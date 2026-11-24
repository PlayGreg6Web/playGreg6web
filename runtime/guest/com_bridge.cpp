#include "guest/com_bridge.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

#include "os/os_shim.h"
#include "os/win32_kernel.h"

namespace playgta6::guest::com {

struct ObjectAccess;

namespace {

constexpr uint64_t kChunkSize = 256 * 1024;
constexpr uint64_t kLargeBlock = 64 * 1024;
constexpr size_t kMaxDiagnostics = 256;
constexpr uint32_t kReleasedVtableSlots = 128;

std::atomic<os::VirtualMemoryArena*> g_arena{nullptr};
// Bumped by reset_com_bridge(): objects and lazy guest threads of an older generation no longer
// touch guest memory.
std::atomic<uint64_t> g_generation{1};

// ---------------------------------------------------------------------------
// Guest memory allocator
// ---------------------------------------------------------------------------

struct GuestHeap {
    std::mutex mutex;
    uint64_t chunk_base = 0;
    uint64_t chunk_used = 0;
    uint64_t chunk_size = 0;
    std::map<uint64_t, std::vector<uint64_t>> free_lists;  // rounded size -> blocks
    std::unordered_map<uint64_t, uint64_t> live;           // block -> rounded size
    std::unordered_map<uint64_t, uint64_t> large;          // own reservation -> size
    uint64_t bytes_in_use = 0;
};

GuestHeap& heap() {
    static GuestHeap instance;
    return instance;
}

// ---------------------------------------------------------------------------
// Object registry, vtables, stubs, diagnostics
// ---------------------------------------------------------------------------

struct InterfaceEntry {
    std::shared_ptr<ComObject> object;
    uint32_t index = 0;
};

struct StubRecord {
    const InterfaceInfo* iface = nullptr;
    uint32_t slot = 0;
    std::string label;        // "IFoo::Name" or "IFoo::slot N"
    std::string method_name;  // declared stub name, may be empty
    bool released_object = false;
    std::atomic<uint64_t> hits{0};
};

struct VtableRecord {
    uint64_t va = 0;
    std::vector<uint64_t> slots;
};

struct Bridge {
    std::shared_mutex registry_mutex;
    std::unordered_map<uint64_t, InterfaceEntry> interfaces;              // interface pointer
    std::unordered_map<uint64_t, std::shared_ptr<ComObject>> objects;     // identity

    std::mutex vtable_mutex;
    std::unordered_map<const InterfaceInfo*, VtableRecord> vtables;
    uint64_t iunknown_thunks[3] = {0, 0, 0};

    std::shared_mutex stub_mutex;
    std::deque<StubRecord> stubs;
    std::unordered_map<uint64_t, StubRecord*> stub_by_address;
    std::map<std::pair<const InterfaceInfo*, uint32_t>, StubRecord*> stub_by_slot;

    std::mutex diag_mutex;
    std::deque<std::string> diags;

    std::atomic<uint64_t> objects_created{0};
    std::atomic<uint64_t> objects_destroyed{0};
    std::atomic<uint64_t> lazy_threads{0};

    std::atomic<GuestThreadInitHook> init_hook{nullptr};
    std::atomic<uint64_t> lazy_stack_size{kDefaultGuestStackSize};
};

Bridge& bridge() {
    static Bridge instance;
    return instance;
}

constexpr InterfaceInfo kReleasedObjectInterface = [] {
    InterfaceInfo info{};
    info.name = "<released object>";
    info.iunknown = false;
    info.slot_count = kReleasedVtableSlots;
    info.policy = SlotPolicy::Trap;
    return info;
}();

std::string iface_label(const InterfaceInfo& iface) {
    return std::string(iface.name != nullptr ? iface.name : "<interface>");
}

void report(const std::string& message) {
    os::log(message);
    Bridge& b = bridge();
    std::lock_guard<std::mutex> lock(b.diag_mutex);
    b.diags.push_back(message);
    while (b.diags.size() > kMaxDiagnostics) {
        b.diags.pop_front();
    }
}

std::string hex(uint64_t value) {
    char buffer[24];
    std::snprintf(buffer, sizeof(buffer), "0x%016" PRIx64, value);
    return buffer;
}

std::string synthetic_name(uint64_t va) {
    SyntheticTarget target;
    if (describe_synthetic(va, &target) && target.name != nullptr) {
        return target.name;
    }
    return "method at " + hex(va);
}

// ---------------------------------------------------------------------------
// Lazy guest threads and state binding
// ---------------------------------------------------------------------------

struct LazyGuestThread {
    GuestThread thread{};
    os::VirtualMemoryArena* arena = nullptr;
    uint64_t generation = 0;
    bool created = false;

    ~LazyGuestThread() {
        if (created && generation == g_generation.load(std::memory_order_acquire)) {
            destroy_guest_thread(*arena, thread);
            bridge().lazy_threads.fetch_sub(1, std::memory_order_relaxed);
        }
    }
};

thread_local GuestState* t_bound_state = nullptr;
thread_local LazyGuestThread t_lazy_thread;

}  // namespace

// ---------------------------------------------------------------------------
// GUIDs
// ---------------------------------------------------------------------------

void detail::invalid_guid_string() {
    os::log("com bridge: malformed GUID string");
    std::abort();
}

std::string format_guid(const Guid& g) {
    char buffer[40];
    std::snprintf(buffer, sizeof(buffer), "{%08" PRIX32 "-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}",
                  g.data1, static_cast<unsigned>(g.data2), static_cast<unsigned>(g.data3),
                  g.data4[0], g.data4[1], g.data4[2], g.data4[3], g.data4[4], g.data4[5],
                  g.data4[6], g.data4[7]);
    return buffer;
}

// ---------------------------------------------------------------------------
// Guest memory
// ---------------------------------------------------------------------------

void set_guest_memory(os::VirtualMemoryArena* arena) {
    g_arena.store(arena, std::memory_order_release);
}

os::VirtualMemoryArena& guest_memory() {
    if (os::VirtualMemoryArena* arena = g_arena.load(std::memory_order_acquire)) {
        return *arena;
    }
    return *os::win32::get_vmem_arena();
}

void* guest_to_host(uint64_t va, size_t len) {
    if (va == 0) {
        return nullptr;
    }
    return guest_memory().host_ptr(va, len == 0 ? 1 : len);
}

bool read_guest(uint64_t va, void* dst, size_t len) {
    return va != 0 && guest_memory().read(va, dst, len);
}

bool write_guest(uint64_t va, const void* src, size_t len) {
    return va != 0 && guest_memory().write(va, src, len);
}

uint64_t guest_alloc(uint64_t size) {
    uint64_t rounded = 0;
    if (!os::align_up_checked(size == 0 ? 16 : size, 16, rounded)) {
        return 0;
    }
    os::VirtualMemoryArena& arena = guest_memory();
    GuestHeap& h = heap();
    std::lock_guard<std::mutex> lock(h.mutex);
    if (rounded > kLargeBlock) {
        const uint64_t va = arena.virtual_alloc(0, rounded, os::kMemReserve | os::kMemCommit,
                                                os::PageProtect::ReadWrite);
        if (va == 0) {
            return 0;
        }
        h.large[va] = rounded;
        h.bytes_in_use += rounded;
        return va;
    }
    uint64_t va = 0;
    auto it = h.free_lists.find(rounded);
    if (it != h.free_lists.end() && !it->second.empty()) {
        va = it->second.back();
        it->second.pop_back();
        const std::vector<uint8_t> zeros(static_cast<size_t>(rounded), 0);
        arena.write(va, zeros.data(), zeros.size());
    } else {
        if (h.chunk_base == 0 || h.chunk_size - h.chunk_used < rounded) {
            const uint64_t chunk = arena.virtual_alloc(0, kChunkSize, os::kMemReserve | os::kMemCommit,
                                                       os::PageProtect::ReadWrite);
            if (chunk == 0) {
                return 0;
            }
            h.chunk_base = chunk;
            h.chunk_used = 0;
            h.chunk_size = kChunkSize;
        }
        va = h.chunk_base + h.chunk_used;
        h.chunk_used += rounded;
    }
    h.live[va] = rounded;
    h.bytes_in_use += rounded;
    return va;
}

void guest_free(uint64_t va) {
    if (va == 0) {
        return;
    }
    GuestHeap& h = heap();
    std::lock_guard<std::mutex> lock(h.mutex);
    if (auto it = h.live.find(va); it != h.live.end()) {
        h.free_lists[it->second].push_back(va);
        h.bytes_in_use -= it->second;
        h.live.erase(it);
        return;
    }
    if (auto it = h.large.find(va); it != h.large.end()) {
        guest_memory().release(va);
        h.bytes_in_use -= it->second;
        h.large.erase(it);
        return;
    }
    report("com bridge: guest_free of unknown block " + hex(va));
}

// ---------------------------------------------------------------------------
// Diagnostics helpers
// ---------------------------------------------------------------------------

void detail::diagnostic(const char* message) {
    report(message);
}

void detail::log_bad_pointer(const char* what, uint64_t va) {
    report(std::string("com bridge: invalid ") + what + " " + hex(va));
}

void detail::fault_bad_stack(GuestState& state, uint64_t va) {
    report("com bridge: guest stack argument not readable at " + hex(va));
    raise_fault(FaultKind::BadStack, &state, state.rip, va);
}

std::vector<std::string> diagnostics() {
    Bridge& b = bridge();
    std::lock_guard<std::mutex> lock(b.diag_mutex);
    return std::vector<std::string>(b.diags.begin(), b.diags.end());
}

void clear_diagnostics() {
    Bridge& b = bridge();
    std::lock_guard<std::mutex> lock(b.diag_mutex);
    b.diags.clear();
}

// ---------------------------------------------------------------------------
// Call context
// ---------------------------------------------------------------------------

uint64_t CallContext::int_arg(unsigned pos) const {
    static constexpr uint32_t kRegs[4] = {GuestReg::Rcx, GuestReg::Rdx, GuestReg::R8, GuestReg::R9};
    if (pos < 4) {
        return state.gpr[kRegs[pos]];
    }
    const uint64_t va = entry_rsp + kGuestStackArgOffset + 8ull * (pos - 4);
    uint64_t value = 0;
    if (!guest_memory().read(va, &value, sizeof(value))) {
        detail::fault_bad_stack(state, va);
    }
    return value;
}

uint64_t CallContext::float_arg(unsigned pos) const {
    if (pos < 4) {
        return state.xmm[pos][0];
    }
    return int_arg(pos);
}

// ---------------------------------------------------------------------------
// Objects
// ---------------------------------------------------------------------------

struct ObjectAccess {
    static std::vector<ComObject::Slot>& slots(ComObject& o) { return o.slots_; }
    static std::atomic<uint32_t>& refs(ComObject& o) { return o.refs_; }
    static void add_slot(ComObject& o, const InterfaceInfo* iface, void* host) {
        o.slots_.push_back(ComObject::Slot{iface, host});
    }

    static void set_layout(ComObject& o, uint64_t base, uint64_t block, uint64_t payload,
                           uint64_t payload_size, const char* name) {
        o.guest_base_ = base;
        o.block_size_ = block;
        o.payload_ = payload;
        o.payload_size_ = payload_size;
        o.class_name_ = name;
    }

    static uint64_t block_size(const ComObject& o) { return o.block_size_; }

    // Final release: unregister, poison the vtable pointers, free the block.
    static void destroy(ComObject& o) {
        Bridge& b = bridge();
        std::shared_ptr<ComObject> owner;
        {
            std::unique_lock<std::shared_mutex> lock(b.registry_mutex);
            auto it = b.objects.find(o.guest_base_);
            if (it == b.objects.end() || it->second.get() != &o) {
                return;  // not registered (bridge reset, or a failed creation)
            }
            owner = std::move(it->second);
            b.objects.erase(it);
            for (size_t i = 0; i < o.slots_.size(); ++i) {
                b.interfaces.erase(o.guest_base_ + 8 * i);
            }
        }
        const uint64_t dead = vtable_address(kReleasedObjectInterface);
        for (size_t i = 0; i < o.slots_.size(); ++i) {
            write_guest(o.guest_base_ + 8 * i, dead);
        }
        guest_free(o.guest_base_);
        b.objects_destroyed.fetch_add(1, std::memory_order_relaxed);
        owner.reset();  // the host object dies now, or when the last in-flight call returns
    }
};

uint32_t ComObject::add_ref() {
    return refs_.fetch_add(1, std::memory_order_relaxed) + 1;
}

uint32_t ComObject::release() {
    const uint32_t before = refs_.fetch_sub(1, std::memory_order_acq_rel);
    if (before == 0) {
        refs_.fetch_add(1, std::memory_order_relaxed);
        report(std::string("com bridge: Release on ") + class_name_ + " " + hex(guest_base_) +
               " with a zero reference count");
        return 0;
    }
    if (before == 1) {
        ObjectAccess::destroy(*this);
    }
    return before - 1;
}

uint64_t ComObject::interface_pointer(const InterfaceInfo& iface) const {
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].iface == &iface) {
            return guest_base_ + 8 * i;
        }
    }
    return 0;
}

HResult ComObject::query_interface(const Guid& iid, uint64_t* out) {
    if (out == nullptr) {
        return E_POINTER_HR;
    }
    *out = 0;
    if (iid == IID_IUnknown_GUID) {
        add_ref();
        *out = guest_base_;
        return S_OK_HR;
    }
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].iface->answers(iid)) {
            add_ref();
            *out = guest_base_ + 8 * i;
            return S_OK_HR;
        }
    }
    return query_interface_extra(iid, out);
}

HResult ComObject::query_interface_extra(const Guid& /*iid*/, uint64_t* out) {
    *out = 0;
    return E_NOINTERFACE_HR;
}

bool detail::register_object(std::shared_ptr<ComObject> object,
                             const char* class_name,
                             uint64_t payload_size,
                             std::span<const InterfaceBinding> bindings) {
    std::vector<uint64_t> vtables;
    vtables.reserve(bindings.size());
    for (const InterfaceBinding& binding : bindings) {
        const uint64_t vt = vtable_address(*binding.iface);
        if (vt == 0) {
            report(std::string("com bridge: cannot materialize vtable of ") + iface_label(*binding.iface));
            return false;
        }
        vtables.push_back(vt);
    }
    const uint64_t header = 8 * bindings.size();
    uint64_t payload_offset = 0;
    os::align_up_checked(header, 16, payload_offset);
    const uint64_t block = payload_offset + payload_size;
    const uint64_t base = guest_alloc(block);
    if (base == 0) {
        report(std::string("com bridge: out of guest memory creating ") + class_name);
        return false;
    }
    for (size_t i = 0; i < vtables.size(); ++i) {
        write_guest(base + 8 * i, vtables[i]);
    }
    ComObject& o = *object;
    ObjectAccess::slots(o).clear();
    for (const InterfaceBinding& binding : bindings) {
        ObjectAccess::add_slot(o, binding.iface, binding.host);
    }
    ObjectAccess::set_layout(o, base, block, payload_size != 0 ? base + payload_offset : 0,
                             payload_size, class_name);
    ObjectAccess::refs(o).store(1, std::memory_order_release);

    Bridge& b = bridge();
    {
        std::unique_lock<std::shared_mutex> lock(b.registry_mutex);
        for (size_t i = 0; i < bindings.size(); ++i) {
            b.interfaces[base + 8 * i] = InterfaceEntry{object, static_cast<uint32_t>(i)};
        }
        b.objects[base] = std::move(object);
    }
    b.objects_created.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool detail::lookup_interface(uint64_t va, Resolved* out) {
    Bridge& b = bridge();
    std::shared_lock<std::shared_mutex> lock(b.registry_mutex);
    auto it = b.interfaces.find(va);
    if (it == b.interfaces.end()) {
        return false;
    }
    const auto& slot = ObjectAccess::slots(*it->second.object)[it->second.index];
    out->object = it->second.object;
    out->iface = slot.iface;
    out->host = slot.host;
    return true;
}

namespace {

detail::Resolved resolve_any_or_fault(GuestState& state, const char* method) {
    const uint64_t self = state.gpr[GuestReg::Rcx];
    detail::Resolved r;
    if (!detail::lookup_interface(self, &r)) {
        report(std::string("com bridge: ") + method + " called on unknown object this=" + hex(self));
        raise_fault(FaultKind::Trap, &state, state.rip, self);
    }
    return r;
}

}  // namespace

detail::Resolved detail::resolve_this_or_fault(GuestState& state, const void* host_tag) {
    const uint64_t self = state.gpr[GuestReg::Rcx];
    Resolved r;
    if (!lookup_interface(self, &r)) {
        report("com bridge: " + synthetic_name(state.rip) + " called on unknown object this=" + hex(self));
        raise_fault(FaultKind::Trap, &state, state.rip, self);
    }
    if (r.iface->host_tag != host_tag) {
        report("com bridge: " + synthetic_name(state.rip) + " called with this=" + hex(self) +
               ", which is a " + iface_label(*r.iface) + " pointer of " + r.object->class_name());
        raise_fault(FaultKind::Trap, &state, state.rip, self);
    }
    return r;
}

// ---------------------------------------------------------------------------
// IUnknown and stub thunks
// ---------------------------------------------------------------------------

namespace {

void qi_thunk(void* raw) {
    GuestState& state = *static_cast<GuestState*>(raw);
    detail::Resolved r = resolve_any_or_fault(state, "IUnknown::QueryInterface");
    const uint64_t riid = state.gpr[GuestReg::Rdx];
    const uint64_t ppv = state.gpr[GuestReg::R8];
    HResult hr = E_POINTER_HR;
    if (ppv != 0) {
        const uint64_t zero = 0;
        if (!write_guest(ppv, zero)) {
            detail::log_bad_pointer("QueryInterface out pointer", ppv);
        } else {
            Guid iid;
            if (riid == 0 || !read_guest(riid, iid)) {
                if (riid != 0) {
                    detail::log_bad_pointer("QueryInterface riid", riid);
                }
            } else {
                uint64_t result = 0;
                hr = r.object->query_interface(iid, &result);
                if (hr >= 0 && !write_guest(ppv, result)) {
                    r.object->release();
                    hr = E_POINTER_HR;
                }
            }
        }
    }
    state.gpr[GuestReg::Rax] = static_cast<uint64_t>(static_cast<int64_t>(hr));
    detail::pop_return(state);
}

void addref_thunk(void* raw) {
    GuestState& state = *static_cast<GuestState*>(raw);
    detail::Resolved r = resolve_any_or_fault(state, "IUnknown::AddRef");
    auto& refs = ObjectAccess::refs(*r.object);
    uint32_t count = refs.load(std::memory_order_relaxed);
    while (count != 0 && !refs.compare_exchange_weak(count, count + 1, std::memory_order_relaxed)) {
    }
    if (count == 0) {
        report(std::string("com bridge: AddRef on released ") + r.object->class_name());
        state.gpr[GuestReg::Rax] = 0;
    } else {
        state.gpr[GuestReg::Rax] = count + 1;
    }
    detail::pop_return(state);
}

void release_thunk(void* raw) {
    GuestState& state = *static_cast<GuestState*>(raw);
    detail::Resolved r = resolve_any_or_fault(state, "IUnknown::Release");
    state.gpr[GuestReg::Rax] = r.object->release();
    detail::pop_return(state);
}

void stub_thunk(void* raw) {
    GuestState& state = *static_cast<GuestState*>(raw);
    Bridge& b = bridge();
    StubRecord* stub = nullptr;
    {
        std::shared_lock<std::shared_mutex> lock(b.stub_mutex);
        auto it = b.stub_by_address.find(state.rip);
        if (it != b.stub_by_address.end()) {
            stub = it->second;
        }
    }
    if (stub == nullptr) {
        report("com bridge: stub entered with unknown rip " + hex(state.rip));
        raise_fault(FaultKind::UnimplementedSlot, &state, state.rip, 0);
    }
    const uint64_t hits = stub->hits.fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t return_address = 0;
    read_guest(state.gpr[GuestReg::Rsp], return_address);
    if (stub->released_object) {
        report("com bridge: call to slot " + std::to_string(stub->slot) +
               " through a released object this=" + hex(state.gpr[GuestReg::Rcx]) +
               " (return address " + hex(return_address) + ")");
        raise_fault(FaultKind::Trap, &state, state.rip, state.gpr[GuestReg::Rcx]);
    }
    if (hits == 1) {
        std::string message = "com bridge: unimplemented " + iface_label(*stub->iface) + " " +
                              format_guid(stub->iface->iid) + " slot " + std::to_string(stub->slot);
        if (!stub->method_name.empty()) {
            message += " (" + stub->method_name + ")";
        }
        message += " called, this=" + hex(state.gpr[GuestReg::Rcx]) + " return address " +
                   hex(return_address);
        report(message);
    }
    if (stub->iface->policy == SlotPolicy::ReturnNotImpl) {
        state.gpr[GuestReg::Rax] = static_cast<uint64_t>(static_cast<int64_t>(E_NOTIMPL_HR));
        detail::pop_return(state);
        return;
    }
    raise_fault(FaultKind::UnimplementedSlot, &state, state.rip, stub->slot);
}

uint64_t register_stub_locked(Bridge& b, const InterfaceInfo& iface, uint32_t slot, const char* name) {
    StubRecord& rec = b.stubs.emplace_back();
    rec.iface = &iface;
    rec.slot = slot;
    rec.released_object = (&iface == &kReleasedObjectInterface);
    rec.method_name = name != nullptr ? name : "";
    rec.label = iface_label(iface) + "::" +
                (rec.method_name.empty() ? "slot " + std::to_string(slot) : rec.method_name) +
                " (unimplemented)";
    const uint64_t va = register_host_callable(&stub_thunk, rec.label.c_str());
    if (va == 0) {
        b.stubs.pop_back();
        return 0;
    }
    std::unique_lock<std::shared_mutex> lock(b.stub_mutex);
    b.stub_by_address[va] = &rec;
    b.stub_by_slot[{&iface, slot}] = &rec;
    return va;
}

}  // namespace

uint64_t vtable_address(const InterfaceInfo& iface) {
    Bridge& b = bridge();
    std::lock_guard<std::mutex> lock(b.vtable_mutex);
    if (auto it = b.vtables.find(&iface); it != b.vtables.end()) {
        return it->second.va;
    }
    if (iface.iunknown && b.iunknown_thunks[0] == 0) {
        b.iunknown_thunks[0] = register_host_callable(&qi_thunk, "IUnknown::QueryInterface");
        b.iunknown_thunks[1] = register_host_callable(&addref_thunk, "IUnknown::AddRef");
        b.iunknown_thunks[2] = register_host_callable(&release_thunk, "IUnknown::Release");
    }
    VtableRecord rec;
    rec.slots.resize(iface.slot_count, 0);
    const uint32_t first = iface.first_method_slot();
    for (uint32_t slot = 0; slot < iface.slot_count; ++slot) {
        uint64_t address = 0;
        if (slot < first) {
            address = b.iunknown_thunks[slot];
        } else if (slot - first < iface.methods.size() && iface.methods[slot - first].thunk != nullptr) {
            const MethodDesc& m = iface.methods[slot - first];
            const std::string label =
                iface_label(iface) + "::" + (m.name != nullptr ? m.name : "slot " + std::to_string(slot));
            address = register_host_callable(m.thunk, label.c_str());
        } else {
            const char* name =
                slot - first < iface.methods.size() ? iface.methods[slot - first].name : nullptr;
            address = register_stub_locked(b, iface, slot, name);
        }
        if (address == 0) {
            return 0;
        }
        rec.slots[slot] = address;
    }
    rec.va = guest_alloc(8ull * std::max<uint32_t>(iface.slot_count, 1));
    if (rec.va == 0 || (iface.slot_count != 0 &&
                        !write_guest(rec.va, rec.slots.data(), rec.slots.size() * sizeof(uint64_t)))) {
        return 0;
    }
    const uint64_t va = rec.va;
    b.vtables.emplace(&iface, std::move(rec));
    return va;
}

uint64_t slot_address(const InterfaceInfo& iface, uint32_t slot) {
    if (vtable_address(iface) == 0) {
        return 0;
    }
    Bridge& b = bridge();
    std::lock_guard<std::mutex> lock(b.vtable_mutex);
    const VtableRecord& rec = b.vtables.at(&iface);
    return slot < rec.slots.size() ? rec.slots[slot] : 0;
}

uint64_t stub_hits(const InterfaceInfo& iface, uint32_t slot) {
    Bridge& b = bridge();
    std::shared_lock<std::shared_mutex> lock(b.stub_mutex);
    auto it = b.stub_by_slot.find({&iface, slot});
    return it != b.stub_by_slot.end() ? it->second->hits.load(std::memory_order_relaxed) : 0;
}

std::vector<StubHit> stub_hit_report() {
    Bridge& b = bridge();
    std::vector<StubHit> out;
    {
        std::shared_lock<std::shared_mutex> lock(b.stub_mutex);
        for (const StubRecord& rec : b.stubs) {
            const uint64_t hits = rec.hits.load(std::memory_order_relaxed);
            if (hits != 0) {
                out.push_back(StubHit{iface_label(*rec.iface), rec.iface->iid, rec.slot, rec.method_name, hits});
            }
        }
    }
    std::sort(out.begin(), out.end(), [](const StubHit& a, const StubHit& c) {
        return a.interface_name != c.interface_name ? a.interface_name < c.interface_name : a.slot < c.slot;
    });
    return out;
}

// ---------------------------------------------------------------------------
// Leak accounting
// ---------------------------------------------------------------------------

BridgeStats bridge_stats() {
    Bridge& b = bridge();
    BridgeStats s;
    s.objects_created = b.objects_created.load(std::memory_order_relaxed);
    s.objects_destroyed = b.objects_destroyed.load(std::memory_order_relaxed);
    {
        std::shared_lock<std::shared_mutex> lock(b.registry_mutex);
        s.live_objects = b.objects.size();
    }
    {
        GuestHeap& h = heap();
        std::lock_guard<std::mutex> lock(h.mutex);
        s.guest_bytes_in_use = h.bytes_in_use;
    }
    s.lazy_guest_threads = b.lazy_threads.load(std::memory_order_relaxed);
    return s;
}

std::vector<LiveObject> live_objects() {
    Bridge& b = bridge();
    std::vector<LiveObject> out;
    std::shared_lock<std::shared_mutex> lock(b.registry_mutex);
    for (const auto& [identity, object] : b.objects) {
        out.push_back(LiveObject{object->class_name(), identity, object->ref_count()});
    }
    std::sort(out.begin(), out.end(),
              [](const LiveObject& a, const LiveObject& c) { return a.identity < c.identity; });
    return out;
}

// ---------------------------------------------------------------------------
// Guest state binding and host -> guest calls
// ---------------------------------------------------------------------------

void detail::push_bound_state(GuestState* state, GuestState** previous) {
    *previous = t_bound_state;
    t_bound_state = state;
}

void detail::pop_bound_state(GuestState* previous) {
    t_bound_state = previous;
}

GuestState* bound_guest_state() {
    return t_bound_state;
}

GuestState& current_guest_state(uint64_t min_stack_size) {
    if (t_bound_state != nullptr) {
        return *t_bound_state;
    }
    LazyGuestThread& lazy = t_lazy_thread;
    const uint64_t generation = g_generation.load(std::memory_order_acquire);
    if (!lazy.created || lazy.generation != generation) {
        lazy = LazyGuestThread{};  // a stale thread from before a reset is forgotten, not freed
        os::VirtualMemoryArena& arena = guest_memory();
        Bridge& b = bridge();
        const uint64_t stack_size =
            std::max(b.lazy_stack_size.load(std::memory_order_relaxed), min_stack_size);
        if (!create_guest_thread(arena, stack_size, lazy.thread)) {
            report("com bridge: cannot allocate a guest stack for this host thread");
            raise_fault(FaultKind::BadStack, nullptr, 0, 0);
        }
        lazy.arena = &arena;
        lazy.generation = generation;
        lazy.created = true;
        b.lazy_threads.fetch_add(1, std::memory_order_relaxed);
        if (GuestThreadInitHook hook = b.init_hook.load(std::memory_order_acquire)) {
            hook(lazy.thread);
        }
    }
    return lazy.thread.state;
}

void set_guest_thread_init_hook(GuestThreadInitHook hook) {
    bridge().init_hook.store(hook, std::memory_order_release);
}

void set_lazy_guest_stack_size(uint64_t bytes) {
    bridge().lazy_stack_size.store(bytes == 0 ? kDefaultGuestStackSize : bytes, std::memory_order_relaxed);
}

uint64_t detail::call_guest_raw(GuestState& state,
                                uint64_t va,
                                const uint64_t* bits,
                                const bool* is_float,
                                size_t count) {
    for (size_t i = 0; i < count && i < 4; ++i) {
        if (is_float[i]) {
            state.xmm[i][0] = bits[i];
            state.xmm[i][1] = 0;
        }
    }
    GuestStateScope scope(state);
    return call_guest(guest_memory(), state, va, std::span<const uint64_t>(bits, count));
}

uint64_t guest_method_address(GuestState& state, uint64_t iface_ptr, uint32_t slot) {
    uint64_t vtable = 0;
    uint64_t target = 0;
    if (iface_ptr == 0 || !read_guest(iface_ptr, vtable) || vtable == 0 ||
        !read_guest(vtable + 8ull * slot, target)) {
        report("com bridge: cannot read vtable slot " + std::to_string(slot) + " of guest object " +
               hex(iface_ptr));
        raise_fault(FaultKind::Trap, &state, state.rip, iface_ptr);
    }
    return target;
}

// ---------------------------------------------------------------------------
// Testing
// ---------------------------------------------------------------------------

void reset_com_bridge() {
    Bridge& b = bridge();
    g_generation.fetch_add(1, std::memory_order_acq_rel);
    std::unordered_map<uint64_t, std::shared_ptr<ComObject>> objects;
    {
        std::unique_lock<std::shared_mutex> lock(b.registry_mutex);
        b.interfaces.clear();
        objects.swap(b.objects);
    }
    objects.clear();  // host destructors run outside the lock
    {
        std::lock_guard<std::mutex> lock(b.vtable_mutex);
        b.vtables.clear();
        b.iunknown_thunks[0] = b.iunknown_thunks[1] = b.iunknown_thunks[2] = 0;
        std::unique_lock<std::shared_mutex> stub_lock(b.stub_mutex);
        b.stub_by_address.clear();
        b.stub_by_slot.clear();
        b.stubs.clear();
    }
    {
        GuestHeap& h = heap();
        std::lock_guard<std::mutex> lock(h.mutex);
        h.chunk_base = h.chunk_used = h.chunk_size = 0;
        h.free_lists.clear();
        h.live.clear();
        h.large.clear();
        h.bytes_in_use = 0;
    }
    clear_diagnostics();
    b.objects_created.store(0, std::memory_order_relaxed);
    b.objects_destroyed.store(0, std::memory_order_relaxed);
    b.lazy_threads.store(0, std::memory_order_relaxed);
    b.init_hook.store(nullptr, std::memory_order_release);
    b.lazy_stack_size.store(kDefaultGuestStackSize, std::memory_order_relaxed);
    t_lazy_thread.created = false;
}

}  // namespace playgta6::guest::com
