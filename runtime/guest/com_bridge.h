#pragma once

#include <atomic>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/guest_runtime.h"
#include "os/vmem.h"

// Guest COM object bridge (O16).
//
// Guest (recompiled x86-64) code talks to most platform components through COM-style interface
// pointers: xgameruntime.dll!QueryApiImpl, d3d12_xs.dll, XCRDAPI!DStorageGetFactory, GameInput,
// MFPlat, CoCreateInstance. This header gives the runtime one way to hand such pointers to guest
// code (host-implemented interfaces) and to call guest-implemented functions and interfaces from
// host code (callbacks).
//
// Object layout in guest memory
// -----------------------------
// A host object implementing interfaces I0..In-1 owns one block of guest memory:
//
//     base + 8*i            vtable pointer of interface i (guest VA of that interface's vtable)
//     base + 8*n (aligned)  optional guest-visible payload (ComObject::payload_address)
//
// The interface pointer for Ii is base + 8*i (so I0's pointer, base, is the object's identity and
// what QueryInterface(IID_IUnknown) returns). Reference count, interface table and the host
// object live host-side: a guest overrun of the block can corrupt neither. Blocks come from a
// small allocator over the bridge's guest memory (set_guest_memory, by default the O7/O5 arena
// win32::get_vmem_arena()), 16-byte aligned and zero-filled.
//
// One vtable exists per declared interface (InterfaceInfo), shared by every object implementing
// it, allocated in guest memory on first use. Each slot holds a synthetic guest address from O9's
// register_host_callable, so `call [rax+8*k]` in lifted code ends in __playgta6_indirect_call,
// which runs the slot's host thunk. Slots 0..2 are IUnknown (QueryInterface, AddRef, Release)
// when InterfaceInfo::iunknown is set; slot k >= first_method_slot() is methods[k - first] or,
// past the declared methods (up to slot_count), an "unimplemented slot" stub.
//
// Method thunks
// -------------
// A thunk is entered like an import thunk (return address already pushed; it pops it like `ret`).
// It reads `this` from rcx, resolves it in the object registry (guest interface pointer ->
// {host object, interface}) and checks that the interface is the one the thunk was generated for
// (a type tag per host interface class), so a stale, foreign or made-up pointer produces a
// diagnostic and a guest fault (FaultKind::Trap, va = the slot's synthetic address, detail = the
// bad `this`) instead of a host crash. The object stays pinned (shared_ptr) for the duration of the
// call even if another thread releases it meanwhile.
//
// Arguments follow the Win64 ABI: Win64 position 0 is `this`; when the method returns a class
// type the hidden result pointer is position 1 (MSVC passes it after `this` for member functions,
// for every user-defined return type, and returns it in rax); then the declared parameters.
// Position p < 4 comes from rcx/rdx/r8/r9 (integers, pointers) or xmm0..3 (float, double),
// p >= 4 from the stack at rsp + 40 + 8*(p - 4). Parameter types are mapped from the C++
// signature of the host member function (see Marshal below):
//
//   integers, bool, enums            low bits of the register / slot (bool = low byte != 0)
//   float, double                    xmm<p> low lane, or the stack slot
//   GuestPtr<T>                      the raw guest address, with checked read/write helpers
//   T* (T trivially copyable / void) guest address translated through guest_to_host() (one seam;
//                                    nullptr for null or unmapped, which also logs a diagnostic)
//   const T& (T trivially copyable)  value read through the guest pointer (REFIID, REFGUID, ...)
//   T by value (trivially copyable)  sizes 1/2/4/8 from the register bits, others through the
//                                    pointer the caller passes (Win64 copies large structs)
//   InterfaceArg<Host>               a guest interface pointer resolved to the host object
//   CallContext& (first param only)  the guest state and `this` of the call; takes no position
//
// Results: integers/enums/bool -> rax; float/double -> xmm0; GuestPtr<T> -> rax; class types ->
// copied to the hidden result pointer, which is returned in rax; void leaves rax alone.
// Escape hatch: Methods<Host>::raw<fn>() binds `void fn(CallContext&)`, which reads its own
// arguments from the state (CallContext::int_arg/float_arg) and writes rax/xmm0 itself.
//
// Declaring an interface (adding one is a table, no hand-written thunk):
//
//     struct IFooHost {                       // host-side C++ interface
//         virtual int32_t Add(int32_t a, int32_t b, GuestPtr<int32_t> out) = 0;
//     };
//     using FooM = com::Methods<IFooHost>;
//     inline constexpr com::MethodDesc kFooMethods[] = {
//         FooM::method<&IFooHost::Add>("Add"),  // slot 3
//         FooM::stub("Reserved"),               // slot 4: named, logging stub
//     };
//     inline constexpr auto kIFoo = com::declare_interface<IFooHost>(
//         "IFoo", com::make_guid("01234567-89ab-cdef-0123-456789abcdef"), kFooMethods,
//         {.slot_count = 16});                 // slots 5..15: unnamed logging stubs
//
//     class Foo : public com::ComObject, public IFooHost { ... };
//     com::Ref<Foo> foo = com::create_object<Foo>(com::implements(kIFoo), ctor args...);
//     *ppv = foo.detach(kIFoo);               // hand the reference to guest code
//
// IUnknown
// --------
// QueryInterface answers IID_IUnknown with the identity pointer and every IID of the object's
// interfaces (InterfaceInfo::iid plus extra_iids, e.g. base interfaces) with that interface's
// pointer, always the same pointer for the same IID, AddRef'ed; otherwise *ppv = 0 and
// E_NOINTERFACE (E_POINTER for a null ppv/riid). ComObject::query_interface_extra lets an object
// add tear-offs. AddRef/Release are atomic; the release that reaches zero unregisters every
// interface pointer, points the guest vtable pointers at a "released object" vtable (any later
// call through a dangling pointer is diagnosed), frees the guest block and drops the bridge's
// reference to the host object. AddRef on a released object is refused with a diagnostic.
//
// Unknown slots
// -------------
// Every stub slot has its own synthetic address. The first call logs
// "com bridge: unimplemented <Interface> {IID} slot N (<name>)" and every call counts a hit
// (stub_hits / stub_hit_report); then the interface's SlotPolicy applies: Trap (default) raises
// FaultKind::UnimplementedSlot through O9's fault handler (va = the stub's synthetic address,
// detail = slot index); ReturnNotImpl sets rax = E_NOTIMPL and returns. The stub identifies its
// slot through GuestState::rip, which __playgta6_indirect_call sets to the target before running
// a host thunk (guest_runtime.h).
//
// Host -> guest calls
// -------------------
// GuestFunction<R(Args...)> calls a guest function pointer, call_guest_method() a vtable slot of
// a guest object, through O9's call_guest on the current thread's guest state: the state bound by
// GuestStateScope (every bridge thunk binds the state it was entered with, so callbacks made from
// inside a guest call reuse that state and stack, below the current rsp), or else a guest thread
// created lazily for this host thread (stack from the bridge's guest memory, released at thread
// exit; set_guest_thread_init_hook can set up its TEB). Arguments: integers/enums/bool, float/
// double (xmm and the integer register, as Win64 does for both), GuestPtr<T> and raw uint64_t
// guest addresses; host pointers are rejected at compile time.
//
// Exceptions and longjmp across host frames (R11, guest_exceptions.h): after every host -> guest
// call the caller checks unwind_pending(state) (guest_runtime.h); while it is set the result is
// meaningless and the caller abandons its work and returns. The thunks of this header take part:
// when the host function they ran returns with an unwind pending they neither store a result nor
// pop the return address, so lifted code continues the unwind. GuestFunction::call_top_level runs
// a callback behind top_level_call_tag() (exceptions leaving it are unhandled).
//
// Threading: the registry, vtables and stub table are thread-safe; lookups take a shared lock.
// Guest memory: every access goes through the bridge arena (VirtualMemoryArena read/write/
// host_ptr); guest addresses are always 64-bit values, never truncated host pointers.

namespace playgta6::guest::com {

using HResult = int32_t;

inline constexpr HResult S_OK_HR = 0;
inline constexpr HResult S_FALSE_HR = 1;
inline constexpr HResult E_NOTIMPL_HR = static_cast<HResult>(0x80004001u);
inline constexpr HResult E_NOINTERFACE_HR = static_cast<HResult>(0x80004002u);
inline constexpr HResult E_POINTER_HR = static_cast<HResult>(0x80004003u);
inline constexpr HResult E_FAIL_HR = static_cast<HResult>(0x80004005u);
inline constexpr HResult E_UNEXPECTED_HR = static_cast<HResult>(0x8000FFFFu);
inline constexpr HResult E_OUTOFMEMORY_HR = static_cast<HResult>(0x8007000Eu);
inline constexpr HResult E_INVALIDARG_HR = static_cast<HResult>(0x80070057u);

// ---------------------------------------------------------------------------
// GUIDs
// ---------------------------------------------------------------------------

struct Guid {
    uint32_t data1 = 0;
    uint16_t data2 = 0;
    uint16_t data3 = 0;
    uint8_t data4[8]{};

    friend constexpr bool operator==(const Guid& a, const Guid& b) {
        if (a.data1 != b.data1 || a.data2 != b.data2 || a.data3 != b.data3) {
            return false;
        }
        for (int i = 0; i < 8; ++i) {
            if (a.data4[i] != b.data4[i]) {
                return false;
            }
        }
        return true;
    }
};
static_assert(sizeof(Guid) == 16 && std::is_trivially_copyable_v<Guid>, "GUID is 16 bytes");

namespace detail {
// Not constexpr: reaching it during constant evaluation is a compile error.
[[noreturn]] void invalid_guid_string();

constexpr uint32_t hex_digit(char c) {
    if (c >= '0' && c <= '9') return static_cast<uint32_t>(c - '0');
    if (c >= 'a' && c <= 'f') return static_cast<uint32_t>(c - 'a' + 10);
    if (c >= 'A' && c <= 'F') return static_cast<uint32_t>(c - 'A' + 10);
    invalid_guid_string();
}
constexpr uint32_t hex_field(const char* s, int digits) {
    uint32_t v = 0;
    for (int i = 0; i < digits; ++i) {
        v = (v << 4) | hex_digit(s[i]);
    }
    return v;
}
}  // namespace detail

// "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" (no braces), usable in constant expressions.
constexpr Guid make_guid(const char (&s)[37]) {
    if (s[8] != '-' || s[13] != '-' || s[18] != '-' || s[23] != '-') {
        detail::invalid_guid_string();
    }
    Guid g{};
    g.data1 = detail::hex_field(s, 8);
    g.data2 = static_cast<uint16_t>(detail::hex_field(s + 9, 4));
    g.data3 = static_cast<uint16_t>(detail::hex_field(s + 14, 4));
    g.data4[0] = static_cast<uint8_t>(detail::hex_field(s + 19, 2));
    g.data4[1] = static_cast<uint8_t>(detail::hex_field(s + 21, 2));
    for (int i = 0; i < 6; ++i) {
        g.data4[2 + i] = static_cast<uint8_t>(detail::hex_field(s + 24 + 2 * i, 2));
    }
    return g;
}

inline constexpr Guid IID_IUnknown_GUID = make_guid("00000000-0000-0000-c000-000000000046");

// "{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}"
std::string format_guid(const Guid& guid);

// ---------------------------------------------------------------------------
// Guest memory seam
// ---------------------------------------------------------------------------

// The arena holding bridge objects, vtables and lazily created guest threads, and through which
// guest pointer arguments are translated. nullptr selects win32::get_vmem_arena() (default).
// Switch it only while no bridge object is alive (tests: before reset_com_bridge()).
void set_guest_memory(os::VirtualMemoryArena* arena);
os::VirtualMemoryArena& guest_memory();

// The one guest -> host pointer translation: host pointer to [va, va + len) when it lies in one
// committed, readable extent, else nullptr (also for va == 0).
void* guest_to_host(uint64_t va, size_t len);
// The reverse, for host pointers guest_to_host (or a marshaled pointer argument) produced: the
// window VA for bytes of a translated-window page, else the arena's guest address. 0 for null.
uint64_t host_to_guest(const void* host);
bool read_guest(uint64_t va, void* dst, size_t len);
bool write_guest(uint64_t va, const void* src, size_t len);

template <typename T>
bool read_guest(uint64_t va, T& out) {
    static_assert(std::is_trivially_copyable_v<T>);
    return read_guest(va, &out, sizeof(T));
}

template <typename T>
bool write_guest(uint64_t va, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    return write_guest(va, &value, sizeof(T));
}

// Zero-filled, 16-byte aligned guest memory (bridge allocator). 0 on failure.
uint64_t guest_alloc(uint64_t size);
void guest_free(uint64_t va);

// A guest pointer argument/result: the raw guest address plus checked accessors.
template <typename T>
class GuestPtr {
public:
    constexpr GuestPtr() = default;
    constexpr explicit GuestPtr(uint64_t va) : va_(va) {}

    constexpr uint64_t va() const { return va_; }
    constexpr bool is_null() const { return va_ == 0; }
    constexpr explicit operator bool() const { return va_ != 0; }

    // Host view of the pointee (nullptr if null or not mapped in one piece).
    T* host() const {
        if constexpr (std::is_void_v<T>) {
            return static_cast<T*>(guest_to_host(va_, 1));
        } else {
            return static_cast<T*>(guest_to_host(va_, sizeof(T)));
        }
    }
    template <typename U = T>
        requires(!std::is_void_v<U>)
    bool read(std::remove_const_t<U>& out) const {
        return va_ != 0 && read_guest(va_, out);
    }
    template <typename U = T>
        requires(!std::is_void_v<U>)
    bool write(const std::remove_const_t<U>& value) const {
        return va_ != 0 && write_guest(va_, value);
    }

private:
    uint64_t va_ = 0;
};

// Guest `void**` / `T**` out-parameter: the guest address of an 8-byte slot.
using OutPtr = GuestPtr<uint64_t>;

// ---------------------------------------------------------------------------
// Interfaces
// ---------------------------------------------------------------------------

class ComObject;
struct InterfaceInfo;

// What a slot without an implementation does after it logged and counted the call.
enum class SlotPolicy : uint8_t {
    Trap = 0,           // FaultKind::UnimplementedSlot through the O9 fault handler (default)
    ReturnNotImpl = 1,  // rax = E_NOTIMPL, return to the caller
};

// One vtable slot after IUnknown. thunk == nullptr declares a named stub.
struct MethodDesc {
    const char* name = nullptr;
    HostThunkFn thunk = nullptr;
};

struct InterfaceInfo {
    const char* name = nullptr;
    Guid iid{};
    std::span<const MethodDesc> methods{};  // slots first_method_slot() ...
    std::span<const Guid> extra_iids{};     // further IIDs answered with this interface (bases)
    uint32_t slot_count = 0;                // total vtable slots, >= first_method_slot() + methods
    SlotPolicy policy = SlotPolicy::Trap;
    bool iunknown = true;                   // slots 0..2 are QueryInterface, AddRef, Release
    const void* host_tag = nullptr;         // identifies the host interface class

    constexpr uint32_t first_method_slot() const { return iunknown ? 3u : 0u; }
    constexpr bool answers(const Guid& id) const {
        if (id == iid) {
            return true;
        }
        for (const Guid& extra : extra_iids) {
            if (id == extra) {
                return true;
            }
        }
        return false;
    }
};

struct InterfaceOptions {
    uint32_t slot_count = 0;  // 0: exactly the declared slots
    SlotPolicy policy = SlotPolicy::Trap;
    bool iunknown = true;
    std::span<const Guid> extra_iids{};
};

namespace detail {
template <typename T>
struct TypeTag {
    static constexpr char value = 0;
};
template <typename T>
constexpr const void* type_tag() {
    return &TypeTag<T>::value;
}
}  // namespace detail

// A declared interface whose methods are implemented by host class `Host`.
template <typename Host>
struct Interface {
    using HostType = Host;
    InterfaceInfo info;
};

template <typename Host>
constexpr Interface<Host> declare_interface(const char* name,
                                            const Guid& iid,
                                            std::span<const MethodDesc> methods,
                                            InterfaceOptions options = {}) {
    InterfaceInfo info{};
    info.name = name;
    info.iid = iid;
    info.methods = methods;
    info.extra_iids = options.extra_iids;
    info.iunknown = options.iunknown;
    info.policy = options.policy;
    const uint32_t declared = info.first_method_slot() + static_cast<uint32_t>(methods.size());
    info.slot_count = options.slot_count > declared ? options.slot_count : declared;
    info.host_tag = detail::type_tag<Host>();
    return Interface<Host>{info};
}

// ---------------------------------------------------------------------------
// Objects
// ---------------------------------------------------------------------------

// Base of every host object exposed to guest code. Derive from ComObject and from the host
// interface classes, create with create_object<T>().
class ComObject {
public:
    ComObject() = default;
    virtual ~ComObject() = default;
    ComObject(const ComObject&) = delete;
    ComObject& operator=(const ComObject&) = delete;

    uint32_t add_ref();
    uint32_t release();
    uint32_t ref_count() const { return refs_.load(std::memory_order_acquire); }

    uint64_t guest_base() const { return guest_base_; }
    uint64_t identity() const { return guest_base_; }
    uint64_t payload_address() const { return payload_; }
    uint64_t payload_size() const { return payload_size_; }
    const char* class_name() const { return class_name_; }
    size_t interface_count() const { return slots_.size(); }

    // Guest pointer of `iface` on this object (no AddRef), 0 if not implemented.
    uint64_t interface_pointer(const InterfaceInfo& iface) const;
    template <typename Host>
    uint64_t interface_pointer(const Interface<Host>& iface) const {
        return interface_pointer(iface.info);
    }

    // COM QueryInterface semantics (AddRefs on success, *out = 0 on failure).
    HResult query_interface(const Guid& iid, uint64_t* out);

protected:
    // Called for IIDs none of the object's interfaces answers; may return a tear-off pointer
    // (AddRef'ed). Default: E_NOINTERFACE.
    virtual HResult query_interface_extra(const Guid& iid, uint64_t* out);

private:
    friend struct ObjectAccess;

    struct Slot {
        const InterfaceInfo* iface = nullptr;
        void* host = nullptr;  // `this` as the interface's host class
    };

    std::atomic<uint32_t> refs_{0};
    uint64_t guest_base_ = 0;
    uint64_t block_size_ = 0;
    uint64_t payload_ = 0;
    uint64_t payload_size_ = 0;
    const char* class_name_ = "";
    std::vector<Slot> slots_;
};

// Owning host reference (one COM reference) to a bridge object.
template <typename T>
class Ref {
public:
    Ref() = default;
    Ref(std::nullptr_t) {}
    // Takes over one reference the caller already owns.
    static Ref adopt(T* object) {
        Ref r;
        r.ptr_ = object;
        return r;
    }
    // Adds a reference.
    static Ref share(T* object) {
        if (object != nullptr) {
            object->add_ref();
        }
        return adopt(object);
    }
    Ref(const Ref& other) : ptr_(other.ptr_) {
        if (ptr_ != nullptr) {
            ptr_->add_ref();
        }
    }
    Ref(Ref&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    Ref& operator=(Ref other) noexcept {
        std::swap(ptr_, other.ptr_);
        return *this;
    }
    ~Ref() { reset(); }

    void reset() {
        if (ptr_ != nullptr) {
            T* p = ptr_;
            ptr_ = nullptr;
            p->release();
        }
    }
    T* get() const { return ptr_; }
    // Gives up ownership of the reference without releasing it (the caller now owns it).
    T* release_ownership() {
        T* p = ptr_;
        ptr_ = nullptr;
        return p;
    }
    T* operator->() const { return ptr_; }
    T& operator*() const { return *ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }

    // Hands this reference to guest code: returns the interface pointer for `iface` and gives up
    // ownership (the guest's Release balances it). 0 (and the reference kept) if the object does
    // not implement `iface`.
    template <typename Host>
    uint64_t detach(const Interface<Host>& iface) {
        if (ptr_ == nullptr) {
            return 0;
        }
        const uint64_t va = ptr_->interface_pointer(iface);
        if (va != 0) {
            ptr_ = nullptr;
        }
        return va;
    }

private:
    T* ptr_ = nullptr;
};

namespace detail {
struct InterfaceBinding {
    const InterfaceInfo* iface;
    void* host;
};

template <typename... Hosts>
struct Implements {
    std::tuple<const Interface<Hosts>*...> ifaces;
};

template <typename T>
concept HasComClassName = requires {
    { T::kComClassName } -> std::convertible_to<const char*>;
};
template <typename T>
concept HasComPayload = requires {
    { T::kComPayloadSize } -> std::convertible_to<uint64_t>;
};

// Allocates the guest block, writes the vtable pointers and registers the interface pointers.
// On success the object holds one reference and the registry owns `object`.
bool register_object(std::shared_ptr<ComObject> object,
                     const char* class_name,
                     uint64_t payload_size,
                     std::span<const InterfaceBinding> bindings);
}  // namespace detail

template <typename... Hosts>
constexpr detail::Implements<Hosts...> implements(const Interface<Hosts>&... ifaces) {
    return detail::Implements<Hosts...>{std::tuple<const Interface<Hosts>*...>(&ifaces...)};
}

// Creates a bridge object implementing the listed interfaces (in vtable-pointer order: the first
// one is the identity). Optional `static constexpr const char* kComClassName` and
// `static constexpr uint64_t kComPayloadSize` on T name the object and reserve guest payload.
// Returns an empty Ref if guest memory is exhausted.
template <typename T, typename... Hosts, typename... Args>
Ref<T> create_object(const detail::Implements<Hosts...>& ifaces, Args&&... args) {
    static_assert(std::is_base_of_v<ComObject, T>, "bridge objects derive from com::ComObject");
    static_assert(sizeof...(Hosts) > 0, "an object implements at least one interface");
    static_assert((std::is_base_of_v<Hosts, T> && ...),
                  "the object class must derive from every host interface class it implements");
    auto object = std::make_shared<T>(std::forward<Args>(args)...);
    T* raw = object.get();
    detail::InterfaceBinding bindings[sizeof...(Hosts)];
    size_t i = 0;
    std::apply(
        [&](const auto*... iface) {
            ((bindings[i++] = detail::InterfaceBinding{
                  &iface->info,
                  static_cast<void*>(static_cast<
                      typename std::remove_cvref_t<decltype(*iface)>::HostType*>(raw))}),
             ...);
        },
        ifaces.ifaces);
    const char* name = "ComObject";
    if constexpr (detail::HasComClassName<T>) {
        name = T::kComClassName;
    }
    uint64_t payload = 0;
    if constexpr (detail::HasComPayload<T>) {
        payload = T::kComPayloadSize;
    }
    if (!detail::register_object(std::move(object), name, payload,
                                 std::span<const detail::InterfaceBinding>(bindings, sizeof...(Hosts)))) {
        return Ref<T>();
    }
    return Ref<T>::adopt(raw);
}

// ---------------------------------------------------------------------------
// Call context and argument marshaling
// ---------------------------------------------------------------------------

struct CallContext {
    GuestState& state;
    uint64_t this_ptr = 0;
    ComObject* object = nullptr;               // nullptr for flat (non-method) thunks
    const InterfaceInfo* iface = nullptr;
    uint64_t entry_rsp = 0;                    // rsp at entry (points at the return address)

    // Raw Win64 argument at position `pos` (0 = rcx/this): integer view and float view.
    // A failed stack read raises FaultKind::BadStack.
    uint64_t int_arg(unsigned pos) const;
    uint64_t float_arg(unsigned pos) const;
    void set_rax(uint64_t value) { state.gpr[GuestReg::Rax] = value; }
    void set_xmm0(uint64_t bits) {
        state.xmm[0][0] = bits;
        state.xmm[0][1] = 0;
    }
};

// A guest interface pointer argument resolved to its host object (pinned for the call).
template <typename Host>
class InterfaceArg {
public:
    InterfaceArg() = default;
    uint64_t va() const { return va_; }
    bool is_null() const { return va_ == 0; }
    // Host interface, nullptr for a null, unknown or wrong-interface pointer.
    Host* get() const { return host_; }
    Host* operator->() const { return host_; }
    explicit operator bool() const { return host_ != nullptr; }
    // Non-null but not a live bridge object implementing Host (a diagnostic was logged).
    bool invalid() const { return va_ != 0 && host_ == nullptr; }
    ComObject* object() const { return object_.get(); }

    static InterfaceArg resolve(uint64_t va);

private:
    uint64_t va_ = 0;
    std::shared_ptr<ComObject> object_;
    Host* host_ = nullptr;
};

namespace detail {
// Binds the state for nested host -> guest calls (see GuestStateScope).
void push_bound_state(GuestState* state, GuestState** previous);
void pop_bound_state(GuestState* previous);
}  // namespace detail

// Makes `state` the one host -> guest calls on this thread use, until destroyed (nests).
class GuestStateScope {
public:
    explicit GuestStateScope(GuestState& state) { detail::push_bound_state(&state, &previous_); }
    ~GuestStateScope() { detail::pop_bound_state(previous_); }
    GuestStateScope(const GuestStateScope&) = delete;
    GuestStateScope& operator=(const GuestStateScope&) = delete;

private:
    GuestState* previous_ = nullptr;
};

namespace detail {

struct Resolved {
    std::shared_ptr<ComObject> object;
    const InterfaceInfo* iface = nullptr;
    void* host = nullptr;
};

// Registry lookup without diagnostics.
bool lookup_interface(uint64_t va, Resolved* out);
// Resolves rcx for a method thunk expecting `host_tag`; on failure logs and raises a guest fault.
Resolved resolve_this_or_fault(GuestState& state, const void* host_tag);
void diagnostic(const char* message);
void log_bad_pointer(const char* what, uint64_t va);
[[noreturn]] void fault_bad_stack(GuestState& state, uint64_t va);

// Pops the thunk's return address (like `ret`), except while an unwind propagates: rsp then holds
// the restored context's value and lifted code continues the unwind.
inline void pop_return(GuestState& state) {
    if (state.unwind_pending == 0) {
        state.gpr[GuestReg::Rsp] += 8;
    }
}

template <typename T>
struct IsGuestPtr : std::false_type {};
template <typename T>
struct IsGuestPtr<GuestPtr<T>> : std::true_type {};
template <typename T>
struct IsInterfaceArg : std::false_type {};
template <typename T>
struct IsInterfaceArg<InterfaceArg<T>> : std::true_type {};

template <typename T>
inline constexpr bool kIsPlainStruct = std::is_class_v<T> && std::is_trivially_copyable_v<T> &&
                                       !IsGuestPtr<T>::value && !IsInterfaceArg<T>::value;

template <typename T>
T int_from_bits(uint64_t raw) {
    if constexpr (std::is_same_v<T, bool>) {
        return (raw & 0xFFu) != 0;
    } else if constexpr (std::is_enum_v<T>) {
        return static_cast<T>(static_cast<std::underlying_type_t<T>>(raw));
    } else {
        return static_cast<T>(raw);
    }
}

template <typename T>
uint64_t bits_from_int(T value) {
    if constexpr (std::is_same_v<T, bool>) {
        return value ? 1u : 0u;
    } else if constexpr (std::is_enum_v<T>) {
        return static_cast<uint64_t>(static_cast<std::underlying_type_t<T>>(value));
    } else if constexpr (std::is_signed_v<T>) {
        return static_cast<uint64_t>(static_cast<int64_t>(value));
    } else {
        return static_cast<uint64_t>(value);
    }
}

inline float f32_from_bits(uint64_t raw) {
    const uint32_t low = static_cast<uint32_t>(raw);
    float value;
    std::memcpy(&value, &low, sizeof(value));
    return value;
}
inline double f64_from_bits(uint64_t raw) {
    double value;
    std::memcpy(&value, &raw, sizeof(value));
    return value;
}
inline uint64_t bits_from_f32(float value) {
    uint32_t low;
    std::memcpy(&low, &value, sizeof(low));
    return low;
}
inline uint64_t bits_from_f64(double value) {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

template <typename T>
T read_struct_arg(uint64_t va) {
    T value{};
    if (va == 0 || !read_guest(va, value)) {
        log_bad_pointer("struct argument", va);
        return T{};
    }
    return value;
}

// Guest -> host parameter conversion for one declared parameter type A.
template <typename A>
struct Marshal {
    using Plain = std::remove_cvref_t<A>;

    static_assert(!std::is_lvalue_reference_v<A> || std::is_const_v<std::remove_reference_t<A>>,
                  "non-const reference parameters are not marshaled: use GuestPtr<T>");
    static_assert(!std::is_rvalue_reference_v<A>, "rvalue reference parameters are not marshaled");

    using Storage = Plain;

    static Storage load(CallContext& ctx, unsigned pos) {
        if constexpr (std::is_same_v<Plain, float>) {
            return f32_from_bits(ctx.float_arg(pos));
        } else if constexpr (std::is_same_v<Plain, double>) {
            return f64_from_bits(ctx.float_arg(pos));
        } else if constexpr (std::is_integral_v<Plain> || std::is_enum_v<Plain>) {
            static_assert(!std::is_reference_v<A>, "pass integers by value");
            return int_from_bits<Plain>(ctx.int_arg(pos));
        } else if constexpr (IsGuestPtr<Plain>::value) {
            return Plain(ctx.int_arg(pos));
        } else if constexpr (IsInterfaceArg<Plain>::value) {
            return Plain::resolve(ctx.int_arg(pos));
        } else if constexpr (std::is_pointer_v<Plain>) {
            using Pointee = std::remove_pointer_t<Plain>;
            static_assert(std::is_void_v<Pointee> || std::is_trivially_copyable_v<Pointee>,
                          "raw pointer parameters must point to trivially copyable data");
            static_assert(!std::is_function_v<Pointee>,
                          "function pointer parameters are guest addresses: use uint64_t or "
                          "GuestFunction");
            const uint64_t va = ctx.int_arg(pos);
            if (va == 0) {
                return nullptr;
            }
            void* host = nullptr;
            if constexpr (std::is_void_v<Pointee>) {
                host = guest_to_host(va, 1);
            } else {
                host = guest_to_host(va, sizeof(Pointee));
            }
            if (host == nullptr) {
                log_bad_pointer("pointer argument", va);
            }
            return static_cast<Plain>(host);
        } else if constexpr (std::is_reference_v<A>) {
            static_assert(std::is_trivially_copyable_v<Plain>,
                          "const reference parameters must be trivially copyable");
            return read_struct_arg<Plain>(ctx.int_arg(pos));
        } else {
            static_assert(kIsPlainStruct<Plain>,
                          "unsupported parameter type (integers, enums, floats, GuestPtr, "
                          "InterfaceArg, data pointers, const T& or trivially copyable structs)");
            constexpr size_t n = sizeof(Plain);
            if constexpr (n == 1 || n == 2 || n == 4 || n == 8) {
                const uint64_t raw = ctx.int_arg(pos);
                Plain value;
                std::memcpy(&value, &raw, n);  // little endian: low bytes of the register
                return value;
            } else {
                return read_struct_arg<Plain>(ctx.int_arg(pos));
            }
        }
    }
};

template <typename R>
inline constexpr bool kReturnsViaHiddenPointer =
    (std::is_class_v<R> || std::is_union_v<R>) && !IsGuestPtr<R>::value;

template <typename R>
void store_result(CallContext& ctx, const R& value, uint64_t hidden) {
    if constexpr (std::is_same_v<R, float>) {
        ctx.set_xmm0(bits_from_f32(value));
    } else if constexpr (std::is_same_v<R, double>) {
        ctx.set_xmm0(bits_from_f64(value));
    } else if constexpr (std::is_integral_v<R> || std::is_enum_v<R>) {
        ctx.set_rax(bits_from_int(value));
    } else if constexpr (IsGuestPtr<R>::value) {
        ctx.set_rax(value.va());
    } else {
        static_assert(kReturnsViaHiddenPointer<R> && std::is_trivially_copyable_v<R>,
                      "unsupported return type (host pointers cannot be returned to the guest; "
                      "use GuestPtr<T> or uint64_t)");
        if (!write_guest(hidden, value)) {
            log_bad_pointer("hidden result pointer", hidden);
        }
        ctx.set_rax(hidden);
    }
}

template <typename F>
struct MemberFn;
template <typename R, typename C, typename... A>
struct MemberFn<R (C::*)(A...)> {
    using Ret = R;
    using Class = C;
    using Params = std::tuple<A...>;
};
template <typename R, typename C, typename... A>
struct MemberFn<R (C::*)(A...) const> : MemberFn<R (C::*)(A...)> {};
template <typename R, typename C, typename... A>
struct MemberFn<R (C::*)(A...) noexcept> : MemberFn<R (C::*)(A...)> {};
template <typename R, typename C, typename... A>
struct MemberFn<R (C::*)(A...) const noexcept> : MemberFn<R (C::*)(A...)> {};

template <typename... A>
struct FirstIsContext : std::false_type {};
template <typename A0, typename... A>
struct FirstIsContext<A0, A...> : std::is_same<A0, CallContext&> {};

template <typename A>
decltype(auto) pass_arg(CallContext& ctx, typename Marshal<A>::Storage& stored) {
    (void)ctx;
    if constexpr (std::is_pointer_v<std::remove_cvref_t<A>> || !std::is_reference_v<A>) {
        return static_cast<std::remove_cvref_t<A>>(stored);
    } else {
        return static_cast<const std::remove_cvref_t<A>&>(stored);
    }
}

// Calls `fn(args...)` with the parameters of `Params` loaded from guest state, then stores the
// result. `first` is the Win64 position of the first declared parameter.
template <typename R, typename... A, size_t... I, typename Call>
void marshal_call(CallContext& ctx, unsigned first, Call&& fn, std::index_sequence<I...>) {
    std::tuple<typename Marshal<A>::Storage...> stored{
        Marshal<A>::load(ctx, first + static_cast<unsigned>(I))...};
    (void)stored;
    if constexpr (std::is_void_v<R>) {
        fn(pass_arg<A>(ctx, std::get<I>(stored))...);
    } else {
        const uint64_t hidden = kReturnsViaHiddenPointer<R> ? ctx.int_arg(first - 1) : 0;
        const R result = fn(pass_arg<A>(ctx, std::get<I>(stored))...);
        if (ctx.state.unwind_pending == 0) {  // an unwind keeps the restored registers
            store_result<R>(ctx, result, hidden);
        }
    }
}

template <typename Host, auto M, typename R, typename Ctx, typename... A>
void invoke_method_with_context(CallContext& ctx, Host* self, std::tuple<Ctx, A...>*) {
    constexpr unsigned first = 1u + (kReturnsViaHiddenPointer<R> ? 1u : 0u);
    marshal_call<R, A...>(
        ctx, first,
        [self, &ctx](auto&&... args) -> R {
            return (self->*M)(ctx, std::forward<decltype(args)>(args)...);
        },
        std::index_sequence_for<A...>{});
}

template <typename Host, auto M, typename R, typename... A>
void invoke_method(CallContext& ctx, Host* self, std::tuple<A...>*) {
    constexpr unsigned first = 1u + (kReturnsViaHiddenPointer<R> ? 1u : 0u);
    if constexpr (FirstIsContext<A...>::value) {
        invoke_method_with_context<Host, M, R>(ctx, self, static_cast<std::tuple<A...>*>(nullptr));
    } else {
        marshal_call<R, A...>(
            ctx, first,
            [self](auto&&... args) -> R { return (self->*M)(std::forward<decltype(args)>(args)...); },
            std::index_sequence_for<A...>{});
    }
}

template <typename Host, auto M>
void method_thunk(void* raw_state) {
    using Traits = MemberFn<decltype(M)>;
    static_assert(std::is_base_of_v<typename Traits::Class, Host> ||
                      std::is_same_v<typename Traits::Class, Host>,
                  "the method must belong to the interface's host class or one of its bases");
    GuestState& state = *static_cast<GuestState*>(raw_state);
    Resolved r = resolve_this_or_fault(state, type_tag<Host>());
    GuestStateScope scope(state);
    CallContext ctx{state, state.gpr[GuestReg::Rcx], r.object.get(), r.iface,
                    state.gpr[GuestReg::Rsp]};
    invoke_method<Host, M, typename Traits::Ret>(ctx, static_cast<Host*>(r.host),
                                                 static_cast<typename Traits::Params*>(nullptr));
    pop_return(state);
}

template <typename Host, void (*F)(CallContext&)>
void raw_method_thunk(void* raw_state) {
    GuestState& state = *static_cast<GuestState*>(raw_state);
    Resolved r = resolve_this_or_fault(state, type_tag<Host>());
    GuestStateScope scope(state);
    CallContext ctx{state, state.gpr[GuestReg::Rcx], r.object.get(), r.iface,
                    state.gpr[GuestReg::Rsp]};
    F(ctx);
    pop_return(state);
}

template <typename F>
struct FreeFn;
template <typename R, typename... A>
struct FreeFn<R (*)(A...)> {
    using Ret = R;
    using Params = std::tuple<A...>;
};
template <typename R, typename... A>
struct FreeFn<R (*)(A...) noexcept> : FreeFn<R (*)(A...)> {};

template <auto F, typename R, typename Ctx, typename... A>
void invoke_free_with_context(CallContext& ctx, std::tuple<Ctx, A...>*) {
    constexpr unsigned first = kReturnsViaHiddenPointer<R> ? 1u : 0u;
    marshal_call<R, A...>(
        ctx, first,
        [&ctx](auto&&... args) -> R { return F(ctx, std::forward<decltype(args)>(args)...); },
        std::index_sequence_for<A...>{});
}

template <auto F, typename R, typename... A>
void invoke_free(CallContext& ctx, std::tuple<A...>*) {
    if constexpr (FirstIsContext<A...>::value) {
        invoke_free_with_context<F, R>(ctx, static_cast<std::tuple<A...>*>(nullptr));
    } else {
        constexpr unsigned first = kReturnsViaHiddenPointer<R> ? 1u : 0u;
        marshal_call<R, A...>(
            ctx, first, [](auto&&... args) -> R { return F(std::forward<decltype(args)>(args)...); },
            std::index_sequence_for<A...>{});
    }
}

template <auto F>
void function_thunk(void* raw_state) {
    using Traits = FreeFn<decltype(F)>;
    GuestState& state = *static_cast<GuestState*>(raw_state);
    GuestStateScope scope(state);
    CallContext ctx{state, 0, nullptr, nullptr, state.gpr[GuestReg::Rsp]};
    invoke_free<F, typename Traits::Ret>(ctx, static_cast<typename Traits::Params*>(nullptr));
    pop_return(state);
}

}  // namespace detail

// Method table builder for interfaces implemented by host class `Host`.
template <typename Host>
struct Methods {
    // A member function of Host (or of a base of Host), marshaled from its signature.
    template <auto M>
    static constexpr MethodDesc method(const char* name) {
        return MethodDesc{name, &detail::method_thunk<Host, M>};
    }
    // `void fn(CallContext&)`: reads its own arguments, sets rax/xmm0 itself.
    template <void (*F)(CallContext&)>
    static constexpr MethodDesc raw(const char* name) {
        return MethodDesc{name, &detail::raw_method_thunk<Host, F>};
    }
    // A known but unimplemented slot (logged with its name, follows the interface's policy).
    static constexpr MethodDesc stub(const char* name) { return MethodDesc{name, nullptr}; }
};

// Guest-state thunk (`void(void* state)`, R5 import-thunk shape) for a flat host function: the
// Win64 arguments are marshaled from F's signature exactly like method parameters (position 0 is
// the first parameter). For guest-callable exports such as QueryApiImpl.
template <auto F>
constexpr HostThunkFn function_thunk() {
    return &detail::function_thunk<F>;
}

// ---------------------------------------------------------------------------
// Unknown-slot accounting and diagnostics
// ---------------------------------------------------------------------------

struct StubHit {
    std::string interface_name;
    Guid iid{};
    uint32_t slot = 0;
    std::string method_name;  // empty for unnamed slots
    uint64_t hits = 0;
};

// Calls so far of `slot` of `iface` (0 for implemented or never-called slots).
uint64_t stub_hits(const InterfaceInfo& iface, uint32_t slot);
template <typename Host>
uint64_t stub_hits(const Interface<Host>& iface, uint32_t slot) {
    return stub_hits(iface.info, slot);
}
// Every stub called at least once, sorted by interface name and slot.
std::vector<StubHit> stub_hit_report();

// Bridge diagnostics (also logged through playgta6::os::log); the newest 256 are kept.
std::vector<std::string> diagnostics();
void clear_diagnostics();

// Guest address of the vtable for `iface` (materialized on first use), 0 on failure.
uint64_t vtable_address(const InterfaceInfo& iface);
// Synthetic guest address in `slot` of `iface`'s vtable, 0 if out of range.
uint64_t slot_address(const InterfaceInfo& iface, uint32_t slot);

// ---------------------------------------------------------------------------
// Leak accounting
// ---------------------------------------------------------------------------

struct BridgeStats {
    uint64_t live_objects = 0;
    uint64_t objects_created = 0;
    uint64_t objects_destroyed = 0;
    uint64_t guest_bytes_in_use = 0;  // bridge allocator (objects, vtables, trampolines' data)
    uint64_t lazy_guest_threads = 0;  // guest threads created for host threads, still alive
};

struct LiveObject {
    std::string class_name;
    uint64_t identity = 0;
    uint32_t refs = 0;
};

BridgeStats bridge_stats();
std::vector<LiveObject> live_objects();

// ---------------------------------------------------------------------------
// Guest state binding and host -> guest calls
// ---------------------------------------------------------------------------

// The state host code on this thread uses for guest calls: the innermost GuestStateScope, else a
// guest thread created on first use for this host thread with a stack of at least
// max(set_lazy_guest_stack_size, min_stack_size) bytes (raises a BadStack fault with a null state
// if the stack cannot be allocated).
GuestState* bound_guest_state();
GuestState& current_guest_state(uint64_t min_stack_size = 0);

// Runs for every lazily created guest thread before its first guest call (e.g. to point
// gs_base at a TEB). nullptr removes it.
using GuestThreadInitHook = void (*)(GuestThread& thread);
void set_guest_thread_init_hook(GuestThreadInitHook hook);
// Stack size of lazily created guest threads (default kDefaultGuestStackSize).
void set_lazy_guest_stack_size(uint64_t bytes);

namespace detail {
template <typename T>
uint64_t guest_arg_bits(const T& value, bool* is_float) {
    *is_float = false;
    if constexpr (std::is_same_v<T, float>) {
        *is_float = true;
        return bits_from_f32(value);
    } else if constexpr (std::is_same_v<T, double>) {
        *is_float = true;
        return bits_from_f64(value);
    } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
        return bits_from_int(value);
    } else if constexpr (IsGuestPtr<T>::value) {
        return value.va();
    } else {
        static_assert(sizeof(T) == 0,
                      "guest calls take integers, enums, floats and guest addresses (GuestPtr / "
                      "uint64_t); host pointers cannot be passed to guest code");
        return 0;
    }
}

uint64_t call_guest_raw(GuestState& state,
                        uint64_t va,
                        const uint64_t* bits,
                        const bool* is_float,
                        size_t count,
                        const void* tag = nullptr);

template <typename R>
R guest_result(GuestState& state, uint64_t rax) {
    if constexpr (std::is_void_v<R>) {
        (void)state;
        (void)rax;
    } else if constexpr (std::is_same_v<R, float>) {
        return f32_from_bits(state.xmm[0][0]);
    } else if constexpr (std::is_same_v<R, double>) {
        return f64_from_bits(state.xmm[0][0]);
    } else if constexpr (std::is_integral_v<R> || std::is_enum_v<R>) {
        return int_from_bits<R>(rax);
    } else if constexpr (IsGuestPtr<R>::value) {
        return R(rax);
    } else {
        static_assert(sizeof(R) == 0, "unsupported guest call result type");
    }
}

template <typename R, typename... A>
R call_guest_typed_tagged(GuestState& state, const void* tag, uint64_t va, const A&... args) {
    constexpr size_t n = sizeof...(A);
    uint64_t bits[n + 1] = {};
    bool is_float[n + 1] = {};
    size_t i = 0;
    ((bits[i] = guest_arg_bits(args, &is_float[i]), ++i), ...);
    (void)i;
    const uint64_t rax = call_guest_raw(state, va, bits, is_float, n, tag);
    return guest_result<R>(state, rax);
}

template <typename R, typename... A>
R call_guest_typed(GuestState& state, uint64_t va, const A&... args) {
    return call_guest_typed_tagged<R>(state, nullptr, va, args...);
}
}  // namespace detail

template <typename Sig>
class GuestFunction;

// A guest function pointer called with the Win64 convention through O9's call_guest.
template <typename R, typename... A>
class GuestFunction<R(A...)> {
public:
    constexpr GuestFunction() = default;
    constexpr explicit GuestFunction(uint64_t va) : va_(va) {}
    constexpr uint64_t address() const { return va_; }
    constexpr explicit operator bool() const { return va_ != 0; }

    R operator()(A... args) const { return call(current_guest_state(), args...); }
    R call(GuestState& state, A... args) const {
        return detail::call_guest_typed<R>(state, va_, args...);
    }
    // As a top-level callback (top_level_call_tag(), guest_runtime.h): an exception leaving the
    // guest function is unhandled instead of reaching the guest frames of an enclosing call.
    R call_top_level(GuestState& state, A... args) const {
        return detail::call_guest_typed_tagged<R>(state, top_level_call_tag(), va_, args...);
    }
    R call_top_level(A... args) const { return call_top_level(current_guest_state(), args...); }

private:
    uint64_t va_ = 0;
};

// Reads the vtable of the guest object `iface_ptr` from guest memory and calls `slot` with
// `this` = iface_ptr. An unreadable object/vtable raises FaultKind::Trap (detail = iface_ptr).
uint64_t guest_method_address(GuestState& state, uint64_t iface_ptr, uint32_t slot);

template <typename R = uint64_t, typename... A>
R call_guest_method(GuestState& state, uint64_t iface_ptr, uint32_t slot, const A&... args) {
    const uint64_t target = guest_method_address(state, iface_ptr, slot);
    return detail::call_guest_typed<R>(state, target, iface_ptr, args...);
}

// ---------------------------------------------------------------------------
// Testing
// ---------------------------------------------------------------------------

// Drops every object (without running guest Release), vtable, stub, counter and diagnostic and
// forgets lazily created guest threads (their stacks are not released). Tests only; call it
// together with reset_guest_runtime() (vtables hold synthetic addresses registered there) and
// never concurrently with guest calls.
void reset_com_bridge();

}  // namespace playgta6::guest::com

template <typename Host>
playgta6::guest::com::InterfaceArg<Host> playgta6::guest::com::InterfaceArg<Host>::resolve(uint64_t va) {
    InterfaceArg arg;
    arg.va_ = va;
    if (va == 0) {
        return arg;
    }
    detail::Resolved r;
    if (!detail::lookup_interface(va, &r) || r.iface->host_tag != detail::type_tag<Host>()) {
        detail::log_bad_pointer("interface argument", va);
        return arg;
    }
    arg.object_ = std::move(r.object);
    arg.host_ = static_cast<Host*>(r.host);
    return arg;
}
