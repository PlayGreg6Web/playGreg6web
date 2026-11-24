#include "guest/guest_exceptions.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <vector>

#include "os/win32_kernel.h"
#include "os/win32_loader.h"

namespace playgta6::guest {

namespace {

using os::VirtualMemoryArena;

constexpr uint32_t kMaxTableEntries = 1u << 20;
constexpr int kMaxChainDepth = 32;
// Frames visited by one dispatch or unwind before the stack is considered corrupt.
constexpr int kMaxFrames = 1 << 20;
constexpr uint64_t kNoPrologOffset = ~uint64_t{0};

// UNWIND_CODE operations.
constexpr uint8_t kUwopPushNonvol = 0;
constexpr uint8_t kUwopAllocLarge = 1;
constexpr uint8_t kUwopAllocSmall = 2;
constexpr uint8_t kUwopSetFpreg = 3;
constexpr uint8_t kUwopSaveNonvol = 4;
constexpr uint8_t kUwopSaveNonvolFar = 5;
constexpr uint8_t kUwopEpilog = 6;  // version 2 only
constexpr uint8_t kUwopSaveXmm128 = 8;
constexpr uint8_t kUwopSaveXmm128Far = 9;
constexpr uint8_t kUwopPushMachframe = 10;

template <typename T>
bool read_guest(VirtualMemoryArena& arena, uint64_t va, T& out) {
    return arena.read(va, &out, sizeof(T));
}

template <typename T>
bool write_guest(VirtualMemoryArena& arena, uint64_t va, const T& value) {
    return arena.write(va, &value, sizeof(T));
}

bool read_u64(VirtualMemoryArena& arena, uint64_t va, uint64_t& out) {
    return read_guest(arena, va, out);
}

// ---------------------------------------------------------------------------
// Function tables
// ---------------------------------------------------------------------------

struct Table {
    uint64_t base = 0;
    uint64_t lo = 0;  // covered VA range [lo, hi)
    uint64_t hi = 0;
    uint64_t table_va = 0;
    bool image = false;
    std::vector<GuestRuntimeFunction> entries;  // sorted by begin
    std::vector<uint32_t> index;                // position of entries[i] in the guest table
};

struct Registry {
    std::shared_mutex mutex;
    std::vector<std::unique_ptr<Table>> tables;
    std::atomic<uint64_t> unhandled_filter{0};
    std::mutex exports_mutex;
    bool exports_registered = false;
};

Registry& registry() {
    static Registry instance;
    return instance;
}

std::unique_ptr<Table> make_table(VirtualMemoryArena& arena,
                                  uint64_t base,
                                  uint64_t table_va,
                                  uint32_t count,
                                  bool image) {
    if (count == 0 || count > kMaxTableEntries) {
        return nullptr;
    }
    std::vector<GuestRuntimeFunction> raw(count);
    if (!arena.read(table_va, raw.data(), raw.size() * sizeof(GuestRuntimeFunction))) {
        return nullptr;
    }
    auto table = std::make_unique<Table>();
    table->base = base;
    table->table_va = table_va;
    table->image = image;
    table->index.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        table->index[i] = i;
    }
    std::stable_sort(table->index.begin(), table->index.end(),
                     [&](uint32_t a, uint32_t b) { return raw[a].begin < raw[b].begin; });
    table->entries.reserve(count);
    uint32_t lo = UINT32_MAX;
    uint32_t hi = 0;
    for (uint32_t i : table->index) {
        table->entries.push_back(raw[i]);
        lo = std::min(lo, raw[i].begin);
        hi = std::max(hi, raw[i].end);
    }
    table->lo = base + lo;
    table->hi = base + hi;
    return table;
}

// Guest VA of the entry covering pc in `t` (not following indirection), or 0.
uint64_t find_in_table(const Table& t, uint64_t pc) {
    if (pc < t.lo || pc >= t.hi) {
        return 0;
    }
    const uint64_t rva = pc - t.base;
    auto it = std::upper_bound(t.entries.begin(), t.entries.end(), rva,
                               [](uint64_t v, const GuestRuntimeFunction& e) { return v < e.begin; });
    if (it == t.entries.begin()) {
        return 0;
    }
    --it;
    if (rva >= it->end) {
        return 0;
    }
    const size_t pos = static_cast<size_t>(it - t.entries.begin());
    return t.table_va + uint64_t{12} * t.index[pos];
}

// Follows indirect entries (UnwindData bit 0) to the entry holding the unwind data.
bool resolve_entry(VirtualMemoryArena& arena, uint64_t image_base, uint64_t& entry_va, GuestRuntimeFunction& entry) {
    if (!read_guest(arena, entry_va, entry)) {
        return false;
    }
    for (int depth = 0; (entry.unwind & 1u) != 0; ++depth) {
        if (depth >= kMaxChainDepth) {
            return false;
        }
        entry_va = image_base + (entry.unwind & ~1u);
        if (!read_guest(arena, entry_va, entry)) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Flags and contexts
// ---------------------------------------------------------------------------

constexpr uint32_t kEflagsBits[7] = {0, 2, 4, 6, 7, 11, 10};  // CF PF AF ZF SF OF DF

uint32_t eflags_from_state(const GuestState& state) {
    uint32_t eflags = 0x202;  // reserved bit 1, IF
    for (uint32_t i = 0; i < 7; ++i) {
        if (state.flags[i] != 0) {
            eflags |= 1u << kEflagsBits[i];
        }
    }
    return eflags;
}

// ---------------------------------------------------------------------------
// Unwind info
// ---------------------------------------------------------------------------

struct UnwindInfoView {
    uint8_t version = 0;
    uint8_t flags = 0;
    uint8_t prolog_size = 0;
    uint8_t count = 0;
    uint8_t frame_register = 0;
    uint8_t frame_offset = 0;  // unscaled
    uint16_t codes[256] = {};
    uint64_t trailer_va = 0;  // handler RVA or chained RUNTIME_FUNCTION
};

bool read_unwind_info(VirtualMemoryArena& arena, uint64_t image_base, uint32_t rva, UnwindInfoView& out) {
    const uint64_t va = image_base + rva;
    uint8_t header[4];
    if (!arena.read(va, header, sizeof(header))) {
        return false;
    }
    out.version = header[0] & 0x7;
    out.flags = header[0] >> 3;
    out.prolog_size = header[1];
    out.count = header[2];
    out.frame_register = header[3] & 0xF;
    out.frame_offset = header[3] >> 4;
    if (out.version != 1 && out.version != 2) {
        return false;
    }
    if (out.count != 0 && !arena.read(va + 4, out.codes, size_t{2} * out.count)) {
        return false;
    }
    out.trailer_va = va + 4 + 2 * ((uint64_t{out.count} + 1) & ~uint64_t{1});
    return true;
}

// Slots used by an unwind code, 0 if invalid.
int code_slots(uint8_t op, uint8_t info, uint8_t version) {
    switch (op) {
        case kUwopPushNonvol:
        case kUwopAllocSmall:
        case kUwopSetFpreg:
            return 1;
        case kUwopAllocLarge:
            return info == 0 ? 2 : (info == 1 ? 3 : 0);
        case kUwopSaveNonvol:
        case kUwopSaveXmm128:
            return 2;
        case kUwopSaveNonvolFar:
        case kUwopSaveXmm128Far:
            return 3;
        case kUwopEpilog:
            return version == 2 ? 1 : 0;
        case kUwopPushMachframe:
            return info <= 1 ? 1 : 0;
        default:
            return 0;
    }
}

// ---------------------------------------------------------------------------
// Epilogs
// ---------------------------------------------------------------------------

struct CodeBytes {
    VirtualMemoryArena& arena;
    bool get(uint64_t va, uint8_t& out) const { return arena.read(va, &out, 1); }
    bool get32(uint64_t va, int32_t& out) const { return arena.read(va, &out, 4); }
};

bool inside_function(uint64_t va, uint64_t image_base, const GuestRuntimeFunction& fe) {
    return va >= image_base + fe.begin && va < image_base + fe.end;
}

// Whether the code at pc is an epilog of `fe` (Windows x64 epilog rules).
bool is_epilog(VirtualMemoryArena& arena, uint64_t pc, uint64_t image_base, const GuestRuntimeFunction& fe) {
    const CodeBytes code{arena};
    uint8_t b0 = 0;
    uint8_t b1 = 0;
    uint8_t b2 = 0;
    if (!code.get(pc, b0)) {
        return false;
    }
    // Optional stack deallocation (with REX.W): add rsp, imm8/imm32 or lea rsp, [reg+disp].
    if ((b0 & 0xF8) == 0x48 && code.get(pc + 1, b1)) {
        if (b1 == 0x81 || b1 == 0x83) {
            if (b0 == 0x48 && code.get(pc + 2, b2) && b2 == 0xC4) {
                pc += b1 == 0x81 ? 7 : 4;
            } else {
                return false;
            }
        } else if (b1 == 0x8D) {
            if ((b0 & 0x06) != 0 || !code.get(pc + 2, b2)) {
                return false;
            }
            if (((b2 >> 3) & 7) != 4 || (b2 & 7) == 4) {
                return false;  // destination must be rsp, no SIB
            }
            if ((b2 >> 6) == 1) {
                pc += 4;
            } else if ((b2 >> 6) == 2) {
                pc += 7;
            } else {
                return false;
            }
        }
    }
    for (int steps = 0; steps < 64; ++steps) {
        uint8_t op = 0;
        if (!code.get(pc, op)) {
            return false;
        }
        uint64_t at = pc;
        if ((op & 0xF0) == 0x40) {  // REX
            ++at;
            if (!code.get(at, op)) {
                return false;
            }
        }
        if (op >= 0x58 && op <= 0x5F) {  // pop reg
            pc = at + 1;
            continue;
        }
        switch (op) {
            case 0xC2:
            case 0xC3:
                return true;
            case 0xF3: {  // rep ret
                uint8_t next = 0;
                return code.get(at + 1, next) && next == 0xC3;
            }
            case 0xE9:
            case 0xEB: {
                int32_t disp = 0;
                if (op == 0xE9) {
                    if (!code.get32(at + 1, disp)) {
                        return false;
                    }
                    pc = at + 5 + static_cast<int64_t>(disp);
                } else {
                    uint8_t d8 = 0;
                    if (!code.get(at + 1, d8)) {
                        return false;
                    }
                    pc = at + 2 + static_cast<int8_t>(d8);
                }
                if (!inside_function(pc, image_base, fe)) {
                    return true;  // tail call out of the function
                }
                continue;
            }
            case 0xFF: {  // jmp [rip+disp32] (tail call through a pointer)
                uint8_t modrm = 0;
                return code.get(at + 1, modrm) && modrm == 0x25;
            }
            default:
                return false;
        }
    }
    return false;
}

bool pop_into(VirtualMemoryArena& arena, GuestContext& ctx, GuestContextPointers* ptrs, unsigned reg) {
    const uint64_t at = ctx.gpr[GuestReg::Rsp];
    uint64_t value = 0;
    if (!read_u64(arena, at, value)) {
        return false;
    }
    ctx.gpr[reg] = value;
    if (ptrs != nullptr) {
        ptrs->gpr[reg] = at;
    }
    ctx.gpr[GuestReg::Rsp] = at + 8;
    return true;
}

bool pop_return(VirtualMemoryArena& arena, GuestContext& ctx, uint64_t extra = 0) {
    uint64_t rip = 0;
    if (!read_u64(arena, ctx.gpr[GuestReg::Rsp], rip)) {
        return false;
    }
    ctx.rip = rip;
    ctx.gpr[GuestReg::Rsp] += 8 + extra;
    return true;
}

// Executes the rest of an epilog starting at pc on the context (is_epilog said yes).
bool interpret_epilog(VirtualMemoryArena& arena,
                      uint64_t pc,
                      uint64_t image_base,
                      const GuestRuntimeFunction& fe,
                      GuestContext& ctx,
                      GuestContextPointers* ptrs) {
    const CodeBytes code{arena};
    for (int steps = 0; steps < 96; ++steps) {
        uint8_t op = 0;
        if (!code.get(pc, op)) {
            return false;
        }
        uint8_t rex = 0;
        if ((op & 0xF0) == 0x40) {
            rex = op & 0x0F;
            ++pc;
            if (!code.get(pc, op)) {
                return false;
            }
        }
        if (op >= 0x58 && op <= 0x5F) {
            if (!pop_into(arena, ctx, ptrs, (op - 0x58u) + ((rex & 1u) << 3))) {
                return false;
            }
            ++pc;
            continue;
        }
        uint8_t modrm = 0;
        int32_t disp = 0;
        switch (op) {
            case 0x81:  // add rsp, imm32
                if (!code.get32(pc + 2, disp)) {
                    return false;
                }
                ctx.gpr[GuestReg::Rsp] += static_cast<int64_t>(disp);
                pc += 6;
                continue;
            case 0x83: {  // add rsp, imm8
                uint8_t imm = 0;
                if (!code.get(pc + 2, imm)) {
                    return false;
                }
                ctx.gpr[GuestReg::Rsp] += static_cast<int8_t>(imm);
                pc += 3;
                continue;
            }
            case 0x8D: {  // lea rsp, [reg+disp8/disp32]
                if (!code.get(pc + 1, modrm)) {
                    return false;
                }
                const unsigned base_reg = (modrm & 7u) + ((rex & 1u) << 3);
                if ((modrm >> 6) == 1) {
                    uint8_t d8 = 0;
                    if (!code.get(pc + 2, d8)) {
                        return false;
                    }
                    ctx.gpr[GuestReg::Rsp] = ctx.gpr[base_reg] + static_cast<int8_t>(d8);
                    pc += 3;
                } else {
                    if (!code.get32(pc + 2, disp)) {
                        return false;
                    }
                    ctx.gpr[GuestReg::Rsp] = ctx.gpr[base_reg] + static_cast<int64_t>(disp);
                    pc += 6;
                }
                continue;
            }
            case 0xC2: {  // ret imm16
                uint8_t lo = 0;
                uint8_t hi = 0;
                if (!code.get(pc + 1, lo) || !code.get(pc + 2, hi)) {
                    return false;
                }
                return pop_return(arena, ctx, static_cast<uint64_t>(lo | (hi << 8)));
            }
            case 0xC3:
            case 0xF3:  // rep ret
            case 0xFF:  // jmp [rip+disp32]: tail call
                return pop_return(arena, ctx);
            case 0xE9:
            case 0xEB: {
                uint64_t target = 0;
                if (op == 0xE9) {
                    if (!code.get32(pc + 1, disp)) {
                        return false;
                    }
                    target = pc + 5 + static_cast<int64_t>(disp);
                } else {
                    uint8_t d8 = 0;
                    if (!code.get(pc + 1, d8)) {
                        return false;
                    }
                    target = pc + 2 + static_cast<int8_t>(d8);
                }
                if (!inside_function(target, image_base, fe)) {
                    return pop_return(arena, ctx);  // tail call
                }
                pc = target;
                continue;
            }
            default:
                return false;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// call_guest boundaries
// ---------------------------------------------------------------------------

enum class TagKind : uint64_t {
    Dispatch = 0x5047'4558'4450'5348ULL,
    Unwind = 0x5047'4558'554E'5744ULL,
    Consolidate = 0x5047'4558'434F'4E53ULL,
};

// Tag of the call_guest activations made by this file (see the header comment).
struct BoundaryTag {
    TagKind kind;
    GuestContext continuation;     // context of the frame above the boundary
    uint64_t establisher = 0;      // dispatch: frame whose handler runs
    uint64_t dispatcher_context = 0;  // unwind: DISPATCHER_CONTEXT of that unwind (guest VA)
};

const BoundaryTag* as_tag(const void* tag) {
    if (tag == nullptr) {
        return nullptr;
    }
    const auto* t = static_cast<const BoundaryTag*>(tag);
    switch (t->kind) {
        case TagKind::Dispatch:
        case TagKind::Unwind:
        case TagKind::Consolidate:
            return t;
    }
    return nullptr;
}

enum class Crossing { Continue, End, Collided };

// The walk reached kGuestReturnSentinel: continue in the frame above the call_guest boundary.
Crossing cross_boundary(VirtualMemoryArena& arena,
                        GuestContext& cur,
                        bool dispatching,
                        uint32_t& flags,
                        uint64_t& nested_frame,
                        const BoundaryTag** collided) {
    const GuestCallBoundary* b = innermost_guest_call();
    while (b != nullptr && b->sentinel_rsp + 8 != cur.gpr[GuestReg::Rsp]) {
        b = b->outer;
    }
    if (b == nullptr || b->outer == nullptr || b->tag == top_level_call_tag()) {
        return Crossing::End;  // a thread's outermost call, or a top-level callback
    }
    const BoundaryTag* tag = as_tag(b->tag);
    if (tag == nullptr) {
        uint64_t rip = 0;
        if (!read_u64(arena, b->caller_rsp, rip)) {
            return Crossing::End;
        }
        cur.rip = rip;
        cur.gpr[GuestReg::Rsp] = b->caller_rsp + 8;
        return Crossing::Continue;
    }
    switch (tag->kind) {
        case TagKind::Dispatch:
            if (dispatching) {
                flags |= kExceptionNestedCall;
                nested_frame = std::max(nested_frame, tag->establisher);
            }
            break;
        case TagKind::Unwind:
            if (!dispatching) {
                *collided = tag;
                return Crossing::Collided;
            }
            break;
        case TagKind::Consolidate:
            break;
    }
    cur = tag->continuation;
    return Crossing::Continue;
}

// Guest stack space below `top`, allocated downwards, 16-byte aligned.
struct Scratch {
    uint64_t top;
    uint64_t alloc(uint64_t size) {
        top = os::align_down(top - size, 16);
        return top;
    }
};

// Calls a guest routine below the scratch area. Returns rax.
uint64_t call_with(VirtualMemoryArena& arena,
                   GuestState& state,
                   const Scratch& scratch,
                   uint64_t va,
                   std::initializer_list<uint64_t> args,
                   const BoundaryTag* tag) {
    state.gpr[GuestReg::Rsp] = scratch.top;
    return call_guest_tagged(arena, state, va, std::span<const uint64_t>(args.begin(), args.size()), tag);
}

[[noreturn]] void unwind_failed(GuestState& state, uint64_t target_ip, uint64_t target_frame) {
    raise_fault(FaultKind::UnwindFailed, &state, target_ip, target_frame);
}

// Final step of every restore: apply the context, in place or through the resume protocol.
RestoreResult finish_restore(VirtualMemoryArena& arena, GuestState& state, uint64_t thunk_rsp, const GuestContext& ctx) {
    uint64_t return_address = 0;
    const bool in_place = read_u64(arena, thunk_rsp, return_address) && ctx.rip == return_address &&
                          ctx.gpr[GuestReg::Rsp] == thunk_rsp + 8;
    apply_context(ctx, state);
    state.rip = ctx.rip;
    if (in_place) {
        return RestoreResult::InPlace;
    }
    state.resume_rip = ctx.rip;
    state.unwind_pending = 1;
    return RestoreResult::Pending;
}

// RtlRestoreContext with the thunk's entry rsp (the return address of the import call).
RestoreResult restore_at(VirtualMemoryArena& arena,
                         GuestState& state,
                         uint64_t thunk_rsp,
                         GuestContext ctx,
                         uint64_t record_va) {
    GuestExceptionRecord rec{};
    if (record_va != 0 && read_guest(arena, record_va, rec) && rec.number_parameters >= 1) {
        if (rec.code == kStatusLongjump) {
            GuestJumpBuffer jb{};
            if (!read_guest(arena, rec.information[0], jb)) {
                unwind_failed(state, ctx.rip, rec.information[0]);
            }
            ctx.gpr[GuestReg::Rbx] = jb.rbx;
            ctx.gpr[GuestReg::Rsp] = jb.rsp;
            ctx.gpr[GuestReg::Rbp] = jb.rbp;
            ctx.gpr[GuestReg::Rsi] = jb.rsi;
            ctx.gpr[GuestReg::Rdi] = jb.rdi;
            ctx.gpr[GuestReg::R12] = jb.r12;
            ctx.gpr[GuestReg::R13] = jb.r13;
            ctx.gpr[GuestReg::R14] = jb.r14;
            ctx.gpr[GuestReg::R15] = jb.r15;
            ctx.rip = jb.rip;
            ctx.mx_csr = jb.mx_csr;
            for (int i = 0; i < 10; ++i) {
                ctx.xmm[6 + i] = jb.xmm[i];
            }
        } else if (rec.code == kStatusUnwindConsolidate) {
            // The callback (e.g. a C++ catch block) runs on top of the current, already unwound
            // stack; a frame walk from inside it continues at the target frame.
            auto tag = std::make_unique<BoundaryTag>();
            tag->kind = TagKind::Consolidate;
            tag->continuation = ctx;
            const Scratch scratch{state.gpr[GuestReg::Rsp]};
            const uint64_t rip = call_with(arena, state, scratch, rec.information[0], {record_va}, tag.get());
            if (state.unwind_pending != 0) {
                return RestoreResult::Pending;
            }
            ctx.rip = rip;
        }
    }
    return finish_restore(arena, state, thunk_rsp, ctx);
}

enum class DispatchOutcome { Continue, Unhandled, Pending };

// RtlDispatchException: walks the frames from `raise` and calls their exception handlers.
DispatchOutcome dispatch(VirtualMemoryArena& arena,
                         GuestState& state,
                         Scratch& scratch,
                         uint64_t record_va,
                         uint64_t context_va,
                         const GuestContext& raise) {
    GuestExceptionRecord rec{};
    if (!read_guest(arena, record_va, rec)) {
        return DispatchOutcome::Unhandled;
    }
    auto tag = std::make_unique<BoundaryTag>();
    tag->kind = TagKind::Dispatch;
    tag->continuation = raise;
    const uint64_t walk_va = scratch.alloc(sizeof(GuestContext));
    const uint64_t dc_va = scratch.alloc(sizeof(GuestDispatcherContext));
    tag->dispatcher_context = dc_va;

    GuestContext cur = raise;
    uint32_t flags = rec.flags;
    uint64_t nested_frame = 0;
    for (int frames = 0; frames < kMaxFrames; ++frames) {
        if (cur.rip == kGuestReturnSentinel) {
            const BoundaryTag* collided = nullptr;
            if (cross_boundary(arena, cur, true, flags, nested_frame, &collided) != Crossing::Continue) {
                return DispatchOutcome::Unhandled;
            }
            continue;
        }
        if (cur.rip == 0) {
            return DispatchOutcome::Unhandled;
        }
        uint64_t image_base = 0;
        const uint64_t entry = lookup_function_entry(arena, cur.rip, &image_base);
        if (entry == 0) {
            if (!pop_return(arena, cur)) {
                return DispatchOutcome::Unhandled;
            }
            continue;
        }
        const uint64_t control_pc = cur.rip;
        uint64_t handler_data = 0;
        uint64_t establisher = 0;
        bool ok = true;
        const uint64_t handler = virtual_unwind(arena, kUnwFlagEHandler, image_base, control_pc, entry, cur,
                                                &handler_data, &establisher, nullptr, &ok);
        if (!ok || (establisher & 7) != 0) {
            write_guest(arena, record_va + offsetof(GuestExceptionRecord, flags), flags | kExceptionStackInvalid);
            return DispatchOutcome::Unhandled;
        }
        if (handler == 0) {
            continue;
        }
        // DispatcherContext.ContextRecord holds the context unwound past this frame (Windows).
        write_guest(arena, walk_va, cur);
        const GuestDispatcherContext dc{control_pc, image_base, entry, establisher, 0, walk_va, handler,
                                        handler_data, 0, 0, 0};
        write_guest(arena, dc_va, dc);
        write_guest(arena, record_va + offsetof(GuestExceptionRecord, flags), flags);
        tag->establisher = establisher;
        const auto disposition = static_cast<uint32_t>(
            call_with(arena, state, scratch, handler, {record_va, establisher, context_va, dc_va}, tag.get()));
        if (state.unwind_pending != 0) {
            return DispatchOutcome::Pending;
        }
        if (nested_frame == establisher) {
            flags &= ~kExceptionNestedCall;
            nested_frame = 0;
        }
        switch (disposition) {
            case kExceptionContinueExecution: {
                uint32_t current = flags;
                read_guest(arena, record_va + offsetof(GuestExceptionRecord, flags), current);
                if ((current & kExceptionNoncontinuable) != 0) {
                    raise_fault(FaultKind::UnhandledException, &state, rec.address, kStatusNoncontinuableException);
                }
                return DispatchOutcome::Continue;
            }
            case kExceptionContinueSearch:
                break;
            case kExceptionNestedException: {
                flags |= kExceptionNestedCall;
                GuestDispatcherContext back{};
                if (read_guest(arena, dc_va, back) && back.establisher_frame > nested_frame) {
                    nested_frame = back.establisher_frame;
                }
                break;
            }
            default:
                raise_fault(FaultKind::UnhandledException, &state, rec.address, kStatusInvalidDisposition);
        }
    }
    return DispatchOutcome::Unhandled;
}

// The top-level filter for an exception nobody handled. Returns true to continue execution.
bool run_unhandled_filter(VirtualMemoryArena& arena,
                          GuestState& state,
                          Scratch& scratch,
                          uint64_t record_va,
                          uint64_t context_va,
                          bool* pending) {
    *pending = false;
    const uint64_t filter = registry().unhandled_filter.load(std::memory_order_acquire);
    if (filter == 0) {
        return false;
    }
    const uint64_t pointers_va = scratch.alloc(sizeof(GuestExceptionPointers));
    write_guest(arena, pointers_va, GuestExceptionPointers{record_va, context_va});
    const auto result = static_cast<int32_t>(call_with(arena, state, scratch, filter, {pointers_va}, nullptr));
    if (state.unwind_pending != 0) {
        *pending = true;
        return false;
    }
    return result == kExceptionContinueExecutionFilter;
}

// ---------------------------------------------------------------------------
// Thunks
// ---------------------------------------------------------------------------

VirtualMemoryArena& process_arena() {
    return *os::win32::get_vmem_arena();
}

GuestState& as_state(void* state) {
    return *static_cast<GuestState*>(state);
}

uint64_t arg(VirtualMemoryArena& arena, const GuestState& s, unsigned index) {
    static constexpr uint32_t kArgRegs[4] = {GuestReg::Rcx, GuestReg::Rdx, GuestReg::R8, GuestReg::R9};
    if (index < 4) {
        return s.gpr[kArgRegs[index]];
    }
    uint64_t value = 0;
    read_u64(arena, s.gpr[GuestReg::Rsp] + kGuestStackArgOffset + 8 * (index - 4), value);
    return value;
}

void finish(GuestState& s, uint64_t rax) {
    s.gpr[GuestReg::Rax] = rax;
    s.gpr[GuestReg::Rsp] += 8;
}

void thunk_rtl_capture_context(void* p) {
    GuestState& s = as_state(p);
    VirtualMemoryArena& arena = process_arena();
    GuestContext ctx{};
    if (capture_caller_context(arena, s, ctx)) {
        write_guest(arena, s.gpr[GuestReg::Rcx], ctx);
    }
    s.gpr[GuestReg::Rsp] += 8;
}

void thunk_rtl_lookup_function_entry(void* p) {
    GuestState& s = as_state(p);
    VirtualMemoryArena& arena = process_arena();
    uint64_t base = 0;
    const uint64_t entry = lookup_function_entry(arena, arg(arena, s, 0), &base);
    if (arg(arena, s, 1) != 0) {
        write_guest(arena, arg(arena, s, 1), entry != 0 ? base : uint64_t{0});
    }
    finish(s, entry);
}

void thunk_rtl_virtual_unwind(void* p) {
    GuestState& s = as_state(p);
    VirtualMemoryArena& arena = process_arena();
    const uint64_t ctx_va = arg(arena, s, 4);
    const uint64_t data_ptr = arg(arena, s, 5);
    const uint64_t frame_ptr = arg(arena, s, 6);
    const uint64_t pointers_va = arg(arena, s, 7);
    GuestContext ctx{};
    if (!read_guest(arena, ctx_va, ctx)) {
        finish(s, 0);
        return;
    }
    GuestContextPointers pointers{};
    if (pointers_va != 0) {
        read_guest(arena, pointers_va, pointers);
    }
    uint64_t handler_data = 0;
    uint64_t establisher = 0;
    const uint64_t handler =
        virtual_unwind(arena, static_cast<uint32_t>(arg(arena, s, 0)), arg(arena, s, 1), arg(arena, s, 2),
                       arg(arena, s, 3), ctx, &handler_data, &establisher, pointers_va != 0 ? &pointers : nullptr);
    write_guest(arena, ctx_va, ctx);
    if (data_ptr != 0) {
        write_guest(arena, data_ptr, handler_data);
    }
    if (frame_ptr != 0) {
        write_guest(arena, frame_ptr, establisher);
    }
    if (pointers_va != 0) {
        write_guest(arena, pointers_va, pointers);
    }
    finish(s, handler);
}

void thunk_rtl_pc_to_file_header(void* p) {
    GuestState& s = as_state(p);
    VirtualMemoryArena& arena = process_arena();
    const uint64_t base = pc_to_image_base(arg(arena, s, 0));
    if (arg(arena, s, 1) != 0) {
        write_guest(arena, arg(arena, s, 1), base);
    }
    finish(s, base);
}

void thunk_rtl_add_function_table(void* p) {
    GuestState& s = as_state(p);
    VirtualMemoryArena& arena = process_arena();
    finish(s, add_function_table(arena, arg(arena, s, 0), static_cast<uint32_t>(arg(arena, s, 1)),
                                 arg(arena, s, 2))
                  ? 1
                  : 0);
}

void thunk_rtl_delete_function_table(void* p) {
    GuestState& s = as_state(p);
    finish(s, delete_function_table(s.gpr[GuestReg::Rcx]) ? 1 : 0);
}

// Both outcomes leave the state as the caller must see it: InPlace continued at the raise site
// (rsp already past the return address), Pending propagates the restore.
void raise_from_thunk(GuestState& s, VirtualMemoryArena& arena, const GuestExceptionRecord& rec) {
    (void)raise_exception(arena, s, rec);
}

void thunk_raise_exception(void* p) {
    GuestState& s = as_state(p);
    VirtualMemoryArena& arena = process_arena();
    GuestExceptionRecord rec{};
    rec.code = static_cast<uint32_t>(arg(arena, s, 0));
    rec.flags = static_cast<uint32_t>(arg(arena, s, 1)) & kExceptionNoncontinuable;
    const uint64_t count = std::min<uint64_t>(static_cast<uint32_t>(arg(arena, s, 2)), kExceptionMaximumParameters);
    const uint64_t args = arg(arena, s, 3);
    if (args != 0 && count != 0) {
        arena.read(args, rec.information, count * sizeof(uint64_t));
        rec.number_parameters = static_cast<uint32_t>(count);
    }
    raise_from_thunk(s, arena, rec);
}

void thunk_rtl_raise_exception(void* p) {
    GuestState& s = as_state(p);
    VirtualMemoryArena& arena = process_arena();
    GuestExceptionRecord rec{};
    if (!read_guest(arena, s.gpr[GuestReg::Rcx], rec)) {
        raise_fault(FaultKind::UnhandledException, &s, 0, kStatusBadStack);
    }
    rec.number_parameters = std::min(rec.number_parameters, kExceptionMaximumParameters);
    raise_from_thunk(s, arena, rec);
}

void thunk_rtl_unwind_ex(void* p) {
    GuestState& s = as_state(p);
    VirtualMemoryArena& arena = process_arena();
    unwind(arena, s, arg(arena, s, 0), arg(arena, s, 1), arg(arena, s, 2), arg(arena, s, 3), arg(arena, s, 4),
           arg(arena, s, 5));
}

void thunk_rtl_unwind(void* p) {
    GuestState& s = as_state(p);
    VirtualMemoryArena& arena = process_arena();
    unwind(arena, s, arg(arena, s, 0), arg(arena, s, 1), arg(arena, s, 2), arg(arena, s, 3), 0, 0);
}

void thunk_rtl_restore_context(void* p) {
    GuestState& s = as_state(p);
    VirtualMemoryArena& arena = process_arena();
    GuestContext ctx{};
    if (!read_guest(arena, s.gpr[GuestReg::Rcx], ctx)) {
        unwind_failed(s, 0, s.gpr[GuestReg::Rcx]);
    }
    restore_context(arena, s, ctx, s.gpr[GuestReg::Rdx]);
}

void thunk_set_unhandled_exception_filter(void* p) {
    GuestState& s = as_state(p);
    finish(s, set_unhandled_exception_filter(s.gpr[GuestReg::Rcx]));
}

void thunk_unhandled_exception_filter(void* p) {
    GuestState& s = as_state(p);
    VirtualMemoryArena& arena = process_arena();
    const uint64_t pointers_va = s.gpr[GuestReg::Rcx];
    GuestExceptionPointers pointers{};
    GuestExceptionRecord rec{};
    if (read_guest(arena, pointers_va, pointers)) {
        read_guest(arena, pointers.exception_record, rec);
    }
    const uint64_t filter = registry().unhandled_filter.load(std::memory_order_acquire);
    if (filter != 0) {
        const auto result = static_cast<int32_t>(call_guest_args(arena, s, filter, pointers_va));
        if (s.unwind_pending != 0) {
            return;
        }
        if (result == kExceptionContinueExecutionFilter) {
            finish(s, static_cast<uint64_t>(static_cast<int64_t>(kExceptionContinueExecutionFilter)));
            return;
        }
    }
    raise_fault(FaultKind::UnhandledException, &s, rec.address, rec.code);
}

constexpr ExceptionExport kExports[] = {
    {"KERNEL32.dll", "RaiseException", &thunk_raise_exception},
    {"KERNEL32.dll", "RtlCaptureContext", &thunk_rtl_capture_context},
    {"KERNEL32.dll", "RtlLookupFunctionEntry", &thunk_rtl_lookup_function_entry},
    {"KERNEL32.dll", "RtlVirtualUnwind", &thunk_rtl_virtual_unwind},
    {"KERNEL32.dll", "RtlUnwindEx", &thunk_rtl_unwind_ex},
    {"KERNEL32.dll", "RtlUnwind", &thunk_rtl_unwind},
    {"KERNEL32.dll", "RtlRestoreContext", &thunk_rtl_restore_context},
    {"KERNEL32.dll", "RtlPcToFileHeader", &thunk_rtl_pc_to_file_header},
    {"KERNEL32.dll", "RtlAddFunctionTable", &thunk_rtl_add_function_table},
    {"KERNEL32.dll", "RtlDeleteFunctionTable", &thunk_rtl_delete_function_table},
    {"KERNEL32.dll", "SetUnhandledExceptionFilter", &thunk_set_unhandled_exception_filter},
    {"KERNEL32.dll", "UnhandledExceptionFilter", &thunk_unhandled_exception_filter},
    {"ntdll.dll", "RtlRaiseException", &thunk_rtl_raise_exception},
    {"ntdll.dll", "RtlCaptureContext", &thunk_rtl_capture_context},
    {"ntdll.dll", "RtlLookupFunctionEntry", &thunk_rtl_lookup_function_entry},
    {"ntdll.dll", "RtlVirtualUnwind", &thunk_rtl_virtual_unwind},
    {"ntdll.dll", "RtlUnwindEx", &thunk_rtl_unwind_ex},
    {"ntdll.dll", "RtlUnwind", &thunk_rtl_unwind},
    {"ntdll.dll", "RtlRestoreContext", &thunk_rtl_restore_context},
    {"ntdll.dll", "RtlPcToFileHeader", &thunk_rtl_pc_to_file_header},
    {"ntdll.dll", "RtlAddFunctionTable", &thunk_rtl_add_function_table},
    {"ntdll.dll", "RtlDeleteFunctionTable", &thunk_rtl_delete_function_table},
};

bool equals_ignore_case(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) {
            return false;
        }
    }
    return true;
}

// "kernel32" and "kernel32.dll" name the same module.
bool same_module(std::string_view a, std::string_view b) {
    auto strip = [](std::string_view m) {
        if (m.size() > 4 && equals_ignore_case(m.substr(m.size() - 4), ".dll")) {
            m.remove_suffix(4);
        }
        return m;
    };
    return equals_ignore_case(strip(a), strip(b));
}

}  // namespace

// ---------------------------------------------------------------------------
// Function tables
// ---------------------------------------------------------------------------

bool register_image_function_table(VirtualMemoryArena& arena,
                                   uint64_t image_base,
                                   uint64_t image_size,
                                   uint64_t table_va,
                                   uint32_t count) {
    if (image_base == 0 || image_size == 0 || image_base + image_size < image_base) {
        return false;
    }
    std::unique_ptr<Table> table = make_table(arena, image_base, table_va, count, true);
    if (table == nullptr) {
        return false;
    }
    table->lo = image_base;
    table->hi = image_base + image_size;
    Registry& reg = registry();
    std::unique_lock lock(reg.mutex);
    for (const auto& t : reg.tables) {
        if (t->lo < table->hi && table->lo < t->hi) {
            return false;
        }
    }
    reg.tables.push_back(std::move(table));
    return true;
}

bool register_image_from_headers(VirtualMemoryArena& arena, uint64_t image_base) {
    uint32_t e_lfanew = 0;
    uint32_t signature = 0;
    uint16_t magic = 0;
    if (!read_guest(arena, image_base + 0x3C, e_lfanew) || !read_guest(arena, image_base + e_lfanew, signature) ||
        signature != 0x00004550) {
        return false;
    }
    const uint64_t optional = image_base + e_lfanew + 4 + 20;
    uint32_t size_of_image = 0;
    uint32_t directory_count = 0;
    uint32_t pdata_rva = 0;
    uint32_t pdata_size = 0;
    if (!read_guest(arena, optional, magic) || magic != 0x20B || !read_guest(arena, optional + 56, size_of_image) ||
        !read_guest(arena, optional + 108, directory_count) || directory_count <= 3 ||
        !read_guest(arena, optional + 112 + 3 * 8, pdata_rva) ||
        !read_guest(arena, optional + 112 + 3 * 8 + 4, pdata_size)) {
        return false;
    }
    return register_image_function_table(arena, image_base, size_of_image, image_base + pdata_rva,
                                         pdata_size / static_cast<uint32_t>(sizeof(GuestRuntimeFunction)));
}

bool add_function_table(VirtualMemoryArena& arena, uint64_t table_va, uint32_t count, uint64_t base) {
    std::unique_ptr<Table> table = make_table(arena, base, table_va, count, false);
    if (table == nullptr) {
        return false;
    }
    Registry& reg = registry();
    std::unique_lock lock(reg.mutex);
    for (const auto& t : reg.tables) {
        if (!t->image && t->table_va == table_va) {
            return false;
        }
    }
    reg.tables.push_back(std::move(table));
    return true;
}

bool delete_function_table(uint64_t table_va) {
    Registry& reg = registry();
    std::unique_lock lock(reg.mutex);
    for (auto it = reg.tables.begin(); it != reg.tables.end(); ++it) {
        if (!(*it)->image && (*it)->table_va == table_va) {
            reg.tables.erase(it);
            return true;
        }
    }
    return false;
}

uint64_t lookup_function_entry(VirtualMemoryArena& arena, uint64_t pc, uint64_t* image_base) {
    uint64_t entry_va = 0;
    uint64_t base = 0;
    {
        Registry& reg = registry();
        std::shared_lock lock(reg.mutex);
        for (const auto& t : reg.tables) {
            if (pc >= t->lo && pc < t->hi) {
                entry_va = find_in_table(*t, pc);
                base = t->base;
                if (entry_va != 0) {
                    break;
                }
            }
        }
    }
    GuestRuntimeFunction entry{};
    if (entry_va == 0 || !resolve_entry(arena, base, entry_va, entry)) {
        return 0;
    }
    if (image_base != nullptr) {
        *image_base = base;
    }
    return entry_va;
}

uint64_t pc_to_image_base(uint64_t pc) {
    Registry& reg = registry();
    std::shared_lock lock(reg.mutex);
    for (const auto& t : reg.tables) {
        if (t->image && pc >= t->lo && pc < t->hi) {
            return t->base;
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Contexts and virtual unwinding
// ---------------------------------------------------------------------------

bool capture_caller_context(VirtualMemoryArena& arena, const GuestState& state, GuestContext& out) {
    out = GuestContext{};
    uint64_t rip = 0;
    if (!read_u64(arena, state.gpr[GuestReg::Rsp], rip)) {
        return false;
    }
    out.context_flags = kContextFull | kContextSegments;
    std::memcpy(out.gpr, state.gpr, sizeof(out.gpr));
    out.gpr[GuestReg::Rsp] = state.gpr[GuestReg::Rsp] + 8;
    out.rip = rip;
    out.eflags = eflags_from_state(state);
    out.mx_csr = state.mxcsr;
    out.flt_mx_csr = state.mxcsr;
    out.flt_mx_csr_mask = 0xFFFF;
    out.control_word = 0x27F;
    out.seg_cs = 0x33;
    out.seg_ds = out.seg_es = out.seg_ss = out.seg_gs = 0x2B;
    out.seg_fs = 0x53;
    for (int i = 0; i < 16; ++i) {
        out.xmm[i] = GuestM128{state.xmm[i][0], state.xmm[i][1]};
    }
    return true;
}

void apply_context(const GuestContext& context, GuestState& state) {
    std::memcpy(state.gpr, context.gpr, sizeof(state.gpr));
    for (uint32_t i = 0; i < 7; ++i) {
        state.flags[i] = static_cast<uint8_t>((context.eflags >> kEflagsBits[i]) & 1u);
    }
    state.mxcsr = context.mx_csr;
    for (int i = 0; i < 16; ++i) {
        state.xmm[i][0] = context.xmm[i].low;
        state.xmm[i][1] = context.xmm[i].high;
    }
}

uint64_t virtual_unwind(VirtualMemoryArena& arena,
                        uint32_t handler_type,
                        uint64_t image_base,
                        uint64_t control_pc,
                        uint64_t function_entry,
                        GuestContext& ctx,
                        uint64_t* handler_data,
                        uint64_t* establisher_frame,
                        GuestContextPointers* ptrs,
                        bool* ok) {
    auto fail = [&]() -> uint64_t {
        if (ok != nullptr) {
            *ok = false;
        }
        return 0;
    };
    if (ok != nullptr) {
        *ok = true;
    }
    uint64_t entry_va = function_entry;
    GuestRuntimeFunction fe{};
    if (!resolve_entry(arena, image_base, entry_va, fe)) {
        return fail();
    }
    uint64_t frame = ctx.gpr[GuestReg::Rsp];
    uint64_t establisher = frame;
    bool machine_frame = false;
    uint64_t prolog_offset = kNoPrologOffset;
    UnwindInfoView info;
    for (int depth = 0;; ++depth) {
        if (depth >= kMaxChainDepth || !read_unwind_info(arena, image_base, fe.unwind, info)) {
            return fail();
        }
        const uint64_t start = image_base + fe.begin;
        prolog_offset = (control_pc >= start && control_pc < start + info.prolog_size) ? control_pc - start
                                                                                       : kNoPrologOffset;
        if (info.frame_register != 0) {
            bool frame_set = prolog_offset == kNoPrologOffset;
            for (int i = 0; !frame_set && i < info.count;) {
                const uint8_t op = (info.codes[i] >> 8) & 0xF;
                const int slots = code_slots(op, static_cast<uint8_t>(info.codes[i] >> 12), info.version);
                if (slots == 0) {
                    return fail();
                }
                if (op == kUwopSetFpreg && (info.codes[i] & 0xFF) <= prolog_offset) {
                    frame_set = true;
                }
                i += slots;
            }
            if (frame_set) {
                frame = ctx.gpr[info.frame_register] - uint64_t{16} * info.frame_offset;
            }
        }
        if (depth == 0 && prolog_offset == kNoPrologOffset && (info.flags & kUnwFlagChainInfo) == 0 &&
            is_epilog(arena, control_pc, image_base, fe)) {
            if (!interpret_epilog(arena, control_pc, image_base, fe, ctx, ptrs)) {
                return fail();
            }
            if (establisher_frame != nullptr) {
                *establisher_frame = frame;
            }
            return 0;
        }
        for (int i = 0; i < info.count;) {
            const uint16_t code = info.codes[i];
            const uint8_t offset = code & 0xFF;
            const uint8_t op = (code >> 8) & 0xF;
            const uint8_t op_info = static_cast<uint8_t>(code >> 12);
            const int slots = code_slots(op, op_info, info.version);
            if (slots == 0 || i + slots > info.count) {
                return fail();
            }
            const uint32_t slot1 = slots > 1 ? info.codes[i + 1] : 0;
            const uint32_t far = slots > 2 ? (slot1 | (uint32_t{info.codes[i + 2]} << 16)) : 0;
            const int at = i;
            i += slots;
            if (prolog_offset != kNoPrologOffset && prolog_offset < offset) {
                continue;  // not executed yet
            }
            switch (op) {
                case kUwopPushNonvol:
                    if (!pop_into(arena, ctx, ptrs, op_info)) {
                        return fail();
                    }
                    break;
                case kUwopAllocLarge:
                    ctx.gpr[GuestReg::Rsp] += op_info == 0 ? uint64_t{slot1} * 8 : far;
                    break;
                case kUwopAllocSmall:
                    ctx.gpr[GuestReg::Rsp] += uint64_t{op_info} * 8 + 8;
                    break;
                case kUwopSetFpreg:
                    ctx.gpr[GuestReg::Rsp] = frame;
                    establisher = frame;
                    break;
                case kUwopSaveNonvol:
                case kUwopSaveNonvolFar: {
                    const uint64_t va = frame + (op == kUwopSaveNonvol ? uint64_t{slot1} * 8 : far);
                    if (!read_u64(arena, va, ctx.gpr[op_info])) {
                        return fail();
                    }
                    if (ptrs != nullptr) {
                        ptrs->gpr[op_info] = va;
                    }
                    break;
                }
                case kUwopSaveXmm128:
                case kUwopSaveXmm128Far: {
                    const uint64_t va = frame + (op == kUwopSaveXmm128 ? uint64_t{slot1} * 16 : far);
                    if (!read_guest(arena, va, ctx.xmm[op_info])) {
                        return fail();
                    }
                    if (ptrs != nullptr) {
                        ptrs->xmm[op_info] = va;
                    }
                    break;
                }
                case kUwopEpilog:
                    break;
                case kUwopPushMachframe: {
                    if (op_info != 0) {
                        ctx.gpr[GuestReg::Rsp] += 8;  // error code
                    }
                    const uint64_t frame_va = ctx.gpr[GuestReg::Rsp];
                    uint64_t rsp = 0;
                    if (!read_u64(arena, frame_va, ctx.rip) || !read_u64(arena, frame_va + 24, rsp)) {
                        return fail();
                    }
                    ctx.gpr[GuestReg::Rsp] = rsp;
                    machine_frame = true;
                    break;
                }
                default:
                    return fail();
            }
            (void)at;
        }
        if ((info.flags & kUnwFlagChainInfo) == 0) {
            break;
        }
        if (!read_guest(arena, info.trailer_va, fe)) {
            return fail();
        }
    }
    if (!machine_frame && !pop_return(arena, ctx)) {
        return fail();
    }
    if (establisher_frame != nullptr) {
        *establisher_frame = establisher;
    }
    if ((info.flags & handler_type & (kUnwFlagEHandler | kUnwFlagUHandler)) == 0 ||
        prolog_offset != kNoPrologOffset) {
        return 0;
    }
    uint32_t handler_rva = 0;
    if (!read_guest(arena, info.trailer_va, handler_rva)) {
        return fail();
    }
    if (handler_data != nullptr) {
        *handler_data = info.trailer_va + 4;
    }
    return image_base + handler_rva;
}

// ---------------------------------------------------------------------------
// Exceptions
// ---------------------------------------------------------------------------

RestoreResult raise_exception(VirtualMemoryArena& arena, GuestState& state, const GuestExceptionRecord& record) {
    const uint64_t thunk_rsp = state.gpr[GuestReg::Rsp];
    GuestContext ctx{};
    if (!capture_caller_context(arena, state, ctx)) {
        raise_fault(FaultKind::BadStack, &state, 0, thunk_rsp);
    }
    Scratch scratch{thunk_rsp};
    const uint64_t context_va = scratch.alloc(sizeof(GuestContext));
    const uint64_t record_va = scratch.alloc(sizeof(GuestExceptionRecord));
    GuestExceptionRecord rec = record;
    if (rec.address == 0) {
        rec.address = ctx.rip;
    }
    if (!write_guest(arena, context_va, ctx) || !write_guest(arena, record_va, rec)) {
        raise_fault(FaultKind::BadStack, &state, rec.address, scratch.top);
    }
    const DispatchOutcome outcome = dispatch(arena, state, scratch, record_va, context_va, ctx);
    if (outcome == DispatchOutcome::Pending) {
        return RestoreResult::Pending;
    }
    if (outcome == DispatchOutcome::Unhandled) {
        bool pending = false;
        const bool resume = run_unhandled_filter(arena, state, scratch, record_va, context_va, &pending);
        if (pending) {
            return RestoreResult::Pending;
        }
        if (!resume) {
            read_guest(arena, record_va, rec);
            raise_fault(FaultKind::UnhandledException, &state, rec.address, rec.code);
        }
    }
    // A handler (or the top-level filter) continued execution with the possibly modified context.
    read_guest(arena, context_va, ctx);
    state.gpr[GuestReg::Rsp] = thunk_rsp;
    return finish_restore(arena, state, thunk_rsp, ctx);
}

RestoreResult unwind(VirtualMemoryArena& arena,
                     GuestState& state,
                     uint64_t target_frame,
                     uint64_t target_ip,
                     uint64_t exception_record,
                     uint64_t return_value,
                     uint64_t original_context,
                     uint64_t history_table) {
    const uint64_t thunk_rsp = state.gpr[GuestReg::Rsp];
    GuestContext cur{};
    if (!capture_caller_context(arena, state, cur)) {
        unwind_failed(state, target_ip, target_frame);
    }
    Scratch scratch{thunk_rsp};
    uint64_t record_va = exception_record;
    if (record_va == 0) {
        record_va = scratch.alloc(sizeof(GuestExceptionRecord));
        GuestExceptionRecord rec{};
        rec.code = kStatusUnwind;
        rec.address = cur.rip;
        write_guest(arena, record_va, rec);
    }
    GuestExceptionRecord rec{};
    if (!read_guest(arena, record_va, rec)) {
        unwind_failed(state, target_ip, target_frame);
    }
    const uint64_t ctx_va = original_context != 0 ? original_context : scratch.alloc(sizeof(GuestContext));
    const uint64_t dc_va = scratch.alloc(sizeof(GuestDispatcherContext));
    auto tag = std::make_unique<BoundaryTag>();
    tag->kind = TagKind::Unwind;
    tag->continuation = cur;
    tag->dispatcher_context = dc_va;

    uint32_t flags = kExceptionUnwinding | (target_frame == 0 ? kExceptionExitUnwind : 0);
    uint64_t nested_unused = 0;
    for (int frames = 0;; ++frames) {
        if (frames >= kMaxFrames) {
            unwind_failed(state, target_ip, target_frame);
        }
        GuestContext next{};
        uint64_t image_base = 0;
        uint64_t entry = 0;
        uint64_t handler = 0;
        uint64_t handler_data = 0;
        uint64_t establisher = 0;
        uint32_t scope_index = 0;
        if (cur.rip == kGuestReturnSentinel) {
            const BoundaryTag* collided = nullptr;
            const Crossing crossing = cross_boundary(arena, cur, false, flags, nested_unused, &collided);
            if (crossing == Crossing::End) {
                unwind_failed(state, target_ip, target_frame);
            }
            if (crossing == Crossing::Continue) {
                continue;
            }
            // Collided unwind: continue with the frame the outer unwind is processing.
            GuestDispatcherContext odc{};
            if (!read_guest(arena, collided->dispatcher_context, odc) || !read_guest(arena, odc.context_record, cur)) {
                unwind_failed(state, target_ip, target_frame);
            }
            cur.rip = odc.control_pc;
            next = cur;
            uint64_t unused_data = 0;
            virtual_unwind(arena, kUnwFlagUHandler, odc.image_base, odc.control_pc, odc.function_entry, next,
                           &unused_data, &establisher);
            image_base = odc.image_base;
            entry = odc.function_entry;
            handler = odc.language_handler;
            handler_data = odc.handler_data;
            scope_index = odc.scope_index;
            flags |= kExceptionCollidedUnwind;
        } else {
            entry = lookup_function_entry(arena, cur.rip, &image_base);
            if (entry == 0) {
                if (!pop_return(arena, cur)) {
                    unwind_failed(state, target_ip, target_frame);
                }
                continue;
            }
            next = cur;
            bool ok = true;
            handler = virtual_unwind(arena, kUnwFlagUHandler, image_base, cur.rip, entry, next, &handler_data,
                                     &establisher, nullptr, &ok);
            if (!ok) {
                unwind_failed(state, target_ip, target_frame);
            }
        }
        if ((establisher & 7) != 0 || (target_frame != 0 && establisher > target_frame)) {
            unwind_failed(state, target_ip, target_frame);  // STATUS_BAD_STACK / INVALID_UNWIND_TARGET
        }
        while (handler != 0) {
            if (establisher == target_frame) {
                flags |= kExceptionTargetUnwind;
            }
            write_guest(arena, record_va + offsetof(GuestExceptionRecord, flags), flags);
            cur.gpr[GuestReg::Rax] = return_value;
            write_guest(arena, ctx_va, cur);
            const GuestDispatcherContext dc{cur.rip, image_base, entry, establisher, target_ip, ctx_va, handler,
                                            handler_data, history_table, scope_index, 0};
            write_guest(arena, dc_va, dc);
            tag->establisher = establisher;
            const auto disposition = static_cast<uint32_t>(
                call_with(arena, state, scratch, handler, {record_va, establisher, ctx_va, dc_va}, tag.get()));
            if (state.unwind_pending != 0) {
                return RestoreResult::Pending;
            }
            flags &= ~(kExceptionTargetUnwind | kExceptionCollidedUnwind);
            if (disposition == kExceptionContinueSearch) {
                break;
            }
            if (disposition != kExceptionCollidedUnwindDisposition) {
                raise_fault(FaultKind::UnhandledException, &state, rec.address, kStatusInvalidDisposition);
            }
            // The handler replaced the dispatcher context with that of the unwind it collided with.
            GuestDispatcherContext back{};
            if (!read_guest(arena, dc_va, back) || !read_guest(arena, back.context_record, cur)) {
                unwind_failed(state, target_ip, target_frame);
            }
            cur.rip = back.control_pc;
            next = cur;
            uint64_t unused_data = 0;
            virtual_unwind(arena, kUnwFlagUHandler, back.image_base, back.control_pc, back.function_entry, next,
                           &unused_data, &establisher);
            image_base = back.image_base;
            entry = back.function_entry;
            handler = back.language_handler;
            handler_data = back.handler_data;
            scope_index = back.scope_index;
            flags |= kExceptionCollidedUnwind;
        }
        if (target_frame != 0 && establisher == target_frame) {
            break;
        }
        cur = next;
    }
    if (rec.code != kStatusUnwindConsolidate) {
        cur.rip = target_ip;
    }
    cur.gpr[GuestReg::Rax] = return_value;
    state.gpr[GuestReg::Rsp] = thunk_rsp;
    return restore_at(arena, state, thunk_rsp, cur, record_va);
}

RestoreResult restore_context(VirtualMemoryArena& arena, GuestState& state, GuestContext context, uint64_t exception_record) {
    return restore_at(arena, state, state.gpr[GuestReg::Rsp], context, exception_record);
}

uint64_t set_unhandled_exception_filter(uint64_t filter) {
    return registry().unhandled_filter.exchange(filter, std::memory_order_acq_rel);
}

// ---------------------------------------------------------------------------
// Thunks and binding
// ---------------------------------------------------------------------------

std::span<const ExceptionExport> exception_exports() {
    return kExports;
}

HostThunkFn exception_thunk_for(std::string_view module, std::string_view name) {
    for (const ExceptionExport& e : kExports) {
        if (name == e.name && same_module(module, e.module)) {
            return e.thunk;
        }
    }
    // API sets (api-ms-win-core-rtlsupport-*, ...) through O7's registry.
    const os::win32::ProcAddress resolved = os::win32::resolve_import(module, name);
    if (resolved != 0) {
        for (const ExceptionExport& e : kExports) {
            if (name == e.name && os::win32::proc_address(e.thunk) == resolved) {
                return e.thunk;
            }
        }
    }
    return nullptr;
}

size_t register_exception_exports() {
    Registry& reg = registry();
    std::lock_guard lock(reg.exports_mutex);
    if (!reg.exports_registered) {
        for (const ExceptionExport& e : kExports) {
            const os::win32::ProcAddress proc = os::win32::proc_address(e.thunk);
            os::win32::register_export(e.module, e.name, 0, proc);
            register_host_callable(e.thunk, e.name, proc);
        }
        reg.exports_registered = true;
    }
    return std::size(kExports);
}

void reset_guest_exceptions() {
    Registry& reg = registry();
    {
        std::unique_lock lock(reg.mutex);
        reg.tables.clear();
    }
    reg.unhandled_filter.store(0, std::memory_order_release);
    std::lock_guard lock(reg.exports_mutex);
    reg.exports_registered = false;
}

}  // namespace playgta6::guest

// ---------------------------------------------------------------------------
// C-ABI entry points
// ---------------------------------------------------------------------------

namespace guest = playgta6::guest;

extern "C" {

int32_t pg6_guest_register_image_function_table(uint64_t image_base,
                                                uint64_t image_size,
                                                uint64_t table_va,
                                                uint32_t count) {
    return guest::register_image_function_table(*playgta6::os::win32::get_vmem_arena(), image_base, image_size,
                                                table_va, count)
               ? 1
               : 0;
}

int32_t pg6_guest_register_image_from_headers(uint64_t image_base) {
    return guest::register_image_from_headers(*playgta6::os::win32::get_vmem_arena(), image_base) ? 1 : 0;
}

size_t pg6_guest_register_exception_exports(void) {
    return guest::register_exception_exports();
}

Pg6HostThunkFn pg6_guest_exception_thunk_for(const char* module, const char* name) {
    if (module == nullptr || name == nullptr) {
        return nullptr;
    }
    return guest::exception_thunk_for(module, name);
}

}  // extern "C"
