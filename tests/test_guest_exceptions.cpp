#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/guest_exceptions.h"
#include "guest/guest_runtime.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"

// Guest exception dispatch / unwinding tests. The "lifted" functions below are hand-written C++
// in the shape the recompiler emits: guest stack in guest memory, real pushes of return
// addresses and nonvolatile registers matching the synthetic .pdata unwind codes, and the resume
// protocol after every call (see after_call).

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_guest_exceptions.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;
using playgta6::os::PageProtect;
using playgta6::os::VirtualMemoryArena;

VirtualMemoryArena& arena() {
    return *os::win32::get_vmem_arena();
}

constexpr uint64_t kBase = 0x140000000ULL;
constexpr uint64_t kImageSize = 0x40000;
constexpr uint64_t va(uint64_t rva) {
    return kBase + rva;
}

// Function layout (RVAs). Each function's .pdata range is [begin, begin + size).
constexpr uint64_t kMain = 0x1000;           // try { mid(); } catch -> kCont1 / kCont2
constexpr uint64_t kMainRet = kMain + 0x15;  // return address of `call mid`
constexpr uint64_t kCont1 = kMain + 0x40;    // catch continuation (consolidated catch funclet)
constexpr uint64_t kCont2 = kMain + 0x50;    // catch continuation (plain RtlUnwindEx target)
constexpr uint64_t kBadCont = kMain + 0x60;  // not a resume target of main
constexpr uint64_t kCatchFunclet = 0x1080;
constexpr uint64_t kMid = 0x1100;            // __finally-style termination handler
constexpr uint64_t kMidRet = kMid + 0x15;
constexpr uint64_t kThrow = 0x1200;          // RaiseException(g_code, g_flags)
constexpr uint64_t kThrowRet = kThrow + 0x20;
constexpr uint64_t kHCatch = 0x1300;         // language handler of main
constexpr uint64_t kHCatchRet = kHCatch + 0x40;
constexpr uint64_t kCallCatch = 0x1400;      // STATUS_UNWIND_CONSOLIDATE callback
constexpr uint64_t kCallCatchRet = kCallCatch + 0x10;
constexpr uint64_t kHFinally = 0x1500;       // language handler of mid / deep
constexpr uint64_t kHFinallyRet = kHFinally + 0x10;
constexpr uint64_t kFin = 0x1600;            // termination handler body (may raise)
constexpr uint64_t kFinRet = kFin + 0x10;
constexpr uint64_t kSetjmp = 0x1700;         // leaf, no .pdata
constexpr uint64_t kMain3 = 0x1800;          // setjmp / longjmp
constexpr uint64_t kMain3SetjmpRet = kMain3 + 0x20;
constexpr uint64_t kMain3DeepRet = kMain3 + 0x30;
constexpr uint64_t kDeep = 0x1900;
constexpr uint64_t kDeepRet = kDeep + 0x10;
constexpr uint64_t kDeep2 = 0x1A00;
constexpr uint64_t kDeep2Ret = kDeep2 + 0x10;
constexpr uint64_t kLongjmp = 0x1B00;
constexpr uint64_t kLongjmpRet = kLongjmp + 0x10;
constexpr uint64_t kFilter = 0x1C00;         // top-level filter, leaf
constexpr uint64_t kRestoreFn = 0x1D00;      // RtlRestoreContext to an address nobody owns
constexpr uint64_t kRestoreFnRet = kRestoreFn + 0x10;

// Data
constexpr uint64_t kPdata = 0x8000;
constexpr uint64_t kUnwind = 0x9000;
constexpr uint64_t kLog = 0xA000;

// Exception codes understood by the miniature language handler.
constexpr uint32_t kCodeConsolidate = 0xE0000001;  // catch via consolidation -> kCont1
constexpr uint32_t kCodePlain = 0xE0000002;        // RtlUnwindEx to kCont2, return value 0x77
constexpr uint32_t kCodeResume = 0xE0000003;       // continue execution, context rax = 0x99
constexpr uint32_t kCodeBadTarget = 0xE0000004;    // RtlUnwindEx to kBadCont
constexpr uint32_t kCodeUnknown = 0xE0000005;      // nobody handles it

// Test knobs read by the guest functions.
uint32_t g_code = 0;
uint32_t g_flags = 0;
bool g_raise_in_finally = false;
std::string g_log;
uint64_t g_after_raise_rax = 0;
uint64_t g_resume_regs[4] = {};  // rbx rsi r12 local at the continuation
uint64_t g_main3_rbx = 0;
uint64_t g_raise_va = 0;
uint64_t g_unwind_ex_va = 0;
uint64_t g_unwind_va = 0;
uint64_t g_restore_va = 0;

// ---------------------------------------------------------------------------
// Guest memory and register helpers
// ---------------------------------------------------------------------------

uint64_t load64(uint64_t at) {
    uint64_t value = 0;
    CHECK(arena().read(at, &value, sizeof(value)));
    return value;
}

void store64(uint64_t at, uint64_t value) {
    CHECK(arena().write(at, &value, sizeof(value)));
}

uint64_t& r(GuestState* s, uint32_t reg) {
    return s->gpr[reg];
}

void push(GuestState* s, uint64_t value) {
    r(s, GuestReg::Rsp) -= 8;
    store64(r(s, GuestReg::Rsp), value);
}

uint64_t pop(GuestState* s) {
    const uint64_t value = load64(r(s, GuestReg::Rsp));
    r(s, GuestReg::Rsp) += 8;
    return value;
}

void ret(GuestState* s) {
    r(s, GuestReg::Rsp) += 8;
}

// `call [rip+iat]` into a runtime thunk (the return address is pushed by the caller).
void call_import(GuestState* s, uint64_t target, uint64_t return_rva) {
    push(s, va(return_rva));
    __playgta6_indirect_call(s, target);
}

// The resume protocol after a call returns, as lifted code implements it.
enum class After { Continue, Return, Resume };

After after_call(GuestState* s, uint64_t entry_rsp, std::initializer_list<uint64_t> targets, uint64_t* resume) {
    if (s->unwind_pending == 0) {
        return After::Continue;
    }
    if (targets.size() == 0 || r(s, GuestReg::Rsp) >= entry_rsp) {
        return After::Return;  // not the owner: propagate without touching the state
    }
    for (uint64_t t : targets) {
        if (va(t) == s->resume_rip) {
            s->unwind_pending = 0;
            *resume = t;
            return After::Resume;
        }
    }
    __playgta6_trap(s, s->resume_rip);  // owner without a block at the resume rip
    return After::Return;
}

GuestExceptionRecord read_record(uint64_t at) {
    GuestExceptionRecord rec{};
    CHECK(arena().read(at, &rec, sizeof(rec)));
    return rec;
}

GuestDispatcherContext read_dc(uint64_t at) {
    GuestDispatcherContext dc{};
    CHECK(arena().read(at, &dc, sizeof(dc)));
    return dc;
}

// ---------------------------------------------------------------------------
// Guest functions
// ---------------------------------------------------------------------------

void sub_mid(GuestState* s);
void sub_throw(GuestState* s);
void sub_fin(GuestState* s);
void sub_deep(GuestState* s);
void sub_deep2(GuestState* s);
void sub_longjmp(GuestState* s);

// main: push rbx; sub rsp, 30h | rbx/rsi/r12 = markers; local [rsp+28h] = 0; call mid
//       normal: rax = 1 | kCont1: rax += 100h | kCont2: rax += 200h | add rsp, 30h; pop rbx; ret
void sub_main(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    push(s, r(s, GuestReg::Rbx));
    r(s, GuestReg::Rsp) -= 0x30;
    r(s, GuestReg::Rbx) = 0xB0B0;
    r(s, GuestReg::Rsi) = 0x5151;
    r(s, GuestReg::R12) = 0x1212;
    store64(r(s, GuestReg::Rsp) + 0x28, 0);
    push(s, va(kMainRet));
    sub_mid(s);
    uint64_t resume = 0;
    switch (after_call(s, entry, {kCont1, kCont2}, &resume)) {
        case After::Return:
            return;
        case After::Continue:
            r(s, GuestReg::Rax) = 1;
            break;
        case After::Resume:
            g_resume_regs[0] = r(s, GuestReg::Rbx);
            g_resume_regs[1] = r(s, GuestReg::Rsi);
            g_resume_regs[2] = r(s, GuestReg::R12);
            g_resume_regs[3] = load64(r(s, GuestReg::Rsp) + 0x28);
            r(s, GuestReg::Rax) += resume == kCont1 ? 0x100 : 0x200;
            break;
    }
    r(s, GuestReg::Rsp) += 0x30;
    r(s, GuestReg::Rbx) = pop(s);
    ret(s);
}

// catch funclet: push rbp; sub rsp, 20h; mov rbp, rdx | [rbp+28h] = CA7Ch; rax = kCont1
void sub_catch_funclet(GuestState* s) {
    push(s, r(s, GuestReg::Rbp));
    r(s, GuestReg::Rsp) -= 0x20;
    r(s, GuestReg::Rbp) = r(s, GuestReg::Rdx);
    store64(r(s, GuestReg::Rbp) + 0x28, 0xCA7C);  // the parent's local, through its frame
    g_log += 'C';
    r(s, GuestReg::Rax) = va(kCont1);
    r(s, GuestReg::Rsp) += 0x20;
    r(s, GuestReg::Rbp) = pop(s);
    ret(s);
}

// mid: push rsi; push rdi; sub rsp, 28h | clobber rsi/rdi; call throw
void sub_mid(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    push(s, r(s, GuestReg::Rsi));
    push(s, r(s, GuestReg::Rdi));
    r(s, GuestReg::Rsp) -= 0x28;
    r(s, GuestReg::Rsi) = 0xDEAD1;
    r(s, GuestReg::Rdi) = 0xDEAD2;
    push(s, va(kMidRet));
    sub_throw(s);
    uint64_t resume = 0;
    if (after_call(s, entry, {}, &resume) == After::Return) {
        return;
    }
    r(s, GuestReg::Rsp) += 0x28;
    r(s, GuestReg::Rdi) = pop(s);
    r(s, GuestReg::Rsi) = pop(s);
    ret(s);
}

// throw: push r12; sub rsp, 20h | r12 = junk; RaiseException(g_code, g_flags, 1, &arg)
void sub_throw(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    push(s, r(s, GuestReg::R12));
    r(s, GuestReg::Rsp) -= 0x20;
    r(s, GuestReg::R12) = 0xBAD12;
    store64(r(s, GuestReg::Rsp) + 0x18, 0x5EED);  // the exception argument
    r(s, GuestReg::Rcx) = g_code;
    r(s, GuestReg::Rdx) = g_flags;
    r(s, GuestReg::R8) = 1;
    r(s, GuestReg::R9) = r(s, GuestReg::Rsp) + 0x18;
    call_import(s, g_raise_va, kThrowRet);
    uint64_t resume = 0;
    if (after_call(s, entry, {}, &resume) == After::Return) {
        return;
    }
    g_after_raise_rax = r(s, GuestReg::Rax);
    r(s, GuestReg::Rsp) += 0x20;
    r(s, GuestReg::R12) = pop(s);
    ret(s);
}

// Miniature C++-style language handler of main (handler data: none needed).
// sub rsp, 0C8h | dispatch: catch by code | unwind: log
void sub_h_catch(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    r(s, GuestReg::Rsp) -= 0xC8;
    const uint64_t rsp = r(s, GuestReg::Rsp);
    const uint64_t record_va = r(s, GuestReg::Rcx);
    const uint64_t establisher = r(s, GuestReg::Rdx);
    const uint64_t context_va = r(s, GuestReg::R8);
    const GuestDispatcherContext dc = read_dc(r(s, GuestReg::R9));
    const GuestExceptionRecord rec = read_record(record_va);
    uint64_t disposition = kExceptionContinueSearch;
    if ((rec.flags & kExceptionUnwinding) != 0) {
        g_log += (rec.flags & kExceptionTargetUnwind) != 0 ? 'T' : 'u';
    } else if (rec.code == kCodeResume) {
        GuestContext ctx{};
        CHECK(arena().read(context_va, &ctx, sizeof(ctx)));
        ctx.gpr[GuestReg::Rax] = 0x99;
        CHECK(arena().write(context_va, &ctx, sizeof(ctx)));
        disposition = kExceptionContinueExecution;
    } else if (rec.code == kCodeConsolidate || rec.code == kCodePlain || rec.code == kCodeBadTarget) {
        uint64_t unwind_record = record_va;
        uint64_t target_ip = rec.code == kCodePlain ? va(kCont2) : va(kBadCont);
        if (rec.code == kCodeConsolidate) {
            // Like FH3/FH4: unwind to the catching frame and run the catch block from
            // RtlRestoreContext through a consolidation callback.
            GuestExceptionRecord consolidate{};
            consolidate.code = kStatusUnwindConsolidate;
            consolidate.flags = kExceptionNoncontinuable;
            consolidate.number_parameters = 3;
            consolidate.information[0] = va(kCallCatch);
            consolidate.information[1] = establisher;
            consolidate.information[2] = va(kCatchFunclet);
            unwind_record = rsp + 0x30;
            CHECK(arena().write(unwind_record, &consolidate, sizeof(consolidate)));
            target_ip = 0;
        }
        r(s, GuestReg::Rcx) = establisher;
        r(s, GuestReg::Rdx) = target_ip;
        r(s, GuestReg::R8) = unwind_record;
        r(s, GuestReg::R9) = rec.code == kCodePlain ? 0x77 : 0;
        store64(rsp + 0x20, dc.context_record);
        store64(rsp + 0x28, dc.history_table);
        call_import(s, g_unwind_ex_va, kHCatchRet);
        uint64_t resume = 0;
        if (after_call(s, entry, {}, &resume) == After::Return) {
            return;
        }
        g_log += '!';  // RtlUnwindEx returned: never expected
    }
    r(s, GuestReg::Rax) = disposition;
    r(s, GuestReg::Rsp) += 0xC8;
    ret(s);
}

// Consolidation callback: sub rsp, 28h | rax = funclet(rcx = 0, rdx = establisher)
void sub_call_catch(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    r(s, GuestReg::Rsp) -= 0x28;
    const GuestExceptionRecord rec = read_record(r(s, GuestReg::Rcx));
    CHECK(rec.code == kStatusUnwindConsolidate);
    r(s, GuestReg::Rcx) = 0;
    r(s, GuestReg::Rdx) = rec.information[1];
    call_import(s, rec.information[2], kCallCatchRet);
    uint64_t resume = 0;
    if (after_call(s, entry, {}, &resume) == After::Return) {
        return;
    }
    r(s, GuestReg::Rsp) += 0x28;
    ret(s);
}

// __C_specific_handler-style handler of mid / deep: runs the termination handler once.
void sub_h_finally(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    r(s, GuestReg::Rsp) -= 0x28;
    const GuestExceptionRecord rec = read_record(r(s, GuestReg::Rcx));
    if ((rec.flags & kExceptionCollidedUnwind) != 0) {
        g_log += 'c';  // already ran when the unwind collided: skip
    } else if ((rec.flags & kExceptionUnwinding) != 0) {
        g_log += 'F';
        r(s, GuestReg::Rcx) = 1;  // AbnormalTermination
        call_import(s, va(kFin), kHFinallyRet);
        uint64_t resume = 0;
        if (after_call(s, entry, {}, &resume) == After::Return) {
            return;
        }
    } else {
        g_log += (rec.flags & kExceptionNestedCall) != 0 ? 'D' : 'd';
    }
    r(s, GuestReg::Rax) = kExceptionContinueSearch;
    r(s, GuestReg::Rsp) += 0x28;
    ret(s);
}

// Termination handler: sub rsp, 28h | optionally raises kCodePlain | log
void sub_fin(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    r(s, GuestReg::Rsp) -= 0x28;
    if (g_raise_in_finally) {
        r(s, GuestReg::Rcx) = kCodePlain;
        r(s, GuestReg::Rdx) = 0;
        r(s, GuestReg::R8) = 0;
        r(s, GuestReg::R9) = 0;
        call_import(s, g_raise_va, kFinRet);
        uint64_t resume = 0;
        if (after_call(s, entry, {}, &resume) == After::Return) {
            return;
        }
    }
    g_log += 'f';
    r(s, GuestReg::Rsp) += 0x28;
    ret(s);
}

// __intrinsic_setjmp(jmp_buf = rcx, frame = rdx): leaf.
void sub_setjmp(GuestState* s) {
    GuestJumpBuffer jb{};
    jb.frame = r(s, GuestReg::Rdx);
    jb.rbx = r(s, GuestReg::Rbx);
    jb.rsp = r(s, GuestReg::Rsp) + 8;
    jb.rbp = r(s, GuestReg::Rbp);
    jb.rsi = r(s, GuestReg::Rsi);
    jb.rdi = r(s, GuestReg::Rdi);
    jb.r12 = r(s, GuestReg::R12);
    jb.r13 = r(s, GuestReg::R13);
    jb.r14 = r(s, GuestReg::R14);
    jb.r15 = r(s, GuestReg::R15);
    jb.rip = load64(r(s, GuestReg::Rsp));
    jb.mx_csr = s->mxcsr;
    CHECK(arena().write(r(s, GuestReg::Rcx), &jb, sizeof(jb)));
    r(s, GuestReg::Rax) = 0;
    ret(s);
}

// main3: push rbp; sub rsp, 140h; lea rbp, [rsp+20h] | if (setjmp(jb, frame) == 0) deep(); ...
void sub_main3(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    push(s, r(s, GuestReg::Rbp));
    r(s, GuestReg::Rsp) -= 0x140;
    r(s, GuestReg::Rbp) = r(s, GuestReg::Rsp) + 0x20;
    r(s, GuestReg::Rbx) = 0x3B3B;
    uint64_t pc = 0;
    for (;;) {
        if (pc == 0) {
            r(s, GuestReg::Rcx) = r(s, GuestReg::Rsp) + 0x30;
            r(s, GuestReg::Rdx) = r(s, GuestReg::Rbp) - 0x20;
            push(s, va(kMain3SetjmpRet));
            sub_setjmp(s);
            pc = kMain3SetjmpRet;
            continue;
        }
        if (r(s, GuestReg::Rax) != 0) {  // second return of setjmp
            g_main3_rbx = r(s, GuestReg::Rbx);
            r(s, GuestReg::Rax) += 0x500;
            break;
        }
        r(s, GuestReg::Rbx) = 0xC0FFEE;  // clobbered; longjmp restores the saved value
        r(s, GuestReg::R13) = r(s, GuestReg::Rsp) + 0x30;  // the jump buffer, for deep2
        push(s, va(kMain3DeepRet));
        sub_deep(s);
        uint64_t resume = 0;
        const After after = after_call(s, entry, {kMain3SetjmpRet}, &resume);
        if (after == After::Return) {
            return;
        }
        if (after == After::Resume) {
            pc = resume;
            continue;
        }
        r(s, GuestReg::Rax) = 1;
        break;
    }
    r(s, GuestReg::Rsp) = r(s, GuestReg::Rbp) - 0x20 + 0x140;
    r(s, GuestReg::Rbp) = pop(s);
    ret(s);
}

// deep: push rsi; sub rsp, 20h | call deep2 (termination handler through h_finally)
void sub_deep(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    push(s, r(s, GuestReg::Rsi));
    r(s, GuestReg::Rsp) -= 0x20;
    push(s, va(kDeepRet));
    sub_deep2(s);
    uint64_t resume = 0;
    if (after_call(s, entry, {}, &resume) == After::Return) {
        return;
    }
    r(s, GuestReg::Rsp) += 0x20;
    r(s, GuestReg::Rsi) = pop(s);
    ret(s);
}

// deep2: sub rsp, 28h | longjmp(jb, 5)
void sub_deep2(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    r(s, GuestReg::Rsp) -= 0x28;
    r(s, GuestReg::Rcx) = r(s, GuestReg::R13);  // main3's jump buffer
    r(s, GuestReg::Rdx) = 5;
    push(s, va(kDeep2Ret));
    sub_longjmp(s);
    uint64_t resume = 0;
    if (after_call(s, entry, {}, &resume) == After::Return) {
        return;
    }
    r(s, GuestReg::Rsp) += 0x28;
    ret(s);
}

// longjmp(jb = rcx, value = rdx), Frame != 0: RtlUnwind(frame, rip, STATUS_LONGJUMP record, value)
void sub_longjmp(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    r(s, GuestReg::Rsp) -= 0xC8;
    const uint64_t rsp = r(s, GuestReg::Rsp);
    GuestJumpBuffer jb{};
    CHECK(arena().read(r(s, GuestReg::Rcx), &jb, sizeof(jb)));
    GuestExceptionRecord rec{};
    rec.code = kStatusLongjump;
    rec.number_parameters = 1;
    rec.information[0] = r(s, GuestReg::Rcx);
    CHECK(arena().write(rsp + 0x30, &rec, sizeof(rec)));
    const uint64_t value = r(s, GuestReg::Rdx);
    r(s, GuestReg::Rcx) = jb.frame;
    r(s, GuestReg::Rdx) = jb.rip;
    r(s, GuestReg::R8) = rsp + 0x30;
    r(s, GuestReg::R9) = value;
    call_import(s, g_unwind_va, kLongjmpRet);
    uint64_t resume = 0;
    if (after_call(s, entry, {}, &resume) == After::Return) {
        return;
    }
    g_log += '!';
    r(s, GuestReg::Rsp) += 0xC8;
    ret(s);
}

// Top-level filter: context rax = 42h, EXCEPTION_CONTINUE_EXECUTION. Leaf.
void sub_filter(GuestState* s) {
    GuestExceptionPointers pointers{};
    CHECK(arena().read(r(s, GuestReg::Rcx), &pointers, sizeof(pointers)));
    GuestContext ctx{};
    CHECK(arena().read(pointers.context_record, &ctx, sizeof(ctx)));
    ctx.gpr[GuestReg::Rax] = 0x42;
    CHECK(arena().write(pointers.context_record, &ctx, sizeof(ctx)));
    g_log += 'U';
    r(s, GuestReg::Rax) = static_cast<uint64_t>(static_cast<int64_t>(kExceptionContinueExecutionFilter));
    ret(s);
}

// RtlRestoreContext to rsp = r13 (above every guest frame), rip = r14.
void sub_restore_fn(GuestState* s) {
    const uint64_t entry = r(s, GuestReg::Rsp);
    r(s, GuestReg::Rsp) -= 0x28;
    GuestContext ctx{};
    ctx.gpr[GuestReg::Rsp] = r(s, GuestReg::R13);
    ctx.rip = r(s, GuestReg::R14);
    const uint64_t ctx_va = os::align_down(r(s, GuestReg::Rsp) - 0x600, 16);
    CHECK(arena().write(ctx_va, &ctx, sizeof(ctx)));
    r(s, GuestReg::Rcx) = ctx_va;
    r(s, GuestReg::Rdx) = 0;
    call_import(s, g_restore_va, kRestoreFnRet);
    uint64_t resume = 0;
    if (after_call(s, entry, {}, &resume) == After::Return) {
        return;
    }
    r(s, GuestReg::Rsp) += 0x28;
    ret(s);
}

const LiftedFunctionEntry kFunctions[] = {
    {va(kMain), sub_main},
    {va(kCatchFunclet), sub_catch_funclet},
    {va(kMid), sub_mid},
    {va(kThrow), sub_throw},
    {va(kHCatch), sub_h_catch},
    {va(kCallCatch), sub_call_catch},
    {va(kHFinally), sub_h_finally},
    {va(kFin), sub_fin},
    {va(kSetjmp), sub_setjmp},
    {va(kMain3), sub_main3},
    {va(kDeep), sub_deep},
    {va(kDeep2), sub_deep2},
    {va(kLongjmp), sub_longjmp},
    {va(kFilter), sub_filter},
    {va(kRestoreFn), sub_restore_fn},
};

// ---------------------------------------------------------------------------
// Synthetic .pdata
// ---------------------------------------------------------------------------

constexpr uint8_t kPush = 0;
constexpr uint8_t kAllocLarge = 1;
constexpr uint8_t kAllocSmall = 2;
constexpr uint8_t kSetFp = 3;
constexpr uint8_t kSaveNonvol = 4;
constexpr uint8_t kSaveNonvolFar = 5;
constexpr uint8_t kSaveXmm = 8;
constexpr uint8_t kSaveXmmFar = 9;
constexpr uint8_t kMachFrame = 10;

uint16_t code(uint8_t prolog_offset, uint8_t op, uint8_t info) {
    return static_cast<uint16_t>(prolog_offset | (op << 8) | (info << 12));
}

struct UnwindSpec {
    uint8_t flags = 0;
    uint8_t prolog = 0;
    uint8_t frame_reg = 0;
    uint8_t frame_off = 0;  // unscaled
    std::vector<uint16_t> codes;
    std::vector<uint8_t> trailer;  // handler RVA + data, or chained RUNTIME_FUNCTION
};

std::vector<uint8_t> encode(const UnwindSpec& u) {
    std::vector<uint8_t> out = {static_cast<uint8_t>(1 | (u.flags << 3)), u.prolog,
                                static_cast<uint8_t>(u.codes.size()),
                                static_cast<uint8_t>(u.frame_reg | (u.frame_off << 4))};
    for (uint16_t c : u.codes) {
        out.push_back(static_cast<uint8_t>(c));
        out.push_back(static_cast<uint8_t>(c >> 8));
    }
    if (u.codes.size() % 2 != 0) {
        out.push_back(0);
        out.push_back(0);
    }
    out.insert(out.end(), u.trailer.begin(), u.trailer.end());
    return out;
}

std::vector<uint8_t> le32(uint32_t v) {
    return {static_cast<uint8_t>(v), static_cast<uint8_t>(v >> 8), static_cast<uint8_t>(v >> 16),
            static_cast<uint8_t>(v >> 24)};
}

struct PdataBuilder {
    std::vector<GuestRuntimeFunction> entries;
    uint64_t next_unwind = kUnwind;

    uint32_t add_unwind(const UnwindSpec& spec) {
        const std::vector<uint8_t> bytes = encode(spec);
        const uint32_t rva = static_cast<uint32_t>(next_unwind);
        CHECK(arena().write(va(rva), bytes.data(), bytes.size()));
        next_unwind = (next_unwind + bytes.size() + 7) & ~uint64_t{7};
        return rva;
    }

    void add(uint64_t begin, uint64_t size, const UnwindSpec& spec) {
        entries.push_back({static_cast<uint32_t>(begin), static_cast<uint32_t>(begin + size), add_unwind(spec)});
    }

    void add_raw(uint64_t begin, uint64_t size, uint32_t unwind) {
        entries.push_back({static_cast<uint32_t>(begin), static_cast<uint32_t>(begin + size), unwind});
    }
};

UnwindSpec handler_spec(UnwindSpec spec, uint8_t flags, uint64_t handler_rva) {
    spec.flags = flags;
    spec.trailer = le32(static_cast<uint32_t>(handler_rva));
    spec.trailer.resize(spec.trailer.size() + 8, 0);  // handler data
    return spec;
}

UnwindSpec alloc_small(uint8_t offset, uint32_t size) {
    return UnwindSpec{0, offset, 0, 0, {code(offset, kAllocSmall, static_cast<uint8_t>(size / 8 - 1))}, {}};
}

UnwindSpec alloc_large(uint8_t offset, uint32_t size) {
    return UnwindSpec{0, offset, 0, 0, {code(offset, kAllocLarge, 0), static_cast<uint16_t>(size / 8)}, {}};
}

// Virtual-unwind fixtures live past the lifted functions.
constexpr uint64_t kVu = 0x3000;          // chained primary: every operation
constexpr uint64_t kVuChunk = 0x3200;     // chained chunk of kVu
constexpr uint64_t kVuEpilog = 0x3300;    // epilog detection
constexpr uint64_t kVuMach = 0x3400;      // machine frame
constexpr uint64_t kVuIndirect = 0x3480;  // indirect entry -> kVuMach's record
constexpr uint64_t kVuHandler = 0x3F00;

void build_image() {
    PdataBuilder p;
    const uint8_t eu = kUnwFlagEHandler | kUnwFlagUHandler;
    p.add(kMain, 0x80, handler_spec({0, 5, 0, 0, {code(5, kAllocSmall, 5), code(1, kPush, GuestReg::Rbx)}, {}}, eu, kHCatch));
    p.add(kCatchFunclet, 0x20, {0, 5, 0, 0, {code(5, kAllocSmall, 3), code(1, kPush, GuestReg::Rbp)}, {}});
    p.add(kMid, 0x80,
          handler_spec({0, 6, 0, 0, {code(6, kAllocSmall, 4), code(2, kPush, GuestReg::Rdi), code(1, kPush, GuestReg::Rsi)}, {}},
                       eu, kHFinally));
    p.add(kThrow, 0x40, {0, 6, 0, 0, {code(6, kAllocSmall, 3), code(2, kPush, GuestReg::R12)}, {}});
    p.add(kHCatch, 0x80, alloc_large(7, 0xC8));
    p.add(kCallCatch, 0x40, alloc_small(4, 0x28));
    p.add(kHFinally, 0x40, alloc_small(4, 0x28));
    p.add(kFin, 0x40, alloc_small(4, 0x28));
    // kSetjmp and kFilter are leaves without .pdata.
    p.add(kMain3, 0x80,
          {0, 13, GuestReg::Rbp, 2,
           {code(13, kSetFp, 0), code(8, kAllocLarge, 0), 0x140 / 8, code(1, kPush, GuestReg::Rbp)}, {}});
    p.add(kDeep, 0x40, handler_spec({0, 5, 0, 0, {code(5, kAllocSmall, 3), code(1, kPush, GuestReg::Rsi)}, {}},
                                    kUnwFlagUHandler, kHFinally));
    p.add(kDeep2, 0x40, alloc_small(4, 0x28));
    p.add(kLongjmp, 0x40, alloc_large(7, 0xC8));
    p.add(kRestoreFn, 0x40, alloc_small(4, 0x28));

    // kVu: push rbp (1), push r15 (3), sub rsp, 1000h (10), lea rbp, [rsp+10h] (15),
    // mov [rsp+18h], rbx (20), mov [rsp+20h], rsi (far, 28), movaps [rsp+30h], xmm6 (34),
    // movaps [rsp+40h], xmm7 (far, 43). EHANDLER kVuHandler.
    UnwindSpec vu{kUnwFlagEHandler, 43, GuestReg::Rbp, 1,
                  {code(43, kSaveXmmFar, 7), 0x40, 0, code(34, kSaveXmm, 6), 3, code(28, kSaveNonvolFar, GuestReg::Rsi),
                   0x20, 0, code(20, kSaveNonvol, GuestReg::Rbx), 3, code(15, kSetFp, 0), code(10, kAllocLarge, 0),
                   0x1000 / 8, code(3, kPush, GuestReg::R15), code(1, kPush, GuestReg::Rbp)},
                  {}};
    vu.trailer = le32(kVuHandler);
    vu.trailer.resize(8, 0);
    const uint32_t vu_unwind = p.add_unwind(vu);
    p.add_raw(kVu, 0x100, vu_unwind);
    // Chunk: sub rsp, 10h at its start, then the primary's codes.
    UnwindSpec chunk{kUnwFlagChainInfo, 0, 0, 0, {code(0, kAllocSmall, 1)}, {}};
    for (uint32_t v : {static_cast<uint32_t>(kVu), static_cast<uint32_t>(kVu + 0x100), vu_unwind}) {
        const std::vector<uint8_t> b = le32(v);
        chunk.trailer.insert(chunk.trailer.end(), b.begin(), b.end());
    }
    p.add(kVuChunk, 0x20, chunk);
    p.add(kVuEpilog, 0x40, {0, 5, 0, 0, {code(5, kAllocSmall, 3), code(1, kPush, GuestReg::Rbx)}, {}});
    p.add(kVuMach, 0x40, {0, 0, 0, 0, {code(0, kMachFrame, 1)}, {}});
    p.add_raw(kVuIndirect, 0x10, 0);  // fixed below: points at kVuMach's record

    std::sort(p.entries.begin(), p.entries.end(),
              [](const GuestRuntimeFunction& a, const GuestRuntimeFunction& b) { return a.begin < b.begin; });
    for (size_t i = 0; i < p.entries.size(); ++i) {
        if (p.entries[i].begin == kVuMach) {
            for (auto& e : p.entries) {
                if (e.begin == kVuIndirect) {
                    e.unwind = static_cast<uint32_t>(kPdata + 12 * i) + 1;
                }
            }
        }
    }
    CHECK(arena().write(va(kPdata), p.entries.data(), p.entries.size() * sizeof(GuestRuntimeFunction)));

    // Minimal PE headers so the image registers from its exception directory.
    const uint32_t e_lfanew = 0x80;
    CHECK(arena().write(va(0x3C), &e_lfanew, 4));
    const uint32_t signature = 0x00004550;
    CHECK(arena().write(va(0x80), &signature, 4));
    const uint64_t optional = 0x80 + 4 + 20;
    const uint16_t magic = 0x20B;
    const uint32_t size_of_image = static_cast<uint32_t>(kImageSize);
    const uint32_t directories = 16;
    const uint32_t pdata[2] = {static_cast<uint32_t>(kPdata),
                               static_cast<uint32_t>(p.entries.size() * sizeof(GuestRuntimeFunction))};
    CHECK(arena().write(va(optional), &magic, 2));
    CHECK(arena().write(va(optional + 56), &size_of_image, 4));
    CHECK(arena().write(va(optional + 108), &directories, 4));
    CHECK(arena().write(va(optional + 112 + 3 * 8), pdata, sizeof(pdata)));
}

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

struct GuestFault {
    FaultInfo info;
};

void throwing_handler(const FaultInfo* info) {
    throw GuestFault{*info};
}

uint64_t thunk_va(const char* name) {
    return register_host_callable(exception_thunk_for("KERNEL32.dll", name), name);
}

struct Fixture {
    GuestThread thread{};
    bool ok = true;

    Fixture() {
        reset_guest_runtime();
        reset_guest_exceptions();
        set_fault_handler(throwing_handler);
        CHECK(register_lifted_functions(kFunctions));
        VirtualMemoryArena& a = arena();
        static bool mapped = false;
        if (!mapped) {
            CHECK(a.reserve(kBase, kImageSize, PageProtect::ReadWrite) == kBase);
            CHECK(a.commit(kBase, kImageSize, PageProtect::ReadWrite));
            mapped = true;
        }
        build_image();
        CHECK(register_image_from_headers(a, kBase));
        CHECK(create_guest_thread(a, 256 * 1024, thread));
        thread.state.mxcsr = kGuestMxcsrDefault;
        g_raise_va = thunk_va("RaiseException");
        g_unwind_ex_va = thunk_va("RtlUnwindEx");
        g_unwind_va = thunk_va("RtlUnwind");
        g_restore_va = thunk_va("RtlRestoreContext");
        g_code = 0;
        g_flags = 0;
        g_raise_in_finally = false;
        g_log.clear();
        g_after_raise_rax = 0;
        g_main3_rbx = 0;
        std::memset(g_resume_regs, 0, sizeof(g_resume_regs));
    }

    ~Fixture() {
        destroy_guest_thread(arena(), thread);
        set_unhandled_exception_filter(0);
    }

    // Runs main; returns the fault kind (0 = none).
    uint32_t run(uint64_t fn, uint64_t* rax = nullptr, std::initializer_list<uint64_t> args = {}) {
        try {
            const uint64_t result =
                call_guest(arena(), thread.state, va(fn), std::span<const uint64_t>(args.begin(), args.size()));
            if (rax != nullptr) {
                *rax = result;
            }
            return 0;
        } catch (const GuestFault& fault) {
            last_fault = fault.info;
            return fault.info.kind;
        }
    }

    FaultInfo last_fault{};
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_layouts() {
    CHECK(sizeof(GuestContext) == 1232);
    CHECK(sizeof(GuestExceptionRecord) == 152);
    CHECK(sizeof(GuestDispatcherContext) == 80);
    CHECK(sizeof(GuestJumpBuffer) == 256);
}

void test_function_tables() {
    Fixture fx;
    uint64_t base = 0;
    const uint64_t entry = lookup_function_entry(arena(), va(kMid + 0x10), &base);
    CHECK(base == kBase);
    GuestRuntimeFunction rf{};
    CHECK(arena().read(entry, &rf, sizeof(rf)));
    CHECK(rf.begin == kMid && rf.end == kMid + 0x80);
    CHECK(entry >= va(kPdata) && (entry - va(kPdata)) % 12 == 0);
    // Leaf code and addresses outside every table.
    CHECK(lookup_function_entry(arena(), va(kSetjmp), &base) == 0);
    CHECK(lookup_function_entry(arena(), 0x7000'0000ULL, &base) == 0);
    // Indirect entries resolve to the record holding the unwind data.
    const uint64_t indirect = lookup_function_entry(arena(), va(kVuIndirect + 4), &base);
    CHECK(arena().read(indirect, &rf, sizeof(rf)));
    CHECK(rf.begin == kVuMach);
    CHECK(pc_to_image_base(va(0x5000)) == kBase);
    CHECK(pc_to_image_base(kBase - 1) == 0);
    // A second image may not overlap.
    CHECK(!register_image_function_table(arena(), kBase + 0x1000, 0x1000, va(kPdata), 1));

    // Dynamic table (RtlAddFunctionTable) with its own base, then removed.
    const uint64_t dyn_base = kBase + 0x20000;
    const GuestRuntimeFunction dyn[2] = {{0x100, 0x180, static_cast<uint32_t>(kUnwind - 0x20000)},
                                         {0x10, 0x20, static_cast<uint32_t>(kUnwind - 0x20000)}};
    CHECK(arena().write(va(0xB000), dyn, sizeof(dyn)));
    CHECK(add_function_table(arena(), va(0xB000), 2, dyn_base));
    CHECK(!add_function_table(arena(), va(0xB000), 2, dyn_base));
    CHECK(lookup_function_entry(arena(), dyn_base + 0x150, &base) == va(0xB000));
    CHECK(lookup_function_entry(arena(), dyn_base + 0x18, &base) == va(0xB000) + 12);
    CHECK(base == dyn_base);
    CHECK(pc_to_image_base(dyn_base + 0x150) == kBase);  // inside the image range, not its own
    CHECK(delete_function_table(va(0xB000)));
    CHECK(!delete_function_table(va(0xB000)));
    CHECK(lookup_function_entry(arena(), dyn_base + 0x150, &base) == 0);
    CHECK(!add_function_table(arena(), va(0xB000), 0, dyn_base));
}

// Builds a context for the kVu fixtures on a fake stack.
constexpr uint64_t kFakeStack = 0x20000;

void test_virtual_unwind_every_operation() {
    Fixture fx;
    // Frame layout at the call site inside the chunk (rsp = S):
    //   S + 10h            = frame (rsp after the primary's allocation)
    //   frame + 18h / 20h  = saved rbx / rsi; frame + 30h / 40h = saved xmm6 / xmm7
    //   frame + 1000h      = saved r15, + 8 saved rbp, + 10h return address
    const uint64_t s_rsp = va(kFakeStack);
    const uint64_t frame = s_rsp + 0x10;
    store64(frame + 0x18, 0xB8B8);
    store64(frame + 0x20, 0x5151);
    const GuestM128 x6{0x66, 0x666}, x7{0x77, 0x777};
    CHECK(arena().write(frame + 0x30, &x6, 16));
    CHECK(arena().write(frame + 0x40, &x7, 16));
    store64(frame + 0x1000, 0x1515);
    store64(frame + 0x1008, 0xBBBB);
    store64(frame + 0x1010, 0x1400099AAULL);

    GuestContext ctx{};
    ctx.gpr[GuestReg::Rsp] = s_rsp;
    ctx.gpr[GuestReg::Rbp] = frame + 0x10;  // lea rbp, [rsp+10h]
    ctx.rip = va(kVuChunk + 8);
    GuestContextPointers ptrs{};
    uint64_t data = 0, establisher = 0;
    bool ok = false;
    const uint64_t entry = lookup_function_entry(arena(), ctx.rip, nullptr);
    const uint64_t handler =
        virtual_unwind(arena(), kUnwFlagEHandler, kBase, ctx.rip, entry, ctx, &data, &establisher, &ptrs, &ok);
    CHECK(ok);
    CHECK(handler == va(kVuHandler));  // from the primary at the end of the chain
    CHECK(data != 0);
    CHECK(establisher == frame);
    CHECK(ctx.rip == 0x1400099AAULL);
    CHECK(ctx.gpr[GuestReg::Rsp] == frame + 0x1018);
    CHECK(ctx.gpr[GuestReg::Rbx] == 0xB8B8 && ctx.gpr[GuestReg::Rsi] == 0x5151);
    CHECK(ctx.gpr[GuestReg::R15] == 0x1515 && ctx.gpr[GuestReg::Rbp] == 0xBBBB);
    CHECK(ctx.xmm[6].low == 0x66 && ctx.xmm[6].high == 0x666 && ctx.xmm[7].low == 0x77);
    CHECK(ptrs.gpr[GuestReg::Rbx] == frame + 0x18 && ptrs.gpr[GuestReg::R15] == frame + 0x1000);
    CHECK(ptrs.xmm[7] == frame + 0x40);
    // No exception handler for a UHANDLER query.
    GuestContext again{};
    again.gpr[GuestReg::Rsp] = s_rsp;
    again.gpr[GuestReg::Rbp] = frame + 0x10;
    again.rip = va(kVuChunk + 8);
    CHECK(virtual_unwind(arena(), kUnwFlagUHandler, kBase, again.rip, entry, again, &data, &establisher) == 0);

    // Inside the primary's prolog after `sub rsp, 1000h` (offset 12): only push/push/alloc have
    // executed; the frame register is not set up yet, so the establisher is rsp.
    GuestContext prolog{};
    prolog.gpr[GuestReg::Rsp] = frame;
    prolog.gpr[GuestReg::Rbp] = 0x1234;  // not a frame pointer yet
    prolog.rip = va(kVu + 12);
    const uint64_t primary = lookup_function_entry(arena(), prolog.rip, nullptr);
    CHECK(virtual_unwind(arena(), kUnwFlagEHandler, kBase, prolog.rip, primary, prolog, &data, &establisher, nullptr,
                         &ok) == 0);
    CHECK(ok);
    CHECK(establisher == frame);
    CHECK(prolog.gpr[GuestReg::R15] == 0x1515 && prolog.gpr[GuestReg::Rbp] == 0xBBBB);
    CHECK(prolog.gpr[GuestReg::Rbx] == 0);  // the save at offset 20 has not happened
    CHECK(prolog.rip == 0x1400099AAULL && prolog.gpr[GuestReg::Rsp] == frame + 0x1018);
}

void test_virtual_unwind_epilogs_and_machine_frames() {
    Fixture fx;
    // kVuEpilog: push rbx; sub rsp, 20h ... at +20h: add rsp, 20h; pop rbx; ret
    //   at +28h: pop rbx; jmp <outside> (tail call)   at +30h: mov rax, rcx (not an epilog)
    const uint8_t epilog[] = {0x48, 0x83, 0xC4, 0x20, 0x5B, 0xC3};
    const uint8_t tail[] = {0x5B, 0xE9, 0x00, 0x10, 0x00, 0x00};
    const uint8_t body[] = {0x48, 0x89, 0xC8};
    CHECK(arena().write(va(kVuEpilog + 0x20), epilog, sizeof(epilog)));
    CHECK(arena().write(va(kVuEpilog + 0x28), tail, sizeof(tail)));
    CHECK(arena().write(va(kVuEpilog + 0x30), body, sizeof(body)));
    const uint64_t sp = va(kFakeStack + 0x2000);
    store64(sp + 0x20, 0xBB);           // saved rbx
    store64(sp + 0x28, 0x140007777ULL);  // return address
    const uint64_t entry = lookup_function_entry(arena(), va(kVuEpilog), nullptr);
    for (uint64_t pc : {kVuEpilog + 0x20, kVuEpilog + 0x30}) {
        GuestContext ctx{};
        ctx.gpr[GuestReg::Rsp] = sp;
        ctx.rip = va(pc);
        bool ok = false;
        uint64_t data = 0, establisher = 0;
        virtual_unwind(arena(), kUnwFlagEHandler, kBase, ctx.rip, entry, ctx, &data, &establisher, nullptr, &ok);
        CHECK(ok);
        CHECK(ctx.gpr[GuestReg::Rbx] == 0xBB);
        CHECK(ctx.rip == 0x140007777ULL && ctx.gpr[GuestReg::Rsp] == sp + 0x30);
    }
    // Part-way through the epilog (after add rsp): emulated from the pop.
    GuestContext mid{};
    mid.gpr[GuestReg::Rsp] = sp + 0x20;
    mid.rip = va(kVuEpilog + 0x28);
    virtual_unwind(arena(), kUnwFlagEHandler, kBase, mid.rip, entry, mid, nullptr, nullptr);
    CHECK(mid.gpr[GuestReg::Rbx] == 0xBB && mid.rip == 0x140007777ULL && mid.gpr[GuestReg::Rsp] == sp + 0x30);

    // Machine frame with error code: [err][rip][cs][eflags][rsp][ss].
    const uint64_t mf = va(kFakeStack + 0x3000);
    store64(mf + 8, 0x140005555ULL);
    store64(mf + 32, 0x1400AAA0ULL);
    GuestContext m{};
    m.gpr[GuestReg::Rsp] = mf;
    m.rip = va(kVuMach + 4);
    bool ok = false;
    virtual_unwind(arena(), kUnwFlagEHandler, kBase, m.rip, lookup_function_entry(arena(), m.rip, nullptr), m, nullptr,
                   nullptr, nullptr, &ok);
    CHECK(ok);
    CHECK(m.rip == 0x140005555ULL && m.gpr[GuestReg::Rsp] == 0x1400AAA0ULL);
}

void test_capture_context_thunk() {
    Fixture fx;
    GuestState& s = fx.thread.state;
    s.gpr[GuestReg::Rbx] = 0x1111;
    s.flags[GuestFlag::ZF] = 1;
    s.xmm[9][1] = 0x99;
    const uint64_t ctx_va = s.gpr[GuestReg::Rsp] - 0x1000;
    s.gpr[GuestReg::Rcx] = ctx_va;
    push(&s, 0x140004444ULL);
    const uint64_t call_rsp = s.gpr[GuestReg::Rsp];
    CHECK(register_exception_exports() == exception_exports().size());
    CHECK(register_exception_exports() == exception_exports().size());  // idempotent
    HostThunkFn capture = exception_thunk_for("api-ms-win-core-rtlsupport-l1-1-0.dll", "RtlCaptureContext");
    CHECK(capture != nullptr);
    CHECK(exception_thunk_for("kernel32", "RtlVirtualUnwind") != nullptr);
    CHECK(exception_thunk_for("KERNEL32.dll", "CreateFileW") == nullptr);
    capture(&s);
    CHECK(s.gpr[GuestReg::Rsp] == call_rsp + 8);
    GuestContext ctx{};
    CHECK(arena().read(ctx_va, &ctx, sizeof(ctx)));
    CHECK(ctx.rip == 0x140004444ULL && ctx.gpr[GuestReg::Rsp] == call_rsp + 8);
    CHECK(ctx.gpr[GuestReg::Rbx] == 0x1111 && (ctx.eflags & 0x40) != 0 && ctx.xmm[9].high == 0x99);
    CHECK(ctx.context_flags == (kContextFull | kContextSegments) && ctx.mx_csr == kGuestMxcsrDefault);
    CHECK(guest_callable_for_proc(os::win32::proc_address(capture)) != 0);
}

void test_throw_caught_two_frames_up_through_catch_funclet() {
    Fixture fx;
    GuestState& s = fx.thread.state;
    const uint64_t top = s.gpr[GuestReg::Rsp];
    s.gpr[GuestReg::Rbx] = 0xCA11E4;
    g_code = kCodeConsolidate;
    g_flags = kExceptionNoncontinuable;
    uint64_t rax = 0;
    CHECK(fx.run(kMain, &rax) == 0);
    // Dispatch reached main's handler past mid's (search), the unwind ran mid's termination
    // handler, main's handler saw the target unwind, then the catch funclet ran.
    CHECK(g_log == "dFfTC");
    CHECK(rax == 0x100);
    CHECK(s.unwind_pending == 0);
    CHECK(s.gpr[GuestReg::Rsp] == top && s.gpr[GuestReg::Rbx] == 0xCA11E4);
    // At the continuation: nonvolatiles restored from mid's and throw's saves, the funclet wrote
    // the parent's local through the establisher frame.
    CHECK(g_resume_regs[0] == 0xB0B0);
    CHECK(g_resume_regs[1] == 0x5151);
    CHECK(g_resume_regs[2] == 0x1212);
    CHECK(g_resume_regs[3] == 0xCA7C);
}

void test_plain_unwind_target_and_return_value() {
    Fixture fx;
    g_code = kCodePlain;
    uint64_t rax = 0;
    CHECK(fx.run(kMain, &rax) == 0);
    CHECK(g_log == "dFfT");
    CHECK(rax == 0x77 + 0x200);
    CHECK(g_resume_regs[1] == 0x5151 && g_resume_regs[2] == 0x1212);
}

void test_continue_execution() {
    Fixture fx;
    g_code = kCodeResume;
    uint64_t rax = 0;
    CHECK(fx.run(kMain, &rax) == 0);
    CHECK(g_log == "d");
    CHECK(g_after_raise_rax == 0x99);  // the handler's modified context, applied in place
    CHECK(rax == 1);

    // Continuing a noncontinuable exception is fatal.
    Fixture fx2;
    g_code = kCodeResume;
    g_flags = kExceptionNoncontinuable;
    CHECK(fx2.run(kMain) == static_cast<uint32_t>(FaultKind::UnhandledException));
    CHECK(fx2.last_fault.detail == kStatusNoncontinuableException);
}

void test_unhandled_exception() {
    Fixture fx;
    g_code = kCodeUnknown;
    CHECK(fx.run(kMain) == static_cast<uint32_t>(FaultKind::UnhandledException));
    CHECK(fx.last_fault.detail == kCodeUnknown);
    CHECK(fx.last_fault.va == va(kThrowRet));
    CHECK(g_log == "d");

    // A top-level filter returning EXCEPTION_CONTINUE_EXECUTION resumes at the raise site.
    Fixture fx2;
    g_code = kCodeUnknown;
    CHECK(set_unhandled_exception_filter(va(kFilter)) == 0);
    uint64_t rax = 0;
    CHECK(fx2.run(kMain, &rax) == 0);
    CHECK(g_log == "dU");
    CHECK(g_after_raise_rax == 0x42);
    CHECK(rax == 1);
}

void test_nested_exception_during_unwind_collides() {
    Fixture fx;
    // The termination handler raises kCodePlain while the unwind for kCodeConsolidate runs.
    // Its dispatch is nested (mid's handler sees EXCEPTION_NESTED_CALL); main catches it and the
    // second unwind collides with the first at mid's frame, so mid's handler is called again with
    // EXCEPTION_COLLIDED_UNWIND instead of running the termination handler twice.
    g_code = kCodeConsolidate;
    g_raise_in_finally = true;
    uint64_t rax = 0;
    CHECK(fx.run(kMain, &rax) == 0);
    CHECK(g_log == "dFDcT");
    CHECK(rax == 0x77 + 0x200);
    CHECK(fx.thread.state.unwind_pending == 0);
    CHECK(g_resume_regs[1] == 0x5151 && g_resume_regs[2] == 0x1212);
}

void test_longjmp_across_frames() {
    Fixture fx;
    GuestState& s = fx.thread.state;
    uint64_t rax = 0;
    CHECK(fx.run(kMain3, &rax) == 0);
    CHECK(rax == 0x505);         // setjmp returned 5 the second time
    CHECK(g_main3_rbx == 0x3B3B);  // nonvolatile restored from the jump buffer
    CHECK(g_log == "Ff");         // deep's termination handler ran during the unwind
    CHECK(s.unwind_pending == 0);
}

void test_unknown_resume_targets_fault() {
    Fixture fx;
    g_code = kCodeBadTarget;
    CHECK(fx.run(kMain) == static_cast<uint32_t>(FaultKind::UnknownResumeTarget));
    CHECK(fx.last_fault.va == va(kBadCont));

    // A restore above every guest frame of the thread has no owner.
    Fixture fx2;
    GuestState& s = fx2.thread.state;
    s.gpr[GuestReg::R13] = fx2.thread.stack_top + 0x100;
    s.gpr[GuestReg::R14] = va(kMain + 0x10);
    CHECK(fx2.run(kRestoreFn) == static_cast<uint32_t>(FaultKind::UnknownResumeTarget));
    CHECK(fx2.last_fault.detail == fx2.thread.stack_top + 0x100);
}

void test_restore_in_place() {
    Fixture fx;
    GuestState& s = fx.thread.state;
    // RtlRestoreContext to its own return address applies the registers and just returns.
    push(&s, 0x140006666ULL);
    const uint64_t call_rsp = s.gpr[GuestReg::Rsp];
    GuestContext ctx{};
    ctx.rip = 0x140006666ULL;
    ctx.gpr[GuestReg::Rsp] = call_rsp + 8;
    ctx.gpr[GuestReg::Rdi] = 0xD1D1;
    ctx.eflags = 0x202 | 1;  // CF
    CHECK(restore_context(arena(), s, ctx, 0) == RestoreResult::InPlace);
    CHECK(s.unwind_pending == 0 && s.gpr[GuestReg::Rsp] == call_rsp + 8 && s.gpr[GuestReg::Rdi] == 0xD1D1);
    CHECK(s.flags[GuestFlag::CF] == 1 && s.flags[GuestFlag::ZF] == 0);
    // Anywhere else goes through the resume protocol.
    push(&s, 0x140006666ULL);
    ctx.rip = 0x140006670ULL;
    CHECK(restore_context(arena(), s, ctx, 0) == RestoreResult::Pending);
    CHECK(s.unwind_pending == 1 && s.resume_rip == 0x140006670ULL);
    s.unwind_pending = 0;
}

}  // namespace

int main() {
    test_layouts();
    test_function_tables();
    test_virtual_unwind_every_operation();
    test_virtual_unwind_epilogs_and_machine_frames();
    test_capture_context_thunk();
    test_throw_caught_two_frames_up_through_catch_funclet();
    test_plain_unwind_target_and_return_value();
    test_continue_execution();
    test_unhandled_exception();
    test_nested_exception_during_unwind_collides();
    test_longjmp_across_frames();
    test_unknown_resume_targets_fault();
    test_restore_in_place();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::puts("test_guest_exceptions: all checks passed");
    return 0;
}
