#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "abi/playgta6_abi.h"
#include "guest/com_bridge.h"
#include "guest/guest_callbacks.h"
#include "guest/guest_runtime.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/win32_loader.h"
#include "os/win32_sync.h"
#include "os/xgameruntime.h"

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_com_bridge.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;
namespace com = playgta6::guest::com;
namespace cbk = playgta6::guest::callbacks;
using com::GuestPtr;
using com::Guid;
using com::HResult;
using com::InterfaceArg;
using com::OutPtr;
using playgta6::os::PageProtect;
using playgta6::os::VirtualMemoryArena;

VirtualMemoryArena* g_arena = nullptr;

uint64_t load64(uint64_t va) {
    uint64_t value = 0;
    CHECK(g_arena->read(va, &value, sizeof(value)));
    return value;
}

int32_t load32(uint64_t va) {
    int32_t value = 0;
    CHECK(g_arena->read(va, &value, sizeof(value)));
    return value;
}

void store64(uint64_t va, uint64_t value) {
    CHECK(g_arena->write(va, &value, sizeof(value)));
}

uint64_t& gpr(GuestState* s, uint32_t reg) {
    return s->gpr[reg];
}

void push(GuestState* s, uint64_t value) {
    gpr(s, GuestReg::Rsp) -= 8;
    store64(gpr(s, GuestReg::Rsp), value);
}

void ret(GuestState* s) {
    gpr(s, GuestReg::Rsp) += 8;
}

uint64_t f64_bits(double d) {
    uint64_t b;
    std::memcpy(&b, &d, 8);
    return b;
}

uint64_t f32_bits(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return b;
}

double bits_f64(uint64_t b) {
    double d;
    std::memcpy(&d, &b, 8);
    return d;
}

// ---------------------------------------------------------------------------
// A hand-written "lifted" caller that does what compiled code does for a COM call: argument
// registers / xmm registers / stack slots above the 32-byte shadow space, `mov rax, [rcx]`,
// `call [rax + 8*slot]` (or a direct `call` of a function pointer), all through
// __playgta6_indirect_call.
// ---------------------------------------------------------------------------

constexpr uint64_t kVCall = 0x140010000ULL;
constexpr uint64_t kGuestAdd = 0x140010100ULL;
constexpr uint64_t kGuestMix = 0x140010200ULL;
constexpr uint64_t kGuestRecord = 0x140010300ULL;
constexpr uint64_t kGuestCompare = 0x140010400ULL;
constexpr uint64_t kGuestCount = 0x140010500ULL;
constexpr uint64_t kGuestReturn5 = 0x140010600ULL;
constexpr uint64_t kGuestThreadStart = 0x140010700ULL;
constexpr uint64_t kGuestObjMethod = 0x140010800ULL;
constexpr uint64_t kGuestAsyncDone = 0x140010900ULL;
constexpr uint64_t kGuestNested = 0x140010A00ULL;

struct Arg {
    uint64_t bits;
    bool is_float;
};

struct VCallSpec {
    uint64_t target = 0;  // 0: virtual call through [args[0]] + 8*slot
    uint32_t slot = 0;
    std::vector<Arg> args;
};

VCallSpec g_spec;
bool g_vcall_balanced = true;

void sub_vcall(GuestState* s) {
    static constexpr uint32_t kRegs[4] = {GuestReg::Rcx, GuestReg::Rdx, GuestReg::R8, GuestReg::R9};
    const size_t n = g_spec.args.size();
    const size_t stack_n = n > 4 ? n - 4 : 0;
    const uint64_t frame = ((0x20 + 8 * stack_n + 15) & ~uint64_t{15}) + 8;
    gpr(s, GuestReg::Rsp) -= frame;
    const uint64_t rsp = gpr(s, GuestReg::Rsp);
    for (size_t i = 0; i < n; ++i) {
        const Arg& a = g_spec.args[i];
        if (i < 4) {
            if (a.is_float) {
                s->xmm[i][0] = a.bits;
                gpr(s, kRegs[i]) = 0xBAD0BAD0BAD0BAD0ULL;  // must not be read
            } else {
                gpr(s, kRegs[i]) = a.bits;
                s->xmm[i][0] = 0xBAD1BAD1BAD1BAD1ULL;
            }
        } else {
            store64(rsp + 0x20 + 8 * (i - 4), a.bits);
        }
    }
    uint64_t target = g_spec.target;
    if (target == 0) {
        target = load64(load64(gpr(s, GuestReg::Rcx)) + 8ull * g_spec.slot);
    }
    push(s, kVCall + 0x40);
    __playgta6_indirect_call(s, target);
    if (gpr(s, GuestReg::Rsp) != rsp) {
        g_vcall_balanced = false;
    }
    gpr(s, GuestReg::Rsp) += frame;
    ret(s);
}

// sub_add(a, b): rax = rcx + rdx
void sub_guest_add(GuestState* s) {
    gpr(s, GuestReg::Rax) = gpr(s, GuestReg::Rcx) + gpr(s, GuestReg::Rdx);
    ret(s);
}

// double mix(double a, float b, int64 c, int64 d, double e): a + b + c + d + e (e on the stack)
void sub_guest_mix(GuestState* s) {
    float b;
    const uint32_t b_bits = static_cast<uint32_t>(s->xmm[1][0]);
    std::memcpy(&b, &b_bits, 4);
    const double e = bits_f64(load64(gpr(s, GuestReg::Rsp) + kGuestStackArgOffset));
    const double r = bits_f64(s->xmm[0][0]) + b + static_cast<double>(static_cast<int64_t>(gpr(s, GuestReg::R8))) +
                     static_cast<double>(static_cast<int64_t>(gpr(s, GuestReg::R9))) + e;
    s->xmm[0][0] = f64_bits(r);
    ret(s);
}

// void record(void* ctx, bool canceled): stores (ctx, canceled) in globals.
std::atomic<uint64_t> g_record_ctx{0};
std::atomic<int> g_record_canceled{-1};
std::atomic<int> g_record_calls{0};
std::atomic<std::thread::id> g_record_thread{};

void sub_guest_record(GuestState* s) {
    g_record_ctx = gpr(s, GuestReg::Rcx);
    g_record_canceled = static_cast<int>(gpr(s, GuestReg::Rdx) & 0xFF);
    g_record_thread = std::this_thread::get_id();
    g_record_calls.fetch_add(1);
    ret(s);
}

// int compare(const int32_t* a, const int32_t* b)
std::atomic<int> g_compare_calls{0};
void sub_guest_compare(GuestState* s) {
    g_compare_calls.fetch_add(1);
    const int32_t a = load32(gpr(s, GuestReg::Rcx));
    const int32_t b = load32(gpr(s, GuestReg::Rdx));
    gpr(s, GuestReg::Rax) = static_cast<uint64_t>(static_cast<int64_t>(a < b ? -1 : (a > b ? 1 : 0)));
    ret(s);
}

// void count(): ++g_count
std::atomic<int> g_count{0};
void sub_guest_count(GuestState* s) {
    g_count.fetch_add(1);
    ret(s);
}

void sub_guest_return5(GuestState* s) {
    gpr(s, GuestReg::Rax) = 5;
    ret(s);
}

// DWORD start(void* p): returns p + 1, records the thread.
std::atomic<std::thread::id> g_start_thread{};
void sub_guest_thread_start(GuestState* s) {
    g_start_thread = std::this_thread::get_id();
    gpr(s, GuestReg::Rax) = gpr(s, GuestReg::Rcx) + 1;
    ret(s);
}

// Guest COM method: rax = this + rdx * 3
void sub_guest_obj_method(GuestState* s) {
    gpr(s, GuestReg::Rax) = gpr(s, GuestReg::Rcx) + gpr(s, GuestReg::Rdx) * 3;
    ret(s);
}

// void done(XAsyncBlock* block): records the block.
std::atomic<uint64_t> g_async_block{0};
void sub_guest_async_done(GuestState* s) {
    g_async_block = gpr(s, GuestReg::Rcx);
    ret(s);
}

// int nested(int x): calls the host method slot in g_nested_slot of g_nested_obj with x, + 1000.
uint64_t g_nested_obj = 0;
uint32_t g_nested_slot = 0;
void sub_guest_nested(GuestState* s) {
    const uint64_t x = gpr(s, GuestReg::Rcx);
    gpr(s, GuestReg::Rsp) -= 0x28;
    gpr(s, GuestReg::Rcx) = g_nested_obj;
    gpr(s, GuestReg::Rdx) = x;
    gpr(s, GuestReg::R8) = 1;
    push(s, kGuestNested + 0x20);
    __playgta6_indirect_call(s, load64(load64(g_nested_obj) + 8ull * g_nested_slot));
    gpr(s, GuestReg::Rsp) += 0x28;
    gpr(s, GuestReg::Rax) = static_cast<uint32_t>(gpr(s, GuestReg::Rax)) + 1000;
    ret(s);
}

const LiftedFunctionEntry kFunctions[] = {
    {kVCall, sub_vcall},
    {kGuestAdd, sub_guest_add},
    {kGuestMix, sub_guest_mix},
    {kGuestRecord, sub_guest_record},
    {kGuestCompare, sub_guest_compare},
    {kGuestCount, sub_guest_count},
    {kGuestReturn5, sub_guest_return5},
    {kGuestThreadStart, sub_guest_thread_start},
    {kGuestObjMethod, sub_guest_obj_method},
    {kGuestAsyncDone, sub_guest_async_done},
    {kGuestNested, sub_guest_nested},
};

// ---------------------------------------------------------------------------
// Fault handling: throw out of the guest call
// ---------------------------------------------------------------------------

struct GuestFault {
    FaultInfo info;
};

void throwing_handler(const FaultInfo* info) {
    throw GuestFault{*info};
}

template <typename F>
bool expect_fault(F&& body, GuestFault* out) {
    try {
        body();
    } catch (const GuestFault& fault) {
        *out = fault;
        return true;
    }
    return false;
}

bool diag_contains(const char* needle) {
    for (const std::string& d : com::diagnostics()) {
        if (d.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Fake interfaces
// ---------------------------------------------------------------------------

struct Vec2 {
    float x;
    float y;
};

struct Big {
    uint64_t a;
    uint64_t b;
    uint64_t c;
};

enum class Mode : uint32_t { A = 1, B = 2 };

constexpr Guid kIidCalc = com::make_guid("6d5140c1-7436-11ce-8034-00aa006009fa");
constexpr Guid kIidCalcBase = com::make_guid("aaaaaaaa-bbbb-cccc-dddd-000000000001");
constexpr Guid kIidOther = com::make_guid("11111111-2222-3333-4444-555555555555");
constexpr Guid kIidLoose = com::make_guid("99999999-2222-3333-4444-555555555555");
constexpr Guid kIidMissing = com::make_guid("deadbeef-0000-0000-0000-000000000000");

struct IOtherHost {
    virtual int32_t Tag() = 0;
};

struct ICalcHost {
    virtual int32_t Add(int32_t a, int32_t b) = 0;
    virtual double Mix(float f, double d, int32_t i, float g, double h) = 0;
    virtual int64_t Sum7(int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f, int64_t g) = 0;
    virtual HResult Create(uint32_t value, OutPtr out_object, GuestPtr<uint32_t> out_value) = 0;
    virtual Big MakeBig(uint64_t seed) = 0;
    virtual uint64_t Structs(Vec2 v, Big b, const Guid& g) = 0;
    virtual bool Flag(bool b, Mode m, uint8_t small) = 0;
    virtual HResult Link(InterfaceArg<IOtherHost> other) = 0;
    virtual int32_t Apply(uint64_t guest_fn, int32_t x) = 0;
    virtual float Half(float x) = 0;
    virtual uint32_t ReadPtr(const uint32_t* value) = 0;
};

void raw_echo(com::CallContext& ctx) {
    // raw method: rax = this + 5th Win64 argument (stack)
    ctx.set_rax(ctx.this_ptr + ctx.int_arg(4));
}

using CalcM = com::Methods<ICalcHost>;
constexpr com::MethodDesc kCalcMethods[] = {
    CalcM::method<&ICalcHost::Add>("Add"),          // 3
    CalcM::method<&ICalcHost::Mix>("Mix"),          // 4
    CalcM::method<&ICalcHost::Sum7>("Sum7"),        // 5
    CalcM::method<&ICalcHost::Create>("Create"),    // 6
    CalcM::method<&ICalcHost::MakeBig>("MakeBig"),  // 7
    CalcM::method<&ICalcHost::Structs>("Structs"),  // 8
    CalcM::method<&ICalcHost::Flag>("Flag"),        // 9
    CalcM::method<&ICalcHost::Link>("Link"),        // 10
    CalcM::method<&ICalcHost::Apply>("Apply"),      // 11
    CalcM::method<&ICalcHost::Half>("Half"),        // 12
    CalcM::method<&ICalcHost::ReadPtr>("ReadPtr"),  // 13
    CalcM::raw<&raw_echo>("RawEcho"),               // 14
    CalcM::stub("NamedButMissing"),                 // 15
};
constexpr Guid kCalcExtra[] = {kIidCalcBase};
constexpr auto kICalc = com::declare_interface<ICalcHost>("ICalc", kIidCalc, kCalcMethods,
                                                          {.slot_count = 20, .extra_iids = kCalcExtra});

using OtherM = com::Methods<IOtherHost>;
constexpr com::MethodDesc kOtherMethods[] = {OtherM::method<&IOtherHost::Tag>("Tag")};
constexpr auto kIOther = com::declare_interface<IOtherHost>("IOther", kIidOther, kOtherMethods);

// Same host class, lenient policy for unknown slots.
constexpr auto kILoose = com::declare_interface<IOtherHost>(
    "ILoose", kIidLoose, kOtherMethods, {.slot_count = 8, .policy = com::SlotPolicy::ReturnNotImpl});

std::atomic<int> g_calc_destroyed{0};

class Calc : public com::ComObject, public ICalcHost, public IOtherHost {
public:
    static constexpr const char* kComClassName = "Calc";
    static constexpr uint64_t kComPayloadSize = 32;

    explicit Calc(int32_t tag) : tag_(tag) {}
    ~Calc() override { g_calc_destroyed.fetch_add(1); }

    int32_t Add(int32_t a, int32_t b) override { return a + b; }
    double Mix(float f, double d, int32_t i, float g, double h) override {
        last_mix = {f, d, i, g, h};
        return f + d + i + g + h;
    }
    int64_t Sum7(int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f, int64_t g) override {
        return a + 10 * b + 100 * c + 1000 * d + 10000 * e + 100000 * f + 1000000 * g;
    }
    HResult Create(uint32_t value, OutPtr out_object, GuestPtr<uint32_t> out_value) override {
        if (out_object.is_null()) {
            return com::E_POINTER_HR;
        }
        com::Ref<Calc> child = com::create_object<Calc>(com::implements(kICalc, kIOther), static_cast<int32_t>(value));
        out_value.write(value * 2);
        out_object.write(child.detach(kIOther));
        return com::S_OK_HR;
    }
    Big MakeBig(uint64_t seed) override { return Big{seed, seed + 1, seed + 2}; }
    uint64_t Structs(Vec2 v, Big b, const Guid& g) override {
        last_vec = v;
        last_big = b;
        last_guid = g;
        return b.a + b.b + b.c + g.data1;
    }
    bool Flag(bool b, Mode m, uint8_t small) override {
        last_flag = b;
        last_mode = m;
        last_small = small;
        return !b;
    }
    HResult Link(InterfaceArg<IOtherHost> other) override {
        if (other.invalid()) {
            return com::E_INVALIDARG_HR;
        }
        if (!other) {
            return com::E_POINTER_HR;
        }
        linked_tag = other->Tag();
        return com::S_OK_HR;
    }
    int32_t Apply(uint64_t guest_fn, int32_t x) override {
        const uint64_t rsp_before = com::current_guest_state().gpr[GuestReg::Rsp];
        const int32_t r = com::GuestFunction<int32_t(int32_t, int32_t)>(guest_fn)(x, 7);
        apply_rsp_ok = com::current_guest_state().gpr[GuestReg::Rsp] == rsp_before;
        return r * 2;
    }
    float Half(float x) override { return x / 2; }
    uint32_t ReadPtr(const uint32_t* value) override { return value != nullptr ? *value + 1 : 0xFFFFFFFFu; }
    int32_t Tag() override { return tag_; }

    struct MixArgs {
        float f;
        double d;
        int32_t i;
        float g;
        double h;
    } last_mix{};
    Vec2 last_vec{};
    Big last_big{};
    Guid last_guid{};
    bool last_flag = false;
    Mode last_mode = Mode::A;
    uint8_t last_small = 0;
    int32_t linked_tag = 0;
    bool apply_rsp_ok = false;

private:
    int32_t tag_;
};

class Loose : public com::ComObject, public IOtherHost {
public:
    int32_t Tag() override { return 77; }
};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

struct Fixture {
    VirtualMemoryArena arena;
    GuestThread thread{};
    uint64_t vcall_entry = 0;

    Fixture() {
        reset_guest_runtime();
        com::reset_com_bridge();
        cbk::reset_callbacks();
        com::set_guest_memory(&arena);
        set_fault_handler(throwing_handler);
        g_arena = &arena;
        CHECK(register_lifted_functions(kFunctions));
        CHECK(create_guest_thread(arena, 256 * 1024, thread));
        g_vcall_balanced = true;
    }

    ~Fixture() {
        com::reset_com_bridge();
        cbk::reset_callbacks();
        destroy_guest_thread(arena, thread);
        com::set_guest_memory(nullptr);
        reset_guest_runtime();
        g_arena = nullptr;
    }

    GuestState& state() { return thread.state; }

    void restore_stack() { thread.state.gpr[GuestReg::Rsp] = thread.stack_top; }

    // Virtual call of `slot` on guest object `obj`: args after `this`.
    uint64_t vcall(uint64_t obj, uint32_t slot, std::initializer_list<Arg> args = {}) {
        g_spec = VCallSpec{};
        g_spec.slot = slot;
        g_spec.args.push_back({obj, false});
        g_spec.args.insert(g_spec.args.end(), args.begin(), args.end());
        return call_guest(arena, thread.state, kVCall);
    }

    // Direct call of a guest-callable address.
    uint64_t dcall(uint64_t target, std::initializer_list<Arg> args = {}) {
        g_spec = VCallSpec{};
        g_spec.target = target;
        g_spec.args.assign(args.begin(), args.end());
        return call_guest(arena, thread.state, kVCall);
    }

    uint64_t guid_in_guest(const Guid& g) {
        const uint64_t va = com::guest_alloc(16);
        CHECK(com::write_guest(va, g));
        return va;
    }
};

Arg I(uint64_t v) {
    return Arg{v, false};
}
Arg F32(float f) {
    return Arg{f32_bits(f), true};
}
Arg F64(double d) {
    return Arg{f64_bits(d), true};
}

int32_t as_hr(uint64_t rax) {
    return static_cast<int32_t>(static_cast<uint32_t>(rax));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_guid_helpers() {
    constexpr Guid g = com::make_guid("01234567-89ab-cdef-0123-456789ABCDEF");
    static_assert(g.data1 == 0x01234567u && g.data2 == 0x89ab && g.data3 == 0xcdef);
    static_assert(g.data4[0] == 0x01 && g.data4[7] == 0xEF);
    CHECK(com::format_guid(g) == "{01234567-89AB-CDEF-0123-456789ABCDEF}");
    CHECK(com::format_guid(com::IID_IUnknown_GUID) == "{00000000-0000-0000-C000-000000000046}");
}

void test_layout_and_marshaling() {
    Fixture fx;
    com::Ref<Calc> calc = com::create_object<Calc>(com::implements(kICalc, kIOther), 42);
    CHECK(calc);
    const uint64_t obj = calc->interface_pointer(kICalc);
    const uint64_t other = calc->interface_pointer(kIOther);
    CHECK(obj == calc->identity());
    CHECK(other == obj + 8);
    CHECK(calc->payload_address() == obj + 16);
    CHECK(calc->payload_size() == 32);
    CHECK(load64(obj) == com::vtable_address(kICalc.info));
    CHECK(load64(other) == com::vtable_address(kIOther.info));
    CHECK(load64(load64(obj) + 8 * 3) == com::slot_address(kICalc.info, 3));
    CHECK(is_synthetic_address(com::slot_address(kICalc.info, 3)));
    CHECK(com::slot_address(kICalc.info, 20) == 0);

    // Add (slot 3)
    CHECK(fx.vcall(obj, 3, {I(40), I(2)}) == 42);
    // int32 result in eax is sign-extended into rax
    CHECK(as_hr(fx.vcall(obj, 3, {I(0xFFFFFFFFu), I(0)})) == -1);
    // Integer arguments use only the low 32 bits.
    CHECK(fx.vcall(obj, 3, {I(0xABCD000000000001ULL), I(1)}) == 2);

    // Mix (slot 4): f in xmm1, d in xmm2, i in r9, g and h on the stack.
    fx.vcall(obj, 4, {F32(1.5f), F64(2.25), I(static_cast<uint64_t>(-3)), F32(4.0f), F64(0.125)});
    CHECK(calc->last_mix.f == 1.5f);
    CHECK(calc->last_mix.d == 2.25);
    CHECK(calc->last_mix.i == -3);
    CHECK(calc->last_mix.g == 4.0f);
    CHECK(calc->last_mix.h == 0.125);
    CHECK(bits_f64(fx.state().xmm[0][0]) == 1.5 + 2.25 - 3 + 4.0 + 0.125);

    // Sum7 (slot 5): 7 args, 4 on the stack
    CHECK(fx.vcall(obj, 5, {I(1), I(2), I(3), I(4), I(5), I(6), I(7)}) == 7654321);

    // Half (slot 12): float result in xmm0
    fx.vcall(obj, 12, {F32(5.0f)});
    float half;
    const uint32_t half_bits = static_cast<uint32_t>(fx.state().xmm[0][0]);
    std::memcpy(&half, &half_bits, 4);
    CHECK(half == 2.5f);

    // Flag (slot 9): bool in the low byte only, enum, uint8
    CHECK(fx.vcall(obj, 9, {I(0xFF00), I(2), I(0x1234)}) == 1);
    CHECK(calc->last_flag == false);
    CHECK(calc->last_mode == Mode::B);
    CHECK(calc->last_small == 0x34);
    CHECK(fx.vcall(obj, 9, {I(0x01), I(1), I(0)}) == 0);
    CHECK(calc->last_flag == true);

    // MakeBig (slot 7): hidden result pointer in rdx after `this`, returned in rax.
    const uint64_t result_buf = com::guest_alloc(sizeof(Big));
    CHECK(fx.vcall(obj, 7, {I(result_buf), I(100)}) == result_buf);
    Big big{};
    CHECK(com::read_guest(result_buf, big));
    CHECK(big.a == 100 && big.b == 101 && big.c == 102);

    // Structs (slot 8): Vec2 (8 bytes) in rdx by value, Big by pointer, GUID by reference.
    Vec2 v{1.0f, -2.0f};
    uint64_t v_bits;
    std::memcpy(&v_bits, &v, 8);
    const uint64_t big_va = com::guest_alloc(sizeof(Big));
    CHECK(com::write_guest(big_va, Big{1, 2, 3}));
    const uint64_t guid_va = fx.guid_in_guest(kIidOther);
    CHECK(fx.vcall(obj, 8, {I(v_bits), I(big_va), I(guid_va)}) == 6 + kIidOther.data1);
    CHECK(calc->last_vec.x == 1.0f && calc->last_vec.y == -2.0f);
    CHECK(calc->last_big.c == 3);
    CHECK(calc->last_guid == kIidOther);

    // ReadPtr (slot 13): raw pointer through the translation seam; null and unmapped -> nullptr.
    const uint64_t u32_va = com::guest_alloc(4);
    const uint32_t seven = 7;
    CHECK(com::write_guest(u32_va, seven));
    CHECK(fx.vcall(obj, 13, {I(u32_va)}) == 8);
    CHECK(fx.vcall(obj, 13, {I(0)}) == 0xFFFFFFFFu);
    com::clear_diagnostics();
    CHECK(fx.vcall(obj, 13, {I(0x7000'0000'0000ULL)}) == 0xFFFFFFFFu);
    CHECK(diag_contains("invalid pointer argument"));

    // Raw method (slot 14): reads the 5th Win64 argument itself.
    CHECK(fx.vcall(obj, 14, {I(0), I(0), I(0), I(0x10)}) == obj + 0x10);

    // Out params (slot 6): new object written to *out_object, value to *out_value.
    const uint64_t out_obj = com::guest_alloc(8);
    const uint64_t out_val = com::guest_alloc(4);
    CHECK(as_hr(fx.vcall(obj, 6, {I(21), I(out_obj), I(out_val)})) == com::S_OK_HR);
    CHECK(static_cast<uint32_t>(load32(out_val)) == 42);
    const uint64_t child_other = load64(out_obj);
    CHECK(child_other != 0);
    CHECK(as_hr(fx.vcall(child_other, 3)) == 21);  // IOther::Tag of the child
    CHECK(as_hr(fx.vcall(obj, 6, {I(21), I(0), I(out_val)})) == com::E_POINTER_HR);
    CHECK(com::bridge_stats().live_objects == 2);
    CHECK(fx.vcall(child_other, 2) == 0);  // Release -> destroyed
    CHECK(com::bridge_stats().live_objects == 1);

    // Link (slot 10): interface argument resolution.
    CHECK(as_hr(fx.vcall(obj, 10, {I(other)})) == com::S_OK_HR);
    CHECK(calc->linked_tag == 42);
    CHECK(as_hr(fx.vcall(obj, 10, {I(0)})) == com::E_POINTER_HR);
    com::clear_diagnostics();
    CHECK(as_hr(fx.vcall(obj, 10, {I(obj)})) == com::E_INVALIDARG_HR);  // ICalc pointer, not IOther
    CHECK(diag_contains("invalid interface argument"));
    CHECK(as_hr(fx.vcall(obj, 10, {I(0x1234560)})) == com::E_INVALIDARG_HR);

    // Flat function thunk with mixed arguments.
    CHECK(g_vcall_balanced);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
}

int64_t flat_fn(int64_t a, double b, int32_t c, float d, int64_t e, double f) {
    return a + static_cast<int64_t>(b * 10) + c + static_cast<int64_t>(d * 100) + e +
           static_cast<int64_t>(f * 1000);
}

double flat_ctx(com::CallContext& ctx, double a, uint64_t b) {
    return a + static_cast<double>(b) + (ctx.object == nullptr ? 0.5 : 0.0);
}

void test_flat_function_thunk() {
    Fixture fx;
    const uint64_t fn = register_host_callable(com::function_thunk<&flat_fn>(), "flat_fn");
    CHECK(fx.dcall(fn, {I(1), F64(2.0), I(3), F32(4.0f), I(5), F64(6.0)}) == 1 + 20 + 3 + 400 + 5 + 6000);
    const uint64_t fn2 = register_host_callable(com::function_thunk<&flat_ctx>(), "flat_ctx");
    fx.dcall(fn2, {F64(1.0), I(2)});
    CHECK(bits_f64(fx.state().xmm[0][0]) == 3.5);
    CHECK(g_vcall_balanced);
}

void test_iunknown() {
    Fixture fx;
    g_calc_destroyed = 0;
    com::Ref<Calc> calc = com::create_object<Calc>(com::implements(kICalc, kIOther), 1);
    const uint64_t obj = calc->interface_pointer(kICalc);
    const uint64_t other = calc->interface_pointer(kIOther);
    const uint64_t ppv = com::guest_alloc(8);
    const uint64_t riid_unknown = fx.guid_in_guest(com::IID_IUnknown_GUID);
    const uint64_t riid_calc = fx.guid_in_guest(kIidCalc);
    const uint64_t riid_base = fx.guid_in_guest(kIidCalcBase);
    const uint64_t riid_other = fx.guid_in_guest(kIidOther);
    const uint64_t riid_missing = fx.guid_in_guest(kIidMissing);

    // Identity: IUnknown from either interface is the same pointer.
    CHECK(as_hr(fx.vcall(other, 0, {I(riid_unknown), I(ppv)})) == com::S_OK_HR);
    CHECK(load64(ppv) == obj);
    CHECK(as_hr(fx.vcall(obj, 0, {I(riid_unknown), I(ppv)})) == com::S_OK_HR);
    CHECK(load64(ppv) == obj);
    // Same IID -> same pointer; extra (base) IID answered by ICalc.
    CHECK(as_hr(fx.vcall(obj, 0, {I(riid_other), I(ppv)})) == com::S_OK_HR);
    CHECK(load64(ppv) == other);
    CHECK(as_hr(fx.vcall(other, 0, {I(riid_calc), I(ppv)})) == com::S_OK_HR);
    CHECK(load64(ppv) == obj);
    CHECK(as_hr(fx.vcall(other, 0, {I(riid_base), I(ppv)})) == com::S_OK_HR);
    CHECK(load64(ppv) == obj);
    CHECK(calc->ref_count() == 6);
    // Unknown IID: E_NOINTERFACE and *ppv = 0.
    store64(ppv, 0x1111);
    CHECK(as_hr(fx.vcall(obj, 0, {I(riid_missing), I(ppv)})) == com::E_NOINTERFACE_HR);
    CHECK(load64(ppv) == 0);
    // Null ppv / null riid: E_POINTER.
    CHECK(as_hr(fx.vcall(obj, 0, {I(riid_calc), I(0)})) == com::E_POINTER_HR);
    CHECK(as_hr(fx.vcall(obj, 0, {I(0), I(ppv)})) == com::E_POINTER_HR);
    CHECK(calc->ref_count() == 6);

    // AddRef / Release return the new count; refs are per object.
    CHECK(fx.vcall(other, 1) == 7);
    for (int i = 0; i < 6; ++i) {
        CHECK(fx.vcall(i % 2 ? obj : other, 2) == static_cast<uint64_t>(6 - i));
    }
    CHECK(calc->ref_count() == 1);

    // Host-side QI.
    uint64_t out = 0;
    CHECK(calc->query_interface(kIidOther, &out) == com::S_OK_HR && out == other);
    CHECK(calc->query_interface(kIidMissing, &out) == com::E_NOINTERFACE_HR && out == 0);
    CHECK(calc->release() == 1);

    // Final release from the guest: destroyed, vtable pointers poisoned.
    Calc* raw = calc.get();
    (void)raw;
    calc.detach(kICalc);  // the guest now owns the last reference
    CHECK(com::bridge_stats().live_objects == 1);
    CHECK(fx.vcall(obj, 2) == 0);
    CHECK(g_calc_destroyed == 1);
    const com::BridgeStats stats = com::bridge_stats();
    CHECK(stats.live_objects == 0);
    CHECK(stats.objects_created == 1);
    CHECK(stats.objects_destroyed == 1);
    CHECK(com::live_objects().empty());
    // A call through the dangling pointer is diagnosed, not a host crash.
    GuestFault fault{};
    com::clear_diagnostics();
    CHECK(expect_fault([&] { fx.vcall(obj, 3, {I(1), I(2)}); }, &fault));
    CHECK(fault.info.kind == static_cast<uint32_t>(FaultKind::Trap));
    CHECK(diag_contains("released object"));
    fx.restore_stack();
}

void test_unknown_slots_and_bad_pointers() {
    Fixture fx;
    com::Ref<Calc> calc = com::create_object<Calc>(com::implements(kICalc, kIOther), 1);
    const uint64_t obj = calc->interface_pointer(kICalc);
    const uint64_t other = calc->interface_pointer(kIOther);
    GuestFault fault{};

    // Unnamed slot past the declared methods: logged once, counted, trapped.
    com::clear_diagnostics();
    CHECK(expect_fault([&] { fx.vcall(obj, 17); }, &fault));
    fx.restore_stack();
    CHECK(fault.info.kind == static_cast<uint32_t>(FaultKind::UnimplementedSlot));
    CHECK(fault.info.detail == 17);
    CHECK(fault.info.va == com::slot_address(kICalc.info, 17));
    {
        char text[512];
        format_fault(fault.info, text, sizeof(text));
        CHECK(std::string(text).find("guest fault: unimplemented COM interface slot at 0x") == 0);
        CHECK(std::string(text).find("(slot 17: ICalc::slot 17 (unimplemented))") != std::string::npos);
    }
    CHECK(diag_contains("unimplemented ICalc {6D5140C1-7436-11CE-8034-00AA006009FA} slot 17"));
    CHECK(expect_fault([&] { fx.vcall(obj, 17); }, &fault));
    fx.restore_stack();
    CHECK(com::stub_hits(kICalc, 17) == 2);
    size_t logged = 0;
    for (const std::string& d : com::diagnostics()) {
        logged += d.find("slot 17") != std::string::npos ? 1 : 0;
    }
    CHECK(logged == 1);

    // Named stub.
    CHECK(expect_fault([&] { fx.vcall(obj, 15); }, &fault));
    fx.restore_stack();
    CHECK(diag_contains("slot 15 (NamedButMissing)"));
    CHECK(com::stub_hits(kICalc, 15) == 1);
    CHECK(com::stub_hits(kICalc, 3) == 0);
    const auto report = com::stub_hit_report();
    CHECK(report.size() == 2);
    CHECK(report.size() == 2 && report[0].slot == 15 && report[0].method_name == "NamedButMissing" &&
          report[1].slot == 17 && report[1].hits == 2 && report[1].iid == kIidCalc);

    // ReturnNotImpl policy.
    com::Ref<Loose> loose = com::create_object<Loose>(com::implements(kILoose));
    const uint64_t lp = loose->identity();
    CHECK(fx.vcall(lp, 3) == 77);
    CHECK(as_hr(fx.vcall(lp, 6)) == com::E_NOTIMPL_HR);
    CHECK(com::stub_hits(kILoose, 6) == 1);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);

    // Bogus `this`: a valid slot address called with a made-up object pointer.
    com::clear_diagnostics();
    const uint64_t add = com::slot_address(kICalc.info, 3);
    CHECK(expect_fault([&] { fx.dcall(add, {I(0x123456780ULL), I(1), I(2)}); }, &fault));
    fx.restore_stack();
    CHECK(fault.info.kind == static_cast<uint32_t>(FaultKind::Trap));
    CHECK(fault.info.detail == 0x123456780ULL);
    CHECK(diag_contains("ICalc::Add called on unknown object"));
    // Wrong interface: an IOther pointer passed as `this` to an ICalc method.
    CHECK(expect_fault([&] { fx.dcall(add, {I(other), I(1), I(2)}); }, &fault));
    fx.restore_stack();
    CHECK(diag_contains("which is a IOther pointer of Calc"));
    // Bogus this for IUnknown slots.
    CHECK(expect_fault([&] { fx.dcall(com::slot_address(kICalc.info, 1), {I(0x42)}); }, &fault));
    fx.restore_stack();
    CHECK(diag_contains("IUnknown::AddRef called on unknown object"));
    CHECK(calc->ref_count() == 1);
}

void test_host_to_guest_calls() {
    Fixture fx;
    // Typed guest function with float and stack arguments.
    com::GuestStateScope scope(fx.state());
    const double mixed =
        com::GuestFunction<double(double, float, int64_t, int64_t, double)>(kGuestMix)(1.5, 2.5f, -1, 10, 0.25);
    CHECK(mixed == 1.5 + 2.5 - 1 + 10 + 0.25);
    CHECK(com::GuestFunction<int64_t(int64_t, int64_t)>(kGuestAdd)(40, 2) == 42);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);

    // Method of a guest-implemented object: vtable built in guest memory.
    const uint64_t vtable = com::guest_alloc(16);
    store64(vtable + 8, kGuestObjMethod);
    const uint64_t object = com::guest_alloc(16);
    store64(object, vtable);
    CHECK(com::call_guest_method<uint64_t>(fx.state(), object, 1, uint64_t{5}) == object + 15);
    GuestFault fault{};
    CHECK(expect_fault([&] { com::call_guest_method(fx.state(), 0x9999000ULL, 1); }, &fault));
    fx.restore_stack();
}

void test_nested_callbacks() {
    Fixture fx;
    com::Ref<Calc> calc = com::create_object<Calc>(com::implements(kICalc, kIOther), 3);
    const uint64_t obj = calc->identity();
    // guest -> host Apply -> guest add(x, 7) -> back: (x + 7) * 2
    CHECK(fx.vcall(obj, 11, {I(kGuestAdd), I(10)}) == 34);
    CHECK(calc->apply_rsp_ok);
    // guest nested() -> host Add(x, 1) via vtable; host -> guest -> host -> guest via GuestFunction
    g_nested_obj = obj;
    g_nested_slot = 3;
    CHECK(fx.vcall(obj, 11, {I(kGuestNested), I(5)}) == ((5 + 1) + 1000) * 2);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
    CHECK(g_vcall_balanced);
}

void test_concurrent_refcounts() {
    Fixture fx;
    g_calc_destroyed = 0;
    com::Ref<Calc> calc = com::create_object<Calc>(com::implements(kICalc, kIOther), 9);
    const uint64_t obj = calc->identity();
    const uint64_t other = calc->interface_pointer(kIOther);
    const uint64_t addref = com::slot_address(kICalc.info, 1);
    const uint64_t release = com::slot_address(kICalc.info, 2);
    const uint64_t tag = com::slot_address(kIOther.info, 3);
    constexpr int kThreads = 8;
    constexpr int kIters = 2000;
    std::atomic<int> errors{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t] {
            GuestThread gt{};
            if (!create_guest_thread(fx.arena, 64 * 1024, gt)) {
                errors.fetch_add(1);
                return;
            }
            for (int i = 0; i < kIters; ++i) {
                const uint64_t self = (i + t) % 2 ? obj : other;
                const uint64_t args[1] = {self};
                if (call_guest(fx.arena, gt.state, addref, args) < 2) {
                    errors.fetch_add(1);
                }
                const uint64_t targs[1] = {other};
                if (call_guest(fx.arena, gt.state, tag, targs) != 9) {
                    errors.fetch_add(1);
                }
                if (call_guest(fx.arena, gt.state, release, args) < 1) {
                    errors.fetch_add(1);
                }
            }
            destroy_guest_thread(fx.arena, gt);
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    CHECK(errors == 0);
    CHECK(calc->ref_count() == 1);
    CHECK(g_calc_destroyed == 0);
    calc.reset();
    CHECK(g_calc_destroyed == 1);
    CHECK(com::bridge_stats().live_objects == 0);
}

void test_leak_accounting() {
    Fixture fx;
    {
        com::Ref<Loose> warm = com::create_object<Loose>(com::implements(kILoose));
    }
    const uint64_t baseline = com::bridge_stats().guest_bytes_in_use;  // vtables only
    std::vector<com::Ref<Calc>> objects;
    for (int i = 0; i < 50; ++i) {
        objects.push_back(com::create_object<Calc>(com::implements(kICalc, kIOther), i));
    }
    CHECK(com::bridge_stats().live_objects == 50);
    CHECK(com::live_objects().size() == 50);
    CHECK(com::live_objects()[0].class_name == "Calc");
    CHECK(com::bridge_stats().guest_bytes_in_use > baseline);
    objects.clear();
    com::Ref<Calc> again = com::create_object<Calc>(com::implements(kICalc, kIOther), 1);
    again.reset();
    const com::BridgeStats stats = com::bridge_stats();
    CHECK(stats.live_objects == 0);
    CHECK(stats.objects_created == 52);
    CHECK(stats.objects_destroyed == 52);
    // Only the new ICalc/IOther vtables remain (the warm-up already materialized ILoose and the
    // released-object vtable); every object block was freed.
    CHECK(stats.guest_bytes_in_use == baseline + 8 * (kICalc.info.slot_count + kIOther.info.slot_count));
}

// --- trampolines ---------------------------------------------------------------------------

template <typename Pred>
bool wait_for(Pred pred) {
    for (int i = 0; i < 2000; ++i) {
        if (pred()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return pred();
}

void test_task_queue_trampolines() {
    Fixture fx;
    const uint64_t submit = register_host_callable(&pg6_gthunk_xtask_queue_submit_callback, "XTaskQueueSubmitCallback");
    const uint64_t terminate = register_host_callable(&pg6_gthunk_xtask_queue_terminate, "XTaskQueueTerminate");

    // Manual queue dispatched from a host thread with no bound guest state: a guest thread is
    // created lazily for it.
    Pg6XTaskQueueHandle queue = nullptr;
    CHECK(pg6_gdk_xtask_queue_create(0, 0, &queue) == 0);  // Manual
    const uint64_t q = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(queue));
    g_record_calls = 0;
    CHECK(as_hr(fx.dcall(submit, {I(q), I(0), I(0xC0FFEE), I(kGuestRecord)})) == 0);
    CHECK(cbk::live_trampolines() == 1);
    CHECK(com::bound_guest_state() == nullptr);
    CHECK(pg6_gdk_xtask_queue_dispatch(queue, 0, 0) == 1);
    CHECK(g_record_calls == 1);
    CHECK(g_record_ctx == 0xC0FFEE);
    CHECK(g_record_canceled == 0);
    CHECK(cbk::live_trampolines() == 0);
    CHECK(com::bridge_stats().lazy_guest_threads == 1);
    // Null callback is rejected by the shim, nothing leaks.
    CHECK(as_hr(fx.dcall(submit, {I(q), I(0), I(1), I(0)})) == com::E_INVALIDARG_HR);
    CHECK(cbk::live_trampolines() == 0);
    // Terminate with a guest completion callback; pending work is cancelled first.
    g_record_calls = 0;
    CHECK(as_hr(fx.dcall(submit, {I(q), I(0), I(0xAA), I(kGuestRecord)})) == 0);
    CHECK(as_hr(fx.dcall(terminate, {I(q), I(1), I(0xBB), I(kGuestRecord)})) == 0);
    CHECK(wait_for([] { return g_record_calls.load() == 2; }));
    CHECK(g_record_ctx == 0xBB);
    CHECK(cbk::live_trampolines() == 0);
    pg6_gdk_xtask_queue_close_handle(queue);

    // Thread-pool queue: the guest callback runs on a worker thread.
    CHECK(pg6_gdk_xtask_queue_create(1, 1, &queue) == 0);  // ThreadPool
    g_record_calls = 0;
    g_record_thread = std::thread::id{};
    CHECK(as_hr(fx.dcall(submit, {I(reinterpret_cast<uintptr_t>(queue)), I(0), I(0x5150), I(kGuestRecord)})) == 0);
    CHECK(wait_for([] { return g_record_calls.load() == 1; }));
    CHECK(g_record_ctx == 0x5150);
    CHECK(g_record_thread.load() != std::this_thread::get_id());
    CHECK(pg6_gdk_xtask_queue_terminate(queue, 1, nullptr, nullptr) == 0);
    pg6_gdk_xtask_queue_close_handle(queue);
    CHECK(cbk::live_trampolines() == 0);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
}

void test_create_thread_trampoline() {
    Fixture fx;
    const uint64_t create = register_host_callable(&pg6_gthunk_create_thread, "CreateThread");
    const uint64_t tid_va = com::guest_alloc(4);
    g_start_thread = std::thread::id{};
    const uint64_t handle = fx.dcall(create, {I(0), I(128 * 1024), I(kGuestThreadStart), I(41), I(0), I(tid_va)});
    CHECK(handle != 0);
    CHECK(pg6_win32_wait_for_single_object(handle, 0xFFFFFFFFu) == 0);
    uint32_t exit_code = 0;
    CHECK(pg6_win32_get_exit_code_thread(handle, &exit_code) == 1);
    CHECK(exit_code == 42);
    CHECK(load32(tid_va) != 0);
    CHECK(g_start_thread.load() != std::this_thread::get_id());
    CHECK(cbk::live_trampolines() == 0);
    pg6_win32_close_handle(handle);
    // Null start routine: the shim fails, no trampoline is created.
    CHECK(fx.dcall(create, {I(0), I(0), I(0), I(0), I(0), I(0)}) == 0);
    CHECK(cbk::live_trampolines() == 0);
}

void test_crt_callbacks() {
    Fixture fx;
    // qsort / bsearch with a guest comparator on guest memory.
    const int32_t values[] = {5, -3, 12, 0, 7, 7, -10, 99, 1};
    constexpr uint64_t n = sizeof(values) / sizeof(values[0]);
    const uint64_t arr = com::guest_alloc(sizeof(values));
    CHECK(com::write_guest(arr, values, sizeof(values)));
    const uint64_t qsort_fn = register_host_callable(&pg6_gthunk_qsort, "qsort");
    g_compare_calls = 0;
    fx.dcall(qsort_fn, {I(arr), I(n), I(4), I(kGuestCompare)});
    CHECK(g_compare_calls > 0);
    int32_t sorted[n];
    CHECK(com::read_guest(arr, sorted, sizeof(sorted)));
    const int32_t expected[n] = {-10, -3, 0, 1, 5, 7, 7, 12, 99};
    CHECK(std::memcmp(sorted, expected, sizeof(sorted)) == 0);

    const uint64_t bsearch_fn = register_host_callable(&pg6_gthunk_bsearch, "bsearch");
    const uint64_t key = com::guest_alloc(4);
    const int32_t twelve = 12;
    CHECK(com::write_guest(key, twelve));
    CHECK(fx.dcall(bsearch_fn, {I(key), I(arr), I(n), I(4), I(kGuestCompare)}) == arr + 7 * 4);
    const int32_t missing = 13;
    CHECK(com::write_guest(key, missing));
    CHECK(fx.dcall(bsearch_fn, {I(key), I(arr), I(n), I(4), I(kGuestCompare)}) == 0);

    // _initterm / _initterm_e over guest function-pointer tables.
    const uint64_t table = com::guest_alloc(32);
    store64(table, kGuestCount);
    store64(table + 8, 0);
    store64(table + 16, kGuestCount);
    g_count = 0;
    fx.dcall(register_host_callable(&pg6_gthunk_initterm, "_initterm"), {I(table), I(table + 24)});
    CHECK(g_count == 2);
    store64(table + 8, kGuestReturn5);
    g_count = 0;
    CHECK(fx.dcall(register_host_callable(&pg6_gthunk_initterm_e, "_initterm_e"), {I(table), I(table + 24)}) == 5);
    CHECK(g_count == 1);
    CHECK(g_vcall_balanced);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
}

void test_async_shadow() {
    Fixture fx;
    const uint64_t block = com::guest_alloc(sizeof(cbk::GuestAsyncBlock));
    cbk::GuestAsyncBlock guest{};
    guest.callback = kGuestAsyncDone;
    guest.context = 0x77;
    CHECK(com::write_guest(block, guest));
    g_async_block = 0;
    Pg6XAsyncBlock* shadow = cbk::async_shadow_begin(block);
    CHECK(shadow != nullptr);
    CHECK(cbk::async_shadow_find(block) == shadow);
    CHECK(pg6_gdk_xuser_add_async(1, shadow) == 0);
    CHECK(wait_for([] { return g_async_block.load() != 0; }));
    CHECK(g_async_block == block);  // the guest callback received the *guest* block
    CHECK(pg6_gdk_xasync_get_status(cbk::async_shadow_find(block), 1) == 0);
    Pg6XUserHandle user = nullptr;
    CHECK(pg6_gdk_xuser_add_result(cbk::async_shadow_find(block), &user) == 0);
    CHECK(user != nullptr);
    pg6_gdk_xuser_close_handle(user);
    cbk::async_shadow_end(block);
    CHECK(cbk::live_async_shadows() == 0);
    CHECK(cbk::async_shadow_begin(0) == nullptr);
}

void host_only_export() {}

void guest_thunk_export(void* state) {
    gpr(static_cast<GuestState*>(state), GuestReg::Rax) = 0x600D;
    ret(static_cast<GuestState*>(state));
}

void test_get_proc_address_thunk() {
    Fixture fx;
    namespace win32 = playgta6::os::win32;
    const uint64_t proc = win32::proc_address(&guest_thunk_export);
    CHECK(win32::register_export("o16_test_module.dll", "GuestEntry", 7, proc));
    CHECK(win32::register_export("o16_test_module.dll", "HostOnly", 0, win32::proc_address(&host_only_export)));
    const uint64_t synthetic = register_host_callable(&guest_thunk_export, "GuestEntry", proc);
    const uint64_t module = win32::GetModuleHandleA("o16_test_module.dll");
    CHECK(module != 0);
    const uint64_t gpa = register_host_callable(&pg6_gthunk_get_proc_address, "GetProcAddress");
    const char name[] = "GuestEntry";
    const uint64_t name_va = com::guest_alloc(sizeof(name));
    CHECK(com::write_guest(name_va, name, sizeof(name)));
    CHECK(fx.dcall(gpa, {I(module), I(name_va)}) == synthetic);
    CHECK(fx.dcall(gpa, {I(module), I(7)}) == synthetic);  // MAKEINTRESOURCE ordinal
    CHECK(call_guest(fx.arena, fx.state(), synthetic) == 0x600D);
    const char host_only[] = "HostOnly";
    CHECK(com::write_guest(name_va, host_only, sizeof(host_only)));
    com::clear_diagnostics();
    CHECK(fx.dcall(gpa, {I(module), I(name_va)}) == 0);
    CHECK(pg6_win32_get_last_error() == win32::ERROR_PROC_NOT_FOUND);
    CHECK(diag_contains("host-only export"));
    const char missing[] = "Missing";
    CHECK(com::write_guest(name_va, missing, sizeof(missing)));
    CHECK(fx.dcall(gpa, {I(module), I(name_va)}) == 0);
    CHECK(fx.state().gpr[GuestReg::Rsp] == fx.thread.stack_top);
}

}  // namespace

int main() {
    test_guid_helpers();
    test_layout_and_marshaling();
    test_flat_function_thunk();
    test_iunknown();
    test_unknown_slots_and_bad_pointers();
    test_host_to_guest_calls();
    test_nested_callbacks();
    test_concurrent_refcounts();
    test_leak_accounting();
    test_task_queue_trampolines();
    test_create_thread_trampoline();
    test_crt_callbacks();
    test_async_shadow();
    test_get_proc_address_thunk();
    if (g_failures != 0) {
        std::fprintf(stderr, "test_com_bridge: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_com_bridge: all tests passed\n");
    return 0;
}
