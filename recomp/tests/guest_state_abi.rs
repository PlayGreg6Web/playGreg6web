//! The lifter's guest state layout must match the shared runtime ABI (`runtime/abi/playgta6_abi.h`).

use playgta6_recomp::lift;

const ABI_HEADER: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../runtime/abi/playgta6_abi.h"
));

/// Value of `inline constexpr uint64_t <name> = <decimal>;` in the ABI header.
fn abi_constant(name: &str) -> u64 {
    let needle = format!("uint64_t {name} = ");
    let start = ABI_HEADER
        .find(&needle)
        .unwrap_or_else(|| panic!("{name} not found in playgta6_abi.h"))
        + needle.len();
    let end = start
        + ABI_HEADER[start..]
            .find(';')
            .expect("unterminated constant");
    ABI_HEADER[start..end]
        .trim()
        .parse()
        .unwrap_or_else(|e| panic!("{name} is not a decimal literal: {e}"))
}

#[test]
fn lifter_state_layout_matches_runtime_abi() {
    let pairs = [
        ("kGuestStateGprOffset", lift::STATE_GPR_OFFSET),
        ("kGuestStateRipOffset", lift::STATE_RIP_OFFSET),
        ("kGuestStateFlagsOffset", lift::STATE_FLAGS_OFFSET),
        ("kGuestStateFsBaseOffset", lift::STATE_FS_BASE_OFFSET),
        ("kGuestStateGsBaseOffset", lift::STATE_GS_BASE_OFFSET),
        ("kGuestStateXmmOffset", lift::STATE_XMM_OFFSET),
        ("kGuestStateXmmStride", lift::STATE_XMM_STRIDE),
        ("kGuestStateYmmHiOffset", lift::STATE_YMM_HI_OFFSET),
        ("kGuestStateYmmHiStride", lift::STATE_YMM_HI_STRIDE),
        ("kGuestStateMxcsrOffset", lift::STATE_MXCSR_OFFSET),
        ("kGuestStateResumeRipOffset", lift::STATE_RESUME_RIP_OFFSET),
        (
            "kGuestStateUnwindPendingOffset",
            lift::STATE_UNWIND_PENDING_OFFSET,
        ),
        ("kGuestStateSize", lift::STATE_SIZE),
        ("kGuestStackArgOffset", lift::CALLEE_STACK_ARG_OFFSET),
    ];
    for (name, lifter) in pairs {
        assert_eq!(abi_constant(name), lifter, "{name} differs from the lifter");
    }
}

#[test]
fn lifter_helpers_match_runtime_abi() {
    for helper in [
        lift::HELPER_INDIRECT_CALL,
        lift::HELPER_TRAP,
        lift::HELPER_UNIMPLEMENTED,
        lift::HELPER_DIVIDE_ERROR,
    ] {
        let decl = format!("void {helper}(playgta6::GuestState* state, uint64_t ");
        assert!(
            ABI_HEADER.contains(&decl),
            "{helper} is not declared in playgta6_abi.h"
        );
    }
}

#[test]
fn lifter_cpu_helpers_match_runtime_abi() {
    for helper in [
        lift::HELPER_CPUID,
        lift::HELPER_RDTSC,
        lift::HELPER_RDTSCP,
        lift::HELPER_XGETBV,
    ] {
        let decl = format!("void {helper}(playgta6::GuestState* state);");
        assert!(
            ABI_HEADER.contains(&decl),
            "{helper} is not declared in playgta6_abi.h"
        );
    }
}

#[test]
fn lifter_guest_base_symbol_matches_runtime_abi() {
    let decl = format!("extern uint64_t {};", lift::GUEST_BASE_SYMBOL);
    assert!(
        ABI_HEADER.contains(&decl),
        "{} is not declared in playgta6_abi.h",
        lift::GUEST_BASE_SYMBOL
    );
    assert_eq!(abi_constant("kGuestAddressSpaceBase"), 0x1_0000_0000);
}

#[test]
fn lifter_window_constants_match_runtime_abi() {
    use playgta6_recomp::lift::memory;
    assert_eq!(abi_constant("kGuestWindowBase"), memory::GUEST_WINDOW_BASE);
    assert_eq!(abi_constant("kGuestWindowSize"), memory::GUEST_WINDOW_SIZE);
    assert_eq!(abi_constant("kGuestWindowSlotBytes"), memory::SLOT_BYTES);
    assert!(ABI_HEADER.contains(&format!(
        "kGuestWindowTrapDelta = {}ull;",
        memory::GUEST_WINDOW_TRAP_DELTA
    )));
    assert!(ABI_HEADER.contains(&format!(
        "extern int64_t {}[];",
        memory::WINDOW_TABLE_SYMBOL
    )));
    for helper in [
        memory::HELPER_GUEST_READ,
        memory::HELPER_GUEST_SCRATCH,
        memory::HELPER_GUEST_WRITE,
        memory::HELPER_GUEST_MEMMOVE,
        memory::HELPER_GUEST_MEMSET,
    ] {
        assert!(
            ABI_HEADER.contains(&format!(" {helper}(")),
            "{helper} is not declared in playgta6_abi.h"
        );
    }
}
