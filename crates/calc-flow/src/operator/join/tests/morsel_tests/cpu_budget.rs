use super::*;

#[test]
fn cpu_aware_probe_scratch_estimate_has_no_internal_byte_cap() {
    let bytes = scratch_bytes(1_000_000, 1_000_000, 3, 32)
        .expect("checked scratch sizing must defer limits to the configured pool");
    assert!(bytes > 512 << 20);
    assert!(scratch_bytes(usize::MAX, 1, 1, 1).is_none());
    assert!(scratch_bytes(1, 1, usize::MAX, 32).is_none());
}

#[test]
fn cpu_aware_probe_result_estimate_has_no_internal_byte_cap() {
    let bytes = funded_bytes(17 << 20, 1, 3)
        .expect("checked result sizing must defer limits to the configured pool");
    assert!(bytes > 16 << 20);
    assert!(funded_bytes(usize::MAX, 1, 1).is_none());
    assert!(funded_bytes(1, usize::MAX, 2).is_none());
}
