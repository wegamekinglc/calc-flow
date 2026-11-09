use super::*;

#[test]
fn test_json_payload_budget_matches_real_serializer_without_copying_values() {
    for value in [
        serde_json::json!(null),
        serde_json::json!({"valid_count":u64::MAX,"value":{"kind":"SignedTotal","value":vec![255_u8;16]},"has_infinity":false}),
        serde_json::json!({"entity":"escaped\"\n\\".repeat(64)}),
    ] {
        let mut plan = None;
        let scan = allocation_counter::measure(|| {
            plan = Some(json(&value, &mut || Ok(())).unwrap());
        });
        assert_eq!(scan.count_total, 0);
        let plan = plan.unwrap();
        let mut payload = None;
        let allocation = allocation_counter::measure(|| {
            payload = Some(serde_json::to_vec(&value).unwrap());
        });
        let payload = payload.unwrap();
        assert_eq!(plan.length, payload.len());
        assert_eq!(plan.capacity, payload.capacity());
        assert!(plan.construction_peak >= allocation.bytes_max);
    }
}
#[test]
fn test_json_payload_prepass_preserves_cancellation() {
    let error = json(&serde_json::json!(1), &mut || {
        Err(CalcFlowError::Cancelled {
            run_id: "payload".into(),
        })
    })
    .err()
    .unwrap();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
}
