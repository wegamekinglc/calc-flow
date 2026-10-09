use super::*;

#[test]
fn test_join_restore_rejects_v2_delta_group_overlap_atomically() {
    let mut restored = operator();
    let original = snapshot(&restored);
    restored.restore(&original).unwrap();
    assert_restored(&restored);
    let mut invalid = original.clone();
    let mut delta = invalid.segments["left-delta-2"].bytes().to_vec();
    let upsert_length = 8 + 8 + 8 + 32 + 8 + 8 + KEY.len();
    let tombstone = 40 + upsert_length;
    assert_eq!(&delta[tombstone..tombstone + 8], &2_u64.to_le_bytes());
    assert_eq!(&delta[tombstone + 8..tombstone + 16], &95_i64.to_le_bytes());
    delta[tombstone..tombstone + 8].copy_from_slice(&9_u64.to_le_bytes());
    delta[tombstone + 8..tombstone + 16].copy_from_slice(&97_i64.to_le_bytes());
    invalid
        .segments
        .insert("left-delta-2".into(), StateSegment::new(delta));
    invalid.inline_metadata.get_mut("metrics").unwrap()["left"]["retained_bytes"] = 284_u64.into();
    let result = restored.restore(&invalid);
    assert!(
        matches!(result, Err(CalcFlowError::CheckpointMismatch { .. })),
        "one row cannot be introduced and removed within the same V2 index: {result:?}"
    );
    assert_restored(&restored);
    assert_eq!(
        original.inline_metadata["metrics"]["left"]["retained_bytes"],
        283
    );
}
