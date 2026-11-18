use super::ColumnarState;
use crate::operator::asof::{
    AsofJoinSide, AsofLatePolicy, AsofStateLimits, StreamAsofJoinSpec, schema,
};
use crate::{Result, StreamingFailureReason};
use datafusion::arrow::{
    array::{Array, Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
use std::{sync::Arc, time::Duration};

fn input_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::Int64, false),
        Field::new("value", DataType::Utf8, true),
    ]))
}

fn spec(tolerance: u64, rows: u64, bytes: u64) -> StreamAsofJoinSpec {
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["sequence".into()],
            prefix.into(),
        )
        .unwrap()
    };
    StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::from_micros(tolerance),
        AsofStateLimits::new(rows, bytes).unwrap(),
    )
    .unwrap()
}

fn record(rows: &[(i64, i64, i64, Option<&str>)]) -> RecordBatch {
    RecordBatch::try_new(
        input_schema(),
        vec![
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.0))),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(rows.iter().map(|row| row.1))
                    .with_timezone("UTC"),
            ),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|row| row.2))),
            Arc::new(StringArray::from(
                rows.iter().map(|row| row.3).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn state(declaration: &StreamAsofJoinSpec, pool: &Arc<dyn MemoryPool>) -> ColumnarState {
    ColumnarState::new(
        "asof-columnar-test",
        declaration,
        [input_schema(), input_schema()],
        pool,
    )
    .unwrap()
}

fn output_schema(declaration: &StreamAsofJoinSpec) -> SchemaRef {
    schema::output_schema(declaration, &input_schema(), &input_schema()).unwrap()
}

fn admit(state: &mut ColumnarState, side: usize, input: &RecordBatch) -> Result<()> {
    state
        .admit(side, input, None, AsofLatePolicy::Error, &mut || Ok(()))
        .map(|_| ())
}

fn with_next_run(
    segments: &mut std::collections::BTreeMap<String, crate::StateSegment>,
    next_run: u64,
) {
    let id = "asof-columnar-v1-control";
    let mut control = serde_json::from_slice::<serde_json::Value>(segments[id].bytes()).unwrap();
    control["next_run"] = serde_json::json!(next_run);
    segments.insert(
        id.into(),
        crate::StateSegment::new(serde_json::to_vec(&control).unwrap()),
    );
}

fn keys_only_snapshot(
    declaration: &StreamAsofJoinSpec,
) -> std::collections::BTreeMap<String, crate::StateSegment> {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut state = state(declaration, &pool);
    admit(&mut state, 1, &record(&[(1, 5, 1, Some("old"))])).unwrap();
    state.evict(super::Progress {
        left_ended: true,
        right_ended: true,
        ..super::Progress::default()
    });
    assert_eq!(state.state_rows(), 0);
    state.capture(1 << 20).unwrap()
}

#[test]
fn test_columnar_restore_rejects_exhausted_run_id() {
    let declaration = spec(0, 100, 1 << 20);
    let mut segments = keys_only_snapshot(&declaration);
    with_next_run(&mut segments, u64::MAX);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    assert!(
        ColumnarState::restore(
            "asof-columnar-test",
            &declaration,
            &[input_schema(), input_schema()],
            &pool,
            &segments,
        )
        .is_err()
    );
}

#[test]
fn test_columnar_admission_reports_exhausted_run_id_without_partial_run() {
    let declaration = spec(0, 100, 1 << 20);
    let mut segments = keys_only_snapshot(&declaration);
    with_next_run(&mut segments, u64::MAX - 1);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut restored = ColumnarState::restore(
        "asof-columnar-test",
        &declaration,
        &[input_schema(), input_schema()],
        &pool,
        &segments,
    )
    .unwrap();
    admit(&mut restored, 1, &record(&[(1, 6, 1, Some("last"))])).unwrap();
    let error = admit(&mut restored, 1, &record(&[(1, 7, 1, Some("overflow"))]))
        .expect_err("exhausted run ID");
    assert!(matches!(
        error,
        crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofCounterOverflow,
            ..
        }
    ));
    assert_eq!(restored.state_rows(), 1);
}

#[test]
fn test_columnar_restore_rejects_forged_run_charges() {
    let declaration = spec(0, 100, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut original = state(&declaration, &pool);
    admit(&mut original, 1, &record(&[(1, 5, 1, Some("payload"))])).unwrap();
    let segments = original.capture(1 << 20).unwrap();
    for field in ["payload_charge", "identity_charge"] {
        let mut damaged = segments.clone();
        let id = "asof-columnar-v1-control";
        let mut control = serde_json::from_slice::<serde_json::Value>(damaged[id].bytes()).unwrap();
        let current = control["runs"][0][field].as_u64().unwrap();
        control["runs"][0][field] = serde_json::json!(current - 1);
        damaged.insert(
            id.into(),
            crate::StateSegment::new(serde_json::to_vec(&control).unwrap()),
        );
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        assert!(
            ColumnarState::restore(
                "asof-columnar-test",
                &declaration,
                &[input_schema(), input_schema()],
                &pool,
                &damaged,
            )
            .is_err(),
            "forged {field}"
        );
    }
}

#[test]
fn test_candidate_is_greatest_time_and_sequence_inside_inclusive_tolerance() {
    let declaration = spec(5, 100, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut state = state(&declaration, &pool);
    admit(
        &mut state,
        1,
        &record(&[(1, 10, 1, Some("10a")), (1, 5, 2, Some("5b"))]),
    )
    .unwrap();
    admit(
        &mut state,
        1,
        &record(&[
            (1, 5, 1, Some("5a")),
            (1, 10, 2, Some("10b")),
            (1, 20, 1, Some("20")),
        ]),
    )
    .unwrap();
    admit(
        &mut state,
        0,
        &record(&[
            (1, 4, 1, Some("before")),
            (1, 9, 1, Some("ties")),
            (1, 10, 1, Some("equal-time")),
            (1, 15, 1, Some("lower-bound")),
            (1, 16, 1, Some("expired")),
        ]),
    )
    .unwrap();

    let selection = state.select(Some(100), false, 100).unwrap();
    let output = state
        .materialize(&selection, &output_schema(&declaration), None)
        .unwrap();
    let right_times = output
        .column(5)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    let right_sequences = output
        .column(6)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let right_values = output
        .column(7)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let expected = [
        None,
        Some((5, 2, "5b")),
        Some((10, 2, "10b")),
        Some((10, 2, "10b")),
        None,
    ];
    for (row, expected) in expected.into_iter().enumerate() {
        assert_eq!(right_times.is_valid(row), expected.is_some());
        if let Some((time, sequence, value)) = expected {
            assert_eq!(right_times.value(row), time);
            assert_eq!(right_sequences.value(row), sequence);
            assert_eq!(right_values.value(row), value);
        }
    }
}

#[test]
fn test_pending_output_uses_typed_key_order_across_runs_not_group_birth_order() {
    let declaration = spec(0, 100, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut state = state(&declaration, &pool);
    admit(
        &mut state,
        0,
        &record(&[
            (2, 5, 0, Some("key-2")),
            (1, 5, 0, Some("key-1-time-5")),
            (1, 4, 2, Some("sequence-2")),
        ]),
    )
    .unwrap();
    admit(&mut state, 0, &record(&[(1, 4, 1, Some("sequence-1"))])).unwrap();

    let selection = state.select(None, true, 100).unwrap();
    let output = state
        .materialize(&selection, &output_schema(&declaration), None)
        .unwrap();
    let keys = output
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let times = output
        .column(1)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    let sequences = output
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let values = output
        .column(3)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(keys.values().as_ref(), [1, 1, 1, 2]);
    assert_eq!(times.values().as_ref(), [4, 4, 5, 5]);
    assert_eq!(sequences.values().as_ref(), [1, 2, 0, 0]);
    assert_eq!(
        values.iter().collect::<Vec<_>>(),
        vec![
            Some("sequence-1"),
            Some("sequence-2"),
            Some("key-1-time-5"),
            Some("key-2"),
        ]
    );
    for column in 4..output.num_columns() {
        assert_eq!(output.column(column).null_count(), 4);
    }
}

#[test]
fn test_strict_finality_commits_only_the_closed_prefix_and_eof_drains() {
    let declaration = spec(0, 100, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut state = state(&declaration, &pool);
    admit(
        &mut state,
        0,
        &record(&[(1, 10, 1, Some("open-at-10")), (1, 11, 1, Some("eleven"))]),
    )
    .unwrap();

    assert!(state.select(Some(10), false, 100).unwrap().is_empty());
    let closed = state.select(Some(11), false, 100).unwrap();
    assert_eq!(closed.len(), 1);
    state.commit(&closed).unwrap();
    assert!(state.select(Some(11), false, 100).unwrap().is_empty());

    let ended = state.select(None, true, 100).unwrap();
    assert_eq!(ended.len(), 1);
    let output = state
        .materialize(&ended, &output_schema(&declaration), None)
        .unwrap();
    let times = output
        .column(1)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    assert_eq!(times.value(0), 11);
    state.commit(&ended).unwrap();
    assert_eq!(state.state_rows(), 0);
}

#[test]
fn test_admission_after_partial_commit_reuses_reclaimed_capacity() {
    let declaration = spec(0, 2_000, 16 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(16 << 20));
    let mut state = state(&declaration, &pool);
    let rows = (0..1_024)
        .map(|row| (1, row, row, Some("pending")))
        .collect::<Vec<_>>();
    admit(&mut state, 0, &record(&rows)).unwrap();
    let prefix = state.select(None, true, 1_000).unwrap();
    state.commit(&prefix).unwrap();
    admit(&mut state, 0, &record(&[(1, 2_000, 2_000, Some("next"))])).unwrap();
    assert_eq!(state.pending_left_rows(), 25);
}

#[test]
fn test_late_drop_excludes_rows_before_key_birth_and_retains_watermark_equality() {
    let declaration = spec(0, 100, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut state = state(&declaration, &pool);
    let admitted = state
        .admit(
            1,
            &record(&[(2, 4, 1, Some("late")), (1, 5, 1, Some("equal"))]),
            Some(5),
            AsofLatePolicy::Drop,
            &mut || Ok(()),
        )
        .unwrap();
    assert_eq!(
        admitted,
        super::AdmissionSummary {
            accepted: 1,
            late: 1
        }
    );
    assert_eq!(state.keys.len(), 1);
    admit(&mut state, 0, &record(&[(1, 5, 1, Some("left"))])).unwrap();
    let selection = state.select(Some(6), false, 10).unwrap();
    assert_eq!(selection.matched(), 1);
}

#[test]
fn test_duplicate_and_row_limit_refusal_roll_back_new_key_births() {
    let declaration = spec(0, 1, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut state = state(&declaration, &pool);
    admit(&mut state, 1, &record(&[(1, 5, 1, Some("resident"))])).unwrap();
    let error = admit(&mut state, 1, &record(&[(1, 5, 1, Some("duplicate"))]))
        .expect_err("resident duplicate");
    assert!(matches!(
        error,
        crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofDuplicateIdentity,
            ..
        }
    ));
    assert_eq!(state.state_rows(), 1);
    assert_eq!(state.keys.len(), 1);

    let error =
        admit(&mut state, 1, &record(&[(2, 6, 1, Some("new-key"))])).expect_err("row limit");
    assert!(matches!(
        error,
        crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofStateLimitExceeded,
            ..
        }
    ));
    assert_eq!(state.state_rows(), 1);
    assert_eq!(state.keys.len(), 1);
}

#[test]
fn test_duplicate_inside_one_batch_installs_no_partial_run() {
    let declaration = spec(0, 100, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut state = state(&declaration, &pool);
    let error = admit(
        &mut state,
        0,
        &record(&[(1, 5, 1, Some("first")), (1, 5, 1, Some("duplicate"))]),
    )
    .expect_err("batch duplicate");
    assert!(matches!(
        error,
        crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofDuplicateIdentity,
            ..
        }
    ));
    assert_eq!(state.state_rows(), 0);
    assert_eq!(state.keys.len(), 0);
}

#[test]
fn test_eviction_preserves_equality_then_retires_payload_before_identity() {
    let declaration = spec(5, 100, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut state = state(&declaration, &pool);
    admit(&mut state, 1, &record(&[(1, 5, 1, Some("right"))])).unwrap();
    admit(&mut state, 0, &record(&[(1, 10, 1, Some("left"))])).unwrap();

    let equality = state.evict(super::Progress {
        left_watermark: Some(10),
        right_watermark: Some(5),
        ..super::Progress::default()
    });
    assert_eq!(equality, super::EvictionSummary::default());
    let selection = state.select(Some(11), false, 10).unwrap();
    assert_eq!(selection.matched(), 1);
    state.commit(&selection).unwrap();

    let payload = state.evict(super::Progress {
        left_watermark: Some(11),
        right_watermark: Some(5),
        ..super::Progress::default()
    });
    assert_eq!(
        payload,
        super::EvictionSummary {
            payloads: 1,
            identities: 0,
        }
    );
    assert_eq!(state.state_rows(), 1);
    assert_eq!(state.right_payload_rows(), 0);
    assert_eq!(state.identity_only_rows(), 1);

    let identity = state.evict(super::Progress {
        left_watermark: Some(11),
        right_watermark: Some(6),
        ..super::Progress::default()
    });
    assert_eq!(
        identity,
        super::EvictionSummary {
            payloads: 0,
            identities: 1,
        }
    );
    assert_eq!(state.state_rows(), 0);
}

#[test]
fn test_eviction_drops_dominated_prefix_but_keeps_newest_tie_at_threshold() {
    let declaration = spec(60, 100, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut state = state(&declaration, &pool);
    admit(
        &mut state,
        1,
        &record(&[
            (1, 0, 1, Some("old")),
            (1, 10, 1, Some("lower-sequence")),
            (1, 10, 2, Some("winner")),
        ]),
    )
    .unwrap();
    admit(&mut state, 0, &record(&[(1, 10, 1, Some("left"))])).unwrap();
    let evicted = state.evict(super::Progress {
        left_watermark: Some(10),
        ..super::Progress::default()
    });
    assert_eq!(
        evicted,
        super::EvictionSummary {
            payloads: 2,
            identities: 0,
        }
    );
    assert_eq!(state.right_payload_rows(), 1);
    let selection = state.select(Some(11), false, 10).unwrap();
    let output = state
        .materialize(&selection, &output_schema(&declaration), None)
        .unwrap();
    let values = output
        .column(7)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(values.value(0), "winner");
}

#[test]
fn test_right_eof_drops_duplicate_identity_but_not_useful_payload() {
    let declaration = spec(5, 100, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut state = state(&declaration, &pool);
    admit(&mut state, 1, &record(&[(1, 10, 1, Some("right"))])).unwrap();
    let evicted = state.evict(super::Progress {
        right_ended: true,
        ..super::Progress::default()
    });
    assert_eq!(
        evicted,
        super::EvictionSummary {
            payloads: 0,
            identities: 1,
        }
    );
    assert_eq!(state.state_rows(), 1);
    assert_eq!(state.right_payload_rows(), 1);
    assert_eq!(state.identity_only_rows(), 0);
}

#[test]
fn test_composite_sequence_threshold_uses_aggregate_extent() {
    let half = i32::MAX as usize / 2;
    assert_eq!(super::run::sequence_tuple_copy_count([half, half]), 1);
    assert_eq!(
        super::run::sequence_tuple_copy_count([half + 1, half + 1]),
        2
    );
}

#[test]
fn test_large_sequence_branch_reserves_an_additional_encoding_copy() {
    assert_eq!(super::run::sequence_copy_count(i32::MAX as usize), 1);
    assert_eq!(super::run::sequence_copy_count(i32::MAX as usize + 1), 2);
}

#[test]
fn test_string_sequence_encoding_is_reserved_before_row_conversion() {
    let declaration = spec(0, 100, 80_000);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(80_000));
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::Utf8, false),
        Field::new("value", DataType::Utf8, true),
    ]));
    let mut state = ColumnarState::new(
        "asof-columnar-test",
        &declaration,
        [schema.clone(), schema.clone()],
        &pool,
    )
    .unwrap();
    let input = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(TimestampMicrosecondArray::from(vec![5]).with_timezone("UTC")),
            Arc::new(StringArray::from(vec!["s".repeat(48_000)])),
            Arc::new(StringArray::from(vec!["payload"])),
        ],
    )
    .unwrap();
    let allocations = allocation_counter::measure(|| {
        let error = state
            .admit(1, &input, None, AsofLatePolicy::Error, &mut || Ok(()))
            .expect_err("sequence encoding exceeds workspace");
        assert!(matches!(
            error,
            crate::CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
                ..
            }
        ));
    });
    assert!(
        allocations.bytes_max < 80_000,
        "sequence encoded before workspace refusal: {allocations:?}"
    );
}

#[test]
fn test_oversized_resident_duplicate_precedes_workspace_refusal() {
    let declaration = spec(0, 100, 40_000);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(40_000));
    let mut state = state(&declaration, &pool);
    admit(&mut state, 1, &record(&[(1, 5, 1, Some("resident"))])).unwrap();
    let large = "x".repeat(48_000);
    let error =
        admit(&mut state, 1, &record(&[(1, 5, 1, Some(&large))])).expect_err("resident duplicate");
    assert!(matches!(
        error,
        crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofDuplicateIdentity,
            ..
        }
    ));
    assert_eq!(state.state_rows(), 1);
}

#[test]
fn test_oversized_admission_is_rejected_before_payload_copy() {
    let declaration = spec(0, 100, 40_000);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(40_000));
    let mut state = state(&declaration, &pool);
    let large = "x".repeat(48_000);
    let input = record(&[(1, 5, 1, Some(&large))]);
    let allocations = allocation_counter::measure(|| {
        let error = admit(&mut state, 1, &input).expect_err("oversized input");
        assert!(matches!(
            error,
            crate::CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
                ..
            }
        ));
    });
    assert!(
        allocations.bytes_max < 40_000,
        "payload was allocated before refusal: {allocations:?}"
    );
}

#[test]
fn test_sliced_payload_does_not_retain_large_backing_allocation() {
    let declaration = spec(10, 100, 16 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(16 << 20));
    let mut state = state(&declaration, &pool);
    let input = record(&[
        (1, 99, 1, Some(&"x".repeat(4 * 1024 * 1024))),
        (1, 100, 2, Some("z")),
        (1, 101, 3, Some(&"x".repeat(4 * 1024 * 1024))),
    ]);
    admit(&mut state, 1, &input.slice(1, 1)).unwrap();
    assert!(state.persistent_bytes() < 16_384);
}

#[test]
fn test_one_byte_short_state_pool_rejects_without_partial_key_or_run() {
    let declaration = spec(0, 100, 1 << 20);
    let input = record(&[(1, 5, 1, Some("payload"))]);
    let reference: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut complete = state(&declaration, &reference);
    admit(&mut complete, 1, &input).unwrap();
    let required = reference.reserved();
    assert!(required > 1);

    let tight: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(required - 1));
    let mut state = state(&declaration, &tight);
    let error = admit(&mut state, 1, &input).expect_err("one-byte-short state");
    assert!(matches!(
        error,
        crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofStateLimitExceeded,
            ..
        }
    ));
    assert_eq!(state.state_rows(), 0);
    assert_eq!(state.keys.len(), 0);
}

#[test]
fn test_projection_interleaves_multiple_owners_and_explicit_unmatched_validity() {
    let declaration = spec(0, 100, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut state = state(&declaration, &pool);
    admit(&mut state, 1, &record(&[(1, 5, 1, Some("right-5"))])).unwrap();
    admit(&mut state, 1, &record(&[(1, 6, 2, Some("right-6"))])).unwrap();
    admit(&mut state, 0, &record(&[(1, 5, 1, Some("left-5"))])).unwrap();
    admit(
        &mut state,
        0,
        &record(&[(1, 6, 1, Some("left-6")), (1, 7, 1, Some("left-7"))]),
    )
    .unwrap();
    let selection = state.select(None, true, 10).unwrap();
    let projected_schema = Arc::new(Schema::new(vec![
        Field::new("right_value", DataType::Utf8, true),
        Field::new("left_value", DataType::Utf8, true),
        Field::new("right_sequence", DataType::Int64, true),
    ]));
    let output = state
        .materialize(
            &selection,
            &projected_schema,
            Some(&[(1, 3), (0, 3), (1, 2)]),
        )
        .unwrap();
    let right = output
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let left = output
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    let sequence = output
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        right.iter().collect::<Vec<_>>(),
        vec![Some("right-5"), Some("right-6"), None]
    );
    assert_eq!(
        left.iter().collect::<Vec<_>>(),
        vec![Some("left-5"), Some("left-6"), Some("left-7")]
    );
    assert_eq!(
        sequence.iter().collect::<Vec<_>>(),
        vec![Some(1), Some(2), None]
    );
}
