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

fn select(
    state: &ColumnarState,
    frontier: Option<i64>,
    ended: bool,
    limit: usize,
) -> super::Selection {
    state.select_prefix(state.finalizable_len(frontier, ended, limit))
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
    // Older v11 checkpoints retained a key dictionary even after all runs retired.
    state
        .restore_keys(&record(&[(1, 5, 1, Some("old"))]).project(&[0]).unwrap())
        .unwrap();
    state.finish_restore(1);
    state.capture(1 << 20).unwrap()
}

#[test]
fn test_visible_latest_right_row_skips_binary_search_only_when_safe() {
    let _ = super::state::take_latest_binary_searches();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    {
        let declaration = spec(5, 100, 1 << 20);
        let mut state = state(&declaration, &pool);
        admit(
            &mut state,
            1,
            &record(&[
                (1, 1, 10, Some("first")),
                (1, 3, 20, Some("middle")),
                (1, 5, 30, Some("latest")),
            ]),
        )
        .unwrap();
        admit(
            &mut state,
            0,
            &record(&[(1, 5, 100, Some("left")), (2, 5, 200, Some("empty"))]),
        )
        .unwrap();
        let selection = select(&state, None, true, 10);
        assert_eq!(selection.matched(), 1);
        assert_eq!(super::state::take_latest_binary_searches(), 0);
    }
    {
        let declaration = spec(0, 100, 1 << 20);
        let mut state = state(&declaration, &pool);
        admit(&mut state, 1, &record(&[(1, 5, 1, Some("future"))])).unwrap();
        admit(&mut state, 0, &record(&[(1, 4, 1, Some("left"))])).unwrap();
        assert_eq!(select(&state, None, true, 10).matched(), 0);
        assert_eq!(super::state::take_latest_binary_searches(), 1);
    }
    {
        let declaration = spec(5, 100, 1 << 20);
        let mut state = state(&declaration, &pool);
        admit(&mut state, 1, &record(&[(1, 5, 1, Some("old"))])).unwrap();
        admit(&mut state, 0, &record(&[(1, 11, 1, Some("left"))])).unwrap();
        assert_eq!(select(&state, None, true, 10).matched(), 0);
        assert_eq!(super::state::take_latest_binary_searches(), 0);
    }
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
fn test_ordered_historical_probes_bound_dense_steps_and_sparse_jumps() {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(16 << 20));
    let declaration = spec(1, 12_000, 16 << 20);
    let mut state = state(&declaration, &pool);
    let rows = (0..2048)
        .flat_map(|time| (0..4).map(move |key| (key, time * 2, time, None)))
        .collect::<Vec<_>>();
    admit(&mut state, 1, &record(&rows)).unwrap();
    let probes = (0..128)
        .chain(2000..2016)
        .flat_map(|time| (0..4).map(move |key| (key, time * 2 + 1, time, None)))
        .collect::<Vec<_>>();
    admit(&mut state, 0, &record(&probes)).unwrap();

    super::state::CANDIDATE_TIME_READS.set(0);
    let selection = select(&state, None, true, probes.len());
    let reads = super::state::CANDIDATE_TIME_READS.replace(0);
    let output = state
        .materialize(&selection, &output_schema(&declaration), None)
        .unwrap();
    assert_eq!(selection.matched(), probes.len());
    assert_eq!(output.column(0), output.column(4));
    assert_eq!(output.column(2), output.column(6));
    // Four adjacent reads per probe, plus one logarithmic jump per key.
    assert!(
        reads <= 4 * probes.len() + 4 * 16,
        "{reads} timestamp reads"
    );

    #[cfg(feature = "benchmark-internals")]
    {
        state.set_legacy_latest_lookup_for_benchmark(true);
        let _ = super::state::take_latest_binary_searches();
        let legacy = select(&state, None, true, probes.len());
        assert_eq!(super::state::take_latest_binary_searches(), probes.len());
        assert_eq!(legacy.right, selection.right);
        assert_eq!(legacy.right_validity, selection.right_validity);
    }
}

#[test]
fn test_candidate_reuses_visible_tail_time_at_inclusive_tolerance() {
    let declaration = spec(1, 100, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let max = i64::MAX;
    let reads = [(max - 3, 2), (max, 3)].map(|(left_time, expected_index)| {
        let mut state = state(&declaration, &pool);
        let right = record(&[
            (1, i64::MIN, 0, Some("first")),
            (1, max - 4, 1, Some("older tie")),
            (1, max - 4, 2, Some("newer tie")),
            (1, max - 1, 3, None),
        ]);
        admit(&mut state, 1, &right).unwrap();
        admit(&mut state, 0, &record(&[(1, left_time, 0, None)])).unwrap();
        super::state::CANDIDATE_TIME_READS.set(0);
        let selection = select(&state, None, true, 1);
        let reads = super::state::CANDIDATE_TIME_READS.replace(0);
        assert_eq!(selection.matched(), 1);
        let output = state
            .materialize(&selection, &output_schema(&declaration), None)
            .unwrap();
        let expected_right = right.slice(expected_index, 1);
        assert_eq!(&output.columns()[4..], expected_right.columns());
        reads
    });
    let [historical, latest] = reads;
    eprintln!("ASOF candidate right timestamp reads: historical={historical}, latest={latest}");
    assert!(historical <= 5);
    assert_eq!(latest, 1);
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

    let selection = select(&state, Some(100), false, 100);
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
            // Each key appends in order; the combined left input does not.
            (2, 5, 0, Some("key-2")),
            (1, 4, 2, Some("sequence-2")),
            (1, 5, 0, Some("key-1-time-5")),
        ]),
    )
    .unwrap();
    admit(&mut state, 0, &record(&[(1, 4, 1, Some("sequence-1"))])).unwrap();

    let selection = select(&state, None, true, 100);
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
fn test_ready_prefix_respects_row_limits_and_timestamp_extremes() {
    let declaration = spec(0, 100, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut state = state(&declaration, &pool);
    assert_eq!(state.finalizable_len(Some(0), false, 8), 0);
    let input = [
        (1, i64::MAX, 0, None),
        (1, 0, 2, None),
        (1, 0, 1, None),
        (1, i64::MIN, 0, None),
    ];
    admit(&mut state, 0, &record(&input)).unwrap();
    super::take_finality_probes();
    assert_eq!(state.finalizable_len(None, false, 4), 0);
    assert_eq!(state.finalizable_len(Some(i64::MIN), true, 2), 2);
    assert_eq!(state.finalizable_len(None, true, usize::MAX), 4);
    assert_eq!(state.finalizable_len(Some(i64::MAX), false, 0), 0);
    assert_eq!(super::take_finality_probes(), 0);
    assert_eq!(state.finalizable_len(Some(i64::MIN), false, 4), 0);
    assert_eq!(state.finalizable_len(Some(0), false, 4), 1);
    assert_eq!(state.finalizable_len(Some(i64::MAX), false, usize::MAX), 3);
    assert_eq!(state.finalizable_len(Some(1), false, 1), 1);
    assert_eq!(state.finalizable_len(Some(1), false, 2), 2);
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

    assert!(select(&state, Some(10), false, 100).is_empty());
    let closed = select(&state, Some(11), false, 100);
    assert_eq!(closed.len(), 1);
    state.commit(&closed).unwrap();
    assert!(select(&state, Some(11), false, 100).is_empty());

    let ended = select(&state, None, true, 100);
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
fn test_fresh_charge_skips_dense_tight_group_columns_and_counts_sparse_input_once() {
    let declaration = spec(0, 8_192, 16 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(16 << 20));
    let mut state = state(&declaration, &pool);
    let batch = |start| {
        record(
            &(start..start + 1_024)
                .map(|row| (row % 64, row / 64, row, None))
                .collect::<Vec<_>>(),
        )
    };
    let mut work = |side, input: &RecordBatch| {
        super::state::PROJECTED_GROUP_VISITS.set(0);
        super::state::PROJECTED_COUNT_UPDATES.set(0);
        admit(&mut state, side, input).unwrap();
        (
            super::state::PROJECTED_GROUP_VISITS.replace(0),
            super::state::PROJECTED_COUNT_UPDATES.replace(0),
        )
    };
    for start in [0, 1_024] {
        let input = batch(start);
        for side in [0, 1] {
            assert_eq!(
                work(side, &input),
                (0, 0),
                "dense fresh append on side {side}"
            );
        }
    }
    let sparse = record(&[(12, 100, 3_000, None)]);
    for side in [0, 1] {
        assert_eq!(work(side, &sparse), (1, 1), "sparse append on side {side}");
    }
}

#[test]
fn test_fresh_projection_reclaims_only_the_touched_restored_right_slots() {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 22));
    let mut source = state(&spec(0, 128, 1 << 20), &pool);
    admit(&mut source, 1, &record(&[(1, 1, 1, None), (2, 1, 2, None)])).unwrap();
    let mut capacities = source.index_capacities();
    let slot = u64::try_from(size_of::<super::RowLocator>()).unwrap();
    capacities.right_payloads[0] += 3 * slot;
    capacities.right_identities[0] += slot;
    capacities.right_identities[1] += 5 * slot;
    source.restore_index_capacities(&capacities).unwrap();
    let segments = source.capture(1 << 20).unwrap();
    let restore = |limit| {
        ColumnarState::restore(
            "asof-columnar-test",
            &spec(0, 128, limit),
            &[input_schema(), input_schema()],
            &pool,
            &segments,
        )
        .unwrap()
    };
    let input = record(
        &(2..10)
            .map(|time| (1, time, time, None))
            .collect::<Vec<_>>(),
    );
    let mut reference = restore(1 << 20);
    admit(&mut reference, 1, &input).unwrap();
    let retained = reference.persistent_bytes();
    let after = reference.index_capacities();
    assert_eq!(after.right_payloads, [9 * slot, slot]);
    assert_eq!(after.right_identities, [9 * slot, 6 * slot]);
    let limit =
        u64::try_from(retained).unwrap() + 2 * u64::try_from(input.num_rows()).unwrap() * slot;
    let mut exact = restore(limit);
    admit(&mut exact, 1, &input).unwrap();
    assert_eq!(exact.persistent_bytes(), retained);
    let mut short = restore(limit - 1);
    assert!(matches!(
        admit(&mut short, 1, &input),
        Err(crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofStateLimitExceeded,
            ..
        })
    ));
    assert_eq!(short.persistent_bytes(), source.persistent_bytes());
}

#[test]
fn test_restore_charge_counts_mixed_right_liveness_at_the_exact_limit() {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let right = record(&[
        (1, 18, 0, None),
        (1, 19, 1, None),
        (1, 20, 2, None),
        (2, 16, 0, None),
        (3, 25, 0, None),
    ]);
    let mixed_run = || {
        let mut source = state(&spec(10, 100, 1 << 20), &pool);
        admit(&mut source, 1, &right).unwrap();
        source.evict(super::Progress {
            left_watermark: Some(20),
            right_watermark: Some(17),
            ..super::Progress::default()
        });
        source.right_runs.remove(&0).unwrap()
    };
    let keys = record(&[(1, 0, 0, None), (2, 0, 0, None), (3, 0, 0, None)])
        .project(&[0])
        .unwrap();
    let empty_index = |limit| {
        let mut target = state(&spec(10, 100, limit), &pool);
        target.restore_keys(&keys).unwrap();
        target
    };
    let run = mixed_run();
    let key_bytes = empty_index(1 << 20).persistent_bytes();
    // Three payload and four identity locators each need one index and one staging slot.
    let bytes = key_bytes
        + run.charge()
        + size_of::<super::run::Run>()
        + 2 * (3 + 4) * size_of::<super::RowLocator>();
    let limit = u64::try_from(bytes).unwrap();
    let mut short = empty_index(limit - 1);
    assert!(matches!(
        short.restore_run(1, 0, run),
        Err(crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofStateLimitExceeded,
            ..
        })
    ));
    assert_eq!(short.persistent_bytes(), key_bytes);
    assert_eq!(short.state_rows(), 0);

    let mut exact = empty_index(limit);
    exact.restore_run(1, 0, mixed_run()).unwrap();
    assert_eq!(exact.state_rows(), 5);
    assert_eq!(exact.identity_only_rows(), 2);
    assert_eq!(exact.right_payload_rows(), 3);
    assert_eq!(
        exact.persistent_bytes(),
        bytes - 7 * size_of::<super::RowLocator>()
    );
}

#[test]
fn test_sparse_fresh_indexes_retain_only_the_charged_locator_slots() {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let rows = (0..64).map(|key| (key, key, 0, None)).collect::<Vec<_>>();
    let input = record(&rows);
    let slot = u64::try_from(size_of::<super::RowLocator>()).unwrap();
    for side in [0, 1] {
        let mut admitted = state(&spec(0, 100, 1 << 20), &pool);
        admit(&mut admitted, side, &input).unwrap();
        let capacities = admitted.index_capacities();
        if side == 0 {
            assert_eq!(capacities.left, vec![slot; rows.len()]);
        } else {
            assert_eq!(capacities.right_payloads, vec![slot; rows.len()]);
            assert_eq!(capacities.right_identities, vec![slot; rows.len()]);
        }

        let retained = admitted.persistent_bytes();
        let exact =
            u64::try_from(retained).unwrap() + 2 * u64::try_from(rows.len()).unwrap() * slot;
        let mut short = state(&spec(0, 100, exact - 1), &pool);
        assert!(matches!(
            admit(&mut short, side, &input),
            Err(crate::CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofStateLimitExceeded,
                ..
            })
        ));
        assert_eq!(short.persistent_bytes(), 0);
        assert_eq!(short.state_rows(), 0);
        let mut tight = state(&spec(0, 100, exact), &pool);
        admit(&mut tight, side, &input).unwrap();
        assert_eq!(tight.persistent_bytes(), retained);
    }
}

#[test]
fn test_partial_commit_does_not_grow_singleton_locator_columns() {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut state = state(&spec(0, 100, 1 << 20), &pool);
    admit(&mut state, 0, &record(&[(1, 7, 0, None), (2, 8, 0, None)])).unwrap();
    let slot = u64::try_from(size_of::<super::RowLocator>()).unwrap();
    let mut capacities = state.index_capacities();
    capacities.left.fill(slot);
    state.restore_index_capacities(&capacities).unwrap();
    let prefix = select(&state, None, true, 1);
    state.commit(&prefix).unwrap();
    assert_eq!(state.index_capacities().left, [0, slot]);
}

#[test]
fn test_restore_preserves_explicit_spare_capacity_from_older_checkpoints() {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut source = state(&spec(0, 100, 1 << 20), &pool);
    admit(&mut source, 1, &record(&[(1, 7, 0, None)])).unwrap();
    let mut capacities = source.index_capacities();
    capacities.right_identities[0] = u64::try_from(64 * size_of::<super::RowLocator>()).unwrap();
    source.restore_index_capacities(&capacities).unwrap();
    let bytes = u64::try_from(source.persistent_bytes()).unwrap();
    let segments = source.capture(1 << 20).unwrap();
    let restored = ColumnarState::restore(
        "asof-columnar-test",
        &spec(0, 100, bytes),
        &[input_schema(), input_schema()],
        &pool,
        &segments,
    )
    .unwrap();
    assert_eq!(u64::try_from(restored.persistent_bytes()).unwrap(), bytes);
    assert_eq!(restored.capture(1 << 20).unwrap(), segments);
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
    let prefix = select(&state, None, true, 1_000);
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
    let selection = select(&state, Some(6), false, 10);
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

fn route_string_groups(keys: StringArray) -> (Vec<usize>, usize) {
    use crate::operator::keyed::{hash_calls, reset_hash_calls};

    let rows = keys.len();
    let values = record(
        &(0..rows)
            .map(|row| (0, i64::try_from(row).unwrap(), 0, None))
            .collect::<Vec<_>>(),
    );
    let mut columns = values.columns().to_vec();
    columns[0] = Arc::new(keys);
    let mut fields = input_schema().fields().to_vec();
    fields[0] = Arc::new(Field::new("key", DataType::Utf8, false));
    let schema = Arc::new(Schema::new(fields));
    let input = RecordBatch::try_new(schema.clone(), columns).unwrap();
    let declaration = spec(0, u64::try_from(rows).unwrap(), 1 << 26);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 26));
    let mut state =
        ColumnarState::new("asof", &declaration, [schema.clone(), schema], &pool).unwrap();
    reset_hash_calls();
    admit(&mut state, 0, &input).unwrap();
    let run = state.left_runs.values().next().unwrap();
    (
        run.groups().iter().map(|group| group.index()).collect(),
        hash_calls(),
    )
}

#[test]
fn repeated_string_key_morsels_reuse_only_exact_previous_groups() {
    use crate::operator::keyed::MORSEL_ROWS;

    let rows = 3 * MORSEL_ROWS + 17;
    // The concatenated bytes match even where the string boundaries change.
    let keys = StringArray::from_iter_values((0..rows).map(|row| {
        let pair = if row < 2 * MORSEL_ROWS {
            ["a", "bc"]
        } else {
            ["ab", "c"]
        };
        pair[row % 2]
    }));
    let (groups, hashes) = route_string_groups(keys);
    for (row, group) in groups.into_iter().enumerate() {
        let offset = if row < 2 * MORSEL_ROWS { 0 } else { 2 };
        assert_eq!(group, offset + row % 2);
    }
    assert_eq!(hashes, 2, "only the two distinct key morsels need routing");
}

#[test]
fn wide_repeated_string_keys_keep_chunked_routing() {
    use crate::operator::keyed::{MORSEL_ROWS, VARIABLE_STEP_BYTES};

    let value = "x".repeat(VARIABLE_STEP_BYTES / MORSEL_ROWS + 1);
    let rows = 2 * MORSEL_ROWS;
    let keys = StringArray::from_iter_values(std::iter::repeat_n(value.as_str(), rows));
    let (groups, hashes) = route_string_groups(keys);
    assert_eq!(groups, vec![0; rows]);
    assert_eq!(hashes, 2);
}

#[test]
fn test_ordered_interleaved_identity_admission_is_linear_on_both_sides() {
    const KEYS: i64 = 64;
    const ROWS: usize = 1024;
    let input = |start: usize| {
        record(
            &(start..start + ROWS)
                .map(|i| {
                    let sequence = i64::try_from(i).unwrap();
                    (sequence % KEYS, sequence / KEYS, sequence, Some("p"))
                })
                .collect::<Vec<_>>(),
        )
    };
    let spec = spec(0, 4096, 1 << 25);
    for side in 0..2 {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 25));
        let mut state = state(&spec, &pool);
        admit(&mut state, side, &input(0)).unwrap();
        super::state::ADMISSION_IDENTITY_COMPARISONS.set(0);
        super::state::INDEX_IDENTITY_COMPARISONS.set(0);
        admit(&mut state, side, &input(ROWS)).unwrap();
        let comparisons = super::state::ADMISSION_IDENTITY_COMPARISONS.replace(0);
        assert_eq!(comparisons, ROWS, "one comparison per row for side {side}");
        let index_comparisons = super::state::INDEX_IDENTITY_COMPARISONS.replace(0);
        let columns = if side == 0 { 1 } else { 2 };
        assert!(
            index_comparisons <= usize::try_from(KEYS).unwrap() * columns,
            "ordered installation revisited identities {index_comparisons} times on side {side}"
        );
        assert_eq!(state.state_rows(), 2 * ROWS);
    }
}

#[test]
fn test_per_key_identity_order_accepts_overlap_and_rejects_nonadjacent_duplicates() {
    let declaration = spec(0, 20, 1 << 20);
    for side in 0..2 {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let mut state = state(&declaration, &pool);
        admit(
            &mut state,
            side,
            &record(&[(1, 5, 1, None), (2, 2, 1, None), (1, 5, 3, None)]),
        )
        .unwrap();
        admit(
            &mut state,
            side,
            &record(&[(2, 2, 2, None), (1, 5, 2, None)]),
        )
        .unwrap();
        let error = admit(
            &mut state,
            side,
            &record(&[(2, 4, 1, None), (1, 5, 1, None)]),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            crate::CalcFlowError::OperatorReason {
                reason_code: StreamingFailureReason::AsofDuplicateIdentity,
                ..
            }
        ));
        assert_eq!(state.state_rows(), 5);
    }
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
    let selection = select(&state, Some(11), false, 10);
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
    assert_eq!(state.persistent_bytes(), 0);
    assert_eq!(pool.reserved(), 0);
    let run_frontier = state.next_run;
    assert!(matches!(
        state.capture(1),
        Err(crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofWorkspaceLimitExceeded,
            ..
        })
    ));
    let segments = state.capture(1 << 20).unwrap();
    let mut state = ColumnarState::restore(
        "asof-columnar-test",
        &declaration,
        &[input_schema(), input_schema()],
        &pool,
        &segments,
    )
    .unwrap();
    assert_eq!(state.next_run, run_frontier);
    assert_eq!(state.capture(1 << 20).unwrap(), segments);
    admit(&mut state, 0, &record(&[(2, 20, 1, Some("new"))])).unwrap();
    assert!(state.next_run > run_frontier);
    let selection = select(&state, Some(21), false, 10);
    assert_eq!(selection.matched(), 0);
    state.commit(&selection).unwrap();
    assert_eq!(state.persistent_bytes(), 0);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn fully_expired_right_runs_retire_without_copying_locator_columns() {
    let declaration = spec(5, 8_192, 8 << 20);
    for cleared_identities in [false, true] {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(32 << 20));
        let mut state = state(&declaration, &pool);
        for start in [0, 2_048_i64] {
            let rows = (start..start + 2_048)
                .map(|i| (i % 4, i, i, Some("right")))
                .collect::<Vec<_>>();
            admit(&mut state, 1, &record(&rows)).unwrap();
        }
        admit(&mut state, 0, &record(&[(1, 10_000, 0, Some("pending"))])).unwrap();
        if cleared_identities {
            let retired = state.evict(super::Progress {
                right_watermark: Some(5_000),
                ..super::Progress::default()
            });
            assert_eq!((retired.payloads, retired.identities), (0, 4_096));
        }
        let mut retired = super::EvictionSummary::default();
        let allocations = allocation_counter::measure(|| {
            retired = state.evict(super::Progress {
                left_watermark: Some(5_000),
                right_watermark: Some(5_000),
                ..super::Progress::default()
            });
        });
        assert_eq!(retired.payloads, 4_096);
        assert_eq!(
            retired.identities,
            if cleared_identities { 0 } else { 4_096 }
        );
        assert_eq!((state.state_rows(), state.right_payload_rows()), (1, 0));
        assert_eq!(state.identity_only_rows(), 0);
        let selection = select(&state, Some(10_001), false, 1);
        assert_eq!((selection.len(), selection.matched()), (1, 0));
        state.commit(&selection).unwrap();
        assert_eq!(pool.reserved(), 0);
        assert!(
            allocations.bytes_total <= 2_048,
            "fully retired right index allocated {} bytes",
            allocations.bytes_total
        );
    }
}

#[test]
fn pending_left_and_equal_right_watermark_preserve_last_keys_boundary_match() {
    let declaration = spec(5, 16, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 22));
    let mut state = state(&declaration, &pool);
    admit(
        &mut state,
        1,
        &record(&[(1, 1, 0, None), (2, 20, 0, Some("boundary"))]),
    )
    .unwrap();
    admit(&mut state, 0, &record(&[(2, 25, 0, None)])).unwrap();
    let evicted = state.evict(super::Progress {
        left_watermark: Some(100),
        right_watermark: Some(20),
        ..super::Progress::default()
    });
    assert_eq!((evicted.payloads, evicted.identities), (1, 1));
    assert_eq!((state.state_rows(), state.right_payload_rows()), (2, 1));
    let segments = state.capture(1 << 20).unwrap();
    let restored = ColumnarState::restore(
        "asof-columnar-test",
        &declaration,
        &[input_schema(), input_schema()],
        &pool,
        &segments,
    )
    .unwrap();
    let selection = select(&restored, Some(26), false, 1);
    assert_eq!(selection.matched(), 1);
    let output = restored
        .materialize(&selection, &output_schema(&declaration), None)
        .unwrap();
    let values = output
        .column(7)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(values.value(0), "boundary");
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
    let selection = select(&state, Some(11), false, 10);
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
fn test_single_run_projection_reuses_indices_across_asof_columns() {
    let declaration = spec(2, 4096, 8 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(8 << 20));
    let mut state = state(&declaration, &pool);
    let left_rows = (0..1024)
        .map(|t| (i64::from(t != 1022), t, t, None))
        .collect::<Vec<_>>();
    let right_rows = (4..1024)
        .step_by(4)
        .flat_map(|t| [(1, t, 1, None), (1, t, 2, None)])
        .collect::<Vec<_>>();
    let left = record(&left_rows);
    admit(&mut state, 1, &record(&right_rows)).unwrap();
    admit(&mut state, 0, &left).unwrap();
    let selection = select(&state, None, true, 1024);
    let schema = Arc::new(
        output_schema(&declaration)
            .project(&[0, 1, 2, 4, 5, 6])
            .unwrap(),
    );
    let projection = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)];
    let mut output = None;
    let measured = allocation_counter::measure(|| {
        output = Some(
            state
                .materialize(&selection, &schema, Some(&projection))
                .unwrap(),
        );
    });
    let matches = (0i64..1024)
        .map(|time| (time >= 4 && time != 1022 && time % 4 <= 2).then_some(time - time % 4));
    let mut columns = left.columns()[..3].to_vec();
    columns.push(Arc::new(
        matches
            .clone()
            .map(|m| m.map(|_| 1))
            .collect::<Int64Array>(),
    ));
    columns.push(Arc::new(
        matches
            .clone()
            .collect::<TimestampMicrosecondArray>()
            .with_timezone("UTC"),
    ));
    columns.push(Arc::new(
        matches.map(|m| m.map(|_| 2)).collect::<Int64Array>(),
    ));
    let output = output.unwrap();
    assert_eq!(output, RecordBatch::try_new(schema, columns).unwrap());
    let keys = output
        .column(3)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        keys.values()[0],
        0,
        "a missing match must not copy a right key into the null slot"
    );
    eprintln!("1024-row single-run ASOF projection: {measured:?}");
    assert!(measured.bytes_total <= 80 * 1024, "{measured:?}");
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
    let selection = select(&state, None, true, 10);
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

mod charge;

#[test]
fn test_complete_asof_runs_share_columns_and_preserve_multi_run_order() {
    let declaration = spec(0, 100, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut state = state(&declaration, &pool);
    for side in [0, 1] {
        admit(
            &mut state,
            side,
            &record(&[(1, 0, 0, None), (1, 1, 1, Some("A"))]),
        )
        .unwrap();
        admit(&mut state, side, &record(&[(1, 2, 2, Some("B"))])).unwrap();
    }
    let schema = output_schema(&declaration).project(&[3, 7]).unwrap();
    let schema = Arc::new(schema);
    let columns = [(0, 3), (1, 3)];
    let first = select(&state, None, true, 2);
    let output = state.materialize(&first, &schema, Some(&columns)).unwrap();
    for (side, locators) in [&first.left, &first.right].into_iter().enumerate() {
        let owner = state.run(side, locators[0].run).payload_record().unwrap();
        assert!(Arc::ptr_eq(output.column(side), owner.column(3)));
    }
    let all = select(&state, None, true, 3);
    let output = state.materialize(&all, &schema, Some(&columns)).unwrap();
    let expected = StringArray::from(vec![None, Some("A"), Some("B")]);
    for column in output.columns() {
        assert_eq!(
            column.as_any().downcast_ref::<StringArray>().unwrap(),
            &expected
        );
    }
}

#[test]
fn test_restored_asof_projection_releases_unselected_ipc_payload() {
    let declaration = spec(0, 100, 32 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(32 << 20));
    let mut source = state(&declaration, &pool);
    let wide = "x".repeat(1 << 20);
    admit(&mut source, 0, &record(&[(1, 1, 9, Some(&wide))])).unwrap();
    let snapshot = source.capture(32 << 20).unwrap();
    let mut restored = ColumnarState::restore(
        "asof-columnar-test",
        &declaration,
        &[input_schema(), input_schema()],
        &pool,
        &snapshot,
    )
    .unwrap();
    let schema = Arc::new(output_schema(&declaration).project(&[2]).unwrap());
    let selected = select(&restored, None, true, 1);
    let output = restored
        .materialize(&selected, &schema, Some(&[(0, 2)]))
        .unwrap();
    let input = restored
        .run(0, selected.left[0].run)
        .payload_record()
        .unwrap();
    let allocation = input.column(2).to_data().buffers()[0].data_ptr();
    assert_eq!(
        allocation,
        input.column(3).to_data().buffers()[1].data_ptr()
    );
    assert_ne!(
        allocation,
        output.column(0).to_data().buffers()[0].data_ptr()
    );
    assert_eq!(
        output
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        9
    );

    restored.commit(&selected).unwrap();
    admit(&mut restored, 0, &record(&[(2, 3, 11, None)])).unwrap();
    let fresh = select(&restored, None, true, 1);
    let output = restored
        .materialize(&fresh, &schema, Some(&[(0, 2)]))
        .unwrap();
    let input = restored.run(0, fresh.left[0].run).payload_record().unwrap();
    assert!(Arc::ptr_eq(input.column(2), output.column(0)));
}

#[test]
fn test_final_left_commit_retires_runs_and_preserves_restored_right_state() {
    let declaration = spec(0, 64, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut source = state(&declaration, &pool);
    admit(&mut source, 1, &record(&[(2, 8, 0, Some("right"))])).unwrap();
    for time in [1, 3] {
        admit(
            &mut source,
            0,
            &record(&[(1, time, 1, None), (2, time + 1, 2, None)]),
        )
        .unwrap();
    }
    let first = select(&source, None, true, 1);
    source.commit(&first).unwrap();
    let saved = source.capture(1 << 20).unwrap();
    let mut restored = ColumnarState::restore(
        "asof-columnar-test",
        &declaration,
        &[input_schema(), input_schema()],
        &pool,
        &saved,
    )
    .unwrap();
    let right = restored.index_capacities();
    let remaining = select(&restored, None, true, 3);
    super::state::COMMITTED_ROW_VISITS.set(0);
    restored.commit(&remaining).unwrap();
    assert_eq!(super::state::COMMITTED_ROW_VISITS.get(), 0);
    let after = restored.index_capacities();
    assert_eq!(after.pending, 0);
    assert_eq!(after.left, [0, 0]);
    assert_eq!(after.right_identities, right.right_identities);
    assert_eq!(after.right_payloads, right.right_payloads);
    assert_eq!(
        (restored.pending_left_rows(), restored.right_payload_rows()),
        (0, 1)
    );
    assert!(restored.left_runs.is_empty());
    let saved = restored.capture(1 << 20).unwrap();
    let mut round_trip = ColumnarState::restore(
        "asof-columnar-test",
        &declaration,
        &[input_schema(), input_schema()],
        &pool,
        &saved,
    )
    .unwrap();
    assert_eq!(round_trip.persistent_bytes(), restored.persistent_bytes());
    admit(&mut round_trip, 0, &record(&[(2, 8, 0, None)])).unwrap();
    assert_eq!(select(&round_trip, None, true, 1).matched(), 1);
}

#[test]
fn test_duplicate_left_commit_rejects_stale_selection_without_retiring_state() {
    let declaration = spec(0, 64, 1 << 20);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut state = state(&declaration, &pool);
    admit(&mut state, 1, &record(&[(1, 1, 1, None)])).unwrap();
    admit(&mut state, 0, &record(&[(1, 1, 1, None)])).unwrap();
    let selection = select(&state, None, true, 1);
    state.commit(&selection).unwrap();
    let bytes = state.persistent_bytes();
    assert!(matches!(
        state.commit(&selection),
        Err(crate::CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::AsofProtocolError,
            ..
        })
    ));
    assert_eq!(state.persistent_bytes(), bytes);
    assert_eq!(state.right_payload_rows(), 1);
}
