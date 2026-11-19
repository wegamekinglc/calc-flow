use super::{
    JoinSide, JoinStateLimits, JoinTimeBounds,
    runs::{Incoming, RunState},
};
use crate::operator::keyed::{GroupId, KeySchema, KeyStore};
use crate::{Batch, BatchMetadata, DataFusionConfig, DataFusionRuntime, EdgeBudget, EventTime};
use datafusion::arrow::{
    array::{Array, ArrayRef, DictionaryArray, Int32Array, Int64Array, ListArray, StringArray},
    datatypes::{DataType, Field, Int32Type, Int64Type, Schema},
    record_batch::RecordBatch,
};
use std::{collections::BTreeMap, sync::Arc};

fn record(keys: &[i64], values: &[i64]) -> RecordBatch {
    RecordBatch::try_from_iter(vec![
        ("key", Arc::new(Int64Array::from(keys.to_vec())) as ArrayRef),
        (
            "value",
            Arc::new(Int64Array::from(values.to_vec())) as ArrayRef,
        ),
    ])
    .unwrap()
}

fn incoming(
    record: &RecordBatch,
    physical: Vec<usize>,
    groups: Vec<GroupId>,
    times: &[i64],
    row_ids: &[u64],
    retain: Vec<bool>,
    next_row_id: u64,
) -> Incoming {
    Incoming::new(
        record.clone(),
        physical,
        groups,
        times.iter().copied().map(EventTime::from_micros).collect(),
        row_ids.to_vec(),
        datafusion::arrow::array::BooleanArray::from(retain),
        next_row_id,
    )
    .unwrap()
}

fn state(match_limit: u64) -> RunState {
    RunState::new(
        "run-test",
        JoinStateLimits::new(100, 1_000_000, match_limit).unwrap(),
        JoinTimeBounds::from_micros(0, 0).unwrap(),
    )
    .unwrap()
}

fn output_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("left_value", DataType::Int64, false),
        Field::new("right_value", DataType::Int64, false),
    ]))
}

fn values(batch: &RecordBatch, column: usize) -> Vec<i64> {
    batch
        .column(column)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec()
}

#[test]
fn test_run_probe_orders_locators_by_time_then_physical_row_id() {
    let mut state = state(100);
    let left = record(&[1, 1, 1], &[10, 11, 12]);
    let left = incoming(
        &left,
        vec![0, 1, 2],
        vec![GroupId::new(0).unwrap(); 3],
        &[20, 10, 20],
        &[0, 1, 2],
        vec![true; 3],
        3,
    );
    let seeded = state
        .apply(
            JoinSide::Left,
            left,
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::new(100, 1_000_000).unwrap(),
        )
        .unwrap();
    assert!(seeded.records.is_empty());

    let right = record(&[1, 1], &[100, 101]);
    let right = incoming(
        &right,
        vec![0, 1],
        vec![GroupId::new(0).unwrap(); 2],
        &[20, 10],
        &[0, 1],
        vec![false; 2],
        2,
    );
    let output = state
        .apply(
            JoinSide::Right,
            right,
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::new(100, 1_000_000).unwrap(),
        )
        .unwrap();
    assert_eq!(output.records.len(), 1);
    assert_eq!(values(&output.records[0], 0), [10, 12, 11]);
    assert_eq!(values(&output.records[0], 1), [100, 100, 101]);
    assert_eq!(output.sequences, [0]);
    assert_eq!(state.rows(JoinSide::Left), 3);
    assert_eq!(state.rows(JoinSide::Right), 0);
}

#[test]
fn test_match_limit_rejects_before_run_or_frontier_commit() {
    let mut state = state(1);
    let left = record(&[1, 1, 1], &[10, 11, 12]);
    let left = incoming(
        &left,
        vec![0, 1, 2],
        vec![GroupId::new(0).unwrap(); 3],
        &[20, 10, 20],
        &[0, 1, 2],
        vec![true; 3],
        3,
    );
    state
        .apply(
            JoinSide::Left,
            left,
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
        )
        .unwrap();
    let before = state.snapshot();

    let right = record(&[1], &[100]);
    let right = incoming(
        &right,
        vec![0],
        vec![GroupId::new(0).unwrap()],
        &[20],
        &[0],
        vec![true],
        1,
    );
    let error = state
        .apply(
            JoinSide::Right,
            right,
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("match limit"));
    assert_eq!(state.rows(JoinSide::Right), 0);
    assert_eq!(state.next_row_id(JoinSide::Right), 0);
    assert_eq!(state.next_output_sequence(), 0);
    assert_eq!(state.snapshot().fingerprint(), before.fingerprint());
}

#[test]
fn test_shared_key_routing_preserves_sparse_ids_and_skips_retention_work() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let mut keys = KeyStore::new(
        KeySchema::groupable(vec![Arc::new(Field::new("key", DataType::Int64, false))]).unwrap(),
        runtime.incremental_reservation("join-run-test"),
    );
    let left_record = record(&[1, 9, 1], &[10, 99, 11]);
    let left = Incoming::route(
        &left_record,
        &[0],
        vec![0, 2],
        vec![EventTime::from_micros(10), EventTime::from_micros(10)],
        vec![0, 2],
        datafusion::arrow::array::BooleanArray::from(vec![true, false]),
        3,
        &mut keys,
        &mut || Ok(()),
    )
    .unwrap();
    let mut state = state(100);
    state
        .apply(
            JoinSide::Left,
            left,
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
        )
        .unwrap();
    assert_eq!(state.rows(JoinSide::Left), 1);
    assert_eq!(state.next_row_id(JoinSide::Left), 3);

    let right_record = record(&[1], &[100]);
    let right = Incoming::route(
        &right_record,
        &[0],
        vec![0],
        vec![EventTime::from_micros(10)],
        vec![0],
        datafusion::arrow::array::BooleanArray::from(vec![false]),
        1,
        &mut keys,
        &mut || Ok(()),
    )
    .unwrap();
    let output = state
        .apply(
            JoinSide::Right,
            right,
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
        )
        .unwrap();
    assert_eq!(values(&output.records[0], 0), [10]);
    assert_eq!(state.rows(JoinSide::Right), 0);
}

#[test]
fn test_retained_row_and_byte_limits_reject_the_whole_run() {
    let source = record(&[1, 1], &[10, 11]);
    let make_incoming = || {
        incoming(
            &source,
            vec![0, 1],
            vec![GroupId::new(0).unwrap(); 2],
            &[10, 10],
            &[0, 1],
            vec![true, true],
            2,
        )
    };
    let mut rows_limited = RunState::new(
        "row-limit",
        JoinStateLimits::new(1, 1_000_000, 100).unwrap(),
        JoinTimeBounds::from_micros(0, 0).unwrap(),
    )
    .unwrap();
    rows_limited
        .apply(
            JoinSide::Left,
            make_incoming(),
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
        )
        .unwrap_err();
    assert_eq!(rows_limited.rows(JoinSide::Left), 0);
    assert_eq!(rows_limited.next_row_id(JoinSide::Left), 0);

    let mut measured = state(100);
    measured
        .apply(
            JoinSide::Left,
            make_incoming(),
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
        )
        .unwrap();
    let bytes = measured.bytes(JoinSide::Left);
    let mut bytes_limited = RunState::new(
        "byte-limit",
        JoinStateLimits::new(100, u64::try_from(bytes - 1).unwrap(), 100).unwrap(),
        JoinTimeBounds::from_micros(0, 0).unwrap(),
    )
    .unwrap();
    bytes_limited
        .apply(
            JoinSide::Left,
            make_incoming(),
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
        )
        .unwrap_err();
    assert_eq!(bytes_limited.rows(JoinSide::Left), 0);
    assert_eq!(bytes_limited.bytes(JoinSide::Left), 0);
    assert_eq!(bytes_limited.next_row_id(JoinSide::Left), 0);
}

fn fanout_state(left_rows: usize) -> (RunState, Incoming) {
    let mut state = state(100);
    let left = record(
        &vec![1; left_rows],
        &(0..i64::try_from(left_rows).unwrap()).collect::<Vec<_>>(),
    );
    let left = incoming(
        &left,
        (0..left_rows).collect(),
        vec![GroupId::new(0).unwrap(); left_rows],
        &vec![10; left_rows],
        &(0..left_rows as u64).collect::<Vec<_>>(),
        vec![true; left_rows],
        left_rows as u64,
    );
    state
        .apply(
            JoinSide::Left,
            left,
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
        )
        .unwrap();
    let right = record(&[1], &[100]);
    let right = incoming(
        &right,
        vec![0],
        vec![GroupId::new(0).unwrap()],
        &[10],
        &[0],
        vec![false],
        1,
    );
    (state, right)
}

fn estimated(record: &RecordBatch, sequence: u64) -> usize {
    Batch::table(
        vec![record.clone()],
        BatchMetadata::new("run-test", sequence, BTreeMap::new()).unwrap(),
    )
    .unwrap()
    .estimated_bytes()
    .unwrap()
}

#[test]
fn test_selection_materialization_preflights_row_and_byte_chunks() {
    let (mut state, right) = fanout_state(3);
    let output = state
        .apply(
            JoinSide::Right,
            right,
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::new(2, 1_000_000).unwrap(),
        )
        .unwrap();
    assert_eq!(
        output
            .records
            .iter()
            .map(RecordBatch::num_rows)
            .collect::<Vec<_>>(),
        [2, 1]
    );
    assert_eq!(output.sequences, [0, 1]);
    assert_eq!(
        output
            .records
            .iter()
            .flat_map(|record| values(record, 0))
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );

    let (mut state, right) = fanout_state(3);
    let whole = state
        .apply(
            JoinSide::Right,
            right,
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
        )
        .unwrap();
    let whole_bytes = estimated(&whole.records[0], 0);
    let (mut state, right) = fanout_state(3);
    let split = state
        .apply(
            JoinSide::Right,
            right,
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::new(100, whole_bytes - 1).unwrap(),
        )
        .unwrap();
    assert!(split.records.len() > 1);
    for index in 0..split.records.len() {
        let record = &split.records[index];
        let sequence = split.sequences[index];
        assert!(estimated(record, sequence) < whole_bytes);
    }
}

#[test]
fn test_one_byte_short_output_row_does_not_commit_input() {
    let (mut measured, right) = fanout_state(1);
    let output = measured
        .apply(
            JoinSide::Right,
            right,
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
        )
        .unwrap();
    let row_bytes = estimated(&output.records[0], 0);

    let (mut state, right) = fanout_state(1);
    let before = state.snapshot();
    let error = state
        .apply(
            JoinSide::Right,
            right,
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::new(100, row_bytes - 1).unwrap(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("output row"));
    assert_eq!(state.rows(JoinSide::Right), 0);
    assert_eq!(state.next_row_id(JoinSide::Right), 0);
    assert_eq!(state.next_output_sequence(), 0);
    assert_eq!(state.snapshot().fingerprint(), before.fingerprint());
}

fn complex_left(label_index: i32, value: i64) -> RecordBatch {
    let labels = DictionaryArray::<Int32Type>::new(
        Int32Array::from(vec![label_index]),
        Arc::new(StringArray::from(vec!["left-a", "left-b", "unused"])),
    );
    let lists = ListArray::from_iter_primitive::<Int64Type, _, _>([Some(vec![Some(value)])]);
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, false),
            Field::new(
                "label",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                false,
            ),
            Field::new("items", lists.data_type().clone(), false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1])),
            Arc::new(labels),
            Arc::new(lists),
        ],
    )
    .unwrap()
}

#[test]
fn test_multi_owner_take_interleave_preserves_dictionary_and_nested_payloads() {
    let dictionary = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    let list = DataType::List(Arc::new(Field::new("item", DataType::Int64, true)));
    let schema = Arc::new(Schema::new(vec![
        Field::new("left_label", dictionary.clone(), false),
        Field::new("left_items", list, false),
        Field::new("right_value", DataType::Int64, false),
    ]));
    let mut state = state(100);
    for (label, value, row_id) in [(0, 7, 0), (1, 8, 1)] {
        let record = complex_left(label, value);
        let left = incoming(
            &record,
            vec![0],
            vec![GroupId::new(0).unwrap()],
            &[10],
            &[row_id],
            vec![true],
            row_id + 1,
        );
        state
            .apply(
                JoinSide::Left,
                left,
                &schema,
                Some(&[1, 2, 4]),
                3,
                EdgeBudget::default(),
            )
            .unwrap();
    }
    let right_record = record(&[1], &[100]);
    let right = incoming(
        &right_record,
        vec![0],
        vec![GroupId::new(0).unwrap()],
        &[10],
        &[0],
        vec![false],
        1,
    );
    super::reset_join_work();
    let output = state
        .apply(
            JoinSide::Right,
            right,
            &schema,
            Some(&[1, 2, 4]),
            3,
            EdgeBudget::default(),
        )
        .unwrap();
    let work = super::join_work();
    assert!(work.output_column_interleaves >= 2);
    assert!(work.output_column_takes >= 1);

    let labels = output.records[0]
        .column(0)
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .unwrap();
    assert_eq!(
        labels.values().len(),
        2,
        "unused dictionary values are detached"
    );
    let labels =
        datafusion::arrow::compute::cast(output.records[0].column(0), &DataType::Utf8).unwrap();
    let labels = labels.as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!((labels.value(0), labels.value(1)), ("left-a", "left-b"));
    let items = output.records[0]
        .column(1)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    for (row, expected) in [7, 8].into_iter().enumerate() {
        assert_eq!(
            items
                .value(row)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .as_ref(),
            [expected]
        );
    }
    assert_eq!(values(&output.records[0], 2), [100, 100]);
}

fn eviction_state() -> RunState {
    let mut state = RunState::new(
        "run-test",
        JoinStateLimits::new(100, 1_000_000, 100).unwrap(),
        JoinTimeBounds::from_micros(10, 10).unwrap(),
    )
    .unwrap();
    for (values, times, first_id) in [([30, 10], [30, 10], 0), ([20, 40], [20, 40], 2)] {
        let record = record(&[1, 1], &values);
        let left = incoming(
            &record,
            vec![0, 1],
            vec![GroupId::new(0).unwrap(); 2],
            &times,
            &[first_id, first_id + 1],
            vec![true, true],
            first_id + 2,
        );
        state
            .apply(
                JoinSide::Left,
                left,
                &output_schema(),
                Some(&[1, 3]),
                2,
                EdgeBudget::default(),
            )
            .unwrap();
    }
    state
}

#[test]
fn test_prefix_eviction_and_snapshot_continue_bitwise() {
    let mut state = eviction_state();
    let none = state
        .evict(JoinSide::Left, 5, EventTime::from_micros(15))
        .unwrap();
    assert_eq!(none.expired_rows, 0);
    assert_eq!(state.rows(JoinSide::Left), 4);

    let expired = state
        .evict(JoinSide::Left, 5, EventTime::from_micros(16))
        .unwrap();
    assert_eq!(expired.expired_rows, 1);
    assert_eq!(expired.retired_runs, 0);
    assert_eq!(state.rows(JoinSide::Left), 3);

    let snapshot = state.snapshot();
    let mut restored = RunState::restore(snapshot).unwrap();
    assert_eq!(
        restored.snapshot().fingerprint(),
        state.snapshot().fingerprint()
    );
    let right_record = record(&[1], &[100]);
    let make_right = || {
        incoming(
            &right_record,
            vec![0],
            vec![GroupId::new(0).unwrap()],
            &[20],
            &[0],
            vec![true],
            1,
        )
    };
    let expected = state
        .apply(
            JoinSide::Right,
            make_right(),
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
        )
        .unwrap();
    let actual = restored
        .apply(
            JoinSide::Right,
            make_right(),
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
        )
        .unwrap();
    assert_eq!(actual.sequences, expected.sequences);
    assert_eq!(actual.records, expected.records);
    assert_eq!(values(&actual.records[0], 0), [20, 30]);
    assert_eq!(
        restored.snapshot().fingerprint(),
        state.snapshot().fingerprint()
    );

    let ended = restored.evict_end(JoinSide::Left).unwrap();
    assert_eq!(ended.expired_rows, 3);
    assert_eq!(ended.retired_runs, 2);
    assert_eq!(restored.rows(JoinSide::Left), 0);
    assert_eq!(restored.bytes(JoinSide::Left), 0);
}

#[test]
fn test_cancellation_before_morsel_leaves_run_state_unchanged() {
    let mut state = state(100);
    let before = state.snapshot();
    let source = record(&[1], &[10]);
    let left = incoming(
        &source,
        vec![0],
        vec![GroupId::new(0).unwrap()],
        &[10],
        &[0],
        vec![true],
        1,
    );
    state
        .apply_checked(
            JoinSide::Left,
            left,
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
            &mut || {
                Err(crate::CalcFlowError::Cancelled {
                    run_id: "run-test".into(),
                })
            },
        )
        .unwrap_err();
    assert_eq!(state.rows(JoinSide::Left), 0);
    assert_eq!(state.next_row_id(JoinSide::Left), 0);
    assert_eq!(state.next_output_sequence(), 0);
    assert_eq!(state.snapshot().fingerprint(), before.fingerprint());
}

#[test]
fn test_production_morsels_share_one_atomic_match_limit_and_commit() {
    let make_state = state;
    let mut state = state(1);
    let left_record = record(&[1], &[10]);
    let left = incoming(
        &left_record,
        vec![0],
        vec![GroupId::new(0).unwrap()],
        &[10],
        &[0],
        vec![true],
        1,
    );
    state
        .apply(
            JoinSide::Left,
            left,
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
        )
        .unwrap();
    let right_record = record(&[1, 1], &[100, 101]);
    let make_right = |row| {
        incoming(
            &right_record,
            vec![row],
            vec![GroupId::new(0).unwrap()],
            &[10],
            &[row as u64],
            vec![false],
            row as u64 + 1,
        )
    };
    let before = state.snapshot();
    let error = state
        .prepare_batch(
            JoinSide::Right,
            vec![make_right(0), make_right(1)],
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
            &mut || Ok(()),
        )
        .err()
        .expect("two morsels share one match limit");
    assert!(error.to_string().contains("match limit"));
    assert_eq!(state.next_row_id(JoinSide::Right), 0);
    assert_eq!(state.next_output_sequence(), 0);
    assert_eq!(state.snapshot().fingerprint(), before.fingerprint());

    drop(state);
    let mut accepted = make_state(100);
    let left = incoming(
        &left_record,
        vec![0],
        vec![GroupId::new(0).unwrap()],
        &[10],
        &[0],
        vec![true],
        1,
    );
    accepted
        .apply(
            JoinSide::Left,
            left,
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
        )
        .unwrap();
    let prepared = accepted
        .prepare_batch(
            JoinSide::Right,
            vec![make_right(0), make_right(1)],
            &output_schema(),
            Some(&[1, 3]),
            2,
            EdgeBudget::default(),
            &mut || Ok(()),
        )
        .unwrap();
    assert_eq!(prepared.outputs().len(), 2);
    assert_eq!(accepted.rows(JoinSide::Right), 0);
    assert_eq!(accepted.next_output_sequence(), 0);
    prepared.commit(&mut accepted).unwrap();
    assert_eq!(accepted.next_row_id(JoinSide::Right), 2);
    assert_eq!(accepted.next_output_sequence(), 2);
}
