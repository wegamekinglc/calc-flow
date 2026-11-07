//! Tests for key-sharded grouped SUM updates: admission, dispatch, and
//! byte-identical snapshots against the serial path.

use super::*;
use crate::runtime::streaming::gather_work::GatherStop;
use crate::{Batch, CancellationToken, DataFusionConfig, JsonMap, StreamJobContext};
use datafusion::arrow::{
    array::{Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
};
use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

static DISPATCHES: AtomicUsize = AtomicUsize::new(0);
static UNITS: AtomicUsize = AtomicUsize::new(0);
static UNIT_ORDINALS: Mutex<Vec<usize>> = Mutex::new(Vec::new());

fn reset() {
    DISPATCHES.store(0, Ordering::SeqCst);
    UNITS.store(0, Ordering::SeqCst);
    UNIT_ORDINALS.lock().clear();
}

pub(super) fn before_unit(_name: &str, ordinal: usize, stop: &GatherStop) -> Result<()> {
    stop.check()?;
    UNIT_ORDINALS.lock().push(ordinal);
    Ok(())
}

pub(super) fn after_unit(_name: &str) {}

pub(super) fn dispatched(_name: &str, units: usize) {
    DISPATCHES.fetch_add(1, Ordering::SeqCst);
    UNITS.fetch_max(units, Ordering::SeqCst);
}

const ROWS: usize = 70_000;
const GROUPS: usize = 3_000;

fn input() -> Batch {
    let keys = (0..ROWS)
        .map(|row| format!("k{:05}", row % GROUPS))
        .collect::<Vec<_>>();
    let values = (0..ROWS)
        .map(|row| i64::try_from(row % 97 + 1).unwrap())
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(keys)) as ArrayRef,
            Arc::new(Int64Array::from(values)) as ArrayRef,
        ],
    )
    .unwrap();
    Batch::table(vec![record], crate::BatchMetadata::default()).unwrap()
}

async fn plan(input: &Batch) -> IncrementalSql {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let query = crate::expression::parse_select_query(
        "SELECT key, SUM(value) AS total FROM events GROUP BY key",
    )
    .unwrap();
    let schema = input.table_payload().unwrap().schema().clone();
    IncrementalSql::plan(&runtime, &query, "events", schema, "key-sharding")
        .await
        .unwrap()
        .unwrap()
}

fn context() -> StreamJobContext {
    StreamJobContext::new(
        973,
        "key-sharding",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn snapshot_rows(records: &[RecordBatch]) -> Vec<(Option<String>, Option<i64>)> {
    records
        .iter()
        .flat_map(|record| {
            let keys = record
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let values = record
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            (0..record.num_rows()).map(move |row| {
                (
                    (!keys.is_null(row)).then(|| keys.value(row).to_owned()),
                    (!values.is_null(row)).then(|| values.value(row)),
                )
            })
        })
        .collect()
}

#[tokio::test]
async fn dense_batches_dispatch_and_match_the_serial_path() {
    reset();
    let input = input();
    let job = context();
    let operator_context = StreamOperatorContext::new(&job, "key-sharding", None);

    let mut sharded = plan(&input).await;
    assert!(sharded.parallel_scratch, "default pool must admit sharding");
    let first = sharded
        .update(&input, &operator_context, "key-sharding")
        .await
        .unwrap();
    let sharded_records = first.records.clone();
    sharded.commit(first);
    let second = sharded
        .update(&input, &operator_context, "key-sharding")
        .await
        .unwrap();
    let sharded_second = second.records.clone();
    sharded.commit(second);

    assert!(
        DISPATCHES.load(Ordering::SeqCst) >= 2,
        "both batches must dispatch"
    );
    assert!(
        UNITS.load(Ordering::SeqCst) >= 2,
        "sharding needs at least two units"
    );
    assert_eq!(
        UNIT_ORDINALS.lock().len(),
        DISPATCHES.load(Ordering::SeqCst) * UNITS.load(Ordering::SeqCst),
        "every unit of every dispatch must run"
    );

    let mut serial = plan(&input).await;
    serial.disable_parallel_scratch_for_test();
    assert!(!serial.parallel_scratch, "forced serial reference");
    let first = serial
        .update(&input, &operator_context, "key-sharding")
        .await
        .unwrap();
    let serial_records = first.records.clone();
    serial.commit(first);
    let second = serial
        .update(&input, &operator_context, "key-sharding")
        .await
        .unwrap();
    let serial_second = second.records.clone();
    serial.commit(second);

    assert_eq!(
        snapshot_rows(&sharded_records),
        snapshot_rows(&serial_records)
    );
    assert_eq!(
        snapshot_rows(&sharded_second),
        snapshot_rows(&serial_second)
    );
    let mut expected = std::collections::BTreeMap::new();
    for row in 0..ROWS {
        *expected
            .entry(format!("k{:05}", row % GROUPS))
            .or_insert(0_i64) += i64::try_from(row % 97 + 1).unwrap();
    }
    let doubled: std::collections::BTreeMap<_, _> = expected
        .iter()
        .map(|(key, value)| (key.clone(), value.checked_mul(2).unwrap()))
        .collect();
    let observed = snapshot_rows(&sharded_second);
    assert_eq!(observed.len(), GROUPS);
    for (key, value) in observed {
        let key = key.expect("dense keys");
        assert_eq!(Some(doubled[&key]), value, "group {key} total");
    }
}

#[tokio::test]
async fn small_batches_stay_serial() {
    reset();
    let keys = (0..1_000).map(|row| format!("k{row}")).collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, true),
        Field::new("value", DataType::Int64, true),
    ]));
    let record = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(keys)) as ArrayRef,
            Arc::new(Int64Array::from(vec![1_i64; 1_000])) as ArrayRef,
        ],
    )
    .unwrap();
    let input = Batch::table(vec![record], crate::BatchMetadata::default()).unwrap();
    let job = context();
    let operator_context = StreamOperatorContext::new(&job, "key-sharding", None);
    let mut plan = plan(&input).await;
    let transaction = plan
        .update(&input, &operator_context, "key-sharding")
        .await
        .unwrap();
    let records = transaction.records.clone();
    plan.commit(transaction);
    assert_eq!(DISPATCHES.load(Ordering::SeqCst), 0);
    assert_eq!(
        records.iter().map(RecordBatch::num_rows).sum::<usize>(),
        1_000
    );
}
