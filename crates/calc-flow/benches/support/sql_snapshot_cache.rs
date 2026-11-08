//! One fresh-process observation of twenty full-snapshot appends.

use super::{
    Batch, BatchMetadata, Collector, SqlOperator, StreamOperator, capture, checkpoint_digest,
};
use calc_flow::benchmark_internals::{PhaseClock, process_observations, settle_job};
use calc_flow::{CancellationToken, JsonMap, StreamJobContext, StreamOperatorContext};
use datafusion::arrow::{
    array::{Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    ipc::reader::FileReader,
    record_batch::RecordBatch,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{ffi::OsString, io::Cursor, sync::Arc, time::Instant};

const GROUPS: usize = 10_000;
const APPENDS: usize = 20;
const SQL: &str = "SELECT account, stock, SUM(value) AS total FROM events GROUP BY account, stock";
type Key = (Option<String>, String);

fn selected_slots(args: &[OsString]) -> (&str, Vec<usize>) {
    assert!(!args.iter().any(|arg| matches!(
        arg.to_str(),
        Some("--samples" | "--check" | "--test" | "--value-type" | "--decimal-scale")
    )));
    let position = args
        .iter()
        .position(|arg| arg == "--snapshot-cache-case")
        .unwrap();
    let mode = args.get(position + 1).and_then(|arg| arg.to_str()).unwrap();
    let mut slots = match mode {
        "sparse" => (9000..GROUPS).collect::<Vec<_>>(),
        "dense" => (0..1000).map(|index| (index * 37) % GROUPS).collect(),
        _ => panic!("--snapshot-cache-case requires sparse or dense"),
    };
    slots.sort_unstable();
    assert_eq!(slots.len(), 1000);
    assert!(slots.windows(2).all(|pair| pair[0] < pair[1]));
    (mode, slots)
}

fn keys() -> Vec<Key> {
    (0..GROUPS)
        .map(|slot| {
            (
                (slot != 0).then(|| format!("account-{:03}", slot / 100)),
                format!("stock-{:03}", slot % 100),
            )
        })
        .collect()
}

fn schema(value_name: &str) -> Schema {
    Schema::new(vec![
        Field::new("account", DataType::Utf8, true),
        Field::new("stock", DataType::Utf8, false),
        Field::new(value_name, DataType::Int64, value_name == "total"),
    ])
}

fn input(keys: &[Key], slots: &[usize], value: i64) -> Batch {
    let accounts = StringArray::from(
        slots
            .iter()
            .map(|&slot| keys[slot].0.as_deref())
            .collect::<Vec<_>>(),
    );
    let stocks = StringArray::from_iter_values(slots.iter().map(|&slot| keys[slot].1.as_str()));
    Batch::table(
        vec![
            RecordBatch::try_new(
                Arc::new(schema("value")),
                vec![
                    Arc::new(accounts),
                    Arc::new(stocks),
                    Arc::new(Int64Array::from(vec![value; slots.len()])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn text_hash(hash: &mut Sha256, value: Option<&str>) {
    hash.update([u8::from(value.is_some())]);
    if let Some(value) = value {
        hash.update(u64::try_from(value.len()).unwrap().to_le_bytes());
        hash.update(value.as_bytes());
    }
}

struct Columns<'a> {
    accounts: &'a StringArray,
    stocks: &'a StringArray,
    values: &'a Int64Array,
}

impl<'a> Columns<'a> {
    fn new(record: &'a RecordBatch, value_name: &str) -> Self {
        assert_eq!(record.schema().as_ref(), &schema(value_name));
        let columns = Self {
            accounts: record.column(0).as_any().downcast_ref().unwrap(),
            stocks: record.column(1).as_any().downcast_ref().unwrap(),
            values: record.column(2).as_any().downcast_ref().unwrap(),
        };
        assert_eq!(columns.stocks.null_count(), 0);
        assert_eq!(columns.values.null_count(), 0);
        columns
    }

    fn key(&self, row: usize) -> (Option<&str>, &str) {
        (
            (!self.accounts.is_null(row)).then(|| self.accounts.value(row)),
            self.stocks.value(row),
        )
    }
}

fn row_digest(batch: &Batch, value_name: &str) -> String {
    assert_eq!(batch.metadata(), &BatchMetadata::default());
    let mut hash = Sha256::new();
    hash.update(b"calc-flow.c1-rows.v1\0");
    hash.update(u64::try_from(batch.num_rows()).unwrap().to_le_bytes());
    for record in batch.table_payload().unwrap().batches() {
        let columns = Columns::new(record, value_name);
        for row in 0..record.num_rows() {
            let (account, stock) = columns.key(row);
            text_hash(&mut hash, account);
            text_hash(&mut hash, Some(stock));
            hash.update([1]);
            hash.update(columns.values.value(row).to_le_bytes());
        }
    }
    hex::encode(hash.finalize())
}

fn validate(batch: &Batch, keys: &[Key], expected: &[i64]) -> String {
    assert_eq!(batch.num_rows(), GROUPS);
    let records = batch.table_payload().unwrap().batches();
    assert_eq!(
        records
            .iter()
            .map(RecordBatch::num_rows)
            .collect::<Vec<_>>(),
        [8192, 1808]
    );
    let mut position = 0;
    for record in records {
        let columns = Columns::new(record, "total");
        for row in 0..record.num_rows() {
            assert_eq!(
                columns.key(row),
                (keys[position].0.as_deref(), keys[position].1.as_str())
            );
            assert_eq!(columns.values.value(row), expected[position]);
            position += 1;
        }
    }
    row_digest(batch, "total")
}

fn take(collector: &mut Collector) -> Batch {
    assert_eq!(collector.snapshots.len(), 1);
    collector.snapshots.pop().unwrap()
}

fn operator() -> SqlOperator {
    SqlOperator::new("c1-snapshot-cache", SQL, vec!["events".into()], Vec::new()).unwrap()
}

fn checkpoint_state(snapshot: &calc_flow::OperatorStateSnapshot) -> Value {
    assert_eq!(snapshot.inline_metadata["state_layout"], 3);
    assert_eq!(snapshot.inline_metadata["state_accounting"], 3);
    assert_eq!(snapshot.inline_metadata["rows"], 30000);
    assert_eq!(
        snapshot
            .segments
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["batch-metadata", "control", "group-state", "logical-schema"]
    );
    let reader =
        FileReader::try_new(Cursor::new(snapshot.segments["group-state"].bytes()), None).unwrap();
    assert_eq!(reader.schema().fields().len(), 3);
    let rows = reader
        .map(|record| record.unwrap().num_rows())
        .sum::<usize>();
    assert_eq!(rows, GROUPS);
    json!({"layout":3,"groups":rows,"columns":3,"logical_rows":30000,"logical_bytes":snapshot.inline_metadata["bytes"],"sha256":checkpoint_digest(snapshot)})
}

async fn recover(
    snapshot: &calc_flow::OperatorStateSnapshot,
    context: &StreamOperatorContext<'_>,
    keys: &[Key],
    expected: &mut [i64],
) -> Value {
    let mut restored = operator();
    restored.restore(snapshot).unwrap();
    let mut collector = Collector::default();
    let empty = input(keys, &[], 0);
    let continuation = input(keys, &[0], 1);
    let input_hashes = [
        row_digest(&empty, "value"),
        row_digest(&continuation, "value"),
    ];
    restored
        .process_data("events", empty, context, &mut collector)
        .await
        .unwrap();
    let first = validate(&take(&mut collector), keys, expected);
    restored
        .process_data("events", continuation, context, &mut collector)
        .await
        .unwrap();
    expected[0] += 1;
    let second = validate(&take(&mut collector), keys, expected);
    restored.reset().unwrap();
    json!({"input_sha256":input_hashes,"snapshot_sha256":[first,second],"validated":true})
}

async fn sample(mode: &str, slots: &[usize]) -> Value {
    let keys = keys();
    let warm = input(&keys, &(0..GROUPS).collect::<Vec<_>>(), 1);
    let inputs = (1..=APPENDS)
        .map(|value| input(&keys, slots, i64::try_from(value).unwrap()))
        .collect::<Vec<_>>();
    let job = StreamJobContext::new(
        1,
        "c1-snapshot-cache",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "c1-snapshot-cache", None);
    let mut operator = operator();
    let mut collector = Collector::default();
    let validation = Instant::now();
    let warm_input_hash = row_digest(&warm, "value");
    let input_validation_seconds = validation.elapsed().as_secs_f64();
    let started = Instant::now();
    operator
        .process_data("events", warm, &context, &mut collector)
        .await
        .unwrap();
    let warm_seconds = started.elapsed().as_secs_f64();
    let original = take(&mut collector);
    let initial = vec![1; GROUPS];
    let mut expected = initial.clone();
    let validation = Instant::now();
    let warm_hash = validate(&original, &keys, &initial);
    let mut validation_seconds = input_validation_seconds + validation.elapsed().as_secs_f64();
    let mut appends = Vec::new();
    for (index, batch) in inputs.iter().enumerate() {
        let started = Instant::now();
        operator
            .process_data("events", batch.clone(), &context, &mut collector)
            .await
            .unwrap();
        let seconds = started.elapsed().as_secs_f64();
        let current = take(&mut collector);
        let started = Instant::now();
        for &slot in slots {
            expected[slot] += i64::try_from(index + 1).unwrap();
        }
        let output_hash = validate(&current, &keys, &expected);
        let held_hash = validate(&original, &keys, &initial);
        assert_eq!(held_hash, warm_hash);
        let input_hash = row_digest(batch, "value");
        validation_seconds += started.elapsed().as_secs_f64();
        appends.push(json!({"index":index,"seconds":seconds,"input_sha256":input_hash,"output_sha256":output_hash,"held_warm_sha256":held_hash}));
    }
    drop(original);
    let started = Instant::now();
    let snapshot = capture(&mut operator, &context).await;
    let checkpoint = checkpoint_state(&snapshot);
    let checkpoint_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let recovery = recover(&snapshot, &context, &keys, &mut expected).await;
    let recovery_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    operator.reset().unwrap();
    drop(operator);
    settle_job(&job).await.unwrap();
    let cleanup_seconds = started.elapsed().as_secs_f64();
    let process_seconds = appends
        .iter()
        .map(|sample| sample["seconds"].as_f64().unwrap())
        .sum::<f64>();
    json!({"schema":"calc-flow.sql-snapshot-cache-invocation.v1","scope":"warm-native-sql-operator-append-full-snapshots","selection":mode,"query":SQL,"metadata":BatchMetadata::default(),"counts":{"warm":{"input_rows":10000,"output_rows":10000,"snapshots":1},"append":{"input_rows":20000,"output_rows":200000,"snapshots":20},"recovery":{"input_rows":1,"output_rows":20000,"snapshots":2},"total":{"input_rows":30001,"output_rows":230000,"snapshots":23},"held_warm_validation_rows":200000},"warm":{"input_sha256":warm_input_hash,"sha256":warm_hash,"chunks":[8192,1808]},"appends":appends,"process_seconds":process_seconds,"warm_seconds":warm_seconds,"validation_seconds":validation_seconds,"checkpoint_seconds":checkpoint_seconds,"recovery_seconds":recovery_seconds,"cleanup_seconds":cleanup_seconds,"checkpoint":checkpoint,"recovery":recovery,"engine_reserved":{"status":"unmeasured"},"engine_limit_basis":"default-runtime-known-infinite","cleanup":true})
}

pub(super) fn run(args: &[OsString]) -> Value {
    let (mode, slots) = selected_slots(args);
    let started = Instant::now();
    let resources = PhaseClock::start("c1-invocation-before-json");
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut report = runtime.block_on(sample(mode, &slots));
    let cleanup = Instant::now();
    drop(runtime);
    report["cleanup_seconds"] =
        json!(report["cleanup_seconds"].as_f64().unwrap() + cleanup.elapsed().as_secs_f64());
    report["total_seconds"] = json!(started.elapsed().as_secs_f64());
    report["resources"] = resources.finish();
    report
        .as_object_mut()
        .unwrap()
        .extend(process_observations().unwrap().as_object().unwrap().clone());
    report
}
