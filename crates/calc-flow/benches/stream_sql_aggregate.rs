//! Cumulative SQL snapshots with separately observed checkpoint preparation.

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, CancellationToken, Epoch, JsonMap, OperatorStateSnapshot, SqlOperator,
    StreamCollector, StreamJobContext, StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{self, Write},
    sync::Arc,
    time::Instant,
};

const ROWS: usize = 100_000;
const QUERY: &str = "SELECT key, SUM(value) AS total, COUNT(value) AS count, MIN(value) AS minimum, MAX(value) AS maximum FROM events GROUP BY key";

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    batches: usize,
    checkpoint_every: usize,
    rows: usize,
    unique_keys: bool,
}

const CASES: [Case; 7] = [
    Case {
        name: "fixed_groups_1_batch",
        batches: 1,
        checkpoint_every: 0,
        rows: ROWS,
        unique_keys: false,
    },
    Case {
        name: "fixed_groups_10_batches",
        batches: 10,
        checkpoint_every: 0,
        rows: ROWS,
        unique_keys: false,
    },
    Case {
        name: "fixed_groups_100_batches",
        batches: 100,
        checkpoint_every: 0,
        rows: ROWS,
        unique_keys: false,
    },
    Case {
        name: "fixed_groups_100_batches_checkpoint_10",
        batches: 100,
        checkpoint_every: 10,
        rows: ROWS,
        unique_keys: false,
    },
    Case {
        name: "fixed_groups_100_batches_checkpoint_1",
        batches: 100,
        checkpoint_every: 1,
        rows: ROWS,
        unique_keys: false,
    },
    Case {
        name: "fixed_groups_1000_batches",
        batches: 1000,
        checkpoint_every: 0,
        rows: ROWS,
        unique_keys: false,
    },
    Case {
        name: "growing_groups_10k_100_batches",
        batches: 100,
        checkpoint_every: 0,
        rows: 10_000,
        unique_keys: true,
    },
];

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, true),
        Field::new("value", DataType::Int64, true),
    ]))
}

fn input_row(row: usize, unique_keys: bool) -> (Option<i64>, Option<i64>) {
    let key = (row % 101 != 0).then_some(if unique_keys {
        row as i64
    } else {
        (row % 64) as i64
    });
    let value = (row % 13 != 0 && key != Some(63)).then_some((row % 257) as i64 - 128);
    (key, value)
}

fn input_batch(start: usize, rows: usize, unique_keys: bool) -> Batch {
    let keys =
        Int64Array::from_iter((start..start + rows).map(|row| input_row(row, unique_keys).0));
    let values =
        Int64Array::from_iter((start..start + rows).map(|row| input_row(row, unique_keys).1));
    Batch::table(
        vec![RecordBatch::try_new(schema(), vec![Arc::new(keys), Arc::new(values)]).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn operator() -> SqlOperator {
    SqlOperator::new("aggregate", QUERY, vec!["events".into()], Vec::new()).unwrap()
}

#[derive(Default)]
struct Collector {
    snapshots: Vec<Batch>,
}

#[async_trait]
impl StreamCollector for Collector {
    async fn emit(&mut self, port: &str, batch: Batch) -> calc_flow::Result<()> {
        assert_eq!(port, "output");
        self.snapshots.push(batch);
        Ok(())
    }
}

#[derive(Default)]
struct Aggregate {
    count: i64,
    sum: i64,
    min: Option<i64>,
    max: Option<i64>,
}

fn update(
    expected: &mut BTreeMap<Option<i64>, Aggregate>,
    start: usize,
    rows: usize,
    unique_keys: bool,
) {
    for row in start..start + rows {
        let (key, value) = input_row(row, unique_keys);
        let aggregate = expected.entry(key).or_default();
        if let Some(value) = value {
            aggregate.count += 1;
            aggregate.sum += value;
            aggregate.min = Some(aggregate.min.map_or(value, |old| old.min(value)));
            aggregate.max = Some(aggregate.max.map_or(value, |old| old.max(value)));
        }
    }
}

fn validate_snapshot(
    batch: &Batch,
    expected: &BTreeMap<Option<i64>, Aggregate>,
) -> (usize, [u8; 32]) {
    let mut seen = BTreeMap::new();
    let names = ["key", "total", "count", "minimum", "maximum"];
    for record in batch.table_payload().unwrap().batches() {
        assert_eq!(record.schema().fields().len(), names.len());
        for (index, name) in names.iter().enumerate() {
            assert_eq!(record.schema().field(index).name(), name);
            assert_eq!(record.schema().field(index).data_type(), &DataType::Int64);
        }
        let integer = |column: usize| {
            record
                .column(column)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
        };
        for row in 0..record.num_rows() {
            let key = (!integer(0).is_null(row)).then(|| integer(0).value(row));
            let optional =
                |column: usize| (!integer(column).is_null(row)).then(|| integer(column).value(row));
            assert!(
                seen.insert(
                    key,
                    (integer(2).value(row), optional(1), optional(3), optional(4))
                )
                .is_none()
            );
            let wanted = expected.get(&key).expect("no unexpected output group");
            assert!(!integer(2).is_null(row));
            assert_eq!(integer(2).value(row), wanted.count);
            for (column, value) in [
                (1, (wanted.count > 0).then_some(wanted.sum)),
                (3, wanted.min),
                (4, wanted.max),
            ] {
                assert_eq!(integer(column).is_null(row), value.is_none());
                if let Some(value) = value {
                    assert_eq!(integer(column).value(row), value);
                }
            }
        }
    }
    assert_eq!(seen.len(), expected.len());
    let mut hash = Sha256::new();
    for (key, (count, sum, min, max)) in &seen {
        let mut nullable = |value: Option<i64>| {
            hash.update([u8::from(value.is_some())]);
            hash.update(value.unwrap_or(0).to_le_bytes());
        };
        nullable(*key);
        hash.update(count.to_le_bytes());
        for value in [sum, min, max] {
            hash.update([u8::from(value.is_some())]);
            hash.update(value.unwrap_or(0).to_le_bytes());
        }
    }
    (seen.len(), hash.finalize().into())
}

async fn capture(
    operator: &mut SqlOperator,
    context: &StreamOperatorContext<'_>,
) -> OperatorStateSnapshot {
    operator.prepare_checkpoint_async(context).await.unwrap();
    operator.checkpoint(Epoch::INITIAL).unwrap()
}

fn sample(runtime: &tokio::runtime::Runtime, case: Case, inputs: &[Batch], recover: bool) -> Value {
    let job = StreamJobContext::new(
        1,
        "sql-aggregate",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "aggregate", None);
    let mut operator = operator();
    let mut collector = Collector::default();
    runtime
        .block_on(operator.process_data(
            "events",
            input_batch(0, 0, case.unique_keys),
            &context,
            &mut collector,
        ))
        .unwrap();
    assert_eq!(collector.snapshots.len(), 1);
    validate_snapshot(&collector.snapshots[0], &BTreeMap::new());
    operator.reset().unwrap();
    collector.snapshots.clear();
    let mut process_seconds = 0.0;
    let mut prepare_seconds = 0.0;
    let mut capture_seconds = 0.0;
    let mut checkpoint_bytes = 0_usize;
    let mut checkpoint_count = 0_usize;
    let mut last_checkpoint = None;
    let started = Instant::now();
    runtime.block_on(async {
        for (index, batch) in inputs.iter().enumerate() {
            let process = Instant::now();
            operator
                .process_data("events", batch.clone(), &context, &mut collector)
                .await
                .unwrap();
            process_seconds += process.elapsed().as_secs_f64();
            if case.checkpoint_every > 0
                && ((index + 1) % case.checkpoint_every == 0 || index + 1 == inputs.len())
            {
                let prepare = Instant::now();
                operator.prepare_checkpoint_async(&context).await.unwrap();
                prepare_seconds += prepare.elapsed().as_secs_f64();
                let capture = Instant::now();
                let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
                capture_seconds += capture.elapsed().as_secs_f64();
                checkpoint_bytes += snapshot
                    .segments
                    .values()
                    .map(|segment| segment.bytes().len())
                    .sum::<usize>();
                checkpoint_count += 1;
                last_checkpoint = Some(snapshot);
            }
        }
    });
    let seconds = started.elapsed().as_secs_f64();
    assert_eq!(collector.snapshots.len(), inputs.len());
    let mut expected = BTreeMap::new();
    let mut output_rows = 0;
    let mut output_hash = Sha256::new();
    for (index, batch) in collector.snapshots.iter().enumerate() {
        update(
            &mut expected,
            index * (case.rows / case.batches),
            case.rows / case.batches,
            case.unique_keys,
        );
        let (rows, digest) = validate_snapshot(batch, &expected);
        output_rows += rows;
        output_hash.update(digest);
    }
    let final_checkpoint = runtime.block_on(capture(&mut operator, &context));
    let segment = final_checkpoint.segments.get("input").unwrap();
    assert_eq!(
        final_checkpoint.inline_metadata["rows"].as_u64(),
        Some(case.rows as u64)
    );
    let checkpoint_sha256 = hex::encode(Sha256::digest(segment.bytes()));
    if let Some(last) = last_checkpoint {
        assert_eq!(last.segments["input"].bytes(), segment.bytes());
    }
    if recover {
        let mut restored = self::operator();
        restored.restore(&final_checkpoint).unwrap();
        let mut recovered = Collector::default();
        runtime
            .block_on(restored.process_data(
                "events",
                input_batch(0, 0, case.unique_keys),
                &context,
                &mut recovered,
            ))
            .unwrap();
        assert_eq!(recovered.snapshots.len(), 1);
        validate_snapshot(&recovered.snapshots[0], &expected);
        runtime
            .block_on(restored.process_data(
                "events",
                input_batch(case.rows, 1, case.unique_keys),
                &context,
                &mut recovered,
            ))
            .unwrap();
        update(&mut expected, case.rows, 1, case.unique_keys);
        assert_eq!(recovered.snapshots.len(), 2);
        validate_snapshot(&recovered.snapshots[1], &expected);
    }
    json!({"seconds":seconds,"process_seconds":process_seconds,"prepare_seconds":prepare_seconds,"capture_seconds":capture_seconds,"checkpoint_bytes":checkpoint_bytes,"checkpoint_count":checkpoint_count,"final_checkpoint_bytes":segment.bytes().len(),"final_checkpoint_sha256":checkpoint_sha256,"output_rows":output_rows,"snapshot_count":collector.snapshots.len(),"input_rows":case.rows,"all_snapshots_sha256":hex::encode(output_hash.finalize()),"validated_all_snapshots":true,"validated_recovery":recover})
}

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    let check = args.iter().any(|arg| arg == "--check" || arg == "--test");
    let samples = args
        .iter()
        .position(|arg| arg == "--samples")
        .map_or(5, |index| args[index + 1].parse::<usize>().unwrap());
    assert!(samples > 0);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let cases = CASES.into_iter().map(|case| {
        let chunk = case.rows / case.batches;
        let inputs = (0..case.rows).step_by(chunk).map(|start| input_batch(start, chunk, case.unique_keys)).collect::<Vec<_>>();
        let oracle = sample(&runtime, case, &inputs, true);
        let observations = if check { Vec::new() } else { (0..samples).map(|_| sample(&runtime, case, &inputs, false)).collect::<Vec<_>>() };
        json!({"name":case.name,"rows":case.rows,"batches":case.batches,"maximum_groups":if case.unique_keys { case.rows - case.rows.div_ceil(101) + 1 } else { 65 },"unique_keys":case.unique_keys,"checkpoint_every":case.checkpoint_every,"query":QUERY,"oracle":oracle,"samples":observations})
    }).collect::<Vec<_>>();
    let report = json!({"schema":"calc-flow.sql-stream-aggregate.v1","scope":"warm-native-operator-cumulative-snapshots","cases":cases});
    let bytes = serde_json::to_vec_pretty(&report).unwrap();
    if let Some(index) = args.iter().position(|arg| arg == "--output") {
        let path = std::path::Path::new(args.get(index + 1).expect("--output needs a path"));
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, bytes).unwrap();
    } else {
        io::stdout().write_all(&bytes).unwrap();
    }
}
