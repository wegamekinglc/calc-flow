//! Cumulative SQL snapshots with separately observed checkpoint preparation.

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, CancellationToken, Epoch, JsonMap, OperatorStateSnapshot, SqlOperator,
    StreamCollector, StreamJobContext, StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{Array, ArrayRef, Int64Array, new_empty_array},
    datatypes::{DataType, Field, Schema, i256},
    record_batch::RecordBatch,
};
use datafusion::common::ScalarValue;
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

#[derive(Clone, Copy)]
struct ValueType {
    bits: u16,
    precision: u8,
    scale: i8,
}

impl ValueType {
    fn total_precision(self) -> u8 {
        let maximum = match self.bits {
            32 => 9,
            64 => 18,
            128 => 38,
            256 => 76,
            _ => 0,
        };
        (self.precision + 10).min(maximum)
    }

    fn data_type(self, total: bool) -> DataType {
        let precision = if total {
            self.total_precision()
        } else {
            self.precision
        };
        match self.bits {
            32 => DataType::Decimal32(precision, self.scale),
            64 => DataType::Decimal64(precision, self.scale),
            128 => DataType::Decimal128(precision, self.scale),
            256 => DataType::Decimal256(precision, self.scale),
            _ => DataType::Int64,
        }
    }

    fn scalar(self, value: Option<i64>, total: bool) -> ScalarValue {
        match self.data_type(total) {
            DataType::Decimal32(p, s) => {
                ScalarValue::Decimal32(value.map(|v| i32::try_from(v).unwrap()), p, s)
            }
            DataType::Decimal64(p, s) => ScalarValue::Decimal64(value, p, s),
            DataType::Decimal128(p, s) => ScalarValue::Decimal128(value.map(i128::from), p, s),
            DataType::Decimal256(p, s) => {
                ScalarValue::Decimal256(value.map(|v| i256::from_i128(i128::from(v))), p, s)
            }
            DataType::Int64 => ScalarValue::Int64(value),
            _ => unreachable!(),
        }
    }

    fn array(self, values: Vec<Option<i64>>) -> ArrayRef {
        if self.bits == 0 {
            return Arc::new(Int64Array::from(values));
        }
        if values.is_empty() {
            return new_empty_array(&self.data_type(false));
        }
        ScalarValue::iter_to_array(values.into_iter().map(|value| self.scalar(value, false)))
            .unwrap()
    }

    fn descriptor(self) -> Value {
        let total = self.total_precision();
        json!({"name":format!("decimal{}",self.bits),"input":{"bits":self.bits,"precision":self.precision,"scale":self.scale},"total":{"bits":self.bits,"precision":total,"scale":self.scale}})
    }
}

fn value_type(args: &[std::ffi::OsString]) -> ValueType {
    let name = args
        .iter()
        .position(|arg| arg == "--value-type")
        .map_or("integer", |index| args[index + 1].to_str().unwrap());
    let (bits, precision) = match name {
        "integer" => (0, 0),
        "decimal32" => (32, 4),
        "decimal64" => (64, 12),
        "decimal128" => (128, 28),
        "decimal256" => (256, 60),
        _ => panic!("unsupported --value-type"),
    };
    let scale = decimal_scale(args, bits);
    ValueType {
        bits,
        precision,
        scale,
    }
}

fn decimal_scale(args: &[std::ffi::OsString], bits: u16) -> i8 {
    let scale = args
        .iter()
        .position(|arg| arg == "--decimal-scale")
        .map_or(2, |index| {
            assert!(bits != 0, "--decimal-scale requires decimal input");
            args[index + 1].to_str().unwrap().parse::<i8>().unwrap()
        });
    assert!(scale == -2 || scale == 2);
    scale
}

fn schema(value_type: ValueType) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, true),
        Field::new("value", value_type.data_type(false), true),
    ]))
}

fn input_row(row: usize, unique_keys: bool) -> (Option<i64>, Option<i64>) {
    let group = i64::try_from(if unique_keys { row } else { row % 64 }).unwrap();
    let key = (row % 101 != 0).then_some(group);
    let value =
        (row % 13 != 0 && key != Some(63)).then_some(i64::try_from(row % 257).unwrap() - 128);
    (key, value)
}

fn input_batch(start: usize, rows: usize, unique_keys: bool, value_type: ValueType) -> Batch {
    let keys = (start..start + rows)
        .map(|row| input_row(row, unique_keys).0)
        .collect::<Int64Array>();
    let values = (start..start + rows)
        .map(|row| input_row(row, unique_keys).1)
        .collect::<Vec<_>>();
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema(value_type),
                vec![Arc::new(keys), value_type.array(values)],
            )
            .unwrap(),
        ],
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

type SnapshotValues = (i64, ScalarValue, ScalarValue, ScalarValue);

fn snapshot_columns(record: &RecordBatch, value_type: ValueType) {
    let names = ["key", "total", "count", "minimum", "maximum"];
    let types = [
        DataType::Int64,
        value_type.data_type(true),
        DataType::Int64,
        value_type.data_type(false),
        value_type.data_type(false),
    ];
    assert_eq!(record.schema().fields().len(), names.len());
    for (index, (name, dtype)) in names.iter().zip(types).enumerate() {
        assert_eq!(record.schema().field(index).name(), *name);
        assert_eq!(record.schema().field(index).data_type(), &dtype);
        assert_eq!(record.schema().field(index).is_nullable(), index != 2);
    }
}

fn validate_snapshot_row(
    record: &RecordBatch,
    row: usize,
    wanted: &Aggregate,
    value_type: ValueType,
) -> SnapshotValues {
    let count = ScalarValue::try_from_array(record.column(2), row).unwrap();
    assert_eq!(count, ScalarValue::Int64(Some(wanted.count)));
    let sum = ScalarValue::try_from_array(record.column(1), row).unwrap();
    let min = ScalarValue::try_from_array(record.column(3), row).unwrap();
    let max = ScalarValue::try_from_array(record.column(4), row).unwrap();
    assert_eq!(
        sum,
        value_type.scalar((wanted.count > 0).then_some(wanted.sum), true)
    );
    assert_eq!(min, value_type.scalar(wanted.min, false));
    assert_eq!(max, value_type.scalar(wanted.max, false));
    (wanted.count, sum, min, max)
}

fn snapshot_values(
    batch: &Batch,
    expected: &BTreeMap<Option<i64>, Aggregate>,
    value_type: ValueType,
) -> BTreeMap<Option<i64>, SnapshotValues> {
    let mut seen = BTreeMap::new();
    for record in batch.table_payload().unwrap().batches() {
        snapshot_columns(record, value_type);
        let keys = record
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for row in 0..record.num_rows() {
            let key = (!keys.is_null(row)).then(|| keys.value(row));
            let wanted = expected.get(&key).expect("no unexpected output group");
            assert!(
                seen.insert(key, validate_snapshot_row(record, row, wanted, value_type))
                    .is_none()
            );
        }
    }
    assert_eq!(seen.len(), expected.len());
    seen
}

fn hash_decimal(
    hash: &mut Sha256,
    bits: u16,
    precision: u8,
    scale: i8,
    present: bool,
    bytes: &[u8],
) {
    hash.update(bits.to_le_bytes());
    hash.update([precision, scale.to_le_bytes()[0], u8::from(present)]);
    hash.update(bytes);
}

fn hash_scalar(hash: &mut Sha256, scalar: &ScalarValue) {
    match scalar {
        ScalarValue::Int64(value) => {
            hash.update([u8::from(value.is_some())]);
            hash.update(value.unwrap_or(0).to_le_bytes());
        }
        ScalarValue::Decimal32(value, p, s) => hash_decimal(
            hash,
            32,
            *p,
            *s,
            value.is_some(),
            &value.unwrap_or(0).to_le_bytes(),
        ),
        ScalarValue::Decimal64(value, p, s) => hash_decimal(
            hash,
            64,
            *p,
            *s,
            value.is_some(),
            &value.unwrap_or(0).to_le_bytes(),
        ),
        ScalarValue::Decimal128(value, p, s) => hash_decimal(
            hash,
            128,
            *p,
            *s,
            value.is_some(),
            &value.unwrap_or(0).to_le_bytes(),
        ),
        ScalarValue::Decimal256(value, p, s) => hash_decimal(
            hash,
            256,
            *p,
            *s,
            value.is_some(),
            &value.unwrap_or(i256::ZERO).to_le_bytes(),
        ),
        _ => panic!("unexpected aggregate scalar"),
    }
}

fn validate_snapshot(
    batch: &Batch,
    expected: &BTreeMap<Option<i64>, Aggregate>,
    value_type: ValueType,
) -> (usize, [u8; 32]) {
    let seen = snapshot_values(batch, expected, value_type);
    let mut hash = Sha256::new();
    for (key, (count, sum, min, max)) in &seen {
        hash_scalar(&mut hash, &ScalarValue::Int64(*key));
        hash.update(count.to_le_bytes());
        for value in [sum, min, max] {
            hash_scalar(&mut hash, value);
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

fn validate_recovery(
    runtime: &tokio::runtime::Runtime,
    case: Case,
    checkpoint: &OperatorStateSnapshot,
    context: &StreamOperatorContext<'_>,
    expected: &mut BTreeMap<Option<i64>, Aggregate>,
    value_type: ValueType,
) -> String {
    let mut hash = Sha256::new();
    let mut restored = operator();
    restored.restore(checkpoint).unwrap();
    let mut recovered = Collector::default();
    runtime
        .block_on(restored.process_data(
            "events",
            input_batch(0, 0, case.unique_keys, value_type),
            context,
            &mut recovered,
        ))
        .unwrap();
    assert_eq!(recovered.snapshots.len(), 1);
    hash.update(validate_snapshot(&recovered.snapshots[0], expected, value_type).1);
    runtime
        .block_on(restored.process_data(
            "events",
            input_batch(case.rows, 1, case.unique_keys, value_type),
            context,
            &mut recovered,
        ))
        .unwrap();
    update(expected, case.rows, 1, case.unique_keys);
    assert_eq!(recovered.snapshots.len(), 2);
    hash.update(validate_snapshot(&recovered.snapshots[1], expected, value_type).1);
    hex::encode(hash.finalize())
}

fn sample(
    runtime: &tokio::runtime::Runtime,
    case: Case,
    inputs: &[Batch],
    recover: bool,
    value_type: ValueType,
) -> Value {
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
            input_batch(0, 0, case.unique_keys, value_type),
            &context,
            &mut collector,
        ))
        .unwrap();
    assert_eq!(collector.snapshots.len(), 1);
    validate_snapshot(&collector.snapshots[0], &BTreeMap::new(), value_type);
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
        let (rows, digest) = validate_snapshot(batch, &expected, value_type);
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
    let recovery_digest = recover.then(|| {
        validate_recovery(
            runtime,
            case,
            &final_checkpoint,
            &context,
            &mut expected,
            value_type,
        )
    });

    let mut observation = json!({"seconds":seconds,"process_seconds":process_seconds,"prepare_seconds":prepare_seconds,"capture_seconds":capture_seconds,"checkpoint_bytes":checkpoint_bytes,"checkpoint_count":checkpoint_count,"final_checkpoint_bytes":segment.bytes().len(),"final_checkpoint_sha256":checkpoint_sha256,"output_rows":output_rows,"snapshot_count":collector.snapshots.len(),"input_rows":case.rows,"all_snapshots_sha256":hex::encode(output_hash.finalize()),"validated_all_snapshots":true,"validated_recovery":recover});
    add_recovery_digest(&mut observation, recovery_digest, value_type);
    observation
}

fn add_recovery_digest(observation: &mut Value, digest: Option<String>, value_type: ValueType) {
    if value_type.bits != 0 {
        observation["recovery_snapshots_sha256"] = digest.map_or(Value::Null, Value::String);
    }
}

fn main() {
    let args = std::env::args_os().collect::<Vec<_>>(); // nosemgrep: args-os
    let value_type = value_type(&args);
    let check = args.iter().any(|arg| arg == "--check" || arg == "--test");
    let samples = args
        .iter()
        .position(|arg| arg == "--samples")
        .map_or(5, |index| {
            args[index + 1]
                .to_str()
                .expect("--samples must be UTF-8")
                .parse::<usize>()
                .unwrap()
        });
    assert!(samples > 0);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let cases = CASES.into_iter().map(|case| {
        let chunk = case.rows / case.batches;
        let inputs = (0..case.rows).step_by(chunk).map(|start| input_batch(start, chunk, case.unique_keys, value_type)).collect::<Vec<_>>();
        let oracle = sample(&runtime, case, &inputs, true, value_type);
        let observations = if check { Vec::new() } else { (0..samples).map(|_| sample(&runtime, case, &inputs, false, value_type)).collect::<Vec<_>>() };
        json!({"name":case.name,"rows":case.rows,"batches":case.batches,"maximum_groups":if case.unique_keys { case.rows - case.rows.div_ceil(101) + 1 } else { 65 },"unique_keys":case.unique_keys,"checkpoint_every":case.checkpoint_every,"query":QUERY,"oracle":oracle,"samples":observations})
    }).collect::<Vec<_>>();
    let mut report = json!({"schema":"calc-flow.sql-stream-aggregate.v1","scope":"warm-native-operator-cumulative-snapshots","cases":cases});
    if value_type.bits != 0 {
        report["schema"] = json!("calc-flow.sql-stream-aggregate.v2");
        report["value_type"] = value_type.descriptor();
    }
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
