//! Window aggregation with independent full-result and recovery checks.

use std::{
    collections::BTreeMap,
    io::{self, Write},
    sync::Arc,
    time::{Duration, Instant},
};

use calc_flow::{
    AggregateFunction, Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, JsonMap,
    OperatorMetadata, OperatorStateSnapshot, StreamJobContext, StreamMessage, StreamOperator,
    StreamOperatorContext, WindowAggregateOperator, WindowSpec,
};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray,
        UInt64Array,
    },
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const ROWS: usize = 100_000;
const FINGERPRINT: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[derive(Clone, Copy)]
enum Keys {
    Integer,
    String,
    Composite,
}

impl Keys {
    fn name(self) -> &'static str {
        match self {
            Self::Integer => "integer",
            Self::String => "string",
            Self::Composite => "composite",
        }
    }

    fn group_type(self) -> DataType {
        match self {
            Self::String => DataType::Utf8,
            _ => DataType::Int64,
        }
    }
}

#[derive(Clone, Copy)]
struct Case {
    keys: Keys,
    groups: usize,
    hopping: bool,
    multiple: bool,
}

impl Case {
    fn name(self) -> String {
        format!(
            "{}_{}_{}_{}",
            self.keys.name(),
            self.groups,
            if self.hopping { "hopping" } else { "tumbling" },
            if self.multiple { "five" } else { "sum" }
        )
    }

    fn config(self) -> Value {
        json!({"rows":ROWS,"groups":self.groups,"keys":self.keys.name(),"hopping":self.hopping,"multiple":self.multiple})
    }

    fn schema(self) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("group", self.keys.group_type(), true),
            Field::new("partition", DataType::Int64, false),
            Field::new("value", DataType::Int64, true),
        ]))
    }

    fn spec(self) -> WindowSpec {
        let spec = if self.hopping {
            WindowSpec::hopping("time", Duration::from_micros(20), Duration::from_micros(10))
                .unwrap()
        } else {
            WindowSpec::tumbling("time", Duration::from_micros(10)).unwrap()
        };
        let groups = if matches!(self.keys, Keys::Composite) {
            vec!["group", "partition"]
        } else {
            vec!["group"]
        };
        let mut spec = spec
            .group_by(groups)
            .unwrap()
            .aggregate(AggregateFunction::Sum, "value", "sum")
            .unwrap();
        if self.multiple {
            for (function, name) in [
                (AggregateFunction::Count, "count"),
                (AggregateFunction::Min, "min"),
                (AggregateFunction::Max, "max"),
                (AggregateFunction::Avg, "avg"),
            ] {
                spec = spec.aggregate(function, "value", name).unwrap();
            }
        }
        spec
    }

    fn operator(self) -> WindowAggregateOperator {
        WindowAggregateOperator::new("window", self.schema(), self.spec()).unwrap()
    }
}

fn group(case: Case, row: usize) -> Option<i64> {
    let key = row % case.groups;
    (key != 0).then(|| i64::try_from(key).unwrap())
}

fn value(case: Case, row: usize) -> Option<i64> {
    (row % case.groups != 0 && row % 17 != 0).then(|| i64::try_from(row % 97).unwrap() - 48)
}

fn input(case: Case) -> Batch {
    let groups: ArrayRef = match case.keys {
        Keys::String => Arc::new(
            (0..ROWS)
                .map(|row| group(case, row).map(|key| format!("key{key:06}")))
                .collect::<StringArray>(),
        ),
        _ => Arc::new(
            (0..ROWS)
                .map(|row| group(case, row))
                .collect::<Int64Array>(),
        ),
    };
    let record = RecordBatch::try_new(
        case.schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![0; ROWS])),
            groups,
            Arc::new(Int64Array::from(vec![7; ROWS])),
            Arc::new(
                (0..ROWS)
                    .map(|row| value(case, row))
                    .collect::<Int64Array>(),
            ),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

#[derive(Default)]
struct Expected {
    sum: i64,
    count: u64,
    min: Option<i64>,
    max: Option<i64>,
}

fn expected(case: Case) -> BTreeMap<Option<i64>, Expected> {
    let mut groups = BTreeMap::<_, Expected>::new();
    for row in 0..ROWS {
        let current = groups.entry(group(case, row)).or_default();
        if let Some(value) = value(case, row) {
            current.sum += value;
            current.count += 1;
            current.min = Some(current.min.map_or(value, |previous| previous.min(value)));
            current.max = Some(current.max.map_or(value, |previous| previous.max(value)));
        }
    }
    groups
}

fn output_schema(case: Case) -> Schema {
    let timestamp = DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC")));
    let mut fields = vec![
        Field::new("window_start", timestamp.clone(), false),
        Field::new("window_end", timestamp, false),
        Field::new("group", case.keys.group_type(), true),
    ];
    if matches!(case.keys, Keys::Composite) {
        fields.push(Field::new("partition", DataType::Int64, false));
    }
    fields.push(Field::new("sum", DataType::Int64, true));
    if case.multiple {
        fields.extend([
            Field::new("count", DataType::UInt64, false),
            Field::new("min", DataType::Int64, true),
            Field::new("max", DataType::Int64, true),
            Field::new("avg", DataType::Float64, true),
        ]);
    }
    Schema::new(fields)
}

fn integer(record: &RecordBatch, name: &str, row: usize) -> Option<i64> {
    let column = record
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    (!column.is_null(row)).then(|| column.value(row))
}

fn result_group(case: Case, record: &RecordBatch, row: usize) -> Option<i64> {
    if !matches!(case.keys, Keys::String) {
        return integer(record, "group", row);
    }
    let column = record
        .column_by_name("group")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    (!column.is_null(row)).then(|| {
        let key = column
            .value(row)
            .strip_prefix("key")
            .unwrap()
            .parse::<i64>()
            .unwrap();
        assert_eq!(column.value(row), format!("key{key:06}"));
        key
    })
}

fn validate_multiple(record: &RecordBatch, row: usize, expected: &Expected) {
    let count = record
        .column_by_name("count")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert!(!count.is_null(row));
    assert_eq!(count.value(row), expected.count);
    assert_eq!(integer(record, "min", row), expected.min);
    assert_eq!(integer(record, "max", row), expected.max);
    let avg = record
        .column_by_name("avg")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(avg.is_null(row), expected.count == 0);
    if expected.count != 0 {
        assert_eq!(
            avg.value(row).to_bits(),
            (f64::from(i32::try_from(expected.sum).unwrap())
                / f64::from(u32::try_from(expected.count).unwrap()))
            .to_bits()
        );
    }
}

fn timestamp(record: &RecordBatch, name: &str, row: usize) -> i64 {
    let column = record
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    assert!(!column.is_null(row));
    column.value(row)
}

fn append_integer(bytes: &mut Vec<u8>, value: Option<i64>) {
    bytes.push(u8::from(value.is_some()));
    bytes.extend_from_slice(&value.unwrap_or_default().to_le_bytes());
}

fn append_multiple(bytes: &mut Vec<u8>, record: &RecordBatch, row: usize) {
    let count = record
        .column_by_name("count")
        .unwrap()
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    bytes.extend_from_slice(&count.value(row).to_le_bytes());
    append_integer(bytes, integer(record, "min", row));
    append_integer(bytes, integer(record, "max", row));
    let average = record
        .column_by_name("avg")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    bytes.push(u8::from(!average.is_null(row)));
    bytes.extend_from_slice(
        &if average.is_null(row) {
            0
        } else {
            average.value(row).to_bits()
        }
        .to_le_bytes(),
    );
}

fn row_bytes(case: Case, record: &RecordBatch, row: usize) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&timestamp(record, "window_start", row).to_le_bytes());
    bytes.extend_from_slice(&timestamp(record, "window_end", row).to_le_bytes());
    append_integer(&mut bytes, result_group(case, record, row));
    if matches!(case.keys, Keys::Composite) {
        bytes.extend_from_slice(&integer(record, "partition", row).unwrap().to_le_bytes());
    }
    append_integer(&mut bytes, integer(record, "sum", row));
    if case.multiple {
        append_multiple(&mut bytes, record, row);
    }
    bytes
}

fn validate_record(
    case: Case,
    record: &RecordBatch,
    oracle: &BTreeMap<Option<i64>, Expected>,
    seen: &mut BTreeMap<(i64, Option<i64>), Vec<u8>>,
) {
    assert_eq!(record.schema().as_ref(), &output_schema(case));
    for row in 0..record.num_rows() {
        let start = timestamp(record, "window_start", row);
        assert!(start == 0 || case.hopping && start == -10);
        assert_eq!(
            timestamp(record, "window_end", row),
            start + if case.hopping { 20 } else { 10 }
        );
        let key = result_group(case, record, row);
        let expected = oracle.get(&key).expect("known group");
        assert_eq!(
            integer(record, "sum", row),
            (expected.count != 0).then_some(expected.sum)
        );
        if matches!(case.keys, Keys::Composite) {
            assert_eq!(integer(record, "partition", row), Some(7));
        }
        if case.multiple {
            validate_multiple(record, row, expected);
        }
        assert!(
            seen.insert((start, key), row_bytes(case, record, row))
                .is_none(),
            "unique window group"
        );
    }
}

fn validate(
    case: Case,
    messages: Vec<StreamMessage>,
    oracle: &BTreeMap<Option<i64>, Expected>,
) -> String {
    let mut seen = BTreeMap::new();
    for message in messages {
        if let Some(batch) = message.as_data() {
            for record in batch.table_payload().unwrap().batches() {
                validate_record(case, record, oracle, &mut seen);
            }
        }
    }
    assert_eq!(seen.len(), case.groups * if case.hopping { 2 } else { 1 });
    let mut digest = Sha256::new();
    for bytes in seen.values() {
        digest.update(bytes);
    }
    hex::encode(digest.finalize())
}

fn checkpoint_info(snapshot: Option<&OperatorStateSnapshot>) -> (usize, Option<String>) {
    let Some(snapshot) = snapshot else {
        return (0, None);
    };
    let mut digest = Sha256::new();
    let mut bytes = 0;
    for (name, segment) in &snapshot.segments {
        digest.update(u64::try_from(name.len()).unwrap().to_le_bytes());
        digest.update(name.as_bytes());
        digest.update(u64::try_from(segment.bytes().len()).unwrap().to_le_bytes());
        digest.update(segment.bytes());
        bytes += segment.bytes().len();
    }
    (bytes, Some(hex::encode(digest.finalize())))
}

fn sample(
    runtime: &tokio::runtime::Runtime,
    case: Case,
    input: &Batch,
    oracle: &BTreeMap<Option<i64>, Expected>,
    recovery: bool,
) -> Value {
    let job = StreamJobContext::new(
        1,
        FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut operator = case.operator();
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let start = Instant::now();
    runtime
        .block_on(operator.process_data("input", input.clone(), &context, &mut collector))
        .unwrap();
    let process_seconds = start.elapsed().as_secs_f64();
    assert!(collector.drain("output").is_empty());
    let snapshot = if recovery {
        Some(operator.checkpoint(Epoch::INITIAL).unwrap())
    } else {
        None
    };
    let (checkpoint_bytes, checkpoint_sha256) = checkpoint_info(snapshot.as_ref());
    let start = Instant::now();
    runtime
        .block_on(operator.on_end(&context, &mut collector))
        .unwrap();
    let end_seconds = start.elapsed().as_secs_f64();
    let digest = validate(case, collector.drain("output"), oracle);
    if let Some(snapshot) = snapshot {
        let mut restored = case.operator();
        restored.restore(&snapshot).unwrap();
        let mut collector = EdgeCollector::new(restored.output_ports().to_vec());
        runtime
            .block_on(restored.on_end(&context, &mut collector))
            .unwrap();
        assert_eq!(validate(case, collector.drain("output"), oracle), digest);
    }
    json!({"seconds":process_seconds+end_seconds,"process_seconds":process_seconds,"end_seconds":end_seconds,"checkpoint_bytes":checkpoint_bytes,"checkpoint_sha256":checkpoint_sha256,"process_output_rows":0,"input_rows":ROWS,"output_rows":case.groups * if case.hopping {2} else {1},"sha256":digest,"validated_all_rows":true,"validated_recovery":recovery})
}

fn check_group_strings() {
    let case = Case {
        keys: Keys::String,
        groups: 4,
        hopping: false,
        multiple: false,
    };
    let schema = Arc::new(Schema::new(vec![Field::new("group", DataType::Utf8, true)]));
    let valid = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StringArray::from(vec![Some("key000001"), None]))],
    )
    .unwrap();
    assert_eq!(result_group(case, &valid, 0), Some(1));
    assert_eq!(result_group(case, &valid, 1), None);
    for spelling in ["key1", "key+1"] {
        let record = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(StringArray::from(vec![Some(spelling)]))],
        )
        .unwrap();
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                result_group(case, &record, 0)
            }))
            .is_err(),
            "reject noncanonical string key {spelling}"
        );
    }
}

fn report(runtime: &tokio::runtime::Runtime, check: bool, samples: usize) -> Value {
    if check {
        check_group_strings();
    }
    let mut cases = Vec::new();
    for keys in [Keys::Integer, Keys::String, Keys::Composite] {
        for groups in [4, 8192] {
            for hopping in [false, true] {
                for multiple in [false, true] {
                    let case = Case {
                        keys,
                        groups,
                        hopping,
                        multiple,
                    };
                    let input = input(case);
                    let expected = expected(case);
                    let oracle = sample(runtime, case, &input, &expected, true);
                    let observations = if check {
                        Vec::new()
                    } else {
                        (0..samples)
                            .map(|_| sample(runtime, case, &input, &expected, false))
                            .collect::<Vec<_>>()
                    };
                    cases.push(json!({"name":case.name(),"config":case.config(),"oracle":oracle,"samples":observations}));
                }
            }
        }
    }
    json!({"schema":"calc-flow.window-groups.v1","scope":"operator-input-and-finalization","cases":cases})
}

fn main() {
    let args = std::env::args_os().collect::<Vec<_>>(); // nosemgrep: args-os
    let check = args.iter().any(|arg| arg == "--check" || arg == "--test");
    let samples = args
        .iter()
        .position(|arg| arg == "--samples")
        .map_or(20, |index| {
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
    let bytes = serde_json::to_vec_pretty(&report(&runtime, check, samples)).unwrap();
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
