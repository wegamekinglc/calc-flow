//! Direct paired pane/legacy evidence with independent values and recovery checks.

use std::{
    collections::BTreeMap,
    error::Error,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use calc_flow::{
    AggregateFunction, Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, JsonMap,
    OperatorMetadata, OperatorStateSnapshot, StreamJobContext, StreamMessage, StreamOperator,
    StreamOperatorContext, WindowAggregateOperator, WindowSpec,
    benchmark_internals::{PhaseClock, process_observations},
};
use datafusion::arrow::{
    array::{Array, Int64Array, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

const ROWS: usize = 100_000;
const WINDOW_MICROS: i64 = 100;
const SLIDE_MICROS: i64 = 10;
const OVERLAP: i64 = WINDOW_MICROS / SLIDE_MICROS;
const FINGERPRINT: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

#[derive(Clone, Copy)]
struct Case {
    groups: usize,
}

impl Case {
    fn id(self) -> String {
        format!(
            "integer-{}-hopping-{WINDOW_MICROS}-{SLIDE_MICROS}",
            self.groups
        )
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("group", DataType::Int64, false),
            Field::new("value", DataType::Int64, true),
        ]))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Expected {
    count: u64,
    minimum: Option<i64>,
    maximum: Option<i64>,
}

struct Observation {
    seconds: f64,
    process_seconds: f64,
    end_seconds: f64,
    input_rows: usize,
    output_rows: usize,
    sha256: String,
    validated_recovery: bool,
    checkpoint_bytes: usize,
    checkpoint_sha256: Option<String>,
    resources: Value,
}

fn window_operator(panes: bool) -> WindowAggregateOperator {
    let spec = WindowSpec::hopping(
        "time",
        Duration::from_micros(u64::try_from(WINDOW_MICROS).unwrap()),
        Duration::from_micros(u64::try_from(SLIDE_MICROS).unwrap()),
    )
    .unwrap()
    .group_by(["group"])
    .unwrap()
    .aggregate(AggregateFunction::Count, "value", "count")
    .unwrap()
    .aggregate(AggregateFunction::Min, "value", "minimum")
    .unwrap()
    .aggregate(AggregateFunction::Max, "value", "maximum")
    .unwrap();
    let mut operator = WindowAggregateOperator::new("window", Case::schema(), spec).unwrap();
    operator.set_pane_execution_for_benchmark(panes);
    operator
}

fn group(case: Case, row: usize) -> i64 {
    i64::try_from(row % case.groups).unwrap()
}

fn value(row: usize) -> Option<i64> {
    (!row.is_multiple_of(17)).then(|| i64::try_from(row % 97).unwrap() - 48)
}

fn input(case: Case) -> Batch {
    let record = RecordBatch::try_new(
        Case::schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![0; ROWS])),
            Arc::new(Int64Array::from_iter_values(
                (0..ROWS).map(|row| group(case, row)),
            )),
            Arc::new((0..ROWS).map(value).collect::<Int64Array>()),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn expected(case: Case) -> BTreeMap<(i64, i64), Expected> {
    let mut groups = BTreeMap::<i64, Expected>::new();
    for row in 0..ROWS {
        let entry = groups.entry(group(case, row)).or_insert(Expected {
            count: 0,
            minimum: None,
            maximum: None,
        });
        if let Some(value) = value(row) {
            entry.count += 1;
            entry.minimum = Some(entry.minimum.map_or(value, |old| old.min(value)));
            entry.maximum = Some(entry.maximum.map_or(value, |old| old.max(value)));
        }
    }
    (1 - OVERLAP..=0)
        .flat_map(|index| {
            let groups = &groups;
            groups
                .iter()
                .map(move |(group, aggregate)| ((index * SLIDE_MICROS, *group), *aggregate))
        })
        .collect()
}

fn integer(record: &RecordBatch, name: &str, row: usize) -> Option<i64> {
    let values = record
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    values.is_valid(row).then(|| values.value(row))
}

fn timestamp(record: &RecordBatch, name: &str, row: usize) -> i64 {
    let values = record
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    assert!(values.is_valid(row));
    values.value(row)
}

fn append_integer(digest: &mut Sha256, value: Option<i64>) {
    digest.update([u8::from(value.is_some())]);
    digest.update(value.unwrap_or_default().to_le_bytes());
}

fn validate(
    messages: Vec<StreamMessage>,
    expected: &BTreeMap<(i64, i64), Expected>,
) -> (usize, String) {
    let mut seen = BTreeMap::new();
    for message in messages {
        let Some(batch) = message.as_data() else {
            continue;
        };
        for record in batch.table_payload().unwrap().batches() {
            for row in 0..record.num_rows() {
                let start = timestamp(record, "window_start", row);
                assert_eq!(timestamp(record, "window_end", row), start + WINDOW_MICROS);
                let key = (
                    start,
                    integer(record, "group", row).expect("non-null group"),
                );
                let counts = record
                    .column_by_name("count")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                assert!(counts.is_valid(row));
                let actual = Expected {
                    count: counts.value(row),
                    minimum: integer(record, "minimum", row),
                    maximum: integer(record, "maximum", row),
                };
                assert_eq!(expected.get(&key), Some(&actual));
                assert!(seen.insert(key, actual).is_none(), "duplicate window group");
            }
        }
    }
    assert_eq!(&seen, expected);
    let rows = seen.len();
    let mut digest = Sha256::new();
    for ((start, group), aggregate) in seen {
        digest.update(start.to_le_bytes());
        digest.update(group.to_le_bytes());
        digest.update(aggregate.count.to_le_bytes());
        append_integer(&mut digest, aggregate.minimum);
        append_integer(&mut digest, aggregate.maximum);
    }
    (rows, hex::encode(digest.finalize()))
}

fn checkpoint_info(snapshot: Option<&OperatorStateSnapshot>) -> (usize, Option<String>) {
    let Some(snapshot) = snapshot else {
        return (0, None);
    };
    let mut bytes = 0;
    let mut digest = Sha256::new();
    for (name, segment) in &snapshot.segments {
        bytes += segment.bytes().len();
        digest.update(name.as_bytes());
        digest.update(segment.bytes());
    }
    (bytes, Some(hex::encode(digest.finalize())))
}

async fn sample(
    input: &Batch,
    expected: &BTreeMap<(i64, i64), Expected>,
    panes: bool,
    recovery: bool,
    deadline: Instant,
) -> Observation {
    let job = StreamJobContext::new(
        1,
        FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut operator = window_operator(panes);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let clock = PhaseClock::start(if recovery {
        "window-input-checkpoint-and-finalize"
    } else {
        "window-input-and-finalize"
    });
    let started = Instant::now();
    operator
        .process_data("input", input.clone(), &context, &mut collector)
        .await
        .unwrap();
    let process_seconds = started.elapsed().as_secs_f64();
    assert!(collector.drain("output").is_empty());
    let snapshot = recovery.then(|| operator.checkpoint(Epoch::INITIAL).unwrap());
    let (checkpoint_bytes, checkpoint_sha256) = checkpoint_info(snapshot.as_ref());
    let started = Instant::now();
    operator.on_end(&context, &mut collector).await.unwrap();
    let end_seconds = started.elapsed().as_secs_f64();
    let resources = clock.finish();
    let (output_rows, sha256) = validate(collector.drain("output"), expected);
    if let Some(snapshot) = snapshot {
        let mut restored = window_operator(panes);
        restored.restore(&snapshot).unwrap();
        let mut restored_output = EdgeCollector::new(restored.output_ports().to_vec());
        restored
            .on_end(&context, &mut restored_output)
            .await
            .unwrap();
        assert_eq!(
            validate(restored_output.drain("output"), expected),
            (output_rows, sha256.clone())
        );
    }
    assert!(Instant::now() < deadline, "measurement deadline exhausted");
    Observation {
        seconds: process_seconds + end_seconds,
        process_seconds,
        end_seconds,
        input_rows: ROWS,
        output_rows,
        sha256,
        validated_recovery: recovery,
        checkpoint_bytes,
        checkpoint_sha256,
        resources,
    }
}

fn compare(left: &Observation, right: &Observation) {
    assert_eq!(left.input_rows, right.input_rows);
    assert_eq!(left.output_rows, right.output_rows);
    assert_eq!(left.sha256, right.sha256);
    assert!(left.seconds.is_finite() && left.seconds > 0.0);
    assert!(right.seconds.is_finite() && right.seconds > 0.0);
    assert_eq!(left.resources["status"], "measured");
    assert_eq!(right.resources["status"], "measured");
}

fn evidence(observation: &Observation, round: usize, pair: usize, panes: bool) -> Value {
    json!({
        "round": round + 1,
        "pair": pair + 1,
        "side": if panes { "candidate" } else { "baseline" },
        "seconds": observation.seconds,
        "process_seconds": observation.process_seconds,
        "end_seconds": observation.end_seconds,
        "input_rows": observation.input_rows,
        "output_rows": observation.output_rows,
        "sha256": observation.sha256,
        "validated_all_rows": true,
        "validated_recovery": observation.validated_recovery,
        "checkpoint_bytes": observation.checkpoint_bytes,
        "checkpoint_sha256": observation.checkpoint_sha256,
        "resources": observation.resources,
    })
}

async fn run_case(case: Case, check: bool, deadline: Instant) -> Value {
    let input = input(case);
    let expected = expected(case);
    if check {
        let legacy = sample(&input, &expected, false, true, deadline).await;
        let panes = sample(&input, &expected, true, true, deadline).await;
        compare(&legacy, &panes);
        return json!({"id":case.id(),"correctness":true,"recovery":true});
    }
    let mut baseline = Vec::new();
    let mut candidate = Vec::new();
    let mut observations = Vec::new();
    for round in 0..2 {
        let legacy = sample(&input, &expected, false, true, deadline).await;
        let panes = sample(&input, &expected, true, true, deadline).await;
        compare(&legacy, &panes);
        let mut legacy_round = Vec::new();
        let mut pane_round = Vec::new();
        for pair in 0..10 {
            let order = if pair % 2 == 0 {
                [(false, &mut legacy_round), (true, &mut pane_round)]
            } else {
                [(true, &mut pane_round), (false, &mut legacy_round)]
            };
            for (panes_enabled, samples) in order {
                let observation = sample(&input, &expected, panes_enabled, false, deadline).await;
                samples.push(observation.seconds);
                observations.push(evidence(&observation, round, pair, panes_enabled));
            }
        }
        baseline.push(legacy_round);
        candidate.push(pane_round);
    }
    json!({
        "id":case.id(),
        "rows":ROWS,
        "groups":case.groups,
        "window_micros":WINDOW_MICROS,
        "slide_micros":SLIDE_MICROS,
        "comparison":"interleaved",
        "correctness":true,
        "baseline":baseline,
        "candidate":candidate,
        "observations":observations,
    })
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let mut check = false;
    let mut output = None;
    let mut arguments = std::env::args_os().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.to_str() {
            Some("--check") => check = true,
            Some("--output") => {
                output = Some(PathBuf::from(arguments.next().ok_or("missing output")?));
            }
            _ => {
                return Err(format!("unknown argument: {}", argument.to_string_lossy()).into());
            }
        }
    }
    let deadline = Instant::now() + Duration::from_secs(600);
    let started = Instant::now();
    let mut cases = Vec::new();
    for groups in [4, 8192] {
        let case = run_case(Case { groups }, check, deadline).await;
        cases.push(case);
    }
    assert!(started.elapsed() < Duration::from_secs(600));
    let report = json!({
        "schema":"calc-flow.window-panes.v1",
        "scope":"operator-input-and-finalization",
        "status":if check {"checked"} else {"completed"},
        "elapsed_seconds":started.elapsed().as_secs_f64(),
        "process_observations":process_observations()?,
        "cases":cases,
    });
    let bytes = serde_json::to_vec_pretty(&report)?;
    if let Some(path) = output {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, bytes)?;
    } else {
        println!("{}", String::from_utf8(bytes)?);
    }
    Ok(())
}
