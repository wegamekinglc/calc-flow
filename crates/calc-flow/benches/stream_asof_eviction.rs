//! Watermark eviction over many keys with sparse and dense expiry.

use async_trait::async_trait;
use calc_flow::{
    AsofJoinSide, AsofStateLimits, Batch, BatchMetadata, CalcFlowError, CancellationToken, Epoch,
    EventTime, IngressProgress, IngressProgressSnapshot, IngressState, JsonMap,
    StreamAsofJoinOperator, StreamAsofJoinSpec, StreamAsofJoinStatus, StreamCollector,
    StreamJobContext, StreamOperator, StreamOperatorContext, StreamingFailureReason,
};
use datafusion::arrow::{
    array::{Array, Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::{self, Write},
    sync::Arc,
    time::{Duration, Instant},
};

const TICKS: usize = 32;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Held,
    Removed,
    None,
    All,
}

impl Mode {
    const fn name(self) -> &'static str {
        match self {
            Self::Held => "sparse_identity_held",
            Self::Removed => "sparse_identity_removed",
            Self::None => "none_expired",
            Self::All => "all_expired",
        }
    }
}

#[derive(Clone, Copy)]
struct Case {
    keys: usize,
    mode: Mode,
}

impl Case {
    fn config(self) -> Value {
        json!({"keys":self.keys,"mode":self.mode.name(),"ticks":self.ticks()})
    }

    const fn ticks(self) -> usize {
        if matches!(self.mode, Mode::All) {
            1
        } else {
            TICKS
        }
    }

    fn frontiers(self, tick: usize) -> (i64, i64) {
        let left = number(if self.mode == Mode::All {
            self.keys
        } else {
            tick
        });
        (left, if self.mode == Mode::Held { 0 } else { left })
    }

    fn evicted(self, tick: usize) -> usize {
        match self.mode {
            Mode::None => 0,
            Mode::All => self.keys,
            Mode::Held | Mode::Removed => tick,
        }
    }

    fn right_time(self, key: usize) -> i64 {
        number(key + if self.mode == Mode::None { 1024 } else { 0 })
    }

    fn probe_time(self, key: usize) -> i64 {
        match self.mode {
            Mode::None => self.right_time(key),
            Mode::All => number(self.keys + key),
            Mode::Held | Mode::Removed => number(key.max(TICKS)),
        }
    }

    fn matched(self, key: usize) -> bool {
        self.mode == Mode::None || self.mode != Mode::All && key >= TICKS
    }
}

fn number(value: usize) -> i64 {
    i64::try_from(value).expect("bounded benchmark integer")
}

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::Int64, false),
        Field::new("value", DataType::Int64, true),
    ]))
}

fn operator() -> StreamAsofJoinOperator {
    let side = |name: &str| {
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["sequence".into()],
            name.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::ZERO,
        AsofStateLimits::new(500_000, 1 << 30).unwrap(),
    )
    .unwrap();
    StreamAsofJoinOperator::new("asof", schema(), schema(), spec).unwrap()
}

fn batch(case: Case, keys: impl Iterator<Item = usize>, left: bool) -> Batch {
    let keys = keys.collect::<Vec<_>>();
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from_iter_values(
                keys.iter().copied().map(number),
            )),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(keys.iter().map(|&key| {
                    if left {
                        case.probe_time(key)
                    } else {
                        case.right_time(key)
                    }
                }))
                .with_timezone("UTC"),
            ),
            Arc::new(Int64Array::from_iter_values(
                keys.iter().copied().map(number),
            )),
            Arc::new(
                keys.iter()
                    .map(|&key| {
                        if left {
                            Some(number(key) * 7)
                        } else {
                            (key % 17 != 0).then_some(number(key) * 3)
                        }
                    })
                    .collect::<Int64Array>(),
            ),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn progress(job: &StreamJobContext, left: i64, right: i64) -> StreamOperatorContext<'_> {
    let snapshot = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(left))),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(right))),
        ),
    ]));
    StreamOperatorContext::with_ingress_progress(
        job,
        "asof",
        Some(EventTime::from_micros(left.min(right))),
        snapshot,
    )
}

#[derive(Default)]
struct Collector {
    output: Vec<Batch>,
}

#[async_trait]
impl StreamCollector for Collector {
    async fn emit(&mut self, _port: &str, batch: Batch) -> calc_flow::Result<()> {
        self.output.push(batch);
        Ok(())
    }
}

fn validate_status(
    case: Case,
    tick: usize,
    status: &StreamAsofJoinStatus,
    previous_bytes: u64,
) -> Value {
    let evicted = case.evicted(tick) as u64;
    let retained = case.keys as u64 - evicted;
    let held = if case.mode == Mode::Held { evicted } else { 0 };
    let (left, right) = case.frontiers(tick);
    assert_eq!(status.retained_right_rows, retained);
    assert_eq!(status.identity_only_rows, held);
    assert_eq!(status.state_rows, retained + held);
    assert_eq!(status.evicted_right_rows, evicted);
    assert_eq!(status.pending_left_rows, 0);
    assert_eq!(status.emitted_left_rows, 0);
    assert_eq!(status.matched_rows, 0);
    assert_eq!(status.unmatched_rows, 0);
    assert_eq!(status.left.accepted_rows, 0);
    assert_eq!(status.right.accepted_rows, case.keys as u64);
    assert_eq!(
        status.left.watermark_micros,
        Some(EventTime::from_micros(left))
    );
    assert_eq!(
        status.right.watermark_micros,
        Some(EventTime::from_micros(right))
    );
    assert_eq!(
        status.output_watermark_micros,
        Some(EventTime::from_micros(left.min(right) - 1))
    );
    assert!(status.state_bytes <= previous_bytes);
    assert_eq!(status.state_bytes != 0, status.state_rows != 0);
    if case.mode == Mode::None {
        assert_eq!(status.state_bytes, previous_bytes);
    }
    json!({"retained_right_rows":retained,"identity_only_rows":held,"state_rows":retained+held,"evicted_right_rows":evicted,"left_watermark":left,"right_watermark":right,"output_watermark":left.min(right)-1,"pending_left_rows":0,"emitted_left_rows":0,"matched_rows":0,"unmatched_rows":0,"right_accepted_rows":case.keys,"left_accepted_rows":0,"state_bytes":status.state_bytes})
}

fn probe_output_schema() -> Schema {
    Schema::new(
        [("left", false), ("right", true)]
            .into_iter()
            .flat_map(|(side, nullable)| {
                schema()
                    .fields()
                    .iter()
                    .map(|field| {
                        field
                            .as_ref()
                            .clone()
                            .with_name(format!("{side}__{}", field.name()))
                            .with_nullable(nullable || field.is_nullable())
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    )
}

fn integer_column(record: &RecordBatch, column: usize) -> &Int64Array {
    record
        .column(column)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
}

fn time_column(record: &RecordBatch, column: usize) -> &TimestampMicrosecondArray {
    record
        .column(column)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap()
}

fn validate_probe_row(case: Case, record: &RecordBatch, row: usize, offset: usize) {
    let key = number(offset);
    let matched = case.matched(offset);
    let integer = |column| integer_column(record, column);
    let time = time_column(record, 1);
    let right_time = time_column(record, 5);
    assert!(integer(0).is_valid(row));
    assert!(time.is_valid(row));
    assert!(integer(2).is_valid(row));
    assert!(integer(3).is_valid(row));
    assert_eq!(integer(0).value(row), key);
    assert_eq!(time.value(row), case.probe_time(offset));
    assert_eq!(integer(2).value(row), key);
    assert_eq!(integer(3).value(row), key * 7);
    assert_eq!(integer(4).is_valid(row), matched);
    assert_eq!(right_time.is_valid(row), matched);
    assert_eq!(integer(6).is_valid(row), matched);
    assert_eq!(integer(7).is_valid(row), matched && offset % 17 != 0);
    if matched {
        assert_eq!(integer(4).value(row), key);
        assert_eq!(right_time.value(row), case.right_time(offset));
        assert_eq!(integer(6).value(row), key);
        if offset % 17 != 0 {
            assert_eq!(integer(7).value(row), key * 3);
        }
    }
}

fn digest_nullable_integer(digest: &mut Sha256, values: &Int64Array, row: usize) {
    digest.update([u8::from(values.is_valid(row))]);
    digest.update(
        if values.is_valid(row) {
            values.value(row)
        } else {
            0
        }
        .to_le_bytes(),
    );
}

fn digest_probe_row(digest: &mut Sha256, record: &RecordBatch, row: usize) {
    digest.update(integer_column(record, 0).value(row).to_le_bytes());
    digest.update(time_column(record, 1).value(row).to_le_bytes());
    digest_nullable_integer(digest, integer_column(record, 6), row);
    digest_nullable_integer(digest, integer_column(record, 7), row);
}

fn validate_probes(case: Case, collector: &Collector) -> String {
    let mut digest = Sha256::new();
    let mut rows = 0;
    let expected_schema = probe_output_schema();
    for output in &collector.output {
        for record in output.table_payload().unwrap().batches() {
            assert_eq!(record.schema().as_ref(), &expected_schema);
            assert_eq!(record.num_columns(), 8);
            for row in 0..record.num_rows() {
                validate_probe_row(case, record, row, rows);
                digest_probe_row(&mut digest, record, row);
                rows += 1;
            }
        }
    }
    assert_eq!(rows, case.keys);
    hex::encode(digest.finalize())
}

fn sample(
    runtime: &tokio::runtime::Runtime,
    case: Case,
    input_batch: &Batch,
    recover: bool,
) -> Value {
    let job = StreamJobContext::new(
        1,
        "asof-eviction",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let input = StreamOperatorContext::new(&job, "asof", None);
    let mut op = operator();
    let mut output = Collector::default();
    runtime
        .block_on(op.process_data("right", input_batch.clone(), &input, &mut output))
        .unwrap();
    let before_state_bytes = op.status().state_bytes;
    assert!(before_state_bytes > 0);
    let contexts = (1..=case.ticks())
        .map(|tick| {
            let (left, right) = case.frontiers(tick);
            progress(&job, left, right)
        })
        .collect::<Vec<_>>();
    let mut tick_seconds = Vec::with_capacity(case.ticks());
    let mut statuses = Vec::with_capacity(case.ticks());
    let mut previous_bytes = before_state_bytes;
    for (tick, context) in contexts.iter().enumerate() {
        let (left, right) = case.frontiers(tick + 1);
        let started = Instant::now();
        runtime
            .block_on(op.on_watermark(
                EventTime::from_micros(left.min(right)),
                context,
                &mut output,
            ))
            .unwrap();
        tick_seconds.push(started.elapsed().as_secs_f64());
        let status = op.status();
        statuses.push(validate_status(case, tick + 1, &status, previous_bytes));
        previous_bytes = status.state_bytes;
    }
    assert!(output.output.is_empty());
    let status = op.status();
    runtime
        .block_on(op.prepare_checkpoint_async(&input))
        .unwrap();
    let snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
    let mut checkpoint_digest = Sha256::new();
    let mut checkpoint_bytes = 0;
    for (name, segment) in &snapshot.segments {
        checkpoint_digest.update((name.len() as u64).to_le_bytes());
        checkpoint_digest.update(name.as_bytes());
        checkpoint_digest.update((segment.bytes().len() as u64).to_le_bytes());
        checkpoint_digest.update(segment.bytes());
        checkpoint_bytes += segment.bytes().len();
    }
    let mut result = json!({"seconds":tick_seconds.iter().sum::<f64>(),"tick_seconds":tick_seconds,"statuses":statuses,"before_state_bytes":before_state_bytes,"output_rows":0,"checkpoint_bytes":checkpoint_bytes,"checkpoint_sha256":hex::encode(checkpoint_digest.finalize()),"validated_status":true,"validated_recovery":recover});
    if recover {
        let mut restored = operator();
        restored.restore(&snapshot).unwrap();
        let recovered = restored.status();
        assert_eq!(recovered.state_bytes, status.state_bytes);
        assert_eq!(recovered.state_rows, status.state_rows);
        assert_eq!(recovered.retained_right_rows, status.retained_right_rows);
        assert_eq!(recovered.identity_only_rows, status.identity_only_rows);
        assert_eq!(
            restored.checkpoint(Epoch::INITIAL).unwrap().segments,
            snapshot.segments
        );
        let last = contexts.last().unwrap();
        if case.mode == Mode::Held {
            let duplicate = batch(case, std::iter::once(0), false);
            let error = runtime
                .block_on(restored.process_data("right", duplicate, last, &mut output))
                .unwrap_err();
            assert!(matches!(
                error,
                CalcFlowError::OperatorReason {
                    reason_code: StreamingFailureReason::AsofDuplicateIdentity,
                    ..
                }
            ));
            assert_eq!(restored.status().right.duplicate_rows, 1);
        }
        let probes = batch(case, 0..case.keys, true);
        runtime
            .block_on(restored.process_data("left", probes, last, &mut output))
            .unwrap();
        runtime
            .block_on(restored.on_end(last, &mut output))
            .unwrap();
        let probe_sha256 = validate_probes(case, &output);
        assert_eq!(restored.status().state_rows, 0);
        assert_eq!(restored.status().state_bytes, 0);
        result["restored_state_rows"] = recovered.state_rows.into();
        result["restored_state_bytes"] = recovered.state_bytes.into();
        result["probe_rows"] = case.keys.into();
        result["probe_sha256"] = probe_sha256.into();
        result["identity_duplicate_rejected"] = (case.mode == Mode::Held).into();
    }
    result
}

fn main() {
    let args = std::env::args_os().collect::<Vec<_>>();
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
    let mut cases = Vec::new();
    for keys in [4096, 65536] {
        for mode in [Mode::Held, Mode::Removed, Mode::None, Mode::All] {
            let case = Case { keys, mode };
            let input = batch(case, 0..keys, false);
            let oracle = sample(&runtime, case, &input, true);
            let observations = if check {
                Vec::new()
            } else {
                (0..samples)
                    .map(|_| sample(&runtime, case, &input, false))
                    .collect::<Vec<_>>()
            };
            cases.push(json!({"name":format!("{}_{keys}",mode.name()),"config":case.config(),"oracle":oracle,"samples":observations}));
        }
    }
    let report = json!({"schema":"calc-flow.asof-eviction.v1","scope":"operator-watermark-eviction","cases":cases});
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
