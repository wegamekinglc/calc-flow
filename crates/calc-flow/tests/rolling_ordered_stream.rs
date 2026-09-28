//! Finality and recovery parity for the columnar ordered rolling path.

use std::sync::Arc;

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, EventTime, JsonMap,
    OperatorMetadata, RollingOperator, RollingSpec, StreamJobContext, StreamOperator,
    StreamOperatorContext,
};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float64Array, LargeStringArray, StringArray, TimestampMicrosecondArray,
        UInt64Array,
    },
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

const FINGERPRINT: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
    ]))
}

fn operator(drop_late: bool) -> RollingOperator {
    let policy = if drop_late {
        serde_json::json!({"kind": "drop", "metrics_version": 1})
    } else {
        serde_json::json!({"kind": "error", "scope": "envelope"})
    };
    let spec: RollingSpec = serde_json::from_value(serde_json::json!({
        "configuration_version": 1, "state_layout_version": 1,
        "partition_by": ["symbol"], "event_time": "ts", "sequence_by": ["sequence"],
        "outputs": [{"kind": "mean", "primitive_version": 1, "input": "price",
                     "output": "mean", "frame": {"kind": "rows", "size": 2}, "min_periods": 1}],
        "allowed_lateness_micros": 0, "late_policy": policy, "value_policy": "stateful_numeric_v1"
    }))
    .unwrap();
    RollingOperator::new("rolling", schema(), spec).unwrap()
}

fn batch(times: &[i64]) -> Batch {
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC"))
                as ArrayRef,
            Arc::new(StringArray::from(vec!["a"; times.len()])),
            Arc::new(UInt64Array::from(
                times
                    .iter()
                    .map(|&t| u64::try_from(t).unwrap())
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                times
                    .iter()
                    .map(|&t| f64::from(i32::try_from(2 * t - 1).unwrap()))
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

struct Stream {
    operator: RollingOperator,
    job: StreamJobContext,
    output: EdgeCollector,
    watermark: Option<EventTime>,
}

impl Stream {
    fn new(drop_late: bool) -> Self {
        let operator = operator(drop_late);
        let output = EdgeCollector::new(operator.output_ports().to_vec());
        Self {
            operator,
            output,
            watermark: None,
            job: StreamJobContext::new(
                1,
                FINGERPRINT,
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
        }
    }

    async fn push(&mut self, times: &[i64]) -> calc_flow::Result<()> {
        let context = StreamOperatorContext::new(&self.job, "rolling", self.watermark);
        self.operator
            .process_data("input", batch(times), &context, &mut self.output)
            .await
    }

    async fn advance(&mut self, time: i64) -> Vec<f64> {
        let watermark = EventTime::from_micros(time);
        let context = StreamOperatorContext::new(&self.job, "rolling", self.watermark);
        self.operator
            .on_watermark(watermark, &context, &mut self.output)
            .await
            .unwrap();
        self.watermark = Some(watermark);
        self.drain()
    }

    fn drain(&mut self) -> Vec<f64> {
        self.output
            .drain("output")
            .iter()
            .flat_map(|message| {
                message
                    .as_data()
                    .unwrap()
                    .table_payload()
                    .unwrap()
                    .batches()
                    .iter()
                    .flat_map(|record| {
                        record
                            .column(4)
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .unwrap()
                            .values()
                            .to_vec()
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

#[tokio::test]
async fn partial_watermark_and_checkpoint_recovery_preserve_pending_rows() {
    let mut stream = Stream::new(false);
    stream.push(&[1, 2, 3]).await.unwrap();
    assert!(stream.drain().is_empty());
    assert_eq!(stream.advance(1).await, [1.0]);
    let snapshot = stream.operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let mut restored = Stream::new(false);
    restored.operator.restore(&snapshot).unwrap();
    restored.watermark = stream.watermark;
    restored.push(&[4, 5]).await.unwrap();
    assert_eq!(restored.advance(4).await, [2.0, 4.0, 6.0]);
    let context = StreamOperatorContext::new(&restored.job, "rolling", restored.watermark);
    restored
        .operator
        .on_end(&context, &mut restored.output)
        .await
        .unwrap();
    assert_eq!(restored.drain(), [8.0]);
}

#[tokio::test]
async fn out_of_order_arrivals_merge_with_the_ordered_pending_prefix() {
    let mut stream = Stream::new(false);
    stream.push(&[2, 4]).await.unwrap();
    stream.push(&[1, 3]).await.unwrap();
    assert_eq!(stream.advance(4).await, [1.0, 2.0, 4.0, 6.0]);
    stream.push(&[5, 6]).await.unwrap();
    assert_eq!(stream.advance(6).await, [8.0, 10.0]);
}

#[tokio::test]
async fn duplicate_envelope_cannot_install_its_nonduplicate_suffix() {
    let mut stream = Stream::new(false);
    stream.push(&[1, 2]).await.unwrap();
    assert!(
        stream
            .push(&[2, 3])
            .await
            .unwrap_err()
            .to_string()
            .contains("duplicate row identity")
    );
    assert_eq!(stream.advance(2).await, [1.0, 2.0]);
    stream.push(&[3]).await.unwrap();
    assert_eq!(stream.advance(3).await, [4.0]);
}

#[tokio::test]
async fn duplicate_late_rows_are_dropped_before_duplicate_validation() {
    let mut stream = Stream::new(true);
    stream.push(&[1, 2]).await.unwrap();
    assert_eq!(stream.advance(2).await, [1.0, 2.0]);
    stream.push(&[1, 1, 3]).await.unwrap();
    assert_eq!(stream.advance(3).await, [4.0]);
    let snapshot = stream.operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert_eq!(snapshot.inline_metadata["metrics"]["late_rows"], 2);
}

#[tokio::test]
async fn empty_envelopes_and_end_preserve_ordered_state() {
    let mut stream = Stream::new(false);
    stream.push(&[]).await.unwrap();
    stream.push(&[1, 2]).await.unwrap();
    stream.push(&[]).await.unwrap();
    let context = StreamOperatorContext::new(&stream.job, "rolling", None);
    stream
        .operator
        .on_end(&context, &mut stream.output)
        .await
        .unwrap();
    assert_eq!(stream.drain(), [1.0, 2.0]);
}

/// Paths that must agree bit for bit: the direct Utf8 route, the encoded
/// `LargeUtf8` route, and the generic typed kernel forced by a count output.
#[derive(Clone, Copy, Debug)]
enum Route {
    Direct,
    Encoded,
    Generic,
}

fn route_schema(route: Route) -> Arc<Schema> {
    let entity = match route {
        Route::Encoded => DataType::LargeUtf8,
        Route::Direct | Route::Generic => DataType::Utf8,
    };
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("symbol", entity, false),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("price", DataType::Float64, true),
    ]))
}

fn route_operator(route: Route) -> RollingOperator {
    let mean = |size: u64| {
        serde_json::json!({"kind": "mean", "primitive_version": 1, "input": "price",
                           "frame": {"kind": "rows", "size": size}, "min_periods": size})
    };
    let mut outputs = vec![
        serde_json::json!({"kind": "mean", "primitive_version": 1, "input": "price",
                           "output": "slow", "frame": {"kind": "rows", "size": 3},
                           "min_periods": 1}),
        serde_json::json!({"kind": "difference", "primitive_version": 1,
                           "left": mean(2), "right": mean(3), "output": "spread"}),
    ];
    if matches!(route, Route::Generic) {
        outputs.push(serde_json::json!({"kind": "count", "primitive_version": 1,
                                        "input": "price", "output": "count",
                                        "frame": {"kind": "rows", "size": 3},
                                        "min_periods": 1}));
    }
    let spec: RollingSpec = serde_json::from_value(serde_json::json!({
        "configuration_version": 1, "state_layout_version": 1,
        "partition_by": ["symbol"], "event_time": "ts", "sequence_by": ["sequence"],
        "outputs": outputs, "allowed_lateness_micros": 0,
        "late_policy": {"kind": "drop", "metrics_version": 1},
        "value_policy": "stateful_numeric_v1"
    }))
    .unwrap();
    RollingOperator::new("rolling", route_schema(route), spec).unwrap()
}

/// Event time, entity, sequence, price.
type RouteRow = (i64, &'static str, u64, Option<f64>);

fn route_batch(route: Route, rows: &[RouteRow]) -> Batch {
    let entities = rows.iter().map(|row| row.1);
    let entity: ArrayRef = match route {
        Route::Encoded => Arc::new(LargeStringArray::from_iter_values(entities)),
        Route::Direct | Route::Generic => Arc::new(StringArray::from_iter_values(entities)),
    };
    let record = RecordBatch::try_new(
        route_schema(route),
        vec![
            Arc::new(
                TimestampMicrosecondArray::from_iter_values(rows.iter().map(|row| row.0))
                    .with_timezone("UTC"),
            ),
            entity,
            Arc::new(UInt64Array::from_iter_values(rows.iter().map(|row| row.2))),
            Arc::new(rows.iter().map(|row| row.3).collect::<Float64Array>()),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

/// One delivery step: an envelope, then an optional watermark and checkpoint.
#[derive(Clone, Debug)]
struct Step {
    rows: Vec<RouteRow>,
    watermark: Option<i64>,
    checkpoint: bool,
}

/// Derived output bits in emission order, or the first error message.
type RouteOutcome = Result<Vec<[Option<u64>; 2]>, String>;

fn derived_bits(output: &mut EdgeCollector, observed: &mut Vec<[Option<u64>; 2]>) {
    for message in output.drain("output") {
        for record in message
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()
        {
            let column = |index: usize| {
                record
                    .column(index)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .clone()
            };
            let (slow, spread) = (column(4), column(5));
            observed.extend((0..record.num_rows()).map(|row| {
                [&slow, &spread]
                    .map(|values| values.is_valid(row).then(|| values.value(row).to_bits()))
            }));
        }
    }
}

async fn run_route(route: Route, steps: &[Step]) -> RouteOutcome {
    let job = StreamJobContext::new(
        1,
        FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let mut operator = route_operator(route);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let mut observed = Vec::new();
    let mut watermark = None;
    for (index, step) in steps.iter().enumerate() {
        let context = StreamOperatorContext::new(&job, "rolling", watermark);
        operator
            .process_data(
                "input",
                route_batch(route, &step.rows),
                &context,
                &mut output,
            )
            .await
            .map_err(|error| error.to_string())?;
        if let Some(next) = step.watermark.map(EventTime::from_micros) {
            let context = StreamOperatorContext::new(&job, "rolling", watermark);
            operator
                .on_watermark(next, &context, &mut output)
                .await
                .map_err(|error| error.to_string())?;
            watermark = Some(next);
        }
        if step.checkpoint {
            let epoch = Epoch::new(u64::try_from(index).unwrap() + 1).unwrap();
            let snapshot = operator.checkpoint(epoch).unwrap();
            operator = route_operator(route);
            operator.restore(&snapshot).unwrap();
        }
        derived_bits(&mut output, &mut observed);
    }
    let context = StreamOperatorContext::new(&job, "rolling", watermark);
    operator
        .on_end(&context, &mut output)
        .await
        .map_err(|error| error.to_string())?;
    derived_bits(&mut output, &mut observed);
    Ok(observed)
}

fn route_price(seed: u8) -> Option<f64> {
    match seed % 12 {
        0 => None,
        1 => Some(f64::NAN),
        2 => Some(f64::INFINITY),
        3 => Some(f64::NEG_INFINITY),
        4 => Some(1e155),
        5 => Some(-1e155),
        other => Some(f64::from(other) * 0.1 - 0.35),
    }
}

/// Canonical rows with one to three entities per tick.
fn canonical_route_rows(ticks: &[(u8, [u8; 3])]) -> Vec<RouteRow> {
    const ENTITIES: [&str; 3] = ["a", "b", "bb"];
    let slots = ticks
        .iter()
        .enumerate()
        .flat_map(|(tick, &(mask, prices))| {
            let time = i64::try_from(tick).unwrap();
            (0..ENTITIES.len())
                .filter(move |&slot| mask & (1 << slot) != 0 || slot == usize::from(mask % 3))
                .map(move |slot| (time, ENTITIES[slot], route_price(prices[slot])))
        });
    slots
        .enumerate()
        .map(|(sequence, (time, entity, price))| {
            (time, entity, u64::try_from(sequence).unwrap(), price)
        })
        .collect()
}

/// Reverses a segment, replays the preceding row, or duplicates its last row.
fn disorder(mut rows: Vec<RouteRow>, previous: Option<RouteRow>, shape: u8) -> Vec<RouteRow> {
    match (shape % 16, previous, rows.last().copied()) {
        (0, _, _) => rows.reverse(),
        (1, Some(previous), _) => rows.push(previous),
        (2, _, Some(last)) => rows.push(last),
        _ => {}
    }
    rows
}

/// Delivers canonical rows in segments; watermarks advance strictly and
/// never pass an undelivered in-order row.
fn route_steps(ticks: &[(u8, [u8; 3])], cuts: &[(u8, u8, bool)]) -> Vec<Step> {
    let canonical = canonical_route_rows(ticks);
    let mut steps = Vec::new();
    let mut cursor = 0;
    let mut last_watermark = None;
    for &(width, shape, checkpoint) in cuts {
        let end = (cursor + usize::from(width % 5)).min(canonical.len());
        let previous = cursor.checked_sub(1).map(|row| canonical[row]);
        let rows = disorder(canonical[cursor..end].to_vec(), previous, shape);
        cursor = end;
        let watermark = canonical
            .get(cursor)
            .map(|row| row.0 - 1)
            .filter(|&next| shape % 3 != 0 && last_watermark.is_none_or(|last| next > last));
        last_watermark = watermark.or(last_watermark);
        steps.push(Step {
            rows,
            watermark,
            checkpoint,
        });
    }
    steps.push(Step {
        rows: canonical[cursor..].to_vec(),
        watermark: None,
        checkpoint: false,
    });
    steps
}

proptest::proptest! {
    #![proptest_config(proptest::prelude::ProptestConfig {
        cases: 96,
        failure_persistence: None,
        ..proptest::prelude::ProptestConfig::default()
    })]

    #[test]
    fn direct_encoded_and_generic_ordered_routes_agree(
        ticks in proptest::collection::vec((0_u8..8, proptest::array::uniform3(0_u8..=255)), 1..16),
        cuts in proptest::collection::vec((0_u8..=255, 0_u8..=255, proptest::bool::weighted(0.2)), 0..10),
    ) {
        let steps = route_steps(&ticks, &cuts);
        let runtime = tokio::runtime::Builder::new_current_thread().build().unwrap();
        let direct = runtime.block_on(run_route(Route::Direct, &steps));
        proptest::prop_assert_eq!(&direct, &runtime.block_on(run_route(Route::Encoded, &steps)));
        proptest::prop_assert_eq!(&direct, &runtime.block_on(run_route(Route::Generic, &steps)));
    }
}
