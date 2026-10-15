use std::{collections::BTreeMap, fmt, sync::Arc};

#[cfg(test)]
use datafusion::execution::memory_pool::GreedyMemoryPool;
use datafusion::{
    arrow::{
        array::{
            Array, ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array,
        },
        datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
        record_batch::RecordBatch,
    },
    execution::memory_pool::{MemoryLimit, MemoryPool, MemoryReservation, UnboundedMemoryPool},
};
use parking_lot::Mutex;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{JoinStateLimits, JoinTimeBounds, StoredRow, StreamJoinOperator, StreamJoinSpec};
use crate::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, Epoch, EventTime, IngressProgress,
    IngressProgressSnapshot, IngressState, JsonMap, OperatorMetadata, OperatorStateSnapshot,
    Result, StreamJobContext, StreamOperator, StreamOperatorContext,
};

const BASE_MICROS: i64 = 1_767_225_600_000_000;
const FULL_ROWS: usize = 4_000_000;
const ENTITIES: usize = 64;
const BATCH_ROWS: usize = 64_000;
const POOL_BYTES: usize = 1 << 30;
const OPERATOR: &str = "cf_stream_join_70034c097d851384";

#[derive(Clone, Debug, Default)]
struct Counters {
    peak: usize,
    phase_peak: usize,
    grows: u64,
    shrinks: u64,
    rejected: u64,
    last_rejection: Option<Rejection>,
}

#[derive(Clone, Debug, Serialize)]
struct Rejection {
    additional_bytes: usize,
    reserved_bytes: usize,
    limit_bytes: Option<usize>,
    consumer_name: String,
}

#[derive(Debug)]
struct ObservedPool {
    id: &'static str,
    limit: Option<usize>,
    pool: Arc<dyn MemoryPool>,
    counters: Mutex<Counters>,
}

#[derive(Debug, Serialize)]
struct PoolSnapshot {
    pool_id: &'static str,
    limit_bytes: Option<usize>,
    reserved_bytes: usize,
    peak_reserved_bytes: usize,
    phase_peak_reserved_bytes: usize,
    successful_grows: u64,
    successful_shrinks: u64,
    failed_grows: u64,
    last_rejection: Option<Rejection>,
}

impl ObservedPool {
    #[cfg(test)]
    fn new(id: &'static str, limit: usize) -> Self {
        Self {
            id,
            limit: Some(limit),
            pool: Arc::new(GreedyMemoryPool::new(limit)),
            counters: Mutex::default(),
        }
    }

    fn unbounded(id: &'static str) -> Self {
        Self {
            id,
            limit: None,
            pool: Arc::new(UnboundedMemoryPool::default()),
            counters: Mutex::default(),
        }
    }

    fn begin_phase(&self) {
        let mut counters = self.counters.lock();
        counters.phase_peak = self.pool.reserved();
    }

    fn snapshot(&self) -> PoolSnapshot {
        let counters = self.counters.lock();
        PoolSnapshot {
            pool_id: self.id,
            limit_bytes: self.limit,
            reserved_bytes: self.pool.reserved(),
            peak_reserved_bytes: counters.peak,
            phase_peak_reserved_bytes: counters.phase_peak,
            successful_grows: counters.grows,
            successful_shrinks: counters.shrinks,
            failed_grows: counters.rejected,
            last_rejection: counters.last_rejection.clone(),
        }
    }

    fn record_growth(&self, counters: &mut Counters) {
        counters.grows += 1;
        let reserved = self.pool.reserved();
        counters.peak = counters.peak.max(reserved);
        counters.phase_peak = counters.phase_peak.max(reserved);
    }
}

impl fmt::Display for ObservedPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "Join checkpoint reservation observer {}",
            self.id
        )
    }
}

impl MemoryPool for ObservedPool {
    fn name(&self) -> &str {
        self.pool.name()
    }
    fn memory_limit(&self) -> MemoryLimit {
        self.pool.memory_limit()
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        let mut counters = self.counters.lock();
        self.pool.grow(reservation, additional);
        self.record_growth(&mut counters);
    }

    fn shrink(&self, reservation: &MemoryReservation, bytes: usize) {
        let mut counters = self.counters.lock();
        self.pool.shrink(reservation, bytes);
        counters.shrinks += 1;
    }

    fn try_grow(
        &self,
        reservation: &MemoryReservation,
        additional: usize,
    ) -> datafusion::common::Result<()> {
        let mut counters = self.counters.lock();
        match self.pool.try_grow(reservation, additional) {
            Ok(()) => {
                self.record_growth(&mut counters);
                Ok(())
            }
            Err(error) => {
                counters.rejected += 1;
                counters.last_rejection = Some(Rejection {
                    additional_bytes: additional,
                    reserved_bytes: self.pool.reserved(),
                    limit_bytes: self.limit,
                    consumer_name: reservation.consumer().name().chars().take(256).collect(),
                });
                Err(error)
            }
        }
    }

    fn reserved(&self) -> usize {
        let _guard = self.counters.lock();
        self.pool.reserved()
    }
}

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(crate::benchmark_internals::invalid(message))
    }
}

fn schemas() -> [SchemaRef; 2] {
    let time = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()));
    [
        Arc::new(Schema::new(vec![
            Field::new("event_time", time.clone(), false),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
        ])),
        Arc::new(Schema::new(vec![
            Field::new("symbol", DataType::Utf8, false),
            Field::new("factor", DataType::Float64, false),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("event_time", time, false),
        ])),
    ]
}

fn quote_time(sequence: usize) -> i64 {
    BASE_MICROS + i64::try_from(sequence / ENTITIES).expect("bounded fixture") * 1_000_000
}

fn price(sequence: usize) -> f64 {
    100.0 + fixture_number(sequence % 257) / 8.0 + fixture_number(sequence % ENTITIES) / 8.0
}

fn fixture_number(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("bounded fixture value"))
}

fn fixture_operator(rows: usize) -> Result<StreamJoinOperator> {
    require(
        (2_560..=FULL_ROWS).contains(&rows) && rows % ENTITIES == 0,
        "checkpoint fixture must preserve the maintained 64-key domain",
    )?;
    let capacity = (ENTITIES + rows.max(BATCH_ROWS)) as u64;
    let span =
        u64::try_from(quote_time(rows - 1) - BASE_MICROS + 1).expect("positive fixture span");
    let spec = StreamJoinSpec::inner(
        ["symbol"],
        ["symbol"],
        "event_time",
        "event_time",
        JoinTimeBounds::from_micros(span, 0)?,
        JoinStateLimits::new(
            capacity,
            (capacity << 10).max(POOL_BYTES as u64),
            (BATCH_ROWS + ENTITIES) as u64,
        )?,
    )?
    .with_prefixes("quote", "dimension")?;
    let [left, right] = schemas();
    StreamJoinOperator::new(OPERATOR, left, right, spec)
}

fn fixture_record(left: bool, start: usize, count: usize) -> Result<RecordBatch> {
    let indices = start..start + count;
    let symbols = (0..ENTITIES)
        .map(|key| format!("S{key:03}"))
        .collect::<Vec<_>>();
    let keys: ArrayRef = Arc::new(StringArray::from_iter_values(
        indices
            .clone()
            .map(|index| symbols[index % ENTITIES].as_str()),
    ));
    let sequence: ArrayRef = Arc::new(UInt64Array::from_iter_values(
        indices.clone().map(|index| index as u64),
    ));
    let time: ArrayRef = Arc::new(
        TimestampMicrosecondArray::from_iter_values(
            indices
                .clone()
                .map(|index| if left { quote_time(index) } else { BASE_MICROS }),
        )
        .with_timezone("UTC"),
    );
    let value: ArrayRef = Arc::new(Float64Array::from_iter_values(indices.map(|index| {
        if left {
            price(index)
        } else {
            fixture_number(index + 1)
        }
    })));
    let [left_schema, right_schema] = schemas();
    RecordBatch::try_new(
        if left { left_schema } else { right_schema },
        if left {
            vec![time, sequence, keys, value]
        } else {
            vec![keys, value, sequence, time]
        },
    )
    .map_err(|error| crate::benchmark_internals::invalid(error.to_string()))
}

fn fixture_description() -> Value {
    let schemas = schemas().map(|schema| {
        schema
            .fields()
            .iter()
            .map(|field| {
                let kind = match field.data_type() {
                    DataType::Timestamp(TimeUnit::Microsecond, Some(zone))
                        if zone.as_ref() == "UTC" =>
                    {
                        "timestamp_us_utc"
                    }
                    DataType::UInt64 => "uint64",
                    DataType::Utf8 => "string",
                    DataType::Float64 => "float64",
                    _ => unreachable!("fixed row3 schema"),
                };
                json!([field.name(), kind, field.is_nullable()])
            })
            .collect::<Vec<_>>()
    });
    json!({"entities": ENTITIES, "base_micros": BASE_MICROS,
        "left_schema": schemas[0], "right_schema": schemas[1]})
}

fn typed<T: 'static>(array: &ArrayRef) -> Result<&T> {
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| crate::benchmark_internals::invalid("row3 oracle Arrow type differs"))
}

fn row_values(columns: &[ArrayRef], row: usize, left: bool) -> Result<(i64, u64, &str, f64)> {
    require(
        columns.len() == 4 && columns.iter().all(|column| !column.is_null(row)),
        "row3 oracle column count or nullability differs",
    )?;
    let [time, sequence, key, value] = if left { [0, 1, 2, 3] } else { [3, 2, 0, 1] };
    Ok((
        typed::<TimestampMicrosecondArray>(&columns[time])?.value(row),
        typed::<UInt64Array>(&columns[sequence])?.value(row),
        typed::<StringArray>(&columns[key])?.value(row),
        typed::<Float64Array>(&columns[value])?.value(row),
    ))
}

fn update_digest(digest: &mut Sha256, time: i64, sequence: u64, key: &str, value: f64) {
    digest.update(time.to_le_bytes());
    digest.update(sequence.to_le_bytes());
    digest.update(key.as_bytes());
    digest.update(value.to_le_bytes());
}

fn retained_identity_matches(
    stored: &StoredRow,
    index: usize,
    values: (i64, u64, &str, f64),
    expected_time: i64,
    expected_value: f64,
    expected_key: &str,
    encoded: &[u8],
) -> bool {
    let (time, sequence, key, value) = values;
    sequence == index as u64
        && stored.row_id == sequence
        && time == expected_time
        && stored.event_time.as_micros() == time
        && key == expected_key
        && value.to_bits() == expected_value.to_bits()
        && stored.encoded_key.as_slice() == encoded
}

fn validate_retained_row<'a>(
    stored: &'a StoredRow,
    index: usize,
    left: bool,
    schema: &SchemaRef,
    keys: &[String],
    encoded: &[Vec<u8>],
) -> Result<(i64, u64, &'a str, f64)> {
    require(
        stored.record.schema_ref() == schema,
        "row3 oracle retained schema differs",
    )?;
    let values = row_values(stored.record.columns(), stored.record.offset(), left)?;
    let expected_time = if left { quote_time(index) } else { BASE_MICROS };
    let expected_value = if left {
        price(index)
    } else {
        fixture_number(index + 1)
    };
    require(
        retained_identity_matches(
            stored,
            index,
            values,
            expected_time,
            expected_value,
            &keys[index % ENTITIES],
            &encoded[index % ENTITIES],
        ),
        "row3 full-state identity, key, time, or value differs",
    )?;
    Ok(values)
}

fn validate_side(rows: &dyn super::Rows, expected: usize, left: bool) -> Result<(String, u64)> {
    require(
        rows.len() == expected,
        "row3 oracle retained row count differs",
    )?;
    let dimension = fixture_record(false, 0, ENTITIES)?;
    let encoded = (0..ENTITIES)
        .map(|index| super::encode_join_key_columns_v1(dimension.columns(), index, &[0]))
        .collect::<Result<Vec<_>>>()?;
    let keys = (0..ENTITIES)
        .map(|key| format!("S{key:03}"))
        .collect::<Vec<_>>();
    let schema = schemas()[usize::from(!left)].clone();
    let mut digest = Sha256::new();
    let mut charged = 0;
    for (index, stored) in rows.iter().enumerate() {
        let (time, sequence, key, value) =
            validate_retained_row(&stored, index, left, &schema, &keys, &encoded)?;
        update_digest(&mut digest, time, sequence, key, value);
        charged += stored.charge;
    }
    Ok((hex::encode(digest.finalize()), charged))
}

fn retained_counts_match(
    status: &super::StreamJoinStatus,
    rows: usize,
    left_bytes: u64,
    right_bytes: u64,
) -> bool {
    status.left.retained_rows == rows as u64
        && status.right.retained_rows == ENTITIES as u64
        && status.left.retained_bytes == left_bytes
        && status.right.retained_bytes == right_bytes
}

fn retained_rows_unclassified(status: &super::StreamJoinStatus) -> bool {
    status.left.evicted_rows == 0
        && status.right.evicted_rows == 0
        && status.left.late_rows == 0
        && status.right.late_rows == 0
}

fn state_counters_match(
    operator: &StreamJoinOperator,
    rows: usize,
    left_bytes: u64,
    right_bytes: u64,
) -> bool {
    let status = operator.status();
    retained_counts_match(&status, rows, left_bytes, right_bytes)
        && retained_rows_unclassified(&status)
        && status.emitted_match_rows == rows as u64
        && !operator.state.ended
        && operator.state.next_left_row_id == rows as u64
        && operator.state.next_right_row_id == ENTITIES as u64
}

fn validate_state(operator: &StreamJoinOperator, rows: usize) -> Result<Value> {
    let (left, left_bytes) = validate_side(&operator.state.left, rows, true)?;
    let (right, right_bytes) = validate_side(&operator.state.right, ENTITIES, false)?;
    require(
        state_counters_match(operator, rows, left_bytes, right_bytes),
        "row3 full-state counters differ",
    )?;
    Ok(
        json!({"validated_all_rows": true, "left_rows": rows, "right_rows": ENTITIES,
        "left_sha256": left, "right_sha256": right}),
    )
}

fn progress(watermark: Option<i64>) -> IngressProgressSnapshot {
    IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, watermark.map(EventTime::from_micros)),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, None),
        ),
    ]))
}

async fn feed(
    operator: &mut StreamJoinOperator,
    job: &StreamJobContext,
    left: bool,
    start: usize,
    count: usize,
) -> Result<()> {
    let record = fixture_record(left, start, count)?;
    let batch = Batch::table(vec![record], BatchMetadata::default())?;
    let context = StreamOperatorContext::with_ingress_progress(
        job,
        OPERATOR,
        None,
        operator.ingress_progress.clone(),
    );
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            if left { "left" } else { "right" },
            batch,
            &context,
            &mut output,
        )
        .await?;
    validate_feed_output(&mut output, left, start, count)
}

fn output_values_match(
    quote: (i64, u64, &str, f64),
    factor: (i64, u64, &str, f64),
    next: usize,
    expected_keys: &[String],
) -> bool {
    quote.1 == next as u64
        && quote.0 == quote_time(next)
        && quote.3.to_bits() == price(next).to_bits()
        && quote.2 == expected_keys[next % ENTITIES]
        && quote.2 == factor.2
        && factor.1 == (next % ENTITIES) as u64
        && factor.0 == BASE_MICROS
        && factor.3.to_bits() == fixture_number(next % ENTITIES + 1).to_bits()
}

fn validate_output_record(
    record: &RecordBatch,
    left: bool,
    mut next: usize,
    expected_keys: &[String],
) -> Result<usize> {
    require(
        left && record.num_columns() == 8,
        "row3 output schema or preload differs",
    )?;
    for offset in 0..record.num_rows() {
        let quote = row_values(&record.columns()[..4], offset, true)?;
        let factor = row_values(&record.columns()[4..], offset, false)?;
        require(
            output_values_match(quote, factor, next, expected_keys),
            "row3 complete output oracle differs",
        )?;
        next += 1;
    }
    Ok(next)
}

fn validate_feed_output(
    output: &mut EdgeCollector,
    left: bool,
    start: usize,
    count: usize,
) -> Result<()> {
    let expected_keys = (0..ENTITIES)
        .map(|key| format!("S{key:03}"))
        .collect::<Vec<_>>();
    let mut next = start;
    for message in output.drain("output") {
        let batch = message
            .as_data()
            .ok_or_else(|| crate::benchmark_internals::invalid("non-data Join output"))?;
        for record in batch.table_payload()?.batches() {
            next = validate_output_record(record, left, next, &expected_keys)?;
        }
    }
    require(
        next == start + if left { count } else { 0 },
        "row3 output rows differ",
    )
}

async fn feed_quotes(
    operator: &mut StreamJoinOperator,
    job: &StreamJobContext,
    start: usize,
    end: usize,
    final_rows: usize,
) -> Result<()> {
    for first in (start..end).step_by(BATCH_ROWS) {
        let stop = (first + BATCH_ROWS).min(end);
        feed(operator, job, true, first, stop - first).await?;
        let maximum = quote_time(stop - 1) + 1;
        let watermark = if stop < final_rows {
            maximum.min(quote_time(stop))
        } else {
            maximum
        };
        let context = StreamOperatorContext::with_ingress_progress(
            job,
            OPERATOR,
            None,
            progress(Some(watermark)),
        );
        operator.on_ingress_progress("left", &context).await?;
    }
    Ok(())
}

fn segments(snapshot: &OperatorStateSnapshot) -> Vec<Value> {
    snapshot
        .segments
        .iter()
        .map(|(name, segment)| {
            json!({
                "name": name, "bytes": segment.bytes().len(),
                "sha256": hex::encode(Sha256::digest(segment.bytes())),
            })
        })
        .collect()
}

fn snapshot_evidence(operator: &StreamJoinOperator, snapshot: &OperatorStateSnapshot) -> Value {
    let status = operator.status();
    let inventory = segments(snapshot);
    let base = inventory
        .iter()
        .filter(|item| {
            item["name"]
                .as_str()
                .is_some_and(|name| name.ends_with("-base"))
        })
        .map(|item| {
            (
                item["name"].as_str().expect("segment name").to_owned(),
                item["sha256"].clone(),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let deltas = snapshot
        .segments
        .keys()
        .filter(|name| name.contains("-delta-"))
        .collect::<Vec<_>>();
    json!({"epoch": snapshot.inline_metadata["epoch"],
        "left_rows": status.left.retained_rows, "right_rows": status.right.retained_rows,
        "left_watermark_micros": status.left.watermark_micros,
        "right_watermark_micros": status.right.watermark_micros,
        "native_index_present": {"left": operator.state.left.1.is_some(), "right": operator.state.right.1.is_some()},
        "base_epoch": snapshot.inline_metadata["v2_inventory"]["base_epoch"],
        "base_sha256": base, "delta_names": deltas,
        "segment_bytes": snapshot.segments.values().map(|segment| segment.bytes().len()).sum::<usize>(),
        "segments": inventory})
}

struct Diagnostic {
    report: Value,
    pools: Vec<Arc<ObservedPool>>,
    active: Option<(Arc<ObservedPool>, &'static str)>,
    cleanup_ok: bool,
}

impl Diagnostic {
    fn new(rows: usize) -> Result<Self> {
        let operator = fixture_operator(rows)?;
        Ok(Self {
            report: json!({"schema": "calc-flow.join-checkpoint-history.v2",
                "case": "join_checkpoint_history_4m", "outcome": "running",
                "check_only": true, "checkpoint_enabled": true,
                "left_rows": rows, "dimension_rows": ENTITIES, "resident_rows": rows + ENTITIES,
                "batch_rows": BATCH_ROWS, "pool_policy": "unbounded", "pool_limit_bytes": null,
                "operator_id": OPERATOR, "join_spec": operator.spec(),
                "fixture": fixture_description(), "reservation_phases": [], "restores": [],
                "owned_cleanup_complete": false}),
            pools: Vec::new(),
            active: None,
            cleanup_ok: true,
        })
    }

    fn operator(
        &mut self,
        rows: usize,
        id: &'static str,
    ) -> Result<(StreamJoinOperator, Arc<ObservedPool>)> {
        let mut operator = fixture_operator(rows)?;
        let pool = Arc::new(ObservedPool::unbounded(id));
        self.pools.push(Arc::clone(&pool));
        operator.runtime.initialize_benchmark_pool(pool.clone())?;
        operator.ingress_progress = progress(None);
        Ok((operator, pool))
    }

    fn start(&mut self, pool: &Arc<ObservedPool>, phase: &'static str) {
        assert!(self.active.is_none(), "one diagnostic phase at a time");
        pool.begin_phase();
        self.active = Some((Arc::clone(pool), phase));
    }

    fn finish(&mut self) {
        if let Some((pool, phase)) = self.active.take() {
            let mut sample = json!(pool.snapshot());
            sample["phase"] = phase.into();
            self.report["reservation_phases"]
                .as_array_mut()
                .expect("phase array")
                .push(sample);
        }
    }

    fn failed_phase(&mut self) {
        if let Some((_, phase)) = &self.active {
            self.report["failed_phase"] = (*phase).into();
        }
    }

    fn restore(
        &mut self,
        rows: usize,
        name: &str,
        pool_id: &'static str,
        snapshot: &OperatorStateSnapshot,
        oracle: &Value,
    ) -> Result<()> {
        let (mut restored, pool) = self.operator(rows, pool_id)?;
        self.start(&pool, pool_id);
        restored.restore(snapshot)?;
        let restored_oracle = validate_state(&restored, rows)?;
        require(
            &restored_oracle == oracle,
            "restored full-state oracle differs",
        )?;
        drop(restored);
        self.report["restores"]
            .as_array_mut()
            .expect("restore array")
            .push(json!({"name": name, "oracle": restored_oracle}));
        self.finish();
        Ok(())
    }

    fn complete(mut self, failure: Option<String>) -> Value {
        self.finish();
        let pools = self
            .pools
            .iter()
            .map(|pool| pool.snapshot())
            .collect::<Vec<_>>();
        let refunded = pools.iter().all(|pool| pool.reserved_bytes == 0);
        self.report["pools"] = json!(pools);
        self.report["owned_cleanup_complete"] = (refunded && self.cleanup_ok).into();
        self.report["outcome"] = if failure.is_none() && refunded && self.cleanup_ok {
            "complete"
        } else {
            "failed"
        }
        .into();
        if let Some(error) = failure {
            self.report["error"] = error.into();
        }
        self.report
    }
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "row3-checkpoint-history",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

async fn prepare(operator: &mut StreamJoinOperator, job: &StreamJobContext) -> Result<()> {
    let context = StreamOperatorContext::with_ingress_progress(
        job,
        OPERATOR,
        None,
        operator.ingress_progress.clone(),
    );
    operator.prepare_checkpoint_async(&context).await
}

async fn fresh_admission(
    diagnostic: &mut Diagnostic,
    rows: usize,
    job: &StreamJobContext,
) -> Result<(StreamJoinOperator, Arc<ObservedPool>, Value)> {
    let (mut operator, pool) = diagnostic.operator(rows, "fresh")?;
    diagnostic.start(&pool, "fresh_admission");
    feed(&mut operator, job, false, 0, ENTITIES).await?;
    feed_quotes(&mut operator, job, 0, rows, rows).await?;
    let oracle = validate_state(&operator, rows)?;
    diagnostic.finish();
    Ok((operator, pool, oracle))
}

fn validate_fresh_captures(
    first_inventory: &[Value],
    second: &OperatorStateSnapshot,
    oracle: &Value,
    operator: &StreamJoinOperator,
    rows: usize,
) -> Result<()> {
    require(
        first_inventory == segments(second),
        "unchanged fresh capture changed segment bytes",
    )?;
    require(
        *oracle == validate_state(operator, rows)?,
        "fresh preparation changed retained state",
    )
}

async fn fresh_phase(
    diagnostic: &mut Diagnostic,
    rows: usize,
    job: &StreamJobContext,
) -> Result<Value> {
    let (mut operator, pool, oracle) = fresh_admission(diagnostic, rows, job).await?;
    for phase in ["fresh_prepare", "fresh_prepare_repeat"] {
        diagnostic.start(&pool, phase);
        prepare(&mut operator, job).await?;
        diagnostic.finish();
    }
    diagnostic.start(&pool, "fresh_capture");
    let first = operator.checkpoint(Epoch::INITIAL)?;
    let first_inventory = segments(&first);
    diagnostic.finish();
    diagnostic.start(&pool, "fresh_prepare_clean");
    prepare(&mut operator, job).await?;
    diagnostic.finish();
    diagnostic.start(&pool, "fresh_capture_clean");
    let second = operator.checkpoint(Epoch::new(2).expect("fixed nonzero epoch"))?;
    validate_fresh_captures(&first_inventory, &second, &oracle, &operator, rows)?;
    diagnostic.finish();
    diagnostic.start(&pool, "fresh_released");
    drop((first, second, operator));
    Ok(json!({"oracle": oracle, "output_rows": rows, "repeated_capture_segments_equal": true}))
}

struct HeldHistory {
    report: Value,
    before: OperatorStateSnapshot,
    after: OperatorStateSnapshot,
    segment_alias: crate::StateSegment,
    pool: Arc<ObservedPool>,
}

impl HeldHistory {
    fn release(self, diagnostic: &mut Diagnostic) -> Result<Value> {
        let Self {
            report,
            before,
            after,
            segment_alias,
            pool,
        } = self;
        diagnostic.start(&pool, "history_snapshots_released");
        let before_release = pool.reserved();
        drop((before, after));
        require(
            before_release > pool.reserved() && pool.reserved() > 0,
            "snapshot credit did not refund independently of the held segment owner",
        )?;
        diagnostic.finish();
        diagnostic.start(&pool, "history_released");
        drop(segment_alias);
        require(
            pool.reserved() == 0,
            "segment alias did not refund its backing and capture-control credit",
        )?;
        Ok(report)
    }
}

fn held_segment(
    snapshot: &OperatorStateSnapshot,
    pool_id: &str,
) -> Result<(crate::StateSegment, Value)> {
    let (name, segment) = snapshot
        .segments
        .iter()
        .filter(|(name, _)| name.starts_with("left-payload-"))
        .max_by_key(|(_, segment)| segment.bytes().len())
        .ok_or_else(|| crate::benchmark_internals::invalid("missing owned left payload segment"))?;
    require(
        segment.has_owner(),
        "held payload segment has no reservation owner",
    )?;
    let evidence = json!({"kind": "StateSegment", "pool_id": pool_id,
        "snapshot_epoch": snapshot.inline_metadata["epoch"],
        "name": name, "bytes": segment.bytes().len(),
        "sha256": hex::encode(Sha256::digest(segment.bytes())),
        "has_owner": true, "credit_scope": "per_segment_backing_plus_shared_capture_controls"});
    Ok((segment.clone(), evidence))
}

async fn captured_history(
    diagnostic: &mut Diagnostic,
    operator: &mut StreamJoinOperator,
    pool: &Arc<ObservedPool>,
    rows: usize,
    job: &StreamJobContext,
) -> Result<(OperatorStateSnapshot, Vec<Value>)> {
    diagnostic.start(pool, "history_checkpoint_1");
    feed(operator, job, false, 0, ENTITIES).await?;
    feed_quotes(operator, job, 0, rows - 3, rows).await?;
    let mut before = operator.checkpoint(Epoch::INITIAL)?;
    let mut checkpoints = vec![snapshot_evidence(operator, &before)];
    diagnostic.finish();
    for (index, phase) in [
        "history_checkpoint_2",
        "history_checkpoint_3",
        "history_checkpoint_4",
    ]
    .into_iter()
    .enumerate()
    {
        diagnostic.start(pool, phase);
        let first = rows - 3 + index;
        feed_quotes(operator, job, first, first + 1, rows).await?;
        before = operator.checkpoint(Epoch::new(index as u64 + 2).expect("fixed nonzero epoch"))?;
        checkpoints.push(snapshot_evidence(operator, &before));
        diagnostic.finish();
    }
    validate_captured_history(&before)?;
    Ok((before, checkpoints))
}

fn validate_captured_history(before: &OperatorStateSnapshot) -> Result<()> {
    let history = before
        .inline_metadata
        .get("v2_inventory")
        .and_then(|value| value.get("deltas"))
        .and_then(Value::as_array)
        .ok_or_else(|| crate::benchmark_internals::invalid("missing four-epoch history"))?;
    require(
        history.len() == 4
            && history
                .iter()
                .enumerate()
                .all(|(index, delta)| delta["epoch"] == index as u64 + 1),
        "history did not record four dirty epochs",
    )
}

async fn prepare_history_replacement(
    diagnostic: &mut Diagnostic,
    operator: &mut StreamJoinOperator,
    pool: &Arc<ObservedPool>,
    job: &StreamJobContext,
) -> Result<()> {
    diagnostic.start(pool, "history_replace");
    prepare(operator, job).await?;
    diagnostic.finish();
    diagnostic.start(pool, "history_prepare_clean");
    prepare(operator, job).await?;
    diagnostic.finish();
    Ok(())
}

fn validate_replaced_history(
    operator: &StreamJoinOperator,
    rows: usize,
    before: &OperatorStateSnapshot,
    before_evidence: &Value,
    after_evidence: &Value,
    oracle: &Value,
) -> Result<Vec<Value>> {
    require(
        before_evidence["base_sha256"] != after_evidence["base_sha256"]
            && before_evidence["base_epoch"] != after_evidence["base_epoch"]
            && after_evidence["delta_names"]
                .as_array()
                .is_some_and(Vec::is_empty),
        "history preparation did not replace the base and fold its deltas",
    )?;
    let old_segments_after = segments(before);
    require(
        before_evidence["segments"] == json!(old_segments_after),
        "held old snapshot bytes changed",
    )?;
    require(
        *oracle == validate_state(operator, rows)?,
        "base replacement changed retained state",
    )?;
    Ok(old_segments_after)
}

fn capture_history_replacement(
    diagnostic: &mut Diagnostic,
    operator: &mut StreamJoinOperator,
    pool: &Arc<ObservedPool>,
    before: &OperatorStateSnapshot,
    before_evidence: &Value,
    oracle: &Value,
    rows: usize,
) -> Result<(OperatorStateSnapshot, Value, Vec<Value>)> {
    diagnostic.start(pool, "history_capture");
    let after = operator.checkpoint(Epoch::new(5).expect("fixed nonzero epoch"))?;
    let after_evidence = snapshot_evidence(operator, &after);
    let old_segments_after = validate_replaced_history(
        operator,
        rows,
        before,
        before_evidence,
        &after_evidence,
        oracle,
    )?;
    diagnostic.finish();
    Ok((after, after_evidence, old_segments_after))
}

fn restore_history_snapshots(
    diagnostic: &mut Diagnostic,
    rows: usize,
    pool: &ObservedPool,
    before: &OperatorStateSnapshot,
    after: &OperatorStateSnapshot,
    oracle: &Value,
) -> Result<()> {
    require(
        pool.reserved() > 0,
        "held snapshot and segment aliases lost their reservation",
    )?;
    diagnostic.finish();
    for (name, pool_id, snapshot) in [
        ("old_history", "restore_old", before),
        ("replacement", "restore_new", after),
    ] {
        diagnostic.restore(rows, name, pool_id, snapshot, oracle)?;
    }
    Ok(())
}

async fn history_phase(
    diagnostic: &mut Diagnostic,
    rows: usize,
    job: &StreamJobContext,
) -> Result<HeldHistory> {
    let (mut operator, pool) = diagnostic.operator(rows, "history")?;
    let (before, checkpoints) =
        captured_history(diagnostic, &mut operator, &pool, rows, job).await?;
    let oracle = validate_state(&operator, rows)?;
    let before_evidence = snapshot_evidence(&operator, &before);
    prepare_history_replacement(diagnostic, &mut operator, &pool, job).await?;
    let (after, after_evidence, old_segments_after) = capture_history_replacement(
        diagnostic,
        &mut operator,
        &pool,
        &before,
        &before_evidence,
        &oracle,
        rows,
    )?;
    let (segment_alias, alias_ownership) = held_segment(&before, pool.id)?;
    diagnostic.start(&pool, "history_source_released");
    drop(operator);
    restore_history_snapshots(diagnostic, rows, &pool, &before, &after, &oracle)?;
    let report = json!({"dirty_epochs": 4, "checkpoints": checkpoints,
        "before": before_evidence, "after": after_evidence,
        "old_segments_after": old_segments_after, "old_segments_unchanged": true,
        "oracle": oracle, "output_rows": rows, "alias_ownership": alias_ownership});
    Ok(HeldHistory {
        report,
        before,
        after,
        segment_alias,
        pool,
    })
}

async fn run_history(rows: usize) -> Result<Value> {
    let mut diagnostic = Diagnostic::new(rows)?;
    let fresh_job = job();
    let result = fresh_phase(&mut diagnostic, rows, &fresh_job).await;
    if result.is_err() {
        diagnostic.failed_phase();
    }
    let cleanup = crate::benchmark_internals::settle_job(&fresh_job).await;
    drop(fresh_job);
    diagnostic.cleanup_ok &= cleanup.is_ok();
    diagnostic.finish();
    let result = result.and_then(|value| cleanup.map(|()| value));
    let fresh = match result {
        Ok(value) => value,
        Err(error) => return Ok(diagnostic.complete(Some(error.to_string()))),
    };
    diagnostic.report["fresh"] = fresh;
    let history_job = job();
    let result = history_phase(&mut diagnostic, rows, &history_job).await;
    if result.is_err() {
        diagnostic.failed_phase();
    }
    let cleanup = crate::benchmark_internals::settle_job(&history_job).await;
    drop(history_job);
    diagnostic.cleanup_ok &= cleanup.is_ok();
    diagnostic.finish();
    match result
        .and_then(|held| cleanup.map(|()| held))
        .and_then(|held| held.release(&mut diagnostic))
    {
        Ok(value) => {
            diagnostic.report["history"] = value;
            Ok(diagnostic.complete(None))
        }
        Err(error) => {
            diagnostic.failed_phase();
            Ok(diagnostic.complete(Some(error.to_string())))
        }
    }
}

pub(crate) async fn run_checkpoint_history_diagnostic() -> Result<Value> {
    run_history(FULL_ROWS).await
}

#[cfg(test)]
#[path = "benchmark_checkpoint_tests.rs"]
mod tests;
