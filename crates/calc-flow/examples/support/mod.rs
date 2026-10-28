//! Shared fixtures and owned source/sink lifecycles for the Rust examples.
#![allow(dead_code)]

use std::{collections::BTreeMap, path::Path, sync::Arc, time::Duration};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, Cursor, Edge, EventTime, ExpressionOperator,
    JobOutcome, JobState, LateErrorScope, LatePolicySpec, ManagedCheckpointRuntime,
    NativeWatermarkCapability, PipelineBuilder, Port, PortEndpoint, ROLLING_CONFIGURATION_VERSION,
    ROLLING_STATE_LAYOUT_VERSION, ReplayPositioning, Result, RollingNumericalProfile,
    RollingOutputSpec, RollingSpec, RollingValuePolicy, SinkBinding, SourceBinding,
    SourceCapabilities, SourceDeliveryCapability, SourceEvent, SourceSchema, SqlOperator,
    StreamExecutionPlan, StreamRequirements, StreamSink, StreamSource, StreamingJob,
    StreamingRunner, UdfRegistry, WatermarkPolicy,
};
use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray,
        UInt64Array,
    },
    datatypes::SchemaRef,
    record_batch::RecordBatch,
    util::pretty::print_batches,
};
use tokio::sync::{Mutex, Notify};

pub type ExampleResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub type Collected = Arc<Mutex<Vec<Batch>>>;

pub fn arguments() -> ExampleResult<Vec<String>> {
    // Discard argv[0]; validated demo options do not make security decisions.
    std::env::args_os() // nosemgrep: args-os
        .skip(1)
        .map(argument_text)
        .collect()
}

fn argument_text(value: std::ffi::OsString) -> ExampleResult<String> {
    value
        .into_string()
        .map_err(|_| "arguments must be UTF-8".into())
}

pub fn ints(values: &[i64]) -> ArrayRef {
    Arc::new(Int64Array::from(values.to_vec()))
}
pub fn uints(values: &[u64]) -> ArrayRef {
    Arc::new(UInt64Array::from(values.to_vec()))
}
pub fn floats(values: &[f64]) -> ArrayRef {
    Arc::new(Float64Array::from(values.to_vec()))
}
pub fn strings(values: &[&str]) -> ArrayRef {
    Arc::new(StringArray::from(values.to_vec()))
}
pub fn times(values: &[i64]) -> ArrayRef {
    Arc::new(TimestampMicrosecondArray::from(values.to_vec()).with_timezone("UTC"))
}

pub fn table(columns: Vec<(&str, ArrayRef)>) -> Batch {
    Batch::table(
        vec![RecordBatch::try_from_iter(columns).expect("valid example columns")],
        BatchMetadata::default(),
    )
    .expect("valid example table")
}

pub fn schema(batch: &Batch) -> SchemaRef {
    Arc::clone(batch.table_payload().expect("table").schema())
}

pub fn expression(
    name: &str,
    formula: &str,
    select: &[&str],
    filter: Option<&str>,
) -> Result<Box<ExpressionOperator>> {
    if !formula.is_empty() && !select.is_empty() {
        let (output, value) = formula.split_once('=').expect("example assignment");
        let projections = select
            .iter()
            .map(|s| {
                if *s == output.trim() {
                    format!("{} AS {}", value.trim(), output.trim())
                } else {
                    (*s).into()
                }
            })
            .collect();
        return Ok(Box::new(ExpressionOperator::new(
            name,
            "",
            projections,
            filter.map(str::to_owned),
            vec![],
        )?));
    }
    Ok(Box::new(ExpressionOperator::new(
        name,
        formula,
        select.iter().map(|s| (*s).into()).collect(),
        filter.map(str::to_owned),
        vec![],
    )?))
}

pub fn sql(name: &str, query: &str, aliases: &[&str]) -> Result<Box<SqlOperator>> {
    Ok(Box::new(SqlOperator::new(
        name,
        query,
        aliases.iter().map(|s| (*s).into()).collect(),
        vec![],
    )?))
}

pub fn edge(from: &str, output: &str, to: &str, input: &str) -> Result<Edge> {
    Ok(Edge::new(
        PortEndpoint::new(from, output)?,
        PortEndpoint::new(to, input)?,
    ))
}

pub fn connect_all(
    builder: PipelineBuilder,
    connections: &[(&str, &str, &str, &str)],
) -> Result<PipelineBuilder> {
    connections
        .iter()
        .try_fold(builder, |builder, &(from, output, to, input)| {
            builder.connect(edge(from, output, to, input)?)
        })
}

pub fn table_port(name: &str, schema: Option<SchemaRef>) -> Result<Port> {
    Port::new(
        name,
        BatchKind::Table,
        true,
        schema.map(|s| s.fields().iter().map(|f| f.as_ref().clone()).collect()),
    )
}

pub fn expression_schemas(
    operator: Box<ExpressionOperator>,
    input: Option<SchemaRef>,
    output: Option<SchemaRef>,
) -> Result<Box<ExpressionOperator>> {
    Ok(Box::new((*operator).with_ports(
        table_port("input", input)?,
        table_port("output", output)?,
    )?))
}

pub fn sql_input_schema(operator: Box<SqlOperator>, input: SchemaRef) -> Result<Box<SqlOperator>> {
    Ok(Box::new((*operator).with_ports(
        vec![table_port("input", Some(input))?],
        table_port("output", None)?,
    )?))
}

pub fn stream_plan(builder: PipelineBuilder) -> Result<StreamExecutionPlan> {
    builder.compile_stream(
        &UdfRegistry::new().snapshot(),
        &StreamRequirements::default(),
    )
}

pub fn best_effort_plan(builder: PipelineBuilder) -> Result<StreamExecutionPlan> {
    builder.compile_stream(
        &UdfRegistry::new().snapshot(),
        &StreamRequirements {
            delivery: BTreeMap::from([("output".into(), calc_flow::DeliveryGuarantee::BestEffort)]),
        },
    )
}

pub fn rolling(outputs: Vec<RollingOutputSpec>) -> RollingSpec {
    RollingSpec {
        configuration_version: ROLLING_CONFIGURATION_VERSION,
        state_layout_version: ROLLING_STATE_LAYOUT_VERSION,
        numerical_profile: RollingNumericalProfile::default(),
        partition_by: vec!["symbol".into()],
        event_time: "ts".into(),
        sequence_by: vec!["seq".into()],
        outputs,
        allowed_lateness_micros: 0,
        late_policy: LatePolicySpec::Error {
            scope: LateErrorScope::Envelope,
        },
        value_policy: RollingValuePolicy::StatefulNumericV1,
    }
}

pub fn cursor(offset: usize) -> Result<Cursor> {
    Cursor::unbound(
        u64::try_from(offset)
            .expect("example offset fits u64")
            .to_be_bytes()
            .to_vec(),
        BTreeMap::from([("offset".into(), serde_json::json!(offset))]),
    )
}

pub fn data(batch: Batch, offset: usize) -> Result<SourceEvent> {
    Ok(SourceEvent::Data {
        batch,
        cursor: cursor(offset)?,
    })
}
pub fn watermark(micros: i64) -> SourceEvent {
    SourceEvent::Watermark(EventTime::from_micros(micros))
}

/// An immutable replay trace; `pause_at` permits a checkpoint before EOF.
pub struct ReplaySource {
    events: Vec<SourceEvent>,
    offset: usize,
    pause_at: Option<usize>,
    pub paused: Arc<Notify>,
    pub opened_at: Arc<Mutex<Vec<usize>>>,
}

impl ReplaySource {
    pub fn new(events: Vec<SourceEvent>) -> Self {
        Self {
            events,
            offset: 0,
            pause_at: None,
            paused: Arc::default(),
            opened_at: Arc::default(),
        }
    }
    pub fn batches(batches: Vec<Batch>) -> Result<Self> {
        Ok(Self::new(
            batches
                .into_iter()
                .enumerate()
                .map(|(i, b)| data(b, i + 1))
                .collect::<Result<_>>()?,
        ))
    }
    pub fn pause_at(mut self, offset: usize) -> Self {
        self.pause_at = Some(offset);
        self
    }
    pub fn binding(self) -> SourceBinding {
        let policy = if self
            .events
            .iter()
            .any(|e| matches!(e, SourceEvent::Watermark(_)))
        {
            WatermarkPolicy::SourceProvided
        } else {
            WatermarkPolicy::Disabled { idle_timeout: None }
        };
        SourceBinding::new(self).with_watermark_policy(policy)
    }
}

#[async_trait]
impl StreamSource for ReplaySource {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::ExactPauseReportAndSeek,
            delivery: SourceDeliveryCapability::Lossless,
            max_batch_rows: self
                .events
                .iter()
                .filter_map(|e| match e {
                    SourceEvent::Data { batch, .. } => Some(batch.num_rows()),
                    _ => None,
                })
                .max()
                .unwrap_or(1)
                .max(1),
            max_batch_bytes: 64 << 20,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: if self
                .events
                .iter()
                .any(|e| matches!(e, SourceEvent::Watermark(_)))
            {
                NativeWatermarkCapability::EmitsNative
            } else {
                NativeWatermarkCapability::NeverEmits
            },
        }
    }
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.offset = cursor
            .as_ref()
            .and_then(|c| c.payload().get("offset"))
            .and_then(serde_json::Value::as_u64)
            .map(|v| usize::try_from(v).expect("example cursor fits usize"))
            .unwrap_or_default();
        self.opened_at.lock().await.push(self.offset);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.pause_at == Some(self.offset) {
            self.paused.notify_one();
            tokio::time::sleep(Duration::from_millis(1)).await;
            return Ok(Some(SourceEvent::Idle));
        }
        let event = self.events.get(self.offset).cloned();
        self.offset += usize::from(event.is_some());
        Ok(event)
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

pub struct CollectSink(pub Collected);
#[async_trait]
impl StreamSink for CollectSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.0.lock().await.push(batch.clone());
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

pub fn runner(
    plan: StreamExecutionPlan,
    sources: BTreeMap<String, SourceBinding>,
    outputs: &BTreeMap<String, Collected>,
    root: &Path,
) -> Result<StreamingRunner> {
    let sinks = outputs
        .iter()
        .map(|(name, batches)| {
            Ok((
                name.clone(),
                vec![SinkBinding::ordinary(
                    name,
                    CollectSink(Arc::clone(batches)),
                )?],
            ))
        })
        .collect::<Result<_>>()?;
    StreamingRunner::new(plan, sources, sinks, ManagedCheckpointRuntime::new(root)?)
}

/// Bound the whole wait and await cancellation if the source stalls.
pub async fn completed(job: &StreamingJob) -> ExampleResult<JobOutcome> {
    let outcome = match tokio::time::timeout(Duration::from_secs(30), job.wait()).await {
        Ok(outcome) => outcome,
        Err(error) => {
            job.cancel().await;
            return Err(error.into());
        }
    };
    assert_eq!(outcome.state, JobState::Completed, "{outcome:?}");
    Ok(outcome)
}

pub async fn collect(
    plan: StreamExecutionPlan,
    sources: BTreeMap<String, SourceBinding>,
) -> ExampleResult<BTreeMap<String, Vec<Batch>>> {
    let outputs: BTreeMap<String, Collected> = plan
        .sink_binding_ids()
        .into_iter()
        .map(|s| (s.to_owned(), Arc::default()))
        .collect();
    let directory = tempfile::tempdir()?;
    let job = runner(plan, sources, &outputs, directory.path())?
        .start()
        .await?;
    completed(&job).await?;
    let mut collected = BTreeMap::new();
    for (name, values) in outputs {
        collected.insert(name, values.lock().await.clone());
    }
    Ok(collected)
}

pub fn i64_values(batches: &[Batch], column: &str) -> Vec<Option<i64>> {
    batches
        .iter()
        .flat_map(|b| b.table_payload().expect("table").batches())
        .flat_map(|b| {
            b.column_by_name(column)
                .expect("column")
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("int64")
                .iter()
        })
        .collect()
}
pub fn f64_values(batches: &[Batch], column: &str) -> Vec<Option<f64>> {
    batches
        .iter()
        .flat_map(|b| b.table_payload().expect("table").batches())
        .flat_map(|b| {
            b.column_by_name(column)
                .expect("column")
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("float64")
                .iter()
        })
        .collect()
}
pub fn check_floats(actual: &[Option<f64>], expected: &[Option<f64>]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        match (a, e) {
            (Some(a), Some(e)) => assert!(
                (a - e).abs() <= 1e-10 * e.abs().max(1.0),
                "{actual:?} != {expected:?}"
            ),
            _ => assert_eq!(a, e),
        }
    }
}
pub fn print(batches: &[Batch]) -> ExampleResult {
    for batch in batches {
        print_batches(batch.table_payload()?.batches())?;
    }
    Ok(())
}

pub fn invalid(field: &str, message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: field.into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::argument_text;

    #[test]
    fn unicode_arguments_are_preserved() {
        assert_eq!(argument_text("计算".into()).unwrap(), "计算");
    }

    #[cfg(unix)]
    #[test]
    fn invalid_unicode_arguments_return_an_error() {
        use std::os::unix::ffi::OsStringExt;
        let value = std::ffi::OsString::from_vec(vec![0xff]);
        assert_eq!(
            argument_text(value).unwrap_err().to_string(),
            "arguments must be UTF-8"
        );
    }
}
