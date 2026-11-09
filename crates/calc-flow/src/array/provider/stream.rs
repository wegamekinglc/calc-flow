use std::{
    any::{Any, TypeId},
    collections::BTreeMap,
    sync::{Arc, Weak},
};

use async_trait::async_trait;
use tokio::sync::OwnedSemaphorePermit;

use super::{
    sequence_pending::Pending,
    worker::{Scope, Workers},
};
use crate::{
    Batch, CalcFlowError, Epoch, EventTime, ExternalOperatorSpec, JsonMap, OperatorMetadata,
    OperatorStateSnapshot, Port, Result, StateSegment, StreamCollector, StreamOperator,
    StreamOperatorContext, StreamOperatorFactory, StreamOperatorLifecycle,
    array::{
        NativeArray, invalid,
        sequence::{
            capture::Captures,
            codec::{self, SequenceSpec, SequenceState},
        },
    },
};

mod checkpoint;
pub(crate) mod inventory;
mod options;
mod preload;
mod schema;

/// Factory for the ordered, checkpointed `native:array_sequence@1` stream provider.
///
/// Each operator owns one independent numerical sequence. A factory shares two
/// bounded worker slots; one operator additionally serializes its pending work.
///
/// ```
/// use calc_flow::{BatchKind, ExternalOperatorSpec, JsonMap, Port, StreamOperatorFactory,
///     array::NativeArraySequenceFactory};
/// # fn main() -> calc_flow::Result<()> {
/// let options: JsonMap = serde_json::from_value(serde_json::json!({
///     "ir_version": 1, "input_dtype": "float64",
///     "kernel": {"function": "ewma", "span": 3}
/// })).expect("options are an object");
/// let spec = ExternalOperatorSpec::new("native", "array_sequence", "1", options)?;
/// let operator = NativeArraySequenceFactory::default().create(&spec,
///     vec![Port::new("input", BatchKind::Array, true, None)?],
///     vec![Port::new("output", BatchKind::Array, true, None)?])?;
/// assert_eq!(operator.name(), "native:array_sequence@1");
/// # Ok(())
/// # }
/// ```
#[derive(Default)]
pub struct NativeArraySequenceFactory {
    workers: Arc<Workers>,
}

impl NativeArraySequenceFactory {
    pub(super) fn with_workers(workers: Arc<Workers>) -> Self {
        Self { workers }
    }

    /// Returns the strict options schema and stream lifecycle descriptor.
    pub fn capability_descriptor() -> JsonMap {
        schema::descriptor()
    }
}

impl StreamOperatorFactory for NativeArraySequenceFactory {
    fn create(
        &self,
        spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Box<dyn StreamOperator>> {
        if (spec.provider(), spec.name(), spec.version()) != ("native", "array_sequence", "1") {
            return Err(invalid("provider", "requires native:array_sequence@1"));
        }
        ports(&inputs, "input", "inputs")?;
        ports(&outputs, "output", "outputs")?;
        let configured = options::parse(spec.options())?;
        let state = Arc::new(SequenceState::new(&configured.spec, configured.state)?);
        Ok(Box::new(NativeStreamOperator {
            configured: Arc::new(configured),
            workers: Arc::clone(&self.workers),
            pending: Pending::default(),
            state,
            captures: Captures::default(),
            capture: None,
            reset_pending: false,
            inputs,
            outputs,
            options: spec.options().clone(),
        }))
    }
}
fn ports(ports: &[Port], name: &str, field: &str) -> Result<()> {
    super::validate::ports(ports, std::iter::once(name), field)?;
    if ports[0].schema().is_some() {
        return Err(invalid(
            field,
            "native sequence ports cannot declare Arrow schemas",
        ));
    }
    Ok(())
}

struct NativeStreamOperator {
    configured: Arc<options::Options>,
    workers: Arc<Workers>,
    pending: Pending,
    state: Arc<SequenceState>,
    captures: Captures,
    capture: Option<Capture>,
    reset_pending: bool,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    options: JsonMap,
}

struct Capture {
    state: Weak<SequenceState>,
    segment: StateSegment,
}
impl Capture {
    fn is_current(&self, state: &Arc<SequenceState>) -> bool {
        Weak::ptr_eq(&self.state, &Arc::downgrade(state))
    }
}

impl OperatorMetadata for NativeStreamOperator {
    fn name(&self) -> &'static str {
        "native:array_sequence@1"
    }
    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }
    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }
    fn configuration(&self) -> JsonMap {
        BTreeMap::from([
            ("provider".into(), "native".into()),
            ("name".into(), "array_sequence".into()),
            ("version".into(), "1".into()),
            (
                "options".into(),
                serde_json::to_value(&self.options).expect("JSON options are serializable"),
            ),
        ])
    }
}

struct Prepared {
    candidate: Option<Arc<SequenceState>>,
    output: Batch,
    _pending: OwnedSemaphorePermit,
}

fn native<'a>(batch: &'a Batch, spec: &SequenceSpec) -> Result<&'a NativeArray> {
    let payload = batch.external_payload().map_err(|_| {
        invalid(
            "inputs.input.backend",
            "requires an owned NativeArray payload",
        )
    })?;
    if Any::type_id(payload.as_ref()) != TypeId::of::<NativeArray>() {
        return Err(invalid(
            "inputs.input.backend",
            "requires an owned NativeArray payload",
        ));
    }
    let array = payload
        .as_any()
        .downcast_ref::<NativeArray>()
        .expect("intrinsic NativeArray identity was checked");
    if array.dtype() != spec.dtype() {
        return Err(invalid(
            "inputs.input.dtype",
            format!(
                "declared {}; received {}",
                dtype(spec.dtype()),
                dtype(array.dtype())
            ),
        ));
    }
    spec.validate_input_shape(array.shape())?;
    Ok(array)
}
fn dtype(dtype: crate::array::ArrayDType) -> &'static str {
    match dtype {
        crate::array::ArrayDType::Float32 => "float32",
        crate::array::ArrayDType::Float64 => "float64",
        crate::array::ArrayDType::Boolean => "bool",
    }
}
fn prepare(
    state: &SequenceState,
    batch: &Batch,
    configured: &options::Options,
    captures: usize,
    pending: OwnedSemaphorePermit,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Prepared> {
    check()?;
    let input = native(batch, &configured.spec)?;
    let prepared = state.prepare_checked(input.view(), &configured.execution, captures, check)?;
    let candidate = prepared.candidate.map(Arc::new);
    let output = Batch::external(Arc::new(prepared.output), batch.metadata().clone())?;
    check()?;
    Ok(Prepared {
        candidate,
        output,
        _pending: pending,
    })
}

impl NativeStreamOperator {
    fn validate_data(
        &self,
        ingress: &str,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        context.check_cancelled()?;
        if ingress != "input" || context.native_static_input("input").is_some() {
            return Err(invalid(
                "inputs.input",
                "requires exactly one dynamic ingress named input",
            ));
        }
        native(batch, &self.configured.spec)?;
        Ok(())
    }
    async fn prepare_data(
        &mut self,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Prepared> {
        let pending = self.pending.enter(context.job()).await?;
        self.apply_reset();
        let state = Arc::clone(&self.state);
        let configured = Arc::clone(&self.configured);
        let captures = self.captures.bytes();
        self.workers
            .run_scoped(
                Scope::stream(context.job()),
                "array_sequence",
                move |check| prepare(&state, &batch, &configured, captures, pending, check),
            )
            .await
    }
    async fn restore_prepared(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let pending = self.pending.enter(context.job()).await?;
        self.apply_reset();
        checkpoint::validate(snapshot, &self.configured.spec)?;
        let configured = Arc::clone(&self.configured);
        let old = self.state.state_bytes();
        let captures = self.captures.clone();
        let snapshot = snapshot.clone();
        let (restored, _pending) = self
            .workers
            .run_scoped(
                Scope::stream(context.job()),
                "array_sequence",
                move |check| {
                    let restored =
                        checkpoint::restore(&configured, old, &captures, &snapshot, check)?;
                    Ok((restored, pending))
                },
            )
            .await?;
        context.check_cancelled()?;
        self.commit_restore(restored);
        Ok(())
    }
    fn apply_reset(&mut self) {
        if self.reset_pending {
            self.capture = None;
            Arc::get_mut(&mut self.state)
                .expect("settled operator owns its committed state")
                .reset();
            self.reset_pending = false;
        }
    }
    fn commit_restore(&mut self, restored: checkpoint::Restored) {
        self.state = restored.state;
        self.capture = Some(Capture {
            state: Arc::downgrade(&self.state),
            segment: restored.segment,
        });
        self.reset_pending = false;
    }
}

fn error_message(message: &str, context: &StreamOperatorContext<'_>) -> String {
    format!(
        "{message}; native:array_sequence@1 operator {:?}, ingress input",
        context.operator_id()
    )
}
fn kernel_error(error: CalcFlowError, context: &StreamOperatorContext<'_>) -> CalcFlowError {
    match error {
        CalcFlowError::InvalidArgument { field, message } => CalcFlowError::InvalidArgument {
            field,
            message: error_message(&message, context),
        },
        error => error,
    }
}
fn restore_error(error: CalcFlowError, context: &StreamOperatorContext<'_>) -> CalcFlowError {
    match error {
        CalcFlowError::CheckpointMismatch { message } => CalcFlowError::CheckpointMismatch {
            message: error_message(&message, context),
        },
        CalcFlowError::InvalidArgument { field, message } => CalcFlowError::CheckpointMismatch {
            message: error_message(&format!("{field}: {message}"), context),
        },
        error => error,
    }
}

#[async_trait]
impl StreamOperator for NativeStreamOperator {
    fn lifecycle(&self) -> StreamOperatorLifecycle {
        StreamOperatorLifecycle::CheckpointedStateful {
            state_version: 1,
            microbatch_invariant: true,
            deterministic: true,
            replay_safe: true,
        }
    }
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.validate_data(ingress, &batch, context)?;
        let prepared = self
            .prepare_data(batch, context)
            .await
            .map_err(|error| kernel_error(error, context))?;
        context.check_cancelled()?;
        output.emit("output", prepared.output).await?;
        if let Some(candidate) = prepared.candidate {
            self.state = candidate;
            self.capture = None;
        }
        Ok(())
    }
    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()
    }
    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()
    }

    async fn prepare_checkpoint_async(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let pending = self.pending.enter(context.job()).await?;
        self.apply_reset();
        if self
            .capture
            .as_ref()
            .is_some_and(|capture| capture.is_current(&self.state))
        {
            return context.check_cancelled();
        }
        let state = Arc::clone(&self.state);
        let configured = Arc::clone(&self.configured);
        let captures = self.captures.clone();
        let (segment, _pending) = self
            .workers
            .run_scoped(
                Scope::stream(context.job()),
                "array_sequence",
                move |check| {
                    let segment =
                        captures.capture(&configured.spec, &state, &configured.execution, check)?;
                    Ok((segment, pending))
                },
            )
            .await?;
        context.check_cancelled()?;
        self.capture = Some(Capture {
            state: Arc::downgrade(&self.state),
            segment,
        });
        Ok(())
    }

    fn checkpoint(&mut self, _epoch: Epoch) -> Result<OperatorStateSnapshot> {
        let _pending = self.pending.try_enter()?;
        let capture = self
            .capture
            .as_ref()
            .filter(|capture| capture.is_current(&self.state))
            .ok_or_else(|| {
                codec::mismatch(
                    "preparation",
                    "prepare_checkpoint_async is required after the latest state change",
                )
            })?;
        Ok(OperatorStateSnapshot {
            inline_metadata: checkpoint::metadata(&self.configured.spec),
            segments: BTreeMap::from([("state".into(), capture.segment.clone())]),
        })
    }
    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        let _pending = self.pending.try_enter()?;
        self.apply_reset();
        let restored = checkpoint::restore(
            &self.configured,
            self.state.state_bytes(),
            &self.captures,
            snapshot,
            &mut || Ok(()),
        )?;
        self.commit_restore(restored);
        Ok(())
    }
    async fn restore_async(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        self.restore_prepared(snapshot, context)
            .await
            .map_err(|error| restore_error(error, context))
    }
    fn reset(&mut self) -> Result<()> {
        self.capture = None;
        self.reset_pending = true;
        if let Ok(_pending) = self.pending.try_enter() {
            self.apply_reset();
        }
        Ok(())
    }
}
