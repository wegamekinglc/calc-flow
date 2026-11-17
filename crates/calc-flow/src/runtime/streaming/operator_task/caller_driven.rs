//! One operator driven by `StreamSession` on the caller's thread: the runner's
//! per-message dispatch and end-of-input code without a task, gates or waits.

use super::{
    AggregateInput, AssertUnwindSafe, CalcFlowError, FutureExt, OperatorBarrierAlignment,
    OperatorInputProgress, OperatorTaskInputs, Result, StreamMessage, StreamMessageKind,
    catch_unwind, control, dispatch_barrier, dispatch_message, finish_operator, ingress_admission,
    panic_message, sync_adapter,
};
use crate::Batch;

pub(in crate::runtime::streaming) struct CallerDrivenOperator {
    node_id: String,
    ingress_names: Vec<String>,
    inputs: OperatorTaskInputs,
    progress: OperatorInputProgress,
    alignment: OperatorBarrierAlignment,
    ended: bool,
    #[cfg(test)]
    panic_on_next_dispatch: bool,
}

impl CallerDrivenOperator {
    /// Resets and prepares the operator like runner entry for a fresh job.
    pub(in crate::runtime::streaming) async fn prepare(
        mut inputs: OperatorTaskInputs,
    ) -> Result<Self> {
        let node_id = inputs.node_id.clone();
        if inputs.ingresses.is_empty() {
            return Err(CalcFlowError::Internal {
                message: format!("operator {node_id:?} has no runtime ingress"),
            });
        }
        match catch_unwind(AssertUnwindSafe(|| inputs.operator.reset())) {
            Ok(result) => result?,
            Err(payload) => return Err(panicked(&node_id, payload.as_ref())),
        }
        contain(&node_id, sync_adapter::prepare(&mut inputs)).await?;
        let ingress_names = inputs.ingresses.keys().cloned().collect();
        let progress = OperatorInputProgress::new(inputs.ingresses.keys());
        Ok(Self {
            node_id,
            ingress_names,
            inputs,
            progress,
            alignment: OperatorBarrierAlignment::default(),
            ended: false,
            #[cfg(test)]
            panic_on_next_dispatch: false,
        })
    }

    pub(in crate::runtime::streaming) fn node_id(&self) -> &str {
        &self.node_id
    }

    #[cfg(test)]
    pub(in crate::runtime::streaming) fn checkpointing(&self) -> bool {
        self.inputs.context.job().checkpointing()
    }

    #[cfg(test)]
    pub(in crate::runtime::streaming) fn panic_on_next_dispatch(&mut self) {
        self.panic_on_next_dispatch = true;
    }

    pub(in crate::runtime::streaming) fn checkpoint(
        &mut self,
        epoch: crate::Epoch,
    ) -> Result<crate::OperatorStateSnapshot> {
        self.inputs.operator.checkpoint(epoch)
    }

    pub(in crate::runtime::streaming) fn restore(
        &mut self,
        snapshot: &crate::OperatorStateSnapshot,
    ) -> Result<()> {
        self.inputs.operator.restore(snapshot)
    }

    pub(in crate::runtime::streaming) fn is_sync(&self) -> bool {
        sync_adapter::is_sync(&self.inputs.operator)
    }

    fn next_sync_envelope(
        &mut self,
        ingress: usize,
    ) -> Result<Option<super::super::local_edge::SessionEnvelope>> {
        let name = &self.ingress_names[ingress];
        self.inputs
            .ingresses
            .get_mut(name)
            .and_then(|ingress| ingress.receiver.as_mut())
            .ok_or_else(|| CalcFlowError::Internal {
                message: format!("operator {:?} lost ingress {name:?}", self.node_id),
            })?
            .try_recv_envelope()
    }

    fn dispatch_sync(&mut self, ingress: usize, message: StreamMessage) -> Result<Option<Batch>> {
        #[cfg(test)]
        let inject_panic = std::mem::take(&mut self.panic_on_next_dispatch);
        match catch_unwind(AssertUnwindSafe(|| {
            #[cfg(test)]
            assert!(!inject_panic, "session dispatch test seam");
            self.dispatch_sync_inner(ingress, message)
        })) {
            Ok(result) => result,
            Err(payload) => Err(panicked(&self.node_id, payload.as_ref())),
        }
    }

    fn dispatch_sync_inner(
        &mut self,
        ingress: usize,
        message: StreamMessage,
    ) -> Result<Option<Batch>> {
        let Self {
            inputs,
            progress,
            alignment,
            ingress_names,
            ..
        } = self;
        let ingress = ingress_names[ingress].as_str();
        match message.kind() {
            StreamMessageKind::Data => sync_data(inputs, progress, ingress, message),
            StreamMessageKind::Watermark => sync_progress(
                inputs,
                progress,
                ingress,
                AggregateInput::Watermark(
                    message
                        .as_watermark()
                        .expect("watermark kind carries event time"),
                ),
            ),
            StreamMessageKind::Idle => {
                sync_progress(inputs, progress, ingress, AggregateInput::Idle)
            }
            StreamMessageKind::EndOfInput => sync_end(inputs, progress, ingress),
            StreamMessageKind::Barrier => {
                dispatch_barrier(inputs, ingress, &message, alignment)?;
                Ok(None)
            }
        }
    }

    fn finish_sync_when_ended(&mut self) -> Result<()> {
        let all_ended = self
            .inputs
            .ingresses
            .values()
            .all(|ingress| ingress.saw_explicit_eof);
        if self.ended || !all_ended {
            return Ok(());
        }
        let result = catch_unwind(AssertUnwindSafe(|| {
            sync_adapter::end_session(
                &mut self.inputs,
                self.progress.input_watermark(),
                self.progress.snapshot()?,
            )
        }));
        match result {
            Ok(result) => result?,
            Err(payload) => return Err(panicked(&self.node_id, payload.as_ref())),
        }
        self.ended = true;
        Ok(())
    }

    /// Empties every ingress queue in FIFO order, then runs `on_end` once
    /// every ingress has seen end-of-input. Panics become operator errors.
    #[allow(
        clippy::large_futures,
        reason = "one dispatch is polled at a time on the session's own runtime; boxing would allocate per message on the push path"
    )]
    pub(in crate::runtime::streaming) async fn drain(&mut self) -> Result<()> {
        for index in 0..self.ingress_names.len() {
            self.drain_ingress(index).await?;
        }
        self.finish_when_ended().await
    }

    #[allow(
        clippy::large_futures,
        reason = "one dispatch is polled at a time on the session's own runtime; boxing would allocate per message on the push path"
    )]
    async fn drain_ingress(&mut self, index: usize) -> Result<()> {
        let Self {
            node_id,
            ingress_names,
            inputs,
            progress,
            alignment,
            #[cfg(test)]
            panic_on_next_dispatch,
            ..
        } = self;
        let name = &ingress_names[index];
        while let Some(message) = next_message(inputs, node_id, name)? {
            #[cfg(test)]
            let inject_panic = std::mem::take(panic_on_next_dispatch);
            let dispatch = async {
                #[cfg(test)]
                assert!(!inject_panic, "session dispatch test seam");
                dispatch_message(inputs, name, message, progress, alignment).await
            };
            contain(node_id, dispatch).await?;
            ingress_admission::acknowledge(inputs, name, progress);
        }
        Ok(())
    }

    /// Runs `on_end` once, after every ingress has seen end-of-input.
    #[allow(
        clippy::large_futures,
        reason = "on_end is polled in place on the session's own runtime"
    )]
    async fn finish_when_ended(&mut self) -> Result<()> {
        let all_ended = self
            .inputs
            .ingresses
            .values()
            .all(|ingress| ingress.saw_explicit_eof);
        if self.ended || !all_ended {
            return Ok(());
        }
        let completed_epoch = self.alignment.completed_epoch;
        let finish = finish_operator(&mut self.inputs, &self.progress, completed_epoch);
        contain(&self.node_id, finish).await?;
        self.ended = true;
        Ok(())
    }
}

/// Runs an all-sync graph without constructing or polling an async future.
pub(in crate::runtime::streaming) fn drain_in_order_sync(
    nodes: &mut [CallerDrivenOperator],
) -> Result<()> {
    for node in 0..nodes.len() {
        let ingresses = nodes[node].ingress_names.len();
        for ingress in 0..ingresses {
            while let Some(envelope) = nodes[node].next_sync_envelope(ingress)? {
                let spent = nodes[node].dispatch_sync(ingress, envelope.message)?;
                if let (Some(producer), Some(batch)) = (envelope.producer, spent) {
                    sync_adapter::recycle(&mut nodes[producer].inputs.operator, batch);
                }
            }
        }
        nodes[node].finish_sync_when_ended()?;
    }
    Ok(())
}

fn sync_data(
    inputs: &mut OperatorTaskInputs,
    progress: &mut OperatorInputProgress,
    ingress: &str,
    message: StreamMessage,
) -> Result<Option<Batch>> {
    let transition = control::evaluate_data(progress, ingress)?;
    sync_adapter::data_session(
        inputs,
        &message,
        transition.input_watermark,
        transition.ingress_progress,
    )?;
    Ok(message.into_data())
}

fn sync_end(
    inputs: &mut OperatorTaskInputs,
    progress: &mut OperatorInputProgress,
    ingress: &str,
) -> Result<Option<Batch>> {
    sync_progress(inputs, progress, ingress, AggregateInput::End)?;
    inputs
        .ingresses
        .get_mut(ingress)
        .expect("session ingress remains registered")
        .saw_explicit_eof = true;
    Ok(None)
}

fn sync_progress(
    inputs: &mut OperatorTaskInputs,
    progress: &mut OperatorInputProgress,
    ingress: &str,
    input: AggregateInput,
) -> Result<Option<Batch>> {
    let transition = control::evaluate_progress(progress, ingress, input)?;
    sync_adapter::progress_session(
        inputs,
        transition.emissions,
        transition.previous_watermark,
        &transition.ingress_progress,
        transition.aggregate_input_frontier,
        &mut progress.output_frontier,
    )?;
    Ok(None)
}

/// Takes the front message of one session ingress.
fn next_message(
    inputs: &mut OperatorTaskInputs,
    node_id: &str,
    name: &str,
) -> Result<Option<StreamMessage>> {
    inputs
        .ingresses
        .get_mut(name)
        .and_then(|ingress| ingress.receiver.as_mut())
        .ok_or_else(|| CalcFlowError::Internal {
            message: format!("operator {node_id:?} lost ingress {name:?}"),
        })?
        .try_recv()
}

async fn contain(node_id: &str, operation: impl Future<Output = Result<()>>) -> Result<()> {
    match AssertUnwindSafe(operation).catch_unwind().await {
        Ok(result) => result,
        Err(payload) => Err(panicked(node_id, payload.as_ref())),
    }
}

fn panicked(node_id: &str, payload: &(dyn std::any::Any + Send)) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: node_id.into(),
        message: format!("panicked: {}", panic_message(payload)),
    }
}
