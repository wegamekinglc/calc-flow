//! One operator driven by `StreamSession` on the caller's thread: the runner's
//! per-message dispatch and end-of-input code without a task, gates or waits.

use super::{
    AssertUnwindSafe, CalcFlowError, FutureExt, OperatorBarrierAlignment, OperatorInputProgress,
    OperatorTaskInputs, Result, catch_unwind, dispatch_message, finish_operator, ingress_admission,
    panic_message,
};

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
    /// Resets the operator like runner entry does for a fresh (unrestored) job.
    pub(in crate::runtime::streaming) fn new(mut inputs: OperatorTaskInputs) -> Result<Self> {
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
    pub(in crate::runtime::streaming) fn panic_on_next_dispatch(&mut self) {
        self.panic_on_next_dispatch = true;
    }

    /// Empties every ingress queue in FIFO order, then runs `on_end` once
    /// every ingress has seen end-of-input. Panics become operator errors.
    #[allow(
        clippy::large_futures,
        reason = "one dispatch is polled at a time on the session's own runtime; boxing would allocate per message on the push path"
    )]
    pub(in crate::runtime::streaming) async fn drain(&mut self) -> Result<()> {
        let Self {
            node_id,
            ingress_names,
            inputs,
            progress,
            alignment,
            ended,
            #[cfg(test)]
            panic_on_next_dispatch,
        } = self;
        for name in ingress_names.iter() {
            while let Some(message) = inputs
                .ingresses
                .get_mut(name)
                .and_then(|ingress| ingress.receiver.as_mut())
                .ok_or_else(|| CalcFlowError::Internal {
                    message: format!("operator {node_id:?} lost ingress {name:?}"),
                })?
                .try_recv()?
            {
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
        }
        if !*ended
            && inputs
                .ingresses
                .values()
                .all(|ingress| ingress.saw_explicit_eof)
        {
            let completed_epoch = alignment.completed_epoch;
            contain(node_id, finish_operator(inputs, progress, completed_epoch)).await?;
            *ended = true;
        }
        Ok(())
    }
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
