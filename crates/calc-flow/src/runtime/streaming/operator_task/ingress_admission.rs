//! One owned prepare task per Join/ASOF ingress. Envelopes become ready before
//! their facts, so receive arbitration never waits for preparation completion.
//! Read-ahead is one envelope and at most 64 fixed-size classification blocks.

#[cfg(test)]
use super::OperatorProgress;
use super::{
    AssertUnwindSafe, Batch, CalcFlowError, CancellationToken, CompiledStreamOperator, EventTime,
    FutureExt, IngressProgress, OperatorCooperation, OperatorEdgeReceiver, OperatorInputProgress,
    OperatorTaskInputs, Result, StreamMessage, StreamMessageKind, TaskFailureSignal, TaskId, mpsc,
    run_operator_loop,
};
use crate::operator::{
    AsofIngressPlan, JoinIngressPlan,
    ingress::{Preparation, Sender},
};
use tokio::{sync::oneshot, task::JoinSet};

enum Plan {
    Join(JoinIngressPlan),
    Asof(AsofIngressPlan),
}

impl Plan {
    async fn prepare(
        &self,
        batch: &Batch,
        watermark: Option<EventTime>,
        output: &Sender,
    ) -> Result<()> {
        match (self, output) {
            (Self::Join(plan), Sender::Join(output)) => plan.prepare(batch, output).await,
            (Self::Asof(plan), Sender::Asof(output)) => {
                let facts = plan.prepare(batch, watermark).await?;
                let _ = output.send(Ok(facts)).await;
                Ok(())
            }
            _ => Err(CalcFlowError::Internal {
                message: "ingress plan does not match its fact sender".into(),
            }),
        }
    }

    fn channel(&self) -> (Sender, Preparation) {
        match self {
            Self::Join(_) => Preparation::join_channel(),
            Self::Asof(_) => Preparation::asof_channel(),
        }
    }
}

struct Envelope {
    message: StreamMessage,
    preparation: Option<Preparation>,
    consumed: oneshot::Sender<Option<EventTime>>,
}

pub(super) struct Receiver {
    ready: mpsc::Receiver<Result<Envelope>>,
    preparation: Option<Preparation>,
    consumed: Option<oneshot::Sender<Option<EventTime>>>,
}

impl Receiver {
    pub(super) async fn recv(&mut self) -> Result<Option<StreamMessage>> {
        let Some(envelope) = self.ready.recv().await else {
            return Ok(None);
        };
        let envelope = envelope?;
        self.preparation = envelope.preparation;
        self.consumed = Some(envelope.consumed);
        Ok(Some(envelope.message))
    }

    pub(super) fn take_preparation(&mut self) -> Option<Preparation> {
        self.preparation.take()
    }

    pub(super) fn acknowledge(&mut self, watermark: Option<EventTime>) {
        if let Some(consumed) = self.consumed.take() {
            let _ = consumed.send(watermark);
        }
    }
}

async fn prepare_ingress(
    mut source: OperatorEdgeReceiver,
    plan: Plan,
    ready: mpsc::Sender<Result<Envelope>>,
    mut watermark: Option<EventTime>,
    #[cfg(test)] observation: &Observation,
) {
    loop {
        let message = match source.recv().await {
            Ok(Some(message)) => message,
            Ok(None) => return,
            Err(error) => {
                let _ = ready.send(Err(error)).await;
                return;
            }
        };
        let eof = message.kind() == StreamMessageKind::EndOfInput;
        let batch = message.as_data().cloned();
        let (facts, preparation) = if batch.is_some() {
            let (sender, preparation) = plan.channel();
            (Some(sender), Some(preparation))
        } else {
            (None, None)
        };
        let (consumed, resume) = oneshot::channel();
        if ready
            .send(Ok(Envelope {
                message,
                preparation,
                consumed,
            }))
            .await
            .is_err()
        {
            return;
        }
        if let (Some(batch), Some(facts)) = (batch, facts) {
            prepare_data(&plan, &batch, watermark, &facts).await;
            #[cfg(test)]
            observation.prepared();
        }
        match resume.await {
            Ok(next) if !eof => watermark = next,
            _ => return,
        }
    }
}

#[tracing::instrument(name = "ingress.admission", level = "debug", skip_all, fields(rows = batch.num_rows()))]
async fn prepare_data(plan: &Plan, batch: &Batch, watermark: Option<EventTime>, facts: &Sender) {
    if let Err(error) = plan.prepare(batch, watermark, facts).await {
        facts.error(error).await;
    }
}

struct Tasks {
    stop: CancellationToken,
    tasks: JoinSet<()>,
}

#[cfg(test)]
struct Observation {
    name: String,
    progress: OperatorProgress,
}

#[cfg(test)]
impl Observation {
    fn new(name: String, progress: OperatorProgress) -> Self {
        progress.0.lock().preparation_tasks_started += 1;
        Self { name, progress }
    }

    fn prepared(&self) {
        *self
            .progress
            .0
            .lock()
            .prepared_ingresses
            .entry(self.name.clone())
            .or_default() += 1;
    }
}

#[cfg(test)]
impl Drop for Observation {
    fn drop(&mut self) {
        self.progress.0.lock().preparation_tasks_exited += 1;
    }
}

impl Tasks {
    fn start(inputs: &mut OperatorTaskInputs, progress: &OperatorInputProgress) -> Result<Self> {
        let plans = inputs
            .ingresses
            .iter()
            .filter(|(_, ingress)| !ingress.saw_explicit_eof)
            .map(|(name, _)| {
                let plan = match &inputs.operator {
                    CompiledStreamOperator::StreamJoin(operator) => {
                        Some(Plan::Join(operator.ingress_plan(name)?))
                    }
                    CompiledStreamOperator::StreamAsofJoin(operator) => {
                        Some(Plan::Asof(operator.ingress_plan(name)?))
                    }
                    _ => None,
                };
                Ok((name.clone(), plan))
            })
            .collect::<Result<Vec<_>>>()?;
        let progress = progress.snapshot()?;
        let stop = CancellationToken::new();
        let mut tasks = JoinSet::new();
        for (name, plan) in plans {
            let Some(plan) = plan else { continue };
            let ingress = inputs.ingresses.get_mut(&name).expect("planned ingress");
            let source = ingress.receiver.take().expect("unprepared ingress");
            let (sender, ready) = mpsc::channel(1);
            ingress.admission = Some(Receiver {
                ready,
                preparation: None,
                consumed: None,
            });
            let stop = stop.clone();
            let cancellation = inputs.context.job().cancellation().clone();
            let watermark = progress.get(&name).and_then(IngressProgress::watermark);
            #[cfg(test)]
            let observation_progress = inputs.progress.clone();
            tasks.spawn(async move {
                #[cfg(test)]
                let observation = Observation::new(name, observation_progress);
                tokio::select! {
                    biased;
                    () = stop.cancelled() => {},
                    () = cancellation.cancelled() => {},
                    () = prepare_ingress(source, plan, sender, watermark, #[cfg(test)] &observation) => {},
                }
            });
        }
        Ok(Self { stop, tasks })
    }

    async fn finish(mut self, task_id: Option<TaskId>) -> Result<()> {
        self.stop.cancel();
        let mut failure = None;
        while let Some(result) = self.tasks.join_next().await {
            if let Err(error) = result {
                failure.get_or_insert_with(|| CalcFlowError::TaskPanicked {
                    task_id: task_id.map_or(0, TaskId::as_u64),
                    message: error.to_string(),
                });
            }
        }
        failure.map_or(Ok(()), Err)
    }
}

pub(super) async fn run(
    inputs: &mut OperatorTaskInputs,
    progress: OperatorInputProgress,
    cooperation: OperatorCooperation,
    failure_signal: Option<&TaskFailureSignal>,
) -> Result<()> {
    if !matches!(
        inputs.operator,
        CompiledStreamOperator::StreamJoin(_) | CompiledStreamOperator::StreamAsofJoin(_)
    ) {
        return run_operator_loop(inputs, progress, cooperation).await;
    }
    let tasks = Tasks::start(inputs, &progress)?;
    let result = AssertUnwindSafe(run_operator_loop(inputs, progress, cooperation))
        .catch_unwind()
        .await;
    if !matches!(&result, Ok(Ok(())))
        && let Some(signal) = failure_signal
    {
        // Preparation workers own the upstream receivers. Publish failure
        // and cancel siblings before cleanup can close those edges.
        signal.converge_after_failure().await;
    }
    let cleanup = tasks.finish(inputs.context.task_id()).await;
    match result {
        Ok(result) => cleanup.and(result),
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

pub(super) fn acknowledge(
    inputs: &mut OperatorTaskInputs,
    ingress_name: &str,
    progress: &OperatorInputProgress,
) {
    if let Some(admission) = inputs
        .ingresses
        .get_mut(ingress_name)
        .and_then(|ingress| ingress.admission.as_mut())
    {
        let watermark = progress
            .cached_snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.get(ingress_name))
            .and_then(IngressProgress::watermark);
        admission.acknowledge(watermark);
    }
}

#[cfg(test)]
mod tests {
    use super::{CalcFlowError, CancellationToken, JoinSet, Tasks, oneshot};

    #[tokio::test]
    async fn test_ingress_admission_panic_cleanup_joins_every_worker() {
        let stop = CancellationToken::new();
        let waiting = stop.clone();
        let mut tasks = JoinSet::new();
        tasks.spawn(async { panic!("prepare worker failed") });
        let (exited, done) = oneshot::channel();
        tasks.spawn(async move {
            waiting.cancelled().await;
            exited.send(()).unwrap();
        });
        let result = Tasks { stop, tasks }.finish(None).await;
        assert!(matches!(result, Err(CalcFlowError::TaskPanicked { .. })));
        done.await
            .expect("other workers must be joined before reporting panic");
    }
}
