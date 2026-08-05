use std::{future::Future, time::Duration};

use chrono::Utc;

use super::{
    EdgeSender, StreamJobContext,
    source_task::{SourceBinding, SourceProgress, spawn_source_tasks},
    supervisor::{TaskFailure, TaskId, TaskSupervisor},
};
use crate::{CalcFlowError, Result};

/// Internal lifecycle state used until M2.4 replaces the public runner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ContinuousJobState {
    Running,
    #[allow(dead_code, reason = "graceful draining is added with M2.4 sinks")]
    Draining,
    Completed,
    Cancelled,
    Failed,
    RecoveryRequired,
}

#[derive(Debug)]
pub(crate) struct ContinuousJobOutcome {
    pub(crate) state: ContinuousJobState,
    pub(crate) errors: Vec<TaskFailure>,
}

/// Minimal owning continuous-job skeleton for M2.1/M2.2.
///
/// The value is deliberately crate-private and terminal methods consume it.
/// Public idempotent job handles, Drop transfer, and the runner-scoped reaper
/// are implemented together in M2.4 instead of being approximated here.
pub(crate) struct ContinuousJob {
    context: StreamJobContext,
    supervisor: TaskSupervisor,
    state: ContinuousJobState,
}

impl ContinuousJob {
    pub(crate) fn new(context: StreamJobContext) -> Self {
        let supervisor = TaskSupervisor::new(context.cancellation().clone());
        Self {
            context,
            supervisor,
            state: ContinuousJobState::Running,
        }
    }

    pub(crate) const fn status(&self) -> ContinuousJobState {
        self.state
    }

    pub(crate) fn spawn<F>(&mut self, name: impl Into<String>, future: F) -> TaskId
    where
        F: Future<Output = Result<()>> + Send + 'static,
    {
        self.supervisor.spawn(name, future)
    }

    pub(crate) fn spawn_source(
        &mut self,
        binding_id: &str,
        binding: SourceBinding,
        outputs: Vec<EdgeSender>,
    ) -> Result<SourceProgress> {
        spawn_source_tasks(
            &mut self.supervisor,
            &self.context,
            binding_id,
            binding,
            outputs,
        )
    }

    pub(crate) async fn wait(mut self) -> ContinuousJobOutcome {
        let deadline = self.context.deadline().copied();
        let deadline_wait = async move {
            match deadline {
                Some(deadline) => {
                    let delay = deadline
                        .signed_duration_since(Utc::now())
                        .to_std()
                        .unwrap_or(Duration::ZERO);
                    tokio::time::sleep(delay).await;
                }
                None => std::future::pending().await,
            }
        };
        tokio::pin!(deadline_wait);
        let report = {
            let join = self.supervisor.join_all();
            tokio::pin!(join);
            tokio::select! {
                biased;
                report = &mut join => Some(report),
                () = &mut deadline_wait => None,
            }
        };
        let report = if let Some(report) = report {
            report
        } else {
            self.supervisor.cancel();
            self.supervisor.join_all().await
        };
        let cancelled = self.context.cancellation().is_cancelled() && report.errors.is_empty();
        outcome(report.errors, cancelled)
    }

    pub(crate) async fn cancel(mut self) -> ContinuousJobOutcome {
        self.supervisor.cancel();
        let report = self.supervisor.join_all().await;
        outcome(report.errors, true)
    }
}

fn outcome(errors: Vec<TaskFailure>, cancelled: bool) -> ContinuousJobOutcome {
    let state = match errors.first() {
        Some(failure) if is_recovery_required(&failure.error) => {
            ContinuousJobState::RecoveryRequired
        }
        Some(_) => ContinuousJobState::Failed,
        None if cancelled => ContinuousJobState::Cancelled,
        None => ContinuousJobState::Completed,
    };
    ContinuousJobOutcome { state, errors }
}

fn is_recovery_required(error: &CalcFlowError) -> bool {
    matches!(
        error,
        CalcFlowError::Io { .. }
            | CalcFlowError::ExternalProvider { .. }
            | CalcFlowError::RecoveryRequired { .. }
    )
}

#[cfg(test)]
mod tests {
    use std::{
        any::Any,
        collections::{BTreeMap, VecDeque},
        future::pending,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    };

    use async_trait::async_trait;
    use serde_json::json;

    use super::{ContinuousJob, ContinuousJobState};
    use crate::{
        Batch, BatchMetadata, CancellationToken, EdgeBudget, ExternalPayload, JsonMap, Result,
        StreamJobContext, StreamMessageKind, edge_channel,
        runtime::streaming::{
            context::StreamTaskKind,
            source_task::{Cursor, SourceBinding, SourceCapabilities, SourceEvent, StreamSource},
        },
    };

    #[derive(Debug)]
    struct Payload;

    impl ExternalPayload for Payload {
        fn backend(&self) -> &'static str {
            "continuous-job-test"
        }

        fn len(&self) -> usize {
            1
        }

        fn estimated_bytes(&self) -> usize {
            1
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn batch() -> Batch {
        Batch::external(Arc::new(Payload), BatchMetadata::default()).unwrap()
    }

    fn cursor(position: u8) -> Cursor {
        Cursor::new(
            vec![position],
            BTreeMap::from([("at".into(), json!(position))]),
        )
        .unwrap()
    }

    fn context(cancellation: CancellationToken) -> StreamJobContext {
        StreamJobContext::new(42, "fingerprint", JsonMap::new(), None, cancellation)
    }

    struct FiniteSource {
        events: VecDeque<Option<SourceEvent>>,
        closed: Arc<AtomicBool>,
    }

    #[async_trait]
    impl StreamSource for FiniteSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            Ok(self.events.pop_front().flatten())
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replayable: true,
                max_batch_rows: 1,
                max_batch_bytes: 1,
            }
        }
    }

    #[tokio::test]
    async fn continuous_job_owns_a_runnable_source_to_recording_consumer_slice() {
        let cancellation = CancellationToken::new();
        let mut job = ContinuousJob::new(context(cancellation));
        let closed = Arc::new(AtomicBool::new(false));
        let source = FiniteSource {
            events: VecDeque::from([
                Some(SourceEvent::Data {
                    batch: batch(),
                    cursor: cursor(1),
                }),
                Some(SourceEvent::Watermark(crate::EventTime::from_micros(10))),
                Some(SourceEvent::Idle),
                Some(SourceEvent::Data {
                    batch: batch(),
                    cursor: cursor(2),
                }),
                None,
            ]),
            closed: Arc::clone(&closed),
        };
        let (sender, mut receiver) = edge_channel(
            "source->recording-consumer",
            EdgeBudget {
                max_rows: 1,
                max_bytes: 1,
            },
        )
        .unwrap();
        job.spawn_source(
            "input",
            SourceBinding::new(Box::new(source), None, 0).unwrap(),
            vec![sender],
        )
        .unwrap();
        let recorded = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let recorded_in_task = Arc::clone(&recorded);
        job.spawn("recording-consumer", async move {
            while let Some(message) = receiver.recv().await? {
                if let Some(data) = message.as_data() {
                    recorded_in_task.lock().push(data.metadata().sequence());
                }
                match message.kind() {
                    StreamMessageKind::Watermark => {
                        recorded_in_task.lock().push(u64::MAX - 2);
                    }
                    StreamMessageKind::Idle => recorded_in_task.lock().push(u64::MAX - 1),
                    StreamMessageKind::EndOfInput => {
                        recorded_in_task.lock().push(u64::MAX);
                        break;
                    }
                    StreamMessageKind::Data | StreamMessageKind::Barrier => {}
                }
            }
            Ok(())
        });

        assert_eq!(job.status(), ContinuousJobState::Running);
        let outcome = job.wait().await;

        assert_eq!(outcome.state, ContinuousJobState::Completed);
        assert!(outcome.errors.is_empty());
        assert_eq!(
            *recorded.lock(),
            [0, u64::MAX - 2, u64::MAX - 1, 1, u64::MAX]
        );
        assert!(closed.load(Ordering::SeqCst));
    }

    struct BlockingSource {
        closed: Arc<AtomicBool>,
    }

    #[async_trait]
    impl StreamSource for BlockingSource {
        async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
            Ok(())
        }

        async fn next(&mut self) -> Result<Option<SourceEvent>> {
            pending().await
        }

        async fn close(&mut self) -> Result<()> {
            self.closed.store(true, Ordering::SeqCst);
            Ok(())
        }

        fn capabilities(&self) -> SourceCapabilities {
            SourceCapabilities {
                replayable: false,
                max_batch_rows: 1,
                max_batch_bytes: 1,
            }
        }
    }

    #[tokio::test]
    async fn explicit_cancel_joins_the_source_and_consumer() {
        let cancellation = CancellationToken::new();
        let mut job = ContinuousJob::new(context(cancellation));
        let closed = Arc::new(AtomicBool::new(false));
        let (sender, mut receiver) =
            edge_channel("source->consumer", EdgeBudget::default()).unwrap();
        job.spawn_source(
            "quiet",
            SourceBinding::new(
                Box::new(BlockingSource {
                    closed: Arc::clone(&closed),
                }),
                None,
                0,
            )
            .unwrap(),
            vec![sender],
        )
        .unwrap();
        let consumer_finished = Arc::new(AtomicBool::new(false));
        let consumer_finished_in_task = Arc::clone(&consumer_finished);
        job.spawn("consumer", async move {
            while receiver.recv().await?.is_some() {}
            consumer_finished_in_task.store(true, Ordering::SeqCst);
            Ok(())
        });

        let outcome = job.cancel().await;

        assert_eq!(outcome.state, ContinuousJobState::Cancelled);
        assert!(outcome.errors.is_empty());
        assert!(closed.load(Ordering::SeqCst));
        assert!(consumer_finished.load(Ordering::SeqCst));
    }

    #[test]
    fn job_context_derives_validated_source_node_and_sink_scopes() {
        let context = context(CancellationToken::new());

        let source = context.for_source("input").unwrap();
        let node = context.for_node("normalize").unwrap();
        let sink = context.for_sink("output").unwrap();

        assert_eq!(source.kind(), StreamTaskKind::Source);
        assert_eq!(node.kind(), StreamTaskKind::Node);
        assert_eq!(sink.kind(), StreamTaskKind::Sink);
        assert_eq!(source.scope_id(), "input");
        assert_eq!(source.job().job_id(), context.job_id());
        assert!(context.for_source("  ").is_err());
    }
}
