//! Cancellation and bounded work shared by finite calls and stream callbacks.
use crate::runtime::streaming::gather_work::{GatherClient, GatherOperatorId, GatherStop};
use crate::{CancellationToken, EdgeBudget, Result, RunContext, StreamOperatorContext};

pub(super) trait KernelContext: Sync {
    fn stream(&self) -> Option<&StreamOperatorContext<'_>>;
    fn check_cancelled(&self) -> Result<()>;
    fn cancellation(&self) -> &CancellationToken;
    fn deadline(&self) -> Option<&chrono::DateTime<chrono::Utc>>;
    fn run_id(&self) -> String;
    fn stop(&self) -> GatherStop;
    fn checkpointing(&self) -> bool {
        self.stream().is_some_and(|c| c.job().checkpointing())
    }
    fn ingress_preparation(&self) -> Option<&crate::operator::ingress::Preparation> {
        self.stream()
            .and_then(StreamOperatorContext::ingress_preparation)
    }
    fn output_budget(&self) -> EdgeBudget;
    fn gather_client(&self, operator: GatherOperatorId) -> GatherClient {
        self.stream()
            .expect("stream worker dispatch")
            .gather_client(operator)
    }
    fn parallelism(&self) -> usize {
        self.stream()
            .map_or(1, |c| c.job().gather_owner().parallelism())
    }
    #[cfg(feature = "benchmark-internals")]
    fn trace(&self) -> &crate::benchmark_internals::JoinCpuTraceOwner {
        static DISABLED: std::sync::OnceLock<crate::benchmark_internals::JoinCpuTraceOwner> =
            std::sync::OnceLock::new();
        self.stream().map_or_else(
            || DISABLED.get_or_init(Default::default),
            |c| c.job().join_cpu_trace(),
        )
    }
}
impl KernelContext for StreamOperatorContext<'_> {
    fn stream(&self) -> Option<&StreamOperatorContext<'_>> {
        Some(self)
    }
    fn output_budget(&self) -> EdgeBudget {
        self.output_budget()
    }
    fn check_cancelled(&self) -> Result<()> {
        self.check_cancelled()
    }
    fn cancellation(&self) -> &CancellationToken {
        self.job().cancellation()
    }
    fn deadline(&self) -> Option<&chrono::DateTime<chrono::Utc>> {
        self.job().deadline()
    }
    fn run_id(&self) -> String {
        self.job().job_id().to_string()
    }
    fn stop(&self) -> GatherStop {
        GatherStop::from_job(self.job())
    }
}
pub(super) struct FiniteContext<'a> {
    run: &'a RunContext,
    output: EdgeBudget,
}
impl<'a> FiniteContext<'a> {
    pub(super) fn new(run: &'a RunContext, max_bytes: u64) -> Self {
        Self {
            run,
            output: EdgeBudget {
                max_rows: 8192,
                max_bytes: usize::try_from(max_bytes)
                    .expect("constructor verified ASOF byte address domain"),
            },
        }
    }
}
impl KernelContext for FiniteContext<'_> {
    fn stream(&self) -> Option<&StreamOperatorContext<'_>> {
        None
    }
    fn output_budget(&self) -> EdgeBudget {
        self.output
    }
    fn check_cancelled(&self) -> Result<()> {
        self.run.check_cancelled()
    }
    fn cancellation(&self) -> &CancellationToken {
        self.run.cancellation()
    }
    fn deadline(&self) -> Option<&chrono::DateTime<chrono::Utc>> {
        self.run.deadline()
    }
    fn run_id(&self) -> String {
        self.run.run_id().to_owned()
    }
    fn stop(&self) -> GatherStop {
        GatherStop::from_run(self.run)
    }
}
