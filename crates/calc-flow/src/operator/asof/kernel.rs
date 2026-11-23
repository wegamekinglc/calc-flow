//! Cancellation and bounded output shared by finite calls and stream callbacks.
use crate::{CancellationToken, EdgeBudget, Result, RunContext, StreamOperatorContext};

pub(super) trait KernelContext: Sync {
    fn check_cancelled(&self) -> Result<()>;
    fn cancellation(&self) -> &CancellationToken;
    fn ingress_preparation(&self) -> Option<&crate::operator::ingress::Preparation>;
    fn output_budget(&self) -> EdgeBudget;
}

impl KernelContext for StreamOperatorContext<'_> {
    fn check_cancelled(&self) -> Result<()> {
        self.check_cancelled()
    }

    fn cancellation(&self) -> &CancellationToken {
        self.job().cancellation()
    }

    fn ingress_preparation(&self) -> Option<&crate::operator::ingress::Preparation> {
        StreamOperatorContext::ingress_preparation(self)
    }

    fn output_budget(&self) -> EdgeBudget {
        self.output_budget()
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
    fn check_cancelled(&self) -> Result<()> {
        self.run.check_cancelled()
    }

    fn cancellation(&self) -> &CancellationToken {
        self.run.cancellation()
    }

    fn ingress_preparation(&self) -> Option<&crate::operator::ingress::Preparation> {
        None
    }

    fn output_budget(&self) -> EdgeBudget {
        self.output
    }
}
