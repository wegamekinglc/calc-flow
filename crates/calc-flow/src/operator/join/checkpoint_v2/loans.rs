use std::sync::Arc;

use super::super::{AdmittedBatch, SidePlan, SqlKeyOwners, StreamJoinOperator, decode_key_pairs};
use super::ContainerFunding;
use crate::Result;
use crate::datafusion::owned;

pub(in crate::operator::join) struct V2SqlOwners {
    inner: SqlKeyOwners,
    _containers: Arc<ContainerFunding>,
}

impl V2SqlOwners {
    pub(in crate::operator::join) fn new(
        inner: SqlKeyOwners,
        containers: Arc<ContainerFunding>,
    ) -> Self {
        Self {
            inner,
            _containers: containers,
        }
    }

    fn finish(self) -> AdmittedBatch {
        let Self {
            inner,
            _containers: containers,
        } = self;
        let admitted = inner.finish();
        drop(containers);
        admitted
    }
}

impl StreamJoinOperator {
    pub(in crate::operator::join) async fn run_v2_owned_key_query(
        &mut self,
        plan: &SidePlan,
        input: owned::Input<V2SqlOwners>,
        admitted: &mut AdmittedBatch,
    ) -> Result<Vec<(u64, u64)>> {
        let result = self
            .runtime
            .runtime()
            .expect("initialized by scratch construction")
            .sql_equality_owned(&self.compiled.equality_query, input, Some(&self.name))
            .await;
        match result {
            Ok(result) => {
                let pairs = decode_key_pairs(result.batch());
                *admitted = result.finish().finish();
                pairs
            }
            Err(failure) => {
                self.finish_v2_owned_key_failure(failure, admitted)?;
                self.legacy_key_retry(plan, admitted).await
            }
        }
    }

    fn finish_v2_owned_key_failure(
        &mut self,
        failure: owned::Failure<V2SqlOwners>,
        admitted: &mut AdmittedBatch,
    ) -> Result<()> {
        let retry = failure
            .error
            .retry_legacy(failure.input.owner().inner.scratch.is_some());
        let (error, input) = failure.into_parts(Some(&self.name));
        *admitted = input.finish().finish();
        if retry {
            Ok(())
        } else {
            Err(error.expect("non-retry failure keeps its source"))
        }
    }
}
