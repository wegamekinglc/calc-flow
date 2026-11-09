//! A separately admitted identity lease survives every candidate-phase adjustment.

use std::sync::Arc;

use super::{Resources, math_requirements::bytes};
use crate::operator::rolling::operator_error;
use crate::{Result, StateBudget, array::RollingWorkerIdentity};

pub(super) fn reserve(
    resources: &Resources,
    node: &str,
    additional: usize,
    budget: StateBudget,
) -> Result<Arc<RollingWorkerIdentity>> {
    let size = allocation_bytes(node, additional)?;
    let lease = Arc::new(resources.reserve(0, size, budget, node)?);
    Ok(RollingWorkerIdentity::new(node, lease))
}

pub(super) fn allocation_bytes(node: &str, additional: usize) -> Result<u64> {
    let identity = RollingWorkerIdentity::allocation_bytes(node)
        .ok_or_else(|| operator_error(node, "state.work_metadata byte bound overflowed"))?;
    bytes::add(
        identity,
        u64::try_from(additional).map_err(|_| bytes::invalid())?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CancellationToken, JsonMap, StreamJobContext, array::RollingScope};

    struct Lease {
        credit: Option<super::super::resources::Credit>,
        settled: Option<tokio::sync::oneshot::Sender<()>>,
    }
    impl Drop for Lease {
        fn drop(&mut self) {
            drop(self.credit.take());
            if let Some(settled) = self.settled.take() {
                let _ = settled.send(());
            }
        }
    }
    #[tokio::test]
    async fn test_lost_worker_waiter_keeps_identity_credit_until_real_settlement() {
        let resources = Resources::default();
        let bytes = allocation_bytes("features", 4096).unwrap();
        let budget = StateBudget::new(8, bytes).unwrap();
        let small = StateBudget::new(8, bytes - 1).unwrap();
        let (settled, receipt) = tokio::sync::oneshot::channel();
        let credit = resources.reserve(0, bytes, budget, "features").unwrap();
        let identity = RollingWorkerIdentity::new(
            "features",
            Arc::new(Lease {
                credit: Some(credit),
                settled: Some(settled),
            }),
        );
        let (started, ready) = tokio::sync::oneshot::channel();
        let (release, blocked) = std::sync::mpsc::channel();
        let workers = resources.workers();
        let job = StreamJobContext::new(
            74,
            "a".repeat(64),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let task = tokio::spawn(async move {
            workers
                .run_rolling_credited(RollingScope::stream(&job), identity, move |_| {
                    let _ = started.send(());
                    blocked.recv().unwrap();
                    Ok(())
                })
                .await
        });
        ready.await.unwrap();
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        assert!(resources.check(0, small, "features").is_err());
        release.send(()).unwrap();
        receipt.await.unwrap();
        resources.check(0, small, "features").unwrap();
    }
    #[tokio::test]
    async fn test_credited_worker_panic_preserves_rolling_node_and_refunds_identity() {
        let resources = Resources::default();
        let budget = StateBudget::new(8, 4096).unwrap();
        let identity = reserve(&resources, "actual-rolling-node", 10, budget).unwrap();
        let job = StreamJobContext::new(
            75,
            "a".repeat(64),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let error = resources
            .workers()
            .run_rolling_credited::<()>(RollingScope::stream(&job), identity, |_| {
                panic!("controlled worker panic")
            })
            .await
            .unwrap_err();
        assert!(
            matches!(error, crate::CalcFlowError::Operator { node_id, .. } if node_id == "actual-rolling-node")
        );
        resources
            .check(0, StateBudget::new(8, 1).unwrap(), "actual-rolling-node")
            .unwrap();
    }
}
