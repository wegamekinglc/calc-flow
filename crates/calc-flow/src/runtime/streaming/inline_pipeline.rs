use std::sync::Arc;

use super::{
    operator_task::caller_driven::{CallerDrivenOperator, drain_in_order_sync},
    supervisor::PreparedLogicalTask,
};
use crate::Result;

pub(in crate::runtime::streaming) type SharedInlinePipeline =
    Arc<tokio::sync::Mutex<InlinePipeline>>;

pub(in crate::runtime::streaming) struct InlinePipeline {
    nodes: Vec<CallerDrivenOperator>,
    reservations: Vec<PreparedLogicalTask>,
    all_sync: bool,
}

impl InlinePipeline {
    pub(in crate::runtime::streaming) fn new_shared(
        nodes: Vec<CallerDrivenOperator>,
        reservations: Vec<PreparedLogicalTask>,
        allow_async: bool,
    ) -> SharedInlinePipeline {
        let all_sync = !allow_async && nodes.iter().all(CallerDrivenOperator::is_sync);
        Arc::new(tokio::sync::Mutex::new(Self {
            nodes,
            reservations,
            all_sync,
        }))
    }

    pub(in crate::runtime::streaming) async fn drain(&mut self) -> Result<()> {
        if self.nodes.iter().any(|node| !node.data_gate_open()) {
            return Ok(());
        }
        if self.all_sync {
            return drain_in_order_sync(&mut self.nodes);
        }
        for (index, node) in self.nodes.iter_mut().enumerate() {
            if let Err(error) = node.drain().await {
                if matches!(error, crate::CalcFlowError::Cancelled { .. }) {
                    return Err(error);
                }
                let source_error = crate::CalcFlowError::Operator {
                    node_id: node.node_id().to_owned(),
                    message: error.to_string(),
                };
                self.reservations.remove(index).fail(error).await;
                return Err(source_error);
            }
        }
        Ok(())
    }
}
