use std::collections::{BTreeMap, BTreeSet};

use super::{OpenedCheckpointRuntime, OperatorProgress};
use crate::{
    CalcFlowError, CancellationToken, Port, Result,
    pipeline::{
        CompiledStreamOperator, OperatorCheckpointCapability, RuntimeStreamNode, StableOperatorId,
        StreamRuntimePlanParts,
    },
    runtime::streaming::{
        context::StreamJobContext,
        operator_task::validate_sql_restore_progress,
        sql_recovery_work::{
            CheckpointWorkOperator, JobSqlRecoveryOwner, SqlRecoveryContext, SqlRestoreRequest,
            SqlWorkIdentity,
        },
    },
};

#[derive(Default)]
pub(super) struct TerminalSqlFrames {
    remaining: Option<std::vec::IntoIter<RuntimeStreamNode>>,
    completed: Vec<RuntimeStreamNode>,
    current: Option<RuntimeStreamNode>,
    frame: Option<NodeFrame>,
    operator: Option<CompiledStreamOperator>,
}

struct NodeFrame {
    node_id: String,
    operator_id: StableOperatorId,
    checkpoint_capability: OperatorCheckpointCapability,
    input_ports: BTreeMap<String, Port>,
    output_ports: BTreeMap<String, Port>,
    ingress_edges: BTreeMap<String, String>,
    output_edges: BTreeMap<String, Vec<String>>,
    late_output_ports: BTreeSet<String>,
}

impl NodeFrame {
    fn split(node: RuntimeStreamNode) -> (Self, CheckpointWorkOperator) {
        let RuntimeStreamNode {
            node_id,
            operator_id,
            operator,
            checkpoint_capability,
            input_ports,
            output_ports,
            ingress_edges,
            output_edges,
            late_output_ports,
        } = node;
        let operator = CheckpointWorkOperator::from_compiled(operator);
        (
            Self {
                node_id,
                operator_id,
                checkpoint_capability,
                input_ports,
                output_ports,
                ingress_edges,
                output_edges,
                late_output_ports,
            },
            operator,
        )
    }

    fn assemble(self, operator: CompiledStreamOperator) -> RuntimeStreamNode {
        RuntimeStreamNode {
            node_id: self.node_id,
            operator_id: self.operator_id,
            operator,
            checkpoint_capability: self.checkpoint_capability,
            input_ports: self.input_ports,
            output_ports: self.output_ports,
            ingress_edges: self.ingress_edges,
            output_edges: self.output_edges,
            late_output_ports: self.late_output_ports,
        }
    }
}

pub(super) async fn restore_terminal(
    plan: &mut StreamRuntimePlanParts,
    frames: &mut TerminalSqlFrames,
    checkpoint: &OpenedCheckpointRuntime,
    context: &StreamJobContext,
    owner: &JobSqlRecoveryOwner,
    launch_cancel: &CancellationToken,
) -> Result<BTreeMap<String, OperatorProgress>> {
    let selected = checkpoint
        .selected
        .as_ref()
        .expect("selected terminal manifest");
    frames.remaining = Some(std::mem::take(&mut plan.nodes).into_iter());
    let mut statuses = BTreeMap::new();
    let mut order = 0;
    while let Some(node) = frames.remaining.as_mut().unwrap().next() {
        frames.current = Some(node);
        let node = frames.current.as_ref().unwrap();
        if !CheckpointWorkOperator::handles(&node.operator) {
            frames.completed.push(frames.current.take().unwrap());
            order += 1;
            continue;
        }
        let id = node.operator_id.as_str();
        let entry = &selected.manifest.operators()[id];
        let snapshot = checkpoint
            .transaction
            .load_operator_state_cancellable(id, entry, context.cancellation())
            .await?;
        let snapshot = terminal_snapshot(node, entry, snapshot)?;
        frames
            .restore_candidate(snapshot, order, context, owner, launch_cancel)
            .await?;
        let frame = frames.frame.take().unwrap();
        let progress = OperatorProgress::default();
        progress.mark_ended();
        statuses.insert(frame.node_id.clone(), progress);
        frames
            .completed
            .push(frame.assemble(frames.operator.take().unwrap()));
        order += 1;
    }
    plan.nodes = std::mem::take(&mut frames.completed);
    frames.remaining = None;
    Ok(statuses)
}

impl TerminalSqlFrames {
    async fn restore_candidate(
        &mut self,
        snapshot: crate::OperatorStateSnapshot,
        order: usize,
        context: &StreamJobContext,
        owner: &JobSqlRecoveryOwner,
        launch_cancel: &CancellationToken,
    ) -> Result<()> {
        let (frame, operator) = NodeFrame::split(self.current.take().unwrap());
        let identity = SqlWorkIdentity {
            node_id: frame.node_id.clone(),
            node_order: order,
            task_id: None,
        };
        self.frame = Some(frame);
        let submitted = owner.submit(SqlRestoreRequest {
            operator,
            snapshot,
            identity,
            context: SqlRecoveryContext::from(context),
            launch_cancel: launch_cancel.clone(),
        });
        let ticket = match submitted.result {
            Ok(ticket) => ticket,
            Err(error) => {
                self.operator = submitted
                    .operator
                    .map(CheckpointWorkOperator::into_compiled);
                return Err(error);
            }
        };
        ticket.join().await?.install_restore(&mut self.operator)
    }
}

fn terminal_snapshot(
    node: &RuntimeStreamNode,
    entry: &crate::OperatorManifestEntry,
    snapshot: crate::OperatorStateSnapshot,
) -> Result<crate::OperatorStateSnapshot> {
    let id = node.operator_id.as_str();
    let mut snapshot = node
        .checkpoint_capability
        .decode_snapshot(node.operator_id.as_str(), snapshot)?;
    if snapshot
        .inline_metadata
        .remove(crate::pipeline::OUTPUT_FRONTIER_METADATA_KEY_V1)
        .is_some()
    {
        return Err(CalcFlowError::CheckpointMismatch {
            message: format!(
                "SQL checkpoint operator {id:?} unexpectedly contains an output frontier"
            ),
        });
    }
    validate_sql_restore_progress(node.ingress_edges.keys(), &entry.progress)?;
    Ok(snapshot)
}
