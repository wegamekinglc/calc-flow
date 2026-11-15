use std::{future::Future, panic::AssertUnwindSafe, sync::Arc};

use async_trait::async_trait;
use futures::FutureExt;

use super::{
    OperatorBarrierAlignment, OperatorInputProgress, OperatorTaskInputs, dispatch_or_cancel,
    finish_operator, indicator_error, ingress_admission, normalize_cancelled_result,
    receive_operator_message, reset_and_acknowledge,
};
use crate::{
    CalcFlowError, Result,
    pipeline::CompiledStreamOperator,
    runtime::streaming::{
        context::wait_for_task_gate,
        edge_queue::Dequeue,
        local_edge::{LocalEdgeOwner, OperatorEdgeReceiver, RegionDrain, RegionEdgeControl},
        supervisor::{
            PreparedLogicalTask, PreparedRegionTask, RetainedTaskResult, TaskFailureSignal, TaskId,
            contain_task_panic,
        },
    },
};

use super::super::runner::operator_fusion::FusionProof;

struct RegionFailure {
    member: usize,
    error: CalcFlowError,
}

impl RegionFailure {
    fn first(error: CalcFlowError) -> Self {
        Self { member: 0, error }
    }

    fn second(error: CalcFlowError) -> Self {
        Self { member: 1, error }
    }
}

pub(crate) struct RegionResources {
    _first: OperatorTaskInputs,
    _second: Option<OperatorTaskInputs>,
    _control: RegionEdgeControl,
    _local_edges: Vec<LocalEdgeOwner>,
}

type RegionOutput = [RetainedTaskResult<Arc<RegionResources>>; 2];

pub(crate) fn prepare_compiled_region(
    members: Vec<(OperatorTaskInputs, PreparedLogicalTask)>,
    proof: &FusionProof,
    local_edges: Vec<LocalEdgeOwner>,
) -> Result<PreparedRegionTask<impl Future<Output = RegionOutput> + use<>>> {
    let [(first, first_token), (second, second_token)] =
        members.try_into().map_err(|_| CalcFlowError::Internal {
            message: "a compiled region requires exactly two logical operators".into(),
        })?;
    if !proof.is_region()
        || !proof.matches(&[&first.node_id, &second.node_id])
        || !matches!(first.operator, CompiledStreamOperator::Rolling(_))
        || !matches!(second.operator, CompiledStreamOperator::Expression(_))
        || first.checkpoint.is_some()
        || second.checkpoint.is_some()
        || first.restore.is_some()
        || second.restore.is_some()
    {
        return Err(CalcFlowError::Internal {
            message: "compiled region admission does not match its prepared operators".into(),
        });
    }
    let control = first
        .outputs
        .get("output")
        .and_then(|outputs| match outputs.as_slice() {
            [sender] => sender.region_control(),
            _ => None,
        })
        .ok_or_else(|| CalcFlowError::Internal {
            message: "compiled region upstream has no single local output".into(),
        })?;
    let downstream = second
        .ingresses
        .get("input")
        .and_then(|ingress| ingress.receiver.as_ref())
        .ok_or_else(|| CalcFlowError::Internal {
            message: "compiled region downstream has no input receiver".into(),
        })?;
    if !matches!(downstream, OperatorEdgeReceiver::Region(_)) || downstream.edge() != control.edge()
    {
        return Err(CalcFlowError::Internal {
            message: "compiled region endpoints do not share one local edge".into(),
        });
    }
    Ok(PreparedLogicalTask::bind_region(
        first_token,
        second_token,
        move |signals| run_region(first, second, control, local_edges, signals),
    ))
}

async fn run_region(
    mut first: OperatorTaskInputs,
    second: OperatorTaskInputs,
    control: RegionEdgeControl,
    local_edges: Vec<LocalEdgeOwner>,
    signals: [TaskFailureSignal; 2],
) -> RegionOutput {
    let mut second = Some(second);
    let mut active = 0;
    let result = AssertUnwindSafe(run_region_body(
        &mut first,
        &mut second,
        &control,
        &signals,
        &mut active,
    ))
    .catch_unwind()
    .await;
    let failure = match result {
        Ok(Ok(())) => None,
        Ok(Err(failure)) => Some(failure),
        Err(payload) => Some(RegionFailure {
            member: active,
            error: CalcFlowError::TaskPanicked {
                task_id: signals[active].task_id().as_u64(),
                message: super::panic_message(payload.as_ref()),
            },
        }),
    };
    let failure = failure.filter(|failure| {
        !(first.context.job().cancellation().is_cancelled()
            && matches!(failure.error, CalcFlowError::Cancelled { .. }))
    });
    let diagnostic = failure.as_ref().and_then(|failure| {
        (failure.member == 0)
            .then(|| indicator_error(&first.node_id, &first.operator, &failure.error))
            .flatten()
    });
    let resources = Arc::new(RegionResources {
        _first: first,
        _second: second,
        _control: control,
        _local_edges: local_edges,
    });
    let (first_result, second_result) = match failure {
        Some(RegionFailure { member: 0, error }) => (Err(error), Ok(())),
        Some(RegionFailure { member: 1, error }) => (Ok(()), Err(error)),
        Some(_) => unreachable!("compiled region has two members"),
        None => (Ok(()), Ok(())),
    };
    [
        RetainedTaskResult {
            indicator_diagnostic: diagnostic,
            result: first_result,
            retained: Arc::clone(&resources),
        },
        RetainedTaskResult {
            indicator_diagnostic: None,
            result: second_result,
            retained: resources,
        },
    ]
}

async fn run_region_body(
    first: &mut OperatorTaskInputs,
    second: &mut Option<OperatorTaskInputs>,
    control: &RegionEdgeControl,
    signals: &[TaskFailureSignal; 2],
    active: &mut usize,
) -> std::result::Result<(), RegionFailure> {
    let Some((mut first_progress, second_progress)) =
        enter_region(first, second, signals, active).await?
    else {
        return Ok(());
    };
    let downstream = RegionDownstream {
        inputs: second.take().expect("downstream inputs are owned once"),
        progress: second_progress,
        alignment: OperatorBarrierAlignment::default(),
        task_id: signals[1].task_id(),
    };
    control
        .install(Box::new(downstream))
        .map_err(RegionFailure::second)?;
    let mut first_alignment = OperatorBarrierAlignment::default();
    loop {
        if first
            .ingresses
            .values()
            .all(|ingress| ingress.saw_explicit_eof)
        {
            *active = 0;
            let result = finish_operator(first, &first_progress, None).await;
            if let Some(error) = control.take_failure() {
                return Err(RegionFailure::second(error));
            }
            normalize_cancelled_result(first, result).map_err(RegionFailure::first)?;
            control.drain_all().await.map_err(RegionFailure::second)?;
            return Ok(());
        }
        *active = 0;
        let received = receive_operator_message(first, &first_alignment, None)
            .await
            .map_err(RegionFailure::first)?;
        let Some((ingress_name, message)) = received else {
            return Ok(());
        };
        let result = dispatch_or_cancel(
            first,
            &ingress_name,
            message,
            &mut first_progress,
            &mut first_alignment,
        )
        .await;
        if let Some(error) = control.take_failure() {
            return Err(RegionFailure::second(error));
        }
        normalize_cancelled_result(first, result).map_err(RegionFailure::first)?;
        ingress_admission::acknowledge(first, &ingress_name, &first_progress);
        control.drain_all().await.map_err(RegionFailure::second)?;
    }
}

async fn enter_region(
    first: &mut OperatorTaskInputs,
    second: &mut Option<OperatorTaskInputs>,
    signals: &[TaskFailureSignal; 2],
    active: &mut usize,
) -> std::result::Result<Option<(OperatorInputProgress, OperatorInputProgress)>, RegionFailure> {
    let second_inputs = second.as_mut().expect("region owns downstream inputs");
    *active = 0;
    if !wait_for_task_gate(
        &mut first.entry_gate,
        &first.launch_cancel,
        first.context.job().cancellation(),
        "operator launch gate closed before release",
    )
    .await
    .map_err(RegionFailure::first)?
    {
        return Ok(None);
    }
    *active = 1;
    if !wait_for_task_gate(
        &mut second_inputs.entry_gate,
        &second_inputs.launch_cancel,
        second_inputs.context.job().cancellation(),
        "operator launch gate closed before release",
    )
    .await
    .map_err(RegionFailure::second)?
    {
        return Ok(None);
    }
    *active = 0;
    let first_progress = reset_and_acknowledge(first, signals[0].task_id()).await;
    *active = 1;
    let second_progress = reset_and_acknowledge(second_inputs, signals[1].task_id()).await;
    let first_progress = first_progress.map_err(RegionFailure::first)?;
    let second_progress = second_progress.map_err(RegionFailure::second)?;
    let (Some(first_progress), Some(second_progress)) = (first_progress, second_progress) else {
        return Ok(None);
    };
    *active = 0;
    if !wait_for_task_gate(
        &mut first.data_gate,
        &first.launch_cancel,
        first.context.job().cancellation(),
        "operator launch gate closed before release",
    )
    .await
    .map_err(RegionFailure::first)?
    {
        return Ok(None);
    }
    *active = 1;
    if !wait_for_task_gate(
        &mut second_inputs.data_gate,
        &second_inputs.launch_cancel,
        second_inputs.context.job().cancellation(),
        "operator launch gate closed before release",
    )
    .await
    .map_err(RegionFailure::second)?
    {
        return Ok(None);
    }
    activate(first, signals[0].task_id());
    activate(second_inputs, signals[1].task_id());
    Ok(Some((first_progress, second_progress)))
}

fn activate(inputs: &mut OperatorTaskInputs, task_id: TaskId) {
    inputs.context.bind_task_id(task_id);
    if let Some(client) = &mut inputs.entity_work {
        client.activate(task_id);
    }
    if let Some(client) = &mut inputs.sql_recovery {
        client.activate(task_id);
    }
}

struct RegionDownstream {
    inputs: OperatorTaskInputs,
    progress: OperatorInputProgress,
    alignment: OperatorBarrierAlignment,
    task_id: TaskId,
}

impl RegionDownstream {
    async fn dispatch_one(&mut self) -> Result<bool> {
        let (ingress_name, message) = {
            let (name, ingress) = self
                .inputs
                .ingresses
                .iter_mut()
                .next()
                .expect("compiled region downstream has one input");
            let receiver = ingress
                .receiver
                .as_mut()
                .expect("region receiver is present");
            match receiver.try_recv_region()? {
                Dequeue::Message(message) => (name.clone(), message),
                Dequeue::Empty => return Ok(false),
                Dequeue::Closed if self.inputs.context.job().cancellation().is_cancelled() => {
                    return Ok(false);
                }
                Dequeue::Closed => {
                    return Err(CalcFlowError::EdgeClosed {
                        edge: ingress.edge_id.clone(),
                    });
                }
            }
        };
        dispatch_or_cancel(
            &mut self.inputs,
            &ingress_name,
            message,
            &mut self.progress,
            &mut self.alignment,
        )
        .await?;
        ingress_admission::acknowledge(&mut self.inputs, &ingress_name, &self.progress);
        if self
            .inputs
            .ingresses
            .values()
            .all(|ingress| ingress.saw_explicit_eof)
        {
            finish_operator(&mut self.inputs, &self.progress, None).await?;
        }
        Ok(true)
    }
}

#[async_trait]
impl RegionDrain for RegionDownstream {
    async fn drain_one(&mut self) -> Result<bool> {
        let result = Box::pin(contain_task_panic(self.task_id, self.dispatch_one()))
            .await
            .and_then(std::convert::identity);
        match result {
            Err(CalcFlowError::Cancelled { .. })
                if self.inputs.context.job().cancellation().is_cancelled() =>
            {
                Ok(false)
            }
            result => result,
        }
    }
}
