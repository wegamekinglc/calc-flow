//! Async runner adapter for prepared synchronous operators.

use std::sync::Arc;

use super::{OperatorTaskInputs, control, send_emission};
use crate::{
    EventTime, IngressProgressSnapshot, Result, StreamMessage,
    operator::{
        PreparedExpression, SyncExpressionOperator,
        rolling_metrics::RollingMetricsRecorder,
        sync_operator::{Emitter, PrepareContext, SyncContext, SyncNode, SyncStreamOperator},
    },
    pipeline::CompiledStreamOperator,
    runtime::streaming::progress::aggregate::ProgressEmissionKind,
};

pub(super) fn is_sync(operator: &CompiledStreamOperator) -> bool {
    matches!(operator, CompiledStreamOperator::SyncExpression(_))
}

pub(super) async fn prepare(inputs: &mut OperatorTaskInputs) -> Result<()> {
    let CompiledStreamOperator::Expression(_) = &inputs.operator else {
        return Ok(());
    };
    let owned = std::mem::replace(&mut inputs.operator, CompiledStreamOperator::CheckpointLoan);
    let CompiledStreamOperator::Expression(operator) = owned else {
        unreachable!("Expression variant checked above")
    };
    let prepared = operator
        .prepare_sync(&PrepareContext::new(inputs.context.job()))
        .await?;
    match prepared {
        PreparedExpression::Async(operator) => {
            inputs.operator = CompiledStreamOperator::Expression(operator);
        }
        PreparedExpression::Sync(operator) => {
            let node_id: Arc<str> = Arc::from(inputs.node_id.as_str());
            let output_budget = inputs.progress.output_budget(&inputs.outputs);
            let late_metrics: Arc<dyn crate::operator::LateMetricSink> =
                Arc::new(inputs.progress.clone());
            let context = SyncContext::new(
                inputs.context.job(),
                Arc::clone(&node_id),
                output_budget,
                late_metrics,
            );
            let emitter = Emitter::new(
                node_id,
                &inputs.output_ports,
                std::mem::take(&mut inputs.outputs),
                &inputs.late_output_ports,
            );
            inputs.operator = CompiledStreamOperator::SyncExpression(Box::new(SyncNode::new(
                operator, context, emitter,
            )));
        }
    }
    Ok(())
}

pub(super) async fn data(
    inputs: &mut OperatorTaskInputs,
    batch: crate::Batch,
    input_watermark: Option<EventTime>,
    ingress_progress: IngressProgressSnapshot,
    rolling: Option<RollingMetricsRecorder>,
) -> Result<()> {
    data_callback(
        inputs,
        batch,
        input_watermark,
        ingress_progress,
        rolling.clone(),
    )?;
    flush(inputs, rolling.as_ref()).await
}

fn data_callback(
    inputs: &mut OperatorTaskInputs,
    batch: crate::Batch,
    input_watermark: Option<EventTime>,
    ingress_progress: IngressProgressSnapshot,
    rolling: Option<RollingMetricsRecorder>,
) -> Result<()> {
    let result = {
        let node = node_mut(&mut inputs.operator);
        node.context_mut().update(input_watermark, ingress_progress);
        node.context_mut().set_rolling_metrics(rolling);
        let (operator, context, emitter) = node.parts_mut();
        operator.on_data(0, batch, context, emitter)
    };
    if result.is_err() {
        node_mut(&mut inputs.operator).emitter_mut().clear();
    }
    result
}

pub(super) async fn progress(
    inputs: &mut OperatorTaskInputs,
    emissions: Vec<ProgressEmissionKind>,
    mut previous_watermark: Option<EventTime>,
    ingress_progress: IngressProgressSnapshot,
    aggregate_input_frontier: Option<EventTime>,
    output_frontier: &mut Option<EventTime>,
) -> Result<()> {
    inputs
        .progress
        .observe_datafusion_runtime(inputs.operator.datafusion_runtime_initialized());
    let mut emit_idle = false;
    for emission in emissions {
        match emission {
            ProgressEmissionKind::Watermark(watermark) => {
                watermark_callback(
                    inputs,
                    watermark,
                    previous_watermark,
                    ingress_progress.clone(),
                )
                .await?;
                previous_watermark = Some(watermark);
            }
            ProgressEmissionKind::Idle => emit_idle = true,
            ProgressEmissionKind::EndOfInput => {}
        }
    }
    let has_non_late_output = node_mut(&mut inputs.operator)
        .emitter()
        .has_non_late_output();
    if let Some(candidate) = control::advancing_output_frontier(
        &inputs.operator,
        has_non_late_output,
        aggregate_input_frontier,
        &ingress_progress,
        *output_frontier,
    )? {
        forward_control(inputs, StreamMessage::watermark(candidate)).await?;
        *output_frontier = Some(candidate);
    }
    if emit_idle && !inputs.operator.suppresses_output_idle() {
        forward_control(inputs, StreamMessage::idle()).await?;
    }
    Ok(())
}

async fn watermark_callback(
    inputs: &mut OperatorTaskInputs,
    watermark: EventTime,
    input_watermark: Option<EventTime>,
    ingress_progress: IngressProgressSnapshot,
) -> Result<()> {
    let processing_timer = inputs.metrics.timer();
    let callback = inputs.progress.begin_rolling(
        crate::operator::rolling_metrics::RollingCallback::Watermark,
        inputs.context.job().cancellation(),
    );
    let rolling = callback
        .as_ref()
        .map(|callback| callback.recorder().clone());
    let result = {
        let node = node_mut(&mut inputs.operator);
        node.context_mut().update(input_watermark, ingress_progress);
        node.context_mut().set_rolling_metrics(rolling.clone());
        let (operator, context, emitter) = node.parts_mut();
        operator.on_watermark(watermark, context, emitter)
    };
    let result = match result {
        Ok(()) => flush(inputs, rolling.as_ref()).await,
        Err(error) => {
            node_mut(&mut inputs.operator).emitter_mut().clear();
            Err(error)
        }
    };
    if let Some(callback) = callback {
        callback.complete(&result);
    }
    result?;
    inputs.metrics.record_watermark(&processing_timer)
}

pub(super) async fn end(
    inputs: &mut OperatorTaskInputs,
    input_watermark: Option<EventTime>,
    ingress_progress: IngressProgressSnapshot,
    rolling: Option<RollingMetricsRecorder>,
) -> Result<()> {
    let result = {
        let node = node_mut(&mut inputs.operator);
        node.context_mut().update(input_watermark, ingress_progress);
        node.context_mut().set_rolling_metrics(rolling.clone());
        let (operator, context, emitter) = node.parts_mut();
        operator.on_end(context, emitter)
    };
    match result {
        Ok(()) => flush(inputs, rolling.as_ref()).await,
        Err(error) => {
            node_mut(&mut inputs.operator).emitter_mut().clear();
            Err(error)
        }
    }
}

async fn flush(
    inputs: &mut OperatorTaskInputs,
    observation: Option<&RollingMetricsRecorder>,
) -> Result<()> {
    let OperatorTaskInputs {
        operator,
        context,
        metrics,
        progress,
        ..
    } = inputs;
    loop {
        let Some(emission) = node_mut(operator).emitter_mut().pop() else {
            return Ok(());
        };
        let node = node_mut(operator);
        let output = node
            .emitter_mut()
            .output_mut(emission.port)
            .expect("validated emission port");
        send_emission(
            &mut output.senders,
            emission.message,
            context.job().cancellation(),
            context.job().job_id(),
            observation,
        )
        .await?;
        metrics.record_output(emission.cost)?;
        progress.record_output()?;
    }
}

pub(super) async fn forward_control(
    inputs: &mut OperatorTaskInputs,
    message: StreamMessage,
) -> Result<()> {
    let OperatorTaskInputs {
        operator, context, ..
    } = inputs;
    let cancellation = context.job().cancellation();
    for output in node_mut(operator).emitter_mut().outputs_mut() {
        if !control::forwards_control(output.late, &message) {
            continue;
        }
        for sender in &mut output.senders {
            tokio::select! {
                biased;
                () = cancellation.cancelled() => {
                    return Err(crate::CalcFlowError::Cancelled {
                        run_id: context.job().job_id().to_string(),
                    });
                }
                result = sender.send(message.clone()) => result?,
            }
        }
    }
    Ok(())
}

pub(super) fn data_session(
    inputs: &mut OperatorTaskInputs,
    message: &StreamMessage,
    input_watermark: Option<EventTime>,
    ingress_progress: IngressProgressSnapshot,
) -> Result<()> {
    let batch = message
        .as_data()
        .expect("session data message carries a batch")
        .clone();
    let cost = crate::runtime::streaming::EnvelopeCost::of_message(message)?;
    inputs.metrics.record_input(cost)?;
    inputs.progress.record_input()?;
    let timer = inputs.metrics.timer();
    let callback = inputs.progress.begin_rolling(
        crate::operator::rolling_metrics::RollingCallback::Data,
        inputs.context.job().cancellation(),
    );
    if let Some(callback) = &callback {
        callback.recorder().add(
            crate::operator::rolling_metrics::RollingWork::InputRows,
            batch.num_rows(),
        );
    }
    let rolling = callback
        .as_ref()
        .map(|callback| callback.recorder().clone());
    let result = data_callback(inputs, batch, input_watermark, ingress_progress, rolling)
        .and_then(|()| flush_session(inputs));
    if let Some(callback) = callback {
        callback.complete(&result);
    }
    inputs
        .progress
        .observe_datafusion_runtime(inputs.operator.datafusion_runtime_initialized());
    result?;
    inputs
        .metrics
        .record_data_completion(&timer, &inputs.progress.snapshot())
}

pub(super) fn progress_session(
    inputs: &mut OperatorTaskInputs,
    emissions: Vec<ProgressEmissionKind>,
    mut previous_watermark: Option<EventTime>,
    ingress_progress: &IngressProgressSnapshot,
    aggregate_input_frontier: Option<EventTime>,
    output_frontier: &mut Option<EventTime>,
) -> Result<()> {
    inputs
        .progress
        .observe_datafusion_runtime(inputs.operator.datafusion_runtime_initialized());
    let mut emit_idle = false;
    for emission in emissions {
        match emission {
            ProgressEmissionKind::Watermark(watermark) => {
                watermark_session(
                    inputs,
                    watermark,
                    previous_watermark,
                    ingress_progress.clone(),
                )?;
                previous_watermark = Some(watermark);
            }
            ProgressEmissionKind::Idle => emit_idle = true,
            ProgressEmissionKind::EndOfInput => {}
        }
    }
    let has_non_late_output = node_mut(&mut inputs.operator)
        .emitter()
        .has_non_late_output();
    if let Some(candidate) = control::advancing_output_frontier(
        &inputs.operator,
        has_non_late_output,
        aggregate_input_frontier,
        ingress_progress,
        *output_frontier,
    )? {
        forward_control_session(inputs, &StreamMessage::watermark(candidate))?;
        *output_frontier = Some(candidate);
    }
    if emit_idle && !inputs.operator.suppresses_output_idle() {
        forward_control_session(inputs, &StreamMessage::idle())?;
    }
    Ok(())
}

fn watermark_session(
    inputs: &mut OperatorTaskInputs,
    watermark: EventTime,
    input_watermark: Option<EventTime>,
    ingress_progress: IngressProgressSnapshot,
) -> Result<()> {
    let timer = inputs.metrics.timer();
    let callback = inputs.progress.begin_rolling(
        crate::operator::rolling_metrics::RollingCallback::Watermark,
        inputs.context.job().cancellation(),
    );
    let rolling = callback
        .as_ref()
        .map(|callback| callback.recorder().clone());
    let result = {
        let node = node_mut(&mut inputs.operator);
        node.context_mut().update(input_watermark, ingress_progress);
        node.context_mut().set_rolling_metrics(rolling);
        let (operator, context, emitter) = node.parts_mut();
        operator.on_watermark(watermark, context, emitter)
    }
    .and_then(|()| flush_session(inputs));
    if let Some(callback) = callback {
        callback.complete(&result);
    }
    result?;
    inputs.metrics.record_watermark(&timer)
}

pub(super) fn end_session(
    inputs: &mut OperatorTaskInputs,
    input_watermark: Option<EventTime>,
    ingress_progress: IngressProgressSnapshot,
) -> Result<()> {
    inputs.progress.record_on_end()?;
    let callback = inputs.progress.begin_rolling(
        crate::operator::rolling_metrics::RollingCallback::End,
        inputs.context.job().cancellation(),
    );
    let rolling = callback
        .as_ref()
        .map(|callback| callback.recorder().clone());
    let result = {
        let node = node_mut(&mut inputs.operator);
        node.context_mut().update(input_watermark, ingress_progress);
        node.context_mut().set_rolling_metrics(rolling);
        let (operator, context, emitter) = node.parts_mut();
        operator.on_end(context, emitter)
    }
    .and_then(|()| flush_session(inputs));
    if let Some(callback) = callback {
        callback.complete(&result);
    }
    result?;
    forward_control_session(inputs, &StreamMessage::end_of_input())?;
    inputs.progress.mark_ended();
    Ok(())
}

fn flush_session(inputs: &mut OperatorTaskInputs) -> Result<()> {
    let OperatorTaskInputs {
        operator,
        metrics,
        progress,
        ..
    } = inputs;
    loop {
        let Some(emission) = node_mut(operator).emitter_mut().pop() else {
            return Ok(());
        };
        let output = node_mut(operator)
            .emitter_mut()
            .output_mut(emission.port)
            .expect("validated session emission port");
        let Some((last, rest)) = output.senders.split_last_mut() else {
            return Err(crate::CalcFlowError::Internal {
                message: "validated session emission lost its routes".into(),
            });
        };
        for sender in rest {
            sender.send_session(emission.message.clone(), &emission.cost)?;
        }
        last.send_session(emission.message, &emission.cost)?;
        metrics.record_output(emission.cost)?;
        progress.record_output()?;
    }
}

fn forward_control_session(inputs: &mut OperatorTaskInputs, message: &StreamMessage) -> Result<()> {
    let cost = crate::runtime::streaming::EnvelopeCost::of_message(message)?;
    for output in node_mut(&mut inputs.operator).emitter_mut().outputs_mut() {
        if !control::forwards_control(output.late, message) {
            continue;
        }
        for sender in &mut output.senders {
            sender.send_session(message.clone(), &cost)?;
        }
    }
    Ok(())
}

pub(super) fn recycle(operator: &mut CompiledStreamOperator, batch: crate::Batch) {
    if let CompiledStreamOperator::SyncExpression(node) = operator {
        node.recycle(batch);
    }
}

fn node_mut(operator: &mut CompiledStreamOperator) -> &mut SyncNode<SyncExpressionOperator> {
    let CompiledStreamOperator::SyncExpression(node) = operator else {
        unreachable!("sync adapter called for an async operator")
    };
    node
}
