use super::*;
use crate::{CalcFlowError, runtime::streaming::supervisor::SupervisionReport};

#[derive(Clone, Copy)]
enum Failure {
    ProducerMetrics,
    ProducerProgress,
    DownstreamMetrics,
}

struct Observation {
    report: SupervisionReport,
    status: crate::StreamAsofJoinStatus,
    accepted: u64,
    output: Vec<Batch>,
    released: bool,
}

async fn observe(failure: Failure) -> Observation {
    let parts = plan(true, false)
        .into_runtime_parts(EdgeBudget::new(64, 1 << 20).unwrap())
        .unwrap();
    let edge = parts.nodes[0].output_edges["output"][0].clone();
    let routes: Vec<_> = parts
        .source_routes
        .iter()
        .map(|(id, route)| (id.clone(), route.target.port.clone()))
        .collect();
    let output = parts.sink_routes.keys().next().unwrap().clone();
    let cancellation = CancellationToken::new();
    let context = StreamJobContext::new(
        14,
        &parts.fingerprint,
        JsonMap::new(),
        None,
        cancellation.clone(),
    );
    let metrics = MetricsRecorder::new(
        parts
            .edges
            .iter()
            .map(|(id, edge)| (id.clone(), edge.budget)),
        parts.source_routes.keys().cloned(),
        parts.nodes.iter().map(|node| node.node_id.clone()),
        parts.sink_routes.keys().cloned(),
    );
    let (commands, _) = mpsc::unbounded_channel();
    let core = Arc::new(JobCore::new(
        LaunchId::new(0),
        14,
        commands,
        metrics.clone(),
        StatusProjection::default(),
        false,
        parts.name.clone(),
    ));
    let Ok(mut runtime) =
        run_operator_entry(parts, &context, &core, &cancellation, BTreeMap::new(), None).await
    else {
        panic!("ASOF entry failed")
    };
    match failure {
        Failure::ProducerMetrics => metrics.preset_operator_outputs_for_test("a_asof", u64::MAX),
        Failure::ProducerProgress => {
            core.runtime_status.lock().nodes["a_asof"].preset_output_count_for_test(u64::MAX);
        }
        Failure::DownstreamMetrics => {
            metrics.preset_operator_outputs_for_test("c_select1", u64::MAX);
        }
    }
    runtime.data_gate.send(true).unwrap();
    let traffic = tokio::time::timeout(Duration::from_secs(5), async {
        for (binding, port) in routes {
            let sender = &mut runtime.source_outputs.get_mut(&binding).unwrap()[0];
            sender
                .send(StreamMessage::data(input_batch(port == "left")))
                .await?;
            sender.send(StreamMessage::end_of_input()).await?;
        }
        let mut emitted = Vec::new();
        let receiver = runtime.sink_inputs.get_mut(&output).unwrap();
        while let Some(message) = receiver.recv().await? {
            if let Some(batch) = message.as_data() {
                emitted.push(batch.clone());
            }
            if message.is_end_of_input() {
                break;
            }
        }
        Ok::<_, CalcFlowError>(emitted)
    })
    .await;
    cancellation.cancel();
    let report = runtime.supervisor.join_all().await;
    let status = core.runtime_status.lock().nodes["a_asof"]
        .snapshot()
        .stream_asof_join
        .clone()
        .unwrap();
    let snapshot = metrics.snapshot();
    let released = runtime.supervisor.registry().snapshot().is_empty()
        && snapshot.edges.values().all(|edge| {
            edge.channel.queue_depth == 0
                && edge.channel.charged_rows == 0
                && edge.channel.charged_bytes == 0
                && !edge.drop_invariant_violated
        });
    Observation {
        report,
        status,
        accepted: snapshot.edges[&edge].input_batches,
        output: traffic
            .expect("failure traffic timed out")
            .expect("failure traffic errored"),
        released,
    }
}

#[tokio::test(flavor = "current_thread")]
async fn a13_producer_post_accept_metrics_failure_does_not_commit_asof_prefix() {
    let observed = observe(Failure::ProducerMetrics).await;
    assert!(observed.released);
    assert_eq!(observed.accepted, 1);
    assert_eq!(
        (
            observed.status.emitted_left_rows,
            observed.status.pending_left_rows
        ),
        (0, 4)
    );
    assert_eq!(
        (observed.status.matched_rows, observed.status.unmatched_rows),
        (0, 0)
    );
    assert_eq!(observed.report.primary_errors().len(), 1);
    assert_eq!(observed.report.primary_errors()[0].task_id, TaskId::new(0));
    assert!(
        matches!(&observed.report.primary_errors()[0].error, CalcFlowError::InvalidArgument { field, .. } if field == "runtime.metrics.a_asof.fully_fanned_out_batches")
    );
    assert!(observed.output.len() <= 1);
}

#[tokio::test(flavor = "current_thread")]
async fn a13_producer_post_accept_progress_failure_does_not_commit_asof_prefix() {
    let observed = observe(Failure::ProducerProgress).await;
    assert!(observed.released);
    assert_eq!(observed.accepted, 1);
    assert_eq!(
        (
            observed.status.emitted_left_rows,
            observed.status.pending_left_rows
        ),
        (0, 4)
    );
    assert_eq!(observed.report.primary_errors().len(), 1);
    assert_eq!(observed.report.primary_errors()[0].task_id, TaskId::new(0));
    assert!(
        matches!(&observed.report.primary_errors()[0].error, CalcFlowError::Internal { message } if message == "operator output batch counter overflowed")
    );
    assert!(observed.output.len() <= 1);
}

#[tokio::test(flavor = "current_thread")]
async fn a13_downstream_failure_after_successful_emit_keeps_asof_commit() {
    let observed = observe(Failure::DownstreamMetrics).await;
    assert!(observed.released);
    assert_eq!(observed.accepted, 1);
    assert_eq!(
        (
            observed.status.emitted_left_rows,
            observed.status.pending_left_rows
        ),
        (4, 0)
    );
    assert_eq!(
        (observed.status.matched_rows, observed.status.unmatched_rows),
        (3, 1)
    );
    assert_eq!(observed.report.primary_errors().len(), 1);
    assert_eq!(observed.report.primary_errors()[0].task_id, TaskId::new(1));
    assert!(
        matches!(&observed.report.primary_errors()[0].error, CalcFlowError::InvalidArgument { field, .. } if field == "runtime.metrics.c_select1.fully_fanned_out_batches")
    );
}
