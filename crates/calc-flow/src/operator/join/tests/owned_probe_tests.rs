use super::*;
use crate::runtime::streaming::gather_work::{
    TestService,
    admission_probe::{AdmissionProbe, AdmissionStage},
};

const PROBE_ROWS: usize = 8_192;

fn probe_operator() -> StreamJoinOperator {
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(20_000, 10_000_000, 8_192).unwrap();
    let operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
    data_work::relax_probe_cost_gate_for_test();
    operator
}

fn assert_literal_output(collector: &mut EdgeCollector, rows: usize, sequence: u64) {
    let messages = collector.drain("output");
    assert_eq!(messages.len(), 1);
    let batch = messages[0].as_data().unwrap();
    assert_eq!(batch.metadata().sequence(), sequence);
    let record = &batch.table_payload().unwrap().batches()[0];
    assert_eq!(record.num_rows(), rows);
    assert_eq!(
        record.column(0).to_data(),
        Int64Array::from(vec![7; rows]).to_data()
    );
    assert_eq!(
        record.column(2).to_data(),
        Int64Array::from(vec![42; rows]).to_data()
    );
    assert_eq!(
        record.column(5).to_data(),
        StringArray::from(vec!["paid"; rows]).to_data()
    );
}

pub(super) async fn assert_owned_process_data(service: &TestService) {
    for mixed in [false, true] {
        assert_abandoned_probe(service, data_work::ProbePhase::Count, false, mixed).await;
        assert_abandoned_probe(service, data_work::ProbePhase::Fill, true, mixed).await;
    }
    let mut operator = probe_operator();
    let job = job().with_gather_owner(service.owner("owned-probe".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right_batch(vec![0]), &context, &mut collector)
        .await
        .unwrap();
    reset_join_work();
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observations = Arc::clone(&observed);
    let paid_owner = job.gather_owner().clone();
    operator.probe_test_hook = Some(Arc::new(move |phase, rows, index| {
        assert!(paid_owner.funding().2 > 0);
        observations
            .lock()
            .unwrap()
            .push((phase, rows, index, std::thread::current().id()));
    }));
    let rows_pointer = operator.state.right.as_ptr() as usize;
    let index_pointer = Arc::as_ptr(operator.state.right.1.as_ref().unwrap()) as usize;
    operator
        .process_data(
            "left",
            left_batch(vec![0; PROBE_ROWS]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    assert_literal_output(&mut collector, PROBE_ROWS, 0);
    assert_eq!(
        join_work().native_range_visits,
        0,
        "large native fill must leave the actor thread"
    );
    assert_eq!(
        join_work().native_boundary_visits,
        0,
        "large native count must leave the actor thread"
    );
    assert!(
        job.gather_owner().funding().1 > 0,
        "large process_data must use the actual paid native worker"
    );
    assert_eq!(operator.state.next_left_row_id, 8_192);
    assert_eq!(operator.state.deltas.pending.iter().count(), 8_193);
    {
        let observed = observed.lock().unwrap();
        assert_eq!(observed.len(), 2);
        assert_eq!(observed[0].0, data_work::ProbePhase::Count);
        assert_eq!(observed[1].0, data_work::ProbePhase::Fill);
        for &(_, rows, index, thread) in &*observed {
            assert_eq!(rows, rows_pointer);
            assert_eq!(index, index_pointer);
            assert_ne!(thread, std::thread::current().id());
        }
    }
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop(operator);
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

struct ProbeGate {
    started: tokio::sync::oneshot::Receiver<(usize, usize, std::thread::ThreadId)>,
    release: std::sync::mpsc::Sender<()>,
}

fn probe_gate(operator: &mut StreamJoinOperator, phase: data_work::ProbePhase) -> ProbeGate {
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let gate = std::sync::Mutex::new(Some((entered, wait)));
    operator.probe_test_hook = Some(Arc::new(move |actual, rows, index| {
        if actual == phase {
            let (entered, wait) = gate.lock().unwrap().take().unwrap();
            entered
                .send((rows, index, std::thread::current().id()))
                .unwrap();
            wait.recv_timeout(Duration::from_secs(10)).unwrap();
        }
    }));
    ProbeGate { started, release }
}

fn assert_probe_released(
    operator: &StreamJoinOperator,
    job: &StreamJobContext,
    pool: &dyn datafusion::execution::memory_pool::MemoryPool,
) {
    assert!(operator.compaction_release.is_none());
    assert!(operator.compaction_cleanup.is_none());
    assert!(operator.probe_control.is_none());
    native_lookup_tests::assert_resident_and_gather_funding(
        pool,
        job,
        native_lookup_tests::state_funding(operator),
    );
}

fn mixed_probe_input() -> Batch {
    let records = [
        (1, -1, false),
        (PROBE_ROWS / 2, 1, false),
        (PROBE_ROWS / 2, 1, true),
    ]
    .into_iter()
    .map(|(rows, time, nullable)| {
        RecordBatch::try_new(
            left_schema(),
            vec![
                Arc::new(Int64Array::from(vec![7; rows])) as ArrayRef,
                Arc::new(TimestampMicrosecondArray::from(vec![time; rows]).with_timezone("UTC")),
                Arc::new(Int64Array::from(vec![(!nullable).then_some(42); rows])),
            ],
        )
        .unwrap()
    })
    .collect();
    Batch::table(records, BatchMetadata::default()).unwrap()
}

async fn assert_abandoned_probe(
    service: &TestService,
    phase: data_work::ProbePhase,
    cancel: bool,
    mixed: bool,
) {
    let mut operator = probe_operator();
    let job_context = job().with_gather_owner(service.owner("abandoned-probe".into()));
    let context = StreamOperatorContext::with_ingress_progress(
        &job_context,
        "match",
        None,
        IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                IngressProgress::new(
                    IngressState::Active,
                    mixed.then_some(EventTime::from_micros(0)),
                ),
            ),
            (
                "right".into(),
                IngressProgress::new(IngressState::Active, None),
            ),
        ])),
    );
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right_batch(vec![0]), &context, &mut collector)
        .await
        .unwrap();
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let weak = Arc::downgrade(operator.state.right.at(0).record.column(0));
    let rows_pointer = operator.state.right.as_ptr() as usize;
    let index_pointer = Arc::as_ptr(operator.state.right.1.as_ref().unwrap()) as usize;
    let ProbeGate { started, release } = probe_gate(&mut operator, phase);
    let input = if mixed {
        mixed_probe_input()
    } else {
        left_batch(vec![0; PROBE_ROWS])
    };
    let inputs = input
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .map(|record| Arc::downgrade(record.column(2)))
        .collect::<Vec<_>>();
    let mut process =
        Box::pin(operator.process_data("left", input.clone(), &context, &mut collector));
    let (rows, index, thread) = tokio::select! {
        observed = started => observed.unwrap(),
        result = &mut process => panic!("probe completed before gate: {result:?}"),
    };
    assert_eq!((rows, index), (rows_pointer, index_pointer));
    assert_ne!(thread, std::thread::current().id());
    if cancel {
        job_context.cancellation().cancel();
        assert!(matches!(
            process.as_mut().await,
            Err(CalcFlowError::Cancelled { .. })
        ));
    }
    drop(process);
    drop(input);
    if mixed {
        assert!(
            inputs[0].upgrade().is_none(),
            "native worker retained dropped record"
        );
        assert!(
            inputs[1].upgrade().is_none(),
            "native worker retained replaced record"
        );
        assert!(
            inputs[2].upgrade().is_some(),
            "native worker lost detached required parent"
        );
    }
    assert_eq!(operator.state.next_left_row_id, 0);
    assert_eq!(operator.state.deltas.pending.iter().count(), 1);
    assert!(collector.drain("output").is_empty());
    assert!(job_context.gather_owner().funding().2 > 0);
    assert!(operator.probe_control.as_ref().unwrap().size() > 0);
    drop(context);
    resume_abandoned_probe(
        operator,
        job_context,
        pool,
        collector,
        (weak, inputs),
        release,
    )
    .await;
}

async fn resume_abandoned_probe(
    mut operator: StreamJoinOperator,
    job_context: StreamJobContext,
    pool: Arc<dyn datafusion::execution::memory_pool::MemoryPool>,
    mut collector: EdgeCollector,
    observed: (std::sync::Weak<dyn Array>, Vec<std::sync::Weak<dyn Array>>),
    release: std::sync::mpsc::Sender<()>,
) {
    let (weak, inputs) = observed;
    operator.reset().unwrap();
    assert!(
        weak.upgrade().is_some(),
        "worker must retain the old generation after reset"
    );
    let resumed_job = job().with_gather_owner(job_context.gather_owner().clone());
    let resumed = StreamOperatorContext::new(&resumed_job, "match", None);
    let mut mutation =
        Box::pin(operator.process_data("right", right_batch(vec![0]), &resumed, &mut collector));
    assert!(futures::poll!(mutation.as_mut()).is_pending());
    drop(mutation);
    assert_eq!(operator.state.next_right_row_id, 0);
    release.send(()).unwrap();
    operator
        .process_data("right", right_batch(vec![0]), &resumed, &mut collector)
        .await
        .unwrap();
    assert!(weak.upgrade().is_none());
    assert!(inputs.iter().all(|input| input.upgrade().is_none()));
    assert_probe_released(&operator, &resumed_job, pool.as_ref());
    operator
        .process_data(
            "left",
            left_batch(vec![0; PROBE_ROWS]),
            &resumed,
            &mut collector,
        )
        .await
        .unwrap();
    assert_literal_output(&mut collector, PROBE_ROWS, 0);
    assert_eq!(operator.state.next_left_row_id, 8_192);
    assert_eq!(operator.state.deltas.pending.iter().count(), 8_193);
    assert_eq!(
        Arc::strong_count(operator.state.right.1.as_ref().unwrap()),
        1
    );
    native_lookup_tests::assert_resident_and_gather_funding(
        pool.as_ref(),
        &resumed_job,
        native_lookup_tests::state_funding(&operator),
    );
    drop(operator);
    drop(resumed);
    assert!(
        resumed_job
            .gather_owner()
            .close_and_drain()
            .await
            .is_empty()
    );
    drop((job_context, resumed_job));
    assert_eq!(pool.reserved(), 0);
}

pub(super) async fn assert_refused_process_data(service: &TestService) {
    let mut operator = with_one_gib_pool(probe_operator());
    let job = job().with_gather_owner(service.owner("refused-probe".into()));
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right_batch(vec![0]), &context, &mut collector)
        .await
        .unwrap();
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let probe = AdmissionProbe::install(
        job.gather_owner(),
        AdmissionStage::Attempt,
        pool.clone(),
        1 << 30,
    );
    operator
        .process_data("left", left_batch(vec![0]), &context, &mut collector)
        .await
        .unwrap();
    assert!(
        probe.take_event().is_none(),
        "small process_data must stay synchronous"
    );
    assert_literal_output(&mut collector, 1, 0);
    assert_eq!(
        Arc::strong_count(operator.state.right.1.as_ref().unwrap()),
        1
    );
    native_lookup_tests::assert_resident_and_gather_funding(
        pool.as_ref(),
        &job,
        native_lookup_tests::state_funding(&operator),
    );
    reset_join_work();
    operator
        .process_data(
            "left",
            left_batch(vec![0; PROBE_ROWS]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let event = probe
        .take_event()
        .expect("large process_data must attempt actual paid admission before serial fallback");
    assert_eq!(event.stage, AdmissionStage::Attempt);
    assert_eq!(event.operator, "match");
    assert_eq!(event.available + 1, event.fee);
    assert_literal_output(&mut collector, PROBE_ROWS, 1);
    assert_eq!(join_work().sql_probe_table_builds, 0);
    assert_eq!(
        join_work().arena_frames,
        0,
        "refused worker admission must reuse recovered probe keys"
    );
    assert_eq!(join_work().key_encodings, 0);
    assert_eq!(operator.state.next_left_row_id, 8_193);
    assert_eq!(operator.state.metrics.emitted_match_rows, 8_193);
    assert_eq!(operator.state.deltas.pending.iter().count(), 8_194);
    drop(operator);
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop((probe, job));
    assert_eq!(pool.reserved(), 0);
}
