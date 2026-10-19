use super::*;
use crate::operator::join::data_work::{ProbePhase, worker::CountWork};

fn source(rows: usize) -> Batch {
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int64Array::from_iter_values(
                (0..rows + 2).map(|row| i64::try_from(row.saturating_sub(1) % 64).unwrap()),
            )),
            Arc::new(TimestampMicrosecondArray::from(vec![0; rows + 2]).with_timezone("UTC")),
            Arc::new(Int64Array::from_iter_values(
                (0..rows + 2).map(|row| i64::try_from(row.saturating_sub(1)).unwrap()),
            )),
        ],
    )
    .unwrap();
    Batch::table(
        (0..rows)
            .step_by(32_768)
            .map(|start| record.slice(1 + start, (rows - start).min(32_768)))
            .collect(),
        BatchMetadata::default(),
    )
    .unwrap()
}

fn verify_rows(output: &mut EdgeCollector, rows: usize, wide: bool) {
    let mut position = 0usize;
    for (sequence, message) in output.drain("output").into_iter().enumerate() {
        let batch = message.as_data().unwrap();
        assert_eq!(
            batch.metadata().sequence(),
            u64::try_from(sequence).unwrap()
        );
        for record in batch.table_payload().unwrap().batches() {
            assert_eq!(record.num_columns(), if wide { 40 } else { 3 });
            for row in 0..record.num_rows() {
                for (column, array) in record.columns().iter().enumerate() {
                    let expected = if wide || column == 0 {
                        position
                    } else {
                        position % 64
                    };
                    assert_eq!(
                        array
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap()
                            .value(row),
                        i64::try_from(expected).unwrap(),
                    );
                }
                position += 1;
            }
        }
    }
    assert_eq!(
        position, rows,
        "partitions must cover every input exactly once"
    );
}

fn observe(operator: &mut StreamJoinOperator) -> Arc<Mutex<[Vec<usize>; 2]>> {
    let observed = Arc::new(Mutex::new([Vec::new(), Vec::new()]));
    let units = Arc::clone(&observed);
    operator.probe_unit_test_hook = Some(Arc::new(move |phase, ordinal, complete| {
        if complete {
            units.lock()[usize::from(phase == ProbePhase::Fill)].push(ordinal);
        }
    }));
    observed
}

fn assert_units(observed: &Mutex<[Vec<usize>; 2]>, capacity: usize, gathered: bool) {
    let mut phases = observed.lock().clone();
    for (phase, ordinals) in phases.iter_mut().enumerate() {
        ordinals.sort_unstable();
        let expected = if phase == 0 || gathered { capacity } else { 0 };
        assert_eq!(
            ordinals,
            &(0..expected).collect::<Vec<_>>(),
            "phase {phase}: Count and Fill must use the captured CPU partition count"
        );
    }
}

async fn output_case(service: &TestService, capacity: usize, refuse: bool, wide: bool) {
    let rows = capacity * 8_192 + 3;
    let mut operator = projected_probe_operator();
    if wide {
        operator.set_output_projection(vec![2; 40]).unwrap();
    }
    let refusal = Arc::new(RefusingPool {
        inner: GreedyMemoryPool::new(1 << 30),
        attempts: AtomicUsize::new(0),
    });
    if refuse {
        operator
            .runtime
            .initialize_benchmark_pool(refusal.clone())
            .unwrap();
    }
    let job = job(service);
    preload_projected_probe(&mut operator, &job).await;
    let observed = observe(&mut operator);
    let context = probe_only_context(&job);
    let source = source(rows);
    let before = source.table_payload().unwrap().batches().to_vec();
    let (mut prepared, expiration) = operator
        .prepare_data_append("left", &source, &context)
        .await
        .unwrap();
    assert!(prepared.deferred.is_some());
    operator
        .finish_morsels(&mut prepared, &context)
        .await
        .unwrap();
    let gathered = !refuse;
    assert_eq!(prepared.gathered.is_some(), gathered);
    if let Some(output) = &prepared.gathered {
        assert_eq!(output.rows(), rows);
        assert_eq!(output.funding_for_test().len(), capacity);
        let funding = output.funding_for_test();
        assert!(funding.iter().all(|(backing, credit)| backing <= credit));
        if wide {
            assert!(funding.iter().map(|(_, credit)| credit).sum::<usize>() > 64 << 20);
        }
    }
    assert_units(&observed, capacity, !refuse);
    if refuse {
        assert!(refusal.attempts.load(Ordering::SeqCst) > 0);
    }
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .finish_data("left", prepared, expiration, &context, &mut output)
        .await
        .unwrap();
    verify_rows(&mut output, rows, wide);
    assert_eq!(source.table_payload().unwrap().batches(), &before);
    assert_eq!(operator.status().left.retained_rows, 0);
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop(operator);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

fn run_output(capacity: usize, refuse: bool, wide: bool) {
    let _gate = ProductionGate::new();
    let service = TestService::new(capacity, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(output_case(&service, capacity, refuse, wide));
    drop(runtime);
    service.shutdown();
}

#[test]
fn cpu_aware_probe_eight_cpus_gather_complete_ordered_partitions() {
    run_output(8, false, false);
}

#[test]
fn cpu_aware_probe_sixteen_cpus_gather_complete_ordered_partitions() {
    run_output(16, false, false);
}

#[test]
fn cpu_aware_probe_sixteen_cpus_optional_refusal_reuses_serial_fallback() {
    run_output(16, true, false);
}

#[test]
fn cpu_aware_probe_sixteen_cpus_large_output_obeys_only_configured_pool() {
    run_output(16, false, true);
}

async fn captured(
    service: &TestService,
    rows: usize,
) -> (StreamJoinOperator, Arc<ProbeInputs>, StreamJobContext) {
    let mut operator = projected_probe_operator();
    let job = job(service);
    preload_projected_probe(&mut operator, &job).await;
    let context = probe_only_context(&job);
    let source = source(rows);
    let plan = operator.begin_batch("left", &source).unwrap();
    let mut bundle = operator.admission_bundle(&plan, &source).unwrap();
    operator
        .admit_batch_records("left", &source, &plan, &context, &mut bundle)
        .await
        .unwrap();
    bundle.finish("match").unwrap();
    let keys = operator
        .native_probe_keys(&plan, &bundle.admitted)
        .unwrap()
        .unwrap();
    assert!(operator.ensure_native_index(false).unwrap());
    let inputs = operator
        .capture_keyed_probe(&plan, bundle.admitted, keys, job.gather_owner())
        .unwrap()
        .ready
        .unwrap();
    (operator, inputs, job)
}

fn run_count_fill(capacity: usize) {
    let _gate = ProductionGate::new();
    super::super::super::PROBE_COST_GATE.set(0);
    let service = TestService::new(capacity, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let rows = 32 * 8_192 + 17;
        let (mut operator, inputs, job) = captured(&service, rows).await;
        let stop = GatherStop::from_job(&job);
        let count = CountWork {
            inputs: Arc::clone(&inputs),
            limit: 1_000_000,
        };
        assert_eq!(
            count.unit_count(),
            capacity,
            "partition count must follow configured CPU capacity"
        );
        let values = (0..capacity)
            .map(|ordinal| count.run(ordinal, &stop).unwrap())
            .collect::<Vec<_>>();
        for (ordinal, value) in values.iter().enumerate() {
            assert_eq!(
                value.count,
                rows / capacity + usize::from(ordinal < rows % capacity)
            );
        }
        let counts =
            Counts::ordered(values, 1_000_000, inputs.control.as_ref().unwrap().clone()).unwrap();
        let (funding, credit) = operator.admit_fill(&counts).unwrap().unwrap();
        let fill = FillWork {
            inputs: Arc::clone(&inputs),
            counts,
            funding,
        };
        assert_eq!(fill.unit_count(), capacity);
        let fragments = (0..capacity)
            .map(|ordinal| fill.run(ordinal, &stop).unwrap())
            .collect();
        let output = pairs::ordered_output(fragments, credit, rows).unwrap();
        assert_eq!(output.pairs.len(), rows);
        for (position, pair) in output.pairs.iter().enumerate() {
            assert_eq!((pair.pos, pair.opposite_index), (position, position % 64));
        }
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop((output, fill, count, inputs, operator, stop, job));
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn cpu_aware_probe_single_cpu_freezes_one_count_and_fill_partition() {
    run_count_fill(1);
}

#[test]
fn cpu_aware_probe_thirty_two_cpus_share_count_and_fill_partition_order() {
    run_count_fill(32);
}

#[test]
fn cpu_aware_probe_dynamic_cursor_controls_are_prepaid() {
    use crate::operator::join::native_lookup::WindowCursor;

    let _gate = ProductionGate::new();
    super::super::super::PROBE_COST_GATE.set(0);
    let runtime = runtime();
    let bytes = [1, 32].map(|capacity| {
        let service = TestService::new(capacity, 1).unwrap();
        let bytes = runtime.block_on(async {
            let (mut operator, inputs, job) = captured(&service, 32 * 8_192).await;
            let bytes = inputs.control.as_ref().unwrap().size();
            let pool = operator
                .runtime
                .runtime()
                .unwrap()
                .incremental_memory_pool();
            drop((operator, inputs, job));
            assert_eq!(pool.reserved(), 0);
            bytes
        });
        service.shutdown();
        bytes
    });
    assert!(
        bytes[1] >= bytes[0] + 31 * 64 * size_of::<Option<WindowCursor>>(),
        "all additional unit-local cursors must be funded before dispatch: {bytes:?}"
    );
}

#[test]
fn cpu_aware_probe_cursor_control_estimate_has_no_internal_byte_cap() {
    let bytes = control::input_bytes(1_000_000, 32, "large-cursors").unwrap();
    assert!(bytes > 512 << 20);
}

#[test]
fn cpu_aware_probe_thirty_two_captured_partitions_gather_with_single_executor() {
    let _gate = ProductionGate::new();
    let capture_service = TestService::new(32, 1).unwrap();
    let executor = TestService::new(1, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let rows = 32 * 8_192 + 17;
        let (mut operator, inputs, capture_job) = captured(&capture_service, rows).await;
        let count = CountWork {
            inputs: Arc::clone(&inputs),
            limit: 1_000_000,
        };
        assert_eq!(count.unit_count(), 32);
        let stop = GatherStop::from_job(&capture_job);
        let values = (0..32)
            .map(|ordinal| count.run(ordinal, &stop).unwrap())
            .collect();
        let counts =
            Counts::ordered(values, 1_000_000, inputs.control.as_ref().unwrap().clone()).unwrap();
        let job = job(&executor);
        let context = StreamOperatorContext::new(&job, "match", None);
        let charges = vec![0; rows];
        let gathered = operator
            .gather_morsels(&inputs, &counts, Some(&charges), &context)
            .await
            .unwrap()
            .expect("small outputs across 32 captured partitions remain eligible");
        assert_eq!(gathered.rows(), rows);
        let funding = gathered.funding_for_test();
        assert_eq!(
            funding.len(),
            32,
            "gather must retain the captured partition count"
        );
        assert!(funding.iter().all(|(backing, credit)| backing <= credit));
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        let mut sequence = 0;
        gathered
            .emit("match", &context, &mut output, &mut sequence)
            .await
            .unwrap();
        verify_rows(&mut output, rows, false);
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop((gathered, counts, count, inputs, operator));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        assert!(
            capture_job
                .gather_owner()
                .close_and_drain()
                .await
                .is_empty()
        );
        drop(context);
        drop((job, capture_job));
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    executor.shutdown();
    capture_service.shutdown();
}
