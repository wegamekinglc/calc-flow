use super::super::super::{join_work, reset_join_work};
use super::*;
use crate::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, JoinStateLimits, JoinTimeBounds,
    JsonMap, OperatorMetadata, StreamJoinSpec, StreamOperator,
};
use datafusion::arrow::{
    array::{Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use std::sync::atomic::{AtomicUsize, Ordering};

const GROUP: usize = 4_096;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
    ]))
}

fn batch(rows: usize) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int64Array::from(vec![7; rows])),
                    Arc::new(TimestampMicrosecondArray::from(vec![0; rows]).with_timezone("UTC")),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

async fn fixture(
    rows: usize,
    opposite: usize,
) -> (StreamJoinOperator, Arc<ProbeInputs>, StreamJobContext) {
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(std::time::Duration::ZERO, std::time::Duration::ZERO).unwrap(),
        JoinStateLimits::new(100_000, 100_000_000, 1_000_000).unwrap(),
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new("probe", schema(), schema(), spec).unwrap();
    let job = StreamJobContext::new(91, "probe", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "probe", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", batch(opposite), &context, &mut output)
        .await
        .unwrap();
    let input = batch(rows);
    let plan = operator.begin_batch("left", &input).unwrap();
    let mut bundle = operator.admission_bundle(&plan, &input).unwrap();
    operator
        .admit_batch_records("left", &input, &plan, &context, &mut bundle)
        .await
        .unwrap();
    bundle.finish("probe").unwrap();
    let keys = operator
        .native_probe_keys(&plan, &bundle.admitted)
        .unwrap()
        .unwrap();
    assert!(operator.ensure_native_index(false).unwrap());
    super::super::relax_probe_cost_gate_for_test();
    let inputs = operator
        .capture_keyed_probe(&plan, bundle.admitted, keys, job.gather_owner())
        .unwrap()
        .ready
        .unwrap();
    (operator, inputs, job)
}

#[tokio::test]
async fn test_probe_stop_checks_scale_with_groups_and_units() {
    // Uneven partitions, misses, short runs, one long run and many nearly
    // group-sized runs exercise both position and aggregate match bounds.
    for (rows, opposite) in [(24_579, 1), (8_193, 0), (4_097, 3), (1, 8_193), (3, 4_095)] {
        let (mut operator, inputs, job) = fixture(rows, opposite).await;
        let full_checks = Arc::new(AtomicUsize::new(0));
        let polls = Arc::new(AtomicUsize::new(0));
        let expensive = full_checks.clone();
        let cheap = polls.clone();
        let stop = GatherStop::from_job(&job).with_check_hook(Arc::new(move |atomic_only| {
            if atomic_only { &cheap } else { &expensive }.fetch_add(1, Ordering::SeqCst);
        }));
        let count = CountWork {
            inputs: inputs.clone(),
            limit: 1_000_000,
        };
        let units = count.unit_count();
        let expected_polls = |width: usize| {
            (0..units)
                .map(|unit| super::super::partition::range(rows, unit).len() * width)
                .map(|work| work.div_ceil(GROUP))
                .sum::<usize>()
        };
        let values = (0..units)
            .map(|unit| count.run(unit, &stop).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values.iter().sum::<usize>(), rows * opposite);
        assert_eq!(full_checks.swap(0, Ordering::SeqCst), units);
        assert_eq!(polls.swap(0, Ordering::SeqCst), expected_polls(1));
        let counts = Counts::ordered(values, 1_000_000).unwrap();
        let (funding, credit) = operator.admit_fill(counts).unwrap().unwrap();
        let fill = FillWork {
            inputs: inputs.clone(),
            counts,
            funding,
        };
        let fragments = (0..units)
            .map(|unit| fill.run(unit, &stop).unwrap())
            .collect();
        let output = pairs::ordered_output(fragments, credit, counts.total).unwrap();
        assert_eq!(full_checks.load(Ordering::SeqCst), 2 * units);
        assert_eq!(
            polls.load(Ordering::SeqCst),
            expected_polls(opposite.max(1))
        );
        assert_eq!(output.pairs.len(), rows * opposite);
        for (ordinal, pair) in output.pairs.iter().enumerate() {
            assert_eq!(
                (pair.pos, pair.opposite_index),
                (ordinal / opposite, ordinal % opposite)
            );
        }
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop((output, fill, count, inputs, operator, stop, job));
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_probe_stops_at_the_next_position_or_match_group() {
    for (rows, opposite, fill_pass) in [
        (8_193, 1, false),
        (8_193, 1, true),
        (1, 8_193, true),
        (3, 4_095, true),
    ] {
        let (mut operator, inputs, job) = fixture(rows, opposite).await;
        let count = CountWork {
            inputs: inputs.clone(),
            limit: 1_000_000,
        };
        let plain = GatherStop::from_job(&job);
        let counts = Counts::ordered(vec![count.run(0, &plain).unwrap()], 1_000_000).unwrap();
        let (funding, credit) = operator.admit_fill(counts).unwrap().unwrap();
        let fill = FillWork {
            inputs: inputs.clone(),
            counts,
            funding,
        };
        let polls = Arc::new(AtomicUsize::new(0));
        let observed = polls.clone();
        let signal = plain.clone();
        let stop = plain.with_check_hook(Arc::new(move |atomic_only| {
            if atomic_only && observed.fetch_add(1, Ordering::SeqCst) == 1 {
                signal.abandon_for_test();
            }
        }));
        reset_join_work();
        let cancelled = if fill_pass {
            fill.run(0, &stop).map(|_| ())
        } else {
            count.run(0, &stop).map(|_| ())
        };
        assert!(matches!(
            cancelled,
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
        assert_eq!(polls.load(Ordering::SeqCst), 2);
        if fill_pass {
            // The iterator can fetch one match before the next group poll.
            assert!(join_work().native_range_visits <= GROUP + 1);
        } else {
            // One first-touch search plus one gallop comparison per ordered
            // row: cursor advance replaces the two per-row binary searches.
            assert_eq!(join_work().native_boundary_visits, GROUP + 1);
        }
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop((fill, credit, count, inputs, operator, stop, job));
        assert_eq!(pool.reserved(), 0);
    }
}

#[test]
fn test_probe_owner_cancel_and_deadline_stop_group_pollers_and_refund() {
    let service = crate::runtime::streaming::gather_work::TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        for deadline in [false, true] {
            for fill_pass in [false, true] {
                let (rows, opposite) = if fill_pass { (1, 8_193) } else { (8_193, 1) };
                let (mut operator, inputs, fixture_job) = fixture(rows, opposite).await;
                let count = CountWork {
                    inputs: inputs.clone(),
                    limit: 1_000_000,
                };
                if fill_pass {
                    let counts = Counts::ordered(
                        vec![count.run(0, &GatherStop::from_job(&fixture_job)).unwrap()],
                        1_000_000,
                    )
                    .unwrap();
                    drop(count);
                    let (funding, credit) = operator.admit_fill(counts).unwrap().unwrap();
                    assert!(credit.is_none());
                    assert_owner_stop(
                        &operator,
                        &service,
                        FillWork {
                            inputs: inputs.clone(),
                            counts,
                            funding,
                        },
                        deadline,
                    )
                    .await;
                } else {
                    assert_owner_stop(&operator, &service, count, deadline).await;
                }
                assert_eq!(operator.state.next_left_row_id, 0);
                assert_eq!(operator.state.next_output_sequence, 0);
                let pool = operator
                    .runtime
                    .runtime()
                    .unwrap()
                    .incremental_memory_pool();
                drop((inputs, operator, fixture_job));
                assert_eq!(pool.reserved(), 0);
            }
        }
    });
    drop(runtime);
    service.shutdown();
}

async fn assert_owner_stop<W: ParallelCpuWork>(
    operator: &StreamJoinOperator,
    service: &crate::runtime::streaming::gather_work::TestService,
    work: W,
    deadline: bool,
) {
    struct Release(std::sync::mpsc::Sender<()>);
    impl Drop for Release {
        fn drop(&mut self) {
            let _ = self.0.send(());
        }
    }
    let job = StreamJobContext::new(
        92,
        "probe-owner-stop",
        JsonMap::new(),
        deadline.then(|| chrono::Utc::now() + chrono::Duration::seconds(2)),
        CancellationToken::new(),
    )
    .with_gather_owner(service.owner("probe-owner-stop".into()));
    let context = StreamOperatorContext::new(&job, "probe", None);
    let scope = job
        .gather_owner()
        .client(GatherOperatorId::new("probe".into()))
        .scope()
        .unwrap();
    let polls = Arc::new(AtomicUsize::new(0));
    let observed = polls.clone();
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, wait) = std::sync::mpsc::channel();
    let release = Release(release);
    let gate = Mutex::new((Some(entered), wait));
    let stop = GatherStop::from_job(&job).with_check_hook(Arc::new(move |atomic_only| {
        if atomic_only && observed.fetch_add(1, Ordering::SeqCst) == 1 {
            let mut gate = gate.lock();
            gate.0.take().unwrap().send(()).unwrap();
            gate.1
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
        }
    }));
    let mut cleanup = None;
    let ticket = scope
        .submit_observed_parallel_work(
            Arc::new(work),
            operator.probe_control.as_ref().unwrap().new_empty(),
            stop,
            job.gather_owner().retain_retirement().unwrap(),
            &mut cleanup,
        )
        .await
        .unwrap();
    let mut finish = Box::pin(finish_probe_ticket(
        ticket,
        &context,
        |_| -> Result<()> { panic!("stopped probe must not install results") },
        None,
    ));
    tokio::select! {
        result = started => result.unwrap(),
        result = &mut finish => panic!("probe did not reach its second group: {result:?}"),
    }
    if !deadline {
        job.cancellation().cancel();
    }
    assert!(matches!(
        finish.await,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert!(job.gather_owner().funding().2 > 0);
    drop(release);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    assert_eq!(polls.load(Ordering::SeqCst), 2);
    let (_, generation, attempt) = job.gather_owner().funding();
    assert_eq!((generation, attempt), (0, 0));
}
