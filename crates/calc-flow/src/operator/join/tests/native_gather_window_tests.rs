use super::super::super::native_lookup::{reset_window_run_searches, window_run_searches};
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
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
use std::time::Duration;

fn window_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
    ]))
}

fn window_batch(key: i64, times: &[i64]) -> Batch {
    keyed_window_batch(vec![key; times.len()], times)
}

fn keyed_window_batch(keys: Vec<i64>, times: &[i64]) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                window_schema(),
                vec![
                    Arc::new(Int64Array::from(keys)),
                    Arc::new(TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC")),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

// A gather probe over one key: event-time-ordered rows across two units, with
// one out-of-order row and one recovering row at the tail of the second unit.
async fn gather_fixture(
    owner: JobGatherOwner,
) -> (StreamJoinOperator, Arc<ProbeInputs>, StreamJobContext) {
    gather_fixture_side(owner, true, false, None).await
}

async fn gather_fixture_side(
    owner: JobGatherOwner,
    incoming_is_left: bool,
    missing: bool,
    pool: Option<Arc<dyn MemoryPool>>,
) -> (StreamJoinOperator, Arc<ProbeInputs>, StreamJobContext) {
    gather_fixture_rows(owner, incoming_is_left, missing, pool, None).await
}

async fn gather_fixture_rows(
    owner: JobGatherOwner,
    incoming_is_left: bool,
    missing: bool,
    pool: Option<Arc<dyn MemoryPool>>,
    fanout: Option<usize>,
) -> (StreamJoinOperator, Arc<ProbeInputs>, StreamJobContext) {
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(Duration::from_micros(10), Duration::from_micros(20)).unwrap(),
        JoinStateLimits::new(100_000, 100_000_000, 1_000_000).unwrap(),
    )
    .unwrap();
    let mut operator =
        StreamJoinOperator::new("probe", window_schema(), window_schema(), spec).unwrap();
    if let Some(pool) = pool {
        operator.runtime.initialize_benchmark_pool(pool).unwrap();
    }
    let job = StreamJobContext::new(91, "probe", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(owner);
    let context = StreamOperatorContext::new(&job, "probe", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let right_times = if let Some(fanout) = fanout {
        vec![10; fanout]
    } else {
        let mut right_times = Vec::new();
        for index in 0..40_i64 {
            let time = index * 3;
            right_times.push(time);
            if index % 4 == 0 {
                right_times.push(time);
            }
        }
        right_times
    };
    operator
        .process_data(
            if incoming_is_left { "right" } else { "left" },
            window_batch(7, &right_times),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    let rows = 16_400;
    let mut probe_times: Vec<i64> = (0..rows - 2).map(|index| i64::from(index) * 10).collect();
    probe_times.extend([5, 15]);
    let input = keyed_window_batch(
        (0..probe_times.len())
            .map(|row| if missing && row % 17 == 0 { 99 } else { 7 })
            .collect(),
        &probe_times,
    );
    let ingress = if incoming_is_left { "left" } else { "right" };
    let plan = operator.begin_batch(ingress, &input).unwrap();
    let mut bundle = operator.admission_bundle(&plan, &input).unwrap();
    operator
        .admit_batch_records(ingress, &input, &plan, &context, &mut bundle)
        .await
        .unwrap();
    bundle.finish("probe").unwrap();
    let keys = operator
        .native_probe_keys(&plan, &bundle.admitted)
        .unwrap()
        .unwrap();
    assert!(operator.ensure_native_index(!incoming_is_left).unwrap());
    let inputs = operator
        .capture_keyed_probe(&plan, bundle.admitted, keys, job.gather_owner())
        .unwrap()
        .ready
        .unwrap();
    assert_eq!(inputs.partition.units(), 2, "fixture must span two units");
    (operator, inputs, job)
}

// Plain uncached window per probe row: the oracle for gathered pairs.
fn expected_pairs(inputs: &ProbeInputs) -> Vec<(usize, usize)> {
    let index = inputs.index.as_ref().expect("owned native index");
    (0..inputs.admitted.len())
        .flat_map(|position| {
            inputs
                .slot(position)
                .into_iter()
                .flat_map(|id| index.range_by_id(id, inputs.range(position)))
                .map(|opposite_index| (position, opposite_index))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn count_windows(inputs: &Arc<ProbeInputs>, job: &StreamJobContext, limit: u64) -> Counts {
    let work = CountWork {
        inputs: Arc::clone(inputs),
        limit,
    };
    let values = (0..work.unit_count())
        .map(|ordinal| work.run(ordinal, &GatherStop::from_job(job)).unwrap())
        .collect();
    Counts::ordered(values, limit, inputs.control.as_ref().unwrap().clone()).unwrap()
}

#[test]
fn test_bulk_window_fill_stops_inside_large_fanout_at_existing_item_quantum() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let service = crate::runtime::streaming::gather_work::TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let (_operator, inputs, job) = gather_fixture_rows(
            service.owner("bulk-window-stop".into()),
            true,
            false,
            None,
            Some(10_000),
        )
        .await;
        let counts = count_windows(&inputs, &job, 1_000_000);
        for full_checks in [false, true] {
            let polls = Arc::new(AtomicUsize::new(0));
            let observed = Arc::clone(&polls);
            let plain = GatherStop::from_job(&job);
            let signal = plain.clone();
            let stop = plain.with_check_hook(Arc::new(move |atomic_only| {
                assert_ne!(atomic_only, full_checks);
                if observed.fetch_add(1, Ordering::SeqCst) == 1 {
                    signal.abandon_for_test();
                }
            }));
            crate::operator::join::reset_join_work();
            let result = if full_checks {
                super::super::worker::fill_all(&inputs, &counts, &stop)
            } else {
                super::super::worker::fill_range(&inputs, &counts, 0, &stop)
            };
            assert!(matches!(
                result,
                Err(crate::CalcFlowError::Cancelled { .. })
            ));
            assert_eq!(polls.load(Ordering::SeqCst), 2);
            let visits = crate::operator::join::join_work().native_range_visits;
            assert!(
                (4_095..=4_097).contains(&visits),
                "a large native window must stop at the existing item quantum: {visits}"
            );
        }
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_gather_probe_windows_use_cursors_instead_of_per_row_searches() {
    let service = crate::runtime::streaming::gather_work::TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(check_gather_windows(service.owner("window-probe".into())));
    drop(runtime);
    service.shutdown();
}

async fn check_gather_windows(owner: JobGatherOwner) {
    let (mut operator, inputs, job) = gather_fixture(owner).await;
    let expected = expected_pairs(&inputs);
    assert!(!expected.is_empty());
    let stop = GatherStop::from_job(&job);
    let count = CountWork {
        inputs: Arc::clone(&inputs),
        limit: 1_000_000,
    };
    let units = count.unit_count();
    reset_window_run_searches();
    let values = (0..units)
        .map(|unit| count.run(unit, &stop).unwrap())
        .collect::<Vec<_>>();
    let counted = window_run_searches();
    let counts =
        Counts::ordered(values, 1_000_000, inputs.control.as_ref().unwrap().clone()).unwrap();
    let (funding, credit) = operator.admit_fill(&counts).unwrap().unwrap();
    let total = counts.total;
    let fill = FillWork {
        inputs: Arc::clone(&inputs),
        counts,
        funding,
    };
    reset_window_run_searches();
    crate::operator::join::matches::reset_pair_pushes();
    let fragments = (0..units)
        .map(|unit| fill.run(unit, &stop).unwrap())
        .collect::<Vec<_>>();
    let filled = window_run_searches();
    assert_eq!(
        crate::operator::join::matches::pair_pushes(),
        0,
        "worker fill must append native windows without per-pair selection bookkeeping"
    );
    let output = pairs::ordered_output(fragments, credit, total).unwrap();
    let identities = output
        .pairs
        .iter()
        .map(|pair| (pair.pos, pair.opposite_index))
        .collect::<Vec<_>>();
    assert_eq!(identities, expected);
    // Count pays one first touch per unit and one out-of-order fallback.
    assert_eq!(counted, 2 * (2 + 1), "count-phase full-run searches");
    assert_eq!(filled, 0, "fill must reuse the completed count windows");
}

#[test]
fn test_interval_cached_windows_preserve_both_sides_boundaries_and_missing_keys() {
    let service = crate::runtime::streaming::gather_work::TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        for incoming_is_left in [false, true] {
            let (mut operator, inputs, job) = gather_fixture_side(
                service.owner("interval-window-parity".into()),
                incoming_is_left,
                true,
                None,
            )
            .await;
            let expected = expected_pairs(&inputs);
            assert!(!expected.is_empty());
            let stop = GatherStop::from_job(&job);
            let count = CountWork {
                inputs: inputs.clone(),
                limit: 1_000_000,
            };
            let values = (0..count.unit_count())
                .map(|unit| count.run(unit, &stop).unwrap())
                .collect();
            let counts =
                Counts::ordered(values, 1_000_000, inputs.control.as_ref().unwrap().clone())
                    .unwrap();
            let (funding, credit) = operator.admit_fill(&counts).unwrap().unwrap();
            let total = counts.total;
            let fill = FillWork {
                inputs: inputs.clone(),
                counts,
                funding,
            };
            reset_window_run_searches();
            let fragments = (0..fill.unit_count())
                .map(|unit| fill.run(unit, &stop).unwrap())
                .collect();
            let searches = window_run_searches();
            let output = pairs::ordered_output(fragments, credit, total).unwrap();
            assert_eq!(
                output
                    .pairs
                    .iter()
                    .map(|pair| (pair.pos, pair.opposite_index))
                    .collect::<Vec<_>>(),
                expected
            );
            assert_eq!(
                searches, 0,
                "cached windows avoid fill searches on either ingress"
            );
        }
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_interval_window_cache_reserves_bounds_and_refunds_owned_count_results() {
    let service = crate::runtime::streaming::gather_work::TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let (mut operator, inputs, job) =
            gather_fixture(service.owner("interval-window-funding".into())).await;
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        let baseline = pool.reserved();
        let work = CountWork {
            inputs: inputs.clone(),
            limit: 1_000_000,
        };
        let stop = GatherStop::from_job(&job);
        let values = (0..work.unit_count())
            .map(|unit| work.run(unit, &stop).unwrap())
            .collect::<Vec<_>>();
        assert!(
            pool.reserved() >= baseline + inputs.admitted.len() * 2 * size_of::<usize>(),
            "count outputs retain prepaid window bounds"
        );
        drop(values);
        assert_eq!(
            pool.reserved(),
            baseline,
            "cached row bounds drop before their lease"
        );
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_interval_cancelled_count_discards_prepaid_partial_windows() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let service = crate::runtime::streaming::gather_work::TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let (mut operator, inputs, job) =
            gather_fixture(service.owner("interval-window-cancel".into())).await;
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        let baseline = pool.reserved();
        let observed = Arc::new(AtomicUsize::new(0));
        let polls = Arc::new(AtomicUsize::new(0));
        let peak = observed.clone();
        let checks = polls.clone();
        let measured = pool.clone();
        let plain = GatherStop::from_job(&job);
        let signal = plain.clone();
        let stop = plain.with_check_hook(Arc::new(move |atomic_only| {
            if atomic_only {
                peak.fetch_max(measured.reserved(), Ordering::SeqCst);
                if checks.fetch_add(1, Ordering::SeqCst) == 1 {
                    signal.abandon_for_test();
                }
            }
        }));
        let work = CountWork {
            inputs: inputs.clone(),
            limit: 1_000_000,
        };
        assert!(matches!(
            work.run(0, &stop),
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
        assert_eq!(polls.load(Ordering::SeqCst), 2);
        assert_eq!(pool.reserved(), baseline);
        assert!(
            observed.load(Ordering::SeqCst)
                >= baseline + inputs.partition.range(0).len() * 2 * size_of::<usize>(),
            "cache must be prepaid before counting and discarded after cancellation"
        );
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_interval_cache_refusal_keeps_uncached_fill_and_exact_pairs() {
    let service = crate::runtime::streaming::gather_work::TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let (mut operator, inputs, job) =
            gather_fixture(service.owner("interval-window-refusal".into())).await;
        let expected = expected_pairs(&inputs);
        let mut inputs = Arc::try_unwrap(inputs).unwrap_or_else(|_| panic!("unique input owner"));
        let original = inputs.control.take();
        let refusing: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(0));
        inputs.control = Some(Arc::new(
            MemoryConsumer::new("cache-refusal").register(&refusing),
        ));
        let inputs = Arc::new(inputs);
        let counts = count_windows(&inputs, &job, 1_000_000);
        assert!((0..counts.units).all(|ordinal| counts.windows(&inputs, ordinal).is_none()));
        assert_eq!(refusing.reserved(), 0);
        let (funding, credit) = operator.admit_fill(&counts).unwrap().unwrap();
        let total = counts.total;
        let fill = FillWork {
            inputs,
            counts,
            funding,
        };
        reset_window_run_searches();
        let fragments = (0..fill.unit_count())
            .map(|ordinal| fill.run(ordinal, &GatherStop::from_job(&job)).unwrap())
            .collect();
        assert_eq!(window_run_searches(), 6);
        let output = pairs::ordered_output(fragments, credit, total).unwrap();
        assert_eq!(
            output
                .pairs
                .iter()
                .map(|pair| (pair.pos, pair.opposite_index))
                .collect::<Vec<_>>(),
            expected
        );
        drop(original);
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_interval_cached_windows_bind_snapshot_and_survive_input_relocation() {
    let service = crate::runtime::streaming::gather_work::TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let (_operator, inputs, job) =
            gather_fixture(service.owner("interval-window-binding".into())).await;
        let expected = expected_pairs(&inputs);
        let index_owners = Arc::strong_count(inputs.index.as_ref().unwrap());
        let mut counts = count_windows(&inputs, &job, 1_000_000);
        assert_eq!(
            Arc::strong_count(inputs.index.as_ref().unwrap()),
            index_owners
        );
        let mut inputs =
            Arc::try_unwrap(inputs).unwrap_or_else(|_| panic!("cache must not pin input owner"));
        assert!((0..counts.units).all(|ordinal| counts.windows(&inputs, ordinal).is_some()));
        inputs.incoming_is_left = false;
        assert!(counts.windows(&inputs, 0).is_none());
        inputs.incoming_is_left = true;
        let original = inputs.control.take();
        inputs.control = Some(Arc::new(original.as_ref().unwrap().new_empty()));
        assert!(
            counts.windows(&inputs, 0).is_none(),
            "another snapshot cannot reuse cached bounds"
        );
        inputs.control = original;
        inputs.bounds = JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap();
        assert!(counts.windows(&inputs, 0).is_none());
        inputs.bounds =
            JoinTimeBounds::new(Duration::from_micros(10), Duration::from_micros(20)).unwrap();
        let inputs = Arc::new(inputs);
        reset_window_run_searches();
        crate::operator::join::matches::reset_pair_pushes();
        let pairs =
            super::super::worker::fill_all(&inputs, &counts, &GatherStop::from_job(&job)).unwrap();
        assert_eq!(
            crate::operator::join::matches::pair_pushes(),
            0,
            "serial fill must append native windows without per-pair selection bookkeeping"
        );
        assert_eq!(
            window_run_searches(),
            0,
            "serial refusal fallback also reuses completed bounds"
        );
        assert_eq!(
            pairs
                .iter()
                .map(|pair| (pair.pos, pair.opposite_index))
                .collect::<Vec<_>>(),
            expected
        );
        assert!(counts.discard_windows());
        assert!(!counts.discard_windows());
        assert!(counts.windows(&inputs, 0).is_none());
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_interval_limit_sentinel_and_final_stop_never_publish_cached_bounds() {
    let service = crate::runtime::streaming::gather_work::TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let (mut operator, inputs, job) =
            gather_fixture(service.owner("interval-window-incomplete".into())).await;
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        let baseline = pool.reserved();
        let work = CountWork {
            inputs: inputs.clone(),
            limit: 1,
        };
        let result = work.run(0, &GatherStop::from_job(&job)).unwrap();
        assert_eq!(result.count, 2);
        assert!(result.windows.is_none());
        assert_eq!(pool.reserved(), baseline);
        let plain = GatherStop::from_job(&job);
        let signal = plain.clone();
        let stop = plain.with_check_hook(Arc::new(move |atomic_only| {
            if !atomic_only {
                signal.abandon_for_test();
            }
        }));
        let work = CountWork {
            inputs,
            limit: 1_000_000,
        };
        assert!(matches!(
            work.run(0, &stop),
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
        assert_eq!(
            pool.reserved(),
            baseline,
            "completed but unaccepted cache is refunded"
        );
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn test_interval_bounded_deferred_count_releases_cache_before_state_admission() {
    let service = crate::runtime::streaming::gather_work::TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        for bounded in [false, true] {
            let pool: Arc<dyn MemoryPool> = if bounded {
                Arc::new(GreedyMemoryPool::new(100_000_000))
            } else {
                Arc::new(datafusion::execution::memory_pool::UnboundedMemoryPool::default())
            };
            let (mut operator, inputs, job) = gather_fixture_side(
                service.owner("interval-window-deferred-budget".into()),
                true,
                false,
                Some(pool.clone()),
            )
            .await;
            let baseline = pool.reserved();
            let counts = count_windows(&inputs, &job, 1_000_000);
            let cached = pool.reserved();
            assert!(cached > baseline);
            let fallback = inputs.control.as_ref().unwrap().new_empty();
            let mut admitted = AdmittedBatch::default();
            let prepared = operator
                .defer_probe(
                    inputs,
                    &mut admitted,
                    counts,
                    fallback,
                    &StreamOperatorContext::new(&job, "probe", None),
                )
                .await
                .unwrap();
            assert_eq!(
                pool.reserved(),
                if bounded { baseline } else { cached },
                "only bounded deferred work releases optional windows before state admission"
            );
            drop(prepared);
            assert!(pool.reserved() <= baseline);
        }
    });
    drop(runtime);
    service.shutdown();
}
