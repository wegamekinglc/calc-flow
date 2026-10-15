use super::*;
use crate::runtime::streaming::gather_work::TestService;

const ROWS: usize = 64_000;

mod boundaries;
mod lifecycle;
#[cfg(feature = "benchmark-internals")]
mod paired;
mod support;

#[test]
fn test_morsel_gathers_before_fill_unit_settlement() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut declaration = spec();
        declaration.limits = JoinStateLimits::new(100_000, 100_000_000, 1_000_000).unwrap();
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
        let job = job().with_gather_owner(service.owner("morsels".into()));
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("right", right_batch(vec![0; 8]), &context, &mut collector)
            .await
            .unwrap();
        let completed = Arc::new(std::sync::Mutex::new(Vec::new()));
        let observed = completed.clone();
        operator.probe_unit_test_hook = Some(Arc::new(move |phase, unit, done| {
            if phase == data_work::ProbePhase::Fill && done {
                assert!(
                    join_work().output_column_takes > 0,
                    "a morsel must gather on its probe worker before settlement"
                );
                observed.lock().unwrap().push(unit);
            }
        }));
        operator
            .process_data("left", left_batch(vec![0; ROWS]), &context, &mut collector)
            .await
            .unwrap();
        let mut units = completed.lock().unwrap().clone();
        units.sort_unstable();
        assert_eq!(units, [0, 1, 2, 3]);
        let messages = collector.drain("output");
        assert_eq!(
            messages
                .iter()
                .map(|message| message.as_data().unwrap().num_rows())
                .sum::<usize>(),
            ROWS * 8
        );
        assert_eq!(operator.status().left.retained_rows, ROWS as u64);
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
    });
    drop(runtime);
    service.shutdown();
}
