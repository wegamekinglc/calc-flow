use super::*;
use crate::runtime::streaming::gather_work::TestService;

#[test]
fn test_cpu_pipeline_retained_preparation_leaves_owner_and_preserves_rows() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let rows = 65_539;
        let mut declaration = spec();
        declaration.limits = JoinStateLimits::new(100_000, 100_000_000, 100_000).unwrap();
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
        let job = job().with_gather_owner(service.owner("retained-cpu".into()));
        let context = StreamOperatorContext::new(&job, "match", None);
        let source = left_batch((0..rows).map(|row| i64::try_from(row).unwrap()).collect());
        let before = source.table_payload().unwrap().batches().to_vec();
        reset_join_work();
        let prepared = operator
            .prepare_batch("left", &source, &context)
            .await
            .unwrap();
        assert_eq!(prepared.chunks.len(), rows);
        assert_eq!(prepared.next_row_id, u64::try_from(rows).unwrap());
        for index in 0..rows {
            let row = prepared.chunks.row(index);
            assert_eq!(row.row_id, u64::try_from(index).unwrap());
            assert_eq!(
                row.event_time,
                EventTime::from_micros(i64::try_from(index).unwrap())
            );
            assert_eq!(row.record.offset(), index);
            assert_eq!(
                row.charge,
                state_row_charge(&before[0], index, &[0], "match").unwrap()
            );
        }
        assert_eq!(source.table_payload().unwrap().batches(), &before);
        assert_eq!(
            join_work().charge_column_resolutions,
            0,
            "large retained charge analysis must execute on native CPU workers"
        );
        assert_eq!(
            operator.state.left.len(),
            0,
            "preparation must not commit state"
        );
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop(prepared);
        drop(operator);
        drop(context);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(job);
        assert_eq!(pool.reserved(), 0);
    });
    drop(runtime);
    service.shutdown();
}
