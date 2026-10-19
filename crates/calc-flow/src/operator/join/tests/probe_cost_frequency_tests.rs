use super::*;
use crate::operator::join::native_lookup::{key_id_reads, reset_key_id_reads};

#[test]
fn scaling_probe_cost_saturates_frequency_products_and_ignores_missing_keys() {
    assert_eq!(
        estimated_key_visits(&[usize::MAX, 5], &[Some(0), Some(1)], |_| 2),
        usize::MAX
    );
    assert_eq!(
        estimated_key_visits(&[0, 2, 9], &[Some(0), None, Some(2)], |id| {
            if id == 0 { usize::MAX } else { 3 }
        }),
        27
    );
    assert_eq!(
        estimated_key_visits(&[usize::MAX], &[], |_| panic!(
            "missing slot cannot be read"
        )),
        0
    );
}

async fn prepare_cost_capture(
    operator: &mut StreamJoinOperator,
    source: &Batch,
    context: &StreamOperatorContext<'_>,
) -> (SidePlan, AdmittedBatch, NativeKeys) {
    let plan = operator.begin_batch("left", source).unwrap();
    let mut bundle = operator.admission_bundle(&plan, source).unwrap();
    operator
        .admit_batch_records("left", source, &plan, context, &mut bundle)
        .await
        .unwrap();
    bundle.finish("match").unwrap();
    assert!(bundle.admitted.iter().all(|row| !row.retain));
    let keys = operator
        .prepare_key_identities(&plan, &bundle.admitted, Some(context))
        .await
        .unwrap()
        .unwrap();
    assert!(operator.ensure_native_index(false).unwrap());
    (plan, bundle.admitted, keys)
}

#[test]
fn scaling_probe_cost_uses_distinct_frequencies_without_reading_row_ids() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let mut operator = operator();
        let job = job(&service);
        for _ in 0..8 {
            preload(&mut operator, &job).await;
        }
        let context = probe_only_context(&job);
        let source = input(false);
        let (plan, admitted, keys) = prepare_cost_capture(&mut operator, &source, &context).await;
        assert_eq!(keys.keys.len(), 64);
        reset_key_id_reads();
        let capture = operator
            .capture_keyed_probe(&plan, admitted, keys, job.gather_owner())
            .unwrap();
        assert!(
            capture.ready.is_some(),
            "probe-only rows still contribute all 512k key-run visits"
        );
        assert_eq!(
            key_id_reads(),
            0,
            "probe admission cost must aggregate keys without another admitted-row scan"
        );
        drop((capture, operator));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
    });
    drop(runtime);
    service.shutdown();
}

#[test]
fn scaling_probe_cost_ignores_unknown_probe_only_keys_without_slot_index_panic() {
    let service = TestService::new(4, 1).unwrap();
    let runtime = runtime();
    runtime.block_on(async {
        let mut operator = operator();
        let job = job(&service);
        preload(&mut operator, &job).await;
        let context = probe_only_context(&job);
        let record = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from_iter_values((0..ROWS).map(|row| {
                    if row.is_multiple_of(2) {
                        999
                    } else {
                        i64::try_from(row % 64).unwrap()
                    }
                }))),
                Arc::new(TimestampMicrosecondArray::from(vec![0; ROWS]).with_timezone("UTC")),
                Arc::new(Int64Array::from_iter_values(
                    (0..ROWS).map(|row| i64::try_from(row).unwrap()),
                )),
            ],
        )
        .unwrap();
        let source = Batch::table(vec![record], BatchMetadata::default()).unwrap();
        let (plan, admitted, keys) = prepare_cost_capture(&mut operator, &source, &context).await;
        assert_eq!(
            (0..ROWS).filter(|&row| keys.id(row) == u32::MAX).count(),
            ROWS / 2
        );
        let capture = operator
            .capture_keyed_probe(&plan, admitted, keys, job.gather_owner())
            .unwrap();
        drop((capture, operator));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
    });
    drop(runtime);
    service.shutdown();
}
