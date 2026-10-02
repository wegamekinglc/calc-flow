use super::*;
use datafusion::arrow::array::Array;

#[tokio::test(flavor = "current_thread")]
async fn test_a03_a10_projected_restore_preserves_multiplicity_and_funding() {
    let (mut operator, left, right) = prefix_fixture();
    operator.set_output_projection(vec![2, 5, 5]).unwrap();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right, &context, &mut output)
        .await
        .unwrap();
    operator
        .process_data("left", left, &context, &mut output)
        .await
        .unwrap();
    let snapshot = operator.capture(Epoch::INITIAL).unwrap();
    assert_eq!(
        snapshot.inline_metadata["layout_version"],
        serde_json::json!(4)
    );
    let (mut restored, _, _) = prefix_fixture();
    restored.set_output_projection(vec![2, 5, 5]).unwrap();
    restored.restore(&snapshot).unwrap();
    assert!(restored.state.right.auxiliary_bytes() > 0);
    assert_eq!(
        restored.runtime.pool.reserved(),
        restored.state.right.auxiliary_bytes()
    );
    let mut output = EdgeCollector::new(restored.output_ports().to_vec());
    workspace::take_output_source_registrations();
    restored.on_end(&context, &mut output).await.unwrap();
    assert_eq!(workspace::take_output_source_registrations(), 2);
    let delivered = output.drain("output");
    assert_eq!(delivered.len(), 1);
    let record = &delivered[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!((record.num_columns(), record.num_rows()), (3, 3));
    let left = record
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(left.values().as_ref(), &[1, 2, 3]);
    for index in [1, 2] {
        let right = record
            .column(index)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(right.values().as_ref(), &[1, 1, 1]);
        assert_eq!(right.null_count(), 0);
    }
    assert_eq!(restored.status.matched_rows, 3);
    assert_eq!(restored.status.state_bytes, 0);
    assert_eq!(restored.runtime.pool.reserved(), 0);
}

#[tokio::test]
async fn test_a03_a10_zero_column_cancelled_prefix_recovers_exactly() {
    let (mut operator, left, right) = prefix_fixture();
    operator.set_output_projection(Vec::new()).unwrap();
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, cancellation.clone());
    let context = StreamOperatorContext::new(&job, "asof", None)
        .with_test_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut preload = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", right, &context, &mut preload)
        .await
        .unwrap();
    operator
        .process_data("left", left, &context, &mut preload)
        .await
        .unwrap();
    let mut stopped = CancelPrefixCollector {
        cancel: cancellation,
        accepted: Vec::new(),
    };
    assert!(matches!(
        operator
            .on_watermark(EventTime::from_micros(103), &context, &mut stopped)
            .await,
        Err(CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(operator.status.pending_left_rows, 2);
    assert_eq!(operator.next_output_sequence, 1);
    assert_eq!(
        operator.runtime.pool.reserved(),
        operator.state.right.auxiliary_bytes()
    );
    let snapshot = operator.capture(Epoch::INITIAL).unwrap();
    let (mut restored, _, _) = prefix_fixture();
    restored.set_output_projection(Vec::new()).unwrap();
    restored.restore(&snapshot).unwrap();
    let job = StreamJobContext::new(2, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "asof", None)
        .with_test_output_budget(EdgeBudget::new(1, 1 << 20).unwrap());
    let mut output = EdgeCollector::new(restored.output_ports().to_vec());
    restored.on_end(&context, &mut output).await.unwrap();
    stopped.accepted.extend(
        output
            .drain("output")
            .into_iter()
            .map(|message| message.as_data().unwrap().clone()),
    );
    assert_eq!(stopped.accepted.len(), 3);
    for (index, batch) in stopped.accepted.iter().enumerate() {
        assert_eq!(batch.metadata().sequence(), index as u64);
        let record = &batch.table_payload().unwrap().batches()[0];
        assert_eq!((record.num_columns(), record.num_rows()), (0, 1));
    }
    assert_eq!(restored.status.emitted_left_rows, 3);
    assert_eq!(restored.status.matched_rows, 3);
    assert_eq!(restored.status.state_bytes, 0);
    assert_eq!(restored.runtime.pool.reserved(), 0);
}
