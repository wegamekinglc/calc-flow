use super::*;

fn keyed_record(left: bool, keys: Vec<i64>, times: Vec<i64>) -> RecordBatch {
    let source = if left {
        left_batch(times)
    } else {
        right_batch(times)
    };
    let record = &source.table_payload().unwrap().batches()[0];
    let mut columns = record.columns().to_vec();
    columns[0] = Arc::new(Int64Array::from(keys));
    RecordBatch::try_new(record.schema(), columns).unwrap()
}

#[tokio::test]
async fn test_persistent_dictionary_reuses_cross_batch_key_frames() {
    Box::pin(assert_cross_batch_keys(true)).await;
}

#[tokio::test]
async fn test_persistent_dictionary_shares_cross_side_canonical_owner() {
    Box::pin(assert_cross_batch_keys(false)).await;
}

async fn assert_cross_batch_keys(check_frames: bool) {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let right = keyed_record(false, vec![7, 7, 8], vec![3, 1, 2]);
    let left = vec![
        keyed_record(true, vec![7, 7], vec![2, 0]),
        keyed_record(true, vec![8], vec![1]),
    ];
    let fees = [(0, 0), (0, 1), (1, 0)].map(|(record, physical)| {
        state_row_charge(&left[record], physical, &[0], "match").unwrap()
    });
    reset_join_work();
    operator
        .process_data(
            "right",
            Batch::table(vec![right.clone()], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    assert_eq!(join_work().key_encodings, 2);
    let canonical = Arc::clone(&operator.state.right[0].encoded_key);
    operator
        .process_data(
            "left",
            Batch::table(left.clone(), BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    assert_cross_batch_output(&mut output);
    for (stored, fee) in operator.state.left.iter().zip(fees) {
        assert_eq!(stored.charge, fee);
    }
    assert_eq!(
        (
            operator.state.next_left_row_id,
            operator.state.next_right_row_id
        ),
        (3, 3)
    );
    assert_eq!(operator.status().emitted_match_rows, 5);
    if check_frames {
        assert_eq!(
            join_work().key_encodings,
            2,
            "known keys on the opposite side must reuse their persistent canonical frames"
        );
    } else {
        assert!(
            Arc::ptr_eq(&canonical, &operator.state.left[0].encoded_key),
            "equal retained keys on both sides must alias one persistent canonical owner"
        );
    }
}

fn assert_cross_batch_output(output: &mut EdgeCollector) {
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    let record = &messages[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!(record.num_rows(), 5);
    let times = |index| {
        record
            .column(index)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .values()
            .to_vec()
    };
    assert_eq!(times(1), [2, 2, 0, 0, 1]);
    assert_eq!(times(4), [1, 3, 1, 3, 2]);
    assert_eq!(
        record.column(2).to_data(),
        Int64Array::from(vec![42; 5]).to_data()
    );
    assert_eq!(
        record.column(5).to_data(),
        StringArray::from(vec!["paid"; 5]).to_data()
    );
}
