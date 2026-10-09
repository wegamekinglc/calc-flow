use super::*;
use datafusion::arrow::{
    array::{
        ArrayRef, BooleanArray, LargeBinaryArray, LargeStringArray, ListArray, NullArray,
        StructArray, TimestampSecondArray,
    },
    buffer::{OffsetBuffer, ScalarBuffer},
};

fn nullable_left_record() -> RecordBatch {
    RecordBatch::try_new(
        left_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7; 3])),
            Arc::new(TimestampMicrosecondArray::from(vec![3, 1, 2]).with_timezone("UTC")),
            Arc::new(Int64Array::from(vec![None, Some(42), None])),
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn test_legacy_admission_keeps_one_parent_and_physical_row_offsets() {
    for nullable in [false, true] {
        let record = if nullable {
            nullable_left_record()
        } else {
            left_batch(vec![3, 1, 2]).table_payload().unwrap().batches()[0].clone()
        };
        let input = Batch::table(vec![record.clone()], BatchMetadata::default()).unwrap();
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        if !nullable {
            operator.set_stream_resources(
                DataFusionConfig {
                    target_partitions: 2,
                    ..DataFusionConfig::default()
                },
                UdfRegistrySnapshot::default(),
            );
        }
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None);
        let prepared = operator
            .prepare_batch("left", &input, &context)
            .await
            .unwrap();
        assert_eq!(prepared.admitted.len(), 3);
        for (offset, row) in prepared.admitted.iter().enumerate() {
            assert_eq!(row.record.offset(), offset, "physical parent row offset");
            assert!(std::ptr::eq(
                row.record.columns(),
                prepared.admitted[0].record.columns()
            ));
            assert_eq!(row.record.column(2).len(), 3);
            assert!(Arc::ptr_eq(row.record.column(2), record.column(2)));
            assert_eq!(row.record.funded_owner(), None);
            assert_eq!(
                row_ipc::RowIpcEncoder::default()
                    .encode(&row.record.view(), "match", "left")
                    .unwrap(),
                row_ipc::RowIpcEncoder::default()
                    .encode(&record.slice(offset, 1), "match", "left")
                    .unwrap()
            );
        }
        drop(prepared);
        let pool = operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop(operator);
        drop(context);
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        assert_eq!(pool.reserved(), 0);
    }
}

fn times(record: &RecordBatch, column: usize) -> Vec<i64> {
    record
        .column(column)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap()
        .values()
        .to_vec()
}

fn amounts(record: &RecordBatch) -> Vec<Option<i64>> {
    record
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .iter()
        .collect()
}

#[tokio::test]
async fn test_same_parent_output_takes_each_column_once_in_both_directions() {
    for incoming_left in [false, true] {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        let left = Batch::table(vec![nullable_left_record()], BatchMetadata::default()).unwrap();
        let (first_side, first, second_side, second) = if incoming_left {
            ("right", right_batch(vec![5, 0]), "left", left)
        } else {
            ("left", left, "right", right_batch(vec![5, 0]))
        };
        operator
            .process_data(first_side, first, &context, &mut collector)
            .await
            .unwrap();
        reset_join_work();
        operator
            .process_data(second_side, second, &context, &mut collector)
            .await
            .unwrap();
        let messages = collector.drain("output");
        assert_eq!(messages.len(), 1);
        let output = &messages[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0];
        assert_eq!(output.num_rows(), 6);
        if incoming_left {
            assert_eq!(times(output, 1), [3, 3, 1, 1, 2, 2]);
            assert_eq!(times(output, 4), [0, 5, 0, 5, 0, 5]);
            assert_eq!(
                amounts(output),
                [None, None, Some(42), Some(42), None, None]
            );
        } else {
            assert_eq!(times(output, 1), [1, 2, 3, 1, 2, 3]);
            assert_eq!(times(output, 4), [5, 5, 5, 0, 0, 0]);
            assert_eq!(
                amounts(output),
                [Some(42), None, None, Some(42), None, None]
            );
        }
        assert_eq!(
            output.schema_ref(),
            operator.output_ports[0].schema().unwrap()
        );
        assert_eq!(join_work().output_column_views, 0);
        assert_eq!(join_work().output_column_takes, 6);
    }
}

fn nested_dictionary(list: bool) -> ArrayRef {
    let unused = "unused".repeat(4_096);
    let dictionary: ArrayRef = Arc::new(DictionaryArray::<Int32Type>::new(
        Int32Array::from(vec![1]),
        Arc::new(StringArray::from(vec![unused.as_str(), "paid"])),
    ));
    let field = Arc::new(Field::new("tag", dictionary.data_type().clone(), false));
    if list {
        Arc::new(ListArray::new(
            field,
            OffsetBuffer::new(ScalarBuffer::from(vec![0, 1])),
            dictionary,
            None,
        ))
    } else {
        Arc::new(StructArray::new(vec![field].into(), vec![dictionary], None))
    }
}

#[tokio::test]
async fn test_nested_dictionary_output_discards_unused_values_under_same_budget() {
    for list in [false, true] {
        let nested = nested_dictionary(list);
        let schema = Arc::new(Schema::new(vec![
            left_schema().field(0).clone(),
            left_schema().field(1).clone(),
            Field::new("amount", nested.data_type().clone(), false),
        ]));
        let record = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![7])),
                Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
                nested,
            ],
        )
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None)
            .with_output_budget(EdgeBudget::new(1, 128).unwrap());
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "left",
                Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        operator
            .process_data("right", right_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        let messages = collector.drain("output");
        assert_eq!(messages.len(), 1);
        let batch = messages[0].as_data().unwrap();
        assert!(batch.estimated_bytes().unwrap() <= 128);
        let nested = batch.table_payload().unwrap().batches()[0].column(2);
        let dictionary = if list {
            nested
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap()
                .values()
        } else {
            nested
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap()
                .column(0)
        };
        let values = dictionary
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .unwrap()
            .values();
        let values = values.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(values.len(), 1);
        assert_eq!(values.value(0), "paid");
        assert_eq!(values.value_data().len(), 4);
    }
}

#[tokio::test]
async fn test_earlier_timestamp_error_precedes_later_row_id_overflow() {
    let schema = Arc::new(Schema::new(vec![
        left_schema().field(0).clone(),
        Field::new(
            "authorized_at",
            DataType::Timestamp(TimeUnit::Second, None),
            false,
        ),
        left_schema().field(2).clone(),
    ]));
    let record = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from(vec![7, 7])),
            Arc::new(TimestampSecondArray::from(vec![i64::MAX, 0])),
            Arc::new(Int64Array::from(vec![None, Some(42)])),
        ],
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
    operator.state.next_left_row_id = u64::MAX - 1;
    let before = operator.status();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let error = operator
        .process_data(
            "left",
            Batch::table(vec![record], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap_err();
    assert_eq!(
        reason_of(&error),
        Some(crate::StreamingFailureReason::JoinTimeConversionFailed)
    );
    assert_eq!(operator.state.next_left_row_id, u64::MAX - 1);
    assert_eq!(operator.status(), before);
    assert!(collector.drain("output").is_empty());
}

#[tokio::test]
async fn test_mixed_parent_output_keeps_concat_fallback_and_pair_order() {
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("left", left_batch(vec![3]), &context, &mut collector)
        .await
        .unwrap();
    let nullable = nullable_left_record().slice(1, 1);
    operator
        .process_data(
            "left",
            Batch::table(vec![nullable], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    reset_join_work();
    operator
        .process_data("right", right_batch(vec![4, 0]), &context, &mut collector)
        .await
        .unwrap();
    let messages = collector.drain("output");
    assert_eq!(messages.len(), 1);
    let output = &messages[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    assert_eq!(times(output, 1), [1, 3, 1, 3]);
    assert_eq!(times(output, 4), [4, 4, 0, 0]);
    assert_eq!(amounts(output), [Some(42); 4]);
    assert_eq!(join_work().output_column_views, 12);
    assert_eq!(join_work().output_column_takes, 3);
}

#[tokio::test]
async fn test_empty_and_singleton_outputs_keep_independent_selected_values() {
    for rows in [0, 1] {
        let record = nullable_left_record().slice(0, rows);
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let schema = operator.output_ports[0].schema().unwrap();
        assert_eq!(
            materialize_output_record(schema, &[], &[], &[], true, "match")
                .unwrap()
                .num_rows(),
            0
        );
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "left",
                Batch::table(vec![record.clone()], BatchMetadata::default()).unwrap(),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        reset_join_work();
        operator
            .process_data("right", right_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        let messages = collector.drain("output");
        assert_eq!(messages.len(), rows);
        assert_eq!(join_work().output_column_views, 0);
        if let Some(message) = messages.first() {
            let output = &message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()[0];
            assert_eq!(amounts(output), [None]);
            assert_eq!(times(output, 1), [3]);
            assert_eq!(join_work().output_column_takes, 6);
            let source = record
                .column(1)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            let taken = output
                .column(1)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            assert_ne!(source.values().as_ptr(), taken.values().as_ptr());
        }
    }
}

fn materialize_parent_column(column: ArrayRef, offsets: &[usize]) -> RecordBatch {
    let left_schema = Arc::new(Schema::new(vec![Field::new(
        "left",
        column.data_type().clone(),
        true,
    )]));
    let parent = Arc::new(RecordBatch::try_new(Arc::clone(&left_schema), vec![column]).unwrap());
    let right_schema = Arc::new(Schema::new(vec![Field::new(
        "right",
        DataType::Int64,
        false,
    )]));
    let right = RecordBatch::try_new(
        Arc::clone(&right_schema),
        vec![Arc::new(Int64Array::from(vec![8]))],
    )
    .unwrap();
    let admitted = offsets
        .iter()
        .enumerate()
        .map(|(row_id, &row)| AdmittedRow {
            record: columnar::RowPayload::Rowed {
                parent: Arc::clone(&parent),
                row,
            },
            event_time: EventTime::from_micros(0),
            row_id: row_id as u64,
            retain: false,
        })
        .collect::<Vec<_>>();
    let opposite = [StoredRow {
        record: right.into(),
        event_time: EventTime::from_micros(0),
        row_id: 0,
        charge: 0,
        encoded_key: Arc::new(Vec::new().into()),
    }];
    let matched = (0..offsets.len())
        .map(|pos| MatchedPair {
            pos,
            opposite_index: 0,
        })
        .collect::<Vec<_>>();
    let output_schema = Arc::new(Schema::new(vec![
        left_schema.field(0).clone(),
        right_schema.field(0).clone(),
    ]));
    materialize_output_record(
        &output_schema,
        &admitted,
        &opposite,
        &matched,
        true,
        "match",
    )
    .unwrap()
}

#[test]
fn test_flat_nullable_gather_matches_old_concat_for_each_supported_shape() {
    assert_flat_nullable_gather_matches_old_concat_for_each_supported_shape();
    #[cfg(target_pointer_width = "64")]
    assert_gather_offsets_above_u32_remain_lossless_without_large_allocation();
}

fn assert_flat_nullable_gather_matches_old_concat_for_each_supported_shape() {
    let columns: Vec<ArrayRef> = vec![
        Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])),
        Arc::new(StringArray::from(vec![Some("猫"), None, Some("x")])),
        Arc::new(LargeStringArray::from(vec![Some("猫"), None, Some("x")])),
        Arc::new(BinaryArray::from_opt_vec(vec![
            Some(b"a".as_slice()),
            None,
            Some(b"b".as_slice()),
        ])),
        Arc::new(LargeBinaryArray::from_opt_vec(vec![
            Some(b"a".as_slice()),
            None,
            Some(b"b".as_slice()),
        ])),
        Arc::new(
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                [Some([1_u8; 4]), None, Some([2_u8; 4])].into_iter(),
                4,
            )
            .unwrap(),
        ),
    ];
    for column in columns {
        let slices = [2, 0, 1, 2].map(|row| column.slice(row, 1));
        let refs = slices.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        let expected = concat_output_column(&refs).unwrap();
        reset_join_work();
        let output = materialize_parent_column(column, &[2, 0, 1, 2]);
        assert_eq!(output.column(0).to_data(), expected.to_data());
        assert_eq!(join_work().output_column_takes, 1);
        assert_eq!(join_work().output_column_views, 4);
    }
}

#[cfg(target_pointer_width = "64")]
fn assert_gather_offsets_above_u32_remain_lossless_without_large_allocation() {
    let offset = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
    reset_join_work();
    let output = materialize_parent_column(Arc::new(NullArray::new(offset + 1)), &[offset]);
    assert_eq!(output.num_rows(), 1);
    assert_eq!(output.column(0).logical_null_count(), 1);
    assert_eq!(join_work().output_column_takes, 1);
    assert_eq!(join_work().output_column_views, 1);
}
