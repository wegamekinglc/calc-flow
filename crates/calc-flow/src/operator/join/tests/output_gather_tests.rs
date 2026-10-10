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
        assert_eq!(join_work().output_source_lookups, 0);
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
async fn test_mixed_parent_output_interleaves_columns_in_pair_order() {
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
    assert_eq!(join_work().output_column_views, 0);
    assert_eq!(join_work().output_column_takes, 3);
    assert_eq!(join_work().output_column_interleaves, 3);
    assert_eq!(join_work().output_source_lookups, 3);
}

fn two_record_batch(left: bool, first: Vec<i64>, second: Vec<i64>) -> Batch {
    let batch = if left { left_batch } else { right_batch };
    let records = [batch(first), batch(second)]
        .into_iter()
        .map(|batch| batch.table_payload().unwrap().batches()[0].clone())
        .collect();
    Batch::table(records, BatchMetadata::default()).unwrap()
}

#[tokio::test]
async fn test_multiple_input_and_retained_chunks_interleave_in_both_directions() {
    for incoming_left in [false, true] {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        let retained_side = if incoming_left { "right" } else { "left" };
        for (first, second) in [(vec![3, 1], vec![2, 0]), (vec![7, 5], vec![6, 4])] {
            operator
                .process_data(
                    retained_side,
                    two_record_batch(!incoming_left, first, second),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
        }
        let retained = if incoming_left {
            &operator.state.right
        } else {
            &operator.state.left
        };
        let owners = retained
            .iter()
            .map(|row| row.record.funded_owner().unwrap().0)
            .collect::<BTreeSet<_>>();
        assert_eq!(owners.len(), 4);
        reset_join_work();
        operator
            .process_data(
                if incoming_left { "left" } else { "right" },
                two_record_batch(incoming_left, vec![2, 0], vec![3, 1]),
                &context,
                &mut collector,
            )
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
        let incoming_times = [
            2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1,
            1, 1, 1,
        ];
        let retained_times = [
            0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4,
            5, 6, 7,
        ];
        assert_eq!(output.num_rows(), 32);
        assert_eq!(
            times(output, 1),
            if incoming_left {
                incoming_times
            } else {
                retained_times
            }
        );
        assert_eq!(
            times(output, 4),
            if incoming_left {
                retained_times
            } else {
                incoming_times
            }
        );
        assert_eq!(amounts(output), [Some(42); 32]);
        assert_eq!(
            output
                .column(5)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>(),
            [Some("paid"); 32]
        );
        assert_eq!(
            output.schema_ref(),
            operator.output_ports[0].schema().unwrap()
        );
        assert_eq!(join_work().output_column_views, 0);
        assert_eq!(join_work().output_column_takes, 0);
        assert_eq!(join_work().output_column_interleaves, 6);
        assert_eq!(
            join_work().output_source_lookups,
            32,
            "one incoming source switch and 31 alternating retained-source switches"
        );
    }
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
    let selections = offsets.iter().map(|&row| (0, row)).collect::<Vec<_>>();
    materialize_parent_columns(vec![column], &selections)
}

fn materialize_parent_columns(
    columns: Vec<ArrayRef>,
    selections: &[(usize, usize)],
) -> RecordBatch {
    let left_schema = Arc::new(Schema::new(vec![Field::new(
        "left",
        columns[0].data_type().clone(),
        true,
    )]));
    let parents = columns
        .into_iter()
        .map(|column| {
            Arc::new(RecordBatch::try_new(Arc::clone(&left_schema), vec![column]).unwrap())
        })
        .collect::<Vec<_>>();
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
    let admitted = selections
        .iter()
        .enumerate()
        .map(|(row_id, &(parent, row))| AdmittedRow {
            record: columnar::RowPayload::Rowed {
                parent: Arc::clone(&parents[parent]),
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
    let matched = (0..selections.len())
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
    for column in flat_nullable_columns() {
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

fn flat_nullable_columns() -> Vec<ArrayRef> {
    vec![
        Arc::new(NullArray::new(3)),
        Arc::new(Int64Array::from(vec![Some(9), None, Some(-7)])),
        Arc::new(
            TimestampMicrosecondArray::from(vec![Some(3), None, Some(1)]).with_timezone("UTC"),
        ),
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
    ]
}

#[test]
fn test_multi_parent_flat_nullable_gather_matches_concat_with_one_interleave() {
    let alternating = vec![(1, 2), (0, 0), (1, 1), (0, 2), (1, 0)];
    let runs = std::iter::repeat_n((0, 2), 32)
        .chain(std::iter::repeat_n((1, 0), 64))
        .chain(std::iter::repeat_n((0, 1), 32))
        .collect::<Vec<_>>();
    for (selections, source_lookups) in [(alternating, 4), (runs, 2)] {
        for column in flat_nullable_columns() {
            let slices = selections
                .iter()
                .map(|&(_, row)| column.slice(row, 1))
                .collect::<Vec<_>>();
            let refs = slices.iter().map(AsRef::as_ref).collect::<Vec<_>>();
            let expected = concat_output_column(&refs).unwrap();
            reset_join_work();
            let output = materialize_parent_columns(vec![Arc::clone(&column), column], &selections);
            assert_eq!(output.column(0).to_data(), expected.to_data());
            assert_eq!(join_work().output_column_takes, 0);
            assert_eq!(join_work().output_column_interleaves, 1);
            assert_eq!(join_work().output_source_lookups, source_lookups);
            assert_eq!(
                join_work().output_column_views,
                selections.len(),
                "only legacy right payloads use row views"
            );
        }
    }
}

#[test]
fn test_multi_parent_gather_discards_unselected_string_backing() {
    let unused = "unused".repeat(4_096);
    let first: ArrayRef = Arc::new(StringArray::from(vec![unused.as_str(), "first"]));
    let second: ArrayRef = Arc::new(StringArray::from(vec!["second", unused.as_str()]));
    reset_join_work();
    let output = materialize_parent_columns(vec![first, second], &[(1, 0), (0, 1), (1, 0)]);
    let strings = output
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        strings.iter().collect::<Vec<_>>(),
        [Some("second"), Some("first"), Some("second")]
    );
    assert_eq!(strings.value_data().len(), 17);
    assert!(output.column(0).get_array_memory_size() < 512);
    assert_eq!(join_work().output_column_interleaves, 1);
    assert_eq!(join_work().output_column_views, 3);
}

#[test]
fn test_multi_parent_nested_dictionary_fallback_discards_unused_values() {
    for list in [false, true] {
        reset_join_work();
        let output = materialize_parent_columns(
            vec![nested_dictionary(list), nested_dictionary(list)],
            &[(1, 0), (0, 0)],
        );
        let nested = output.column(0);
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
        assert_eq!(values.iter().collect::<Vec<_>>(), [Some("paid")]);
        assert_eq!(values.value_data().len(), 4);
        assert_eq!(join_work().output_column_interleaves, 0);
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

#[test]
#[cfg(target_pointer_width = "64")]
fn test_multi_parent_gather_offsets_above_u32_remain_lossless() {
    let offset = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
    reset_join_work();
    let output = materialize_parent_columns(
        vec![
            Arc::new(NullArray::new(offset + 1)),
            Arc::new(NullArray::new(1)),
        ],
        &[(0, offset), (1, 0)],
    );
    assert_eq!(output.num_rows(), 2);
    assert_eq!(output.column(0).logical_null_count(), 2);
    assert_eq!(join_work().output_column_interleaves, 1);
    assert_eq!(join_work().output_column_views, 2);
}
