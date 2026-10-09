use super::*;
use datafusion::arrow::{array::MapArray, buffer::OffsetBuffer};

// Frozen v1 logical fees add Map(1 + 4), entry Struct(1), and key "a"(1 + 4 + 1).
const MAP_CHARGES: [u64; 2] = [232, 235];

fn wrap_map(record: &RecordBatch) -> RecordBatch {
    let dictionary = record
        .column(2)
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .unwrap();
    let values = Arc::clone(dictionary.values());
    let count = values.len();
    let entries = StructArray::new(
        vec![
            Arc::new(Field::new("key", DataType::Utf8, false)),
            Arc::new(Field::new("value", values.data_type().clone(), false)),
        ]
        .into(),
        vec![Arc::new(StringArray::from(vec!["a"; count])), values],
        None,
    );
    let offsets = OffsetBuffer::new(ScalarBuffer::from(
        (0..=count)
            .map(|offset| i32::try_from(offset).unwrap())
            .collect::<Vec<_>>(),
    ));
    let maps = MapArray::try_new(
        Arc::new(Field::new("entries", entries.data_type().clone(), false)),
        offsets,
        entries,
        None,
        false,
    )
    .unwrap();
    let dictionary = Arc::new(
        DictionaryArray::<Int32Type>::try_new(dictionary.keys().clone(), Arc::new(maps)).unwrap(),
    );
    let original = record.schema();
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            original.field(0).clone(),
            original.field(1).clone(),
            tagged_field("payload", dictionary.data_type().clone())
                .as_ref()
                .clone(),
        ],
        original.metadata().clone(),
    ));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::clone(record.column(0)),
            Arc::clone(record.column(1)),
            dictionary,
        ],
    )
    .unwrap()
}

fn selected_map_values(record: &RecordBatch) -> (&StructArray, usize) {
    let dictionary = record
        .column(2)
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .unwrap();
    let maps = dictionary
        .values()
        .as_any()
        .downcast_ref::<MapArray>()
        .unwrap();
    let index = usize::try_from(dictionary.keys().value(0)).unwrap();
    let start = usize::try_from(maps.value_offsets()[index]).unwrap();
    assert_eq!(
        maps.value_offsets()[index + 1] - maps.value_offsets()[index],
        1
    );
    assert_eq!(
        maps.keys()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(start),
        "a"
    );
    (
        maps.values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap(),
        start,
    )
}

fn assert_map_rows(operator: &StreamJoinOperator) {
    assert_eq!(
        (operator.state.left.len(), operator.state.right.len()),
        (2, 0)
    );
    assert_eq!(operator.status().left.retained_bytes, 467);
    for (index, (time, charge, label, tag, bits)) in [
        (95, 232, "payload-00000000", "tag00", 0x8000_0000_0000_0000),
        (96, 235, "payload-00000010", "tag10", 0x7ff8_0000_0000_0042),
    ]
    .into_iter()
    .enumerate()
    {
        let row = &operator.state.left[index];
        assert_eq!(
            (row.row_id, row.event_time.as_micros(), row.charge),
            (u64::try_from(index).unwrap(), time, charge)
        );
        assert_eq!(row.encoded_key.as_slice(), KEY);
        let record = row.record.view();
        let (values, selected) = selected_map_values(&record);
        assert_eq!(
            values
                .column(0)
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap()
                .value(selected),
            label
        );
        assert_dictionary_value(values.column(1), selected, tag);
        assert_spans(values.column(2), selected);
        assert_reading(values.column(3), selected);
        assert_eq!(
            values
                .column(6)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .value(selected)
                .to_bits(),
            bits
        );
        assert_eq!(&record.schema(), operator.input_schema(0));
    }
}

fn assert_bad_map_schema_is_atomic(
    operator: &mut StreamJoinOperator,
    initial: &RecordBatch,
    final_batch: &RecordBatch,
) {
    let schema = Arc::new(Schema::new_with_metadata(
        final_batch.schema().fields().clone(),
        HashMap::from([("fixture".into(), "corrupted-map-metadata".into())]),
    ));
    let initial = RecordBatch::try_new(Arc::clone(&schema), initial.columns().to_vec()).unwrap();
    let final_batch = RecordBatch::try_new(schema, final_batch.columns().to_vec()).unwrap();
    let invalid = composite_snapshot_with_charges(
        operator,
        &delta_ipc_with_charges(&initial, &final_batch, MAP_CHARGES),
        MAP_CHARGES,
    );
    let state = Arc::clone(&operator.state.left.0);
    let containers = Arc::clone(operator.v2_containers.as_ref().unwrap());
    let metrics = operator.status();
    let pool = operator.checkpoint_preload_test_pool().unwrap();
    let paid = pool.reserved();
    let error = operator.restore(&invalid).unwrap_err();
    assert!(
        matches!(error, CalcFlowError::CheckpointMismatch { .. }),
        "{error}"
    );
    assert!(error.to_string().contains("schema"), "{error}");
    assert!(Arc::ptr_eq(&operator.state.left.0, &state));
    assert!(Arc::ptr_eq(
        operator.v2_containers.as_ref().unwrap(),
        &containers
    ));
    assert_eq!(operator.status(), metrics);
    assert_eq!(pool.reserved(), paid);
    assert_map_rows(operator);
}

#[test]
fn test_v2_reader_behavior_restores_map_dictionary_deltas() {
    let (initial, final_batch) = dictionary_batches();
    let initial = wrap_map(&initial);
    let final_batch = wrap_map(&final_batch);
    let mut operator = StreamJoinOperator::new(
        "v2-match",
        final_batch.schema(),
        final_batch.schema(),
        operator().spec.clone(),
    )
    .unwrap();
    for (row, charge) in MAP_CHARGES.into_iter().enumerate() {
        assert_eq!(
            state_row_charge(&final_batch, row, &[0], "v2-match").unwrap(),
            charge
        );
    }
    let snapshot = composite_snapshot_with_charges(
        &operator,
        &delta_ipc_with_charges(&initial, &final_batch, MAP_CHARGES),
        MAP_CHARGES,
    );
    let original = snapshot.clone();
    operator.restore(&snapshot).unwrap();
    assert_map_rows(&operator);
    assert_bad_map_schema_is_atomic(&mut operator, &initial, &final_batch);
    assert_eq!(snapshot.inline_metadata, original.inline_metadata);
    assert_eq!(snapshot.segments, original.segments);
    let record = operator.state.left[1].record.view();
    let (values, _) = selected_map_values(&record);
    let buffer = values
        .column(0)
        .as_any()
        .downcast_ref::<StringViewArray>()
        .unwrap()
        .data_buffers()
        .last()
        .unwrap()
        .clone();
    drop(record);
    let pool = operator.checkpoint_preload_test_pool().unwrap();
    let containers = Arc::downgrade(operator.v2_containers.as_ref().unwrap());
    drop(operator);
    assert!(containers.upgrade().is_none());
    assert!(
        pool.reserved() > 0,
        "the final nested Map value Buffer retains backing credit"
    );
    assert!(
        buffer
            .as_slice()
            .windows(16)
            .any(|bytes| bytes == b"payload-00000010")
    );
    drop(buffer);
    assert_eq!(pool.reserved(), 0);
}
