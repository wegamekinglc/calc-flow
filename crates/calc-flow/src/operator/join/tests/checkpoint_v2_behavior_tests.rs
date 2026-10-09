use super::*;
use datafusion::arrow::{
    array::{
        BooleanArray, FixedSizeListArray, Float64Array, ListViewArray, NullArray, RunArray,
        StringViewArray, StructArray, UnionArray,
    },
    buffer::{Buffer, ScalarBuffer},
    datatypes::{UnionFields, UnionMode},
    ipc::{
        self,
        writer::{
            CompressionContext, DictionaryHandling, DictionaryTracker, IpcDataGenerator,
            write_message,
        },
    },
};
use std::collections::HashMap;

const CHARGES: [u64; 2] = [220, 223];

#[path = "checkpoint_v2_map_behavior_tests.rs"]
mod map_tests;

fn tagged_field(name: &str, data_type: DataType) -> Arc<Field> {
    let nullable = data_type == DataType::Null;
    Arc::new(
        Field::new(name, data_type, nullable).with_metadata(HashMap::from([(
            "source".into(),
            "nested-delta-fixture".into(),
        )])),
    )
}

fn tags(count: usize) -> ArrayRef {
    let values = StringArray::from_iter_values((0..count).map(|index| format!("tag{index:02}")));
    Arc::new(
        DictionaryArray::<Int32Type>::try_new(
            Int32Array::from_iter_values((0..count).map(|index| i32::try_from(index).unwrap())),
            Arc::new(values),
        )
        .unwrap(),
    )
}

fn labels() -> ArrayRef {
    let mut builder =
        datafusion::arrow::array::StringViewBuilder::with_capacity(11).with_fixed_block_size(16);
    for index in 0..11 {
        builder.append_value(format!("payload-{index:08}"));
    }
    let result = builder.finish();
    assert_eq!(result.data_buffers().len(), 11);
    Arc::new(result)
}

fn spans() -> ArrayRef {
    let values = FixedSizeListArray::new(
        Arc::new(Field::new("number", DataType::Int32, false)),
        2,
        Arc::new(Int32Array::from(vec![10, 20, 30, 40])),
        None,
    );
    Arc::new(
        ListViewArray::try_new(
            Arc::new(Field::new("pair", values.data_type().clone(), false)),
            ScalarBuffer::from(vec![0_i32; 11]),
            ScalarBuffer::from(vec![2_i32; 11]),
            Arc::new(values),
            None,
        )
        .unwrap(),
    )
}

fn readings() -> ArrayRef {
    let fields = UnionFields::try_new(
        [0, 1],
        [
            Field::new("integer", DataType::Int64, false),
            Field::new("flag", DataType::Boolean, false),
        ],
    )
    .unwrap();
    let mut ids = vec![0_i8; 11];
    ids[5] = 1;
    let result = UnionArray::try_new(
        fields,
        ScalarBuffer::from(ids),
        Some(ScalarBuffer::from(vec![0_i32; 11])),
        vec![
            Arc::new(Int64Array::from(vec![42])),
            Arc::new(BooleanArray::from(vec![true])),
        ],
    )
    .unwrap();
    assert!(matches!(
        result.data_type(),
        DataType::Union(_, UnionMode::Dense)
    ));
    Arc::new(result)
}

fn dictionary_values() -> StructArray {
    let repeated = RunArray::<Int32Type>::try_new(
        &Int32Array::from(vec![10, 11]),
        &StringArray::from(vec!["r", "last"]),
    )
    .unwrap();
    let mut prices = vec![-0.0_f64; 11];
    prices[10] = f64::from_bits(0x7ff8_0000_0000_0042);
    let columns: Vec<ArrayRef> = vec![
        labels(),
        tags(11),
        spans(),
        readings(),
        Arc::new(repeated),
        Arc::new(NullArray::new(11)),
        Arc::new(Float64Array::from(prices)),
    ];
    let fields = [
        "label", "tag", "spans", "reading", "repeat", "empty", "price",
    ]
    .into_iter()
    .zip(&columns)
    .map(|(name, array)| tagged_field(name, array.data_type().clone()))
    .collect::<Vec<_>>();
    StructArray::new(fields.into(), columns, None)
}

fn composite_schema(values: &StructArray) -> Arc<Schema> {
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("key", DataType::Int64, false),
            Field::new(
                "at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            tagged_field(
                "payload",
                DataType::Dictionary(
                    Box::new(DataType::Int32),
                    Box::new(values.data_type().clone()),
                ),
            )
            .as_ref()
            .clone(),
        ],
        HashMap::from([("fixture".into(), "dictionary-delta".into())]),
    ))
}

fn composite_record(schema: Arc<Schema>, values: StructArray, last: i32) -> RecordBatch {
    let dictionary =
        DictionaryArray::<Int32Type>::try_new(Int32Array::from(vec![0, last]), Arc::new(values))
            .unwrap();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![7, 7])),
            Arc::new(TimestampMicrosecondArray::from(vec![95, 96]).with_timezone("UTC")),
            Arc::new(dictionary),
        ],
    )
    .unwrap()
}

fn dictionary_batches() -> (RecordBatch, RecordBatch) {
    let values = dictionary_values();
    let schema = composite_schema(&values);
    let mut initial_columns = values
        .columns()
        .iter()
        .map(|array| array.slice(0, 10))
        .collect::<Vec<_>>();
    initial_columns[1] = tags(10);
    let initial = StructArray::new(values.fields().clone(), initial_columns, None);
    (
        composite_record(Arc::clone(&schema), initial, 9),
        composite_record(schema, values, 10),
    )
}

fn delta_ipc(initial: &RecordBatch, final_batch: &RecordBatch) -> Vec<u8> {
    delta_ipc_with_charges(initial, final_batch, CHARGES)
}

fn delta_ipc_with_charges(
    initial: &RecordBatch,
    final_batch: &RecordBatch,
    charges: [u64; 2],
) -> Vec<u8> {
    let options = IpcWriteOptions::try_new(8, false, MetadataVersion::V5)
        .unwrap()
        .with_dictionary_handling(DictionaryHandling::Delta);
    let generator = IpcDataGenerator {};
    let mut tracker = DictionaryTracker::new(false);
    let mut compression = CompressionContext::default();
    let schema = generator.schema_to_bytes_with_dictionary_tracker(
        &initial.schema(),
        &mut tracker,
        &options,
    );
    let (base_dictionaries, _) = generator
        .encode(initial, &mut tracker, &options, &mut compression)
        .unwrap();
    let (delta_dictionaries, record) = generator
        .encode(final_batch, &mut tracker, &options, &mut compression)
        .unwrap();
    assert!(base_dictionaries.len() >= 2);
    assert_eq!(delta_dictionaries.len(), 2);
    for dictionary in &delta_dictionaries {
        assert!(
            ipc::root_as_message(&dictionary.ipc_message)
                .unwrap()
                .header_as_dictionary_batch()
                .unwrap()
                .isDelta()
        );
    }
    let mut bytes = Vec::new();
    write_message(&mut bytes, schema, &options).unwrap();
    for dictionary in base_dictionaries.into_iter().chain(delta_dictionaries) {
        write_message(&mut bytes, dictionary, &options).unwrap();
    }
    write_message(&mut bytes, record, &options).unwrap();
    bytes.extend_from_slice(&[0xff; 4]);
    bytes.extend_from_slice(&0_i32.to_le_bytes());
    let mut reader = StreamReader::try_new(Cursor::new(&bytes), None).unwrap();
    let decoded = reader.next().unwrap().unwrap();
    assert_eq!(decoded.num_rows(), 2);
    for (row, charge) in charges.into_iter().enumerate() {
        let actual = state_row_charge(&decoded, row, &[0], "v2-match").unwrap();
        assert_eq!(actual, charge, "upstream decoded row {row}: {decoded:?}");
    }
    assert!(
        reader.next().is_none(),
        "dictionary deltas precede exactly one final record batch"
    );
    bytes
}

fn composite_snapshot(operator: &StreamJoinOperator, ipc: &[u8]) -> OperatorStateSnapshot {
    composite_snapshot_with_charges(operator, ipc, CHARGES)
}

fn composite_snapshot_with_charges(
    operator: &StreamJoinOperator,
    ipc: &[u8],
    charges: [u64; 2],
) -> OperatorStateSnapshot {
    let mut bytes = header(*b"CFJPAY2\0", 0);
    bytes.extend_from_slice(&2_u64.to_le_bytes());
    bytes.extend_from_slice(&u64::try_from(ipc.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(ipc);
    let payload = Payload {
        side: 0,
        ids: vec![0, 1],
        digest: Sha256::digest(&bytes).into(),
        segment: StateSegment::new(bytes),
    };
    let sha256 = hex::encode(payload.digest);
    let mut right = header(*b"CFJIDX2\0", 1);
    right.extend_from_slice(&0_u64.to_le_bytes());
    right.extend_from_slice(&0_u64.to_le_bytes());
    let metadata = JoinCheckpointMetadata {
        layout_version: 2,
        spec: operator.spec.clone(),
        next_left_row_id: 2,
        next_right_row_id: 0,
        next_output_sequence: 0,
        ended: false,
        epoch: 1,
        metrics: JoinMetrics {
            left: SideMetrics {
                retained_rows: 2,
                retained_bytes: charges.into_iter().sum(),
                ..SideMetrics::default()
            },
            ..JoinMetrics::default()
        },
    };
    let Value::Object(mut metadata) = serde_json::to_value(metadata).unwrap() else {
        panic!("metadata object")
    };
    metadata.insert("v2_inventory".into(), serde_json::json!({
        "codec_version": 2, "base_epoch": 1, "deltas": [],
        "payloads": [{"side": "left", "sha256": sha256, "rows": 2, "bytes": payload.segment.bytes().len()}],
    }));
    OperatorStateSnapshot {
        inline_metadata: metadata.into_iter().collect(),
        segments: BTreeMap::from([
            ("left-base".into(), base(&payload, &[95, 96], &charges)),
            ("right-base".into(), StateSegment::new(right)),
            (format!("left-payload-{sha256}"), payload.segment),
        ]),
    }
}

fn assert_dictionary_value(column: &ArrayRef, row: usize, expected: &str) {
    let array = column
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .unwrap();
    let values = array
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        values.value(usize::try_from(array.keys().value(row)).unwrap()),
        expected
    );
}

fn assert_spans(column: &ArrayRef, row: usize) {
    let array = column.as_any().downcast_ref::<ListViewArray>().unwrap();
    let selected = array.value(row);
    let pairs = selected
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    assert_eq!(pairs.len(), 2);
    for (index, expected) in [[10, 20], [30, 40]].iter().enumerate() {
        let value = pairs.value(index);
        assert_eq!(
            value
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .as_ref(),
            expected
        );
    }
}

fn assert_reading(column: &ArrayRef, row: usize) {
    let array = column.as_any().downcast_ref::<UnionArray>().unwrap();
    assert_eq!(array.type_id(row), 0);
    let child = array.child(0);
    let offset = array.value_offset(row);
    assert_eq!(
        child
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(offset),
        42
    );
    let flags = array
        .child(1)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    assert_eq!(flags.len(), 1);
    assert!(flags.value(0));
}

fn assert_composite_rows(operator: &StreamJoinOperator) {
    assert_eq!(
        (operator.state.left.len(), operator.state.right.len()),
        (2, 0)
    );
    assert_eq!(
        (
            operator.state.next_left_row_id,
            operator.state.next_right_row_id,
            operator.state.next_output_sequence
        ),
        (2, 0, 0)
    );
    assert_eq!(operator.state.last_checkpoint_epoch, Epoch::new(1));
    assert_eq!(operator.status().left.retained_bytes, 443);
    for (index, (time, charge, label, tag, repeated, bits)) in [
        (
            95,
            220,
            "payload-00000000",
            "tag00",
            "r",
            0x8000_0000_0000_0000,
        ),
        (
            96,
            223,
            "payload-00000010",
            "tag10",
            "last",
            0x7ff8_0000_0000_0042,
        ),
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
        let dictionary = record
            .column(2)
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .unwrap();
        let values = dictionary
            .values()
            .as_any()
            .downcast_ref::<StructArray>()
            .unwrap();
        let selected = usize::try_from(dictionary.keys().value(0)).unwrap();
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
        let repeated_values = values
            .column(4)
            .as_any()
            .downcast_ref::<RunArray<Int32Type>>()
            .unwrap();
        assert_eq!(
            repeated_values
                .values()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(repeated_values.get_physical_index(selected)),
            repeated
        );
        let nulls = values
            .column(5)
            .as_any()
            .downcast_ref::<NullArray>()
            .unwrap();
        assert_eq!((nulls.len(), nulls.logical_null_count()), (11, 11));
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

fn assert_schema_rejection(
    operator: &mut StreamJoinOperator,
    initial: &RecordBatch,
    final_batch: &RecordBatch,
) {
    let schema = Arc::new(Schema::new_with_metadata(
        final_batch.schema().fields().clone(),
        HashMap::from([("fixture".into(), "corrupted-metadata".into())]),
    ));
    let initial = RecordBatch::try_new(Arc::clone(&schema), initial.columns().to_vec()).unwrap();
    let final_batch = RecordBatch::try_new(schema, final_batch.columns().to_vec()).unwrap();
    let invalid = composite_snapshot(operator, &delta_ipc(&initial, &final_batch));
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
    assert_composite_rows(operator);
}

fn last_value_buffer(operator: &StreamJoinOperator) -> Buffer {
    let record = operator.state.left[1].record.view();
    let dictionary = record
        .column(2)
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .unwrap();
    let values = dictionary
        .values()
        .as_any()
        .downcast_ref::<StructArray>()
        .unwrap();
    values
        .column(0)
        .as_any()
        .downcast_ref::<StringViewArray>()
        .unwrap()
        .data_buffers()
        .last()
        .unwrap()
        .clone()
}

#[test]
fn test_v2_reader_behavior_restores_dictionary_deltas_and_nested_values() {
    let (initial, final_batch) = dictionary_batches();
    let mut operator = StreamJoinOperator::new(
        "v2-match",
        final_batch.schema(),
        final_batch.schema(),
        operator().spec.clone(),
    )
    .unwrap();
    for (row, charge) in CHARGES.into_iter().enumerate() {
        assert_eq!(
            state_row_charge(&final_batch, row, &[0], "v2-match").unwrap(),
            charge
        );
    }
    let snapshot = composite_snapshot(&operator, &delta_ipc(&initial, &final_batch));
    let original = snapshot.clone();
    operator.restore(&snapshot).unwrap();
    assert_composite_rows(&operator);
    assert_schema_rejection(&mut operator, &initial, &final_batch);
    assert_eq!(snapshot.inline_metadata, original.inline_metadata);
    assert_eq!(snapshot.segments, original.segments);
    let pool = operator.checkpoint_preload_test_pool().unwrap();
    let containers = Arc::downgrade(operator.v2_containers.as_ref().unwrap());
    let buffer = last_value_buffer(&operator);
    drop(operator);
    assert!(containers.upgrade().is_none());
    assert!(
        pool.reserved() > 0,
        "the actual final nested Buffer retains its backing credit"
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
