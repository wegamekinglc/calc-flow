use std::sync::Arc;

use datafusion::arrow::{
    array::{
        Array, ArrayRef, BinaryArray, Int32Array, ListArray, StringViewArray, UInt8Array,
        UInt64Array, builder::StringDictionaryBuilder,
    },
    datatypes::{DataType, Field, Schema},
    datatypes::{Int8Type, Int32Type},
    ipc::{
        self,
        writer::{FileWriter, IpcWriteOptions},
    },
    record_batch::RecordBatch,
};

use super::preflight;
use crate::CalcFlowError;

fn fixture() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("_state_kind", DataType::UInt8, false),
        Field::new("_entity_id", DataType::UInt64, false),
        Field::new("_entity_position", DataType::UInt64, true),
        Field::new("_indicator_state", DataType::Binary, true),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(UInt8Array::from(vec![0, 2])),
        Arc::new(UInt64Array::from(vec![0, 0])),
        Arc::new(UInt64Array::from(vec![Some(0), None])),
        Arc::new(BinaryArray::from(vec![None, None])),
    ];
    RecordBatch::try_new(schema, columns).unwrap()
}

fn encode(record: &RecordBatch, options: IpcWriteOptions) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut writer =
        FileWriter::try_new_with_options(&mut bytes, &record.schema(), options).unwrap();
    writer.write(record).unwrap();
    writer.finish().unwrap();
    drop(writer);
    bytes
}

fn assert_invalid(bytes: &[u8], expected: &Schema, reason: &str) {
    let result = preflight(bytes, expected, 100, &mut || Ok(()));
    let Err(CalcFlowError::Format { message }) = result else {
        panic!("invalid IPC must be rejected before Arrow arrays are allocated");
    };
    assert!(message.contains(reason), "{message}");
}

#[test]
fn test_canonical_preflight_borrows_input_without_allocating() {
    let record = fixture();
    let bytes = encode(&record, IpcWriteOptions::default());
    let mut accepted = false;
    let measured = allocation_counter::measure(|| {
        let plan = preflight(&bytes, &record.schema(), 100, &mut || Ok(())).unwrap();
        assert_eq!(plan.record_rows, 2);
        assert_eq!(plan.entity_rows, 1);
        assert_eq!(plan.pending_rows, 1);
        assert!(plan.record_body.length > 0);
        assert!(plan.record_block.length > plan.record_body.length);
        assert!(plan.metadata_bytes > plan.footer_bytes);
        assert_eq!(plan.field_count, record.num_columns());
        assert_eq!(plan.node_count, record.num_columns());
        assert_eq!(plan.buffer_count, 9);
        assert_eq!(plan.variadic_count, 0);
        assert_eq!(plan.dictionary_count, 0);
        assert_eq!(plan.dictionary_body_bytes, 0);
        assert_eq!(plan.dictionary_block_bytes, 0);
        assert_eq!(plan.metadata_pairs, 0);
        assert_eq!(plan.metadata_string_bytes, 0);
        assert_eq!(
            plan.field_name_bytes,
            record
                .schema()
                .fields()
                .iter()
                .map(|field| field.name().len())
                .sum::<usize>()
        );
        assert_eq!(plan.timezone_string_bytes, 0);
        assert_eq!(plan.union_type_ids, 0);
        assert_eq!(plan.history_rows, 0);
        assert_eq!(plan.summary_rows, 0);
        assert_eq!(plan.kinds.bytes(&bytes).unwrap(), [0, 2]);
        assert_eq!(plan.entity_ids.length, 16);
        assert_eq!(plan.positions.length, 16);
        assert_eq!(plan.position_validity.bytes(&bytes).unwrap()[0] & 3, 1);
        accepted = true;
    });
    assert!(accepted);
    assert_eq!(
        measured.count_total, 0,
        "valid preflight must borrow metadata and buffers"
    );
}

#[test]
fn test_preflight_rejects_wrong_schema_and_row_limit() {
    let record = fixture();
    let bytes = encode(&record, IpcWriteOptions::default());
    assert_invalid(&bytes, &Schema::empty(), "schema");
    let result = preflight(&bytes, &record.schema(), 1, &mut || Ok(()));
    assert!(matches!(result, Err(CalcFlowError::Format { message }) if message.contains("rows")));
}

#[test]
fn test_preflight_rejects_truncated_or_noncanonical_file_envelopes() {
    let record = fixture();
    let bytes = encode(&record, IpcWriteOptions::default());
    assert_invalid(&bytes[..bytes.len() - 1], &record.schema(), "footer");
    let mut bad_header = bytes.clone();
    bad_header[0] = b'X';
    assert_invalid(&bad_header, &record.schema(), "header");
    let mut bad_length = bytes;
    let end = bad_length.len();
    bad_length[end - 10..end - 6].copy_from_slice(&i32::MAX.to_le_bytes());
    assert_invalid(&bad_length, &record.schema(), "footer");
}

#[test]
fn test_preflight_rejects_compression_before_decoding_body() {
    let record = fixture();
    let options = IpcWriteOptions::default()
        .try_with_compression(Some(ipc::CompressionType::ZSTD))
        .unwrap();
    let bytes = encode(&record, options);
    assert_invalid(&bytes, &record.schema(), "compression");
}

#[test]
fn test_preflight_preserves_cancellation() {
    let record = fixture();
    let bytes = encode(&record, IpcWriteOptions::default());
    let result = preflight(&bytes, &record.schema(), 100, &mut || {
        Err(CalcFlowError::Cancelled {
            run_id: "ipc-cancel".into(),
        })
    });
    assert!(matches!(result, Err(CalcFlowError::Cancelled { run_id }) if run_id == "ipc-cancel"));
}

fn with_extra(record: &RecordBatch, column: ArrayRef) -> RecordBatch {
    let mut fields = record.schema().fields().to_vec();
    fields.push(Arc::new(Field::new(
        "extra",
        column.data_type().clone(),
        true,
    )));
    let mut columns = record.columns().to_vec();
    columns.push(column);
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

#[test]
fn test_preflight_accepts_nested_dictionary_and_variadic_buffers() {
    let mut dictionary = StringDictionaryBuilder::<Int8Type>::new();
    dictionary.append("same").unwrap();
    dictionary.append("same").unwrap();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>([
            Some(vec![Some(1), None]),
            None,
        ])),
        Arc::new(dictionary.finish()),
        Arc::new(StringViewArray::from(vec![
            Some("a sufficiently long external string"),
            None,
        ])),
        Arc::new(Int32Array::from(vec![1, 2])),
    ];
    for array in arrays {
        let record = with_extra(&fixture(), array);
        let bytes = encode(&record, IpcWriteOptions::default());
        let plan = preflight(&bytes, &record.schema(), 100, &mut || Ok(())).unwrap();
        assert_eq!(plan.record_rows, 2);
        assert_eq!(plan.dictionary_bodies().count(), plan.dictionary_count);
        assert_eq!(
            plan.dictionary_bodies()
                .map(|body| body.length)
                .sum::<usize>(),
            plan.dictionary_body_bytes
        );
    }
}

fn metadata_offset(bytes: &[u8]) -> (usize, ipc::RecordBatch<'_>) {
    let (footer, _) = super::footer(bytes).unwrap();
    let block = footer.recordBatches().unwrap().get(0);
    let start = usize::try_from(block.offset()).unwrap() + 8;
    let length = usize::try_from(block.metaDataLength()).unwrap() - 8;
    let message = ipc::root_as_message(&bytes[start..start + length]).unwrap();
    (start, message.header_as_record_batch().unwrap())
}

fn relative(bytes: &[u8], part: &[u8]) -> usize {
    (part.as_ptr() as usize)
        .checked_sub(bytes.as_ptr() as usize)
        .unwrap()
}

#[test]
fn test_preflight_rejects_forged_rows_nodes_and_buffer_extents_before_arrays() {
    let record = fixture();
    let bytes = encode(&record, IpcWriteOptions::default());
    let (start, batch) = metadata_offset(&bytes);
    let row_field = usize::from(batch._tab.vtable().get(ipc::RecordBatch::VT_LENGTH));
    assert!(row_field > 0);
    let row_offset = start + batch._tab.loc() + row_field;
    let node_offset = relative(&bytes, batch.nodes().unwrap().bytes());
    let buffer_offset = relative(&bytes, batch.buffers().unwrap().bytes());
    for (offset, value, reason) in [
        (row_offset, i64::MAX, "rows"),
        (node_offset, i64::MAX, "node rows"),
        (buffer_offset + 8, i64::MAX, "buffer length"),
        (buffer_offset + 16, 0, "buffer length, overlap or alignment"),
    ] {
        let mut corrupted = bytes.clone();
        corrupted[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        assert_invalid(&corrupted, &record.schema(), reason);
    }
}

#[test]
fn test_preflight_rejects_variadic_count_before_reader_collects_buffers() {
    let record = with_extra(
        &fixture(),
        Arc::new(StringViewArray::from(vec![
            Some("external string data larger than inline"),
            None,
        ])),
    );
    let bytes = encode(&record, IpcWriteOptions::default());
    let (_, batch) = metadata_offset(&bytes);
    let offset = relative(&bytes, batch.variadicBufferCounts().unwrap().bytes());
    let mut corrupted = bytes;
    corrupted[offset..offset + 8].copy_from_slice(&i64::MAX.to_le_bytes());
    assert_invalid(&corrupted, &record.schema(), "variadic buffer count");
}
