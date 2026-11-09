use super::*;
use datafusion::arrow::{
    array::{
        Array, ArrayRef, Int16Array, Int32Array, ListViewArray, RunArray, StringViewArray,
        UnionArray,
    },
    buffer::{NullBuffer, ScalarBuffer},
    datatypes::{Field, Int16Type, Schema, UnionFields},
    ipc::{self, writer::FileWriter},
    record_batch::RecordBatch,
};
use std::sync::Arc;

fn encode(array: ArrayRef) -> Vec<u8> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        array.data_type().clone(),
        true,
    )]));
    let record = RecordBatch::try_new(schema, vec![array]).unwrap();
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, &record.schema()).unwrap();
    writer.write(&record).unwrap();
    writer.finish().unwrap();
    drop(writer);
    bytes
}
fn parts(bytes: &[u8]) -> (ipc::RecordBatch<'_>, Span) {
    let (footer, _) = super::super::footer(bytes).unwrap();
    let block = footer.recordBatches().unwrap().get(0);
    let start = usize::try_from(block.offset()).unwrap();
    let metadata = usize::try_from(block.metaDataLength()).unwrap();
    let message = ipc::root_as_message(&bytes[start + 8..start + metadata]).unwrap();
    assert_eq!(message.version(), ipc::MetadataVersion::V5);
    (
        message.header_as_record_batch().unwrap(),
        Span {
            offset: start + metadata,
            length: usize::try_from(block.bodyLength()).unwrap(),
        },
    )
}
fn validate(bytes: &[u8], dtype: &DataType) -> Result<Counts> {
    let (record, body) = parts(bytes);
    dictionary(bytes, record, body, dtype, &mut || Ok(()))
}
fn buffer_start(bytes: &[u8], index: usize) -> usize {
    let (record, body) = parts(bytes);
    body.offset + usize::try_from(record.buffers().unwrap().get(index).offset()).unwrap()
}
#[test]
fn test_offset_and_word_helpers_keep_checked_bounds_sign_and_width() {
    for width in [2, 4, 8] {
        let one = 1_i64.to_le_bytes();
        assert_eq!(offset(&one[..width], 0, width).unwrap(), 1);
        assert!(offset(&[255; 8][..width], 0, width).is_err());
        assert!(offset(&one[..width], 1, width).is_err());
    }
    assert!(offset(&[0; 8], 0, 3).is_err());
    assert!(offset(&[0; 8], usize::MAX, 8).is_err());
    assert!(word(&[0; 4], usize::MAX).is_err());
    assert_eq!(word(&[1, 0, 0, 0], 0).unwrap(), 1);
}
#[test]
fn test_all_valid_info_avoids_reading_absent_validity_bytes() {
    let valid = ArrayInfo {
        length: 1,
        nulls: 0,
        validity: Span {
            offset: usize::MAX,
            length: 1,
        },
        ..Default::default()
    };
    assert!(valid.valid(&[], 0).unwrap());
    let missing = ArrayInfo { nulls: 1, ..valid };
    assert!(missing.valid(&[], 0).is_err());
}
#[test]
fn test_masked_byte_view_values_are_skipped_but_valid_values_are_checked() {
    let array = StringViewArray::from(vec![Some("a long external string value"), None]);
    let dtype = array.data_type().clone();
    let mut bytes = encode(Arc::new(array));
    let start = buffer_start(&bytes, 1);
    bytes[start + 16..start + 32].fill(255);
    validate(&bytes, &dtype).unwrap();
    bytes[start..start + 16].fill(255);
    assert!(validate(&bytes, &dtype).is_err());
}
#[test]
fn test_masked_list_view_offsets_and_sizes_are_not_interpreted() {
    let array = ListViewArray::try_new(
        Arc::new(Field::new("item", DataType::Int32, true)),
        ScalarBuffer::from(vec![0, 0]),
        ScalarBuffer::from(vec![1, 0]),
        Arc::new(Int32Array::from(vec![42])),
        Some(NullBuffer::from(vec![true, false])),
    )
    .unwrap();
    let dtype = array.data_type().clone();
    let mut bytes = encode(Arc::new(array));
    let offsets = buffer_start(&bytes, 1);
    let sizes = buffer_start(&bytes, 2);
    bytes[offsets + 4..offsets + 8].copy_from_slice(&i32::MIN.to_le_bytes());
    bytes[sizes + 4..sizes + 8].copy_from_slice(&i32::MIN.to_le_bytes());
    validate(&bytes, &dtype).unwrap();
    bytes[offsets..offsets + 4].copy_from_slice(&i32::MIN.to_le_bytes());
    assert!(validate(&bytes, &dtype).is_err());
}
fn union_array(dense: bool) -> UnionArray {
    let fields = UnionFields::try_new(
        [0, 1],
        [
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int32, true),
        ],
    )
    .unwrap();
    let (offsets, children) = if dense {
        (
            Some(ScalarBuffer::from(vec![0, 0])),
            vec![
                Arc::new(Int32Array::from(vec![1])) as ArrayRef,
                Arc::new(Int32Array::from(vec![2])),
            ],
        )
    } else {
        (
            None,
            vec![
                Arc::new(Int32Array::from(vec![1, 0])) as ArrayRef,
                Arc::new(Int32Array::from(vec![0, 2])),
            ],
        )
    };
    UnionArray::try_new(fields, ScalarBuffer::from(vec![0, 1]), offsets, children).unwrap()
}
#[test]
fn test_union_modes_and_run_end_values_keep_canonical_v5_layouts() {
    for dense in [false, true] {
        let array = union_array(dense);
        let dtype = array.data_type().clone();
        let bytes = encode(Arc::new(array));
        validate(&bytes, &dtype).unwrap();
    }
    let run = RunArray::<Int16Type>::try_new(
        &Int16Array::from(vec![1, 2]),
        &Int32Array::from(vec![5, 7]),
    )
    .unwrap();
    let dtype = run.data_type().clone();
    validate(&encode(Arc::new(run)), &dtype).unwrap();
}
