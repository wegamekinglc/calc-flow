//! Existing compressed files remain readable after the zstd security upgrade.
#![cfg(feature = "file")]

use std::{io::Cursor, sync::Arc};

use calc_flow::{DecodeBounds, FormatDecoder};
use calc_flow_connectors::parquet::ParquetCodec;
use datafusion::arrow::{
    array::{ArrayRef, Int64Array, StringArray},
    compute::concat_batches,
    datatypes::{DataType, Field, Schema},
    ipc::{
        CompressionType,
        reader::FileReader,
        writer::{FileWriter, IpcWriteOptions},
    },
    record_batch::RecordBatch,
};
use parquet::{
    arrow::ArrowWriter,
    basic::{Compression, ZstdLevel},
    file::properties::WriterProperties,
};

fn expected() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("value", DataType::Int64, false),
        Field::new("tag", DataType::Utf8, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from_iter_values(0..128)) as ArrayRef,
            Arc::new(
                (0..128)
                    .map(|i| (i % 7 != 0).then_some("snapshot"))
                    .collect::<StringArray>(),
            ),
        ],
    )
    .unwrap()
}

fn read_ipc(bytes: &[u8]) -> RecordBatch {
    let reader = FileReader::try_new(Cursor::new(bytes), None).unwrap();
    let schema = reader.schema();
    let batches = reader.collect::<Result<Vec<_>, _>>().unwrap();
    concat_batches(&schema, &batches).unwrap()
}

#[test]
fn test_zstd_ipc_reads_existing_files_and_roundtrips_multiple_batches() {
    let expected = expected();
    assert_eq!(
        read_ipc(include_bytes!("fixtures/zstd/snapshot.arrow")),
        expected
    );
    let options = IpcWriteOptions::default()
        .try_with_compression(Some(CompressionType::ZSTD))
        .unwrap();
    let mut bytes = Vec::new();
    let mut writer =
        FileWriter::try_new_with_options(&mut bytes, &expected.schema(), options).unwrap();
    writer.write(&expected.slice(0, 64)).unwrap();
    writer.write(&expected.slice(64, 64)).unwrap();
    writer.finish().unwrap();
    drop(writer);
    assert_eq!(read_ipc(&bytes), expected);
}

fn read_parquet(bytes: &[u8]) -> RecordBatch {
    let codec = ParquetCodec::new("1").unwrap();
    let batch = codec
        .decode(bytes, &DecodeBounds::new(128, 1 << 20).unwrap(), &[])
        .unwrap();
    let table = batch.table_payload().unwrap();
    concat_batches(table.schema(), table.batches()).unwrap()
}

#[test]
fn test_zstd_parquet_preserves_optional_codec_and_existing_files() {
    let expected = expected();
    let codec = ParquetCodec::new("1").unwrap();
    let existing = codec.decode(
        include_bytes!("fixtures/zstd/snapshot.parquet"),
        &DecodeBounds::new(128, 1 << 20).unwrap(),
        &[],
    );
    // The connector leaves this upstream codec optional. Verify its error
    // boundary in default builds, and compatibility when parquet/zstd is enabled.
    if let Err(error) = existing {
        assert!(
            error
                .to_string()
                .contains("Disabled feature at compile time: zstd"),
            "{error}"
        );
        return;
    }
    assert_eq!(
        read_parquet(include_bytes!("fixtures/zstd/snapshot.parquet")),
        expected
    );
    let properties = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::default()))
        .set_max_row_group_row_count(Some(64))
        .build();
    let mut bytes = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut bytes, expected.schema(), Some(properties)).unwrap();
    writer.write(&expected).unwrap();
    writer.close().unwrap();
    assert_eq!(read_parquet(&bytes), expected);
}
