use std::sync::Arc;

use arrow::{
    array::{TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use calc_flow::{ArrowFieldSpec, Batch, BatchMetadata, DecodeBounds, FormatDecoder, FormatEncoder};
use calc_flow_connectors::{csv::CsvCodec, json_lines::JsonLinesCodec};

fn fields() -> Vec<ArrowFieldSpec> {
    vec![
        ArrowFieldSpec {
            name: "key".into(),
            data_type: "uint64".into(),
            nullable: false,
        },
        ArrowFieldSpec {
            name: "time".into(),
            data_type: "timestamp[us, UTC]".into(),
            nullable: true,
        },
    ]
}

fn expected() -> Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::UInt64, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            true,
        ),
    ]));
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt64Array::from(vec![1, 2, 3])),
                    Arc::new(
                        TimestampMicrosecondArray::from(vec![Some(105), Some(-1), None])
                            .with_timezone("UTC"),
                    ),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

fn check(codec: &(impl FormatDecoder + FormatEncoder), bytes: &[u8]) {
    let bounds = DecodeBounds::new(3, 4_096).unwrap();
    let declared = fields();
    let saved = declared.clone();
    let original = expected();
    let decoded = codec.decode(bytes, &bounds, &declared).unwrap();
    assert_eq!(
        decoded.table_payload().unwrap().schema(),
        original.table_payload().unwrap().schema()
    );
    assert_eq!(
        decoded.table_payload().unwrap().batches(),
        original.table_payload().unwrap().batches()
    );
    let roundtrip = codec
        .decode(&codec.encode(&original).unwrap(), &bounds, &declared)
        .unwrap();
    assert_eq!(
        roundtrip.table_payload().unwrap().batches(),
        original.table_payload().unwrap().batches()
    );
    assert_eq!(declared, saved);
    assert_eq!(original.metadata(), &BatchMetadata::default());
}

#[test]
fn test_csv_utc_microseconds_preserve_offsets_nulls_and_roundtrip() {
    let codec = CsvCodec::new("1", true).unwrap();
    check(
        &codec,
        b"key,time\n1,1970-01-01T09:00:00.000105+09:00\n2,1969-12-31T23:59:59.999999Z\n3,\n",
    );
}

#[test]
fn test_json_utc_microseconds_preserve_offsets_nulls_and_roundtrip() {
    let codec = JsonLinesCodec::new("1").unwrap();
    check(&codec, b"{\"key\":1,\"time\":\"1970-01-01T09:00:00.000105+09:00\"}\n{\"key\":2,\"time\":\"1969-12-31T23:59:59.999999Z\"}\n{\"key\":3,\"time\":null}\n");
}

#[cfg(feature = "file")]
#[test]
fn test_parquet_utc_microseconds_preserve_nulls_and_roundtrip() {
    let codec = calc_flow_connectors::parquet::ParquetCodec::new("1").unwrap();
    check(&codec, &codec.encode(&expected()).unwrap());
}

#[test]
fn test_schema_timezone_vocabulary_remains_exact() {
    for dtype in [
        "timestamp[us, utc]",
        "timestamp[us, Europe/London]",
        "timestamp[ns, UTC]",
    ] {
        let mut schema = fields();
        schema[1].data_type = dtype.into();
        let error = calc_flow_connectors::arrow_schema::schema_from_spec(&schema).unwrap_err();
        assert!(
            matches!(error, calc_flow::CalcFlowError::InvalidArgument { field, .. } if field == "schema field time")
        );
    }
}
