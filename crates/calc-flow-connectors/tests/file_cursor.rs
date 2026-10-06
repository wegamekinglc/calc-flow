#![cfg(feature = "file")]

use std::{collections::BTreeMap, path::Path, sync::Arc};

use calc_flow::{ArrowFieldSpec, Batch, BatchMetadata, Cursor, SourceEvent, StreamSource};
use calc_flow_connectors::{FileSource, FileSourceConfig};
use datafusion::arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use serde_json::json;

fn source(path: &Path, format: &str) -> FileSource {
    let options = BTreeMap::from([
        ("path".into(), json!(path.display().to_string())),
        ("format".into(), json!(format)),
        ("max_batch_rows".into(), json!(2)),
        (
            "schema".into(),
            json!([ArrowFieldSpec {
                name: "value".into(),
                data_type: "int64".into(),
                nullable: false,
            }]),
        ),
    ]);
    FileSource::new(FileSourceConfig::from_options(&options).unwrap()).unwrap()
}

async fn data(source: &mut FileSource) -> (Batch, Cursor) {
    match source.next().await.unwrap().unwrap() {
        SourceEvent::Data { batch, cursor } => (batch, cursor),
        _ => panic!("expected file data"),
    }
}

fn assert_batch(actual: &Batch, expected: &Batch) {
    assert_eq!(actual.metadata(), expected.metadata());
    let actual = actual.table_payload().unwrap();
    let expected = expected.table_payload().unwrap();
    assert_eq!(actual.schema(), expected.schema());
    assert_eq!(actual.batches(), expected.batches());
}

fn json_files(root: &Path) {
    std::fs::write(root.join("00.json"), b"\n").unwrap();
    std::fs::write(
        root.join("01.json"),
        b"{\"value\":1}\n{\"value\":2}\n{\"value\":3}\n",
    )
    .unwrap();
    std::fs::write(root.join("02.json"), b"{\"value\":4}\n").unwrap();
}

#[tokio::test]
async fn json_cursor_restores_exact_batches_and_metadata() {
    let directory = tempfile::tempdir().unwrap();
    json_files(directory.path());
    let mut live = source(directory.path(), "json");
    live.open(None).await.unwrap();
    let (_, cut) = data(&mut live).await;
    let mut expected = Vec::new();
    for _ in 0..2 {
        expected.push(data(&mut live).await);
    }
    assert!(live.next().await.unwrap().is_none());
    live.close().await.unwrap();
    let mut restored = source(directory.path(), "json");
    restored.open(Some(cut)).await.unwrap();
    for (batch, cursor) in expected {
        let (actual, position) = data(&mut restored).await;
        assert_batch(&actual, &batch);
        assert_eq!(position, cursor);
    }
    assert!(restored.next().await.unwrap().is_none());
    restored.close().await.unwrap();
}

#[tokio::test]
async fn csv_and_parquet_cursors_restore_exact_metadata() {
    use calc_flow::FormatEncoder;
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        false,
    )]));
    for format in ["csv", "parquet"] {
        let directory = tempfile::tempdir().unwrap();
        for value in 1..=2 {
            let record = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(vec![value]))],
            )
            .unwrap();
            let batch = Batch::table(vec![record], BatchMetadata::default()).unwrap();
            let bytes = if format == "csv" {
                calc_flow_connectors::csv::CsvCodec::new("1", true)
                    .unwrap()
                    .encode(&batch)
                    .unwrap()
            } else {
                calc_flow_connectors::parquet::ParquetCodec::new("1")
                    .unwrap()
                    .encode(&batch)
                    .unwrap()
            };
            std::fs::write(directory.path().join(format!("{value:02}.{format}")), bytes).unwrap();
        }
        let mut live = source(directory.path(), format);
        live.open(None).await.unwrap();
        let (_, cut) = data(&mut live).await;
        let (expected, cursor) = data(&mut live).await;
        live.close().await.unwrap();
        let mut restored = source(directory.path(), format);
        restored.open(Some(cut)).await.unwrap();
        let (actual, position) = data(&mut restored).await;
        assert_batch(&actual, &expected);
        assert_eq!(position, cursor);
        assert!(restored.next().await.unwrap().is_none());
        restored.close().await.unwrap();
    }
}

#[tokio::test]
async fn file_cursor_seek_and_reopen_restore_metadata() {
    let directory = tempfile::tempdir().unwrap();
    json_files(directory.path());
    let mut live = source(directory.path(), "json");
    live.open(None).await.unwrap();
    let (first, cut) = data(&mut live).await;
    let (second, _) = data(&mut live).await;
    live.close().await.unwrap();
    live.open(Some(cut)).await.unwrap();
    assert_batch(&data(&mut live).await.0, &second);
    live.close().await.unwrap();
    live.open(None).await.unwrap();
    assert_batch(&data(&mut live).await.0, &first);
    live.close().await.unwrap();
}

#[tokio::test]
async fn file_cursor_rejects_missing_or_inconsistent_position() {
    let directory = tempfile::tempdir().unwrap();
    json_files(directory.path());
    let mut live = source(directory.path(), "json");
    live.open(None).await.unwrap();
    let (_, cut) = data(&mut live).await;
    live.close().await.unwrap();
    for field in ["file", "row", "sequence"] {
        let mut payload = cut.payload().clone();
        payload.remove(field);
        let cursor = Cursor::unbound(cut.order().to_vec(), payload).unwrap();
        assert!(live.open(Some(cursor)).await.is_err(), "missing {field}");
    }
    for (field, value) in [
        ("row", json!(0)),
        ("sequence", json!(0)),
        ("extra", json!(1)),
    ] {
        let mut payload = cut.payload().clone();
        payload.insert(field.into(), value);
        let cursor = Cursor::unbound(cut.order().to_vec(), payload).unwrap();
        assert!(live.open(Some(cursor)).await.is_err(), "invalid {field}");
    }
    let mut order = cut.order().to_vec();
    order[0] = 1;
    let cursor = Cursor::unbound(order, cut.payload().clone()).unwrap();
    assert!(live.open(Some(cursor)).await.is_err());
}

#[tokio::test]
async fn file_cursor_sequence_exhaustion_returns_error() {
    let directory = tempfile::tempdir().unwrap();
    json_files(directory.path());
    let mut live = source(directory.path(), "json");
    live.open(None).await.unwrap();
    let (_, cut) = data(&mut live).await;
    live.close().await.unwrap();
    let mut payload = cut.payload().clone();
    payload.insert("sequence".into(), json!(u64::MAX));
    let cursor = Cursor::unbound(cut.order().to_vec(), payload).unwrap();
    live.open(Some(cursor)).await.unwrap();
    for _ in 0..2 {
        let error = live.next().await.unwrap_err();
        assert!(error.to_string().contains("batch sequence exhausted"));
    }
    live.close().await.unwrap();
}
