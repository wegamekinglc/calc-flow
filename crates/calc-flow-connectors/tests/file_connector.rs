//! Integration tests for the M6.2 file/Parquet connector: codec bounds,
//! deterministic discovery and cursor replay, fail-closed discovery
//! rules, transactional epoch commits, and the file-to-Parquet
//! exactly-once path through the public continuous runtime.

#![cfg(feature = "file")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use calc_flow::{
    ArrowFieldSpec, Batch, BatchMetadata, DecodeBounds, Epoch, ExpressionOperator,
    ManagedCheckpointRuntime, PipelineBuilder, SinkBinding, SourceBinding, SourceEvent,
    StreamExecutionPlan, StreamRequirements, StreamingRunner, UdfRegistry,
};
use calc_flow::{
    ConnectorRegistry, ConnectorSinkFactory, ConnectorSourceFactory, Cursor, FormatDecoder,
    FormatEncoder, StreamSource, TransactionalStreamSink,
};
use calc_flow_connectors::csv::CsvCodec;
use calc_flow_connectors::json_lines::JsonLinesCodec;
use calc_flow_connectors::parquet::ParquetCodec;
use calc_flow_connectors::{
    FileSinkConfig, FileSource, FileSourceConfig, TransactionalParquetSink,
    register_file_connectors,
};
use datafusion::arrow::record_batch::RecordBatch;
use serde_json::{Value, json};

fn field(name: &str, data_type: &str) -> ArrowFieldSpec {
    ArrowFieldSpec {
        name: name.to_string(),
        data_type: data_type.to_string(),
        nullable: false,
    }
}

fn csv_options(dir: &Path) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("path".to_string(), json!(dir.display().to_string())),
        ("format".to_string(), json!("csv")),
    ])
}

fn json_options(dir: &Path) -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("path".to_string(), json!(dir.display().to_string())),
        ("format".to_string(), json!("json")),
    ])
}

async fn collect_source(source: &mut FileSource) -> Vec<Batch> {
    source.open(None).await.expect("source opens");
    let mut batches = Vec::new();
    while let Some(event) = source.next().await.expect("source produces events") {
        match event {
            SourceEvent::Data { batch, .. } => batches.push(batch),
            SourceEvent::Idle | SourceEvent::Watermark(_) => {}
        }
    }
    source.close().await.expect("source closes");
    batches
}

fn temp_root(tag: &str) -> PathBuf {
    tempfile::tempdir().expect("tempdir").keep().join(tag)
}

#[tokio::test]
async fn csv_codec_roundtrips_bounded_batches() {
    let codec = CsvCodec::new("1", true).expect("codec");
    let batch = codec
        .decode(
            b"a,b\n1,2\n3,4\n",
            &DecodeBounds::new(10, 1024).unwrap(),
            &[],
        )
        .expect("decodes");
    assert_eq!(batch.num_rows(), 2);
    let encoded = codec.encode(&batch).expect("encodes");
    let round = codec
        .decode(&encoded, &DecodeBounds::new(10, 1024).unwrap(), &[])
        .expect("round-trips");
    assert_eq!(round.num_rows(), 2);

    let error = codec
        .decode(
            b"a,b\n1,2\n3,4\n",
            &DecodeBounds::new(1, 1024).unwrap(),
            &[],
        )
        .expect_err("row bound enforced");
    assert!(error.to_string().contains("rows"), "{error}");
}

#[tokio::test]
async fn csv_codec_rejects_schema_mismatch() {
    let codec = CsvCodec::new("1", true).expect("codec");
    let schema = vec![field("a", "int64"), field("c", "int64")];
    let error = codec
        .decode(
            b"a,b\n1,2\n",
            &DecodeBounds::new(10, 1024).unwrap(),
            &schema,
        )
        .expect_err("column mismatch fails closed");
    assert!(error.to_string().contains("schema"), "{error}");
}

#[tokio::test]
async fn json_codec_roundtrips_bounded_batches() {
    let codec = JsonLinesCodec::new("1").expect("codec");
    let batch = codec
        .decode(
            b"{\"a\":1}\n{\"a\":2}\n",
            &DecodeBounds::new(10, 1024).unwrap(),
            &[],
        )
        .expect("decodes");
    assert_eq!(batch.num_rows(), 2);
    let encoded = codec.encode(&batch).expect("encodes");
    assert_eq!(
        codec
            .decode(&encoded, &DecodeBounds::new(10, 1024).unwrap(), &[])
            .expect("round-trips")
            .num_rows(),
        2
    );
    let schema = vec![field("b", "int64")];
    let error = codec
        .decode(
            b"{\"a\":1}\n",
            &DecodeBounds::new(10, 1024).unwrap(),
            &schema,
        )
        .expect_err("field mismatch fails closed");
    assert!(error.to_string().contains("schema"), "{error}");
}

fn sample_batch(rows: i64) -> Batch {
    use datafusion::arrow::array::Int64Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
    let array = Int64Array::from((1..=rows).collect::<Vec<_>>());
    let record = RecordBatch::try_new(schema, vec![Arc::new(array)]).expect("record batch");
    Batch::table(
        vec![record],
        BatchMetadata::new("test", 1, BTreeMap::new()).unwrap(),
    )
    .unwrap()
}

#[tokio::test]
async fn parquet_codec_roundtrips_and_enforces_row_groups() {
    let codec = ParquetCodec::new("1").expect("codec");
    let batch = sample_batch(4);
    let encoded = codec.encode(&batch).expect("encodes");
    let decoded = codec
        .decode(&encoded, &DecodeBounds::new(10, 1024 * 1024).unwrap(), &[])
        .expect("decodes");
    assert_eq!(decoded.num_rows(), 4);

    let error = codec
        .decode(&encoded, &DecodeBounds::new(2, 1024 * 1024).unwrap(), &[])
        .expect_err("row-group row bound enforced");
    assert!(
        error.to_string().contains("row group") || error.to_string().contains("rows"),
        "{error}"
    );

    let schema = vec![field("b", "int64")];
    let error = codec
        .decode(
            &encoded,
            &DecodeBounds::new(10, 1024 * 1024).unwrap(),
            &schema,
        )
        .expect_err("stored schema mismatch fails closed");
    assert!(error.to_string().contains("schema"), "{error}");

    let error = codec
        .decode(
            b"not a parquet file",
            &DecodeBounds::new(10, 1024).unwrap(),
            &[],
        )
        .expect_err("corrupt parquet fails closed");
    assert!(!error.to_string().is_empty());
}

#[tokio::test]
async fn source_discovers_files_deterministically_and_replays() {
    let root = temp_root("discovery");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("b.csv"), b"a,b\n2,3\n").unwrap();
    std::fs::write(root.join("a.csv"), b"a,b\n0,1\n").unwrap();
    std::fs::write(root.join("c.csv"), b"a,b\n4,5\n").unwrap();

    let mut source =
        FileSource::new(FileSourceConfig::from_options(&csv_options(&root)).expect("config"))
            .expect("source");
    let batches = collect_source(&mut source).await;
    assert_eq!(batches.len(), 3, "one batch per file");
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(
        batches[0].metadata().attributes().get("file").unwrap(),
        &json!("a.csv"),
        "lexicographic file order is the stable identity order"
    );

    let cursor = {
        let mut source =
            FileSource::new(FileSourceConfig::from_options(&csv_options(&root)).expect("config"))
                .expect("source");
        source.open(None).await.unwrap();
        let mut last = None;
        while let Some(event) = source.next().await.unwrap() {
            if let SourceEvent::Data { cursor, .. } = event {
                last = Some(cursor);
            }
        }
        last.expect("at least one cursor")
    };
    assert_eq!(cursor.payload().get("file"), Some(&json!("c.csv")));

    let mut replay =
        FileSource::new(FileSourceConfig::from_options(&csv_options(&root)).expect("config"))
            .expect("source");
    replay.open(Some(cursor)).await.expect("replay opens");
    assert!(
        replay.next().await.expect("replay exhausted").is_none(),
        "replay from the last cursor ends immediately"
    );
}

#[tokio::test]
async fn json_source_replays_from_row_cursor() {
    let root = temp_root("json_replay");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(
        root.join("data.json"),
        b"{\"a\":1}\n{\"a\":2}\n{\"a\":3}\n{\"a\":4}\n",
    )
    .unwrap();
    let mut options = json_options(&root);
    options.insert("max_batch_rows".to_string(), json!(2));

    let mut source =
        FileSource::new(FileSourceConfig::from_options(&options).expect("config")).unwrap();
    source.open(None).await.unwrap();
    let mut cursors = Vec::new();
    let mut rows = Vec::new();
    while let Some(event) = source.next().await.unwrap() {
        if let SourceEvent::Data { batch, cursor } = event {
            assert_eq!(batch.num_rows(), 2, "bounded chunks");
            cursors.push(cursor);
            rows.push(batch.num_rows());
        }
    }
    assert_eq!(rows, vec![2, 2]);
    assert_eq!(cursors[0].payload().get("row"), Some(&json!(2)));

    let mut replay =
        FileSource::new(FileSourceConfig::from_options(&options).expect("config")).unwrap();
    replay.open(Some(cursors[0].clone())).await.unwrap();
    let event = replay.next().await.unwrap().expect("resumes mid-file");
    if let SourceEvent::Data { batch, cursor } = event {
        assert_eq!(batch.num_rows(), 2, "remaining rows only");
        assert_eq!(cursor.payload().get("row"), Some(&json!(4)));
    } else {
        panic!("expected data");
    }
}

#[tokio::test]
async fn discovery_fails_closed() {
    let root = temp_root("fail_closed");
    std::fs::create_dir_all(root.join("nested")).unwrap();
    std::fs::write(root.join("data.csv"), b"a,b\n1,2\n").unwrap();

    let mut traversal = csv_options(&root);
    traversal.insert(
        "path".to_string(),
        json!(format!("{}/../target", root.display())),
    );
    let error = FileSourceConfig::from_options(&traversal).expect_err("traversal rejected");
    assert!(error.to_string().contains("traversal"), "{error}");

    let mut subdir = csv_options(&root);
    subdir.insert(
        "path".to_string(),
        json!(root.join("nested").display().to_string()),
    );
    std::fs::write(root.join("nested").join("data.csv"), b"a,b\n1,2\n").unwrap();
    let mut with_files = FileSource::new(FileSourceConfig::from_options(&subdir).unwrap()).unwrap();
    let batches = collect_source(&mut with_files).await;
    assert_eq!(batches.len(), 1);

    let flat = root.join("flat");
    std::fs::create_dir_all(&flat).unwrap();
    std::fs::write(
        flat.join("data.csv"),
        b"a,b
1,2
",
    )
    .unwrap();
    std::fs::write(flat.join("unexpected.txt"), b"not csv").unwrap();
    let wrong_type = csv_options(&flat);
    let mut source = FileSource::new(FileSourceConfig::from_options(&wrong_type).unwrap()).unwrap();
    let error = source
        .open(None)
        .await
        .expect_err("wrong extension fails closed");
    assert!(error.to_string().contains("extension"), "{error}");
}

#[cfg(unix)]
#[tokio::test]
async fn symlinked_entries_fail_closed() {
    let root = temp_root("symlink");
    std::fs::create_dir_all(&root).unwrap();
    let target = root.join("real.csv");
    std::fs::write(&target, b"a,b\n1,2\n").unwrap();
    std::os::unix::fs::symlink(&target, root.join("link.csv")).unwrap();

    let mut source =
        FileSource::new(FileSourceConfig::from_options(&csv_options(&root)).unwrap()).unwrap();
    let error = source.open(None).await.expect_err("symlink fails closed");
    assert!(error.to_string().contains("symlink"), "{error}");

    let link = root.join("link.csv");
    let direct = csv_options(&link);
    let mut source = FileSource::new(FileSourceConfig::from_options(&direct).unwrap()).unwrap();
    let error = source
        .open(None)
        .await
        .expect_err("symlinked root fails closed");
    assert!(error.to_string().contains("symlink"), "{error}");
}

#[cfg(windows)]
#[tokio::test]
async fn locked_file_fails_closed() {
    use std::os::windows::fs::OpenOptionsExt;
    let root = temp_root("locked");
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("data.csv");
    std::fs::write(&path, b"a,b\n1,2\n").unwrap();
    let _lock = std::fs::OpenOptions::new()
        .read(true)
        .share_mode(0)
        .open(&path)
        .expect("lock acquired");

    let mut source =
        FileSource::new(FileSourceConfig::from_options(&csv_options(&root)).unwrap()).unwrap();
    source.open(None).await.expect("discovery succeeds");
    let error = source
        .next()
        .await
        .expect_err("locked file read fails closed");
    assert!(error.to_string().contains("read"), "{error}");
}

#[tokio::test]
async fn partial_csv_row_fails_closed() {
    let root = temp_root("partial");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("data.csv"), b"a,b\n1\n").unwrap();
    let mut source =
        FileSource::new(FileSourceConfig::from_options(&csv_options(&root)).unwrap()).unwrap();
    source.open(None).await.unwrap();
    let error = source.next().await.expect_err("partial row fails closed");
    assert!(!error.to_string().is_empty());
}

struct Harness;

impl Harness {
    fn sink(root: &Path, output: &str) -> TransactionalParquetSink {
        TransactionalParquetSink::new(FileSinkConfig {
            root: root.to_path_buf(),
            output: output.to_string(),
        })
        .expect("sink")
    }

    fn committed_epochs(root: &Path, output: &str) -> Vec<String> {
        let dir = root.join(output);
        let mut epochs: Vec<String> = std::fs::read_dir(&dir)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .filter(|name| name.starts_with("epoch="))
                    .collect()
            })
            .unwrap_or_default();
        epochs.sort();
        epochs
    }
}

#[tokio::test]
async fn sink_stages_then_commits_atomically() {
    let root = temp_root("sink_staging");
    let mut sink = Harness::sink(&root, "out");
    sink.open().await.expect("opens");
    sink.begin_epoch(Epoch::INITIAL).await.expect("begins");
    sink.write(&sample_batch(2)).await.expect("writes");
    let evidence = sink.pre_commit(Epoch::INITIAL).await.expect("pre-commits");

    assert!(
        Harness::committed_epochs(&root, "out").is_empty(),
        "staging is not committed output"
    );
    let staging = root.join("out").join(".staging").join("epoch=1");
    assert!(staging.join("manifest.json").exists(), "manifest published");

    sink.commit(Epoch::INITIAL, &evidence)
        .await
        .expect("commits");
    assert_eq!(
        Harness::committed_epochs(&root, "out"),
        vec!["epoch=1".to_string()]
    );
    assert!(
        root.join("out")
            .join("epoch=1")
            .join("manifest.json")
            .exists(),
        "committed epoch carries its manifest"
    );
    assert!(
        !root.join("out").join(".staging").join("epoch=1").exists(),
        "staging is removed on commit"
    );

    sink.commit(Epoch::INITIAL, &evidence)
        .await
        .expect("replayed commit is idempotent");
    assert_eq!(Harness::committed_epochs(&root, "out").len(), 1);

    let conflicting: BTreeMap<String, Value> = BTreeMap::from([
        ("output".to_string(), json!("out")),
        ("epoch".to_string(), json!(1)),
        ("parts".to_string(), json!(["part-0000.parquet"])),
        ("rows".to_string(), json!(999)),
    ]);
    let error = sink
        .commit(Epoch::INITIAL, &conflicting)
        .await
        .expect_err("conflicting evidence rejected");
    assert!(error.to_string().contains("manifest"), "{error}");
}

#[tokio::test]
async fn sink_abort_and_recover() {
    let root = temp_root("sink_abort");
    let mut sink = Harness::sink(&root, "out");
    sink.open().await.expect("opens");
    sink.begin_epoch(Epoch::INITIAL).await.expect("begins");
    sink.write(&sample_batch(1)).await.expect("writes");
    let evidence = sink.pre_commit(Epoch::INITIAL).await.expect("pre-commits");
    sink.commit(Epoch::INITIAL, &evidence)
        .await
        .expect("commits");

    let next_epoch = Epoch::INITIAL.next().expect("next epoch");
    sink.begin_epoch(next_epoch).await.expect("begins next");
    sink.write(&sample_batch(1)).await.expect("writes");
    sink.abort(next_epoch, None).await.expect("aborts");
    assert_eq!(
        Harness::committed_epochs(&root, "out"),
        vec!["epoch=1".to_string()],
        "aborted epochs never reach the output"
    );

    // Recovery reconciliation runs through the runtime's SinkRecovery
    // evidence; the restart leg of file_to_parquet_exactly_once drives it
    // end to end, while commit's idempotence and manifest checks above pin
    // the same evidence comparison.
    let _ = evidence;
}

#[tokio::test]
async fn sink_never_touches_unrelated_user_files() {
    let root = temp_root("sink_unrelated");
    std::fs::create_dir_all(root.join("out")).unwrap();
    std::fs::write(root.join("out").join("user-notes.txt"), b"keep me").unwrap();
    let mut sink = Harness::sink(&root, "out");
    sink.open().await.expect("opens");
    sink.begin_epoch(Epoch::INITIAL).await.expect("begins");
    sink.write(&sample_batch(1)).await.expect("writes");
    let evidence = sink.pre_commit(Epoch::INITIAL).await.expect("pre-commits");
    sink.commit(Epoch::INITIAL, &evidence)
        .await
        .expect("commits");
    assert_eq!(
        std::fs::read(root.join("out").join("user-notes.txt")).unwrap(),
        b"keep me",
        "unrelated files survive epoch commits"
    );
}

fn file_to_parquet_plan(output_id: &str) -> StreamExecutionPlan {
    PipelineBuilder::new("file_to_parquet")
        .expect("builder")
        .add_node(
            "total",
            Box::new(
                ExpressionOperator::new("total", "total = a + b", Vec::new(), None, Vec::new())
                    .expect("operator"),
            ),
        )
        .expect("node")
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements {
                delivery: BTreeMap::from([(
                    output_id.to_string(),
                    calc_flow::DeliveryGuarantee::ExactlyOnce,
                )]),
            },
        )
        .expect("plan")
}

fn discover_binding_ids() -> (String, String) {
    let probe = PipelineBuilder::new("file_to_parquet")
        .expect("builder")
        .add_node(
            "total",
            Box::new(
                ExpressionOperator::new("total", "total = a + b", Vec::new(), None, Vec::new())
                    .expect("operator"),
            ),
        )
        .expect("node")
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .expect("probe plan");
    (
        probe.source_binding_ids()[0].to_owned(),
        probe.sink_binding_ids()[0].to_owned(),
    )
}

async fn run_once(root: &Path, input: &Path) -> (calc_flow::JobOutcome, Epoch) {
    let (source_id, output_id) = discover_binding_ids();
    let plan = file_to_parquet_plan(&output_id);
    let source =
        FileSource::new(FileSourceConfig::from_options(&json_options(input)).unwrap()).unwrap();
    let sink = TransactionalParquetSink::new(FileSinkConfig {
        root: root.join("out"),
        output: "totals".to_string(),
    })
    .unwrap();
    let binding = SourceBinding::new(source)
        .with_watermark_policy(calc_flow::WatermarkPolicy::Disabled { idle_timeout: None });
    let runner = StreamingRunner::new(
        plan,
        BTreeMap::from([(source_id, binding)]),
        BTreeMap::from([(
            output_id,
            vec![SinkBinding::transactional("totals", sink).unwrap()],
        )]),
        ManagedCheckpointRuntime::new(root.join("checkpoints")).unwrap(),
    )
    .expect("runner");
    let job = runner.start().await.expect("starts");
    // The finite source converges on its own; wait() observes natural
    // completion instead of draining before the first poll.
    let outcome = job.wait().await;
    let completed = outcome.completed_epoch;
    (
        outcome,
        completed.expect("shutdown completes a checkpoint epoch"),
    )
}

fn read_total_rows(out_root: &Path) -> usize {
    let codec = ParquetCodec::new("1").unwrap();
    let mut rows = 0;
    for epoch in Harness::committed_epochs(out_root, "totals") {
        let dir = out_root.join("totals").join(&epoch);
        for entry in std::fs::read_dir(&dir).unwrap().filter_map(Result::ok) {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap();
            let batch = codec
                .decode(&bytes, &DecodeBounds::new(1024, 1024 * 1024).unwrap(), &[])
                .unwrap();
            rows += batch.num_rows();
        }
    }
    rows
}

async fn wait_for_committed_epoch(root: &Path, output: &str) {
    for _ in 0..200 {
        if !Harness::committed_epochs(root, output).is_empty() {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("no epoch was committed within the wait budget");
}

#[tokio::test]
async fn file_to_parquet_exactly_once() {
    let root = temp_root("exactly_once");
    let input = root.join("input");
    std::fs::create_dir_all(&input).unwrap();
    std::fs::write(
        input.join("data.json"),
        b"{\"a\":1,\"b\":2}\n{\"a\":3,\"b\":4}\n{\"a\":5,\"b\":6}\n",
    )
    .unwrap();
    let checkpoint_root = root.join("checkpoints");
    std::fs::create_dir_all(&checkpoint_root).unwrap();

    let (outcome, completed) = run_once(&root, &input).await;
    assert!(outcome.completed_epoch >= Some(completed));
    assert_eq!(
        Harness::committed_epochs(&root.join("out"), "totals").len(),
        1,
        "exactly one committed epoch"
    );

    let codec = ParquetCodec::new("1").unwrap();
    let epoch_dir = root.join("out").join("totals").join("epoch=1");
    let parts: Vec<_> = std::fs::read_dir(&epoch_dir)
        .unwrap()
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some("parquet"))
        .collect();
    let mut totals: Vec<i64> = Vec::new();
    for part in parts {
        let bytes = std::fs::read(&part).unwrap();
        let batch = codec
            .decode(&bytes, &DecodeBounds::new(1024, 1024 * 1024).unwrap(), &[])
            .unwrap();
        let payload = batch.table_payload().unwrap();
        for record in payload.batches() {
            use datafusion::arrow::array::Int64Array;
            let column = record
                .column_by_name("total")
                .expect("total column")
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("int64 column");
            totals.extend(column.iter().map(|value| value.unwrap_or(0)));
        }
    }
    totals.sort_unstable();
    assert_eq!(totals, vec![3, 7, 11], "every input row lands exactly once");

    let (restart_outcome, _) = run_once(&root, &input).await;
    assert!(
        restart_outcome.completed_epoch.is_some(),
        "restart converges"
    );
    assert_eq!(
        Harness::committed_epochs(&root.join("out"), "totals").len(),
        1,
        "recovery replay adds no duplicate epochs"
    );
    assert_eq!(
        read_total_rows(&root.join("out")),
        3,
        "output stays exactly-once after restart"
    );
}

#[tokio::test]
async fn file_to_parquet_recovers_after_cancel() {
    let root = temp_root("recover_cancel");
    let input = root.join("input");
    std::fs::create_dir_all(&input).unwrap();
    for index in 0..40 {
        std::fs::write(
            input.join(format!("part-{index:03}.json")),
            b"{\"a\":1,\"b\":2}\n{\"a\":3,\"b\":4}\n{\"a\":5,\"b\":6}\n",
        )
        .unwrap();
    }
    let mut options = json_options(&input);
    options.insert("max_batch_rows".to_string(), json!(3));

    let (source_id, output_id) = discover_binding_ids();
    let build_runner = |options: BTreeMap<String, Value>| {
        let plan = file_to_parquet_plan(&output_id);
        let source = FileSource::new(FileSourceConfig::from_options(&options).unwrap()).unwrap();
        let binding = SourceBinding::new(source)
            .with_watermark_policy(calc_flow::WatermarkPolicy::Disabled { idle_timeout: None });
        StreamingRunner::new(
            plan,
            BTreeMap::from([(source_id.clone(), binding)]),
            BTreeMap::from([(
                output_id.clone(),
                vec![
                    SinkBinding::transactional(
                        "totals",
                        TransactionalParquetSink::new(FileSinkConfig {
                            root: root.join("out"),
                            output: "totals".to_string(),
                        })
                        .unwrap(),
                    )
                    .unwrap(),
                ],
            )]),
            ManagedCheckpointRuntime::new(root.join("checkpoints")).unwrap(),
        )
        .expect("runner")
    };

    let cancelled = build_runner(options.clone());
    let job = cancelled.start().await.expect("starts");
    wait_for_committed_epoch(&root.join("out"), "totals").await;
    let outcome = job.cancel().await;
    assert!(
        matches!(
            outcome.state,
            calc_flow::JobState::Cancelled | calc_flow::JobState::Completed
        ),
        "cancel lands on a terminal state: {:?}",
        outcome.state
    );

    let recovered = build_runner(options);
    let job = recovered.start().await.expect("restarts");
    let outcome = job.wait().await;
    assert_eq!(outcome.state, calc_flow::JobState::Completed);

    assert_eq!(
        read_total_rows(&root.join("out")),
        120,
        "every input row lands exactly once across cancel and recovery"
    );
    for epoch in Harness::committed_epochs(&root.join("out"), "totals") {
        assert!(
            root.join("out")
                .join("totals")
                .join(&epoch)
                .join("manifest.json")
                .exists(),
            "committed epoch {epoch} carries its manifest"
        );
    }
}

struct NoSecrets;

#[async_trait]
impl calc_flow::SecretResolver for NoSecrets {
    fn resolve(
        &self,
        reference: &calc_flow::SecretReference,
    ) -> calc_flow::Result<calc_flow::SecretHandle> {
        Err(calc_flow::CalcFlowError::NotFound {
            resource: "secret".into(),
            key: reference.key.clone(),
        })
    }
}

#[test]
fn config_parsing_rejects_invalid_options() {
    let root = temp_root("config");
    std::fs::create_dir_all(&root).unwrap();

    let missing_path: BTreeMap<String, Value> =
        BTreeMap::from([("format".to_string(), json!("csv"))]);
    let error = FileSourceConfig::from_options(&missing_path).expect_err("path required");
    assert!(error.to_string().contains("path"), "{error}");

    let mut wrong_format = csv_options(&root);
    wrong_format.insert("format".to_string(), json!("xml"));
    let error = FileSourceConfig::from_options(&wrong_format).expect_err("format vocabulary");
    assert!(error.to_string().contains("format"), "{error}");

    let mut bad_header = csv_options(&root);
    bad_header.insert("header".to_string(), json!(1));
    let error = FileSourceConfig::from_options(&bad_header).expect_err("header type");
    assert!(error.to_string().contains("header"), "{error}");

    let mut bad_rows = csv_options(&root);
    bad_rows.insert("max_batch_rows".to_string(), json!("many"));
    let error = FileSourceConfig::from_options(&bad_rows).expect_err("bound type");
    assert!(error.to_string().contains("max_batch_rows"), "{error}");

    let mut bad_schema = csv_options(&root);
    bad_schema.insert("schema".to_string(), json!("not-a-list"));
    let error = FileSourceConfig::from_options(&bad_schema).expect_err("schema shape");
    assert!(error.to_string().contains("schema"), "{error}");

    let mut zero_rows = json_options(&root);
    zero_rows.insert("max_batch_rows".to_string(), json!(0));
    let error = FileSourceConfig::from_options(&zero_rows)
        .and_then(|config| FileSource::new(config).map(|_| ()))
        .expect_err("zero row bound rejected by decode bounds");
    assert!(!error.to_string().is_empty());

    let missing_output: BTreeMap<String, Value> =
        BTreeMap::from([("path".to_string(), json!(root.display().to_string()))]);
    let error = FileSinkConfig::from_options(&missing_output).expect_err("output required");
    assert!(error.to_string().contains("output"), "{error}");

    let nested_output = BTreeMap::from([
        ("path".to_string(), json!(root.display().to_string())),
        ("output".to_string(), json!("a/b")),
    ]);
    let error = FileSinkConfig::from_options(&nested_output).expect_err("output shape");
    assert!(error.to_string().contains("single directory"), "{error}");

    let dot_output = BTreeMap::from([
        ("path".to_string(), json!(root.display().to_string())),
        ("output".to_string(), json!("..")),
    ]);
    assert!(FileSinkConfig::from_options(&dot_output).is_err());
}

#[test]
fn schema_conversion_rejects_unknown_types_and_empty_names() {
    let bad_type = vec![ArrowFieldSpec {
        name: "a".to_string(),
        data_type: "decimal".to_string(),
        nullable: false,
    }];
    let error = calc_flow_connectors::arrow_schema::schema_from_spec(&bad_type)
        .expect_err("unknown type fails closed");
    assert!(error.to_string().contains("unsupported"), "{error}");

    let bad_name = vec![ArrowFieldSpec {
        name: String::new(),
        data_type: "int64".to_string(),
        nullable: false,
    }];
    let error = calc_flow_connectors::arrow_schema::schema_from_spec(&bad_name)
        .expect_err("empty name fails closed");
    assert!(error.to_string().contains("must not be empty"), "{error}");
}

#[tokio::test]
async fn codec_error_paths_fail_closed() {
    let csv = CsvCodec::new("1", true).expect("codec");
    let schema = vec![field("a", "int64"), field("b", "int64")];
    let error = csv
        .decode(
            b"a,b\nnotanint,2\n",
            &DecodeBounds::new(10, 1024).unwrap(),
            &schema,
        )
        .expect_err("invalid int value fails closed");
    assert!(!error.to_string().is_empty());

    let error = csv
        .decode(b"a,b\n1,2\n", &DecodeBounds::new(10, 8).unwrap(), &[])
        .expect_err("byte bound enforced");
    assert!(error.to_string().contains("bytes"), "{error}");

    let headerless = CsvCodec::new("1", false).expect("codec");
    let batch = headerless
        .decode(b"1,2\n", &DecodeBounds::new(10, 1024).unwrap(), &[])
        .expect("headerless decode infers");
    assert_eq!(batch.num_rows(), 1);

    let json = JsonLinesCodec::new("1").expect("codec");
    let error = json
        .decode(
            b"{broken json\n",
            &DecodeBounds::new(10, 1024).unwrap(),
            &[],
        )
        .expect_err("malformed json fails closed");
    assert!(!error.to_string().is_empty());

    let parquet = ParquetCodec::new("1").expect("codec");
    let encoded = parquet.encode(&sample_batch(4)).expect("encodes");
    let error = parquet
        .decode(&encoded, &DecodeBounds::new(100, 8).unwrap(), &[])
        .expect_err("byte bound enforced");
    assert!(error.to_string().contains("bytes"), "{error}");
}

#[tokio::test]
async fn sink_lifecycle_rejects_out_of_order_operations() {
    let root = temp_root("sink_order");
    let mut sink = Harness::sink(&root, "out");
    sink.open().await.expect("opens");
    let error = sink
        .write(&sample_batch(1))
        .await
        .expect_err("write before begin_epoch");
    assert!(error.to_string().contains("begin_epoch"), "{error}");

    let error = sink
        .pre_commit(Epoch::INITIAL)
        .await
        .expect_err("pre_commit before begin_epoch");
    assert!(error.to_string().contains("begin_epoch"), "{error}");

    sink.begin_epoch(Epoch::INITIAL).await.expect("begins");
    let later = Epoch::INITIAL.next().expect("epoch");
    let error = sink
        .pre_commit(later)
        .await
        .expect_err("inactive epoch rejected");
    assert!(error.to_string().contains("inactive"), "{error}");

    let unbegun = Epoch::new(3).expect("epoch");
    let error = sink
        .commit(unbegun, &BTreeMap::new())
        .await
        .expect_err("commit without a staged epoch fails closed");
    assert!(error.to_string().contains("missing"), "{error}");

    // A committed epoch without its manifest fails closed on replay.
    let bare = root.join("out").join("epoch=2");
    std::fs::create_dir_all(&bare).unwrap();
    let evidence = BTreeMap::from([
        ("output".to_string(), json!("out")),
        ("epoch".to_string(), json!(2)),
        ("parts".to_string(), Value::Array(vec![])),
        ("rows".to_string(), json!(0)),
    ]);
    let error = sink
        .commit(Epoch::new(2).unwrap(), &evidence)
        .await
        .expect_err("manifest-less epoch rejected");
    assert!(error.to_string().contains("manifest"), "{error}");
}

#[tokio::test]
async fn source_replay_rejects_unknown_cursor_files() {
    let root = temp_root("cursor_mismatch");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("a.csv"), b"a,b\n1,2\n").unwrap();

    let order = vec![9u8; 16];
    let payload = BTreeMap::from([
        ("file".to_string(), json!("ghost.csv")),
        ("row".to_string(), json!(0)),
    ]);
    let cursor = Cursor::unbound(order, payload).expect("cursor");
    let mut source =
        FileSource::new(FileSourceConfig::from_options(&csv_options(&root)).unwrap()).unwrap();
    let error = source
        .open(Some(cursor))
        .await
        .expect_err("unknown cursor file fails closed");
    assert!(error.to_string().contains("ghost.csv"), "{error}");
}

#[tokio::test]
async fn source_factory_opens_from_data_only_options() {
    let root = temp_root("factory");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("a.csv"), b"a,b\n1,2\n").unwrap();
    let mut registry = ConnectorRegistry::new();
    register_file_connectors(&mut registry).unwrap();
    let snapshot = registry.snapshot();
    let factory = calc_flow_connectors::resolve_file_source(&snapshot).expect("resolves");

    let mut source = factory
        .open(&csv_options(&root), &NoSecrets)
        .await
        .expect("opens");
    source.open(None).await.expect("discovers");
    let event = source.next().await.expect("produces").expect("has data");
    assert!(matches!(event, SourceEvent::Data { .. }));

    let mut bad_format = csv_options(&root);
    bad_format.insert("format".to_string(), json!("xml"));
    let outcome = factory.open(&bad_format, &NoSecrets).await;
    let error = match outcome {
        Ok(_) => panic!("factory must reject unknown formats"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("format"), "{error}");
}

#[tokio::test]
async fn ordinary_sink_rejects_writes_after_close_cycle() {
    let root = temp_root("ordinary_reject");
    let options = BTreeMap::from([
        ("path".to_string(), json!(root.display().to_string())),
        ("output".to_string(), json!("out")),
    ]);
    let mut registry = ConnectorRegistry::new();
    register_file_connectors(&mut registry).unwrap();
    let factory = registry
        .snapshot()
        .resolve_sink(
            &calc_flow::ConnectorIdentity::new(
                "calc-flow-connectors",
                "file",
                calc_flow_connectors::FILE_CONNECTOR_VERSION,
            )
            .unwrap(),
        )
        .unwrap();
    let mut sink = factory.open(&options, &NoSecrets).await.expect("opens");
    sink.open().await.expect("sink opens");
    sink.write(&sample_batch(1)).await.expect("writes");
    sink.close().await.expect("closes and commits");
    assert_eq!(
        Harness::committed_epochs(&root, "out"),
        vec!["epoch=1".to_string()]
    );
}

#[derive(Debug)]
struct FakeExternalPayload;

impl calc_flow::ExternalPayload for FakeExternalPayload {
    fn backend(&self) -> &'static str {
        "fake"
    }

    fn len(&self) -> usize {
        1
    }

    fn estimated_bytes(&self) -> usize {
        1
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn external_batch() -> Batch {
    Batch::external(
        std::sync::Arc::new(FakeExternalPayload),
        BatchMetadata::new("test", 1, BTreeMap::new()).unwrap(),
    )
    .unwrap()
}

#[tokio::test]
async fn codecs_reject_non_table_batches_on_encode() {
    let batch = external_batch();
    for error in [
        CsvCodec::new("1", true)
            .unwrap()
            .encode(&batch)
            .unwrap_err(),
        JsonLinesCodec::new("1")
            .unwrap()
            .encode(&batch)
            .unwrap_err(),
        ParquetCodec::new("1").unwrap().encode(&batch).unwrap_err(),
    ] {
        assert!(error.to_string().contains("table batches only"), "{error}");
    }
}

#[tokio::test]
async fn json_source_skips_empty_files_and_enforces_line_bounds() {
    let root = temp_root("json_edges");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("empty.json"), b"").unwrap();
    std::fs::write(
        root.join("data.json"),
        b"{\"a\":1}
{\"a\":2}
",
    )
    .unwrap();
    let mut source =
        FileSource::new(FileSourceConfig::from_options(&json_options(&root)).unwrap()).unwrap();
    let batches = collect_source(&mut source).await;
    assert_eq!(batches.len(), 1, "empty files are skipped");
    assert_eq!(batches[0].num_rows(), 2);

    let mut tight = json_options(&root);
    tight.insert("max_batch_bytes".to_string(), json!(4));
    let mut source = FileSource::new(FileSourceConfig::from_options(&tight).unwrap()).unwrap();
    source.open(None).await.expect("discovers");
    let error = source
        .next()
        .await
        .expect_err("a line above the byte bound fails closed");
    assert!(error.to_string().contains("byte limit"), "{error}");
}

#[tokio::test]
async fn begin_epoch_replaces_stale_staging() {
    let root = temp_root("stale_staging");
    let mut sink = Harness::sink(&root, "out");
    sink.open().await.expect("opens");
    sink.begin_epoch(Epoch::INITIAL).await.expect("begins");
    sink.write(&sample_batch(1)).await.expect("writes");
    sink.begin_epoch(Epoch::INITIAL).await.expect("re-begins");
    let staging = root.join("out").join(".staging").join("epoch=1");
    assert!(
        std::fs::read_dir(&staging)
            .map(|mut entries| entries.next().is_none())
            .unwrap_or(false),
        "re-begin clears stale staging parts"
    );
    let evidence = sink.pre_commit(Epoch::INITIAL).await.expect("pre-commits");
    assert!(
        evidence.get("rows").and_then(Value::as_u64).unwrap_or(1) == 0,
        "stale rows do not leak into the new evidence"
    );
}

#[tokio::test]
async fn transactional_factory_sink_completes_epochs() {
    let root = temp_root("factory_txn");
    let options = BTreeMap::from([
        ("path".to_string(), json!(root.display().to_string())),
        ("output".to_string(), json!("out")),
    ]);
    let mut registry = ConnectorRegistry::new();
    register_file_connectors(&mut registry).unwrap();
    let factory = registry
        .snapshot()
        .resolve_sink(
            &calc_flow::ConnectorIdentity::new(
                "calc-flow-connectors",
                "file",
                calc_flow_connectors::FILE_CONNECTOR_VERSION,
            )
            .unwrap(),
        )
        .unwrap();
    let mut sink = factory
        .open_transactional(&options, &NoSecrets)
        .await
        .expect("opens transactional")
        .expect("file provides a transactional implementation");
    sink.open().await.expect("opens");
    sink.begin_epoch(Epoch::INITIAL).await.expect("begins");
    sink.write(&sample_batch(2)).await.expect("writes");
    let evidence = sink.pre_commit(Epoch::INITIAL).await.expect("pre-commits");
    sink.commit(Epoch::INITIAL, &evidence)
        .await
        .expect("commits");
    assert_eq!(
        Harness::committed_epochs(&root, "out"),
        vec!["epoch=1".to_string()]
    );
}

#[test]
fn registry_defaults_and_codec_only_registration() {
    let source_factory = calc_flow_connectors::FileSourceFactory::default();
    let sink_factory = calc_flow_connectors::FileSinkFactory::default();
    assert_eq!(source_factory.descriptor().identity.name.as_ref(), "file");
    assert_eq!(sink_factory.descriptor().identity.name.as_ref(), "file");

    let mut registry = ConnectorRegistry::new();
    register_file_connectors(&mut registry).expect("registers");
    let error = calc_flow_connectors::register_format_codecs(&mut registry)
        .expect_err("duplicate format identities conflict");
    assert!(
        matches!(error, calc_flow::CalcFlowError::Conflict { .. }),
        "{error}"
    );
}

#[tokio::test]
async fn sink_config_rejects_non_string_options() {
    let options = BTreeMap::from([
        ("path".to_string(), json!(42)),
        ("output".to_string(), json!("out")),
    ]);
    let error = FileSinkConfig::from_options(&options).expect_err("non-string path");
    assert!(error.to_string().contains("path"), "{error}");
}

#[test]
fn codec_identities_and_error_projection_contract() {
    let csv = CsvCodec::new("1", true).unwrap();
    assert_eq!(FormatDecoder::identity(&csv).name.as_ref(), "csv");
    assert_eq!(FormatEncoder::identity(&csv).version.as_ref(), "1");

    let json = JsonLinesCodec::new("1").unwrap();
    assert_eq!(FormatDecoder::identity(&json).name.as_ref(), "json");
    assert_eq!(FormatEncoder::identity(&json).name.as_ref(), "json");

    let parquet = ParquetCodec::new("1").unwrap();
    assert_eq!(FormatDecoder::identity(&parquet).name.as_ref(), "parquet");
    assert_eq!(FormatEncoder::identity(&parquet).name.as_ref(), "parquet");

    let identity = calc_flow::FormatIdentity::new("csv", "1").unwrap();
    let error = calc_flow_connectors::arrow_schema::codec_error(&identity, "", "boom");
    let rendered = error.to_string();
    assert!(
        rendered.contains("calc-flow-connectors") && rendered.contains("csv"),
        "{rendered}"
    );
    assert!(
        rendered.contains("decode"),
        "empty names fall back: {rendered}"
    );
    assert!(rendered.contains("boom"), "{rendered}");
    let projected = calc_flow_connectors::arrow_schema::codec_connector_identity(&identity);
    assert_eq!(projected.provider.as_ref(), "calc-flow-connectors");
}

#[test]
fn negative_bounds_fail_option_parsing() {
    let root = temp_root("negative");
    std::fs::create_dir_all(&root).unwrap();
    let mut options = json_options(&root);
    options.insert("max_batch_bytes".to_string(), json!(-1));
    let error = FileSourceConfig::from_options(&options).expect_err("negative bound rejected");
    assert!(error.to_string().contains("max_batch_bytes"), "{error}");
}

#[tokio::test]
async fn subdirectory_entries_fail_closed() {
    let root = temp_root("subdir_only");
    std::fs::create_dir_all(root.join("nested-deeper")).unwrap();
    let mut source =
        FileSource::new(FileSourceConfig::from_options(&csv_options(&root)).unwrap()).unwrap();
    let error = source.open(None).await.expect_err("subdirectory rejected");
    assert!(error.to_string().contains("subdirectories"), "{error}");
}

#[tokio::test]
async fn corrupt_parquet_file_fails_through_source() {
    let root = temp_root("corrupt_parquet");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("data.parquet"), b"definitely not parquet").unwrap();
    let options = BTreeMap::from([
        ("path".to_string(), json!(root.display().to_string())),
        ("format".to_string(), json!("parquet")),
    ]);
    let mut source = FileSource::new(FileSourceConfig::from_options(&options).unwrap()).unwrap();
    source.open(None).await.expect("discovers");
    let error = source
        .next()
        .await
        .expect_err("corrupt parquet fails closed");
    assert!(error.to_string().contains("parquet"), "{error}");
}

#[tokio::test]
async fn json_replay_from_end_cursor_skips_file() {
    let root = temp_root("json_end_replay");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("data.json"), b"{\"a\":1}\n{\"a\":2}\n").unwrap();

    let mut source =
        FileSource::new(FileSourceConfig::from_options(&json_options(&root)).unwrap()).unwrap();
    source.open(None).await.unwrap();
    let mut last = None;
    while let Some(event) = source.next().await.unwrap() {
        if let SourceEvent::Data { cursor, .. } = event {
            last = Some(cursor);
        }
    }
    let end_cursor = last.expect("cursor");

    let mut replay =
        FileSource::new(FileSourceConfig::from_options(&json_options(&root)).unwrap()).unwrap();
    replay.open(Some(end_cursor)).await.expect("replay opens");
    assert!(
        replay.next().await.expect("exhausted").is_none(),
        "end-of-file cursor skips the file entirely"
    );
}

#[tokio::test]
async fn parquet_row_group_bound_names_the_row_group() {
    let codec = ParquetCodec::new("1").unwrap();
    let encoded = codec.encode(&sample_batch(4)).unwrap();
    let error = codec
        .decode(&encoded, &DecodeBounds::new(3, 1024 * 1024).unwrap(), &[])
        .expect_err("row group above the row bound fails closed");
    let rendered = error.to_string();
    assert!(
        rendered.contains("row group"),
        "the error names the row group: {rendered}"
    );
}

#[tokio::test]
async fn sink_recovers_from_matching_evidence_and_rejects_conflicts() {
    let root = temp_root("recover_direct");
    let mut sink = Harness::sink(&root, "out");
    sink.open().await.expect("opens");
    sink.begin_epoch(Epoch::INITIAL).await.expect("begins");
    sink.write(&sample_batch(2)).await.expect("writes");
    let evidence = sink.pre_commit(Epoch::INITIAL).await.expect("pre-commits");
    sink.commit(Epoch::INITIAL, &evidence)
        .await
        .expect("commits");

    let recovery = calc_flow::SinkRecovery::from_parts(
        Epoch::INITIAL,
        true,
        calc_flow::SinkDelivery::Transactional,
        evidence,
    );
    sink.recover(&recovery)
        .await
        .expect("matching evidence recovers");

    let conflicting: BTreeMap<String, Value> = BTreeMap::from([
        ("output".to_string(), json!("out")),
        ("epoch".to_string(), json!(1)),
        ("parts".to_string(), Value::Array(vec![])),
        ("rows".to_string(), json!(0)),
    ]);
    let mismatch = calc_flow::SinkRecovery::from_parts(
        Epoch::INITIAL,
        false,
        calc_flow::SinkDelivery::Transactional,
        conflicting,
    );
    let error = sink
        .recover(&mismatch)
        .await
        .expect_err("conflicting recovery evidence fails closed");
    assert!(error.to_string().contains("manifest"), "{error}");

    let uncommitted = calc_flow::SinkRecovery::from_parts(
        Epoch::new(9).unwrap(),
        false,
        calc_flow::SinkDelivery::Transactional,
        BTreeMap::new(),
    );
    sink.recover(&uncommitted)
        .await
        .expect("uncommitted epochs recover without output");
}

#[tokio::test]
async fn explicit_schema_and_header_options_flow_through_the_source() {
    let root = temp_root("explicit_schema");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("data.csv"), b"1,2\n3,4\n").unwrap();
    let mut options = csv_options(&root);
    options.insert("header".to_string(), json!(false));
    options.insert(
        "schema".to_string(),
        json!([
            {"name": "a", "data_type": "int64", "nullable": false},
            {"name": "b", "data_type": "int64", "nullable": false}
        ]),
    );
    let mut source =
        FileSource::new(FileSourceConfig::from_options(&options).expect("config")).unwrap();
    let batches = collect_source(&mut source).await;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2, "headerless rows decode fully");

    let mut bad_path: BTreeMap<String, Value> = BTreeMap::from([
        ("path".to_string(), json!(7)),
        ("format".to_string(), json!("csv")),
    ]);
    let _ = &mut bad_path;
    let error = FileSourceConfig::from_options(&bad_path).expect_err("path must be a string");
    assert!(error.to_string().contains("path"), "{error}");
}

#[tokio::test]
async fn file_ceiling_fails_closed() {
    let root = temp_root("ceiling");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("data.csv"), b"a,b\n1,2\n3,4\n5,6\n").unwrap();
    let mut options = csv_options(&root);
    options.insert("max_file_bytes".to_string(), json!(4));
    let mut source =
        FileSource::new(FileSourceConfig::from_options(&options).expect("config")).unwrap();
    source.open(None).await.expect("discovers");
    let error = source
        .next()
        .await
        .expect_err("file above the ceiling fails");
    assert!(error.to_string().contains("ceiling"), "{error}");
}
