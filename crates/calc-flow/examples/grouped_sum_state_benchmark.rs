//! Focused before/after evidence for grouped integer SUM state processing.
//!
//! See `benchmarks/sum-state/README.md` for the paired collection contract.

#[path = "support/stock_account_data.rs"]
mod stock_account_data;
mod support;

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, Cursor, DataFusionConfig, DataFusionParallelismMode, ExecutionOptions,
    PipelineBuilder, SinkBinding, SourceBinding, SourceCapabilities, SourceEvent, StateBudget,
    StreamRuntimeConfig, StreamSink, StreamSource, StreamingRunner, UdfRegistry, WatermarkPolicy,
};
use datafusion::arrow::{
    array::{Array, Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    ipc::{reader::StreamReader, writer::StreamWriter},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Cursor as IoCursor, path::Path, sync::Arc, time::Instant};
use stock_account_data::{DemoSource, Options, trade_schema};
use support::ExampleResult;
use tokio::sync::Mutex;

type Positions = BTreeMap<(String, String), i64>;
type Snapshots = Arc<Mutex<Vec<Batch>>>;

fn options(rows: u64, accounts: u32, batch_size: u32) -> Options {
    Options {
        rows,
        accounts,
        stocks: 100,
        batch_size,
        quote_rows: 1,
    }
}

async fn generated(options: Options) -> ExampleResult<Vec<Batch>> {
    let mut source = DemoSource::trades(options);
    source.open(None).await?;
    let mut batches = Vec::new();
    while let Some(event) = source.next().await? {
        let SourceEvent::Data { batch, .. } = event else {
            return Err("unexpected source control".into());
        };
        batches.push(batch);
    }
    source.close().await?;
    Ok(batches)
}

async fn generated_fixtures() -> ExampleResult<[(&'static str, Vec<Batch>); 2]> {
    let dense = generated(options(2_000_000, 100, 100_000)).await?;
    let mut sparse = generated(options(0, 100, 10_000)).await?;
    // All 10k groups exist; only 100 groups receive subsequent updates.
    sparse.extend(generated(options(2_000, 1, 100)).await?.into_iter().skip(1));
    Ok([("stock", dense), ("sparse", sparse)])
}

async fn prepare(directory: &Path) -> ExampleResult {
    tokio::fs::create_dir(directory).await?;
    for (name, batches) in generated_fixtures().await? {
        let path = directory.join(format!("{name}.arrow"));
        let report =
            tokio::task::spawn_blocking(move || fixture_report(name, &path, &batches)).await??;
        println!("{report}");
    }
    Ok(())
}

fn write_fixture_batches(
    writer: &mut StreamWriter<std::fs::File>,
    batches: &[Batch],
) -> ExampleResult {
    for batch in batches {
        for record in batch.table_payload()?.batches() {
            writer.write(record)?;
        }
    }
    Ok(())
}

struct FixtureReference {
    sha256: Vec<String>,
    rows: Vec<usize>,
}

fn fixture_reference(batches: &[Batch]) -> ExampleResult<FixtureReference> {
    let mut expected = Positions::new();
    let mut reference = FixtureReference {
        sha256: Vec::new(),
        rows: Vec::new(),
    };
    for batch in batches {
        update_reference(&mut expected, batch)?;
        reference.sha256.push(position_hash(&expected)?);
        reference.rows.push(expected.len());
    }
    Ok(reference)
}

fn position_schema() -> Schema {
    Schema::new(vec![
        Field::new("account", DataType::Utf8, true),
        Field::new("stock", DataType::Utf8, true),
        Field::new("position", DataType::Int64, true),
    ])
}

fn fixture_report(name: &str, path: &Path, batches: &[Batch]) -> ExampleResult<Value> {
    let mut writer = StreamWriter::try_new(std::fs::File::create(path)?, &trade_schema())?;
    write_fixture_batches(&mut writer, batches)?;
    writer.finish()?;
    let bytes = std::fs::read(path)?;
    let reference = fixture_reference(batches)?;
    Ok(json!({
        "snapshot_sha256": reference.sha256,
        "snapshot_rows": reference.rows,
        "snapshot_schema_sha256": schema_hash(&position_schema()),
        "fixture": name,
        "input_sha256": hex::encode(Sha256::digest(bytes)),
        "input_rows": batches.iter().map(Batch::num_rows).sum::<usize>(),
        "batch_rows": batches.iter().map(Batch::num_rows).collect::<Vec<_>>(),
        "schema_sha256": schema_hash(&trade_schema()),
    }))
}

fn schema_hash(schema: &Schema) -> String {
    let fields = schema
        .fields()
        .iter()
        .map(|f| (f.name(), format!("{:?}", f.data_type()), f.is_nullable()))
        .collect::<Vec<_>>();
    hex::encode(Sha256::digest(serde_json::to_vec(&fields).unwrap()))
}

struct Preloaded {
    batches: Arc<[Batch]>,
    offset: usize,
    capabilities: SourceCapabilities,
}

#[async_trait]
impl StreamSource for Preloaded {
    fn capabilities(&self) -> SourceCapabilities {
        self.capabilities.clone()
    }
    async fn open(&mut self, cursor: Option<Cursor>) -> calc_flow::Result<()> {
        assert!(cursor.is_none());
        self.offset = 0;
        Ok(())
    }
    async fn next(&mut self) -> calc_flow::Result<Option<SourceEvent>> {
        let Some(batch) = self.batches.get(self.offset).cloned() else {
            return Ok(None);
        };
        self.offset += 1;
        Ok(Some(SourceEvent::Data {
            batch,
            cursor: Cursor::unbound(self.offset.to_be_bytes().to_vec(), BTreeMap::new())?,
        }))
    }
    async fn close(&mut self) -> calc_flow::Result<()> {
        self.offset = self.batches.len();
        Ok(())
    }
}

struct SnapshotSink(Snapshots);
#[async_trait]
impl StreamSink for SnapshotSink {
    async fn open(&mut self) -> calc_flow::Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> calc_flow::Result<()> {
        self.0.lock().await.push(batch.clone());
        Ok(())
    }
    async fn close(&mut self) -> calc_flow::Result<()> {
        Ok(())
    }
}

fn graph(partitions: usize) -> calc_flow::Result<PipelineBuilder> {
    let config = DataFusionConfig {
        target_partitions: partitions,
        min_rows_per_partition: 1,
        parallelism_mode: DataFusionParallelismMode::Fixed,
        ..DataFusionConfig::default()
    };
    support::connect_all(PipelineBuilder::new("sum-state-evidence")?.with_datafusion_config(config)
        .add_node("deltas",support::expression("deltas","",&["account","stock","CASE WHEN side = 'buy' THEN quantity ELSE -quantity END AS delta"],Some("flow_type = 'trade'"))?)?
        .add_node("positions",support::sql("positions","SELECT account, stock, SUM(delta) AS position FROM input GROUP BY account, stock",&["input"])?)?,&[("deltas","output","positions","input")])
}

fn update_reference(expected: &mut Positions, batch: &Batch) -> ExampleResult {
    for record in batch.table_payload()?.batches() {
        let strings = |name: &str| {
            record
                .column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
        };
        let account = strings("account");
        let stock = strings("stock");
        let side = strings("side");
        let flow = strings("flow_type");
        let quantity = record
            .column_by_name("quantity")
            .unwrap()
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for row in 0..record.num_rows() {
            assert!(record.columns().iter().all(|a| !a.is_null(row)));
            if flow.value(row) == "trade" {
                let delta = if side.value(row) == "buy" {
                    quantity.value(row)
                } else {
                    -quantity.value(row)
                };
                *expected
                    .entry((account.value(row).to_owned(), stock.value(row).to_owned()))
                    .or_default() += delta;
            }
        }
    }
    Ok(())
}

fn position_hash(positions: &Positions) -> ExampleResult<String> {
    let rows = positions
        .iter()
        .map(|((a, s), v)| (a, s, v))
        .collect::<Vec<_>>();
    Ok(hex::encode(Sha256::digest(serde_json::to_vec(&rows)?)))
}

fn validate_output(batch: &Batch, expected: &Positions) -> ExampleResult<(String, String)> {
    let table = batch.table_payload()?;
    assert_eq!(
        table
            .schema()
            .fields()
            .iter()
            .map(|f| (f.name().as_str(), f.data_type().clone()))
            .collect::<Vec<_>>(),
        vec![
            ("account", DataType::Utf8),
            ("stock", DataType::Utf8),
            ("position", DataType::Int64)
        ]
    );
    let mut actual = Positions::new();
    for record in table.batches() {
        let accounts = record
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let stocks = record
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let values = record
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for row in 0..record.num_rows() {
            assert!(record.columns().iter().all(|array| !array.is_null(row)));
            assert!(
                actual
                    .insert(
                        (accounts.value(row).to_owned(), stocks.value(row).to_owned()),
                        values.value(row)
                    )
                    .is_none(),
                "duplicate output key"
            );
        }
    }
    assert_eq!(&actual, expected, "complete per-snapshot arithmetic oracle");
    Ok((position_hash(&actual)?, schema_hash(table.schema())))
}

#[cfg(target_os = "linux")]
fn clock_ns(cpu: bool) -> i64 {
    use nix::time::{ClockId, clock_gettime};
    let value = clock_gettime(if cpu {
        ClockId::CLOCK_PROCESS_CPUTIME_ID
    } else {
        ClockId::CLOCK_MONOTONIC
    })
    .unwrap();
    value.tv_sec() * 1_000_000_000 + value.tv_nsec()
}
#[cfg(not(target_os = "linux"))]
fn clock_ns(_: bool) -> i64 {
    panic!("this measurement requires Linux clocks")
}

fn stream_config() -> ExampleResult<StreamRuntimeConfig> {
    Ok(StreamRuntimeConfig {
        sql_compact_state_budget: Some(StateBudget::new(
            10_000,
            StateBudget::default().max_bytes(),
        )?),
        ..options(0, 100, 100_000).runtime_config()?
    })
}

fn stream_runner(
    batches: &Arc<[Batch]>,
    state: &Snapshots,
    partitions: usize,
) -> ExampleResult<StreamingRunner> {
    let source = Preloaded {
        batches: Arc::clone(batches),
        offset: 0,
        capabilities: DemoSource::trades(options(0, 100, 100_000)).capabilities(),
    };
    let config = stream_config()?;
    let runner = StreamingRunner::without_checkpoints(
        support::best_effort_plan(graph(partitions)?)?,
        BTreeMap::from([(
            "input".into(),
            SourceBinding::new(source)
                .with_watermark_policy(WatermarkPolicy::Disabled { idle_timeout: None }),
        )]),
        BTreeMap::from([(
            "output".into(),
            vec![SinkBinding::ordinary(
                "snapshots",
                SnapshotSink(Arc::clone(state)),
            )?],
        )]),
    )?;
    Ok(runner.with_runtime_config(config)?)
}

async fn execute_stream(
    batches: &Arc<[Batch]>,
    partitions: usize,
) -> ExampleResult<(Vec<Batch>, Value)> {
    let state: Snapshots = Arc::default();
    let job = stream_runner(batches, &state, partitions)?.start().await?;
    let outcome = job.wait().await;
    assert_eq!(outcome.state, calc_flow::JobState::Completed, "{outcome:?}");
    let status = job.status();
    assert_eq!(status.task_count, 0);
    let snapshots = std::mem::take(&mut *state.lock().await);
    Ok((snapshots, serde_json::to_value(status)?))
}

fn batch_input(batches: &[Batch]) -> calc_flow::Result<Batch> {
    Batch::table(
        batches
            .iter()
            .flat_map(|b| b.table_payload().unwrap().batches().iter().cloned())
            .collect(),
        BatchMetadata::default(),
    )
}

async fn execute_batch(batches: &[Batch], partitions: usize) -> ExampleResult<(Vec<Batch>, Value)> {
    let input = batch_input(batches)?;
    let run = graph(partitions)?
        .compile_batch(&UdfRegistry::new().snapshot())?
        .execute(
            BTreeMap::from([("input".into(), input)]),
            ExecutionOptions::default(),
        )
        .await?;
    Ok((
        vec![run.outputs["output"].clone()],
        json!({"queries":run.datafusion_metrics,"nodes":run.node_timings}),
    ))
}

async fn execute_sample(
    batches: &Arc<[Batch]>,
    mode: &str,
    partitions: usize,
) -> ExampleResult<(Vec<Batch>, Value)> {
    if mode == "stream" {
        execute_stream(batches, partitions).await
    } else {
        execute_batch(batches, partitions).await
    }
}

#[derive(Default)]
struct ValidatedSnapshots {
    sha256: Vec<String>,
    schema_sha256: Vec<String>,
}

fn validate_stream_snapshots(
    batches: &[Batch],
    output: &[Batch],
) -> ExampleResult<ValidatedSnapshots> {
    assert_eq!(output.len(), batches.len());
    let mut expected = Positions::new();
    let mut validated = ValidatedSnapshots::default();
    for (input, snapshot) in batches.iter().zip(output) {
        update_reference(&mut expected, input)?;
        let (digest, schema) = validate_output(snapshot, &expected)?;
        validated.sha256.push(digest);
        validated.schema_sha256.push(schema);
    }
    Ok(validated)
}

fn validate_batch_snapshot(
    batches: &[Batch],
    output: &[Batch],
) -> ExampleResult<ValidatedSnapshots> {
    let mut expected = Positions::new();
    for input in batches {
        update_reference(&mut expected, input)?;
    }
    let (digest, schema) = validate_output(&output[0], &expected)?;
    Ok(ValidatedSnapshots {
        sha256: vec![digest],
        schema_sha256: vec![schema],
    })
}

fn validate_snapshots(
    batches: &[Batch],
    output: &[Batch],
    mode: &str,
) -> ExampleResult<ValidatedSnapshots> {
    if mode == "stream" {
        validate_stream_snapshots(batches, output)
    } else {
        validate_batch_snapshot(batches, output)
    }
}

async fn sample(
    batches: Arc<[Batch]>,
    input_hash: &str,
    mode: &str,
    partitions: usize,
    index: usize,
) -> ExampleResult<Value> {
    let cpu_start = clock_ns(true);
    let start_ns = clock_ns(false);
    let started = Instant::now();
    let (output, diagnostics) = execute_sample(&batches, mode, partitions).await?;
    let seconds = started.elapsed().as_secs_f64();
    let end_ns = clock_ns(false);
    let cpu_ns = clock_ns(true) - cpu_start;
    let validated = validate_snapshots(&batches, &output, mode)?;
    Ok(json!({
        "process_observations": calc_flow::benchmark_internals::process_observations()?,
        "schema": "calc-flow.sum-state.v1",
        "mode": mode,
        "partitions": partitions,
        "sample": index,
        "pid": std::process::id(),
        "seconds": seconds,
        "process_cpu_ns": cpu_ns,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "input_sha256": input_hash,
        "input_schema_sha256": schema_hash(&trade_schema()),
        "input_rows": batches.iter().map(Batch::num_rows).sum::<usize>(),
        "batch_rows": batches.iter().map(Batch::num_rows).collect::<Vec<_>>(),
        "snapshot_rows": output.iter().map(Batch::num_rows).collect::<Vec<_>>(),
        "snapshot_sha256": validated.sha256,
        "snapshot_schema_sha256": validated.schema_sha256,
        "validated_all_snapshots": true,
        "diagnostics": diagnostics,
    }))
}

struct SampleRequest<'a> {
    input: &'a str,
    mode: &'a str,
    partitions: usize,
    repeats: usize,
}

fn sample_request(args: &[String]) -> ExampleResult<SampleRequest<'_>> {
    if args.len() != 4 {
        return Err("INPUT.arrow stream|batch PARTITIONS REPEATS".into());
    }
    let mode = args[1].as_str();
    assert!(matches!(mode, "stream" | "batch"));
    let partitions = args[2].parse::<usize>()?;
    let repeats = args[3].parse::<usize>()?;
    assert!(partitions > 0 && repeats > 0);
    Ok(SampleRequest {
        input: &args[0],
        mode,
        partitions,
        repeats,
    })
}

struct LoadedInput {
    batches: Arc<[Batch]>,
    sha256: String,
}

async fn load_input(path: &str) -> ExampleResult<LoadedInput> {
    let bytes = tokio::fs::read(path).await?;
    let sha256 = hex::encode(Sha256::digest(&bytes));
    let reader = StreamReader::try_new(IoCursor::new(bytes), None)?;
    assert_eq!(reader.schema(), trade_schema());
    let batches = reader
        .map(|r| Batch::table(vec![r.unwrap()], BatchMetadata::default()).unwrap())
        .collect::<Vec<_>>()
        .into();
    Ok(LoadedInput { batches, sha256 })
}

async fn run_samples(request: SampleRequest<'_>) -> ExampleResult {
    let input = load_input(request.input).await?;
    for index in 0..request.repeats {
        println!(
            "{}",
            sample(
                Arc::clone(&input.batches),
                &input.sha256,
                request.mode,
                request.partitions,
                index
            )
            .await?
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> ExampleResult {
    let args = support::arguments()?;
    if args.first().map(String::as_str) == Some("--prepare") {
        return prepare(Path::new(args.get(1).ok_or("--prepare directory")?)).await;
    }
    run_samples(sample_request(&args)?).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_evidence_fixture_report_and_input_round_trip() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("small.arrow");
        let batches = generated(options(10, 1, 100)).await.unwrap();
        let fixture_path = path.clone();
        let report =
            tokio::task::spawn_blocking(move || fixture_report("small", &fixture_path, &batches))
                .await
                .unwrap()
                .unwrap();
        let loaded = load_input(path.to_str().unwrap()).await.unwrap();
        assert_eq!(report["fixture"], "small");
        assert_eq!(report["input_rows"], 110);
        assert_eq!(report["batch_rows"], json!([100, 10]));
        assert_eq!(report["snapshot_rows"], json!([100, 100]));
        assert_eq!(
            report["snapshot_sha256"],
            json!([
                "20d9043ab930d576cf6058bc58643415bff68c0ada8145fd7bb792f3058ff63e",
                "cf3f1be7cadc06df5d94095a3a3a1309e67b48c87e495876867cb64ad64638b0",
            ])
        );
        assert_eq!(report["input_sha256"], loaded.sha256);
        assert_eq!(
            loaded
                .batches
                .iter()
                .map(Batch::num_rows)
                .collect::<Vec<_>>(),
            [100, 10]
        );
        let loaded_reference = fixture_reference(&loaded.batches).unwrap();
        assert_eq!(report["snapshot_sha256"], json!(loaded_reference.sha256));
        assert_eq!(
            report["snapshot_schema_sha256"],
            schema_hash(&position_schema())
        );
    }

    #[test]
    fn test_evidence_sample_request_keeps_cli_contract() {
        let args = ["input.arrow", "stream", "8", "6"].map(str::to_owned);
        let request = sample_request(&args).unwrap();
        assert_eq!(request.input, "input.arrow");
        assert_eq!(request.mode, "stream");
        assert_eq!(request.partitions, 8);
        assert_eq!(request.repeats, 6);
        assert!(sample_request(&args[..3]).is_err());
        let invalid_number = ["input.arrow", "batch", "many", "1"].map(str::to_owned);
        assert!(sample_request(&invalid_number).is_err());
    }
}
