//! Shared project, fixtures, and managed lifecycle for connector examples.
#![allow(dead_code)]
use arrow::{
    array::{Float64Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchMetadata, ConnectorRegistry, DecodeBounds, DeliveryGuarantee, FormatDecoder,
    FormatEncoder, JobState, JobStatus, ManagedCheckpointRuntime, ProjectSpec, ProviderRegistry,
    StreamRequirements, StreamingRunner, UdfRegistry, compile_stream_project,
};
use calc_flow_connectors::{
    csv::CsvCodec, json_lines::JsonLinesCodec, parquet::ParquetCodec, register_file_connectors,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

pub type ExampleResult<T = ()> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub fn arguments() -> ExampleResult<Vec<String>> {
    // Discard argv[0]; validated demo options do not make security decisions.
    std::env::args_os() // nosemgrep: args-os
        .skip(1)
        .map(argument_text)
        .collect()
}
fn argument_text(value: std::ffi::OsString) -> ExampleResult<String> {
    value
        .into_string()
        .map_err(|_| "arguments must be UTF-8".into())
}
pub fn registry() -> calc_flow::Result<ConnectorRegistry> {
    let mut registry = ConnectorRegistry::new();
    register_file_connectors(&mut registry)?;
    Ok(registry)
}
pub fn connector(name: &str) -> Value {
    json!({"provider":"calc-flow-connectors","name":name,"version":if name=="mysql" {"1.0.0"}else{"2.0.0"}})
}
pub fn order_schema(nullable: bool) -> Value {
    json!([{"name":"id","data_type":"int64","nullable":nullable},{"name":"quantity","data_type":"int64","nullable":nullable},{"name":"price","data_type":"float64","nullable":nullable}])
}
pub fn secret(key: &str) -> Value {
    json!({"url":{"resolver":"environment","key":key}})
}
pub fn require_env(key: &str) -> ExampleResult {
    if std::env::var(key).is_ok_and(|v| !v.is_empty()) {
        Ok(())
    } else {
        Err(format!("Set {key}; see examples/README.md for service preparation").into())
    }
}
pub fn source(name: &str, options: Value, format: Option<&str>, secrets: Option<Value>) -> Value {
    let mut value =
        json!({"binding":"input","connector":connector(name),"watermark":{"policy":"disabled"}});
    value["options"] = options;
    if let Some(format) = format {
        value["format"] = json!({"name":format,"version":"1"});
    }
    if let Some(secrets) = secrets {
        value["secrets"] = secrets;
    }
    value
}
pub fn file_source(root: &Path, format: &str) -> Value {
    source(
        "file",
        json!({"path":root.join(format!("orders.{format}")),"format":format,"schema":order_schema(true),"header":true,"max_batch_rows":8192}),
        Some(format),
        None,
    )
}
pub fn file_sink(root: &Path, delivery: &str) -> Value {
    json!({"binding":"output","connector":connector("file"),"format":{"name":"parquet","version":"1"},"options":{"path":root,"output":"results"},"delivery":delivery})
}
pub fn project(root: &Path, source: Value, sink: Value) -> ExampleResult<ProjectSpec> {
    let mut document = json!({"format_version":3,"id":"order-totals","name":"order-totals","runtime":{"mode":"stream","options":{}},
        "graph":{"name":"order-totals","nodes":[{"id":"totals","operator":{"kind":"expression","select":["id","CAST(quantity AS DOUBLE) * price AS total"],"filter":"quantity > 0"}}]},
        "state":{"root":root.join("state"),"retention":2}
    });
    document["sources"] = Value::Array(vec![source]);
    document["sinks"] = Value::Array(vec![sink]);
    Ok(serde_json::from_value(document)?)
}

pub fn orders(include_zero: bool) -> calc_flow::Result<Batch> {
    let (ids, quantities, prices) = if include_zero {
        (vec![1, 2, 3], vec![2, 3, 0], vec![10.0, 20.0, 99.0])
    } else {
        (vec![1, 2], vec![2, 3], vec![10.0, 20.0])
    };
    let record = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, true),
            Field::new("quantity", DataType::Int64, true),
            Field::new("price", DataType::Float64, true),
        ])),
        vec![
            Arc::new(Int64Array::from(ids)),
            Arc::new(Int64Array::from(quantities)),
            Arc::new(Float64Array::from(prices)),
        ],
    )
    .expect("sample schema");
    Batch::table(vec![record], BatchMetadata::default())
}
pub async fn write_orders(root: &Path, format: &str, include_zero: bool) -> ExampleResult {
    tokio::fs::create_dir_all(root).await?;
    let batch = orders(include_zero)?;
    let bytes = encode_orders(format, &batch)?;
    tokio::fs::write(root.join(format!("orders.{format}")), bytes).await?;
    Ok(())
}
fn encode_orders(format: &str, batch: &Batch) -> ExampleResult<Vec<u8>> {
    let encoder: Box<dyn FormatEncoder> = match format {
        "csv" => Box::new(CsvCodec::new("1", true)?),
        "json" => Box::new(JsonLinesCodec::new("1")?),
        "parquet" => Box::new(ParquetCodec::new("1")?),
        _ => return Err("unsupported sample format".into()),
    };
    Ok(encoder.encode(batch)?)
}

fn parquet_paths(root: &Path) -> ExampleResult<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            paths.extend(parquet_paths(&path)?);
        } else if path.extension().is_some_and(|s| s == "parquet") {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}
pub async fn check_totals(root: &Path) -> ExampleResult<Vec<f64>> {
    let root = root.join("results");
    let rows = tokio::task::spawn_blocking(move || read_totals(&root)).await??;
    assert_eq!(rows, [(1, 20.0), (2, 60.0)]);
    Ok(rows.into_iter().map(|(_, total)| total).collect())
}

fn read_output(path: &Path) -> ExampleResult<Batch> {
    Ok(ParquetCodec::new("1")?.decode(
        &std::fs::read(path)?,
        &DecodeBounds::new(8192, 8 << 20)?,
        &[],
    )?)
}

fn read_totals(root: &Path) -> ExampleResult<Vec<(i64, f64)>> {
    let mut rows = Vec::new();
    for path in parquet_paths(root)? {
        let batch = read_output(&path)?;
        for record in batch.table_payload()?.batches() {
            let ids = record
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let totals = record
                .column_by_name("total")
                .unwrap()
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            rows.extend((0..record.num_rows()).map(|i| (ids.value(i), totals.value(i))));
        }
    }
    rows.sort_by_key(|r| r.0);
    Ok(rows)
}

pub async fn execute(
    root: &Path,
    source: Value,
    sink: Value,
    registry: &ConnectorRegistry,
    continuous: bool,
    expected: DeliveryGuarantee,
) -> ExampleResult<JobStatus> {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
    let runner = project_runner(root, source, sink, registry)?;
    let (job, cleanup) = start_before(runner, deadline).await?;
    let result = wait_before(&job, deadline, continuous).await;
    let status = job.status();
    drop(job);
    cleanup.await?;
    let outcome = result?;
    assert_eq!(outcome.state, JobState::Completed, "{outcome:?}");
    assert_eq!(
        status.sinks.values().map(|s| s.delivered_rows).sum::<u64>(),
        2
    );
    assert_eq!(status.delivery["output"].effective, expected);
    Ok(status)
}

fn project_runner(
    root: &Path,
    source: Value,
    sink: Value,
    registry: &ConnectorRegistry,
) -> ExampleResult<StreamingRunner> {
    let project = project(root, source, sink)?;
    let plan = compile_stream_project(
        &project,
        &ProviderRegistry::default(),
        &UdfRegistry::new().snapshot(),
        &registry.snapshot(),
        &StreamRequirements::default(),
    )?;
    Ok(StreamingRunner::new(
        plan,
        BTreeMap::new(),
        BTreeMap::new(),
        ManagedCheckpointRuntime::new(root.join("state"))?,
    )?)
}

async fn start_before(
    runner: StreamingRunner,
    deadline: tokio::time::Instant,
) -> ExampleResult<(
    calc_flow::StreamingJob,
    impl Future<Output = calc_flow::Result<()>>,
)> {
    let (start, cleanup) = runner.start_with_cleanup();
    let job = match tokio::time::timeout_at(deadline, start).await {
        Ok(Ok(job)) => job,
        Ok(Err(error)) => {
            cleanup.await?;
            return Err(error.into());
        }
        Err(error) => {
            cleanup.await?;
            return Err(error.into());
        }
    };
    Ok((job, cleanup))
}

async fn wait_for_sample(job: &calc_flow::StreamingJob, continuous: bool) -> calc_flow::JobOutcome {
    if continuous {
        loop {
            let status = job.status();
            if status.state != JobState::Running
                || status.sinks.values().map(|s| s.delivered_rows).sum::<u64>() >= 2
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        job.shutdown().await
    } else {
        job.wait().await
    }
}

async fn wait_before(
    job: &calc_flow::StreamingJob,
    deadline: tokio::time::Instant,
    continuous: bool,
) -> ExampleResult<calc_flow::JobOutcome> {
    match tokio::time::timeout_at(deadline, wait_for_sample(job, continuous)).await {
        Ok(outcome) => Ok(outcome),
        Err(error) => {
            job.cancel().await;
            Err(error.into())
        }
    }
}

pub async fn read_source(
    source: Value,
    registry: &ConnectorRegistry,
    continuous: bool,
    delivery: &str,
) -> ExampleResult {
    let root = tempfile::tempdir()?;
    let name = match source["format"]["name"].as_str() {
        Some("protobuf") => "kafka protobuf",
        Some("custom") => "kafka custom decoder",
        _ => source["connector"]["name"]
            .as_str()
            .ok_or("source connector name")?,
    }
    .to_owned();
    let expected = if delivery == "at_least_once" {
        DeliveryGuarantee::AtLeastOnce
    } else {
        DeliveryGuarantee::BestEffort
    };
    let status = execute(
        root.path(),
        source,
        file_sink(root.path(), delivery),
        registry,
        continuous,
        expected,
    )
    .await?;
    let totals = check_totals(root.path()).await?;
    println!("{name} -> Parquet: {totals:?}");
    println!(
        "delivery: {}",
        serde_json::to_string(&status.delivery["output"])?
    );
    Ok(())
}

pub async fn write_sink(
    name: &str,
    mut options: Value,
    secrets: Option<Value>,
    registry: &ConnectorRegistry,
) -> ExampleResult {
    let root = tempfile::tempdir()?;
    write_orders(root.path(), "parquet", true).await?;
    if name != "mysql" {
        options["pipeline"] = json!(root.path().file_name().unwrap().to_string_lossy());
        options["output"] = json!("totals");
    }
    let (delivery, expected) = sink_delivery(name);
    let mut sink = json!({"binding":"output","connector":connector(name),"options":options,"delivery":delivery});
    if let Some(secrets) = secrets {
        sink["secrets"] = secrets;
    }
    if name == "kafka" {
        sink["format"] = json!({"name":"json","version":"1"});
    }
    let status = execute(
        root.path(),
        file_source(root.path(), "parquet"),
        sink,
        registry,
        false,
        expected,
    )
    .await?;
    println!("Parquet -> {name}: wrote 2 rows; expected (id, total) = (1, 20.0), (2, 60.0)");
    println!(
        "delivery: {}",
        serde_json::to_string(&status.delivery["output"])?
    );
    Ok(())
}

fn sink_delivery(name: &str) -> (&'static str, DeliveryGuarantee) {
    if name == "kafka" {
        ("exactly_once", DeliveryGuarantee::ExactlyOnce)
    } else {
        ("at_least_once", DeliveryGuarantee::AtLeastOnce)
    }
}

#[cfg(test)]
mod tests {
    use super::argument_text;

    #[test]
    fn unicode_arguments_are_preserved() {
        assert_eq!(argument_text("计算".into()).unwrap(), "计算");
    }

    #[cfg(unix)]
    #[test]
    fn invalid_unicode_arguments_return_an_error() {
        use std::os::unix::ffi::OsStringExt;
        let value = std::ffi::OsString::from_vec(vec![0xff]);
        assert_eq!(
            argument_text(value).unwrap_err().to_string(),
            "arguments must be UTF-8"
        );
    }
}
