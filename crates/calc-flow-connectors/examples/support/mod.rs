//! Shared project, fixtures, and managed lifecycle for connector examples.
#![allow(dead_code)]
use arrow::{
    array::{Float64Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchMetadata, ConnectorRegistry, DecodeBounds, DeliveryGuarantee, FormatDecoder,
    FormatEncoder, JobState, ManagedCheckpointRuntime, ProjectSpec, ProviderRegistry,
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
    let bytes = match format {
        "csv" => CsvCodec::new("1", true)?.encode(&batch)?,
        "json" => JsonLinesCodec::new("1")?.encode(&batch)?,
        "parquet" => ParquetCodec::new("1")?.encode(&batch)?,
        _ => return Err("unsupported sample format".into()),
    };
    tokio::fs::write(root.join(format!("orders.{format}")), bytes).await?;
    Ok(())
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
pub async fn check_totals(root: &Path) -> ExampleResult {
    let root = root.join("results");
    let rows = tokio::task::spawn_blocking(move || -> ExampleResult<Vec<(i64, f64)>> {
        let mut rows = Vec::new();
        for path in parquet_paths(&root)? {
            let batch = ParquetCodec::new("1")?.decode(
                &std::fs::read(path)?,
                &DecodeBounds::new(8192, 8 << 20)?,
                &[],
            )?;
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
    })
    .await??;
    assert_eq!(rows, [(1, 20.0), (2, 60.0)]);
    println!("Parquet totals: {rows:?}");
    Ok(())
}

pub async fn execute(
    root: &Path,
    source: Value,
    sink: Value,
    registry: &ConnectorRegistry,
    continuous: bool,
    expected: DeliveryGuarantee,
) -> ExampleResult {
    let project = project(root, source, sink)?;
    let plan = compile_stream_project(
        &project,
        &ProviderRegistry::default(),
        &UdfRegistry::new().snapshot(),
        &registry.snapshot(),
        &StreamRequirements::default(),
    )?;
    let (start, cleanup) = StreamingRunner::new(
        plan,
        BTreeMap::new(),
        BTreeMap::new(),
        ManagedCheckpointRuntime::new(root.join("state"))?,
    )?
    .start_with_cleanup();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(60);
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
    let result = tokio::time::timeout_at(deadline, async {
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
    })
    .await;
    let outcome = match result {
        Ok(outcome) => outcome,
        Err(error) => {
            job.cancel().await;
            drop(job);
            cleanup.await?;
            return Err(error.into());
        }
    };
    let status = job.status();
    drop(job);
    cleanup.await?;
    assert_eq!(outcome.state, JobState::Completed, "{outcome:?}");
    assert_eq!(
        status.sinks.values().map(|s| s.delivered_rows).sum::<u64>(),
        2
    );
    assert_eq!(status.delivery["output"].effective, expected);
    println!("delivery: {:?}", status.delivery["output"]);
    Ok(())
}
pub async fn read_source(
    source: Value,
    registry: &ConnectorRegistry,
    continuous: bool,
    delivery: &str,
) -> ExampleResult {
    let root = tempfile::tempdir()?;
    let expected = if delivery == "at_least_once" {
        DeliveryGuarantee::AtLeastOnce
    } else {
        DeliveryGuarantee::BestEffort
    };
    execute(
        root.path(),
        source,
        file_sink(root.path(), delivery),
        registry,
        continuous,
        expected,
    )
    .await?;
    check_totals(root.path()).await
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
    let delivery = if name == "kafka" {
        "exactly_once"
    } else {
        "at_least_once"
    };
    let mut sink = json!({"binding":"output","connector":connector(name),"options":options,"delivery":delivery});
    if let Some(secrets) = secrets {
        sink["secrets"] = secrets;
    }
    if name == "kafka" {
        sink["format"] = json!({"name":"json","version":"1"});
    }
    execute(
        root.path(),
        file_source(root.path(), "parquet"),
        sink,
        registry,
        false,
        if name == "kafka" {
            DeliveryGuarantee::ExactlyOnce
        } else {
            DeliveryGuarantee::AtLeastOnce
        },
    )
    .await?;
    println!("Parquet -> {name}: wrote (id,total) = (1,20.0), (2,60.0)");
    Ok(())
}
