//! Exploratory probe for the `parallelism_mode="auto_rows"` policy.
//!
//! Measures the same 2M-row batch SQL shapes under four partition policies:
//! fixed p1, fixed p4, `auto` p4 (single-input graphs fall back to p1 without
//! entity metadata), and `auto_rows` p4 (rows alone select four partitions).
//! Outputs are compared across configurations before timings are reported.
//! Set `AUTO_ROWS_PROBE_PARTITIONS` to change the requested partition count
//! (default four) and run under `taskset` to control worker capacity.
//!
//! Run with:
//! ```bash
//! cargo run -p calc-flow --release --example auto_rows_probe
//! ```

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use calc_flow::{
    Batch, BatchMetadata, DataFusionConfig, DataFusionParallelismMode, ExecutionOptions,
    PipelineBuilder, SqlOperator, UdfRegistry,
};
use datafusion::arrow::{
    array::{Array, Int64Array},
    record_batch::RecordBatch,
};

type ProbeResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

const ROWS: usize = 2_000_000;
const BATCHES: usize = 244;
const SAMPLES: usize = 7;
const KEYS: usize = 64;

#[derive(Clone, Copy)]
enum Shape {
    Projection,
    GroupBy,
    Join,
}

impl Shape {
    const fn name(self) -> &'static str {
        match self {
            Self::Projection => "projection",
            Self::GroupBy => "group_by",
            Self::Join => "join",
        }
    }

    const fn query(self) -> &'static str {
        match self {
            Self::Projection => "SELECT value + 1 AS result FROM input WHERE value >= 0",
            Self::GroupBy => "SELECT key, SUM(value) AS total FROM input GROUP BY key",
            Self::Join => {
                "SELECT facts.key, SUM(facts.value) AS total FROM facts \
                 JOIN dimension ON facts.key = dimension.key GROUP BY facts.key"
            }
        }
    }

    const fn aliases(self) -> &'static [&'static str] {
        match self {
            Self::Join => &["facts", "dimension"],
            Self::Projection | Self::GroupBy => &["input"],
        }
    }
}

fn record(rows: usize, start: i64) -> ProbeResult<RecordBatch> {
    let keys = Arc::new(Int64Array::from_iter_values(
        (0..rows).map(|row| i64::try_from(row % KEYS).unwrap()),
    )) as Arc<dyn Array>;
    let values = Arc::new(Int64Array::from_iter_values(
        (0..rows).map(|row| i64::try_from(row).unwrap() + start),
    )) as Arc<dyn Array>;
    Ok(RecordBatch::try_from_iter(vec![
        ("key", keys),
        ("value", values),
    ])?)
}

fn batch(rows: usize, batches: usize) -> ProbeResult<Batch> {
    let width = rows / batches;
    (0..batches)
        .map(|index| record(width, i64::try_from(index * width).unwrap()))
        .collect::<ProbeResult<Vec<_>>>()
        .map(|records| Batch::table(records, BatchMetadata::default()).unwrap())
}

fn config(mode: DataFusionParallelismMode, partitions: usize) -> DataFusionConfig {
    DataFusionConfig {
        target_partitions: partitions,
        parallelism_mode: mode,
        max_partitions: partitions,
        min_rows_per_partition: 65_536,
        small_rows_threshold: 10_001,
        batch_size: 8_192,
        ..DataFusionConfig::default()
    }
}

fn plan(shape: Shape, config: DataFusionConfig) -> ProbeResult<calc_flow::BatchExecutionPlan> {
    let mut builder = PipelineBuilder::new(shape.name())?.with_datafusion_config(config);
    builder = builder.add_node(
        "query",
        Box::new(SqlOperator::new(
            "query",
            shape.query(),
            shape
                .aliases()
                .iter()
                .map(|alias| (*alias).to_owned())
                .collect(),
            Vec::new(),
        )?),
    )?;
    Ok(builder.compile_batch(&UdfRegistry::new().snapshot())?)
}

fn batch_with_entities(rows: usize, batches: usize, entities: usize) -> ProbeResult<Batch> {
    let batch = batch(rows, batches)?;
    let metadata = batch
        .metadata()
        .clone()
        .with_datafusion_active_entities(entities)?;
    Ok(batch.with_metadata(metadata))
}

fn inputs(
    plan: &calc_flow::BatchExecutionPlan,
    entities: Option<usize>,
) -> ProbeResult<BTreeMap<String, Batch>> {
    let facts = match entities {
        Some(entities) => batch_with_entities(ROWS, BATCHES, entities)?,
        None => batch(ROWS, BATCHES)?,
    };
    let dimension = batch(KEYS, 1)?;
    plan.external_inputs()
        .iter()
        .map(|(name, port)| {
            let value = match port.port.as_str() {
                "facts" | "input" => facts.clone(),
                "dimension" => dimension.clone(),
                other => return Err(format!("unexpected input {other}").into()),
            };
            Ok((name.clone(), value))
        })
        .collect()
}

fn output_rows(batch: &Batch) -> ProbeResult<Vec<Vec<i64>>> {
    let table = batch.table_payload()?;
    let mut rows = table
        .batches()
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(move |row| {
                (0..record.num_columns())
                    .map(|column| {
                        let values = record
                            .column(column)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .expect("integer probe output")
                            .values();
                        values[row]
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    rows.sort_unstable();
    Ok(rows)
}

async fn measure(
    shape: Shape,
    mode: DataFusionParallelismMode,
    partitions: usize,
    entities: Option<usize>,
    expected: &mut Option<Vec<Vec<i64>>>,
) -> ProbeResult<()> {
    let plan = plan(shape, config(mode, partitions))?;
    let inputs = inputs(&plan, entities)?;
    let mut durations = Vec::with_capacity(SAMPLES);
    let mut effective = 0;
    let mut output_partitions = 0;
    for _ in 0..SAMPLES {
        let started = Instant::now();
        let result = plan
            .execute(inputs.clone(), ExecutionOptions::default())
            .await?;
        durations.push(started.elapsed().as_secs_f64() * 1_000.0);
        let metric = &result.datafusion_metrics[0];
        effective = metric.effective_target_partitions;
        output_partitions = metric.output_partition_count;
        let rows = output_rows(&result.outputs["output"])?;
        match expected {
            Some(expected) => assert_eq!(
                &rows,
                expected,
                "probe outputs differ for {mode:?} p{partitions} on {}",
                shape.name()
            ),
            None => *expected = Some(rows),
        }
    }
    durations.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let entity_label = entities.map_or_else(|| "none".to_owned(), |value| value.to_string());
    println!(
        "query={} mode={mode:?} entities={entity_label} requested_partitions={partitions} effective_partitions={effective} output_partitions={output_partitions} median_ms={:.2}",
        shape.name(),
        durations[durations.len() / 2]
    );
    Ok(())
}

#[tokio::main]
async fn main() -> ProbeResult<()> {
    let request: usize = std::env::var("AUTO_ROWS_PROBE_PARTITIONS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(4);
    println!(
        "input_rows={ROWS} input_batches={BATCHES} samples={SAMPLES} requested_partitions={request}"
    );
    for shape in [Shape::Projection, Shape::GroupBy, Shape::Join] {
        let mut expected: Option<Vec<Vec<i64>>> = None;
        measure(
            shape,
            DataFusionParallelismMode::Fixed,
            1,
            None,
            &mut expected,
        )
        .await?;
        measure(
            shape,
            DataFusionParallelismMode::Fixed,
            request,
            None,
            &mut expected,
        )
        .await?;
        measure(
            shape,
            DataFusionParallelismMode::Auto,
            request,
            None,
            &mut expected,
        )
        .await?;
        measure(
            shape,
            DataFusionParallelismMode::AutoRows,
            request,
            None,
            &mut expected,
        )
        .await?;
        measure(
            shape,
            DataFusionParallelismMode::Auto,
            request,
            Some(KEYS),
            &mut expected,
        )
        .await?;
    }
    Ok(())
}
