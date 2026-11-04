//! Exploratory probe for the input-partitioned `MemTable` experiment.
//!
//! This example is not a regression gate. It measures the same row-local and
//! grouped batch SQL shapes with 1, 4, 8, and 16 input partitions so the
//! experiment can decide whether input partitioning reaches the physical plan
//! and whether it changes end-to-end latency. Run it with:
//!
//! ```bash
//! cargo run -p calc-flow --release --features benchmark-internals \
//!   --example input_partitions_probe
//! ```
//!
//! Every configuration is checked against the single-partition result before
//! its timings are reported.

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use calc_flow::{
    Batch, BatchMetadata, DataFusionConfig, ExecutionOptions, PipelineBuilder, SqlOperator,
    UdfRegistry,
};
use datafusion::arrow::{
    array::{Array, Int64Array},
    record_batch::RecordBatch,
};

type ProbeResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

const ROWS: usize = 2_000_000;
const BATCHES: usize = 244;
const SAMPLES: usize = 7;
const PARTITIONS: [usize; 4] = [1, 4, 8, 16];
const QUERIES: [(&str, &str); 2] = [
    (
        "projection",
        "SELECT value + 1 AS result FROM input WHERE value >= 0",
    ),
    (
        "group_by",
        "SELECT key, SUM(value) AS total FROM input GROUP BY key",
    ),
];

fn input_batch() -> ProbeResult<Batch> {
    let width = ROWS / BATCHES;
    let mut records = Vec::with_capacity(BATCHES);
    for batch in 0..BATCHES {
        let start = i64::try_from(batch * width)?;
        let keys = Arc::new(Int64Array::from_iter_values(
            (0..width).map(|row| i64::try_from(row % 64).unwrap()),
        )) as Arc<dyn Array>;
        let values = Arc::new(Int64Array::from_iter_values(
            (0..width).map(|row| i64::try_from(row).unwrap() + start),
        )) as Arc<dyn Array>;
        records.push(RecordBatch::try_from_iter(vec![
            ("key", keys),
            ("value", values),
        ])?);
    }
    Ok(Batch::table(records, BatchMetadata::default())?)
}

fn plan(query: &str, input_partitions: usize) -> ProbeResult<calc_flow::BatchExecutionPlan> {
    Ok(PipelineBuilder::new("input partitions probe")?
        .with_datafusion_config(DataFusionConfig {
            target_partitions: 16,
            min_rows_per_partition: 65_536,
            small_rows_threshold: 10_001,
            batch_size: 8_192,
            ..DataFusionConfig::default()
        })
        .add_node(
            "query",
            Box::new(SqlOperator::new(
                "query",
                query,
                vec!["input".into()],
                Vec::new(),
            )?),
        )?
        .compile_batch(&UdfRegistry::new().snapshot())?
        .with_input_partitions(input_partitions)?)
}

fn result_values(output: &Batch) -> ProbeResult<Vec<i64>> {
    let table = output.table_payload()?;
    let column = table.batches()[0].num_columns() - 1;
    let mut values = table
        .batches()
        .iter()
        .flat_map(|batch| {
            batch
                .column(column)
                .as_any()
                .downcast_ref::<Int64Array>()
                .expect("integer probe output")
                .values()
                .to_vec()
        })
        .collect::<Vec<_>>();
    values.sort_unstable();
    Ok(values)
}

async fn measure_configuration(
    name: &str,
    query: &str,
    input: &Batch,
    input_partitions: usize,
    expected: &mut Option<Vec<i64>>,
) -> ProbeResult<()> {
    let plan = plan(query, input_partitions)?;
    let mut durations = Vec::with_capacity(SAMPLES);
    let mut output_partitions = 0;
    for _ in 0..SAMPLES {
        let started = Instant::now();
        let result = plan
            .execute(
                BTreeMap::from([("input".to_owned(), input.clone())]),
                ExecutionOptions::default(),
            )
            .await?;
        durations.push(started.elapsed().as_secs_f64() * 1_000.0);
        output_partitions = result.datafusion_metrics[0].output_partition_count;
        let values = result_values(&result.outputs["output"])?;
        match expected {
            Some(expected) => assert_eq!(
                &values, expected,
                "probe outputs differ for input_partitions={input_partitions}"
            ),
            None => *expected = Some(values),
        }
    }
    durations.sort_by(|left, right| left.partial_cmp(right).unwrap());
    println!(
        "query={name} input_partitions={input_partitions} output_partitions={output_partitions} median_ms={:.2}",
        durations[durations.len() / 2]
    );
    Ok(())
}

#[tokio::main]
async fn main() -> ProbeResult<()> {
    let input = input_batch()?;
    println!("input_rows={ROWS} input_batches={BATCHES} samples={SAMPLES}");
    for (name, query) in QUERIES {
        let mut expected: Option<Vec<i64>> = None;
        for input_partitions in PARTITIONS {
            measure_configuration(name, query, &input, input_partitions, &mut expected).await?;
        }
    }
    Ok(())
}
