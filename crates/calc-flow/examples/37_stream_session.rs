//! Push ordered rows and recover a caller-driven stream session.

use std::{collections::BTreeMap, error::Error, path::Path, sync::Arc};

use calc_flow::arrow::{
    array::{ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchMetadata, EventTime, ManagedCheckpointRuntime, PipelineBuilder, RollingOperator,
    RollingSpec, SessionOutput, StreamExecutionPlan, StreamRequirements, StreamRuntimeConfig,
    StreamSession, UdfRegistry,
};

const BASE_MICROS: i64 = 1_767_225_600_000_000;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("seq", DataType::UInt64, false),
        Field::new("price", DataType::Float64, false),
    ]))
}

fn rows(start: u64, prices: &[f64]) -> Result<Batch, Box<dyn Error>> {
    let first_micros = i64::try_from(start)?;
    let count = i64::try_from(prices.len())?;
    let end = start + u64::try_from(prices.len())?;
    let timestamps = (0..count)
        .map(|offset| BASE_MICROS + (first_micros + offset) * 1_000_000)
        .collect::<Vec<_>>();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(TimestampMicrosecondArray::from(timestamps).with_timezone("UTC")),
        Arc::new(StringArray::from(vec!["A"; prices.len()])),
        Arc::new(UInt64Array::from((start..end).collect::<Vec<_>>())),
        Arc::new(Float64Array::from(prices.to_vec())),
    ];
    let record = RecordBatch::try_new(schema(), columns)?;
    Ok(Batch::table(vec![record], BatchMetadata::default())?)
}

fn mean_values(
    output: &SessionOutput,
    output_id: &str,
) -> Result<Vec<Option<f64>>, Box<dyn Error>> {
    let mut values = Vec::new();
    for batch in output.batches(output_id) {
        for record in batch.table_payload()?.batches() {
            let index = record.schema().index_of("mean_price")?;
            let means = record
                .column(index)
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or("mean_price is not Float64")?;
            values.extend(means.iter());
        }
    }
    Ok(values)
}

fn build_plan() -> Result<StreamExecutionPlan, Box<dyn Error>> {
    let spec: RollingSpec = serde_json::from_value(serde_json::json!({
        "configuration_version": 1, "state_layout_version": 1,
        "partition_by": ["symbol"], "event_time": "ts", "sequence_by": ["seq"],
        "outputs": [{"kind": "mean", "primitive_version": 1, "input": "price",
            "output": "mean_price", "frame": {"kind": "rows", "size": 2}, "min_periods": 1}],
        "allowed_lateness_micros": 0, "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1",
    }))?;
    let rolling = RollingOperator::new("rolling", schema(), spec)?;
    Ok(PipelineBuilder::new("quotes")?
        .add_node("rolling", rolling)?
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )?)
}

fn push_quotes(
    session: &mut StreamSession,
) -> Result<(SessionOutput, SessionOutput), Box<dyn Error>> {
    let input = session.input_ids()[0].to_owned();
    let first = session.push(
        &input,
        rows(0, &[10.0, 12.0])?,
        Some(EventTime::from_micros(BASE_MICROS + 1_000_000)),
    )?;
    let second = session.push(&input, rows(2, &[15.0])?, None)?;
    Ok((first, second))
}

fn run_session() -> Result<(SessionOutput, SessionOutput, SessionOutput, String), Box<dyn Error>> {
    // Progress comes from caller-supplied watermarks and finish.
    let mut session = StreamSession::new(
        build_plan()?,
        &StreamRuntimeConfig::default(),
        BTreeMap::new(),
    )?;
    let output = session.output_ids()[0].to_owned();
    let (first, second) = push_quotes(&mut session)?;
    let final_output = session.finish()?;
    Ok((first, second, final_output, output))
}

fn checkpoint_buffered_quote(
    root: &Path,
    config: &StreamRuntimeConfig,
) -> Result<u64, Box<dyn Error>> {
    let mut session = StreamSession::new_with_checkpoints(
        build_plan()?,
        config,
        BTreeMap::new(),
        ManagedCheckpointRuntime::new(root)?,
    )?;
    push_quotes(&mut session)?;
    Ok(session.checkpoint()?.as_u64())
}

fn recover_buffered_quote() -> Result<(u64, Vec<Option<f64>>), Box<dyn Error>> {
    let directory = tempfile::tempdir()?;
    let root = directory.path().join("managed");
    let config = StreamRuntimeConfig {
        checkpointing: true,
        ..Default::default()
    };
    let epoch = checkpoint_buffered_quote(&root, &config)?;

    // Closing the first session leaves its checkpoint available at the same root.
    let mut restored = StreamSession::restore(
        build_plan()?,
        &config,
        BTreeMap::new(),
        ManagedCheckpointRuntime::new(&root)?,
    )?;
    let result = restored.finish()?;
    Ok((epoch, mean_values(&result, restored.output_ids()[0])?))
}

fn check_recovery(expected: &[Option<f64>]) -> Result<(), Box<dyn Error>> {
    let (epoch, restored) = recover_buffered_quote()?;
    if epoch != 1 || restored != expected {
        return Err(format!("unexpected recovery at epoch {epoch}: {restored:?}").into());
    }
    println!("checkpoint epoch: {epoch}");
    println!("restored finish: {restored:?}");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let (first, second, final_output, output) = run_session()?;
    let first = mean_values(&first, &output)?;
    let second = mean_values(&second, &output)?;
    let final_values = mean_values(&final_output, &output)?;
    let actual = (&first, &second, &final_values);
    let expected = (
        &vec![Some(10.0), Some(11.0)],
        &Vec::new(),
        &vec![Some(13.5)],
    );
    if actual != expected {
        return Err(format!("unexpected session means: {actual:?}").into());
    }
    println!("first push: {first:?}");
    println!("second push: {second:?}");
    println!("finish: {final_values:?}");

    check_recovery(&final_values)
}
