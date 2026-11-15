//! Push ordered rows through one stateful table session.

use std::{collections::BTreeMap, error::Error, sync::Arc};

use calc_flow::arrow::{
    array::{ArrayRef, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchMetadata, EventTime, PipelineBuilder, RollingOperator, RollingSpec, SessionOutput,
    StreamRequirements, StreamRuntimeConfig, StreamSession, UdfRegistry,
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
    let timestamps = (0..prices.len())
        .map(|offset| BASE_MICROS + (start as i64 + offset as i64) * 1_000_000)
        .collect::<Vec<_>>();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(TimestampMicrosecondArray::from(timestamps).with_timezone("UTC")),
        Arc::new(StringArray::from(vec!["A"; prices.len()])),
        Arc::new(UInt64Array::from(
            (start..start + prices.len() as u64).collect::<Vec<_>>(),
        )),
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

fn main() -> Result<(), Box<dyn Error>> {
    let spec: RollingSpec = serde_json::from_value(serde_json::json!({
        "configuration_version": 1, "state_layout_version": 1,
        "partition_by": ["symbol"], "event_time": "ts", "sequence_by": ["seq"],
        "outputs": [{"kind": "mean", "primitive_version": 1, "input": "price",
            "output": "mean_price", "frame": {"kind": "rows", "size": 2}, "min_periods": 1}],
        "allowed_lateness_micros": 0, "late_policy": {"kind": "error", "scope": "envelope"},
        "value_policy": "stateful_numeric_v1",
    }))?;
    let rolling = RollingOperator::new("rolling", schema(), spec)?;
    let plan = PipelineBuilder::new("quotes")?
        .add_node("rolling", rolling)?
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )?;

    // No generated policy: progress comes only from explicit watermarks and finish.
    let mut session = StreamSession::new(plan, &StreamRuntimeConfig::default(), BTreeMap::new())?;
    let (input, output) = (
        session.input_ids()[0].to_owned(),
        session.output_ids()[0].to_owned(),
    );
    let first = session.push(
        &input,
        rows(0, &[10.0, 12.0])?,
        Some(EventTime::from_micros(BASE_MICROS + 1_000_000)),
    )?;
    let second = session.push(&input, rows(2, &[15.0])?, None)?;
    let final_output = session.finish()?;

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
    Ok(())
}
