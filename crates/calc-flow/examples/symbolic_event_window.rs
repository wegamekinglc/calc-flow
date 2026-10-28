//! Python named example: shared minute-window state and derived price ranges.
mod support;
use calc_flow::{
    AggregateFunction, Batch, BatchMetadata, OperatorMetadata, PipelineBuilder,
    WindowAggregateOperator, WindowSpec,
};
use datafusion::arrow::{
    array::{Float64Array, Int64Array, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use std::{collections::BTreeMap, sync::Arc, time::Duration};
use support::{
    ExampleResult, ReplaySource, check_floats, collect, data, expression, f64_values, i64_values,
    schema, stream_plan, strings, table, uints, watermark,
};

fn trade_data() -> ExampleResult<Batch> {
    let input = table(vec![
        (
            "ts",
            Arc::new(
                TimestampMicrosecondArray::from(vec![
                    Some(5_000_000),
                    Some(35_000_000),
                    Some(59_000_000),
                    Some(15_000_000),
                    Some(60_000_000),
                    Some(95_000_000),
                    None,
                ])
                .with_timezone("UTC"),
            ),
        ),
        ("symbol", strings(&["A", "A", "A", "B", "A", "A", "A"])),
        ("trade_id", uints(&[1, 2, 3, 4, 5, 6, 7])),
        (
            "quantity",
            Arc::new(Int64Array::from(vec![
                Some(10),
                None,
                Some(30),
                None,
                Some(7),
                Some(13),
                Some(999),
            ])),
        ),
        (
            "price",
            Arc::new(Float64Array::from(vec![
                Some(100.0),
                Some(102.0),
                None,
                None,
                Some(110.0),
                Some(90.0),
                Some(999.0),
            ])),
        ),
    ]);
    let record = &input.table_payload()?.batches()[0];
    let mut fields = record.schema().fields().to_vec();
    fields[1] = Arc::new(Field::new("symbol", DataType::Utf8, true));
    Ok(Batch::table(
        vec![RecordBatch::try_new(
            Arc::new(Schema::new(fields)),
            record.columns().to_vec(),
        )?],
        BatchMetadata::default(),
    )?)
}

#[tokio::main]
async fn main() -> ExampleResult {
    let input = trade_data()?;
    let plan = stream_plan(program(schema(&input))?)?;
    let source = ReplaySource::new(vec![
        data(input, 1)?,
        watermark(60_000_000),
        watermark(120_000_000),
    ]);
    let outputs = collect(plan, BTreeMap::from([("input".into(), source.binding())])).await?;
    check_outputs(&outputs)
}

fn check_schema_and_rows(minute: &[Batch]) -> ExampleResult {
    let expected_schema = Schema::new(vec![
        Field::new(
            "window_start",
            DataType::Timestamp(
                datafusion::arrow::datatypes::TimeUnit::Microsecond,
                Some("UTC".into()),
            ),
            false,
        ),
        Field::new(
            "window_end",
            DataType::Timestamp(
                datafusion::arrow::datatypes::TimeUnit::Microsecond,
                Some("UTC".into()),
            ),
            false,
        ),
        Field::new("symbol", DataType::Utf8, true),
        Field::new("trade_count", DataType::UInt64, false),
        Field::new("volume", DataType::Int64, true),
        Field::new("low", DataType::Float64, true),
        Field::new("high", DataType::Float64, true),
        Field::new("avg_price", DataType::Float64, true),
    ]);
    assert_eq!(schema(&minute[0]).as_ref(), &expected_schema);
    let expected = RecordBatch::try_new(
        Arc::new(expected_schema),
        vec![
            support::times(&[0, 0, 60_000_000]),
            support::times(&[60_000_000, 60_000_000, 120_000_000]),
            strings(&["A", "B", "A"]),
            uints(&[3, 1, 2]),
            Arc::new(Int64Array::from(vec![Some(40), None, Some(20)])),
            Arc::new(Float64Array::from(vec![Some(100.0), None, Some(90.0)])),
            Arc::new(Float64Array::from(vec![Some(102.0), None, Some(110.0)])),
            Arc::new(Float64Array::from(vec![Some(101.0), None, Some(100.0)])),
        ],
    )?;
    let actual = datafusion::arrow::compute::concat_batches(
        &expected.schema(),
        minute
            .iter()
            .flat_map(|b| b.table_payload().unwrap().batches()),
    )?;
    assert_eq!(actual, expected);
    Ok(())
}

fn window_operator(
    schema: datafusion::arrow::datatypes::SchemaRef,
) -> calc_flow::Result<WindowAggregateOperator> {
    let spec = WindowSpec::tumbling("ts", Duration::from_secs(60))?
        .group_by(["symbol"])?
        .aggregate(AggregateFunction::Count, "trade_id", "trade_count")?
        .aggregate(AggregateFunction::Sum, "quantity", "volume")?
        .aggregate(AggregateFunction::Min, "price", "low")?
        .aggregate(AggregateFunction::Max, "price", "high")?
        .aggregate(AggregateFunction::Avg, "price", "avg_price")?;
    WindowAggregateOperator::new("window", schema, spec)
}

fn branch_nodes(
    window: WindowAggregateOperator,
) -> calc_flow::Result<Vec<(&'static str, calc_flow::NodeOperator)>> {
    let schema = window.output_ports()[0].schema().unwrap().clone();
    let minute = support::expression_schemas(
        expression("minute", "", &["*"], None)?,
        Some(schema.clone()),
        None,
    )?;
    let summary = support::expression_schemas(
        expression("summary", "price_range = high - low", &[], None)?,
        Some(schema),
        None,
    )?;
    Ok(vec![
        ("window", window.into()),
        ("minute", minute.into()),
        ("summary", summary.into()),
    ])
}

fn program(schema: datafusion::arrow::datatypes::SchemaRef) -> calc_flow::Result<PipelineBuilder> {
    let nodes = branch_nodes(window_operator(schema)?)?;
    let builder = nodes.into_iter().try_fold(
        PipelineBuilder::new("minute-bars")?,
        |builder, (id, node)| builder.add_node(id, node),
    )?;
    support::connect_all(
        builder,
        &[
            ("window", "output", "minute", "input"),
            ("window", "output", "summary", "input"),
        ],
    )
}

fn check_outputs(outputs: &BTreeMap<String, Vec<Batch>>) -> ExampleResult {
    let minute = &outputs["minute.output"];
    let summary = &outputs["summary.output"];
    assert_eq!(i64_values(minute, "volume"), [Some(40), None, Some(20)]);
    check_floats(
        &f64_values(minute, "avg_price"),
        &[Some(101.0), None, Some(100.0)],
    );
    assert_eq!(
        f64_values(summary, "price_range"),
        [Some(2.0), None, Some(20.0)]
    );
    check_schema_and_rows(minute)?;
    support::print(minute)?;
    support::print(summary)
}
