//! Python 09: compose native rolling, cross-section, and expression stages.
mod support;
use calc_flow::{
    CrossSectionOperator, CrossSectionSpec, ExecutionOptions, OperatorMetadata, PipelineBuilder,
    RollingOperator, RollingSpec, UdfRegistry,
};
use serde_json::json;
use std::collections::BTreeMap;
use support::{
    ExampleResult, check_floats, expression, f64_values, floats, schema, sql, strings, table, times,
};

#[tokio::main]
async fn main() -> ExampleResult {
    let quotes = table(vec![
        ("ts", times(&[1_000_000, 1_000_000, 2_000_000, 2_000_000])),
        ("symbol", strings(&["AAA", "BBB", "AAA", "BBB"])),
        ("price", floats(&[100.0, 50.0, 102.0, 48.0])),
        ("volume", floats(&[10.0, 20.0, 30.0, 15.0])),
    ]);
    let plan = financial_graph(&schema(&quotes))?.compile_batch(&UdfRegistry::new().snapshot())?;
    let run = plan
        .execute(
            BTreeMap::from([("input".into(), quotes)]),
            ExecutionOptions::default(),
        )
        .await?;
    let output = [run.outputs["output"].clone()];
    assert_eq!(output[0].num_rows(), 4);
    assert_eq!(schema(&output[0]).fields().len(), 16);
    check_floats(
        &f64_values(&output, "momentum_1"),
        &[None, None, Some(0.02), Some(-0.04)],
    );
    check_floats(
        &f64_values(&output, "price_mean_3"),
        &[Some(100.0), Some(50.0), Some(101.0), Some(49.0)],
    );
    check_floats(
        &f64_values(&output, "volume_z"),
        &[Some(-1.0), Some(1.0), Some(1.0), Some(-1.0)],
    );
    support::print(&output)
}

fn financial_graph(
    input_schema: &datafusion::arrow::datatypes::Schema,
) -> calc_flow::Result<PipelineBuilder> {
    let (rolling, means, cross) = native_stages(input_schema)?;
    let nodes = graph_nodes(rolling, means, cross)?;
    let builder = nodes.into_iter().try_fold(
        PipelineBuilder::new("financial-features")?,
        |builder, (id, node)| builder.add_node(id, node),
    )?;
    support::connect_all(
        builder,
        &[
            ("log", "output", "rolling", "input"),
            ("rolling", "output", "gains", "input"),
            ("gains", "output", "means", "input"),
            ("means", "output", "cross", "input"),
            ("cross", "output", "signals", "input"),
        ],
    )
}

fn with_fields(
    schema: &datafusion::arrow::datatypes::Schema,
    names: &[&str],
) -> datafusion::arrow::datatypes::SchemaRef {
    let mut fields = schema.fields().to_vec();
    fields.extend(names.iter().map(|name| {
        std::sync::Arc::new(datafusion::arrow::datatypes::Field::new(
            *name,
            datafusion::arrow::datatypes::DataType::Float64,
            true,
        ))
    }));
    std::sync::Arc::new(datafusion::arrow::datatypes::Schema::new(fields))
}

fn native_stages(
    input_schema: &datafusion::arrow::datatypes::Schema,
) -> calc_flow::Result<(RollingOperator, RollingOperator, CrossSectionOperator)> {
    let rolling = rolling_stage(with_fields(input_schema, &["log_price"]))?;
    let means = mean_stage(with_fields(
        rolling.output_ports()[0].schema().unwrap(),
        &["gain", "loss"],
    ))?;
    let cross = cross_stage(means.output_ports()[0].schema().unwrap().clone())?;
    Ok((rolling, means, cross))
}

fn rolling_stage(
    schema: datafusion::arrow::datatypes::SchemaRef,
) -> calc_flow::Result<RollingOperator> {
    let rolling_spec: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":2,"partition_by":["symbol"],"event_time":"ts","sequence_by":["ts"],
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},"value_policy":"stateful_numeric_v1",
        "outputs":[
            {"kind":"lag","primitive_version":1,"input":"price","output":"previous","periods":1},
            {"kind":"lag","primitive_version":1,"input":"log_price","output":"previous_log_price","periods":1},
            {"kind":"delta","primitive_version":1,"input":"price","output":"change","periods":1},
            {"kind":"mean","primitive_version":1,"input":"price","output":"price_mean_3","frame":{"kind":"rows","size":3},"min_periods":1},
            {"kind":"stddev","primitive_version":1,"input":"price","output":"price_stddev_3","frame":{"kind":"rows","size":3},"min_periods":2,"ddof":0},
            {"kind":"ewma","primitive_version":1,"input":"price","output":"price_ema_3","span":3,"min_periods":1},
            {"kind":"ewma","primitive_version":1,"input":"price","output":"fast","span":2,"min_periods":1},
            {"kind":"ewma","primitive_version":1,"input":"price","output":"slow","span":4,"min_periods":1}
        ]
    })).map_err(|e|support::invalid("spec",e.to_string()))?;
    RollingOperator::new("rolling", schema, rolling_spec)
}

fn mean_stage(
    schema: datafusion::arrow::datatypes::SchemaRef,
) -> calc_flow::Result<RollingOperator> {
    let mean_spec: RollingSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":1,"partition_by":["symbol"],"event_time":"ts","sequence_by":["ts"],
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},"value_policy":"stateful_numeric_v1",
        "outputs":[
            {"kind":"mean","primitive_version":1,"input":"gain","output":"average_gain","frame":{"kind":"rows","size":3},"min_periods":1},
            {"kind":"mean","primitive_version":1,"input":"loss","output":"average_loss","frame":{"kind":"rows","size":3},"min_periods":1}
        ]
    })).map_err(|e|support::invalid("spec",e.to_string()))?;
    RollingOperator::new("means", schema, mean_spec)
}

fn cross_stage(
    schema: datafusion::arrow::datatypes::SchemaRef,
) -> calc_flow::Result<CrossSectionOperator> {
    let cross_spec: CrossSectionSpec = serde_json::from_value(json!({
        "configuration_version":1,"state_layout_version":1,"event_time":"ts","entity_by":["symbol"],"partition_by":[],"sequence_by":["ts"],"grouping":{"kind":"exact_time"},
        "allowed_lateness_micros":0,"late_policy":{"kind":"error","scope":"envelope"},"value_policy":"nan_exclude_preserve_v1",
        "outputs":[{"kind":"zscore","primitive_version":1,"input":"volume","output":"volume_z","min_samples":2,"ddof":0}]
    })).map_err(|e|support::invalid("spec",e.to_string()))?;
    CrossSectionOperator::new("cross", schema, cross_spec)
}

fn graph_nodes(
    rolling: RollingOperator,
    means: RollingOperator,
    cross: CrossSectionOperator,
) -> calc_flow::Result<Vec<(&'static str, calc_flow::NodeOperator)>> {
    let log_schema = rolling.input_ports()[0].schema().unwrap().clone();
    let rolling_schema = rolling.output_ports()[0].schema().unwrap().clone();
    let gain_schema = means.input_ports()[0].schema().unwrap().clone();
    let cross_schema = cross.output_ports()[0].schema().unwrap().clone();
    let log = support::expression_schemas(
        expression("log", "", &["*", "LN(price) AS log_price"], None)?,
        None,
        Some(log_schema),
    )?;
    let gains = support::expression_schemas(
        expression(
            "gains",
            "",
            &[
                "*",
                "CASE WHEN change IS NULL THEN NULL ELSE LEAST(GREATEST(change,0.0),1e100) END AS gain",
                "CASE WHEN change IS NULL THEN NULL ELSE LEAST(GREATEST(-change,0.0),1e100) END AS loss",
            ],
            None,
        )?,
        Some(rolling_schema),
        Some(gain_schema),
    )?;
    let signals = support::sql_input_schema(sql("signals", SIGNAL_SQL, &["input"])?, cross_schema)?;
    Ok(vec![
        ("log", log.into()),
        ("rolling", rolling.into()),
        ("gains", gains.into()),
        ("means", means.into()),
        ("cross", cross.into()),
        ("signals", signals.into()),
    ])
}

const SIGNAL_SQL: &str = "SELECT ts, symbol, price, volume, price / previous - 1.0 AS momentum_1, previous_log_price, LN(price / previous) AS log_return_1, price_mean_3, price_ema_3, fast - slow AS price_macd_2_4, price_stddev_3, price_mean_3 + 2.0 * price_stddev_3 AS bollinger_upper_3, price_mean_3 - 2.0 * price_stddev_3 AS bollinger_lower_3, 100.0 - 100.0 / (1.0 + average_gain / average_loss) AS rsi_3, volume_z, price / previous - 1.0 - volume_z * 0.01 AS liquidity_adjusted_momentum FROM input ORDER BY ts, symbol";
