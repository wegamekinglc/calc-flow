//! Python 31: compact positions/latest quotes feed complete-replacement SQL.
#[path = "support/stock_account_data.rs"]
mod stock_account_data;
mod support;
use calc_flow::{
    BatchKind, EdgeBudget, NodeOperator, OperatorMetadata, PipelineBuilder, Port, SinkBinding,
    SnapshotSqlOperator, SourceBinding, StateBudget, StreamRuntimeConfig, StreamingRunner,
    WatermarkPolicy,
};
use datafusion::arrow::datatypes::{DataType, Field};
use std::{collections::BTreeMap, sync::Arc, time::Instant};
use stock_account_data::{DemoSource, LastSink, Latest, Options, check_valuation};
use support::{ExampleResult, best_effort_plan, expression, sql};

fn port(name: &str, fields: Vec<Field>) -> calc_flow::Result<Port> {
    Port::new(name, BatchKind::Table, true, Some(fields))
}

#[tokio::main]
async fn main() -> ExampleResult {
    let options = Options::parse(true)?;
    let started_at = Instant::now();
    let last: Latest = Arc::default();
    let job = configured_runner(options, &last)?.start().await?;
    let outcome = job.wait().await;
    assert_eq!(outcome.state, calc_flow::JobState::Completed, "{outcome:?}");
    let last = last.lock().await;
    let result = last.as_ref().ok_or("no valuation snapshots")?;
    let elapsed = started_at.elapsed();
    check_valuation(result, options)?;
    support::print(std::slice::from_ref(result))?;
    println!("total elapsed: {:.3} s", elapsed.as_secs_f64());
    Ok(())
}

fn runtime_config() -> calc_flow::Result<StreamRuntimeConfig> {
    Ok(StreamRuntimeConfig {
        checkpointing: false,
        edge_budget: EdgeBudget::new(100_000, 64 << 20)?,
        sql_compact_state_budget: Some(StateBudget::new(100_000, 64 << 20)?),
        sql_snapshot_state_budget: Some(StateBudget::new(200_000, 128 << 20)?),
        ..StreamRuntimeConfig::default()
    })
}

fn configured_runner(options: Options, last: &Latest) -> calc_flow::Result<StreamingRunner> {
    let source = |s| {
        SourceBinding::new(s)
            .with_watermark_policy(WatermarkPolicy::Disabled { idle_timeout: None })
    };
    let sources = BTreeMap::from([
        ("deltas.input".into(), source(DemoSource::trades(options))),
        ("latest.input".into(), source(DemoSource::quotes(options))),
    ]);
    let sinks = BTreeMap::from([(
        "output".into(),
        vec![SinkBinding::ordinary(
            "market-values",
            LastSink(Arc::clone(last)),
        )?],
    )]);
    StreamingRunner::without_checkpoints(best_effort_plan(graph()?)?, sources, sinks)?
        .with_runtime_config(runtime_config()?)
}

fn position_fields() -> Vec<Field> {
    vec![
        Field::new("account", DataType::Utf8, true),
        Field::new("stock", DataType::Utf8, true),
        Field::new("position", DataType::Int64, true),
    ]
}
fn quote_fields() -> Vec<Field> {
    vec![
        Field::new("stock", DataType::Utf8, true),
        Field::new("latest_price", DataType::Float64, true),
    ]
}

fn position_operator() -> calc_flow::Result<calc_flow::SqlOperator> {
    let operator = sql(
        "positions",
        "SELECT account, stock, SUM(delta) AS position FROM input GROUP BY account, stock",
        &["input"],
    )?;
    (*operator).with_ports(
        vec![port(
            "input",
            vec![
                Field::new("account", DataType::Utf8, true),
                Field::new("stock", DataType::Utf8, true),
                Field::new("delta", DataType::Int64, true),
            ],
        )?],
        port("output", position_fields())?,
    )
}

fn quote_operator() -> calc_flow::Result<calc_flow::SqlOperator> {
    let operator = sql(
        "latest",
        "SELECT stock, LAST_VALUE(latest_price ORDER BY \"timestamp\") AS latest_price FROM input GROUP BY stock",
        &["input"],
    )?;
    (*operator).with_ports(
        vec![port(
            "input",
            stock_account_data::quote_schema()
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .collect(),
        )?],
        port("output", quote_fields())?,
    )
}

fn snapshot_operator() -> calc_flow::Result<SnapshotSqlOperator> {
    SnapshotSqlOperator::new("values","SELECT p.account, SUM(p.position * q.latest_price) AS market_value FROM p LEFT JOIN q ON p.stock = q.stock GROUP BY p.account ORDER BY p.account",vec!["p".into(),"q".into()],vec![])?
        .with_ports(vec![port("p",position_fields())?,port("q",quote_fields())?],port("output",vec![Field::new("account",DataType::Utf8,true),Field::new("market_value",DataType::Float64,true)])?)
}

fn graph() -> calc_flow::Result<PipelineBuilder> {
    let positions = position_operator()?;
    let quotes = quote_operator()?;
    let values = snapshot_operator()?;
    let delta_schema = positions.input_ports()[0].schema().unwrap().clone();
    let deltas = support::expression_schemas(
        expression(
            "deltas",
            "",
            &[
                "account",
                "stock",
                "CASE WHEN side = 'buy' THEN quantity ELSE -quantity END AS delta",
            ],
            Some("flow_type = 'trade'"),
        )?,
        None,
        Some(delta_schema),
    )?;
    let nodes = vec![
        ("deltas", deltas.into()),
        ("positions", Box::new(positions).into()),
        ("latest", Box::new(quotes).into()),
        ("values", NodeOperator::SnapshotSql(values)),
    ];
    let builder = nodes.into_iter().try_fold(
        PipelineBuilder::new("account-market-values")?,
        |builder, (id, node)| builder.add_node(id, node),
    )?;
    support::connect_all(
        builder,
        &[
            ("deltas", "output", "positions", "input"),
            ("positions", "output", "values", "p"),
            ("latest", "output", "values", "q"),
        ],
    )
}
