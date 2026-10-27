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
use std::{collections::BTreeMap, sync::Arc};
use stock_account_data::{DemoSource, LastSink, Latest, Options, check_valuation};
use support::{ExampleResult, best_effort_plan, edge, expression, sql};

fn port(name: &str, fields: Vec<Field>) -> calc_flow::Result<Port> {
    Port::new(name, BatchKind::Table, true, Some(fields))
}
#[tokio::main]
async fn main() -> ExampleResult {
    let options = Options::parse(true)?;
    let plan = graph()?;
    let source = |s| {
        SourceBinding::new(s)
            .with_watermark_policy(WatermarkPolicy::Disabled { idle_timeout: None })
    };
    let sources = BTreeMap::from([
        ("deltas.input".into(), source(DemoSource::trades(options))),
        ("latest.input".into(), source(DemoSource::quotes(options))),
    ]);
    let last: Latest = Arc::default();
    let job = StreamingRunner::without_checkpoints(
        plan,
        sources,
        BTreeMap::from([(
            "output".into(),
            vec![SinkBinding::ordinary(
                "market-values",
                LastSink(Arc::clone(&last)),
            )?],
        )]),
    )?
    .with_runtime_config(StreamRuntimeConfig {
        checkpointing: false,
        edge_budget: EdgeBudget::new(100_000, 64 << 20)?,
        sql_compact_state_budget: Some(StateBudget::new(100_000, 64 << 20)?),
        sql_snapshot_state_budget: Some(StateBudget::new(200_000, 128 << 20)?),
        ..StreamRuntimeConfig::default()
    })?
    .start()
    .await?;
    let outcome = job.wait().await;
    assert_eq!(outcome.state, calc_flow::JobState::Completed, "{outcome:?}");
    let last = last.lock().await;
    let result = last.as_ref().ok_or("no valuation snapshots")?;
    check_valuation(result, options)?;
    support::print(std::slice::from_ref(result))
}

fn graph() -> calc_flow::Result<calc_flow::StreamExecutionPlan> {
    let position_fields = vec![
        Field::new("account", DataType::Utf8, true),
        Field::new("stock", DataType::Utf8, true),
        Field::new("position", DataType::Int64, true),
    ];
    let quote_fields = vec![
        Field::new("stock", DataType::Utf8, true),
        Field::new("latest_price", DataType::Float64, true),
    ];
    let positions = sql(
        "positions",
        "SELECT account, stock, SUM(delta) AS position FROM input GROUP BY account, stock",
        &["input"],
    )?;
    let positions = (*positions).with_ports(
        vec![port(
            "input",
            vec![
                Field::new("account", DataType::Utf8, true),
                Field::new("stock", DataType::Utf8, true),
                Field::new("delta", DataType::Int64, true),
            ],
        )?],
        port("output", position_fields.clone())?,
    )?;
    let quotes = sql(
        "latest",
        "SELECT stock, LAST_VALUE(latest_price ORDER BY \"timestamp\") AS latest_price FROM input GROUP BY stock",
        &["input"],
    )?;
    let quotes = (*quotes).with_ports(
        vec![port(
            "input",
            stock_account_data::quote_schema()
                .fields()
                .iter()
                .map(|f| f.as_ref().clone())
                .collect(),
        )?],
        port("output", quote_fields.clone())?,
    )?;
    let values=SnapshotSqlOperator::new("values","SELECT p.account, SUM(p.position * q.latest_price) AS market_value FROM p LEFT JOIN q ON p.stock = q.stock GROUP BY p.account ORDER BY p.account",vec!["p".into(),"q".into()],vec![])?
        .with_ports(vec![port("p",position_fields)?,port("q",quote_fields)?],port("output",vec![Field::new("account",DataType::Utf8,true),Field::new("market_value",DataType::Float64,true)])?)?;
    let delta_schema = positions.input_ports()[0].schema().unwrap().clone();
    best_effort_plan(
        PipelineBuilder::new("account-market-values")?
            .add_node(
                "deltas",
                support::expression_schemas(
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
                )?,
            )?
            .add_node("positions", Box::new(positions))?
            .add_node("latest", Box::new(quotes))?
            .add_node("values", NodeOperator::SnapshotSql(values))?
            .connect(edge("deltas", "output", "positions", "input")?)?
            .connect(edge("positions", "output", "values", "p")?)?
            .connect(edge("latest", "output", "values", "q")?)?,
    )
}
