//! Python 12: bounded event-time join followed by an expression stage.
mod support;
use calc_flow::{
    JoinStateLimits, JoinTimeBounds, PipelineBuilder, StreamJoinOperator, StreamJoinSpec,
};
use std::{collections::BTreeMap, time::Duration};
use support::{
    ExampleResult, ReplaySource, collect, data, edge, expression, floats, i64_values, ints, schema,
    stream_plan, table, times, uints, watermark,
};

#[tokio::main]
async fn main() -> ExampleResult {
    let left = table(vec![
        ("account_id", ints(&[1, 2, 3])),
        (
            "authorized_at",
            times(&[10_000_000, 20_000_000, 30_000_000]),
        ),
        ("sequence", uints(&[1, 1, 1])),
        ("amount", floats(&[100.0, 50.0, 25.0])),
    ]);
    let right = table(vec![
        ("account_id", ints(&[1, 2, 1])),
        ("paid_at", times(&[12_000_000, 16_000_000, 20_000_000])),
        ("sequence", uints(&[1, 1, 2])),
        ("paid_amount", floats(&[100.0, 45.0, 100.0])),
    ]);
    let spec = StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "authorized_at",
        "paid_at",
        JoinTimeBounds::new(Duration::from_secs(5), Duration::from_secs(2))?,
        JoinStateLimits::new(10_000, 16 << 20, 100_000)?,
    )?
    .with_prefixes("authorization", "payment")?;
    let join = StreamJoinOperator::new("join", schema(&left), schema(&right), spec)?;
    let output_schema = calc_flow::OperatorMetadata::output_ports(&join)[0]
        .schema()
        .unwrap()
        .clone();
    let plan = stream_plan(
        PipelineBuilder::new("symbolic-stream-join")?
            .add_node("join", join)?
            .add_node(
                "matches",
                support::expression_schemas(
                    expression(
                        "matches",
                        "amount_matches = authorization__amount = payment__paid_amount",
                        &[],
                        None,
                    )?,
                    Some(output_schema),
                    None,
                )?,
            )?
            .connect(edge("join", "output", "matches", "input")?)?,
    )?;
    let left_batches = left.table_payload()?.batches()[0].clone();
    let right_batches = right.table_payload()?.batches()[0].clone();
    let to_batch =
        |record| calc_flow::Batch::table(vec![record], calc_flow::BatchMetadata::default());
    let left_events = vec![
        data(to_batch(left_batches.slice(0, 1))?, 1)?,
        data(to_batch(left_batches.slice(1, 1))?, 2)?,
        data(to_batch(left_batches.slice(2, 1))?, 3)?,
        watermark(60_000_000),
    ];
    let right_events = vec![
        data(to_batch(right_batches.slice(0, 2))?, 1)?,
        data(to_batch(right_batches.slice(2, 1))?, 2)?,
        watermark(60_000_000),
    ];
    let results = collect(
        plan,
        BTreeMap::from([
            ("left".into(), ReplaySource::new(left_events).binding()),
            ("right".into(), ReplaySource::new(right_events).binding()),
        ]),
    )
    .await?;
    let mut ids = i64_values(&results["output"], "authorization__account_id");
    ids.sort();
    assert_eq!(ids, [Some(1), Some(2)]);
    let matches: Vec<_> = results["output"]
        .iter()
        .flat_map(|b| b.table_payload().unwrap().batches())
        .flat_map(|b| {
            b.column_by_name("amount_matches")
                .unwrap()
                .as_any()
                .downcast_ref::<datafusion::arrow::array::BooleanArray>()
                .unwrap()
                .iter()
        })
        .collect();
    assert_eq!(matches.iter().filter(|v| **v == Some(true)).count(), 1);
    assert_eq!(matches.iter().filter(|v| **v == Some(false)).count(), 1);
    support::print(&results["output"])
}
