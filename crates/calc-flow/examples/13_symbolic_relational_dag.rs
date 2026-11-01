//! Python 13: compose two native bounded joins through exact intermediate schemas.
mod support;
use calc_flow::{
    JoinStateLimits, JoinTimeBounds, OperatorMetadata, PipelineBuilder, StreamJoinOperator,
    StreamJoinSpec,
};
use std::{collections::BTreeMap, time::Duration};
use support::{
    ExampleResult, ReplaySource, collect, data, expression, f64_values, floats, ints, schema,
    stream_plan, table, times, uints, watermark,
};

fn input(time: &str, value: &str, seconds: i64, amount: f64) -> calc_flow::Batch {
    table(vec![
        ("account_id", ints(&[7])),
        (time, times(&[seconds * 1_000_000])),
        ("sequence", uints(&[1])),
        (value, floats(&[amount])),
    ])
}
fn spec(
    left_key: &str,
    left_time: &str,
    right_time: &str,
    left_prefix: &str,
    right_prefix: &str,
) -> calc_flow::Result<StreamJoinSpec> {
    StreamJoinSpec::inner(
        [left_key],
        ["account_id"],
        left_time,
        right_time,
        JoinTimeBounds::new(Duration::from_secs(5), Duration::from_secs(2))?,
        JoinStateLimits::new(10_000, 16 << 20, 100_000)?,
    )?
    .with_prefixes(left_prefix, right_prefix)
}
#[tokio::main]
async fn main() -> ExampleResult {
    let auth = input("authorized_at", "authorized_amount", 10, 100.0);
    let payment = input("paid_at", "paid_amount", 11, 100.0);
    let settlement = input("settled_at", "fee", 12, 2.5);
    let plan = stream_plan(graph(&auth, &payment, &settlement)?)?;
    let sources = [
        ("left", auth),
        ("matched.right", payment),
        ("settled.right", settlement),
    ]
    .into_iter()
    .map(|(name, batch)| {
        Ok((
            name.into(),
            ReplaySource::new(vec![data(batch, 1)?, watermark(60_000_000)]).binding(),
        ))
    })
    .collect::<calc_flow::Result<BTreeMap<_, _>>>()?;
    let result = collect(plan, sources).await?;
    assert_eq!(f64_values(&result["output"], "net_amount"), [Some(97.5)]);
    support::print_columns(
        &result["output"],
        &["matched__authorization__account_id", "net_amount"],
    )
}

fn joins(
    auth: &calc_flow::Batch,
    payment: &calc_flow::Batch,
    settlement: &calc_flow::Batch,
) -> calc_flow::Result<(StreamJoinOperator, StreamJoinOperator)> {
    let first = StreamJoinOperator::new(
        "matched",
        schema(auth),
        schema(payment),
        spec(
            "account_id",
            "authorized_at",
            "paid_at",
            "authorization",
            "payment",
        )?,
    )?;
    let second = StreamJoinOperator::new(
        "settled",
        first.output_ports()[0].schema().unwrap().clone(),
        schema(settlement),
        spec(
            "authorization__account_id",
            "authorization__authorized_at",
            "settled_at",
            "matched",
            "settlement",
        )?,
    )?;
    Ok((first, second))
}

fn graph(
    auth: &calc_flow::Batch,
    payment: &calc_flow::Batch,
    settlement: &calc_flow::Batch,
) -> calc_flow::Result<PipelineBuilder> {
    let (first, second) = joins(auth, payment, settlement)?;
    let output_schema = second.output_ports()[0].schema().unwrap().clone();
    let net = support::expression_schemas(
        expression(
            "net",
            "net_amount = matched__authorization__authorized_amount - settlement__fee",
            &[],
            None,
        )?,
        Some(output_schema),
        None,
    )?;
    let nodes: Vec<(&str, calc_flow::NodeOperator)> = vec![
        ("matched", first.into()),
        ("settled", second.into()),
        ("net", net.into()),
    ];
    let builder = nodes.into_iter().try_fold(
        PipelineBuilder::new("symbolic-relational-dag")?,
        |builder, (id, node)| builder.add_node(id, node),
    )?;
    support::connect_all(
        builder,
        &[
            ("matched", "output", "settled", "left"),
            ("settled", "output", "net", "input"),
        ],
    )
}
