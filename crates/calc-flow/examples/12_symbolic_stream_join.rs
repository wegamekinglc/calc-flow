//! Python 12: bounded event-time join followed by an expression stage.
mod support;
use calc_flow::{
    JoinStateLimits, JoinTimeBounds, PipelineBuilder, StreamJoinOperator, StreamJoinSpec,
};
use std::{collections::BTreeMap, time::Duration};
use support::{
    ExampleResult, ReplaySource, collect, data, expression, floats, i64_values, ints, schema,
    stream_plan, table, times, uints, watermark,
};

#[tokio::main]
async fn main() -> ExampleResult {
    let (left, right) = input_tables();
    let plan = stream_plan(graph(&left, &right)?)?;
    let sources = BTreeMap::from([
        ("left".into(), events(&left, &[(0, 1), (1, 1), (2, 1)])?),
        ("right".into(), events(&right, &[(0, 2), (2, 1)])?),
    ]);
    let results = collect(plan, sources).await?;
    check_results(&results)
}

fn input_tables() -> (calc_flow::Batch, calc_flow::Batch) {
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
    (left, right)
}

fn join_spec() -> calc_flow::Result<StreamJoinSpec> {
    StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "authorized_at",
        "paid_at",
        JoinTimeBounds::new(Duration::from_secs(5), Duration::from_secs(2))?,
        JoinStateLimits::new(10_000, 16 << 20, 100_000)?,
    )?
    .with_prefixes("authorization", "payment")
}

fn graph(left: &calc_flow::Batch, right: &calc_flow::Batch) -> calc_flow::Result<PipelineBuilder> {
    let join = StreamJoinOperator::new("join", schema(left), schema(right), join_spec()?)?;
    let output_schema = calc_flow::OperatorMetadata::output_ports(&join)[0]
        .schema()
        .unwrap()
        .clone();
    let matches = support::expression_schemas(
        expression(
            "matches",
            "amount_matches = authorization__amount = payment__paid_amount",
            &[],
            None,
        )?,
        Some(output_schema),
        None,
    )?;
    let builder = PipelineBuilder::new("symbolic-stream-join")?
        .add_node("join", join)?
        .add_node("matches", matches)?;
    support::connect_all(builder, &[("join", "output", "matches", "input")])
}

fn events(
    batch: &calc_flow::Batch,
    segments: &[(usize, usize)],
) -> calc_flow::Result<calc_flow::SourceBinding> {
    let record = &batch.table_payload()?.batches()[0];
    let mut events = segments
        .iter()
        .enumerate()
        .map(|(index, &(offset, rows))| {
            let batch = calc_flow::Batch::table(
                vec![record.slice(offset, rows)],
                calc_flow::BatchMetadata::default(),
            )?;
            data(batch, index + 1)
        })
        .collect::<calc_flow::Result<Vec<_>>>()?;
    events.push(watermark(60_000_000));
    Ok(ReplaySource::new(events).binding())
}

fn check_results(results: &BTreeMap<String, Vec<calc_flow::Batch>>) -> ExampleResult {
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
    support::print_columns(
        &results["output"],
        &["authorization__account_id", "amount_matches"],
    )
}
