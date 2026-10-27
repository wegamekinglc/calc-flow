//! Python 21: one source fans out to two named stream outputs.
mod support;
use calc_flow::PipelineBuilder;
use std::collections::BTreeMap;
use support::{
    ExampleResult, ReplaySource, collect, edge, expression, i64_values, ints, stream_plan, table,
};

#[tokio::main]
async fn main() -> ExampleResult {
    let plan = stream_plan(
        PipelineBuilder::new("branches")?
            .add_node("source", expression("source", "", &["value"], None)?)?
            .add_node(
                "double",
                expression("double", "value2 = value * 2", &["value2"], None)?,
            )?
            .add_node(
                "large",
                expression("large", "", &["value"], Some("value >= 2"))?,
            )?
            .connect(edge("source", "output", "double", "input")?)?
            .connect(edge("source", "output", "large", "input")?)?,
    )?;
    let source = ReplaySource::batches(vec![
        table(vec![("value", ints(&[1, 2]))]),
        table(vec![("value", ints(&[3]))]),
    ])?;
    let results = collect(plan, BTreeMap::from([("input".into(), source.binding())])).await?;
    assert_eq!(
        i64_values(&results["double.output"], "value2"),
        [Some(2), Some(4), Some(6)]
    );
    assert_eq!(
        i64_values(&results["large.output"], "value"),
        [Some(2), Some(3)]
    );
    for (name, batches) in results {
        println!("{name}");
        support::print(&batches)?;
    }
    Ok(())
}
