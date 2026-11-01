//! Python 26: consume normal and late branches from one rolling state owner.
mod support;
use calc_flow::{
    LatePolicySpec, PipelineBuilder, RollingFrameSpec, RollingOperator, RollingOutputSpec,
};
use std::collections::BTreeMap;
use support::{
    ExampleResult, ReplaySource, collect, data, f64_values, floats, rolling, schema, stream_plan,
    strings, table, times, uints, watermark,
};

#[tokio::main]
async fn main() -> ExampleResult {
    let input = table(vec![
        ("ts", times(&[0, 2])),
        ("symbol", strings(&["a", "a"])),
        ("seq", uints(&[0, 1])),
        ("x", floats(&[10.0, 20.0])),
    ]);
    let plan = plan(&input)?;
    let output_names = plan
        .sink_binding_ids()
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let source = ReplaySource::new(vec![watermark(1), data(input, 2)?, watermark(2)]);
    let result = collect(plan, BTreeMap::from([("input".into(), source.binding())])).await?;
    let normal = output_names
        .iter()
        .find(|name| name.ends_with("output"))
        .expect("normal output");
    let late = output_names
        .iter()
        .find(|name| name.ends_with("late"))
        .expect("late output");
    assert_eq!(f64_values(&result[normal], "x"), [Some(20.0)]);
    assert_eq!(f64_values(&result[late], "x"), [Some(10.0)]);
    println!(
        "{{'normal': {}, 'late': {}}}",
        support::format_values(&f64_values(&result[normal], "x")),
        support::format_values(&f64_values(&result[late], "x")),
    );
    Ok(())
}

fn plan(input: &calc_flow::Batch) -> calc_flow::Result<calc_flow::StreamExecutionPlan> {
    let mut spec = rolling(vec![RollingOutputSpec::Mean {
        primitive_version: 1,
        input: "x".into(),
        output: "avg".into(),
        frame: RollingFrameSpec::Rows { size: 2 },
        min_periods: 1,
    }]);
    spec.late_policy = LatePolicySpec::SideOutput {
        metrics_version: 1,
        schema_version: 1,
    };
    stream_plan(PipelineBuilder::new("late")?.add_node(
        "rolling",
        RollingOperator::new("rolling", schema(input), spec)?,
    )?)
}
