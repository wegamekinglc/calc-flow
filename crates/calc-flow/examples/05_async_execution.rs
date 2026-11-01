//! Python 05: await execution while another Tokio task remains live.
mod support;
use calc_flow::{ExecutionOptions, PipelineBuilder, UdfRegistry};
use std::collections::BTreeMap;
use support::{ExampleResult, expression, i64_values, ints, table};

#[tokio::main]
async fn main() -> ExampleResult {
    let plan = PipelineBuilder::new("async-example")?
        .add_node(
            "calculate",
            expression("calculate", "total = a + b", &["total"], None)?,
        )?
        .compile_batch(&UdfRegistry::new().snapshot())?;
    let options = ExecutionOptions {
        settings: BTreeMap::from([(
            "request".into(),
            serde_json::json!({"source":"async-example"}),
        )]),
        deadline: Some(chrono::Utc::now() + chrono::Duration::seconds(30)),
        ..ExecutionOptions::default()
    };
    let heartbeat = tokio::spawn(async {
        tokio::task::yield_now().await;
        "event loop remained live"
    });
    let result = plan
        .execute(
            BTreeMap::from([(
                "input".into(),
                table(vec![("a", ints(&[1, 3])), ("b", ints(&[2, 4]))]),
            )]),
            options,
        )
        .await?;
    println!("{}", heartbeat.await?);
    assert_eq!(
        i64_values(&[result.outputs["output"].clone()], "total"),
        [Some(3), Some(7)]
    );
    support::print(&[result.outputs["output"].clone()])
}
