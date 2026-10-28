//! Python 19: reusable expression stages around a read-only SQL stage.
mod support;
use calc_flow::{ExecutionOptions, PipelineBuilder, UdfRegistry};
use std::collections::BTreeMap;
use support::{ExampleResult, connect_all, expression, f64_values, floats, ints, sql, table};

fn pipeline() -> calc_flow::Result<PipelineBuilder> {
    let builder = PipelineBuilder::new("sql-expression-pipeline")?
        .add_node(
            "gross",
            expression(
                "gross",
                "gross = CAST(quantity AS DOUBLE) * price",
                &[],
                None,
            )?,
        )?
        .add_node(
            "sql",
            sql(
                "sql",
                "SELECT order_id, gross FROM input WHERE gross >= 20 ORDER BY order_id",
                &["input"],
            )?,
        )?
        .add_node(
            "discount",
            expression("discount", "net = gross * 0.9", &["order_id", "net"], None)?,
        )?;
    connect_all(
        builder,
        &[
            ("gross", "output", "sql", "input"),
            ("sql", "output", "discount", "input"),
        ],
    )
}

#[tokio::main]
async fn main() -> ExampleResult {
    let input = table(vec![
        ("order_id", ints(&[1, 2, 3])),
        ("quantity", ints(&[2, 1, 3])),
        ("price", floats(&[10.0, 5.0, 10.0])),
    ]);
    let result = pipeline()?
        .compile_batch(&UdfRegistry::new().snapshot())?
        .execute(
            BTreeMap::from([("input".into(), input)]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(
        f64_values(&[result.outputs["output"].clone()], "net"),
        [Some(18.0), Some(27.0)]
    );
    support::print(&[result.outputs["output"].clone()])
}
