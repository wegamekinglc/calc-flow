//! Python 01: order expressions, filtering, and two named outputs.
mod support;
use calc_flow::{ExecutionOptions, PipelineBuilder, UdfRegistry};
use std::collections::BTreeMap;
use support::{ExampleResult, edge, expression, i64_values, ints, strings, table};

#[tokio::main]
async fn main() -> ExampleResult {
    let orders = table(vec![
        ("order_id", strings(&["A-100", "A-101", "A-102"])),
        ("quantity", ints(&[3, 1, 4])),
        ("unit_price", ints(&[10, 12, 10])),
    ]);
    let registry = UdfRegistry::new().snapshot();
    let plan = PipelineBuilder::new("large-orders")?
        .add_node(
            "gross",
            expression("gross", "gross = quantity * unit_price", &[], None)?,
        )?
        .add_node(
            "large",
            expression(
                "large",
                "fee = CAST(gross AS DOUBLE) / 10.0",
                &["order_id", "gross"],
                Some("gross >= 20"),
            )?,
        )?
        .connect(edge("gross", "output", "large", "input")?)?
        .compile_batch(&registry)?;
    let result = plan
        .execute(
            BTreeMap::from([("input".into(), orders.clone())]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(
        i64_values(&[result.outputs["output"].clone()], "gross"),
        [Some(30), Some(40)]
    );
    support::print(&[result.outputs["output"].clone()])?;

    let program = PipelineBuilder::new("order-outputs")?
        .add_node(
            "source",
            expression("source", "gross = quantity * unit_price", &[], None)?,
        )?
        .add_node(
            "totals",
            expression("totals", "", &["order_id", "gross"], None)?,
        )?
        .add_node(
            "quantities",
            expression("quantities", "", &["order_id", "quantity"], None)?,
        )?
        .connect(edge("source", "output", "totals", "input")?)?
        .connect(edge("source", "output", "quantities", "input")?)?
        .compile_batch(&registry)?;
    let result = program
        .execute(
            BTreeMap::from([("input".into(), orders)]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(
        i64_values(&[result.outputs["totals.output"].clone()], "gross"),
        [Some(30), Some(12), Some(40)]
    );
    assert_eq!(
        i64_values(&[result.outputs["quantities.output"].clone()], "quantity"),
        [Some(3), Some(1), Some(4)]
    );
    println!("named outputs: {:?}", result.outputs.keys());
    Ok(())
}
