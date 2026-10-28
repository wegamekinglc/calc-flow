//! Python 01: order expressions, filtering, and two named outputs.
mod support;
use calc_flow::{Batch, ExecutionOptions, PipelineBuilder, UdfRegistry};
use std::collections::BTreeMap;
use support::{ExampleResult, connect_all, expression, i64_values, ints, strings, table};

#[tokio::main]
async fn main() -> ExampleResult {
    let orders = table(vec![
        ("order_id", strings(&["A-100", "A-101", "A-102"])),
        ("quantity", ints(&[3, 1, 4])),
        ("unit_price", ints(&[10, 12, 10])),
    ]);
    filtered_orders(&orders).await?;
    named_outputs(orders).await
}

fn filtered_graph() -> calc_flow::Result<PipelineBuilder> {
    let builder = PipelineBuilder::new("large-orders")?
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
        )?;
    connect_all(builder, &[("gross", "output", "large", "input")])
}

async fn filtered_orders(orders: &Batch) -> ExampleResult {
    let plan = filtered_graph()?.compile_batch(&UdfRegistry::new().snapshot())?;
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
    support::print(&[result.outputs["output"].clone()])
}

fn named_graph() -> calc_flow::Result<PipelineBuilder> {
    let builder = PipelineBuilder::new("order-outputs")?
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
        )?;
    connect_all(
        builder,
        &[
            ("source", "output", "totals", "input"),
            ("source", "output", "quantities", "input"),
        ],
    )
}

async fn named_outputs(orders: Batch) -> ExampleResult {
    let program = named_graph()?.compile_batch(&UdfRegistry::new().snapshot())?;
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
