//! Python 01: order expressions, filtering, and two named outputs.
mod support;
use calc_flow::{
    Batch, CompileMode, CompileOptions, ExecutionOptions, InputOptions, Program, Result, RunResult,
    TableExpr, UdfRegistry, arrow::datatypes::DataType, lit, table_input,
};
use std::collections::BTreeMap;
use support::{ExampleResult, i64_values, ints, strings, table};

fn large_orders(source: &TableExpr) -> Result<TableExpr> {
    let gross = source.column("quantity")? * source.column("unit_price")?;
    let fee = gross.clone().cast(DataType::Float64)? / lit(10.0_f64)?;
    let enriched = source.with_columns(vec![gross.alias("gross")?, fee.alias("fee")?])?;
    let large = enriched.filter(enriched.column("gross")?.ge(lit(20_i64)?))?;
    large.select(vec![
        large.column("order_id")?.alias("order_id")?,
        large.column("gross")?.alias("gross")?,
    ])
}

async fn execute(program: &Program, orders: Batch) -> Result<RunResult> {
    let udfs = UdfRegistry::new();
    let plan = program
        .compile_batch(&udfs.snapshot(), &CompileOptions::default())
        .await?;
    plan.execute(
        BTreeMap::from([("orders".into(), orders)]),
        ExecutionOptions::default(),
    )
    .await
}

#[tokio::main]
async fn main() -> ExampleResult {
    let orders = table(vec![
        ("order_id", strings(&["A-100", "A-101", "A-102"])),
        ("quantity", ints(&[3, 1, 4])),
        ("unit_price", ints(&[10, 12, 10])),
    ]);
    let source = table_input(
        "orders",
        orders.table_payload()?.schema().clone(),
        InputOptions::default(),
    )?;
    let filtered = Program::new(
        "large-orders",
        CompileMode::Batch,
        BTreeMap::from([("large".into(), large_orders(&source)?)]),
    )?;
    let filtered_result = execute(&filtered, orders.clone()).await?;
    let large = std::slice::from_ref(&filtered_result.outputs["large"]);
    assert_eq!(i64_values(large, "gross"), [Some(30), Some(40)]);
    support::print(large)?;

    let gross = source.column("quantity")? * source.column("unit_price")?;
    let totals = source.select(vec![
        source.column("order_id")?.alias("order_id")?,
        gross.alias("gross")?,
    ])?;
    let quantities = source.select(vec![
        source.column("order_id")?.alias("order_id")?,
        source.column("quantity")?.alias("quantity")?,
    ])?;
    let named = Program::new(
        "order-outputs",
        CompileMode::Batch,
        BTreeMap::from([("totals".into(), totals), ("quantities".into(), quantities)]),
    )?;
    let result = execute(&named, orders).await?;
    assert_eq!(
        i64_values(std::slice::from_ref(&result.outputs["totals"]), "gross"),
        [Some(30), Some(12), Some(40)]
    );
    assert_eq!(
        i64_values(
            std::slice::from_ref(&result.outputs["quantities"]),
            "quantity",
        ),
        [Some(3), Some(1), Some(4)]
    );
    let names = ["totals", "quantities"];
    assert_eq!(result.outputs.len(), names.len());
    println!("named outputs: {names:?}");
    Ok(())
}
