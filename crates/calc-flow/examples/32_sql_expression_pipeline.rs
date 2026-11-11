//! Reuse typed expression stages around native SQL, without graph wiring.
use calc_flow::arrow::{
    array::{Float64Array, Int64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchMetadata, CompileMode, CompileOptions, ExecutionOptions, InputOptions, Program,
    Result, TableExpr, UdfRegistry, lit, table_input,
};
use std::{collections::BTreeMap, sync::Arc};

fn add_gross(table: &TableExpr) -> Result<TableExpr> {
    table.with_columns(vec![
        (table.column("quantity")?.cast(DataType::Float64)? * table.column("price")?)
            .alias("gross")?,
    ])
}
fn discounted(table: &TableExpr) -> Result<TableExpr> {
    table.select(vec![
        table.column("order_id")?.alias("order_id")?,
        (table.column("gross")? * lit(0.9_f64)?).alias("net")?,
    ])
}

#[tokio::main]
async fn main() -> Result<()> {
    let (schema, input) = orders()?;
    let batch = declaration(schema)?;
    verify_wrappers(&batch).await?;
    let actual = execute(&batch, input).await?;
    assert_eq!(actual, vec![Some(18.0), Some(27.0)]);
    println!("net = {actual:?}");
    Ok(())
}

fn orders() -> Result<(Arc<Schema>, Batch)> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_id", DataType::Int64, false),
        Field::new("quantity", DataType::Int64, false),
        Field::new("price", DataType::Float64, false),
    ]));
    let input = Batch::table(
        vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![1, 2, 3])),
                    Arc::new(Int64Array::from(vec![2, 1, 3])),
                    Arc::new(Float64Array::from(vec![10.0, 5.0, 10.0])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )?;
    Ok((schema, input))
}
fn declaration(schema: Arc<Schema>) -> Result<Program> {
    let source = table_input("orders", schema, InputOptions::default())?;
    let gross = add_gross(&source)?;
    let selected =
        gross.sql("SELECT order_id, gross FROM input WHERE gross >= 20 ORDER BY order_id")?;
    let result = discounted(&selected)?.sql("SELECT order_id, net FROM input")?;
    let batch = Program::new(
        "sql-expression-pipeline",
        CompileMode::Batch,
        BTreeMap::from([("gross".into(), gross), ("result".into(), result)]),
    )?;
    Ok(batch)
}
async fn verify_wrappers(batch: &Program) -> Result<()> {
    let stream = batch.for_mode(CompileMode::Stream)?;
    assert_eq!(batch.fingerprint(), stream.fingerprint());
    let udfs = UdfRegistry::new().snapshot();
    assert_eq!(
        batch.collect_schema(&udfs).await?["result"].field(1).name(),
        "net"
    );
    Ok(())
}
async fn execute(batch: &Program, input: Batch) -> Result<Vec<Option<f64>>> {
    let udfs = UdfRegistry::new().snapshot();
    let plan = batch
        .compile_batch(&udfs, &CompileOptions::default())
        .await?;
    let output = plan
        .execute(
            BTreeMap::from([("orders".into(), input)]),
            ExecutionOptions::default(),
        )
        .await?;
    let records = output.outputs["result"].table_payload()?.batches();
    let actual: Vec<_> = records
        .iter()
        .flat_map(|record| {
            record
                .column(1)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .iter()
                .collect::<Vec<_>>()
        })
        .collect();
    Ok(actual)
}
