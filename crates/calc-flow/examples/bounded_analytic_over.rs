//! Standalone core example: one final analytic relation over a bounded history.
use calc_flow::arrow::{
    array::{Int64Array, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use calc_flow::{
    AnalyticExpr, Batch, BatchMetadata, CompileMode, CompileOptions, ExecutionOptions,
    InputOptions, NullPlacement, OverOrder, OverSpec, Program, Result, RowsFrame, SortDirection,
    SortKey, StateBudget, UdfRegistry, table_input,
};
use std::{collections::BTreeMap, sync::Arc};

#[tokio::main]
async fn main() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("v", DataType::Int64, true),
        Field::new("seq", DataType::UInt64, false),
    ]));
    let program = analytic_program(schema.clone())?;
    let plan = program
        .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
        .await?;
    let result = plan
        .execute(
            BTreeMap::from([("events".into(), input(schema)?)]),
            ExecutionOptions::default(),
        )
        .await?;
    verify_result(&result.outputs["out"])?;
    println!("complete H=[3,1,2]; sorted v=[1,2,3], rn=[1,2,3], prev=[NULL,1,2], total=[1,3,6]");
    Ok(())
}
fn analytic_order() -> Result<OverOrder> {
    OverOrder::new(
        vec![],
        vec![SortKey::new(
            "v",
            SortDirection::Ascending,
            NullPlacement::Last,
        )?],
        vec!["seq".into()],
    )
}
fn expressions() -> Result<Vec<AnalyticExpr>> {
    Ok(vec![
        AnalyticExpr::row_number("rn")?,
        AnalyticExpr::lag("v", "prev")?,
        AnalyticExpr::sum("v", RowsFrame::cumulative(), "total")?,
    ])
}
fn analytic_program(schema: Arc<Schema>) -> Result<Program> {
    let events = table_input("events", schema, InputOptions::default())?;
    let out = events.over(
        expressions()?,
        OverSpec::bounded_source(analytic_order()?, StateBudget::new(9, 8 << 20)?)?,
    )?;
    Program::new(
        "features",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), out)]),
    )
}
fn input(schema: Arc<Schema>) -> Result<Batch> {
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int64Array::from(vec![3, 1, 2])),
                    Arc::new(UInt64Array::from(vec![30, 10, 20])),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
}
fn verify_result(batch: &Batch) -> Result<()> {
    let record = &batch.table_payload()?.batches()[0];
    let sums = record
        .column_by_name("total")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(
        sums.iter().collect::<Vec<_>>(),
        vec![Some(1), Some(3), Some(6)]
    );
    Ok(())
}
