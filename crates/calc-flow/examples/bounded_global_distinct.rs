//! A finite final relation with `copy_visible_v2` owned peak accounting.
//! Temporarily borrowed external ingress and other graph/source owners are excluded.
use calc_flow::CompileOptions;
use calc_flow::{
    Batch, BatchMetadata, CompileMode, DistinctSpec, ExecutionOptions, InputOptions, Program,
    Result, StateBudget, UdfRegistry,
    arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    },
    table_input,
};
use std::{collections::BTreeMap, sync::Arc};

#[tokio::main]
async fn main() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let events = table_input("events", schema.clone(), InputOptions::default())?;
    let out = events.distinct(
        vec!["x".into()],
        DistinctSpec::bounded_source(StateBudget::new(5, 8 << 20)?),
    )?;
    let program = Program::new(
        "unique",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), out)]),
    )?;
    let plan = program
        .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
        .await?;
    let record = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(vec![
            Some(1),
            None,
            Some(1),
            None,
            Some(2),
        ]))],
    )
    .unwrap();
    let input = Batch::table(vec![record], BatchMetadata::default())?;
    let result = plan
        .execute(
            BTreeMap::from([("events".into(), input)]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(result.outputs["out"].num_rows(), 3);
    println!(
        "{} distinct tuples (order unspecified)",
        result.outputs["out"].num_rows()
    );
    Ok(())
}
