use calc_flow::arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchMetadata, CompileMode, CompileOptions, ExecutionOptions, InputOptions, Program,
    UdfRegistry, table_input,
};
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

fn schemas() -> Vec<Schema> {
    let field = Field::new("x", DataType::Int64, true);
    vec![
        Schema::new(vec![field.clone()]),
        Schema::new_with_metadata(
            vec![field.clone()],
            HashMap::from([("origin".into(), "consumer".into())]),
        ),
        Schema::new(vec![
            field.with_metadata(HashMap::from([("unit".into(), "USD".into())])),
        ]),
    ]
}

fn program(schema: Arc<Schema>) -> calc_flow::Result<Program> {
    let source = table_input("events", schema, InputOptions::default())?;
    Program::new(
        "metadata",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), source)]),
    )
}

fn input(schema: Arc<Schema>) -> calc_flow::Result<Batch> {
    let record =
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![Some(1)]))]).unwrap();
    Batch::table(vec![record], BatchMetadata::default())
}

#[tokio::test]
async fn exact_metadata_inputs_execute_without_mutating_caller_batches() -> calc_flow::Result<()> {
    for schema in schemas() {
        check_schema(Arc::new(schema)).await?;
    }
    Ok(())
}

async fn check_schema(schema: Arc<Schema>) -> calc_flow::Result<()> {
    let plan = program(schema.clone())?
        .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
        .await?;
    let batch = input(schema.clone())?;
    let result = plan
        .execute(
            BTreeMap::from([("events".into(), batch.clone())]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(
        result.outputs["out"].table_payload()?.batches()[0].num_rows(),
        1
    );
    assert_eq!(batch.table_payload()?.schema(), &schema);
    Ok(())
}
