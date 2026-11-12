//! Run identical `DataFrame` and SQL unions with the standalone Rust core.

use calc_flow::arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchMetadata, CompileMode, CompileOptions, ExecutionOptions, InputOptions, Program,
    Result, StreamRequirements, UdfRegistry, table_input,
};
use std::{collections::BTreeMap, sync::Arc};

fn data(schema: Arc<Schema>, values: Vec<Option<i64>>) -> Result<Batch> {
    Batch::table(
        vec![RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values))]).unwrap()],
        BatchMetadata::default(),
    )
}

fn declaration(schema: Arc<Schema>) -> Result<Program> {
    let a = table_input("a", schema.clone(), InputOptions::default())?;
    let b = table_input("b", schema, InputOptions::default())?;
    let df = a.union_all(&b)?;
    let sql = calc_flow::sql::sql(
        "SELECT x FROM a UNION ALL SELECT x FROM b",
        BTreeMap::from([("a".into(), a), ("b".into(), b)]),
    )?;
    Program::new(
        "union",
        CompileMode::Batch,
        BTreeMap::from([("df".into(), df), ("sql".into(), sql)]),
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let batch = declaration(schema.clone())?;
    let udfs = UdfRegistry::new().snapshot();
    let plan = batch
        .compile_batch(&udfs, &CompileOptions::default())
        .await?;
    let result = plan
        .execute(input_batches(schema)?, ExecutionOptions::default())
        .await?;
    assert_outputs(&result.outputs)?;
    batch
        .for_mode(CompileMode::Stream)?
        .compile_stream(
            &udfs,
            &StreamRequirements::default(),
            &CompileOptions::default(),
        )
        .await?;
    println!("DataFrame and SQL: duplicate/NULL oracle passed; both lifecycles compile");
    Ok(())
}

fn assert_outputs(outputs: &BTreeMap<String, Batch>) -> Result<()> {
    for output in outputs.values() {
        let mut values: Vec<_> = output
            .table_payload()?
            .batches()
            .iter()
            .flat_map(|record| {
                record
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .iter()
                    .collect::<Vec<_>>()
            })
            .collect();
        values.sort();
        assert_eq!(values, vec![None, Some(1), Some(1), Some(1), Some(2)]);
    }
    Ok(())
}

fn input_batches(schema: Arc<Schema>) -> Result<BTreeMap<String, Batch>> {
    Ok(BTreeMap::from([
        (
            "a".into(),
            data(schema.clone(), vec![Some(1), None, Some(1)])?,
        ),
        ("b".into(), data(schema, vec![Some(1), Some(2)])?),
    ]))
}
