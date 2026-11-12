use calc_flow::arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchMetadata, CancellationToken, DistinctOperator, DistinctSpec, EdgeCollector,
    JsonMap, OperatorMetadata, Result, StateBudget, StreamJobContext, StreamOperator,
    StreamOperatorContext,
};
use std::{collections::HashMap, sync::Arc};

#[tokio::test]
async fn test_distinct_retains_fresh_schema_and_metadata_without_raw_arcs() -> Result<()> {
    let mut spare = String::with_capacity(8 << 20);
    spare.push_str("tag");
    let field = Arc::new(
        Field::new("x", DataType::Int64, true)
            .with_metadata(HashMap::from([("meaning".into(), spare)])),
    );
    let schema = Arc::new(Schema::new(vec![field.clone()]));
    let weak_schema = Arc::downgrade(&schema);
    let weak_field = Arc::downgrade(&field);
    let mut op = DistinctOperator::new(
        "owned_schema",
        schema.clone(),
        vec!["x".into()],
        DistinctSpec::bounded_source(StateBudget::new(2, 1 << 20)?),
    )?;
    let raw = Batch::table(
        vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from(vec![Some(1)]))],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )?;
    drop(field);
    drop(schema);
    let job = StreamJobContext::new(1, "test", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "owned_schema", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("input", raw, &context, &mut output).await?;
    assert!(
        weak_schema.upgrade().is_none(),
        "node must not retain raw SchemaRef"
    );
    assert!(
        weak_field.upgrade().is_none(),
        "node must not retain raw FieldRef/metadata"
    );
    assert_eq!(
        op.input_ports()[0].schema().unwrap().field(0).metadata()["meaning"],
        "tag"
    );
    assert!(
        op.input_ports()[0].schema().unwrap().field(0).metadata()["meaning"].capacity() < 1 << 20
    );
    Ok(())
}
