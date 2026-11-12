use calc_flow::arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use calc_flow::{
    Batch, BatchKind, BatchMetadata, ExecutionOptions, PipelineBuilder, Port, Result, UdfRegistry,
    UnionOperator,
};
use std::{collections::BTreeMap, sync::Arc};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]))
}

fn data(parts: &[&[Option<i64>]]) -> Batch {
    Batch::table(
        parts
            .iter()
            .map(|values| {
                RecordBatch::try_new(schema(), vec![Arc::new(Int64Array::from(values.to_vec()))])
                    .unwrap()
            })
            .collect(),
        BatchMetadata::default(),
    )
    .unwrap()
}

fn values(batch: &Batch) -> Vec<Option<i64>> {
    batch
        .table_payload()
        .unwrap()
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
        .collect()
}

fn wrong_schema_data() -> Batch {
    let schema = Arc::new(Schema::new(vec![Field::new("y", DataType::Int64, true)]));
    Batch::table(
        vec![
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![Some(100)]))])
                .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}

#[tokio::test]
async fn finite_union_reuses_one_plan_without_old_rows() -> Result<()> {
    let ports = ["left", "right"]
        .into_iter()
        .map(|name| {
            Port::new(
                name,
                BatchKind::Table,
                true,
                Some(vec![Field::new("x", DataType::Int64, true)]),
            )
            .unwrap()
        })
        .collect();
    let plan = PipelineBuilder::new("finite")?
        .add_node("merge", Box::new(UnionOperator::new("merge", ports)?))?
        .compile_batch(&UdfRegistry::new().snapshot())?;
    let cases = [
        (
            data(&[&[Some(1), None], &[Some(1)]]),
            data(&[&[Some(1), Some(2)]]),
            vec![Some(1), None, Some(1), Some(1), Some(2)],
        ),
        (data(&[&[Some(9)]]), data(&[&[]]), vec![Some(9)]),
        (data(&[&[]]), data(&[&[]]), vec![]),
    ];
    for (left, right, expected) in cases {
        let out = plan
            .execute(
                BTreeMap::from([("left".into(), left), ("right".into(), right)]),
                ExecutionOptions::default(),
            )
            .await?;
        assert_eq!(values(&out.outputs["output"]), expected);
        assert_eq!(out.outputs["output"].table_payload()?.schema(), &schema());
        assert_eq!(out.outputs["output"].metadata().source(), "merge");
        assert_eq!(out.outputs["output"].metadata().sequence(), 0);
        assert!(out.outputs["output"].metadata().attributes().is_empty());
        assert!(
            plan.execute(
                BTreeMap::from([
                    ("left".into(), wrong_schema_data()),
                    ("right".into(), data(&[&[]]))
                ]),
                ExecutionOptions::default(),
            )
            .await
            .is_err()
        );
        assert!(
            plan.execute(
                BTreeMap::from([("left".into(), data(&[&[Some(100)]]))]),
                ExecutionOptions::default(),
            )
            .await
            .is_err()
        );
        let cancellation = calc_flow::CancellationToken::new();
        cancellation.cancel();
        assert!(
            plan.execute(
                BTreeMap::from([
                    ("left".into(), data(&[&[Some(100)]])),
                    ("right".into(), data(&[&[]]))
                ]),
                ExecutionOptions {
                    cancellation,
                    ..Default::default()
                },
            )
            .await
            .is_err()
        );
    }
    Ok(())
}

#[tokio::test]
async fn finite_union_rejects_non_exact_or_optional_ports_in_analysis() -> Result<()> {
    use calc_flow::{AnalysisOptions, StreamRequirements, SupportStatus};
    for (kind, required, fields) in [
        (BatchKind::Table, true, None),
        (BatchKind::Array, true, None),
        (
            BatchKind::Table,
            false,
            Some(vec![Field::new("x", DataType::Int64, true)]),
        ),
    ] {
        let ports = ["left", "right"]
            .into_iter()
            .map(|name| Port::new(name, kind, required, fields.clone()).unwrap())
            .collect();
        let builder = PipelineBuilder::new("shape")?
            .add_node("merge", Box::new(UnionOperator::new("merge", ports)?))?;
        let report = builder
            .analyze(&UdfRegistry::new().snapshot(), &AnalysisOptions::default())
            .await?;
        assert_eq!(report.nodes[0].batch.support, SupportStatus::Unsupported);
        assert!(report.diagnostics.iter().any(
            |issue| issue.code == "unsupported_mode" && issue.message.contains("exact schema")
        ));
        builder.compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )?;
    }
    Ok(())
}

#[test]
fn finite_union_has_no_checkpoint_state() -> Result<()> {
    use calc_flow::BatchOperator;
    let ports = ["left", "right"]
        .into_iter()
        .map(|name| {
            Port::new(
                name,
                BatchKind::Table,
                true,
                Some(vec![Field::new("x", DataType::Int64, true)]),
            )
            .unwrap()
        })
        .collect();
    let mut operator = UnionOperator::new("merge", ports)?;
    assert_eq!(BatchOperator::snapshot(&operator)?, serde_json::Value::Null);
    assert!(BatchOperator::restore(&mut operator, &serde_json::json!({"rows": 1})).is_err());
    BatchOperator::reset(&mut operator)?;
    Ok(())
}

#[tokio::test]
async fn finite_union_enforces_existing_table_result_limit() -> Result<()> {
    use calc_flow::arrow::array::NullArray;
    let fields = vec![Field::new("x", DataType::Null, true)];
    let schema = Arc::new(Schema::new(fields.clone()));
    let plan = null_union_plan(&fields)?;
    let record =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(NullArray::new(50_000_001))]).unwrap();
    let input = Batch::table(vec![record], BatchMetadata::default())?;
    let result = plan
        .execute(
            BTreeMap::from([("left".into(), input.clone()), ("right".into(), input)]),
            ExecutionOptions::default(),
        )
        .await;
    assert!(
        result.is_err(),
        "union must enforce the existing 100M table result row limit"
    );
    let empty = Batch::table(
        vec![RecordBatch::new_empty(schema)],
        BatchMetadata::default(),
    )?;
    let out = plan
        .execute(
            BTreeMap::from([("left".into(), empty.clone()), ("right".into(), empty)]),
            ExecutionOptions::default(),
        )
        .await?;
    assert_eq!(
        out.outputs["output"].table_payload()?.batches()[0].num_rows(),
        0
    );
    Ok(())
}

fn null_union_plan(fields: &[Field]) -> Result<calc_flow::BatchExecutionPlan> {
    let ports = ["left", "right"]
        .into_iter()
        .map(|name| Port::new(name, BatchKind::Table, true, Some(fields.to_vec())).unwrap())
        .collect();
    PipelineBuilder::new("limit")?
        .add_node("merge", Box::new(UnionOperator::new("merge", ports)?))?
        .compile_batch(&UdfRegistry::new().snapshot())
}
