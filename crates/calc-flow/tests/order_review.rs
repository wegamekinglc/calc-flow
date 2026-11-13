use calc_flow::{
    Batch, BatchMetadata, BoundedOrderOperator, ExecutionOptions, NullPlacement, OperatorMetadata,
    PipelineBuilder, Result, SortDirection, SortKey, SortSpec, StateBudget, UdfRegistry,
    arrow::{
        array::Int64Array,
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    },
};
use std::{collections::BTreeMap, future::Future, sync::Arc, task::Poll};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]))
}
fn operator(name: &str, limit: u64) -> Result<BoundedOrderOperator> {
    BoundedOrderOperator::new(
        name,
        schema(),
        vec![SortKey::new(
            "x",
            SortDirection::Ascending,
            NullPlacement::Last,
        )?],
        Some(1),
        SortSpec::bounded_source(StateBudget::new(200_000, limit)?)?,
    )
}
fn batch(values: Vec<i64>) -> Batch {
    Batch::table(
        vec![RecordBatch::try_new(schema(), vec![Arc::new(Int64Array::from(values))]).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn test_cancelled_finite_worker_does_not_charge_the_next_execute() {
    // Original full-input peak plus the new 261-byte owned name role.
    let op = operator("order", 11_262_843).unwrap();
    let plan = PipelineBuilder::new("cancel")
        .unwrap()
        .add_node("order", op)
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap();
    let input = plan.single_external_input().unwrap().to_owned();
    let mut seed = 10_u64;
    let large = batch(
        (0..200_000)
            .map(|_| {
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                i64::try_from(seed >> 1).unwrap()
            })
            .collect(),
    );
    let tiny = batch(vec![9]);
    for data in [large.clone(), tiny.clone()] {
        plan.execute(
            BTreeMap::from([(input.clone(), data)]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    }
    let mut pending = Box::pin(plan.execute(
        BTreeMap::from([(input.clone(), large)]),
        ExecutionOptions::default(),
    ));
    std::future::poll_fn(|cx| {
        assert!(matches!(pending.as_mut().poll(cx), Poll::Pending));
        Poll::Ready(())
    })
    .await;
    drop(pending);
    let result = plan
        .execute(BTreeMap::from([(input, tiny)]), ExecutionOptions::default())
        .await
        .unwrap();
    let output = result
        .outputs
        .values()
        .next()
        .unwrap()
        .table_payload()
        .unwrap();
    assert_eq!(result.outputs.values().next().unwrap().num_rows(), 1);
    assert_eq!(
        output.batches()[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0),
        9
    );
}

#[test]
fn test_owned_name_is_prepaid_at_the_declaration_boundary() {
    let name = "n".repeat(1 << 20);
    assert!(
        operator(&name, 40_000).is_err(),
        "a copied name cannot exceed the entire quota"
    );
    // Base declaration 25616 + 1048576 visible name bytes + 256 descriptor bytes.
    for (limit, success) in [(1_074_448, true), (1_074_447, false)] {
        let result = operator(&name, limit);
        assert_eq!(result.is_ok(), success, "name boundary {limit}");
        if let Ok(op) = result {
            assert_eq!(op.name(), name);
            assert_ne!(op.name().as_ptr(), name.as_ptr());
        }
    }
}
