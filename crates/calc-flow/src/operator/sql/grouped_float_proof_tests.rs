use super::*;
use crate::{
    Batch, BatchMetadata, CancellationToken, DataFusionRuntime, StreamJobContext,
    StreamOperatorContext,
};
use datafusion::{
    arrow::{
        array::{Float32Array, Float64Array, Int64Array},
        datatypes::{Field, Schema},
        record_batch::RecordBatch,
    },
    execution::memory_pool::MemoryConsumer,
};

async fn plan(dtype: &DataType) -> (DataFusionRuntime, super::super::IncrementalSql, Batch) {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, true),
        Field::new("value", dtype.clone(), true),
    ]));
    let query = crate::expression::parse_select_query(
        "SELECT key, MIN(value), MAX(value), COUNT(*) FROM events GROUP BY key",
    )
    .unwrap();
    let plan =
        super::super::IncrementalSql::plan(&runtime, &query, "events", schema.clone(), "proof")
            .await
            .unwrap()
            .unwrap();
    let values: datafusion::arrow::array::ArrayRef = match dtype {
        DataType::Float32 => Arc::new(Float32Array::from(vec![Some(1.0); 3])),
        DataType::Float64 => Arc::new(Float64Array::from(vec![Some(1.0); 3])),
        _ => unreachable!(),
    };
    let record = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(vec![Some(1); 3])), values],
    )
    .unwrap();
    (
        runtime,
        plan,
        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
    )
}

#[tokio::test]
async fn test_grouped_float_capacity_covers_real_native_arbitrary_jump() {
    for dtype in [DataType::Float32, DataType::Float64] {
        let (runtime, plan, _) = plan(&dtype).await;
        for expression in &plan.aggregates[..2] {
            let mut accumulator = expression.create_groups_accumulator().unwrap();
            let value = match dtype {
                DataType::Float32 => ScalarValue::Float32(Some(1.0)),
                _ => ScalarValue::Float64(Some(1.0)),
            };
            for count in [6, 7] {
                accumulator
                    .update_batch(
                        &[value.to_array_of_size(count).unwrap()],
                        &(0..count).collect::<Vec<_>>(),
                        None,
                        count,
                    )
                    .unwrap();
            }
            let states = accumulator
                .state(datafusion::logical_expr::EmitTo::All)
                .unwrap();
            let bytes = states[0].to_data().buffers()[0].capacity();
            assert!(bytes <= capacity(7, "proof").unwrap() * dtype.primitive_width().unwrap());
        }
        assert!(capacity(usize::MAX, "proof").is_err());
        drop(plan);
        assert_eq!(runtime.incremental_memory_pool().reserved(), 0);
    }
}

#[tokio::test]
async fn test_grouped_float_proof_growth_refusal_refunds_and_preserves_state() {
    for dtype in [DataType::Float32, DataType::Float64] {
        let (runtime, mut plan, batch) = plan(&dtype).await;
        let job = StreamJobContext::new(
            921,
            "proof",
            crate::JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "proof", None);
        let transaction = plan.update(&batch, &context, "proof").await.unwrap();
        plan.commit(transaction);
        let before = plan.export_native_state("proof", || Ok(())).unwrap();
        let policy = plan.checkpoint_policy();
        let needed = headroom(plan.groups.len(), 10_003, 8, &plan.aggregates, "proof").unwrap();
        let pool = runtime.incremental_memory_pool();
        let basis = pool.reserved();
        let pressure = MemoryConsumer::new("proof-pressure").register(&pool);
        pressure.try_grow((1 << 30) - basis - (needed - 1)).unwrap();
        let record = RecordBatch::try_new(
            batch.table_payload().unwrap().schema().clone(),
            vec![
                Arc::new(Int64Array::from(vec![Some(1); 10_003])),
                ScalarValue::try_from(&dtype)
                    .unwrap()
                    .to_array_of_size(10_003)
                    .unwrap(),
            ],
        )
        .unwrap();
        let held = pool.reserved();
        assert!(plan.prepare_grouped_proof(&[record], "proof").is_err());
        assert_eq!(pool.reserved(), held);
        assert_eq!(plan.checkpoint_policy(), policy);
        drop(pressure);
        assert_eq!(pool.reserved(), basis);
        let after = plan.export_native_state("proof", || Ok(())).unwrap();
        assert_eq!(before.records(), after.records());
        let query = crate::expression::parse_select_query(
            "SELECT key, MIN(value), MAX(value), COUNT(*) FROM events GROUP BY key",
        )
        .unwrap();
        let schema = batch.table_payload().unwrap().schema().clone();
        let mut candidate = super::super::IncrementalSql::plan_sync(
            &runtime,
            &query,
            "events",
            schema.clone(),
            schema,
            "proof",
        )
        .unwrap()
        .unwrap();
        let candidate_policy = candidate.checkpoint_policy();
        let needed = headroom(plan.groups.len(), 3, 8, &plan.aggregates, "proof").unwrap();
        let basis = pool.reserved();
        let pressure = MemoryConsumer::new("cold-proof-pressure").register(&pool);
        pressure.try_grow((1 << 30) - basis - (needed - 1)).unwrap();
        let held = pool.reserved();
        assert!(
            candidate
                .restore_grouped_proof(&policy, plan.groups.len(), 3, "proof")
                .is_err()
        );
        assert_eq!(candidate.checkpoint_policy(), candidate_policy);
        assert_eq!(pool.reserved(), held);
        drop(pressure);
        assert_eq!(pool.reserved(), basis);
        drop((plan, candidate, before, after, batch));
        assert_eq!(pool.reserved(), 0);
    }
}
