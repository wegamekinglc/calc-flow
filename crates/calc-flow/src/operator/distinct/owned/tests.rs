use super::*;
use crate::state_memory_pool::OwnedStatePool;
use datafusion::{arrow::array::Int64Array, execution::memory_pool::UnboundedMemoryPool};
use std::cell::Cell;

thread_local! {static TARGET_ALLOCATIONS:Cell<usize>=const {Cell::new(0)};}
pub(super) fn before_target_allocation() {
    TARGET_ALLOCATIONS.with(|n| n.set(n.get() + 1));
}

#[test]
fn test_empty_history_reserves_all_array_descriptors_before_constructing_them() {
    let columns = (0..32).map(|n| format!("k{n}")).collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(
        columns
            .iter()
            .map(|name| Field::new(name, DataType::Int64, true))
            .collect::<Vec<_>>(),
    ));
    let quota = Arc::new(OwnedStatePool::new(
        Arc::new(UnboundedMemoryPool::default()),
        "empty distinct",
        "limit",
    ));
    quota.set_limit(Some(8192)).unwrap();
    let pool: Arc<dyn MemoryPool> = quota;
    TARGET_ALLOCATIONS.with(|n| n.set(0));
    assert!(combine(&[], &schema, &columns, &pool, &|| Ok(())).is_err());
    assert_eq!(
        TARGET_ALLOCATIONS.with(Cell::get),
        0,
        "raw empty arrays must not be allocated before their complete descriptor role is admitted"
    );
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_distinct_declaration_compacts_owned_spec_strings() {
    use crate::{DistinctHistory, DistinctOperator, DistinctSpec, EventTime, StateBudget};
    let budget = StateBudget::new(2, 1 << 20).unwrap();
    let spec = DistinctSpec::closed_window(
        "ts",
        EventTime::from_micros(0),
        EventTime::from_micros(10),
        budget,
    )
    .unwrap();
    let mut wire = serde_json::to_value(spec).unwrap();
    let mut spare = String::with_capacity(8 << 20);
    spare.push_str("ts");
    wire["history"]["event_time_column"] = serde_json::Value::String(spare);
    let spec: DistinctSpec = serde_json::from_value(wire).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("q", DataType::Int64, true),
    ]));
    let node = DistinctOperator::new("spec_owned", schema, vec!["q".into()], spec).unwrap();
    let DistinctHistory::ClosedWindow {
        event_time_column, ..
    } = node.spec.history()
    else {
        panic!("window required")
    };
    assert!(
        event_time_column.capacity() < 1024,
        "node must not retain an 8 MiB spare allocation as a two-byte configuration string"
    );
}

#[test]
fn test_copy_target_exact_budget_and_one_byte_short_reject_before_allocation() {
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let raw = Batch::table(
        vec![RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1]))]).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap();
    // Independent literal: schema/base 4610, record 256, column descriptor 256,
    // aligned i64 values 64, metadata 4096. No external parent-size inference.
    for (limit, succeeds) in [(9282, true), (9281, false)] {
        let quota = Arc::new(OwnedStatePool::new(
            Arc::new(UnboundedMemoryPool::default()),
            "test distinct",
            "limit",
        ));
        quota.set_limit(Some(limit)).unwrap();
        let pool: Arc<dyn MemoryPool> = quota.clone();
        TARGET_ALLOCATIONS.with(|n| n.set(0));
        let result = normalize(&raw, &["x".into()], &pool, None, &|| Ok(()));
        assert_eq!(result.is_ok(), succeeds);
        if let Ok(owned) = result {
            assert_eq!(owned.charge(), 9282);
            assert_eq!(TARGET_ALLOCATIONS.with(Cell::get), 2);
            assert_eq!(quota.take_peak(), 9282);
        } else {
            assert_eq!(
                TARGET_ALLOCATIONS.with(Cell::get),
                0,
                "schema/array target must not exist at B-1"
            );
        }
        assert_eq!(pool.reserved(), 0);
    }
}

async fn close_with_budget(limit: u64) -> (bool, usize, usize) {
    use crate::{
        CancellationToken, DataFusionConfig, DataFusionParallelismMode, DistinctOperator,
        DistinctSpec, EdgeCollector, Epoch, JsonMap, OperatorMetadata, StateBudget,
        StreamJobContext, StreamOperator, StreamOperatorContext,
    };
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, true)]));
    let mut op = DistinctOperator::new(
        "peak",
        schema.clone(),
        vec!["x".into()],
        DistinctSpec::bounded_source(StateBudget::new(2, limit).unwrap()),
    )
    .unwrap();
    op.set_datafusion_config(DataFusionConfig {
        parallelism_mode: DataFusionParallelismMode::Fixed,
        target_partitions: 1,
        ..DataFusionConfig::default()
    });
    let job = StreamJobContext::new(1, "peak", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "peak", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    TARGET_ALLOCATIONS.with(|n| n.set(0));
    let raw = Batch::table(
        vec![RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 1]))]).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap();
    op.process_data("input", raw, &context, &mut output)
        .await
        .unwrap();
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
    let result = op.on_end(&context, &mut output).await;
    if result.is_err() {
        assert!(output.drain("output").is_empty());
        assert!(!op.stream.as_ref().unwrap().closed);
        assert_eq!(op.stream.as_ref().unwrap().accepted, 2);
        assert_eq!(
            op.checkpoint(Epoch::INITIAL).unwrap().segments["control"],
            before.segments["control"]
        );
    } else {
        let batches = output.drain("output");
        assert_eq!(batches.len(), 1);
        let values = batches[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(values.iter().collect::<Vec<_>>(), vec![Some(1)]);
    }
    (
        result.is_ok(),
        op.pool.take_peak(),
        TARGET_ALLOCATIONS.with(Cell::get),
    )
}

#[tokio::test]
async fn test_query_and_compact_output_inclusive_peak_with_live_capture() {
    let (ok, peak, allocations) = close_with_budget(8 << 20).await;
    assert!(ok);
    assert_eq!(allocations, 4);
    let (ok, exact, allocations) = close_with_budget(peak as u64).await;
    assert!(ok, "exact measured lease schedule must fit inclusively");
    assert_eq!(exact, peak);
    assert_eq!(allocations, 4);
    let (ok, _, allocations) = close_with_budget(peak as u64 - 1).await;
    assert!(!ok);
    assert_eq!(
        allocations, 2,
        "B-1 must reject before compact schema/array target allocation"
    );
}
