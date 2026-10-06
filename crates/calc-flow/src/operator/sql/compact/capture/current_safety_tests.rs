use std::mem::size_of;

use datafusion::{
    arrow::{datatypes::Schema, record_batch::RecordBatch},
    execution::memory_pool::{MemoryLimit, MemoryPool},
};
use serde_json::{Value, json};

use super::super::storage_tests::{input, operator, rows, same_snapshot, schema};
use super::*;
use crate::{
    CalcFlowError, CancellationToken, DataFusionConfig, DataFusionRuntime, EdgeCollector, Epoch,
    JsonMap, StreamJobContext, StreamOperatorContext,
    operator::{OperatorMetadata, StateBudget, StateSegment, StreamOperator},
};

async fn process(
    operator: &mut SqlOperator,
    batch: Batch,
    context: &StreamOperatorContext<'_>,
) -> Batch {
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", batch, context, &mut output)
        .await
        .unwrap();
    let result = output.drain("output");
    assert_eq!(result.len(), 1);
    assert!(operator.retained.is_none());
    assert!(operator.compact.is_some());
    result[0].as_data().unwrap().clone()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        61,
        "compact",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn changed_schema(batch: &Batch) -> Batch {
    let records = batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .map(|record| {
            let schema = Arc::new(Schema::new_with_metadata(
                record.schema().fields().clone(),
                [("origin".into(), "empty-different".into())].into(),
            ));
            RecordBatch::try_new(schema, record.columns().to_vec()).unwrap()
        })
        .collect();
    Batch::table(records, batch.metadata().clone()).unwrap()
}

fn latest() -> BatchMetadata {
    BatchMetadata::new(
        "latest-来源",
        97,
        JsonMap::from([(
            "nested".into(),
            json!({"timestamp": 19, "null": null, "flags": [true, false]}),
        )]),
    )
    .unwrap()
}

async fn oracle(sequence: u64) -> Batch {
    DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            "SELECT key, SUM(value) AS total, COUNT(*) AS rows FROM events GROUP BY key",
            &BTreeMap::from([(
                "events".into(),
                input(false, &[Some("a"), None], &[Some(2), None], sequence),
            )]),
            Some("independent-oracle"),
        )
        .await
        .unwrap()
}

fn assert_output(actual: &Batch, expected: &Batch, metadata: &BatchMetadata) {
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(rows(actual), rows(expected));
    assert_eq!(actual.metadata(), metadata);
    assert_eq!(actual.table_payload().unwrap().batches().len(), 1);
}

async fn empty_schema_case(restore: bool) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "compact", None);
    let mut state = operator(false, false);
    drop(
        process(
            &mut state,
            input(false, &[Some("a"), None], &[Some(2), None], 0),
            &context,
        )
        .await,
    );
    let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
    if restore {
        let mut recovered = operator(false, false);
        recovered.restore(&snapshot).unwrap();
        state = recovered;
    }
    let compact = state.compact.as_ref().unwrap();
    assert!(compact.projection().is_none());
    assert!(state.input_ports()[0].schema().is_none());
    let ledger = compact.ledger;
    let logical = compact.columns.logical.clone();
    let physical = compact.columns.physical.clone();
    state
        .set_state_budget(StateBudget::new(ledger.rows, ledger.bytes).unwrap())
        .unwrap();
    let expected = oracle(0).await;
    let metadata = latest();
    let empty = changed_schema(&input(false, &[], &[], 1)).with_metadata(metadata.clone());
    let actual = process(&mut state, empty, &context).await;
    assert_output(&actual, &expected, &metadata);
    let compact = state.compact.as_ref().unwrap();
    assert_eq!(compact.ledger, ledger);
    assert!(Arc::ptr_eq(&compact.columns.logical, &logical));
    assert!(Arc::ptr_eq(&compact.columns.physical, &physical));
    let recaptured = state.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(
        recaptured.segments["logical-schema"].bytes(),
        snapshot.segments["logical-schema"].bytes()
    );
    assert_eq!(
        recaptured.segments["group-state"].bytes(),
        snapshot.segments["group-state"].bytes()
    );
    let mut recovered = operator(false, false);
    recovered.restore(&recaptured).unwrap();
    let actual = process(
        &mut recovered,
        input(false, &[], &[], 2).with_metadata(metadata.clone()),
        &context,
    )
    .await;
    assert_output(&actual, &expected, &metadata);
}

#[tokio::test]
async fn test_current_compact_bound_none_empty_schema_metadata_is_ignored() {
    empty_schema_case(false).await;
}

#[tokio::test]
async fn test_current_compact_restored_none_empty_schema_metadata_is_ignored() {
    empty_schema_case(true).await;
}

#[tokio::test]
async fn test_current_compact_empty_schema_strict_boundaries() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "compact", None);
    for (wide, declared, empty) in [
        (false, true, true),
        (true, false, true),
        (false, false, false),
    ] {
        let mut state = operator(wide, declared);
        drop(
            process(
                &mut state,
                input(wide, &[Some("a")], &[Some(2)], 0),
                &context,
            )
            .await,
        );
        let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
        let ledger = state.compact.as_ref().unwrap().ledger;
        let incoming = if empty {
            input(wide, &[], &[], 1)
        } else {
            input(wide, &[Some("b")], &[Some(9)], 1)
        };
        let mut collector = EdgeCollector::new(state.output_ports().to_vec());
        let result = state
            .process_data(
                "events",
                changed_schema(&incoming),
                &context,
                &mut collector,
            )
            .await;
        if declared {
            assert!(matches!(result, Err(CalcFlowError::Compile { .. })));
        } else {
            assert!(matches!(result, Err(CalcFlowError::InvalidArgument { .. })));
        }
        assert!(collector.drain("output").is_empty());
        assert_eq!(state.compact.as_ref().unwrap().ledger, ledger);
        same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
    }
    let mut declared = operator(false, true);
    let mut collector = EdgeCollector::new(declared.output_ports().to_vec());
    assert!(
        declared
            .process_data(
                "events",
                changed_schema(&input(false, &[], &[], 0)),
                &context,
                &mut collector,
            )
            .await
            .is_err()
    );
    assert!(declared.compact.is_none());
    assert!(collector.drain("output").is_empty());
    let mut dynamic = operator(false, false);
    drop(
        process(
            &mut dynamic,
            changed_schema(&input(false, &[], &[], 0)),
            &context,
        )
        .await,
    );
    assert_ne!(
        dynamic.compact.as_ref().unwrap().columns.logical,
        schema(false)
    );
    let before = dynamic.checkpoint(Epoch::INITIAL).unwrap();
    let mut collector = EdgeCollector::new(dynamic.output_ports().to_vec());
    assert!(
        dynamic
            .process_data(
                "events",
                input(false, &[Some("a")], &[Some(2)], 1),
                &context,
                &mut collector,
            )
            .await
            .is_err()
    );
    assert!(collector.drain("output").is_empty());
    same_snapshot(&before, &dynamic.checkpoint(Epoch::INITIAL).unwrap());
}

#[tokio::test]
async fn test_current_compact_carried_state_releases_prior_capture_credit() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "compact", None);
    let mut state = operator(false, false);
    drop(
        process(
            &mut state,
            input(false, &[Some("a"), None], &[Some(2), None], 0),
            &context,
        )
        .await,
    );
    let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
    let payload = snapshot.segments["group-state"].bytes_arc();
    drop(snapshot);
    let pool = pool(&state);
    let basis = pool.reserved();
    for sequence in 1..10 {
        let previous = Arc::downgrade(
            state
                .compact
                .as_ref()
                .unwrap()
                .capture
                .as_ref()
                .unwrap()
                .checkpoint_fee_for_test(),
        );
        let batch = input(false, &[], &[], sequence);
        let metadata = batch.metadata().clone();
        drop(process(&mut state, batch, &context).await);
        if sequence % 2 == 0 {
            state.prepare_compact_capture_async(&context).await.unwrap();
        }
        let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
        assert!(Arc::ptr_eq(
            &payload,
            &snapshot.segments["group-state"].bytes_arc()
        ));
        let actual: BatchMetadata =
            serde_json::from_slice(snapshot.segments["batch-metadata"].bytes()).unwrap();
        assert_eq!(actual, metadata);
        drop(snapshot);
        assert!(
            previous.upgrade().is_none(),
            "carried state retains an old capture reservation"
        );
        assert_eq!(
            pool.reserved(),
            basis,
            "unchanged state accumulates reserved bytes"
        );
    }
    drop((state, payload));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

fn large_metadata() -> BatchMetadata {
    let values = (0..65_536)
        .map(|index| {
            if index % 2 == 0 {
                Value::Bool(true)
            } else {
                Value::Null
            }
        })
        .collect();
    BatchMetadata::new(
        "large",
        23,
        JsonMap::from([("large".into(), Value::Array(values))]),
    )
    .unwrap()
}

fn pool(state: &SqlOperator) -> Arc<dyn MemoryPool> {
    state.retention_runtime().unwrap().incremental_memory_pool()
}

fn actual_metadata_bytes(capture: &CompactCapture) -> usize {
    capture.metadata.attributes()["large"]
        .as_array()
        .unwrap()
        .capacity()
        * size_of::<Value>()
}

async fn ownerless_snapshot(context: &StreamOperatorContext<'_>) -> OperatorStateSnapshot {
    let mut source = operator(false, false);
    let metadata = large_metadata();
    drop(
        process(
            &mut source,
            input(false, &[Some("a"), None], &[Some(2), None], 0).with_metadata(metadata),
            context,
        )
        .await,
    );
    let paid = source.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(paid.inline_metadata["state_layout"], json!(3));
    let source_pool = pool(&source);
    let external = OperatorStateSnapshot {
        inline_metadata: paid.inline_metadata.clone(),
        segments: paid
            .segments
            .iter()
            .map(|(name, segment)| (name.clone(), StateSegment::new(segment.bytes().to_vec())))
            .collect(),
    };
    drop(paid);
    drop(source);
    assert_eq!(source_pool.reserved(), 0);
    external
}

#[tokio::test]
async fn test_current_compact_restored_capture_metadata_keeps_actual_paid_owner() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "compact", None);
    let external = ownerless_snapshot(&context).await;
    let mut state = operator(false, false);
    state.restore(&external).unwrap();
    let pool = pool(&state);
    let capture = state
        .compact
        .as_ref()
        .unwrap()
        .capture
        .as_ref()
        .unwrap()
        .clone();
    assert_eq!(capture.metadata, large_metadata());
    let actual_bytes = actual_metadata_bytes(&capture);
    let weak = Arc::downgrade(&capture);
    let lease = Arc::downgrade(capture.checkpoint_fee_for_test());
    let metadata = latest();
    let expected = oracle(0).await;
    let actual = process(
        &mut state,
        input(false, &[], &[], 1).with_metadata(metadata.clone()),
        &context,
    )
    .await;
    assert_output(&actual, &expected, &metadata);
    drop(actual);
    drop(expected);
    assert!(Arc::ptr_eq(
        state.compact.as_ref().unwrap().capture.as_ref().unwrap(),
        &capture
    ));
    let paid_after_empty = pool.reserved();
    state.reset().unwrap();
    drop(state);
    let paid_after_reset = pool.reserved();
    let snapshot = capture.snapshot.clone();
    drop(capture);
    assert!(weak.upgrade().is_none());
    assert!(lease.upgrade().is_some());
    assert!(pool.reserved() > 0);
    drop(snapshot);
    drop(external);
    assert!(lease.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
    assert!(
        paid_after_empty >= actual_bytes,
        "metadata survives tiny replacement: paid {paid_after_empty}, actual {actual_bytes}"
    );
    assert!(
        paid_after_reset >= actual_bytes,
        "capture survives reset: paid {paid_after_reset}, actual {actual_bytes}"
    );
}

#[tokio::test]
async fn test_current_compact_restore_pool_refusal_is_atomic_and_refunds() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "compact", None);
    let external = ownerless_snapshot(&context).await;
    let mut state = operator(false, false);
    drop(
        process(
            &mut state,
            input(false, &[Some("a"), None], &[Some(2), None], 0),
            &context,
        )
        .await,
    );
    let before = state.checkpoint(Epoch::INITIAL).unwrap();
    let pool = pool(&state);
    let baseline = pool.reserved();
    let hold = state
        .retention_runtime()
        .unwrap()
        .incremental_reservation("metadata-refusal-control");
    let MemoryLimit::Finite(limit) = pool.memory_limit() else {
        panic!("metadata refusal control requires a bounded actual pool");
    };
    hold.try_grow(limit - baseline - 65_536).unwrap();
    let charged = pool.reserved();
    let error = state.restore(&external).unwrap_err();
    assert!(matches!(&error, CalcFlowError::DataFusion { .. }));
    assert!(error.to_string().contains("Resources exhausted"), "{error}");
    assert_eq!(pool.reserved(), charged);
    drop(hold);
    assert_eq!(pool.reserved(), baseline);
    same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
    let metadata = latest();
    let actual = process(
        &mut state,
        input(false, &[], &[], 1).with_metadata(metadata.clone()),
        &context,
    )
    .await;
    let expected = oracle(0).await;
    assert_output(&actual, &expected, &metadata);
    drop(actual);
    drop(expected);
    state.reset().unwrap();
    drop(state);
    drop(before);
    drop(external);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_current_compact_normal_capture_pays_metadata_until_last_owner_drop() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "compact", None);
    let mut state = operator(false, false);
    drop(
        process(
            &mut state,
            input(false, &[Some("a"), None], &[Some(2), None], 0).with_metadata(large_metadata()),
            &context,
        )
        .await,
    );
    drop(state.checkpoint(Epoch::INITIAL).unwrap());
    let pool = pool(&state);
    let capture = state
        .compact
        .as_ref()
        .unwrap()
        .capture
        .as_ref()
        .unwrap()
        .clone();
    let capacity = actual_metadata_bytes(&capture);
    drop(
        process(
            &mut state,
            input(false, &[], &[], 1).with_metadata(latest()),
            &context,
        )
        .await,
    );
    state.reset().unwrap();
    drop(state);
    assert!(pool.reserved() >= capacity);
    drop(capture);
    assert_eq!(pool.reserved(), 0);
}
