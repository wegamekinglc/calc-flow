use std::{cell::Cell, collections::BTreeMap, sync::Arc};

use datafusion::{
    arrow::{
        array::{ArrayRef, Int64Array},
        record_batch::RecordBatch,
    },
    execution::memory_pool::MemoryPool,
};

use super::*;
use crate::{
    BatchMetadata, CalcFlowError, CancellationToken, DataFusionConfig, DataFusionRuntime,
    EdgeCollector, Epoch, JsonMap, StreamJobContext,
    operator::{OperatorMetadata, StreamOperator, StreamOperatorContext},
};
use serde_json::json;

const QUERY: &str = "SELECT SUM(value) AS total, COUNT(*) AS rows FROM events";

fn input(source: &str, values: &[i64]) -> Batch {
    let record = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from(values.to_vec())) as ArrayRef,
    )])
    .unwrap();
    Batch::table(
        vec![record],
        BatchMetadata::new(source, 0, JsonMap::new()).unwrap(),
    )
    .unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "sql-prepared-restore",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

async fn push(operator: &mut SqlOperator, batch: Batch, job: &StreamJobContext) -> Batch {
    let context = StreamOperatorContext::new(job, "totals", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("events", batch, &context, &mut collector)
        .await
        .unwrap();
    let emitted = collector.drain("output");
    assert_eq!(emitted.len(), 1);
    emitted[0].as_data().unwrap().clone()
}

async fn seeded(values: &[i64], job: &StreamJobContext) -> SqlOperator {
    let mut operator = SqlOperator::new("totals", QUERY, vec!["events".into()], vec![]).unwrap();
    let _output = push(&mut operator, input("direct-prepare-seed", values), job).await;
    operator
}

fn pool(operator: &SqlOperator) -> Arc<dyn MemoryPool> {
    operator
        .retention_runtime()
        .unwrap()
        .incremental_memory_pool()
}

fn same_capture(left: &OperatorStateSnapshot, right: &OperatorStateSnapshot) {
    assert_eq!(left.inline_metadata, right.inline_metadata);
    assert_eq!(
        left.segments.keys().collect::<Vec<_>>(),
        right.segments.keys().collect::<Vec<_>>()
    );
    for (name, segment) in &left.segments {
        assert_eq!(segment.bytes(), right.segments[name].bytes());
    }
}

#[tokio::test]
async fn test_prepared_current_restore_is_paid_and_installed_only_when_observed() {
    let job = job();
    let mut target = seeded(&[1, 2], &job).await;
    let mut source = seeded(&[10, 20], &job).await;
    let committed = target.checkpoint(Epoch::INITIAL).unwrap();
    let replacement = source.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(
        replacement
            .inline_metadata
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [
            "batch_metadata_sha256",
            "bytes",
            "query_sha256",
            "rows",
            "state_accounting",
            "state_layout"
        ]
    );
    assert_eq!(
        replacement
            .segments
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["batch-metadata", "input"]
    );
    let pool = pool(&target);
    let before = pool.reserved();
    let prepared = target.prepare_restore(&replacement, &|| Ok(())).unwrap();
    assert!(pool.reserved() > before);
    assert!(target.incremental.is_some());
    same_capture(&committed, &target.checkpoint(Epoch::INITIAL).unwrap());
    drop(prepared);
    assert_eq!(pool.reserved(), before);
    let prepared = target.prepare_restore(&replacement, &|| Ok(())).unwrap();
    target.install_restore(prepared);
    assert!(target.incremental.is_none());
    same_capture(&replacement, &target.checkpoint(Epoch::INITIAL).unwrap());
    let observed = push(&mut target, input("direct-prepare-continued", &[4]), &job).await;
    let oracle = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            QUERY,
            &BTreeMap::from([(
                "events".into(),
                input("direct-prepare-continued", &[10, 20, 4]),
            )]),
            Some("independent-prepared-prefix"),
        )
        .await
        .unwrap();
    assert_eq!(observed.metadata(), oracle.metadata());
    assert_eq!(
        observed.table_payload().unwrap().schema(),
        oracle.table_payload().unwrap().schema()
    );
    assert_eq!(
        observed.table_payload().unwrap().batches(),
        oracle.table_payload().unwrap().batches()
    );
}

#[tokio::test]
async fn test_cancelled_and_empty_prepared_restore_preserve_committed_state() {
    let job = job();
    let mut target = seeded(&[1, 2], &job).await;
    let mut source = seeded(&[10, 20], &job).await;
    let committed = target.checkpoint(Epoch::INITIAL).unwrap();
    let replacement = source.checkpoint(Epoch::INITIAL).unwrap();
    let pool = pool(&target);
    let before = pool.reserved();
    for stop in 1..=4 {
        let calls = Cell::new(0);
        let check = || {
            calls.set(calls.get() + 1);
            if calls.get() == stop {
                Err(CalcFlowError::Cancelled {
                    run_id: "sql-prepared-restore".into(),
                })
            } else {
                Ok(())
            }
        };
        let error = target.prepare_restore(&replacement, &check).err().unwrap();
        assert!(matches!(error, CalcFlowError::Cancelled { .. }));
        assert_eq!(calls.get(), stop);
        assert!(target.incremental.is_some());
        same_capture(&committed, &target.checkpoint(Epoch::INITIAL).unwrap());
        assert_eq!(pool.reserved(), before);
    }
    let empty = OperatorStateSnapshot {
        inline_metadata: JsonMap::new(),
        segments: BTreeMap::new(),
    };
    let prepared = target.prepare_restore(&empty, &|| Ok(())).unwrap();
    same_capture(&committed, &target.checkpoint(Epoch::INITIAL).unwrap());
    assert_eq!(pool.reserved(), before);
    drop(prepared);
    same_capture(&committed, &target.checkpoint(Epoch::INITIAL).unwrap());
    let prepared = target.prepare_restore(&empty, &|| Ok(())).unwrap();
    target.install_restore(prepared);
    assert!(target.retained.is_none());
    assert!(target.incremental.is_none());
    assert!(!target.incremental_checked);
    assert!(
        target
            .checkpoint(Epoch::INITIAL)
            .unwrap()
            .segments
            .is_empty()
    );
}

#[tokio::test]
async fn test_invalid_and_unfunded_prepared_restore_preserve_previous_state() {
    let job = job();
    let mut target = seeded(&[1, 2], &job).await;
    let mut source = seeded(&[10, 20], &job).await;
    let committed = target.checkpoint(Epoch::INITIAL).unwrap();
    let replacement = source.checkpoint(Epoch::INITIAL).unwrap();
    let pool = pool(&target);
    let before = pool.reserved();
    let mut invalid = replacement.clone();
    invalid.inline_metadata.insert("rows".into(), json!(999));
    assert!(matches!(
        target.prepare_restore(&invalid, &|| Ok(())),
        Err(CalcFlowError::Format { .. })
    ));
    same_capture(&committed, &target.checkpoint(Epoch::INITIAL).unwrap());
    assert_eq!(pool.reserved(), before);
    let pressure = target
        .retention_runtime()
        .unwrap()
        .incremental_reservation("prepared-restore-pressure");
    pressure.try_grow((1 << 30) - pool.reserved()).unwrap();
    assert!(matches!(
        target.prepare_restore(&replacement, &|| Ok(())),
        Err(CalcFlowError::DataFusion { .. })
    ));
    same_capture(&committed, &target.checkpoint(Epoch::INITIAL).unwrap());
    assert!(target.incremental.is_some());
    drop(pressure);
    assert_eq!(pool.reserved(), before);
    let prepared = target.prepare_restore(&replacement, &|| Ok(())).unwrap();
    target.install_restore(prepared);
    same_capture(&replacement, &target.checkpoint(Epoch::INITIAL).unwrap());
}
