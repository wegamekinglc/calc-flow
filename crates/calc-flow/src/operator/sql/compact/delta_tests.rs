use std::io::Cursor;

use datafusion::arrow::{
    array::{ArrayRef, Int64Array, StringArray},
    ipc::{reader::FileReader, writer::FileWriter},
    record_batch::RecordBatch,
};

use super::storage_tests::{input, operator, process, rows, same_snapshot};
use super::*;
use crate::{
    Batch, CalcFlowError, CancellationToken, DataFusionConfig, DataFusionRuntime, Epoch, JsonMap,
    OperatorStateSnapshot, StateSegment, StreamJobContext, operator::StreamOperator,
};
use datafusion::common::ScalarValue;
use std::{collections::BTreeMap, sync::Arc};

const QUERY: &str = "SELECT key, SUM(value) AS total, COUNT(*) AS rows FROM events GROUP BY key";

fn job() -> StreamJobContext {
    StreamJobContext::new(
        61,
        "compact",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

async fn seed(state: &mut SqlOperator, context: &StreamOperatorContext<'_>) -> Vec<Batch> {
    let names = (0..64)
        .map(|index| format!("key-{index:02}"))
        .collect::<Vec<_>>();
    let keys = names
        .iter()
        .map(|key| Some(key.as_str()))
        .collect::<Vec<_>>();
    let values = (0..64).map(Some).collect::<Vec<_>>();
    drop(process(state, input(false, &keys, &values, 0), context).await);
    vec![input(false, &keys, &values, 0)]
}

fn delta_ids(snapshot: &OperatorStateSnapshot) -> Vec<String> {
    snapshot
        .segments
        .keys()
        .filter(|id| id.starts_with("group-delta-"))
        .cloned()
        .collect()
}

struct Reject;

#[async_trait::async_trait]
impl StreamCollector for Reject {
    async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
        Err(CalcFlowError::Operator {
            node_id: "collector".into(),
            message: "reject output".into(),
        })
    }
}

#[tokio::test]
async fn test_compact_delta_refused_new_group_refunds_and_keeps_capture() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "compact", None);
    let mut state = operator(false, false);
    let mut history = seed(&mut state, &context).await;
    let before = state.checkpoint(Epoch::INITIAL).unwrap();
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let basis = pool.reserved();
    assert!(
        state
            .process_data(
                "events",
                input(false, &[Some("new")], &[Some(3)], 1),
                &context,
                &mut Reject
            )
            .await
            .is_err()
    );
    assert_eq!(
        pool.reserved(),
        basis,
        "refused output must release staged group capacity"
    );
    same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
    history.push(input(false, &[Some("new")], &[Some(3)], 1));
    let actual = process(
        &mut state,
        input(false, &[Some("new")], &[Some(3)], 1),
        &context,
    )
    .await;
    assert_prefix(&actual, &history).await;
    let after = state.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(delta_ids(&after).len(), 1);
    drop((state, actual, before, after, history));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_compact_delta_repeated_groups_compact_and_cold_continue() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "compact", None);
    let mut state = operator(false, false);
    let mut history = seed(&mut state, &context).await;
    let mut previous = state.checkpoint(Epoch::INITIAL).unwrap();
    for sequence in 1..=34 {
        for _ in 0..2 {
            history.push(input(
                false,
                &[Some("key-07"), None],
                &[Some(5), None],
                sequence,
            ));
            let actual = process(
                &mut state,
                input(false, &[Some("key-07"), None], &[Some(5), None], sequence),
                &context,
            )
            .await;
            assert_prefix(&actual, &history).await;
        }
        if sequence % 2 == 0 {
            state.prepare_compact_capture_async(&context).await.unwrap();
        }
        let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
        let frames = if sequence <= 32 {
            usize::try_from(sequence).unwrap()
        } else {
            usize::try_from(sequence).unwrap() - 33
        };
        assert_eq!(delta_ids(&snapshot).len(), frames);
        assert_eq!(
            Arc::ptr_eq(
                &previous.segments["group-state"].bytes_arc(),
                &snapshot.segments["group-state"].bytes_arc()
            ),
            sequence != 33
        );
        let mut restored = operator(false, false);
        restored.restore(&snapshot).unwrap();
        same_snapshot(&snapshot, &restored.checkpoint(Epoch::INITIAL).unwrap());
        state = restored;
        previous = snapshot;
    }
    history.push(input(false, &[Some("new"), None], &[Some(3), Some(9)], 35));
    let actual = process(
        &mut state,
        input(false, &[Some("new"), None], &[Some(3), Some(9)], 35),
        &context,
    )
    .await;
    assert_prefix(&actual, &history).await;
}

fn replace_control(snapshot: &mut OperatorStateSnapshot, value: &serde_json::Value) {
    let segment = StateSegment::new(serde_json::to_vec(value).unwrap());
    snapshot
        .inline_metadata
        .insert("control_sha256".into(), serde_json::json!(segment.sha256()));
    snapshot.segments.insert("control".into(), segment);
}

fn replace_delta(snapshot: &mut OperatorStateSnapshot, columns: Vec<ArrayRef>) {
    let id = delta_ids(snapshot).remove(0);
    let record = FileReader::try_new(Cursor::new(snapshot.segments[&id].bytes()), None)
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let fields = record
        .schema()
        .fields()
        .iter()
        .zip(&columns)
        .map(|(field, array)| {
            field
                .as_ref()
                .clone()
                .with_nullable(field.is_nullable() || array.null_count() != 0)
        })
        .collect::<Vec<_>>();
    let schema = Arc::new(datafusion::arrow::datatypes::Schema::new_with_metadata(
        fields,
        record.schema().metadata().clone(),
    ));
    let record = RecordBatch::try_new(schema, columns).unwrap();
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, record.schema().as_ref()).unwrap();
    writer.write(&record).unwrap();
    writer.finish().unwrap();
    drop(writer);
    let segment = StateSegment::new(bytes);
    let mut control: serde_json::Value =
        serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
    control["group_log"]["frames"][0]["sha256"] = serde_json::json!(segment.sha256());
    control["group_log"]["frames"][0]["groups"] = serde_json::json!(record.num_rows());
    snapshot.segments.insert(id, segment);
    replace_control(snapshot, &control);
}

#[tokio::test]
async fn test_compact_delta_invalid_states_with_valid_digests_refund_restore() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "compact", None);
    let mut state = operator(false, false);
    drop(seed(&mut state, &context).await);
    drop(state.checkpoint(Epoch::INITIAL).unwrap());
    drop(
        process(
            &mut state,
            input(false, &[Some("key-07")], &[Some(5)], 1),
            &context,
        )
        .await,
    );
    let before = state.checkpoint(Epoch::INITIAL).unwrap();
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    for counts in [
        vec![Some(-1)],
        vec![Some(0)],
        vec![Some(3)],
        vec![None],
        vec![Some(2), Some(2)],
    ] {
        let length = counts.len();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec!["key-07"; length])),
            Arc::new(Int64Array::from(vec![12; length])),
            Arc::new(Int64Array::from(counts)),
        ];
        let mut malformed = before.clone();
        replace_delta(&mut malformed, columns);
        let basis = pool.reserved();
        assert!(state.restore(&malformed).is_err());
        assert_eq!(pool.reserved(), basis);
        same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
    }
}

#[tokio::test]
async fn test_compact_delta_carried_frames_release_previous_capture_credit() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "compact", None);
    let mut state = operator(false, false);
    drop(seed(&mut state, &context).await);
    drop(state.checkpoint(Epoch::INITIAL).unwrap());
    drop(
        process(
            &mut state,
            input(false, &[Some("key-07")], &[Some(5)], 1),
            &context,
        )
        .await,
    );
    let initial = state.checkpoint(Epoch::INITIAL).unwrap();
    let id = delta_ids(&initial).remove(0);
    let base = initial.segments["group-state"].bytes_arc();
    let frame = initial.segments[&id].bytes_arc();
    drop(initial);
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let basis = pool.reserved();
    for sequence in 2..10 {
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
        drop(process(&mut state, input(false, &[], &[], sequence), &context).await);
        if sequence % 2 == 0 {
            state.prepare_compact_capture_async(&context).await.unwrap();
        }
        let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
        assert!(Arc::ptr_eq(
            &base,
            &snapshot.segments["group-state"].bytes_arc()
        ));
        assert!(Arc::ptr_eq(&frame, &snapshot.segments[&id].bytes_arc()));
        drop(snapshot);
        assert!(previous.upgrade().is_none());
        assert_eq!(pool.reserved(), basis);
    }
    drop((state, base, frame));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_compact_delta_corrupt_inventory_and_control_leave_live_state_unchanged() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "compact", None);
    let mut state = operator(false, false);
    drop(seed(&mut state, &context).await);
    drop(state.checkpoint(Epoch::INITIAL).unwrap());
    for sequence in 1..=2 {
        drop(
            process(
                &mut state,
                input(false, &[Some("key-07")], &[Some(5)], sequence),
                &context,
            )
            .await,
        );
        drop(state.checkpoint(Epoch::INITIAL).unwrap());
    }
    let before = state.checkpoint(Epoch::INITIAL).unwrap();
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let ids = delta_ids(&before);
    let mut malformed = Vec::new();
    let mut missing = before.clone();
    missing.segments.remove(&ids[0]);
    malformed.push(missing);
    let mut extra = before.clone();
    extra.segments.insert(
        "group-delta-unlisted".into(),
        before.segments[&ids[0]].clone(),
    );
    malformed.push(extra);
    for change in 0..7 {
        let mut snapshot = before.clone();
        let mut value: serde_json::Value =
            serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
        match change {
            0 => {
                value.as_object_mut().unwrap().remove("group_log");
            }
            1 => {
                value["group_log"]["frames"]
                    .as_array_mut()
                    .unwrap()
                    .reverse();
            }
            2 => {
                value["group_log"]["frames"][0]["groups"] = serde_json::json!(2);
            }
            3 => {
                value["group_log"]["frames"][0]["sha256"] = serde_json::json!("0".repeat(64));
            }
            4 => {
                value["group_log"]["generation"] = serde_json::json!(99);
            }
            5 => {
                value["group_log"]["base_ledger"]["rows"] = serde_json::json!(63);
            }
            _ => {
                value["group_log"]["frames"][1]["ledger"]["rows"] = serde_json::json!(65);
            }
        }
        replace_control(&mut snapshot, &value);
        malformed.push(snapshot);
    }
    for snapshot in malformed {
        let basis = pool.reserved();
        assert!(state.restore(&snapshot).is_err());
        assert_eq!(pool.reserved(), basis);
        same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
    }
}

#[tokio::test]
async fn test_compact_delta_cancelled_capture_preserves_dirty_updates() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "compact", None);
    let mut state = operator(false, false);
    let mut history = seed(&mut state, &context).await;
    let before = state.checkpoint(Epoch::INITIAL).unwrap();
    history.push(input(false, &[Some("key-07")], &[Some(5)], 1));
    drop(
        process(
            &mut state,
            input(false, &[Some("key-07")], &[Some(5)], 1),
            &context,
        )
        .await,
    );
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let basis = pool.reserved();
    for stop in 1..=8 {
        let calls = std::cell::Cell::new(0);
        assert!(
            state
                .prepare_compact_capture(&|| {
                    calls.set(calls.get() + 1);
                    if calls.get() == stop {
                        Err(CalcFlowError::Cancelled {
                            run_id: "capture".into(),
                        })
                    } else {
                        Ok(())
                    }
                })
                .is_err()
        );
        assert_eq!(pool.reserved(), basis);
        same_snapshot(
            &before,
            &state
                .compact
                .as_ref()
                .unwrap()
                .capture
                .as_ref()
                .unwrap()
                .snapshot,
        );
    }
    let after = state.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(delta_rows(&after)[0][1], ScalarValue::Int64(Some(12)));
    let mut restored = operator(false, false);
    restored.restore(&after).unwrap();
    history.push(input(false, &[], &[], 2));
    let actual = process(&mut restored, input(false, &[], &[], 2), &context).await;
    assert_prefix(&actual, &history).await;
}

async fn assert_prefix(actual: &Batch, history: &[Batch]) {
    let batch = Batch::table(
        history
            .iter()
            .flat_map(|batch| batch.table_payload().unwrap().batches().to_vec())
            .collect(),
        history.last().unwrap().metadata().clone(),
    )
    .unwrap();
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            QUERY,
            &BTreeMap::from([("events".into(), batch)]),
            Some("oracle"),
        )
        .await
        .unwrap();
    assert_eq!(rows(actual), rows(&expected));
    assert_eq!(actual.metadata(), expected.metadata());
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
}

fn delta_rows(snapshot: &OperatorStateSnapshot) -> Vec<Vec<ScalarValue>> {
    let segments = snapshot
        .segments
        .iter()
        .filter(|(id, _)| id.starts_with("group-delta-"))
        .collect::<Vec<_>>();
    assert_eq!(
        segments.len(),
        1,
        "one sparse update must encode one delta segment"
    );
    FileReader::try_new(Cursor::new(segments[0].1.bytes()), None)
        .unwrap()
        .flat_map(|record| {
            let record = record.unwrap();
            (0..record.num_rows())
                .map(|row| {
                    record
                        .columns()
                        .iter()
                        .map(|array| ScalarValue::try_from_array(array, row).unwrap())
                        .collect()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[tokio::test]
async fn test_compact_sparse_update_captures_only_changed_group_and_restores() {
    let job = StreamJobContext::new(
        61,
        "compact",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "compact", None);
    let mut state = operator(false, false);
    let names = (0..64)
        .map(|index| format!("key-{index:02}"))
        .collect::<Vec<_>>();
    let keys = names
        .iter()
        .map(|key| Some(key.as_str()))
        .collect::<Vec<_>>();
    let values = (0..64).map(Some).collect::<Vec<_>>();
    let first = input(false, &keys, &values, 0);
    let mut history = vec![input(false, &keys, &values, 0)];
    let actual = process(&mut state, first, &context).await;
    assert_prefix(&actual, &history).await;
    drop(actual);
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let before = state.checkpoint(Epoch::INITIAL).unwrap();
    let second = input(false, &[Some("key-07")], &[Some(5)], 1);
    history.push(input(false, &[Some("key-07")], &[Some(5)], 1));
    let actual = process(&mut state, second, &context).await;
    assert_prefix(&actual, &history).await;
    drop(actual);
    state.prepare_compact_capture_async(&context).await.unwrap();
    let after = state.checkpoint(Epoch::INITIAL).unwrap();
    assert!(
        Arc::ptr_eq(
            &before.segments["group-state"].bytes_arc(),
            &after.segments["group-state"].bytes_arc()
        ),
        "sparse capture must carry the unchanged base allocation"
    );
    assert_eq!(
        delta_rows(&after),
        vec![vec![
            ScalarValue::Utf8(Some("key-07".into())),
            ScalarValue::Int64(Some(12)),
            ScalarValue::Int64(Some(2))
        ]]
    );
    drop(state);
    let mut recovered = operator(false, false);
    StreamOperator::restore(&mut recovered, &after).unwrap();
    let continuation = input(
        false,
        &[Some("key-07"), Some("new")],
        &[Some(10), Some(3)],
        2,
    );
    history.push(input(
        false,
        &[Some("key-07"), Some("new")],
        &[Some(10), Some(3)],
        2,
    ));
    let actual = process(&mut recovered, continuation, &context).await;
    assert_prefix(&actual, &history).await;
    drop((actual, recovered, before, after, history));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}
