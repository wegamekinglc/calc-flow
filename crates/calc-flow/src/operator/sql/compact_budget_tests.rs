use super::super::{SqlOperator, encode_sql_state};
use crate::{
    Batch, BatchKind, BatchMetadata, Epoch, JsonMap, OperatorMetadata, Port, Result, StateBudget,
    StreamOperator, StreamOperatorContext,
};
use crate::{CancellationToken, EdgeCollector, StreamJobContext};
use datafusion::arrow::array::{ArrayRef, Int64Array, StringArray};
use datafusion::arrow::record_batch::RecordBatch;
use std::sync::Arc;

const QUERY: &str = "SELECT k, LAST_VALUE(v ORDER BY t) AS v FROM input GROUP BY k";
const ROOM: u64 = 16 << 20;

fn batch(keys: &[&str], value: i64) -> Batch {
    let record = RecordBatch::try_from_iter(vec![
        ("k", Arc::new(StringArray::from(keys.to_vec())) as ArrayRef),
        (
            "v",
            Arc::new(Int64Array::from(vec![value; keys.len()])) as ArrayRef,
        ),
        (
            "t",
            Arc::new(Int64Array::from(vec![value; keys.len()])) as ArrayRef,
        ),
    ])
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        440,
        "budget",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn operator(query: &str, groups: u64, bytes: u64) -> SqlOperator {
    let mut op = SqlOperator::new("budget", query, vec!["input".into()], vec![]).unwrap();
    op.set_compact_state_budget(Some(StateBudget::new(groups, bytes).unwrap()))
        .unwrap();
    op
}

async fn process(
    op: &mut SqlOperator,
    input: Batch,
    context: &StreamOperatorContext<'_>,
) -> Result<()> {
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    let result = op.process_data("input", input, context, &mut output).await;
    if result.is_err() {
        assert!(output.drain("output").is_empty());
    }
    result
}

fn state(op: &SqlOperator) -> Vec<u8> {
    let export = op
        .incremental
        .as_ref()
        .unwrap()
        .export_native_state("budget", || Ok(()))
        .unwrap();
    encode_sql_state(&Batch::table(export.records().to_vec(), BatchMetadata::default()).unwrap())
        .unwrap()
}

fn bytes(op: &SqlOperator) -> usize {
    op.retention_runtime().unwrap().compact_state_bytes()
}

#[tokio::test]
async fn test_compact_budget_group_limit_rejects_atomically() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "budget", None);
    let mut op = operator(QUERY, 2, ROOM);
    process(&mut op, batch(&["a", "b"], 1), &context)
        .await
        .unwrap();
    let before = state(&op);
    let fee = bytes(&op);
    let error = process(&mut op, batch(&["a", "c"], 2), &context)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("group budget"), "{error}");
    assert_eq!(state(&op), before);
    assert_eq!(bytes(&op), fee);
    assert_eq!(op.compact.as_ref().unwrap().ledger.rows, 2);
    process(&mut op, batch(&["b"], 3), &context).await.unwrap();
    assert_eq!(op.incremental.as_ref().unwrap().group_count(), 2);
}

#[tokio::test]
async fn test_compact_budget_counts_only_live_filtered_groups_and_checks_reconfiguration() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "budget", None);
    let mut op = operator(
        "SELECT k, COUNT(*) AS n FROM input WHERE v > 0 GROUP BY k",
        1,
        ROOM,
    );
    process(&mut op, batch(&["ignored-a", "ignored-b"], -1), &context)
        .await
        .unwrap();
    process(&mut op, batch(&["a", "a"], 1), &context)
        .await
        .unwrap();
    assert_eq!(op.incremental.as_ref().unwrap().group_count(), 1);
    let before = state(&op);
    let original_budget = op.compact_state_budget;
    assert!(
        op.set_compact_state_budget(Some(StateBudget::new(1, 1).unwrap()))
            .is_err()
    );
    assert_eq!(op.compact_state_budget, original_budget);
    assert_eq!(state(&op), before);
    process(&mut op, batch(&["a"], 2), &context).await.unwrap();
    assert!(process(&mut op, batch(&["b"], 2), &context).await.is_err());
}

#[tokio::test]
async fn test_compact_budget_byte_boundary_and_candidate_copy() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "budget", None);
    let mut probe = operator(QUERY, 2, ROOM);
    process(&mut probe, batch(&["a"], 1), &context)
        .await
        .unwrap();
    let first_peak = probe.retention_runtime().unwrap().take_compact_state_peak();
    for (limit, succeeds) in [(first_peak, true), (first_peak - 1, false)] {
        let mut op = operator(QUERY, 2, limit as u64);
        let result = process(&mut op, batch(&["a"], 1), &context).await;
        assert_eq!(result.is_ok(), succeeds, "limit {limit}: {result:?}");
        if !succeeds {
            assert!(op.compact.is_none());
            assert_eq!(bytes(&op), 0);
        }
    }
    let before = state(&probe);
    let fee = bytes(&probe);
    process(&mut probe, batch(&["a"], 2), &context)
        .await
        .unwrap();
    let update_peak = probe.retention_runtime().unwrap().take_compact_state_peak();
    assert!(
        update_peak > fee,
        "candidate must pay while the old winner lives"
    );
    let mut op = operator(QUERY, 2, ROOM);
    process(&mut op, batch(&["a"], 1), &context).await.unwrap();
    op.set_compact_state_budget(Some(StateBudget::new(2, (update_peak - 1) as u64).unwrap()))
        .unwrap();
    let error = process(&mut op, batch(&["a"], 2), &context)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("compact state budget"),
        "{error}"
    );
    assert_eq!(state(&op), before);
    assert_eq!(bytes(&op), fee);
    assert_eq!(op.compact.as_ref().unwrap().ledger.rows, 1);
}

#[tokio::test]
async fn test_compact_budget_counts_capture_and_held_snapshots() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "budget", None);
    let mut op = operator(QUERY, 8, ROOM);
    process(&mut op, batch(&["a", "b", "c", "d"], 1), &context)
        .await
        .unwrap();
    let fee = bytes(&op);
    let dirty = op.incremental.as_ref().unwrap().checkpoint_changes();
    op.set_compact_state_budget(Some(StateBudget::new(8, fee as u64).unwrap()))
        .unwrap();
    assert!(op.checkpoint(Epoch::new(1).unwrap()).is_err());
    assert!(op.prepare_checkpoint_async(&context).await.is_err());
    assert!(op.compact.as_ref().unwrap().capture.is_none());
    assert_eq!(bytes(&op), fee);
    assert_eq!(op.incremental.as_ref().unwrap().checkpoint_changes(), dirty);
    op.set_compact_state_budget(Some(StateBudget::new(8, ROOM).unwrap()))
        .unwrap();
    let snapshot = op.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let captured = bytes(&op);
    assert!(captured > fee);
    drop(op.checkpoint(Epoch::new(1).unwrap()).unwrap());
    assert_eq!(bytes(&op), captured);
    op.reset().unwrap();
    assert!(bytes(&op) > 0, "held snapshot retains its ownership lease");
    drop(snapshot);
    assert_eq!(bytes(&op), 0);
}

#[tokio::test]
async fn test_compact_budget_preserves_old_cumulative_input_quota() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "budget", None);
    let mut op = operator(QUERY, 1, ROOM);
    op.set_state_budget(StateBudget::new(2, ROOM).unwrap())
        .unwrap();
    process(&mut op, batch(&["a", "a"], 1), &context)
        .await
        .unwrap();
    let before = state(&op);
    let error = process(&mut op, batch(&["a"], 2), &context)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("retained input"), "{error}");
    assert_eq!(state(&op), before);
    assert_eq!(op.compact.as_ref().unwrap().ledger.rows, 2);
}

#[tokio::test]
async fn test_compact_budget_requires_supported_aggregate_but_allows_row_sql() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "budget", None);
    let unsupported = "SELECT k, COUNT(DISTINCT v) FROM input GROUP BY k";
    let mut op = operator(unsupported, 1, ROOM);
    let error = process(&mut op, batch(&["a"], 1), &context)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("requires compact execution"),
        "{error}"
    );
    assert!(op.retained.is_none());
    let mut declared = SqlOperator::new("declared", unsupported, vec!["input".into()], vec![])
        .unwrap()
        .with_ports(
            vec![
                Port::with_schema_ref(
                    "input",
                    BatchKind::Table,
                    true,
                    Some(batch(&["a"], 1).table_payload().unwrap().schema().clone()),
                )
                .unwrap(),
            ],
            Port::with_schema_ref("output", BatchKind::Table, true, None).unwrap(),
        )
        .unwrap();
    assert!(
        declared
            .set_compact_state_budget(Some(StateBudget::new(1, ROOM).unwrap()))
            .is_err()
    );
    assert!(declared.compact_state_budget.is_none());
    let mut row_sql = operator("SELECT v FROM input", 1, 1);
    process(&mut row_sql, batch(&["a", "b", "c"], 1), &context)
        .await
        .unwrap();
    assert_eq!(row_sql.compact_state_budget, None);
}

#[tokio::test]
async fn test_compact_budget_restores_full_and_delta_with_actual_charges() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "budget", None);
    let mut donor = operator(QUERY, 16, ROOM);
    process(
        &mut donor,
        batch(&["a", "b", "c", "d", "e", "f", "g", "h"], 1),
        &context,
    )
    .await
    .unwrap();
    let full = donor.checkpoint(Epoch::new(1).unwrap()).unwrap();
    process(&mut donor, batch(&["i"], 2), &context)
        .await
        .unwrap();
    let delta = donor.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert!(
        delta
            .segments
            .keys()
            .any(|id| id.starts_with("group-delta-"))
    );
    let mut restored = operator(QUERY, 16, ROOM);
    restored.restore(&full).unwrap();
    assert_eq!(restored.incremental.as_ref().unwrap().group_count(), 8);
    restored.restore(&delta).unwrap();
    assert_eq!(state(&restored), state(&donor));
    assert!(bytes(&restored) > 0);
    let mut limited = operator(QUERY, 8, ROOM);
    process(&mut limited, batch(&["original"], 1), &context)
        .await
        .unwrap();
    let before = state(&limited);
    let fee = bytes(&limited);
    assert!(
        limited
            .restore(&delta)
            .unwrap_err()
            .to_string()
            .contains("group budget")
    );
    assert_eq!(state(&limited), before);
    assert_eq!(bytes(&limited), fee);
    let mut byte_limited = operator(QUERY, 16, (bytes(&restored) - 1) as u64);
    assert!(byte_limited.restore(&delta).is_err());
    assert!(byte_limited.compact.is_none());
    assert_eq!(bytes(&byte_limited), 0);
}

#[tokio::test]
async fn test_compact_budget_rejects_nonempty_retained_restore_without_losing_history() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "budget", None);
    let query = "SELECT k, COUNT(DISTINCT v) FROM input GROUP BY k";
    let mut donor = SqlOperator::new("budget", query, vec!["input".into()], vec![]).unwrap();
    process(&mut donor, batch(&["a"], 1), &context)
        .await
        .unwrap();
    let snapshot = donor.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let mut receiver = operator(query, 8, ROOM);
    assert!(
        receiver
            .restore(&snapshot)
            .unwrap_err()
            .to_string()
            .contains("nonempty retained checkpoint")
    );
    assert!(receiver.retained.is_none());
    assert_eq!(donor.retained.as_ref().unwrap().rows, 1);
}
