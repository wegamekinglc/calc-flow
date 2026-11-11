//! Focused failure injection entry points for the single-alias replacement policy.
use super::*;
use crate::{BatchKind, BatchMetadata, CancellationToken, EdgeCollector, StreamJobContext};
use datafusion::arrow::{
    array::{Array, Int64Array},
    record_batch::RecordBatch,
};
use std::cell::Cell;

fn batch(parts: &[&[i64]]) -> Batch {
    Batch::table(
        parts
            .iter()
            .map(|values| {
                RecordBatch::try_from_iter(vec![(
                    "x",
                    Arc::new(Int64Array::from(values.to_vec())) as Arc<dyn Array>,
                )])
                .unwrap()
            })
            .collect(),
        BatchMetadata::default(),
    )
    .unwrap()
}
async fn operator(query: &str) -> SnapshotSqlOperator {
    let schema = batch(&[&[]]).table_payload().unwrap().schema().clone();
    let declaration = SqlOperator::new("replacement", query, vec!["p".into()], vec![]).unwrap();
    let output = declaration
        .infer_schema(&BTreeMap::from([("p".into(), schema.clone())]))
        .await
        .unwrap();
    let declaration = declaration
        .with_ports(
            vec![Port::with_schema_ref("p", BatchKind::Table, true, Some(schema)).unwrap()],
            Port::with_schema_ref("output", BatchKind::Table, true, Some(output)).unwrap(),
        )
        .unwrap();
    let mut operator = SnapshotSqlOperator::for_single_alias(declaration).unwrap();
    operator
        .set_budget(Some(StateBudget::new(100, 1 << 24).unwrap()))
        .unwrap();
    operator
}
fn job() -> StreamJobContext {
    StreamJobContext::new(
        1,
        "replacement",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}
fn drain(out: &mut EdgeCollector) -> Vec<Batch> {
    out.drain("output")
        .iter()
        .map(|m| m.as_data().unwrap().clone())
        .collect()
}
fn signature(snapshot: &OperatorStateSnapshot) -> (JsonMap, BTreeMap<String, String>) {
    (
        snapshot.inline_metadata.clone(),
        snapshot
            .segments
            .iter()
            .map(|(name, s)| (name.clone(), s.sha256().into()))
            .collect(),
    )
}
fn sum(batch: &Batch) -> Option<i64> {
    let a = batch.table_payload().unwrap().batches()[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    (!a.is_null(0)).then(|| a.value(0))
}
struct Reject;
#[async_trait]
impl StreamCollector for Reject {
    async fn emit(&mut self, _: &str, _: Batch) -> Result<()> {
        Err(state_error("injected emit rejection"))
    }
}
#[tokio::test]
async fn replacement_commit_restore_and_failed_candidates() {
    let mut op = operator("SELECT SUM(x) AS total FROM p").await;
    let active_job = job();
    let context = StreamOperatorContext::new(&active_job, "replacement", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("p", batch(&[&[1]]), &context, &mut out)
        .await
        .unwrap();
    assert_eq!(sum(&drain(&mut out)[0]), Some(1));
    let saved = op.checkpoint(Epoch::INITIAL).unwrap();
    let control: serde_json::Value =
        serde_json::from_slice(saved.segments["control"].bytes()).unwrap();
    assert_eq!(control["state_kind"], "sql_replacement");
    assert_eq!(control["state_layout"], 6);
    assert_eq!(control["initialized"], true);
    let before = op.pool.as_ref().unwrap().reserved();
    assert!(
        op.process_data("p", batch(&[&[9]]), &context, &mut Reject)
            .await
            .is_err()
    );
    assert_eq!(op.pool.as_ref().unwrap().reserved(), before);
    assert_eq!(
        signature(&op.checkpoint(Epoch::INITIAL).unwrap()),
        signature(&saved)
    );
    op.set_budget(Some(StateBudget::new(1, 1 << 24).unwrap()))
        .unwrap();
    assert!(
        op.process_data("p", batch(&[&[2, 3]]), &context, &mut out)
            .await
            .is_err()
    );
    assert_eq!(
        signature(&op.checkpoint(Epoch::INITIAL).unwrap()),
        signature(&saved)
    );
    op.set_budget(Some(StateBudget::new(100, 1 << 24).unwrap()))
        .unwrap();
    let token = CancellationToken::new();
    token.cancel();
    let cancelled = StreamJobContext::new(2, "cancelled", JsonMap::new(), None, token);
    let cancelled_context = StreamOperatorContext::new(&cancelled, "replacement", None);
    assert!(
        op.process_data("p", batch(&[&[8]]), &cancelled_context, &mut out)
            .await
            .is_err()
    );
    assert_eq!(
        signature(&op.checkpoint(Epoch::INITIAL).unwrap()),
        signature(&saved)
    );
    let mut restored = operator("SELECT SUM(x) AS total FROM p").await;
    restored.restore(&saved).unwrap();
    for target in [&mut op, &mut restored] {
        target
            .process_data("p", batch(&[&[2], &[3]]), &context, &mut out)
            .await
            .unwrap();
        let events = drain(&mut out);
        assert_eq!(events.len(), 1);
        assert_eq!(sum(&events[0]), Some(5));
        target
            .process_data("p", batch(&[&[]]), &context, &mut out)
            .await
            .unwrap();
        assert_eq!(sum(&drain(&mut out)[0]), None);
    }
}
#[tokio::test]
async fn replacement_initialization_and_query_failure_are_atomic() {
    let mut op = operator("SELECT SUM(x) AS total FROM p").await;
    let active_job = job();
    let context = StreamOperatorContext::new(&active_job, "replacement", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    let uninitialized = op.checkpoint(Epoch::INITIAL).unwrap();
    assert!(
        op.process_data("p", batch(&[&[1]]), &context, &mut Reject)
            .await
            .is_err()
    );
    assert_eq!(
        signature(&op.checkpoint(Epoch::INITIAL).unwrap()),
        signature(&uninitialized)
    );
    op.process_data("p", batch(&[&[]]), &context, &mut out)
        .await
        .unwrap();
    let empty = op.checkpoint(Epoch::INITIAL).unwrap();
    assert_ne!(signature(&empty), signature(&uninitialized));
    let good_query = op.validated.clone();
    // Inject a failing query at the executor boundary, leaving declaration identity unchanged.
    op.validated = parse_select_query("SELECT CAST('bad' AS BIGINT) FROM p").unwrap();
    assert!(
        op.process_data("p", batch(&[&[1]]), &context, &mut out)
            .await
            .is_err()
    );
    assert_eq!(
        signature(&op.checkpoint(Epoch::INITIAL).unwrap()),
        signature(&empty)
    );
    op.validated = good_query;
    op.process_data("p", batch(&[&[6]]), &context, &mut out)
        .await
        .unwrap();
    assert_eq!(sum(&drain(&mut out).pop().unwrap()), Some(6));
    let mut fresh = operator("SELECT SUM(x) AS total FROM p").await;
    fresh.restore(&uninitialized).unwrap();
    assert!(fresh.relations[0].is_none());
    fresh.restore(&empty).unwrap();
    assert_eq!(fresh.relations[0].as_ref().unwrap().batch.num_rows(), 0);
    for stop in 1..20 {
        let calls = Cell::new(0);
        let check = || {
            calls.set(calls.get() + 1);
            if calls.get() == stop {
                Err(state_error("injected preparation failure"))
            } else {
                Ok(())
            }
        };
        let before = fresh.pool.as_ref().unwrap().reserved();
        drop(fresh.prepare_restore(&op.checkpoint(Epoch::INITIAL).unwrap(), &check));
        assert_eq!(fresh.pool.as_ref().unwrap().reserved(), before);
        fresh.capture = None;
        let before = fresh.pool.as_ref().unwrap().reserved();
        calls.set(0);
        drop(fresh.prepare_checkpoint_work(&check));
        assert_eq!(fresh.pool.as_ref().unwrap().reserved(), before);
    }
}
#[tokio::test]
async fn replacement_byte_ledger_is_inclusive_and_pays_old_candidate_output() {
    let mut op = operator("SELECT SUM(x) AS total FROM p").await;
    let active_job = job();
    let context = StreamOperatorContext::new(&active_job, "replacement", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    let input = batch(&[&[1]]);
    op.process_data("p", input.clone(), &context, &mut out)
        .await
        .unwrap();
    let produced = drain(&mut out).pop().unwrap();
    let pool = op.pool.as_ref().unwrap().clone();
    let input_charge = backing_bytes(&input).unwrap() + 4096;
    assert_eq!(pool.reserved(), input_charge);
    let output_charge = backing_bytes(&produced).unwrap() + 4096;
    let limit = 2 * input_charge + output_charge;
    op.set_budget(Some(StateBudget::new(1, (limit - 1) as u64).unwrap()))
        .unwrap();
    assert!(
        op.process_data("p", input.clone(), &context, &mut out)
            .await
            .is_err()
    );
    assert_eq!(pool.reserved(), input_charge);
    op.set_budget(Some(StateBudget::new(1, limit as u64).unwrap()))
        .unwrap();
    op.process_data("p", input, &context, &mut out)
        .await
        .unwrap();
    assert_eq!(pool.reserved(), input_charge);
    op.set_budget(Some(StateBudget::new(1, 1 << 24).unwrap()))
        .unwrap();
    let saved = op.checkpoint(Epoch::INITIAL).unwrap();
    let with_capture = pool.reserved();
    assert!(with_capture > input_charge);
    op.set_budget(Some(
        StateBudget::new(1, (with_capture + input_charge + output_charge - 1) as u64).unwrap(),
    ))
    .unwrap();
    assert!(
        op.process_data("p", batch(&[&[2]]), &context, &mut out)
            .await
            .is_err()
    );
    assert_eq!(pool.reserved(), with_capture);
    op.reset().unwrap();
    assert!(pool.reserved() > 0);
    drop(saved);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn replacement_restore_rejects_legacy_unknown_and_insufficient_budget() {
    let mut op = operator("SELECT SUM(x) AS total FROM p").await;
    let active_job = job();
    let context = StreamOperatorContext::new(&active_job, "replacement", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("p", batch(&[&[1, 2]]), &context, &mut out)
        .await
        .unwrap();
    let saved = op.checkpoint(Epoch::INITIAL).unwrap();
    let mut legacy = (*op.declaration).clone();
    assert!(legacy.restore(&saved).is_err());
    legacy
        .process_data("p", batch(&[&[1]]), &context, &mut out)
        .await
        .unwrap();
    let old = legacy.checkpoint(Epoch::INITIAL).unwrap();
    assert!(op.restore(&old).is_err());
    for (field, value) in [
        ("state_kind", serde_json::json!("future")),
        ("state_layout", serde_json::json!(7)),
        ("state_accounting", serde_json::json!(2)),
        ("initialized", serde_json::json!(false)),
    ] {
        let mut bad = saved.clone();
        let mut control: serde_json::Value =
            serde_json::from_slice(bad.segments["control"].bytes()).unwrap();
        control[field] = value;
        let segment = crate::StateSegment::new(serde_json::to_vec(&control).unwrap());
        bad.inline_metadata
            .insert("control_sha256".into(), serde_json::json!(segment.sha256()));
        bad.segments.insert("control".into(), segment);
        assert!(op.restore(&bad).is_err());
        assert_eq!(
            signature(&op.checkpoint(Epoch::INITIAL).unwrap()),
            signature(&saved)
        );
    }
    let mut target = operator("SELECT SUM(x) AS total FROM p").await;
    target
        .process_data("p", batch(&[&[9]]), &context, &mut out)
        .await
        .unwrap();
    target
        .set_budget(Some(StateBudget::new(1, 1 << 24).unwrap()))
        .unwrap();
    assert!(target.restore(&saved).is_err());
    assert_eq!(sum(&target.relations[0].as_ref().unwrap().batch), Some(9));
}

#[tokio::test]
async fn replacement_grouped_empty_and_whole_output_edge_limits() {
    let mut op = operator("SELECT x, COUNT(*) AS n FROM p GROUP BY x").await;
    let active_job = job();
    let context = StreamOperatorContext::new(&active_job, "replacement", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("p", batch(&[&[1, 2]]), &context, &mut out)
        .await
        .unwrap();
    assert_eq!(drain(&mut out)[0].num_rows(), 2);
    let saved = op.checkpoint(Epoch::INITIAL).unwrap();
    let produced = op
        .runtime
        .runtime()
        .unwrap()
        .snapshot_sql(
            &op.validated,
            &BTreeMap::from([("p".into(), batch(&[&[3, 4]]))]),
            "replacement",
            crate::EdgeBudget::new(10, 1 << 20).unwrap(),
        )
        .await
        .unwrap();
    let exact = produced.estimated_bytes().unwrap();
    let too_small = StreamOperatorContext::new(&active_job, "replacement", None)
        .with_output_budget(crate::EdgeBudget::new(2, exact - 1).unwrap());
    assert!(
        op.process_data("p", batch(&[&[3, 4]]), &too_small, &mut out)
            .await
            .is_err()
    );
    assert_eq!(
        signature(&op.checkpoint(Epoch::INITIAL).unwrap()),
        signature(&saved)
    );
    let exact_context = StreamOperatorContext::new(&active_job, "replacement", None)
        .with_output_budget(crate::EdgeBudget::new(2, exact).unwrap());
    op.process_data("p", batch(&[&[3, 4]]), &exact_context, &mut out)
        .await
        .unwrap();
    op.process_data("p", batch(&[&[]]), &context, &mut out)
        .await
        .unwrap();
    assert_eq!(drain(&mut out).pop().unwrap().num_rows(), 0);
    let empty = op.checkpoint(Epoch::INITIAL).unwrap();
    op.on_end(&context, &mut out).await.unwrap();
    assert!(drain(&mut out).is_empty());
    assert_eq!(
        signature(&op.checkpoint(Epoch::INITIAL).unwrap()),
        signature(&empty)
    );
}

async fn replacement_sql() -> SqlOperator {
    let mut op = *operator("SELECT SUM(x) AS total FROM p").await.declaration;
    op.enable_replacement().unwrap();
    op.set_replacement_budget(StateBudget::new(100, 1 << 24).unwrap())
        .unwrap();
    op
}

#[tokio::test]
async fn test_replacement_legacy_input_budget_limits_each_candidate() {
    let mut op = replacement_sql().await;
    let active_job = job();
    let context = StreamOperatorContext::new(&active_job, "replacement", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("p", batch(&[&[1]]), &context, &mut out)
        .await
        .unwrap();
    assert_eq!(sum(&drain(&mut out)[0]), Some(1));
    let saved = op.checkpoint(Epoch::INITIAL).unwrap();
    let candidate = batch(&[&[2, 3]]);
    let bytes = candidate.estimated_bytes().unwrap() as u64;
    for (budget, field) in [
        (StateBudget::new(1, bytes).unwrap(), "max_rows"),
        (StateBudget::new(2, bytes - 1).unwrap(), "max_bytes"),
    ] {
        op.set_state_budget(budget).unwrap();
        let error = op
            .process_data("p", candidate.clone(), &context, &mut out)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains(&format!("replacement.p: sql_state_budget.{field}")));
        assert!(error.contains("required=") && error.contains("limit="));
        assert!(drain(&mut out).is_empty());
        assert_eq!(
            signature(&op.checkpoint(Epoch::INITIAL).unwrap()),
            signature(&saved)
        );
    }
    op.set_state_budget(StateBudget::new(2, bytes).unwrap())
        .unwrap();
    for (input, expected) in [(candidate, 5), (batch(&[&[4, 5]]), 9)] {
        op.process_data("p", input, &context, &mut out)
            .await
            .unwrap();
        let events = drain(&mut out);
        assert_eq!(events.len(), 1);
        assert_eq!(sum(&events[0]), Some(expected));
    }
}

#[tokio::test]
async fn test_replacement_restore_honors_tighter_legacy_input_budget() {
    let mut original = replacement_sql().await;
    let active_job = job();
    let context = StreamOperatorContext::new(&active_job, "replacement", None);
    let mut out = EdgeCollector::new(original.output_ports().to_vec());
    let input = batch(&[&[2, 3]]);
    let bytes = input.estimated_bytes().unwrap() as u64;
    original
        .process_data("p", input, &context, &mut out)
        .await
        .unwrap();
    assert_eq!(sum(&drain(&mut out)[0]), Some(5));
    let saved = original.checkpoint(Epoch::INITIAL).unwrap();
    let mut target = replacement_sql().await;
    target
        .process_data("p", batch(&[&[9]]), &context, &mut out)
        .await
        .unwrap();
    assert_eq!(sum(&drain(&mut out)[0]), Some(9));
    let before = target.checkpoint(Epoch::INITIAL).unwrap();
    for (budget, field) in [
        (StateBudget::new(1, bytes).unwrap(), "max_rows"),
        (StateBudget::new(2, bytes - 1).unwrap(), "max_bytes"),
    ] {
        target.set_state_budget(budget).unwrap();
        let error = target.restore(&saved).unwrap_err().to_string();
        assert!(error.contains(&format!("replacement.p: sql_state_budget.{field}")));
        assert_eq!(
            signature(&target.checkpoint(Epoch::INITIAL).unwrap()),
            signature(&before)
        );
        assert!(drain(&mut out).is_empty());
    }
    target
        .set_state_budget(StateBudget::new(2, bytes).unwrap())
        .unwrap();
    target.restore(&saved).unwrap();
    assert!(drain(&mut out).is_empty());
    assert_eq!(
        signature(&target.checkpoint(Epoch::INITIAL).unwrap()),
        signature(&saved)
    );
    for op in [&mut original, &mut target] {
        op.process_data("p", batch(&[&[4], &[5]]), &context, &mut out)
            .await
            .unwrap();
        let events = drain(&mut out);
        assert_eq!(events.len(), 1);
        assert_eq!(sum(&events[0]), Some(9));
    }
}

#[tokio::test]
async fn test_replacement_cancelled_pending_emit_refunds_candidate_and_output() {
    struct Pending(Arc<tokio::sync::Notify>);
    #[async_trait]
    impl StreamCollector for Pending {
        async fn emit(&mut self, _: &str, _: Batch) -> Result<()> {
            self.0.notify_one();
            std::future::pending().await
        }
    }
    let mut op = operator("SELECT SUM(x) AS total FROM p").await;
    let active_job = job();
    let context = StreamOperatorContext::new(&active_job, "replacement", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    let input = batch(&[&[1]]);
    op.process_data("p", input.clone(), &context, &mut out)
        .await
        .unwrap();
    let produced = drain(&mut out).pop().unwrap();
    let saved = op.checkpoint(Epoch::INITIAL).unwrap();
    let pool = op.pool.as_ref().unwrap().clone();
    let before = pool.reserved();
    let entered = Arc::new(tokio::sync::Notify::new());
    let mut pending = Pending(entered.clone());
    let mut work = Box::pin(op.process_data("p", input.clone(), &context, &mut pending));
    tokio::select! {
        () = entered.notified() => {},
        result = &mut work => panic!("emit must remain pending: {result:?}"),
    }
    assert_eq!(
        pool.reserved(),
        before + backing_bytes(&input).unwrap() + backing_bytes(&produced).unwrap() + 8192
    );
    active_job.cancellation().cancel();
    drop(work);
    assert_eq!(pool.reserved(), before);
    assert!(drain(&mut out).is_empty());
    assert_eq!(
        signature(&op.checkpoint(Epoch::INITIAL).unwrap()),
        signature(&saved)
    );
    let retry_job = job();
    let retry_context = StreamOperatorContext::new(&retry_job, "replacement", None);
    op.process_data("p", batch(&[&[2, 3]]), &retry_context, &mut out)
        .await
        .unwrap();
    let events = drain(&mut out);
    assert_eq!(events.len(), 1);
    assert_eq!(sum(&events[0]), Some(5));
}

#[tokio::test]
async fn test_replacement_successful_emit_commits_when_collector_cancels() {
    struct CancelAfterEmit {
        cancellation: CancellationToken,
        emitted: Vec<Batch>,
    }
    #[async_trait]
    impl StreamCollector for CancelAfterEmit {
        async fn emit(&mut self, port: &str, batch: Batch) -> Result<()> {
            assert_eq!(port, "output");
            self.emitted.push(batch);
            self.cancellation.cancel();
            tokio::task::yield_now().await;
            Ok(())
        }
    }
    let mut op = operator("SELECT SUM(x) AS total FROM p").await;
    let active_job = job();
    let context = StreamOperatorContext::new(&active_job, "replacement", None);
    let mut out = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("p", batch(&[&[1]]), &context, &mut out)
        .await
        .unwrap();
    assert_eq!(sum(&drain(&mut out)[0]), Some(1));
    let saved = op.checkpoint(Epoch::INITIAL).unwrap();
    let mut collector = CancelAfterEmit {
        cancellation: active_job.cancellation().clone(),
        emitted: Vec::new(),
    };
    op.process_data("p", batch(&[&[2, 3]]), &context, &mut collector)
        .await
        .unwrap();
    assert!(context.check_cancelled().is_err());
    assert_eq!(collector.emitted.len(), 1);
    assert_eq!(sum(&collector.emitted[0]), Some(5));
    let committed = op.checkpoint(Epoch::INITIAL).unwrap();
    assert_ne!(signature(&committed), signature(&saved));
    let mut restored = operator("SELECT SUM(x) AS total FROM p").await;
    restored.restore(&committed).unwrap();
    assert_eq!(restored.relations[0].as_ref().unwrap().batch.num_rows(), 2);
    let retry_job = job();
    let retry_context = StreamOperatorContext::new(&retry_job, "replacement", None);
    for target in [&mut op, &mut restored] {
        target
            .process_data("p", batch(&[&[4]]), &retry_context, &mut out)
            .await
            .unwrap();
        let events = drain(&mut out);
        assert_eq!(events.len(), 1);
        assert_eq!(sum(&events[0]), Some(4));
    }
}
