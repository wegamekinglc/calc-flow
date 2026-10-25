use super::*;
use crate::{CancellationToken, EdgeCollector, StreamJobContext};
use datafusion::{
    arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray, TimestampMicrosecondArray},
    common::ScalarValue,
};
use std::sync::Arc;

mod corruption;
mod faults;
mod recovery;

const QUERY: &str = "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp) AS latest_price FROM input GROUP BY stock";
type Row = (Option<&'static str>, Option<f64>, Option<i64>, Option<i64>);

fn input(rows: &[Row]) -> Batch {
    let record = RecordBatch::try_from_iter(vec![
        (
            "stock",
            Arc::new(StringArray::from(
                rows.iter().map(|r| r.0).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "latest_price",
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.1).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
        (
            "timestamp",
            Arc::new(
                TimestampMicrosecondArray::from(rows.iter().map(|r| r.2).collect::<Vec<_>>())
                    .with_timezone("UTC"),
            ) as ArrayRef,
        ),
        (
            "seq",
            Arc::new(Int64Array::from(
                rows.iter().map(|r| r.3).collect::<Vec<_>>(),
            )) as ArrayRef,
        ),
    ])
    .unwrap();
    let schema = Arc::new(datafusion::arrow::datatypes::Schema::new(
        record
            .schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone().with_nullable(true))
            .collect::<Vec<_>>(),
    ));
    let record = RecordBatch::try_new(schema, record.columns().to_vec()).unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn operator(query: &str) -> SqlOperator {
    SqlOperator::new("ordered_last", query, vec!["input".into()], vec![]).unwrap()
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        431,
        "ordered-last",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

async fn process(op: &mut SqlOperator, batch: Batch, ctx: &StreamOperatorContext<'_>) -> Batch {
    let mut collector = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("input", batch, ctx, &mut collector)
        .await
        .unwrap();
    collector.drain("output")[0].as_data().unwrap().clone()
}

fn values(batch: &Batch) -> BTreeMap<String, Vec<ScalarValue>> {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                let cells = record
                    .columns()
                    .iter()
                    .map(|a| ScalarValue::try_from_array(a, row).unwrap())
                    .collect::<Vec<_>>();
                (format!("{:?}", cells[0]), cells)
            })
        })
        .collect()
}

#[tokio::test]
async fn test_ordered_last_original_example_is_compact_and_keeps_first_tie() {
    let ctx_job = job();
    let ctx = StreamOperatorContext::new(&ctx_job, "ordered_last", None);
    let mut op = operator(QUERY);
    let first = input(&[
        (Some("A"), Some(10.0), Some(-1), Some(0)),
        (Some("A"), Some(20.0), Some(-1), Some(0)),
    ]);
    let result = process(&mut op, first, &ctx).await;
    assert_eq!(
        values(&result).values().next().unwrap()[1],
        ScalarValue::Float64(Some(10.0))
    );
    assert!(op.compact.is_some(), "example 31 must use compact state");
    assert!(op.retained.is_none());
}

#[tokio::test]
async fn test_ordered_last_every_prefix_matches_datafusion_across_batch_splits() {
    let rows = [
        (Some("A"), Some(1.0), Some(-10), Some(0)),
        (Some("B"), None, Some(5), Some(1)),
        (Some("A"), Some(2.0), Some(3), Some(1)),
        (Some("A"), Some(9.0), Some(3), Some(1)),
        (None, None, None, None),
        (Some("A"), None, None, Some(2)),
        (Some("B"), Some(4.0), Some(0), Some(0)),
        (None, Some(7.0), Some(-1), Some(0)),
        (Some("A"), Some(8.0), Some(3), Some(3)),
    ];
    for (order, treatment) in [
        ("timestamp", ""),
        ("timestamp ASC NULLS FIRST", "RESPECT NULLS"),
        ("timestamp DESC NULLS LAST", "IGNORE NULLS"),
        ("timestamp DESC", ""),
        (
            "timestamp ASC NULLS FIRST, seq DESC NULLS LAST",
            "IGNORE NULLS",
        ),
    ] {
        let query = format!(
            "SELECT stock, LAST_VALUE(latest_price ORDER BY {order}) {treatment} AS latest_price FROM input GROUP BY stock ORDER BY stock"
        );
        for width in [1, 3, 20] {
            let mut op = operator(&query);
            let ctx_job = job();
            let ctx = StreamOperatorContext::new(&ctx_job, "ordered_last", None);
            let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
            let mut prefix = Vec::new();
            for part in rows.chunks(width) {
                prefix.extend_from_slice(part);
                let actual = process(&mut op, input(part), &ctx).await;
                let expected = runtime
                    .sql(
                        &query,
                        &BTreeMap::from([("input".into(), input(&prefix))]),
                        Some("oracle"),
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    values(&actual),
                    values(&expected),
                    "{query}; width={width}; prefix={}",
                    prefix.len()
                );
                assert!(op.compact.is_some(), "{query}");
                assert!(op.retained.is_none());
            }
        }
    }
}

fn same_snapshot(before: &OperatorStateSnapshot, after: &OperatorStateSnapshot) {
    assert_eq!(before.inline_metadata, after.inline_metadata);
    let digests = |snapshot: &OperatorStateSnapshot| {
        snapshot
            .segments
            .iter()
            .map(|(id, segment)| (id.clone(), segment.sha256().to_owned()))
            .collect::<BTreeMap<_, _>>()
    };
    assert_eq!(digests(before), digests(after));
}

#[tokio::test]
async fn test_ordered_last_state_capacity_and_charge_stabilize_at_eight_keys() {
    let ctx_job = job();
    let ctx = StreamOperatorContext::new(&ctx_job, "ordered_last", None);
    let mut op = operator(QUERY);
    let keys = ["A", "B", "C", "D", "E", "F", "G", "H"];
    let mut baseline = None;
    for start in (0i32..10_000).step_by(100) {
        let rows = (start..start + 100)
            .map(|n| {
                (
                    Some(keys[usize::try_from(n % 8).unwrap()]),
                    Some(f64::from(n)),
                    Some(i64::from(n)),
                    Some(0),
                )
            })
            .collect::<Vec<_>>();
        let batch = input(&rows);
        let weak = batch.table_payload().unwrap().batches()[0]
            .columns()
            .iter()
            .map(Arc::downgrade)
            .collect::<Vec<_>>();
        drop(process(&mut op, batch, &ctx).await);
        assert!(
            weak.iter().all(|a| a.upgrade().is_none()),
            "input arrays must be released"
        );
        assert!(op.retained.is_none());
        let native = op.incremental.as_ref().unwrap();
        assert_eq!(
            native
                .native_descriptor("ordered_last")
                .unwrap()
                .group_count,
            8
        );
        let current = (
            native.capacity_charge(),
            op.stream_state
                .runtime()
                .unwrap()
                .incremental_memory_pool()
                .reserved(),
        );
        assert_eq!(*baseline.get_or_insert(current), current);
    }
    let snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["state_layout"], json!(5));
    assert_eq!(snapshot.inline_metadata["rows"], json!(10_000));
    assert!(!snapshot.segments.contains_key("input-retained"));
}

struct Reject(Option<CancellationToken>);

#[async_trait]
impl StreamCollector for Reject {
    async fn emit(&mut self, _: &str, _: Batch) -> Result<()> {
        if let Some(token) = &self.0 {
            token.cancel();
            return Err(CalcFlowError::Cancelled {
                run_id: "ordered-last".into(),
            });
        }
        Err(CalcFlowError::Operator {
            node_id: "ordered_last".into(),
            message: "injected emit failure".into(),
        })
    }
}

#[tokio::test]
async fn test_ordered_last_emit_failure_and_cancellation_preserve_winners_and_ledger() {
    for cancel in [false, true] {
        let token = CancellationToken::new();
        let ctx_job =
            StreamJobContext::new(439, "ordered-last", JsonMap::new(), None, token.clone());
        let ctx = StreamOperatorContext::new(&ctx_job, "ordered_last", None);
        let mut op = operator(QUERY);
        drop(
            process(
                &mut op,
                input(&[(Some("A"), Some(10.0), Some(1), Some(0))]),
                &ctx,
            )
            .await,
        );
        let before = op.checkpoint(Epoch::INITIAL).unwrap();
        let pool = op.stream_state.runtime().unwrap().incremental_memory_pool();
        let reserved = pool.reserved();
        let next = input(&[
            (Some("A"), Some(20.0), Some(2), Some(0)),
            (Some("B"), None, None, None),
        ]);
        let mut reject = Reject(cancel.then_some(token));
        assert!(
            op.process_data("input", next.clone(), &ctx, &mut reject)
                .await
                .is_err()
        );
        assert_eq!(pool.reserved(), reserved);
        same_snapshot(&before, &op.checkpoint(Epoch::INITIAL).unwrap());
        let retry_job = job();
        let retry_ctx = StreamOperatorContext::new(&retry_job, "ordered_last", None);
        let result = process(&mut op, next, &retry_ctx).await;
        assert_eq!(result.num_rows(), 2);
        assert_eq!(
            op.checkpoint(Epoch::INITIAL).unwrap().inline_metadata["rows"],
            json!(3)
        );
    }
}

#[tokio::test]
async fn test_ordered_last_layout_five_roundtrip_preserves_null_groups_and_ties() {
    let query = "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp ASC NULLS FIRST, seq DESC NULLS LAST) IGNORE NULLS AS price FROM input GROUP BY stock";
    let ctx_job = job();
    let ctx = StreamOperatorContext::new(&ctx_job, "ordered_last", None);
    let mut op = operator(query);
    let parts = [
        vec![
            (Some("A"), Some(10.0), Some(-1), Some(0)),
            (None, None, None, None),
        ],
        vec![
            (Some("A"), Some(20.0), Some(5), Some(2)),
            (Some("B"), None, Some(5), Some(0)),
        ],
        vec![
            (Some("A"), Some(30.0), Some(5), Some(2)),
            (None, Some(40.0), Some(-5), Some(0)),
        ],
        vec![
            (Some("A"), Some(50.0), Some(4), Some(0)),
            (Some("B"), Some(60.0), Some(8), Some(1)),
        ],
    ];
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let mut prefix = Vec::new();
    for rows in parts {
        prefix.extend_from_slice(&rows);
        let actual = process(&mut op, input(&rows), &ctx).await;
        let expected = runtime
            .sql(
                query,
                &BTreeMap::from([("input".into(), input(&prefix))]),
                None,
            )
            .await
            .unwrap();
        assert_eq!(values(&actual), values(&expected));
        op.prepare_checkpoint_async(&ctx).await.unwrap();
        let snapshot = op.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(snapshot.inline_metadata["state_layout"], json!(5));
        let mut restored = operator(query);
        StreamOperator::restore(&mut restored, &snapshot).unwrap();
        same_snapshot(&snapshot, &restored.checkpoint(Epoch::INITIAL).unwrap());
        op = restored;
    }
}

#[tokio::test]
async fn test_ordered_last_failed_restore_keeps_live_state() {
    let ctx_job = job();
    let ctx = StreamOperatorContext::new(&ctx_job, "ordered_last", None);
    let mut op = operator(QUERY);
    drop(
        process(
            &mut op,
            input(&[(Some("A"), Some(10.0), Some(1), Some(0))]),
            &ctx,
        )
        .await,
    );
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
    for layout in [3, 4, 6] {
        let mut damaged = before.clone();
        damaged
            .inline_metadata
            .insert("state_layout".into(), json!(layout));
        assert!(StreamOperator::restore(&mut op, &damaged).is_err());
        same_snapshot(&before, &op.checkpoint(Epoch::INITIAL).unwrap());
    }
    let mut other = operator(
        "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp DESC) AS latest_price FROM input GROUP BY stock",
    );
    drop(
        process(
            &mut other,
            input(&[(Some("B"), Some(20.0), Some(2), Some(0))]),
            &ctx,
        )
        .await,
    );
    let incompatible = other.checkpoint(Epoch::INITIAL).unwrap();
    assert!(StreamOperator::restore(&mut op, &incompatible).is_err());
    same_snapshot(&before, &op.checkpoint(Epoch::INITIAL).unwrap());
}

#[tokio::test]
async fn test_ordered_last_nonempty_layout_four_restore_stays_retained() {
    let ctx_job = job();
    let ctx = StreamOperatorContext::new(&ctx_job, "ordered_last", None);
    let mut legacy = operator(QUERY);
    legacy.incremental_checked = true;
    let first = input(&[(Some("A"), Some(10.0), Some(1), Some(0))]);
    drop(process(&mut legacy, first, &ctx).await);
    let snapshot = legacy.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["state_layout"], json!(4));
    let mut restored = operator(QUERY);
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    let actual = process(
        &mut restored,
        input(&[(Some("A"), Some(20.0), Some(1), Some(0))]),
        &ctx,
    )
    .await;
    assert_eq!(
        values(&actual).values().next().unwrap()[1],
        ScalarValue::Float64(Some(10.0))
    );
    assert!(restored.compact.is_none());
    assert_eq!(
        restored.checkpoint(Epoch::INITIAL).unwrap().inline_metadata["state_layout"],
        json!(4)
    );
}

#[tokio::test]
async fn test_ordered_last_input_quota_failure_keeps_committed_state() {
    let ctx_job = job();
    let ctx = StreamOperatorContext::new(&ctx_job, "ordered_last", None);
    let mut op = operator(QUERY);
    op.set_state_budget(StateBudget::new(1, 1 << 20).unwrap())
        .unwrap();
    drop(
        process(
            &mut op,
            input(&[(Some("A"), Some(10.0), Some(1), Some(0))]),
            &ctx,
        )
        .await,
    );
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
    let mut collector = EdgeCollector::new(op.output_ports().to_vec());
    assert!(
        op.process_data(
            "input",
            input(&[(Some("A"), Some(20.0), Some(2), Some(0))]),
            &ctx,
            &mut collector
        )
        .await
        .is_err()
    );
    assert!(collector.drain("output").is_empty());
    same_snapshot(&before, &op.checkpoint(Epoch::INITIAL).unwrap());
}

#[tokio::test]
async fn test_ordered_last_unsupported_shapes_keep_retained_execution() {
    let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
    let ctx_job = job();
    let ctx = StreamOperatorContext::new(&ctx_job, "ordered_last", None);
    for query in [
        "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp), COUNT(*) FROM input GROUP BY stock",
        "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp) FROM input WHERE latest_price > 0 GROUP BY stock",
        "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp) FILTER (WHERE latest_price > 0) FROM input GROUP BY stock",
        "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp) FROM input GROUP BY stock LIMIT 1",
        "SELECT stock, LAST_VALUE(latest_price ORDER BY timestamp) + 1 FROM input GROUP BY stock",
    ] {
        let mut op = operator(query);
        let mut prefix = Vec::new();
        for row in [
            (Some("A"), Some(10.0), Some(1), Some(0)),
            (Some("A"), Some(20.0), Some(2), Some(0)),
        ] {
            prefix.push(row);
            let actual = process(&mut op, input(&[row]), &ctx).await;
            let expected = runtime
                .sql(
                    query,
                    &BTreeMap::from([("input".into(), input(&prefix))]),
                    None,
                )
                .await
                .unwrap();
            assert_eq!(values(&actual), values(&expected));
            assert!(op.compact.is_none(), "{query}");
            assert_eq!(
                op.checkpoint(Epoch::INITIAL).unwrap().inline_metadata["state_layout"],
                json!(4)
            );
        }
    }
}
