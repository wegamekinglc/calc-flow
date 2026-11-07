use super::*;
use crate::{CalcFlowError, EventTime, StreamMessageKind, runtime::streaming::supervisor::TaskId};

#[derive(Clone, Copy, Debug)]
enum Refusal {
    LocalBudget,
    SinkBudget,
    HeadCast,
    SqlType,
    CompactRequired,
    UppercaseAlias,
}

fn refusal_parts(case: Refusal) -> StreamRuntimePlanParts {
    let alias = if matches!(case, Refusal::UppercaseAlias) {
        "Trades"
    } else {
        "trades"
    };
    let mut parts = if matches!(case, Refusal::SqlType | Refusal::HeadCast) {
        let value = if matches!(case, Refusal::HeadCast) {
            "CAST(quantity AS BIGINT) AS delta"
        } else {
            "quantity AS delta"
        };
        let head = ExpressionOperator::new(
            HEAD,
            "",
            vec!["account".into(), "stock".into(), value.into()],
            None,
            vec![],
        )
        .unwrap();
        build_plan(head, position_sql(alias, false), alias)
            .into_runtime_parts(EdgeBudget::default())
            .unwrap()
    } else {
        plan(alias, true)
            .into_runtime_parts(EdgeBudget::default())
            .unwrap()
    };
    match case {
        Refusal::LocalBudget => {
            let edge = parts
                .edges
                .values_mut()
                .find(|edge| edge.kind == RuntimeEdgeKind::Internal)
                .unwrap();
            edge.budget = EdgeBudget::new(1, 1 << 20).unwrap();
        }
        Refusal::SinkBudget => {
            let edge = parts.sink_routes.values().next().unwrap().edge_id.clone();
            parts.edges.get_mut(&edge).unwrap().budget = EdgeBudget::new(100, 1).unwrap();
        }
        _ => {}
    }
    parts
}

fn refusal_input(case: Refusal) -> Batch {
    match case {
        Refusal::CompactRequired => lifecycle::floating_key_input(0, &[2, 5]),
        Refusal::SqlType | Refusal::HeadCast => {
            let batch = input(0, &[2, 5], true);
            let record = &batch.table_payload().unwrap().batches()[0];
            let fields = record
                .schema()
                .fields()
                .iter()
                .enumerate()
                .map(|(index, field)| {
                    if index == 3 {
                        Field::new("quantity", DataType::Utf8, false)
                    } else {
                        field.as_ref().clone()
                    }
                })
                .collect::<Vec<_>>();
            let mut columns = record.columns().to_vec();
            columns[3] = Arc::new(StringArray::from(vec!["invalid", "invalid"]));
            Batch::table(
                vec![RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()],
                batch.metadata().clone(),
            )
            .unwrap()
        }
        _ => input(0, &[2, 5], true),
    }
}

async fn refuse(case: Refusal, fused: bool) -> String {
    let mut harness = harness(refusal_parts(case), fused).await;
    harness.runtime.data_gate.send(true).unwrap();
    harness.send(StreamMessage::data(refusal_input(case))).await;
    let _ = harness.runtime.source_outputs.values_mut().next().unwrap()[0]
        .send(StreamMessage::end_of_input())
        .await;
    let report = harness.runtime.supervisor.join_all().await;
    assert_eq!(report.primary_errors().len(), 1, "{case:?}: {report:?}");
    let upstream = matches!(case, Refusal::LocalBudget | Refusal::HeadCast);
    let failure = &report.primary_errors()[0];
    assert_eq!(failure.task_id, TaskId::new(u64::from(!upstream)));
    assert_eq!(
        failure.task_name,
        format!("operator:{}", if upstream { HEAD } else { TAIL })
    );
    if matches!(case, Refusal::LocalBudget | Refusal::SinkBudget) {
        assert!(matches!(
            failure.error,
            CalcFlowError::EdgeBudgetExceeded { .. }
        ));
    }
    let metrics = harness.core.metrics.snapshot();
    assert_eq!(
        metrics.nodes[HEAD].fully_fanned_out_batches,
        u64::from(!upstream)
    );
    assert_eq!(metrics.nodes[TAIL].fully_fanned_out_batches, 0);
    harness.runtime.sink_inputs.clear();
    harness.core.sql_recovery.close_admission();
    assert!(harness.core.sql_recovery.drain().await.is_empty());
    assert_eq!(harness.core.sql_recovery.live_counts(), (0, 0));
    assert!(
        harness
            .core
            .metrics
            .snapshot()
            .edges
            .values()
            .all(|edge| edge.channel.charged_rows == 0 && edge.channel.charged_bytes == 0)
    );
    failure.error.to_string()
}

#[tokio::test(flavor = "current_thread")]
async fn p1b_failures_keep_the_original_node_boundary_and_both_edge_budgets() {
    for case in [
        Refusal::LocalBudget,
        Refusal::SinkBudget,
        Refusal::HeadCast,
        Refusal::SqlType,
        Refusal::CompactRequired,
        Refusal::UppercaseAlias,
    ] {
        let reference = refuse(case, false).await;
        let fused = refuse(case, true).await;
        assert_eq!(reference, fused, "{case:?}");
    }
}

#[tokio::test(flavor = "current_thread")]
async fn p1b_quantity_negation_overflow_keeps_the_existing_wrapped_snapshot() {
    for fused in [false, true] {
        let parts = plan("trades", true)
            .into_runtime_parts(EdgeBudget::default())
            .unwrap();
        let mut harness = harness(parts, fused).await;
        harness.runtime.data_gate.send(true).unwrap();
        harness
            .send(StreamMessage::data(input(0, &[1, i64::MIN], true)))
            .await;
        let message = harness.receive().await;
        let table = message.as_data().unwrap().table_payload().unwrap();
        assert_eq!(
            table.batches()[0]
                .column_by_name("position")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            i64::MIN + 1
        );
        harness.send(StreamMessage::end_of_input()).await;
        assert!(harness.receive().await.is_end_of_input());
        harness.finish().await;
    }
}

async fn saturate(harness: &mut Harness) -> tokio::task::JoinHandle<crate::Result<()>> {
    harness.runtime.data_gate.send(true).unwrap();
    let mut source = harness
        .runtime
        .source_outputs
        .pop_first()
        .unwrap()
        .1
        .pop()
        .unwrap();
    let producer = tokio::spawn(async move {
        for sequence in 0..8 {
            source
                .send(StreamMessage::data(input(sequence, &[1, 2], true)))
                .await?;
        }
        source.send(StreamMessage::end_of_input()).await
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        while harness
            .core
            .metrics
            .snapshot()
            .edges
            .values()
            .filter(|edge| edge.channel.blocked_sends > 0)
            .count()
            < 2
        {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    producer
}

#[tokio::test(flavor = "current_thread")]
async fn p1b_full_local_and_sink_queues_cancel_or_resume_without_losing_snapshots() {
    for cancel in [false, true] {
        let parts = plan("trades", true)
            .into_runtime_parts(EdgeBudget::new(2, 1 << 20).unwrap())
            .unwrap();
        let mut harness = harness(parts, true).await;
        let producer = saturate(&mut harness).await;
        if cancel {
            harness.cancellation.cancel();
            harness.finish().await;
            assert!(producer.await.unwrap().is_err());
            harness.runtime.sink_inputs.clear();
        } else {
            let mut snapshots = 0;
            loop {
                let message = harness.receive().await;
                if message.as_data().is_some() {
                    snapshots += 1;
                }
                if message.is_end_of_input() {
                    break;
                }
            }
            assert_eq!(snapshots, 8);
            producer.await.unwrap().unwrap();
            harness.finish().await;
        }
        assert!(
            harness
                .core
                .metrics
                .snapshot()
                .edges
                .values()
                .all(|edge| edge.channel.charged_rows == 0 && edge.channel.charged_bytes == 0)
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn p1b_empty_source_preserves_control_fifo_without_starting_sql() {
    for fused in [false, true] {
        let parts = plan("trades", false)
            .into_runtime_parts(EdgeBudget::default())
            .unwrap();
        let mut harness = harness(parts, fused).await;
        harness.runtime.data_gate.send(true).unwrap();
        harness
            .send(StreamMessage::watermark(EventTime::from_micros(1)))
            .await;
        harness.send(StreamMessage::idle()).await;
        harness.send(StreamMessage::end_of_input()).await;
        for expected in [
            StreamMessageKind::Watermark,
            StreamMessageKind::Idle,
            StreamMessageKind::EndOfInput,
        ] {
            assert_eq!(harness.receive().await.kind(), expected);
        }
        harness.finish().await;
        assert!(
            harness
                .core
                .runtime_status
                .lock()
                .nodes
                .values()
                .all(|node| !node.snapshot().datafusion_runtime_created)
        );
    }
}
