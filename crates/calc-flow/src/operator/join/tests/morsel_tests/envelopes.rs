use super::{FundedMorsel, GatheredOutput};
use crate::{
    Batch, BatchKind, BatchMetadata, CancellationToken, EdgeBudget, EdgeCollector,
    ExpressionOperator, JsonMap, OperatorMetadata, Port, StreamCollector, StreamJobContext,
    StreamOperator, StreamOperatorContext,
};
use datafusion::arrow::{
    array::{ArrayRef, Int64Array, StringArray},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
use std::sync::Arc;

fn gathered(units: usize) -> (GatheredOutput, Arc<dyn MemoryPool>) {
    from_records(
        (0..units)
            .map(|unit| {
                let start = i64::try_from(unit * 128 / units).unwrap();
                let end = i64::try_from((unit + 1) * 128 / units).unwrap();
                RecordBatch::try_from_iter(vec![(
                    "value",
                    Arc::new(Int64Array::from_iter_values(start..end)) as ArrayRef,
                )])
                .unwrap()
            })
            .collect(),
    )
}

fn from_records(records: Vec<RecordBatch>) -> (GatheredOutput, Arc<dyn MemoryPool>) {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(16 << 20));
    let morsels = records
        .into_iter()
        .map(|record| {
            let credit = MemoryConsumer::new("output-envelope-test").register(&pool);
            credit
                .try_grow(record.get_array_memory_size() + 4096)
                .unwrap();
            FundedMorsel::new(vec![record], credit).unwrap()
        })
        .collect();
    let control = MemoryConsumer::new("output-envelope-controls").register(&pool);
    control.try_grow(4096).unwrap();
    (
        GatheredOutput {
            morsels,
            _control: control,
        },
        pool,
    )
}

#[tokio::test]
async fn test_gathered_output_envelopes_charge_each_utf8_and_validity_slice() {
    let record = RecordBatch::try_from_iter(vec![
        (
            "text",
            Arc::new(StringArray::from(vec![
                Some("x"),
                None,
                Some("中"),
                Some(""),
            ])) as ArrayRef,
        ),
        (
            "value",
            Arc::new(Int64Array::from(vec![Some(0), None, Some(2), Some(3)])) as ArrayRef,
        ),
    ])
    .unwrap();
    let bytes = Batch::table(vec![record.clone()], BatchMetadata::default())
        .unwrap()
        .estimated_bytes()
        .unwrap();
    let (gathered, pool) = from_records(vec![record; 4]);
    let job = job();
    let budget = EdgeBudget::new(7, bytes * 2).unwrap();
    let context = StreamOperatorContext::new(&job, "join", None).with_output_budget(budget);
    let mut output = collector();
    let mut sequence = 0;
    gathered
        .emit("join", &context, &mut output, &mut sequence)
        .await
        .unwrap();
    let messages = output.drain("output");
    assert_eq!(
        messages
            .iter()
            .map(|message| message.as_data().unwrap().num_rows())
            .collect::<Vec<_>>(),
        [7, 7, 2]
    );
    for message in &messages {
        let batch = message.as_data().unwrap();
        let separate = batch
            .table_payload()
            .unwrap()
            .batches()
            .iter()
            .map(|record| {
                Batch::table(vec![record.clone()], BatchMetadata::default())
                    .unwrap()
                    .estimated_bytes()
                    .unwrap()
            })
            .sum::<usize>();
        assert_eq!(batch.estimated_bytes().unwrap(), separate);
        assert!(separate <= budget.max_bytes);
    }
    drop((messages, gathered));
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_gathered_output_envelopes_reject_late_oversized_row_before_emitting() {
    let records = ["a", "oversized"].map(|text| {
        RecordBatch::try_from_iter(vec![(
            "text",
            Arc::new(StringArray::from(vec![text])) as ArrayRef,
        )])
        .unwrap()
    });
    let (gathered, pool) = from_records(records.into());
    let job = job();
    let context = StreamOperatorContext::new(&job, "join", None)
        .with_output_budget(EdgeBudget::new(10, 9).unwrap());
    let mut output = collector();
    let mut sequence = 0;
    let error = gathered
        .emit("join", &context, &mut output, &mut sequence)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("byte budget"), "{error}");
    assert!(output.drain("output").is_empty());
    assert_eq!(sequence, 0);
    drop(gathered);
    assert_eq!(pool.reserved(), 0);
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        619,
        "output-envelopes",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

fn collector() -> EdgeCollector {
    EdgeCollector::new(vec![
        Port::new("output", BatchKind::Table, true, None).unwrap(),
    ])
}

#[tokio::test]
async fn test_gathered_output_envelopes_pack_cpu_units_and_feed_expression() {
    for units in [8, 32] {
        let (gathered, pool) = gathered(units);
        let job = job();
        let context = StreamOperatorContext::new(&job, "join", None)
            .with_output_budget(EdgeBudget::new(128, usize::MAX).unwrap());
        let mut output = collector();
        let mut sequence = 0;
        gathered
            .emit("join", &context, &mut output, &mut sequence)
            .await
            .unwrap();
        let messages = output.drain("output");
        assert_eq!(messages.len(), 1, "CPU units must share one edge envelope");
        assert_eq!(sequence, 1);
        let batch = messages[0].as_data().unwrap();
        assert_eq!(batch.num_rows(), 128);
        let records = batch.table_payload().unwrap().batches();
        assert_eq!(records.len(), units);
        for (record, morsel) in records.iter().zip(&gathered.morsels) {
            assert!(Arc::ptr_eq(record.column(0), morsel.records[0].column(0)));
        }
        let mut expression = ExpressionOperator::new(
            "project",
            "",
            vec!["value + 1 AS next".into()],
            None,
            vec![],
        )
        .unwrap();
        let mut computed = EdgeCollector::new(expression.output_ports().to_vec());
        expression
            .process_data("input", batch.clone(), &context, &mut computed)
            .await
            .unwrap();
        let actual = computed
            .drain("output")
            .iter()
            .flat_map(|message| {
                message
                    .as_data()
                    .unwrap()
                    .table_payload()
                    .unwrap()
                    .batches()
                    .iter()
                    .flat_map(|record| {
                        record
                            .column(0)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap()
                            .values()
                            .iter()
                            .copied()
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, (1..=128).collect::<Vec<_>>());
        drop((computed, expression, messages, gathered));
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_gathered_output_envelopes_follow_row_and_byte_budgets_across_units() {
    for units in [8, 32] {
        for budget in [
            EdgeBudget::new(50, usize::MAX).unwrap(),
            EdgeBudget::new(128, 50 * size_of::<i64>()).unwrap(),
        ] {
            let (gathered, pool) = gathered(units);
            let job = job();
            let context = StreamOperatorContext::new(&job, "join", None).with_output_budget(budget);
            let mut output = collector();
            let mut sequence = 7;
            gathered
                .emit("join", &context, &mut output, &mut sequence)
                .await
                .unwrap();
            let messages = output.drain("output");
            assert_eq!(
                messages
                    .iter()
                    .map(|message| message.as_data().unwrap().num_rows())
                    .collect::<Vec<_>>(),
                [50, 50, 28],
                "edge layout must not depend on the CPU unit count"
            );
            let mut position = 0;
            for (index, message) in messages.iter().enumerate() {
                let batch = message.as_data().unwrap();
                assert_eq!(batch.metadata().sequence(), 7 + index as u64);
                assert!(batch.num_rows() <= budget.max_rows);
                assert!(batch.estimated_bytes().unwrap() <= budget.max_bytes);
                for record in batch.table_payload().unwrap().batches() {
                    let values = record
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    for value in values.values() {
                        assert_eq!(*value, position);
                        position += 1;
                    }
                }
            }
            assert_eq!(position, 128);
            assert_eq!(sequence, 10);
            drop((messages, gathered));
            assert_eq!(pool.reserved(), 0);
        }
    }
}

#[tokio::test]
async fn test_gathered_output_envelopes_validate_sequence_before_any_emission() {
    let (gathered, pool) = gathered(8);
    let job = job();
    let context = StreamOperatorContext::new(&job, "join", None);
    let mut output = collector();
    let mut sequence = u64::MAX - 1;
    gathered
        .emit("join", &context, &mut output, &mut sequence)
        .await
        .expect("one packed envelope fits the last sequence slot");
    assert_eq!(sequence, u64::MAX);
    assert_eq!(output.drain("output").len(), 1);

    let context = context.with_output_budget(EdgeBudget::new(50, usize::MAX).unwrap());
    let mut sequence = u64::MAX - 2;
    let error = gathered
        .emit("join", &context, &mut output, &mut sequence)
        .await
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("sequence overflowed before emission")
    );
    assert!(output.drain("output").is_empty());
    assert_eq!(sequence, u64::MAX - 2);
    drop(gathered);
    assert_eq!(pool.reserved(), 0);
}

struct BlockedCollector {
    entered: Option<tokio::sync::oneshot::Sender<()>>,
    release: tokio::sync::oneshot::Receiver<()>,
    pool: Arc<dyn MemoryPool>,
}

#[async_trait::async_trait]
impl StreamCollector for BlockedCollector {
    async fn emit(&mut self, _: &str, batch: Batch) -> crate::Result<()> {
        assert_eq!(batch.table_payload()?.batches().len(), 8);
        assert!(self.pool.reserved() > 0);
        self.entered.take().unwrap().send(()).unwrap();
        (&mut self.release).await.unwrap();
        assert!(self.pool.reserved() > 0);
        drop(batch);
        Err(crate::CalcFlowError::Operator {
            node_id: "join".into(),
            message: "blocked collector failed".into(),
        })
    }
}

#[tokio::test]
async fn test_gathered_output_envelopes_keep_funding_through_blocked_collector_cleanup() {
    for mode in ["failure", "cancel", "abandon"] {
        let (gathered, pool) = gathered(8);
        let weak = Arc::downgrade(gathered.morsels[0].records[0].column(0));
        let token = CancellationToken::new();
        let cancelled = token.clone();
        let (entered, entering) = tokio::sync::oneshot::channel();
        let (release, released) = tokio::sync::oneshot::channel();
        let mut output = BlockedCollector {
            entered: Some(entered),
            release: released,
            pool: pool.clone(),
        };
        let mut emitting = Box::pin(async move {
            let job = job();
            let context = StreamOperatorContext::new(&job, "join", None);
            let mut sequence = 0;
            let result = tokio::select! {
                result = gathered.emit("join", &context, &mut output, &mut sequence) => result,
                () = cancelled.cancelled() => Err(crate::CalcFlowError::Cancelled { run_id: "619".into() }),
            };
            (result, sequence)
        });
        tokio::select! {
            entered = entering => entered.unwrap(),
            result = &mut emitting => panic!("emission stopped before blocking: {result:?}"),
        }
        assert!(pool.reserved() > 0);
        assert!(weak.upgrade().is_some());
        match mode {
            "failure" => release.send(()).unwrap(),
            "cancel" => token.cancel(),
            "abandon" => {}
            _ => unreachable!(),
        }
        if mode != "abandon" {
            let (result, sequence) = (&mut emitting).await;
            assert!(result.is_err());
            assert_eq!(sequence, 0);
        }
        drop(emitting);
        assert!(weak.upgrade().is_none());
        assert_eq!(pool.reserved(), 0);
    }
}
