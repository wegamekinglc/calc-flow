use super::*;
use crate::{
    AnalyticExpr, CancellationToken, EdgeCollector, Epoch, OverOrder, OverSpec, RowsFrame,
    StreamJobContext,
};
use datafusion::arrow::{
    array::{Int64Array, LargeStringArray, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

fn operator() -> BoundedOrderOperator {
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", DataType::Int64, true),
        Field::new("seq", DataType::UInt64, false),
        Field::new("text", DataType::LargeUtf8, true),
    ]));
    let order = OverOrder::new(
        vec![],
        vec![
            SortKey::new(
                "x",
                crate::SortDirection::Ascending,
                crate::NullPlacement::Last,
            )
            .unwrap(),
        ],
        vec!["seq".into()],
    )
    .unwrap();
    BoundedOrderOperator::new_over(
        "over",
        schema,
        vec![
            AnalyticExpr::row_number("rn").unwrap(),
            AnalyticExpr::lag("text", "previous").unwrap(),
            AnalyticExpr::sum("x", RowsFrame::preceding(1).unwrap(), "total").unwrap(),
        ],
        OverSpec::bounded_source(order, crate::StateBudget::new(10, 8 << 20).unwrap()).unwrap(),
    )
    .unwrap()
}
fn raw(op: &BoundedOrderOperator, empty: bool) -> Batch {
    let n = if empty { 0 } else { 3 };
    Batch::table(
        vec![
            RecordBatch::try_new(
                op.input[0].schema().unwrap().clone(),
                vec![
                    Arc::new(Int64Array::from(vec![Some(1); n])),
                    Arc::new(UInt64Array::from((0..n as u64).collect::<Vec<_>>())),
                    Arc::new(LargeStringArray::from(vec![Some("owned string"); n])),
                ],
            )
            .unwrap(),
        ],
        crate::BatchMetadata::default(),
    )
    .unwrap()
}
#[tokio::test]
async fn test_over_owned_peak_boundaries_cover_new_columns_empty_and_live_captures() {
    for empty in [false, true] {
        let op = operator();
        let run = crate::RunContext::new(JsonMap::new(), None, CancellationToken::new()).unwrap();
        let input = raw(&op, empty);
        let (pool, declaration) = op.finite_pool().unwrap();
        let result = op
            .finite(&input, &run, pool.clone(), declaration)
            .await
            .unwrap();
        let peak = pool.take_peak();
        drop(result);
        assert_eq!(pool.reserved(), 0);
        for (limit, success) in [(peak, true), (peak - 1, false)] {
            let (pool, declaration) = op.finite_pool().unwrap();
            pool.set_limit(Some(limit as u64)).unwrap();
            let result = op.finite(&input, &run, pool.clone(), declaration).await;
            assert_eq!(
                result.is_ok(),
                success,
                "analytic finite empty={empty} B={peak} limit={limit}: {:?}",
                result.as_ref().err()
            );
            assert!(pool.take_peak() <= limit);
            drop(result);
            assert_eq!(pool.reserved(), 0);
        }
        eprintln!("bounded_over finite empty={empty} owned_peak={peak}");
    }
    let job = StreamJobContext::new(1, "over", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "over", None);
    let mut probe = operator();
    let mut output = EdgeCollector::new(probe.output_ports().to_vec());
    probe
        .process_data("input", raw(&probe, false), &context, &mut output)
        .await
        .unwrap();
    let capture = probe.checkpoint(Epoch::INITIAL).unwrap();
    let capture_peak = probe.pool.take_peak();
    probe.on_end(&context, &mut output).await.unwrap();
    let final_peak = probe.pool.take_peak();
    for (limit, success) in [(final_peak, true), (final_peak - 1, false)] {
        // A fresh job includes native worker-generation funding in every trial.
        let trial_job =
            StreamJobContext::new(1, "over", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&trial_job, "over", None);
        let mut op = operator();
        op.process_data("input", raw(&op, false), &context, &mut output)
            .await
            .unwrap();
        let held = op.checkpoint(Epoch::INITIAL).unwrap();
        op.pool.take_peak();
        op.pool.set_limit(Some(limit as u64)).unwrap();
        let result = op.on_end(&context, &mut output).await;
        let observed = op.pool.take_peak();
        eprintln!("analytic stream close B={final_peak} limit={limit} observed={observed}");
        assert_eq!(
            result.is_ok(),
            success,
            "capture+final B={final_peak} limit={limit} observed={observed}: {result:?}"
        );
        if !success {
            assert_eq!(
                op.checkpoint(Epoch::INITIAL).unwrap().segments,
                held.segments
            );
        }
        assert!(op.pool.take_peak() <= limit);
    }
    let mut target = operator();
    target.restore_capture(&capture).unwrap();
    let restore_peak = target.pool.take_peak();
    for (limit, success) in [(restore_peak, true), (restore_peak - 1, false)] {
        let mut op = operator();
        op.pool.set_limit(Some(limit as u64)).unwrap();
        let result = op.restore_capture(&capture);
        assert_eq!(
            result.is_ok(),
            success,
            "restore B={restore_peak} limit={limit}: {result:?}"
        );
        assert_eq!(
            op.stream.as_ref().unwrap().accepted,
            if success { 3 } else { 0 }
        );
        assert!(op.pool.take_peak() <= limit);
    }
    eprintln!(
        "bounded_over capture_peak={capture_peak} capture+final_peak={final_peak} restore_peak={restore_peak}"
    );
}
#[tokio::test]
async fn test_over_raw_owner_released_before_await_and_edge_failure_does_not_close() {
    let mut op = operator();
    let raw = raw(&op, false);
    let values = Arc::new(Int64Array::from(vec![1; 1_048_576]));
    let observed = values.values().inner().clone();
    let sliced = values.slice(5, 3);
    let record = &raw.table_payload().unwrap().batches()[0];
    let raw = Batch::table(
        vec![
            RecordBatch::try_new(
                record.schema(),
                vec![
                    Arc::new(sliced),
                    record.column(1).clone(),
                    record.column(2).clone(),
                ],
            )
            .unwrap(),
        ],
        crate::BatchMetadata::default(),
    )
    .unwrap();
    drop(values);
    let job = StreamJobContext::new(1, "over", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "over", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("input", raw, &context, &mut output)
        .await
        .unwrap();
    assert_eq!(observed.strong_count(), 1);
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
    let context = context.with_output_budget(crate::EdgeBudget {
        max_rows: 2,
        max_bytes: 1 << 20,
    });
    assert!(op.on_end(&context, &mut output).await.is_err());
    assert!(!op.stream.as_ref().unwrap().closed);
    assert_eq!(
        op.checkpoint(Epoch::INITIAL).unwrap().segments,
        before.segments
    );
    assert!(output.drain("output").is_empty());
}

#[tokio::test]
async fn test_over_provisional_output_roles_cannot_use_the_earlier_partial_quote() {
    // The earlier quote omitted the provisional output's retained input role,
    // schema-building owners and nullable byte-take range scratch.
    let op = operator();
    let input = raw(&op, false);
    let run = crate::RunContext::new(JsonMap::new(), None, CancellationToken::new()).unwrap();
    let (pool, declaration) = op.finite_pool().unwrap();
    pool.set_limit(Some(110_170)).unwrap();
    let result = op.finite(&input, &run, pool, declaration).await;
    assert!(
        result.is_err(),
        "the audited participating roles exceed the earlier partial quote"
    );
}

#[tokio::test]
async fn test_over_worker_generation_budget_is_typed_and_does_not_close() {
    let probe_job =
        StreamJobContext::new(1, "over", JsonMap::new(), None, CancellationToken::new());
    let probe_context = StreamOperatorContext::new(&probe_job, "over", None);
    let mut probe = operator();
    let mut output = EdgeCollector::new(probe.output_ports().to_vec());
    probe
        .process_data("input", raw(&probe, false), &probe_context, &mut output)
        .await
        .unwrap();
    let history = probe.close_input(&probe_context).unwrap().unwrap().0;
    let work = probe
        .final_work(history, probe.pool.clone(), probe.declaration.clone())
        .unwrap();
    let limit = probe.pool.reserved() + 1024;
    drop(work);
    let job = StreamJobContext::new(2, "over", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "over", None);
    let mut op = operator();
    op.process_data("input", raw(&op, false), &context, &mut output)
        .await
        .unwrap();
    op.pool.set_limit(Some(limit as u64)).unwrap();
    let error = op.on_end(&context, &mut output).await.unwrap_err();
    assert!(
        matches!(
            error,
            crate::CalcFlowError::OperatorReason {
                reason_code: crate::StreamingFailureReason::StateBudgetExceeded,
                ..
            }
        ),
        "{error}"
    );
    assert_eq!(op.stream.as_ref().unwrap().accepted, 3);
    assert!(!op.stream.as_ref().unwrap().closed);
    assert!(output.drain("output").is_empty());
}
