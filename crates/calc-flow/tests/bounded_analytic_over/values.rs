use calc_flow::arrow::{
    array::{ArrayRef, Int8Array, Int64Array, LargeStringArray, UInt8Array, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use calc_flow::{
    AnalyticExpr, Batch, BatchMetadata, BoundedOrderOperator, CancellationToken, EdgeCollector,
    Epoch, JsonMap, NullPlacement, OperatorMetadata, OverOrder, OverSpec, RowsFrame, SortDirection,
    SortKey, StateBudget, StreamJobContext, StreamOperator, StreamOperatorContext,
};
use std::{collections::BTreeMap, sync::Arc};
fn input(values: ArrayRef) -> Batch {
    let n = values.len();
    let schema = Arc::new(Schema::new(vec![
        Field::new("score", DataType::UInt64, false),
        Field::new("seq", DataType::UInt64, false),
        Field::new("v", values.data_type().clone(), true),
    ]));
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(UInt64Array::from((0..n as u64).collect::<Vec<_>>())),
                    Arc::new(UInt64Array::from((0..n as u64).collect::<Vec<_>>())),
                    values,
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}
fn make(raw: &Batch, expressions: Vec<AnalyticExpr>) -> BoundedOrderOperator {
    let order = OverOrder::new(
        vec![],
        vec![SortKey::new("score", SortDirection::Ascending, NullPlacement::Last).unwrap()],
        vec!["seq".into()],
    )
    .unwrap();
    BoundedOrderOperator::new_over(
        "over",
        raw.table_payload().unwrap().schema().clone(),
        expressions,
        OverSpec::bounded_source(order, StateBudget::new(10, 8 << 20).unwrap()).unwrap(),
    )
    .unwrap()
}
#[tokio::test]
async fn test_over_sum_checks_every_output_without_wrapping_or_cancellation_masking() {
    for values in [
        vec![Some(i64::MAX), Some(1)],
        vec![Some(i64::MAX), Some(1), Some(-1)],
    ] {
        let raw = input(Arc::new(Int64Array::from(values)));
        let mut op = make(
            &raw,
            vec![AnalyticExpr::sum("v", RowsFrame::cumulative(), "total").unwrap()],
        );
        let job = StreamJobContext::new(1, "over", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "over", None);
        let mut output = EdgeCollector::new(op.output_ports().to_vec());
        op.process_data("input", raw, &context, &mut output)
            .await
            .unwrap();
        let before = op.checkpoint(Epoch::INITIAL).unwrap();
        let error = op
            .on_end(&context, &mut output)
            .await
            .unwrap_err()
            .to_string();
        assert!(error.contains("arithmetic_overflow"), "{error}");
        assert!(output.drain("output").is_empty());
        assert_eq!(
            op.checkpoint(Epoch::INITIAL).unwrap().segments,
            before.segments
        );
    }
    let raw = input(Arc::new(UInt64Array::from(vec![u64::MAX, 1])));
    let mut op = make(
        &raw,
        vec![AnalyticExpr::sum("v", RowsFrame::cumulative(), "total").unwrap()],
    );
    let job = StreamJobContext::new(1, "over", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "over", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("input", raw, &context, &mut output)
        .await
        .unwrap();
    assert!(
        op.on_end(&context, &mut output)
            .await
            .unwrap_err()
            .to_string()
            .contains("arithmetic_overflow")
    );
}
#[tokio::test]
async fn test_over_current_row_and_wide_frames_preserve_null_and_integer_types() {
    use calc_flow::{BatchOperator, BatchOperatorContext, RunContext};
    for (values, dtype) in [
        (
            Arc::new(Int8Array::from(vec![None, Some(0), Some(120)])) as ArrayRef,
            DataType::Int64,
        ),
        (
            Arc::new(UInt8Array::from(vec![None, Some(0), Some(250)])) as ArrayRef,
            DataType::UInt64,
        ),
        (
            Arc::new(Int64Array::from(vec![Some(i64::MAX), Some(1)])) as ArrayRef,
            DataType::Int64,
        ),
    ] {
        let raw = input(values);
        let mut op = make(
            &raw,
            vec![AnalyticExpr::sum("v", RowsFrame::preceding(0).unwrap(), "current").unwrap()],
        );
        let run = RunContext::new(JsonMap::new(), None, CancellationToken::new()).unwrap();
        let context = BatchOperatorContext { run: &run };
        let out = BatchOperator::process(
            &mut op,
            &BTreeMap::from([("input".into(), raw.clone())]),
            &context,
        )
        .await
        .unwrap();
        let record = &out["output"].table_payload().unwrap().batches()[0];
        let current = record.column_by_name("current").unwrap();
        assert_eq!(current.data_type(), &dtype);
        assert_eq!(current.len(), raw.num_rows());
        assert_eq!(
            current.null_count(),
            raw.table_payload().unwrap().batches()[0]
                .column_by_name("v")
                .unwrap()
                .null_count()
        );
    }
}
#[tokio::test]
async fn test_over_lag_large_utf8_respects_null_and_detaches_foreign_owner() {
    let values = Arc::new(LargeStringArray::from(vec![Some("a"), None, Some("z")]));
    let observed = values.value_data().as_ptr();
    let raw = input(values);
    let mut op = make(&raw, vec![AnalyticExpr::lag("v", "prev").unwrap()]);
    let job = StreamJobContext::new(1, "over", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "over", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("input", raw, &context, &mut output)
        .await
        .unwrap();
    op.on_end(&context, &mut output).await.unwrap();
    let finals = output.drain("output");
    let record = &finals[0]
        .as_data()
        .unwrap()
        .table_payload()
        .unwrap()
        .batches()[0];
    let lag = record
        .column_by_name("prev")
        .unwrap()
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .unwrap();
    assert_eq!(lag.iter().collect::<Vec<_>>(), vec![None, Some("a"), None]);
    assert_ne!(lag.value_data().as_ptr(), observed);
}

#[tokio::test]
async fn test_over_owned_budget_error_identifies_the_analytic_node() {
    use calc_flow::{
        BatchOperator, BatchOperatorContext, CalcFlowError, RunContext, StreamingFailureReason,
    };
    let raw = input(Arc::new(Int64Array::from(vec![1, 2, 3])));
    let order = OverOrder::new(
        vec![],
        vec![SortKey::new("score", SortDirection::Ascending, NullPlacement::Last).unwrap()],
        vec!["seq".into()],
    )
    .unwrap();
    let mut op = BoundedOrderOperator::new_over(
        "analytic",
        raw.table_payload().unwrap().schema().clone(),
        vec![AnalyticExpr::row_number("rn").unwrap()],
        OverSpec::bounded_source(order, StateBudget::new(10, 80_000).unwrap()).unwrap(),
    )
    .unwrap();
    let run = RunContext::new(JsonMap::new(), None, CancellationToken::new()).unwrap();
    let error = BatchOperator::process(
        &mut op,
        &BTreeMap::from([("input".into(), raw)]),
        &BatchOperatorContext { run: &run },
    )
    .await
    .unwrap_err();
    let CalcFlowError::OperatorReason {
        node_id,
        reason_code,
        ..
    } = error
    else {
        panic!("expected typed budget failure: {error}");
    };
    assert_eq!(reason_code, StreamingFailureReason::StateBudgetExceeded);
    assert_eq!(node_id, "analytic");
}
