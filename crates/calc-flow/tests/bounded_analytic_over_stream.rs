#[path = "bounded_analytic_over/common.rs"]
mod common;
use calc_flow::{
    CancellationToken, EdgeCollector, Epoch, EventTime, JsonMap, OperatorMetadata,
    StreamJobContext, StreamOperator, StreamOperatorContext,
};
use common::{assert_oracle, data, operator};

#[tokio::test]
async fn test_over_nine_row_window_checkpoint_and_real_close() {
    let job = StreamJobContext::new(1, "over", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "over", None);
    let mut op = operator(false);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("input", data(true), &context, &mut output)
        .await
        .unwrap();
    op.on_watermark(EventTime::from_micros(9), &context, &mut output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    let open = op.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(open.inline_metadata["state_kind"], "bounded_over_v1");
    let mut restored = operator(false);
    restored.restore(&open).unwrap();
    restored
        .process_data("input", data(false), &context, &mut output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
    restored
        .on_watermark(EventTime::from_micros(10), &context, &mut output)
        .await
        .unwrap();
    let finals = output.drain("output");
    assert_eq!(finals.len(), 1);
    assert_oracle(finals[0].as_data().unwrap());
    let closed = restored.checkpoint(Epoch::INITIAL).unwrap();
    let control: serde_json::Value =
        serde_json::from_slice(closed.segments["control"].bytes()).unwrap();
    assert_eq!(control["accepted_rows"], 9);
    assert_eq!(control["retained_rows"], 0);
    let mut closed_restore = operator(false);
    closed_restore.restore(&closed).unwrap();
    closed_restore.on_end(&context, &mut output).await.unwrap();
    assert!(output.drain("output").is_empty());
}

fn control(snapshot: &calc_flow::OperatorStateSnapshot) -> serde_json::Value {
    serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap()
}
fn late_data() -> calc_flow::Batch {
    use calc_flow::arrow::{
        array::{ArrayRef, Int64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
        record_batch::RecordBatch,
    };
    use std::sync::Arc;
    let record = |ts: Vec<Option<i64>>, seq: Vec<u64>| {
        let n = ts.len();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(StringArray::from(vec!["A"; n])),
            Arc::new(TimestampMicrosecondArray::from(ts)),
            Arc::new(Int64Array::from(vec![0; n])),
            Arc::new(UInt64Array::from(seq)),
            Arc::new(Int64Array::from(vec![1; n])),
        ];
        RecordBatch::try_new(common::schema(), columns).unwrap()
    };
    calc_flow::Batch::table(
        vec![
            record(vec![Some(0), Some(10)], vec![50, 99]),
            record(vec![Some(9), None], vec![51, 100]),
        ],
        calc_flow::BatchMetadata::default(),
    )
    .unwrap()
}
#[tokio::test]
async fn test_over_post_close_drop_restores_totals_and_error_is_atomic() {
    for drop in [false, true] {
        let job = StreamJobContext::new(1, "over", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "over", None);
        let mut op = operator(drop);
        let mut output = EdgeCollector::new(op.output_ports().to_vec());
        for first in [true, false] {
            op.process_data("input", data(first), &context, &mut output)
                .await
                .unwrap();
        }
        op.on_watermark(EventTime::from_micros(10), &context, &mut output)
            .await
            .unwrap();
        assert_eq!(output.drain("output").len(), 1);
        let before = op.checkpoint(Epoch::INITIAL).unwrap();
        let result = op
            .process_data("input", late_data(), &context, &mut output)
            .await;
        assert_eq!(result.is_ok(), drop, "{result:?}");
        let after = op.checkpoint(Epoch::INITIAL).unwrap();
        if !drop {
            assert_eq!(before.segments, after.segments);
            continue;
        }
        assert_eq!(control(&after)["accepted_rows"], 9);
        assert_eq!(control(&after)["late_metrics"]["late_rows"], 2);
        assert_eq!(control(&after)["late_metrics"]["affected_batches"], 1);
        assert_eq!(control(&after)["late_metrics"]["max_lateness_micros"], 10);
        assert!(output.drain("output").is_empty());
        let mut restored = operator(true);
        restored.restore(&after).unwrap();
        restored
            .on_watermark(EventTime::from_micros(12), &context, &mut output)
            .await
            .unwrap();
        let single = late_data().table_payload().unwrap().batches()[0].slice(0, 1);
        // ts=0 gives max lateness 12 at the new effective W.
        let raw =
            calc_flow::Batch::table(vec![single], calc_flow::BatchMetadata::default()).unwrap();
        restored
            .process_data("input", raw, &context, &mut output)
            .await
            .unwrap();
        let totals = control(&restored.checkpoint(Epoch::INITIAL).unwrap());
        assert_eq!(totals["late_metrics"]["late_rows"], 3);
        assert_eq!(totals["late_metrics"]["affected_batches"], 2);
        assert_eq!(totals["late_metrics"]["max_lateness_micros"], 12);
        restored.on_end(&context, &mut output).await.unwrap();
        assert!(output.drain("output").is_empty());
    }
}
struct RefuseFinal;
#[async_trait::async_trait]
impl calc_flow::StreamCollector for RefuseFinal {
    async fn emit(&mut self, _: &str, _: calc_flow::Batch) -> calc_flow::Result<()> {
        Err(calc_flow::CalcFlowError::Internal {
            message: "refused analytic final".into(),
        })
    }
}
#[tokio::test]
async fn test_over_duplicate_candidate_emit_cancel_and_identity_are_atomic() {
    let cancellation = CancellationToken::new();
    let job = StreamJobContext::new(1, "over", JsonMap::new(), None, cancellation.clone());
    let context = StreamOperatorContext::new(&job, "over", None);
    let mut op = operator(false);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("input", data(true), &context, &mut output)
        .await
        .unwrap();
    let before = op.checkpoint(Epoch::INITIAL).unwrap();
    let error = op
        .process_data("input", data(true), &context, &mut output)
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("unique"));
    assert!(error.contains("over.spec.order.sequence_by"), "{error}");
    assert_eq!(
        op.checkpoint(Epoch::INITIAL).unwrap().segments,
        before.segments
    );
    assert!(
        op.on_end(&context, &mut RefuseFinal)
            .await
            .unwrap_err()
            .to_string()
            .contains("refused analytic final")
    );
    assert_eq!(
        op.checkpoint(Epoch::INITIAL).unwrap().segments,
        before.segments
    );
    let changed = vec![calc_flow::AnalyticExpr::row_number("different").unwrap()];
    let mut target = calc_flow::BoundedOrderOperator::new_over(
        "over",
        common::schema(),
        changed,
        common::spec(false),
    )
    .unwrap();
    assert!(
        target
            .restore(&before)
            .unwrap_err()
            .to_string()
            .contains("identity")
    );
    assert!(operator(true).restore(&before).is_err());
    cancellation.cancel();
    assert!(op.on_end(&context, &mut output).await.is_err());
    assert_eq!(
        op.checkpoint(Epoch::INITIAL).unwrap().segments,
        before.segments
    );
    assert!(output.drain("output").is_empty());
}

#[path = "bounded_analytic_over/public_stream.rs"]
mod public_stream;
#[tokio::test]
async fn test_over_public_rust_df_sql_nine_row_modes_and_export_reload() {
    use calc_flow::{
        AnalysisOptions, Comparison, ComparisonCutoff, ComparisonOrdering, CompileMode,
        CompileOptions, ExecutionOptions, HistoryScope, InputOptions, PortabilityRequest, Program,
        SupportStatus, UdfRegistry, table_input,
    };
    use std::collections::BTreeMap;
    let input = table_input("events", common::schema(), InputOptions::default()).unwrap();
    let df = input
        .over(common::expressions(), common::spec(false))
        .unwrap();
    let query = "SELECT *, ROW_NUMBER() OVER (PARTITION BY p ORDER BY score,seq) AS rn, LAG(v,1) OVER (PARTITION BY p ORDER BY score,seq) AS prev, SUM(v) OVER (PARTITION BY p ORDER BY score,seq ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total, SUM(v) OVER (PARTITION BY p ORDER BY score,seq ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) AS last_two FROM CF_OVER_WINDOW('a','ts',0,10,9,8388608,ARRAY['seq'])";
    let sql = calc_flow::sql::sql(query, BTreeMap::from([("a".into(), input)])).unwrap();
    let program = Program::new(
        "nine",
        CompileMode::Batch,
        BTreeMap::from([("df".into(), df), ("sql".into(), sql)]),
    )
    .unwrap();
    let registry = UdfRegistry::new().snapshot();
    let request = PortabilityRequest::new(
        Comparison::FinalRelation,
        HistoryScope::EventWindow,
        ComparisonCutoff::ClosedWindow,
    )
    .unwrap()
    .with_ordering(ComparisonOrdering::Declared);
    let report = program
        .analyze(
            &registry,
            &AnalysisOptions {
                portability: Some(request),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        report.portability,
        Some(SupportStatus::Supported),
        "{report:?}"
    );
    let records = [data(true), data(false)]
        .iter()
        .flat_map(|b| b.table_payload().unwrap().batches().to_vec())
        .collect();
    let raw = calc_flow::Batch::table(records, calc_flow::BatchMetadata::default()).unwrap();
    let plan = program
        .compile_batch(&registry, &CompileOptions::default())
        .await
        .unwrap();
    let results = plan
        .execute(
            BTreeMap::from([("events".into(), raw.clone())]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    for batch in results.outputs.values() {
        assert_oracle(batch);
    }
    let project = program.to_project(&registry).await.unwrap();
    let roundtrip: calc_flow::ProjectSpec =
        serde_json::from_value(serde_json::to_value(project).unwrap()).unwrap();
    let reloaded = calc_flow::compile_project(
        &roundtrip,
        &calc_flow::ProviderRegistry::default(),
        &registry,
    )
    .unwrap();
    let physical = reloaded.single_external_input().unwrap().to_owned();
    let results = reloaded
        .execute(
            BTreeMap::from([(physical, raw)]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    for batch in results.outputs.values() {
        assert_oracle(batch);
    }
    let stream = program
        .for_mode(CompileMode::Stream)
        .unwrap()
        .compile_stream(
            &registry,
            &calc_flow::StreamRequirements::default(),
            &CompileOptions::default(),
        )
        .await
        .unwrap();
    public_stream::verify(stream).await;
}

#[tokio::test]
async fn test_over_empty_history_emits_one_exact_typed_final() {
    let job = StreamJobContext::new(1, "over", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "over", None);
    let mut op = operator(false);
    let expected = op.output_ports()[0].schema().unwrap().clone();
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.on_end(&context, &mut output).await.unwrap();
    let finals = output.drain("output");
    assert_eq!(finals.len(), 1);
    let final_batch = finals[0].as_data().unwrap();
    assert_eq!(final_batch.num_rows(), 0);
    assert_eq!(final_batch.table_payload().unwrap().schema(), &expected);
    op.on_end(&context, &mut output).await.unwrap();
    assert!(output.drain("output").is_empty());
}

async fn finish(
    mut op: calc_flow::BoundedOrderOperator,
    inputs: Vec<calc_flow::Batch>,
) -> calc_flow::Batch {
    let job = StreamJobContext::new(1, "over", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "over", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    for data in inputs {
        op.process_data("input", data, &context, &mut output)
            .await
            .unwrap();
    }
    op.on_end(&context, &mut output).await.unwrap();
    let finals = output.drain("output");
    assert_eq!(finals.len(), 1);
    finals[0].as_data().unwrap().clone()
}
fn ints(batch: &calc_flow::Batch, name: &str) -> Vec<Option<i64>> {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|r| {
            r.column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<calc_flow::arrow::array::Int64Array>()
                .unwrap()
                .iter()
        })
        .collect()
}
fn uints(batch: &calc_flow::Batch, name: &str) -> Vec<u64> {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|r| {
            r.column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<calc_flow::arrow::array::UInt64Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect()
}
#[tokio::test]
async fn test_over_desc_nulls_first_and_u32_max_rows_frame() {
    let mut wire = serde_json::to_value(common::spec(false)).unwrap();
    wire["order"]["order_by"][0]["direction"] = serde_json::json!("descending");
    wire["order"]["order_by"][0]["nulls"] = serde_json::json!("first");
    let spec = serde_json::from_value(wire).unwrap();
    let op = calc_flow::BoundedOrderOperator::new_over(
        "over",
        common::schema(),
        common::expressions(),
        spec,
    )
    .unwrap();
    let output = finish(op, vec![data(true), data(false)]).await;
    assert_eq!(
        uints(&output, "seq"),
        vec![13, 12, 10, 11, 22, 21, 40, 30, 31]
    );
    assert_eq!(
        ints(&output, "total"),
        vec![
            Some(5),
            Some(5),
            Some(8),
            Some(10),
            None,
            Some(7),
            None,
            Some(4),
            Some(4)
        ]
    );
    assert_eq!(
        ints(&output, "prev"),
        vec![
            None,
            Some(5),
            None,
            Some(3),
            None,
            None,
            None,
            None,
            Some(4)
        ]
    );
    let expressions = vec![
        calc_flow::AnalyticExpr::sum(
            "v",
            calc_flow::RowsFrame::preceding(u64::from(u32::MAX)).unwrap(),
            "large",
        )
        .unwrap(),
    ];
    let op = calc_flow::BoundedOrderOperator::new_over(
        "over",
        common::schema(),
        expressions,
        common::spec(false),
    )
    .unwrap();
    let output = finish(op, vec![data(true), data(false)]).await;
    assert_eq!(
        ints(&output, "large"),
        vec![
            Some(3),
            Some(5),
            Some(5),
            Some(10),
            Some(7),
            Some(7),
            None,
            Some(4),
            Some(4)
        ]
    );
}
#[tokio::test]
async fn test_over_negative_window_shift_preserves_membership() {
    use calc_flow::arrow::{array::TimestampMicrosecondArray, record_batch::RecordBatch};
    use std::sync::Arc;
    let mut wire = serde_json::to_value(common::spec(false)).unwrap();
    wire["history"]["start_micros"] = serde_json::json!(-100);
    wire["history"]["end_micros"] = serde_json::json!(-90);
    let spec = serde_json::from_value(wire).unwrap();
    let op = calc_flow::BoundedOrderOperator::new_over(
        "over",
        common::schema(),
        common::expressions(),
        spec,
    )
    .unwrap();
    let inputs = [true, false]
        .into_iter()
        .map(|first| {
            let batch = data(first);
            let record = &batch.table_payload().unwrap().batches()[0];
            let times = record
                .column_by_name("ts")
                .unwrap()
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            let mut columns = record.columns().to_vec();
            columns[1] = Arc::new(TimestampMicrosecondArray::from(
                times.iter().map(|t| t.map(|t| t - 100)).collect::<Vec<_>>(),
            ));
            calc_flow::Batch::table(
                vec![RecordBatch::try_new(record.schema(), columns).unwrap()],
                calc_flow::BatchMetadata::default(),
            )
            .unwrap()
        })
        .collect();
    assert_oracle(&finish(op, inputs).await);
}
#[tokio::test]
async fn test_over_composite_null_partition_keys_compare_equal() {
    use calc_flow::arrow::{
        array::{Int64Array, StringArray, UInt64Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    };
    use calc_flow::{
        AnalyticExpr, NullPlacement, OverOrder, OverSpec, RowsFrame, SortDirection, SortKey,
        StateBudget,
    };
    use std::sync::Arc;
    let schema = Arc::new(Schema::new(vec![
        Field::new("p", DataType::Utf8, true),
        Field::new("q", DataType::Int64, true),
        Field::new("score", DataType::Int64, false),
        Field::new("seq", DataType::UInt64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let raw = calc_flow::Batch::table(
        vec![
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec![
                        None,
                        None,
                        Some("A"),
                        Some("A"),
                        Some("A"),
                        Some("A"),
                    ])),
                    Arc::new(Int64Array::from(vec![
                        None,
                        None,
                        None,
                        None,
                        Some(1),
                        Some(1),
                    ])),
                    Arc::new(Int64Array::from(vec![1, 0, 1, 0, 0, 1])),
                    Arc::new(UInt64Array::from(vec![0, 1, 2, 3, 4, 5])),
                    Arc::new(Int64Array::from(vec![2, 3, 4, 5, 7, 8])),
                ],
            )
            .unwrap(),
        ],
        calc_flow::BatchMetadata::default(),
    )
    .unwrap();
    let order = OverOrder::new(
        vec!["p".into(), "q".into()],
        vec![SortKey::new("score", SortDirection::Ascending, NullPlacement::Last).unwrap()],
        vec!["seq".into()],
    )
    .unwrap();
    let spec = OverSpec::bounded_source(order, StateBudget::new(6, 8 << 20).unwrap()).unwrap();
    let op = calc_flow::BoundedOrderOperator::new_over(
        "over",
        schema,
        vec![
            AnalyticExpr::row_number("rn").unwrap(),
            AnalyticExpr::sum("v", RowsFrame::cumulative(), "total").unwrap(),
        ],
        spec,
    )
    .unwrap();
    let output = finish(op, vec![raw]).await;
    assert_eq!(uints(&output, "seq"), vec![4, 5, 3, 2, 1, 0]);
    assert_eq!(uints(&output, "rn"), vec![1, 2, 1, 2, 1, 2]);
    assert_eq!(
        ints(&output, "total"),
        vec![Some(7), Some(15), Some(5), Some(9), Some(3), Some(5)]
    );
}
