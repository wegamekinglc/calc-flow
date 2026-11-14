use calc_flow::arrow::{
    array::{Int64Array, UInt64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use calc_flow::{
    AnalyticExpr, Batch, BatchMetadata, CompileMode, CompileOptions, ExecutionOptions,
    InputOptions, NullPlacement, OverOrder, OverSpec, Program, RowsFrame, SortDirection, SortKey,
    StateBudget, UdfRegistry, table_input,
};
use std::{collections::BTreeMap, sync::Arc};

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("v", DataType::Int64, true),
        Field::new("seq", DataType::UInt64, false),
    ]))
}
async fn plan() -> calc_flow::BatchExecutionPlan {
    let input = table_input("events", schema(), InputOptions::default()).unwrap();
    let order = OverOrder::new(
        vec![],
        vec![SortKey::new("v", SortDirection::Ascending, NullPlacement::Last).unwrap()],
        vec!["seq".into()],
    )
    .unwrap();
    let out = input
        .over(
            vec![
                AnalyticExpr::row_number("rn").unwrap(),
                AnalyticExpr::lag("v", "prev").unwrap(),
                AnalyticExpr::sum("v", RowsFrame::cumulative(), "total").unwrap(),
            ],
            OverSpec::bounded_source(order, StateBudget::new(9, 8 << 20).unwrap()).unwrap(),
        )
        .unwrap();
    Program::new(
        "over",
        CompileMode::Batch,
        BTreeMap::from([("out".into(), out)]),
    )
    .unwrap()
    .compile_batch(&UdfRegistry::new().snapshot(), &CompileOptions::default())
    .await
    .unwrap()
}
fn raw(values: Vec<i64>) -> Batch {
    let count = values.len();
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(Int64Array::from(values)),
                    Arc::new(UInt64Array::from((0..count as u64).collect::<Vec<_>>())),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}
fn integers(batch: &Batch, name: &str) -> Vec<Option<i64>> {
    batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|r| {
            r.column_by_name(name)
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
        })
        .collect()
}
#[tokio::test]
async fn test_over_batch_history_and_plan_isolation() {
    let plan = plan().await;
    for (values, expected, sums) in [
        (vec![3, 1, 2], vec![1, 2, 3], vec![1, 3, 6]),
        (vec![9], vec![9], vec![9]),
        (vec![], vec![], vec![]),
    ] {
        let count = values.len();
        let result = plan
            .execute(
                BTreeMap::from([("events".into(), raw(values))]),
                ExecutionOptions::default(),
            )
            .await
            .unwrap();
        let out = &result.outputs["out"];
        let table = out.table_payload().unwrap();
        assert_eq!(
            table.schema().field_with_name("rn").unwrap(),
            &Field::new("rn", DataType::UInt64, false)
        );
        assert_eq!(
            integers(out, "v"),
            expected.iter().copied().map(Some).collect::<Vec<_>>()
        );
        assert_eq!(
            integers(out, "total"),
            sums.into_iter().map(Some).collect::<Vec<_>>()
        );
        let rn = table
            .batches()
            .iter()
            .flat_map(|r| {
                r.column_by_name("rn")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        assert_eq!(rn, (1..=count as u64).collect::<Vec<_>>());
        let previous = if count == 0 {
            vec![]
        } else {
            std::iter::once(None)
                .chain(expected[..count - 1].iter().copied().map(Some))
                .collect()
        };
        assert_eq!(integers(out, "prev"), previous);
    }
}
