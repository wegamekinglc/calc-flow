use calc_flow::arrow::{
    array::{ArrayRef, Int64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use calc_flow::{
    AnalyticExpr, Batch, BatchMetadata, BoundedOrderOperator, EventTime, LatePolicySpec,
    NullPlacement, OverOrder, OverSpec, OverWindow, RowsFrame, SortDirection, SortKey, StateBudget,
};
use std::sync::Arc;

pub fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("p", DataType::Utf8, true),
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
        Field::new("score", DataType::Int64, true),
        Field::new("seq", DataType::UInt64, false),
        Field::new("v", DataType::Int64, true),
    ]))
}
pub fn expressions() -> Vec<AnalyticExpr> {
    vec![
        AnalyticExpr::row_number("rn").unwrap(),
        AnalyticExpr::lag("v", "prev").unwrap(),
        AnalyticExpr::sum("v", RowsFrame::cumulative(), "total").unwrap(),
        AnalyticExpr::sum("v", RowsFrame::preceding(1).unwrap(), "last_two").unwrap(),
    ]
}
pub fn spec(drop: bool) -> OverSpec {
    OverSpec::closed_window(
        OverWindow::new("ts", EventTime::from_micros(0), EventTime::from_micros(10)).unwrap(),
        OverOrder::new(
            vec!["p".into()],
            vec![SortKey::new("score", SortDirection::Ascending, NullPlacement::Last).unwrap()],
            vec!["seq".into()],
        )
        .unwrap(),
        StateBudget::new(9, 8 << 20).unwrap(),
    )
    .unwrap()
    .with_late_policy(if drop {
        LatePolicySpec::Drop { metrics_version: 1 }
    } else {
        LatePolicySpec::Error {
            scope: calc_flow::LateErrorScope::Envelope,
        }
    })
    .unwrap()
}
pub fn operator(drop: bool) -> BoundedOrderOperator {
    BoundedOrderOperator::new_over("over", schema(), expressions(), spec(drop)).unwrap()
}
pub fn data(first: bool) -> Batch {
    let columns: Vec<ArrayRef> = if first {
        vec![
            Arc::new(StringArray::from(vec![Some("A"), Some("A"), Some("B")])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(0),
                Some(1),
                Some(9),
            ])),
            Arc::new(Int64Array::from(vec![Some(2), Some(1), None])),
            Arc::new(UInt64Array::from(vec![12, 11, 22])),
            Arc::new(Int64Array::from(vec![None, Some(2), None])),
        ]
    } else {
        vec![
            Arc::new(StringArray::from(vec![
                Some("A"),
                Some("A"),
                Some("B"),
                Some("C"),
                None,
                None,
                Some("A"),
                Some("A"),
            ])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(9),
                Some(8),
                Some(2),
                Some(5),
                Some(3),
                Some(4),
                Some(10),
                None,
            ])),
            Arc::new(Int64Array::from(vec![
                Some(1),
                Some(3),
                Some(0),
                Some(0),
                Some(1),
                Some(1),
                Some(0),
                Some(0),
            ])),
            Arc::new(UInt64Array::from(vec![10, 13, 21, 40, 30, 31, 99, 100])),
            Arc::new(Int64Array::from(vec![
                Some(3),
                Some(5),
                Some(7),
                None,
                Some(4),
                None,
                Some(100),
                Some(100),
            ])),
        ]
    };
    Batch::table(
        vec![RecordBatch::try_new(schema(), columns).unwrap()],
        BatchMetadata::default(),
    )
    .unwrap()
}
pub fn assert_oracle(batch: &Batch) {
    let table = batch.table_payload().unwrap();
    assert_eq!(batch.num_rows(), 9);
    let seq = table
        .batches()
        .iter()
        .flat_map(|r| {
            r.column_by_name("seq")
                .unwrap()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    assert_eq!(seq, vec![10, 11, 12, 13, 21, 22, 40, 30, 31]);
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
    assert_eq!(rn, vec![1, 2, 3, 4, 1, 2, 1, 1, 2]);
    for (name, expected) in [
        (
            "prev",
            vec![
                None,
                Some(3),
                Some(2),
                None,
                None,
                Some(7),
                None,
                None,
                Some(4),
            ],
        ),
        (
            "total",
            vec![
                Some(3),
                Some(5),
                Some(5),
                Some(10),
                Some(7),
                Some(7),
                None,
                Some(4),
                Some(4),
            ],
        ),
        (
            "last_two",
            vec![
                Some(3),
                Some(5),
                Some(2),
                Some(5),
                Some(7),
                Some(7),
                None,
                Some(4),
                Some(4),
            ],
        ),
    ] {
        let actual = table
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
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{name}");
    }
}
