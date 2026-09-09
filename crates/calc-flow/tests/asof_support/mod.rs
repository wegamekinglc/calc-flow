use calc_flow::{
    AsofJoinSide, AsofStateLimits, Batch, BatchMetadata, StreamAsofJoinOperator, StreamAsofJoinSpec,
};
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use std::{sync::Arc, time::Duration};

pub fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "time",
            DataType::Timestamp(
                datafusion::arrow::datatypes::TimeUnit::Microsecond,
                Some("UTC".into()),
            ),
            false,
        ),
        Field::new("sequence", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

pub fn spec(tolerance: u64) -> StreamAsofJoinSpec {
    StreamAsofJoinSpec::new(
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["sequence".into()],
            "left".into(),
        )
        .unwrap(),
        AsofJoinSide::new(
            vec!["key".into()],
            "time".into(),
            vec!["sequence".into()],
            "right".into(),
        )
        .unwrap(),
        Duration::from_micros(tolerance),
        AsofStateLimits::new(100_000, 64 * 1024 * 1024).unwrap(),
    )
    .unwrap()
}

pub fn operator(tolerance: u64) -> StreamAsofJoinOperator {
    StreamAsofJoinOperator::new("asof", schema(), schema(), spec(tolerance)).unwrap()
}

pub fn batch(rows: &[(&str, i64, i64, i64)]) -> Batch {
    Batch::table(
        vec![
            RecordBatch::try_new(
                schema(),
                vec![
                    Arc::new(StringArray::from(
                        rows.iter().map(|r| r.0).collect::<Vec<_>>(),
                    )),
                    Arc::new(
                        TimestampMicrosecondArray::from(
                            rows.iter().map(|r| r.1).collect::<Vec<_>>(),
                        )
                        .with_timezone("UTC"),
                    ),
                    Arc::new(Int64Array::from(
                        rows.iter().map(|r| r.2).collect::<Vec<_>>(),
                    )),
                    Arc::new(Int64Array::from(
                        rows.iter().map(|r| r.3).collect::<Vec<_>>(),
                    )),
                ],
            )
            .unwrap(),
        ],
        BatchMetadata::default(),
    )
    .unwrap()
}
