mod asof_support;

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use calc_flow::{
    AsofJoinSide, AsofStateLimits, Batch, BatchMetadata, CancellationToken, EdgeCollector,
    EventTime, IngressProgress, IngressProgressSnapshot, IngressState, JsonMap, OperatorMetadata,
    StreamAsofJoinOperator, StreamAsofJoinSpec, StreamJobContext, StreamOperator,
    StreamOperatorContext,
};
use datafusion::arrow::{
    array::{BooleanArray, Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};

fn watermark_context(job: &StreamJobContext, time: i64) -> StreamOperatorContext<'_> {
    let progress = IngressProgressSnapshot::new(BTreeMap::from([
        (
            "left".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(time))),
        ),
        (
            "right".into(),
            IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(time))),
        ),
    ]));
    StreamOperatorContext::with_ingress_progress(job, "asof", None, progress)
}

fn values(output: &mut EdgeCollector) -> Vec<Option<i64>> {
    let mut values = Vec::new();
    for message in output.drain("output") {
        for batch in message
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()
        {
            let column = batch
                .column_by_name("right__value")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            values.extend(column.iter());
        }
    }
    values
}

#[tokio::test]
async fn time_extremes_and_maximum_tolerance_do_not_wrap_or_end_at_max_watermark() {
    const MAX_T: u64 = 9_007_199_254_740_991;
    for (left, right, tolerance, expected) in [
        (i64::MIN, i64::MIN, MAX_T, Some(7)),
        (i64::MIN, i64::MIN + 1, MAX_T, None),
        (i64::MAX, i64::MIN, MAX_T, None),
        (i64::MAX, i64::MAX - 9_007_199_254_740_991, MAX_T, Some(7)),
        (i64::MAX, i64::MAX, 0, Some(7)),
    ] {
        let mut op = asof_support::operator(tolerance);
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let initial = StreamOperatorContext::new(&job, "asof", None);
        let mut output = EdgeCollector::new(op.output_ports().to_vec());
        op.process_data(
            "right",
            asof_support::batch(&[("A", right, 1, 7)]),
            &initial,
            &mut output,
        )
        .await
        .unwrap();
        op.process_data(
            "left",
            asof_support::batch(&[("A", left, 1, 99)]),
            &initial,
            &mut output,
        )
        .await
        .unwrap();
        let equal = watermark_context(&job, left);
        op.on_watermark(EventTime::from_micros(left), &equal, &mut output)
            .await
            .unwrap();
        assert!(
            values(&mut output).is_empty(),
            "watermark equality is not EOF"
        );
        op.on_end(&equal, &mut output).await.unwrap();
        assert_eq!(
            values(&mut output),
            [expected],
            "left={left}, right={right}"
        );
        assert_eq!(op.status().emitted_left_rows, 1);
    }
}

fn composite_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("venue", DataType::Boolean, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::Int64, false),
        Field::new("text", DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn composite_batch(rows: &[(i64, bool, i64, &str, i64)]) -> Batch {
    let record = RecordBatch::try_new(
        composite_schema(),
        vec![
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.0).collect::<Vec<_>>(),
            )),
            Arc::new(BooleanArray::from(
                rows.iter().map(|row| row.1).collect::<Vec<_>>(),
            )),
            Arc::new(TimestampMicrosecondArray::from(vec![100; rows.len()]).with_timezone("UTC")),
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.2).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter().map(|row| row.3).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                rows.iter().map(|row| row.4).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

#[tokio::test]
async fn composite_keys_and_numeric_then_utf8_sequence_choose_the_same_row() {
    let side = |prefix: &str| {
        AsofJoinSide::new(
            vec!["key".into(), "venue".into()],
            "time".into(),
            vec!["sequence".into(), "text".into()],
            prefix.into(),
        )
        .unwrap()
    };
    let spec = StreamAsofJoinSpec::new(
        side("left"),
        side("right"),
        Duration::ZERO,
        AsofStateLimits::new(100, 1_048_576).unwrap(),
    )
    .unwrap();
    let rows = [
        (-1, true, -1, "中", 1),
        (-1, true, 2, "a", 2),
        (-1, true, 2, "é", 3),
        (-1, true, 2, "中", 4),
        (-1, false, 999, "z", 5),
    ];
    for reverse in [false, true] {
        let mut op = StreamAsofJoinOperator::new(
            "asof",
            composite_schema(),
            composite_schema(),
            spec.clone(),
        )
        .unwrap();
        let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new());
        let cx = StreamOperatorContext::new(&job, "asof", None);
        let mut output = EdgeCollector::new(op.output_ports().to_vec());
        let mut ordered = rows;
        if reverse {
            ordered.reverse();
        }
        for row in ordered {
            op.process_data("right", composite_batch(&[row]), &cx, &mut output)
                .await
                .unwrap();
        }
        op.process_data(
            "left",
            composite_batch(&[(-1, true, 1, "trade", 99)]),
            &cx,
            &mut output,
        )
        .await
        .unwrap();
        op.on_end(&cx, &mut output).await.unwrap();
        assert_eq!(values(&mut output), [Some(4)]);
    }
}

#[test]
fn native_constructor_rejects_nullable_wrong_time_and_unordered_identity_types() {
    let valid = asof_support::schema();
    for replacement in [
        Field::new("key", DataType::Utf8, true),
        Field::new("key", DataType::Float64, false),
        Field::new("key", DataType::LargeUtf8, false),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::Float64, false),
    ] {
        let fields: Vec<_> = valid
            .fields()
            .iter()
            .map(|field| {
                if field.name() == replacement.name() {
                    Arc::new(replacement.clone())
                } else {
                    field.clone()
                }
            })
            .collect();
        let result = StreamAsofJoinOperator::new(
            "asof",
            valid.clone(),
            Arc::new(Schema::new(fields)),
            asof_support::spec(10),
        );
        assert!(result.is_err(), "unexpectedly accepted {replacement:?}");
    }
}
