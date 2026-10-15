use super::*;
use datafusion::arrow::{
    array::{
        BinaryViewArray, FixedSizeListArray, Int16Array, LargeBinaryArray, LargeListArray,
        LargeListViewArray, LargeStringArray, ListArray, ListViewArray, MapArray, RunArray,
        StringViewArray, StructArray, UnionArray, make_array, new_null_array,
    },
    buffer::{NullBuffer, OffsetBuffer, ScalarBuffer},
    datatypes::{Int16Type, UnionFields, UnionMode},
};

pub(super) const ROWS: usize = 70;

pub(super) fn validity(length: usize) -> NullBuffer {
    NullBuffer::from((0..length).map(|row| row % 3 != 0).collect::<Vec<_>>())
}

pub(super) fn fixed() -> Vec<ArrayRef> {
    let types = [
        DataType::Null,
        DataType::Boolean,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::UInt8,
        DataType::UInt16,
        DataType::UInt32,
        DataType::UInt64,
        DataType::Float16,
        DataType::Float32,
        DataType::Float64,
        DataType::Date32,
        DataType::Date64,
        DataType::Time32(TimeUnit::Second),
        DataType::Time32(TimeUnit::Millisecond),
        DataType::Time64(TimeUnit::Microsecond),
        DataType::Time64(TimeUnit::Nanosecond),
        DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
        DataType::Duration(TimeUnit::Second),
        DataType::Interval(IntervalUnit::YearMonth),
        DataType::Interval(IntervalUnit::DayTime),
        DataType::Interval(IntervalUnit::MonthDayNano),
        DataType::Decimal32(9, 2),
        DataType::Decimal64(18, 2),
        DataType::Decimal128(38, 2),
        DataType::Decimal256(76, 2),
        DataType::FixedSizeBinary(5),
    ];
    types
        .iter()
        .map(|data_type| {
            let data = new_null_array(data_type, ROWS).to_data();
            if matches!(data_type, DataType::Null) {
                return make_array(data);
            }
            make_array(
                data.into_builder()
                    .nulls(Some(validity(ROWS)))
                    .build()
                    .unwrap(),
            )
        })
        .collect()
}

pub(super) fn variable() -> Vec<ArrayRef> {
    let strings = (0..ROWS)
        .map(|row| (row % 3 != 0).then(|| "x".repeat(row)))
        .collect::<Vec<_>>();
    let bytes = strings
        .iter()
        .map(|value| value.as_deref().map(str::as_bytes))
        .collect::<Vec<_>>();
    vec![
        Arc::new(StringArray::from(strings.clone())),
        Arc::new(LargeStringArray::from(strings.clone())),
        Arc::new(StringViewArray::from(strings.clone())),
        Arc::new(BinaryArray::from(bytes.clone())),
        Arc::new(LargeBinaryArray::from(bytes.clone())),
        Arc::new(BinaryViewArray::from(bytes)),
    ]
}

pub(super) fn nested() -> Vec<ArrayRef> {
    let child = Arc::new(
        (0..ROWS * 2)
            .map(|row| (row % 5 != 0).then_some("child"))
            .collect::<StringArray>(),
    ) as ArrayRef;
    let field = Arc::new(Field::new("item", DataType::Utf8, true));
    let nulls = Some(validity(ROWS));
    let entries = StructArray::new(
        vec![
            Field::new("key", DataType::Int32, false),
            Field::new("value", DataType::Utf8, true),
        ]
        .into(),
        vec![
            Arc::new(Int32Array::from_iter_values(
                0..i32::try_from(ROWS * 2).unwrap(),
            )),
            Arc::clone(&child),
        ],
        None,
    );
    vec![
        Arc::new(ListArray::new(
            Arc::clone(&field),
            OffsetBuffer::from_lengths([2; ROWS]),
            Arc::clone(&child),
            nulls.clone(),
        )),
        Arc::new(LargeListArray::new(
            Arc::clone(&field),
            OffsetBuffer::from_lengths([2; ROWS]),
            Arc::clone(&child),
            nulls.clone(),
        )),
        Arc::new(ListViewArray::new(
            Arc::clone(&field),
            ScalarBuffer::from(vec![3_i32; ROWS]),
            ScalarBuffer::from(vec![5_i32; ROWS]),
            Arc::clone(&child),
            nulls.clone(),
        )),
        Arc::new(LargeListViewArray::new(
            Arc::clone(&field),
            ScalarBuffer::from(vec![2_i64; ROWS]),
            ScalarBuffer::from(vec![4_i64; ROWS]),
            Arc::clone(&child),
            nulls.clone(),
        )),
        Arc::new(FixedSizeListArray::new(
            field,
            2,
            Arc::clone(&child),
            nulls.clone(),
        )),
        Arc::new(StructArray::new(
            vec![Field::new("text", DataType::Utf8, true)].into(),
            vec![child.slice(2, ROWS)],
            nulls.clone(),
        )),
        Arc::new(MapArray::new(
            Arc::new(Field::new("entries", entries.data_type().clone(), false)),
            OffsetBuffer::from_lengths([2; ROWS]),
            entries,
            nulls,
            false,
        )),
        dictionary(),
        Arc::new(
            RunArray::<Int32Type>::try_new(
                &Int32Array::from(vec![2, i32::try_from(ROWS).unwrap()]),
                &StringArray::from(vec![Some("run"), None]),
            )
            .unwrap(),
        ),
        union(UnionMode::Dense),
        union(UnionMode::Sparse),
    ]
}

pub(super) fn dictionary() -> ArrayRef {
    Arc::new(DictionaryArray::<Int32Type>::new(
        (0..ROWS)
            .map(|row| (row % 4 != 0).then(|| i32::try_from(row % 3).unwrap()))
            .collect::<Int32Array>(),
        Arc::new(StringArray::from(vec![
            Some("short"),
            None,
            Some("long dictionary value"),
        ])),
    ))
}

pub(super) fn unsupported_dictionary() -> ArrayRef {
    Arc::new(DictionaryArray::<Int16Type>::new(
        Int16Array::from(vec![Some(0), None]),
        Arc::new(StringArray::from(vec!["value"])),
    ))
}

fn union(mode: UnionMode) -> ArrayRef {
    let fields = UnionFields::try_new(
        vec![0, 1],
        vec![
            Field::new("number", DataType::Int64, true),
            Field::new("text", DataType::Utf8, true),
        ],
    )
    .unwrap();
    let offsets = (mode == UnionMode::Dense).then(|| {
        (0..ROWS)
            .map(|row| i32::try_from(row / 2).unwrap())
            .collect::<ScalarBuffer<_>>()
    });
    Arc::new(
        UnionArray::try_new(
            fields,
            (0..ROWS)
                .map(|row| i8::try_from(row % 2).unwrap())
                .collect::<ScalarBuffer<_>>(),
            offsets,
            vec![
                Arc::new(Int64Array::from(vec![Some(7); ROWS])),
                Arc::new(
                    (0..ROWS)
                        .map(|row| (row % 3 != 0).then_some("union"))
                        .collect::<StringArray>(),
                ),
            ],
        )
        .unwrap(),
    )
}

pub(super) fn record(payload: &[ArrayRef]) -> RecordBatch {
    let rows = payload.first().map_or(ROWS, Array::len);
    let mut columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![7; rows])),
        Arc::new(TimestampMicrosecondArray::from(vec![0; rows]).with_timezone("UTC")),
    ];
    columns.extend_from_slice(payload);
    let fields = columns
        .iter()
        .enumerate()
        .map(|(index, column)| Field::new(format!("c{index}"), column.data_type().clone(), true))
        .collect::<Vec<_>>();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap()
}

pub(super) fn operator(record: &RecordBatch) -> StreamJoinOperator {
    StreamJoinOperator::new(
        "match",
        record.schema(),
        record.schema(),
        StreamJoinSpec::inner(
            ["c0"],
            ["c0"],
            "c1",
            "c1",
            JoinTimeBounds::new(Duration::from_micros(10), Duration::from_micros(10)).unwrap(),
            JoinStateLimits::new(10_000, 10_000_000, 10_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}
