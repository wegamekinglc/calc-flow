use super::*;

fn record(values: ArrayRef) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("value", values.data_type().clone(), true),
        ])),
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![0; values.len()])),
            values,
        ],
    )
    .unwrap()
}

fn spec(functions: &[AggregateFunction]) -> WindowSpec {
    functions.iter().enumerate().fold(
        WindowSpec::tumbling("time", Duration::from_micros(10)).unwrap(),
        |spec, (ordinal, &function)| {
            spec.aggregate(function, "value", &format!("result_{ordinal}"))
                .unwrap()
        },
    )
}

fn string_record(large: bool, values: &[Option<&str>]) -> RecordBatch {
    let array: ArrayRef = if large {
        Arc::new(LargeStringArray::from(values.to_vec()))
    } else {
        Arc::new(StringArray::from(values.to_vec()))
    };
    record(array)
}

fn extremes(minimum: &str, maximum: &str) -> AccumulatorRow {
    AccumulatorRow {
        group_values: vec![],
        aggregates: vec![
            AccumulatorValue::Min(Some(ScalarValue::String(minimum.into()))),
            AccumulatorValue::Max(Some(ScalarValue::String(maximum.into()))),
        ],
    }
}

#[test]
fn unchanged_string_extremes_do_not_copy_input_values() {
    for large in [false, true] {
        let values = [None, Some("aa\0aa"), Some("mid\0dle"), Some("é"), Some("𐀀")];
        let input = std::iter::once(Some("outside-slice"))
            .chain(values.into_iter().cycle().take(8192))
            .collect::<Vec<_>>();
        let record = string_record(large, &input).slice(1, 8192);
        let spec = spec(&[AggregateFunction::Min, AggregateFunction::Max]);
        let operator =
            WindowAggregateOperator::new("window", record.schema(), spec.clone()).unwrap();
        let columns = RecordColumns::new(&record, &spec, &operator.compiled, "window").unwrap();
        let mut row = extremes("aa\0aa", "𐀀");
        let allocation = allocation_counter::measure(|| {
            for index in 0..record.num_rows() {
                update_accumulators(&mut row, &columns, index, &spec, "window").unwrap();
            }
        });
        assert_eq!(state(&row), state(&extremes("aa\0aa", "𐀀")));
        assert_eq!(allocation.count_total, 0, "large={large}: {allocation:?}");
    }
}

#[test]
fn string_extremes_copy_only_new_witnesses() {
    for large in [false, true] {
        let record = string_record(
            large,
            &[
                Some("m"),
                None,
                Some("a"),
                Some("z"),
                Some("a"),
                Some("z"),
                Some("á"),
                Some("zz"),
            ],
        );
        let spec = spec(&[AggregateFunction::Min, AggregateFunction::Max]);
        let operator =
            WindowAggregateOperator::new("window", record.schema(), spec.clone()).unwrap();
        let columns = RecordColumns::new(&record, &spec, &operator.compiled, "window").unwrap();
        let mut row = extremes("m", "m");
        let allocation = allocation_counter::measure(|| {
            for index in 0..record.num_rows() {
                update_accumulators(&mut row, &columns, index, &spec, "window").unwrap();
            }
        });
        assert_eq!(state(&row), state(&extremes("a", "á")));
        assert_eq!(allocation.count_total, 3, "large={large}: {allocation:?}");
    }
}

fn reference_update(
    row: &mut AccumulatorRow,
    columns: &RecordColumns<'_>,
    index: usize,
    spec: &WindowSpec,
) -> Result<()> {
    for (ordinal, ((aggregate, (column, _)), accumulator)) in spec
        .aggregates
        .iter()
        .zip(&columns.aggregates)
        .zip(&mut row.aggregates)
        .enumerate()
    {
        if let Some(value) = aggregate_input(aggregate.function, column, index) {
            update_accumulator(accumulator, aggregate.function, value).map_err(|message| {
                operator_error(
                    "window",
                    &format!("window.aggregates[{ordinal}] update failed: {message}"),
                )
            })?;
        }
    }
    Ok(())
}

fn state(row: &AccumulatorRow) -> Vec<String> {
    row.aggregates
        .iter()
        .map(|value| match value {
            AccumulatorValue::FloatSum(value) => format!(
                "{:?}",
                value.map(|sum| (sum.sum.to_bits(), sum.correction.to_bits()))
            ),
            AccumulatorValue::FloatAverage { sum, count } => {
                format!("{}:{}:{count}", sum.sum.to_bits(), sum.correction.to_bits())
            }
            AccumulatorValue::SignedAverage { sum, count } => format!("signed:{sum}:{count}"),
            AccumulatorValue::UnsignedAverage { sum, count } => format!("unsigned:{sum}:{count}"),
            _ => format!("{:?}", finalize_accumulator(value).unwrap()),
        })
        .collect()
}

fn assert_reference(values: ArrayRef, functions: &[AggregateFunction]) {
    let original = record(values);
    let record = original.slice(1, original.num_rows() - 1);
    let spec = spec(functions);
    let operator = WindowAggregateOperator::new("window", record.schema(), spec.clone()).unwrap();
    let columns = RecordColumns::new(&record, &spec, &operator.compiled, "window").unwrap();
    let mut actual = new_accumulator_row(&spec, &operator.compiled, vec![]);
    let mut expected = actual.clone();
    for index in 0..record.num_rows() {
        reference_update(&mut expected, &columns, index, &spec).unwrap();
        update_accumulators(&mut actual, &columns, index, &spec, "window").unwrap();
        assert_eq!(
            state(&actual),
            state(&expected),
            "type={}, row={index}",
            record.column(1).data_type()
        );
    }
}

#[test]
fn numeric_updates_preserve_every_prefix_and_float_state_bits() {
    let functions = [
        AggregateFunction::Count,
        AggregateFunction::Sum,
        AggregateFunction::Min,
        AggregateFunction::Max,
        AggregateFunction::Avg,
    ];
    let signed = vec![Some(99), None, Some(-4), Some(1), Some(0), Some(7)];
    let unsigned = vec![Some(99), None, Some(4), Some(1), Some(0), Some(7)];
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int8Array::from(signed.clone())),
        Arc::new(Int16Array::from(
            signed
                .iter()
                .map(|value| value.map(i16::from))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int32Array::from(
            signed
                .iter()
                .map(|value| value.map(i32::from))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            signed
                .iter()
                .map(|value| value.map(i64::from))
                .collect::<Vec<_>>(),
        )),
        Arc::new(UInt8Array::from(unsigned.clone())),
        Arc::new(UInt16Array::from(
            unsigned
                .iter()
                .map(|value| value.map(u16::from))
                .collect::<Vec<_>>(),
        )),
        Arc::new(UInt32Array::from(
            unsigned
                .iter()
                .map(|value| value.map(u32::from))
                .collect::<Vec<_>>(),
        )),
        Arc::new(UInt64Array::from(
            unsigned
                .iter()
                .map(|value| value.map(u64::from))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float32Array::from(vec![
            Some(99.0),
            None,
            Some(1e10),
            Some(1.0),
            Some(-1e10),
            Some(-0.0),
            Some(0.0),
            Some(f32::from_bits(0x7fc0_abcd)),
            Some(f32::INFINITY),
            Some(f32::NEG_INFINITY),
        ])),
        Arc::new(Float64Array::from(vec![
            Some(99.0),
            None,
            Some(1e16),
            Some(1.0),
            Some(-1e16),
            Some(-0.0),
            Some(0.0),
            Some(f64::from_bits(0x7ff8_0000_0000_abcd)),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
        ])),
    ];
    for array in arrays {
        assert_reference(array, &functions);
    }
}

#[test]
fn opaque_count_and_temporal_extremes_preserve_generic_results() {
    use datafusion::arrow::array::BinaryArray;
    assert_reference(
        Arc::new(BinaryArray::from(vec![
            Some(b"outside".as_slice()),
            Some(b"".as_slice()),
            None,
            Some(b"data".as_slice()),
        ])),
        &[AggregateFunction::Count],
    );
    let functions = [
        AggregateFunction::Count,
        AggregateFunction::Min,
        AggregateFunction::Max,
    ];
    assert_reference(
        Arc::new(BooleanArray::from(vec![
            Some(true),
            None,
            Some(false),
            Some(true),
        ])),
        &functions,
    );
    assert_reference(
        Arc::new(Date32Array::from(vec![Some(99), None, Some(-4), Some(7)])),
        &functions,
    );
    assert_reference(
        Arc::new(Date64Array::from(vec![
            Some(99),
            None,
            Some(-86_400_000),
            Some(86_400_000),
        ])),
        &functions,
    );
    assert_reference(
        Arc::new(TimestampMicrosecondArray::from(vec![
            Some(99),
            None,
            Some(-4),
            Some(7),
        ])),
        &functions,
    );
}

#[test]
fn numeric_failures_preserve_first_aggregate_and_partial_scratch_state() {
    let record = record(Arc::new(Int64Array::from(vec![1])));
    let spec = spec(&[
        AggregateFunction::Count,
        AggregateFunction::Sum,
        AggregateFunction::Avg,
    ]);
    let operator = WindowAggregateOperator::new("window", record.schema(), spec.clone()).unwrap();
    let columns = RecordColumns::new(&record, &spec, &operator.compiled, "window").unwrap();
    for (count, sum, average_count) in [
        (u64::MAX, i64::MAX, 0),
        (0, i64::MAX, u64::MAX),
        (0, 0, u64::MAX),
    ] {
        let mut actual = AccumulatorRow {
            group_values: vec![],
            aggregates: vec![
                AccumulatorValue::Count(count),
                AccumulatorValue::SignedSum(Some(i128::from(sum))),
                AccumulatorValue::SignedAverage {
                    sum: 0,
                    count: average_count,
                },
            ],
        };
        let mut expected = actual.clone();
        let reference = reference_update(&mut expected, &columns, 0, &spec).unwrap_err();
        let error = update_accumulators(&mut actual, &columns, 0, &spec, "window").unwrap_err();
        assert_eq!(error.to_string(), reference.to_string());
        assert_eq!(state(&actual), state(&expected));
    }
}
