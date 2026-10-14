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

/// Picks a column value, nulling every `null_nth`-th roll.
fn pick<T>(roll: u64, null_nth: u64, choose: impl Fn(u64) -> T) -> Option<T> {
    if roll.is_multiple_of(null_nth) {
        None
    } else {
        Some(choose(roll))
    }
}

fn columnar_equivalence_record() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new("group", DataType::Utf8, true),
        Field::new("signed", DataType::Int64, true),
        Field::new("unsigned", DataType::UInt64, true),
        Field::new("float", DataType::Float64, true),
        Field::new("small", DataType::Float32, true),
    ]));
    // Deterministic pseudo-random rows: out-of-order times, repeats, nulls in
    // every column, late rows against the advanced watermark, extreme floats.
    let groups = ["alpha", "beta", "gamma"];
    let floats = [
        0.5,
        -1.25,
        1e16,
        -1e16,
        f64::from_bits(0x7ff8_0000_0000_abcd),
        f64::INFINITY,
        f64::NEG_INFINITY,
        -0.0,
    ];
    let mut times = Vec::new();
    let mut group_column = Vec::new();
    let mut signed = Vec::new();
    let mut unsigned = Vec::new();
    let mut float_column = Vec::new();
    let mut small = Vec::new();
    let mut seed = 0x1234_5678_9abc_def0_u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(0x6364_2364_6905)
            .wrapping_add(1_441_269_409);
        seed >> 11
    };
    for roll in (0..512).map(|_| next()) {
        times.push(pick(roll, 8, |r| i64::try_from(r % 45).unwrap() - 5));
        group_column.push(pick(roll, 11, |r| groups[usize::try_from(r % 3).unwrap()]));
        signed.push(pick(roll, 5, |r| i64::try_from(r % 2001).unwrap() - 1000));
        unsigned.push(pick(roll, 7, |r| r % 10_000));
        float_column.push(pick(roll, 6, |r| {
            floats[usize::try_from(r / 6).unwrap() % floats.len()]
        }));
        small.push(pick(roll, 4, |r| {
            f32::from(u16::try_from(r % 1000).unwrap()) * 0.25 - 100.0
        }));
    }
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(TimestampMicrosecondArray::from(times)),
            Arc::new(StringArray::from(group_column)),
            Arc::new(Int64Array::from(signed)),
            Arc::new(UInt64Array::from(unsigned)),
            Arc::new(Float64Array::from(float_column)),
            Arc::new(Float32Array::from(small)),
        ],
    )
    .unwrap()
}

fn columnar_equivalence_spec() -> WindowSpec {
    WindowSpec::hopping("time", Duration::from_micros(30), Duration::from_micros(10))
        .unwrap()
        .group_by(["group"])
        .unwrap()
        .aggregate(AggregateFunction::Count, "float", "count_float")
        .unwrap()
        .aggregate(AggregateFunction::Sum, "signed", "sum_signed")
        .unwrap()
        .aggregate(AggregateFunction::Sum, "unsigned", "sum_unsigned")
        .unwrap()
        .aggregate(AggregateFunction::Sum, "float", "sum_float")
        .unwrap()
        .aggregate(AggregateFunction::Min, "signed", "min_signed")
        .unwrap()
        .aggregate(AggregateFunction::Max, "float", "max_float")
        .unwrap()
        .aggregate(AggregateFunction::Avg, "float", "avg_float")
        .unwrap()
        .aggregate(AggregateFunction::Avg, "small", "avg_small")
        .unwrap()
}

fn assert_same_update(actual: &InputBatchUpdate, reference: &InputBatchUpdate) {
    assert_eq!(
        format!("{:?}", actual.metrics),
        format!("{:?}", reference.metrics),
        "late metrics diverged"
    );
    assert_eq!(
        actual.usage.rows, reference.usage.rows,
        "logical row charge diverged"
    );
    assert_eq!(
        actual.usage.bytes, reference.usage.bytes,
        "logical byte charge diverged"
    );
    let mut actual_keys = actual.accumulators.keys().collect::<Vec<_>>();
    let mut reference_keys = reference.accumulators.keys().collect::<Vec<_>>();
    actual_keys.sort();
    reference_keys.sort();
    assert_eq!(actual_keys, reference_keys, "key sets diverged");
    for key in actual_keys {
        let left = &actual.accumulators[key];
        let right = &reference.accumulators[key];
        assert_eq!(
            format!("{:?}", left.group_values),
            format!("{:?}", right.group_values),
            "group values diverged at {key:?}"
        );
        assert_eq!(
            state(left),
            state(right),
            "accumulator bits diverged at {key:?}"
        );
    }
}

/// The columnar two-pass fast path must reproduce the row-wise reference
/// kernel bit for bit: same keys, same accumulator states (including
/// compensated float bits), same logical usage.
#[test]
fn columnar_fast_path_matches_row_reference_bitwise() {
    let record = columnar_equivalence_record();
    let spec = columnar_equivalence_spec();
    let job = crate::StreamJobContext::new(
        1,
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let mut fast = WindowAggregateOperator::new("window", record.schema(), spec.clone()).unwrap();
    let mut rowwise = WindowAggregateOperator::new("window", record.schema(), spec).unwrap();
    rowwise.force_rowwise = true;
    let batch = Batch::table(
        vec![
            record.slice(0, 300),
            record.slice(300, record.num_rows() - 300),
        ],
        crate::BatchMetadata::default(),
    )
    .unwrap();
    // Two batches cross the live-state seeding path; the later watermark
    // admits some late rows on the second pass.
    let first = StreamOperatorContext::new(&job, "window", None);
    let later = StreamOperatorContext::new(&job, "window", Some(EventTime::from_micros(21)));
    for context in [&first, &later] {
        let fast_update = fast.prepare_input_batch(&batch, context).unwrap();
        let reference_update = rowwise.prepare_input_batch(&batch, context).unwrap();
        assert_same_update(&fast_update, &reference_update);
        let metrics = fast_update.metrics;
        fast.install_input_update(fast_update, metrics, false, context);
        rowwise.install_input_update(reference_update, metrics, false, context);
    }
}
