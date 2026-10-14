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

fn equivalence_job() -> crate::StreamJobContext {
    crate::StreamJobContext::new(
        1,
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    )
}

/// The default columnar operator and its forced row-wise reference twin.
fn operator_pair(
    schema: SchemaRef,
    spec: WindowSpec,
) -> (WindowAggregateOperator, WindowAggregateOperator) {
    let fast = WindowAggregateOperator::new("window", Arc::clone(&schema), spec.clone()).unwrap();
    let mut rowwise = WindowAggregateOperator::new("window", schema, spec).unwrap();
    rowwise.force_rowwise = true;
    (fast, rowwise)
}

/// Feeds one two-record batch through both kernels twice: the second pass
/// crosses the live-state seeding path under a watermark that makes some
/// rows late.
fn assert_columnar_matches_reference(
    record: &RecordBatch,
    spec: WindowSpec,
    split: usize,
    watermark: i64,
) {
    let job = equivalence_job();
    let (mut fast, mut rowwise) = operator_pair(record.schema(), spec);
    let batch = Batch::table(
        vec![
            record.slice(0, split),
            record.slice(split, record.num_rows() - split),
        ],
        crate::BatchMetadata::default(),
    )
    .unwrap();
    let first = StreamOperatorContext::new(&job, "window", None);
    let later = StreamOperatorContext::new(&job, "window", Some(EventTime::from_micros(watermark)));
    for context in [&first, &later] {
        let fast_update = fast.prepare_input_batch(&batch, context).unwrap();
        let reference_update = rowwise.prepare_input_batch(&batch, context).unwrap();
        assert_same_update(&fast_update, &reference_update);
        let metrics = fast_update.metrics;
        fast.install_input_update(fast_update, metrics, false, context);
        rowwise.install_input_update(reference_update, metrics, false, context);
    }
}

/// The columnar two-pass fast path must reproduce the row-wise reference
/// kernel bit for bit: same keys, same accumulator states (including
/// compensated float bits), same logical usage.
#[test]
fn columnar_fast_path_matches_row_reference_bitwise() {
    assert_columnar_matches_reference(
        &columnar_equivalence_record(),
        columnar_equivalence_spec(),
        300,
        21,
    );
}

fn xorshift_rolls(count: usize) -> Vec<u64> {
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    (0..count)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        })
        .collect()
}

/// One nullable column drawn from independent bits of each roll.
fn nullable<T>(
    rolls: &[u64],
    shift: u32,
    null_nth: u64,
    value: impl Fn(u64) -> T,
) -> Vec<Option<T>> {
    rolls
        .iter()
        .map(|&roll| pick(roll >> shift, null_nth, &value))
        .collect()
}

fn offset(modulus: u64, shift: i64) -> impl Fn(u64) -> i64 {
    move |roll| i64::try_from(roll % modulus).unwrap() - shift
}

/// Every non-string input type the columnar kernels dispatch on, plus
/// integer and string group columns, with nulls in every column except
/// `dense`, which takes the null-free loops.
fn wide_equivalence_record() -> RecordBatch {
    let rolls = xorshift_rolls(600);
    let small_floats = [1.5_f32, -2.25, 3.0e7, -3.0e7, f32::NAN, f32::INFINITY, -0.0];
    let fields = [
        ("time", DataType::Timestamp(TimeUnit::Microsecond, None)),
        ("g_int", DataType::Int32),
        ("g_str", DataType::Utf8),
        ("i8", DataType::Int8),
        ("i32", DataType::Int32),
        ("u16", DataType::UInt16),
        ("u32", DataType::UInt32),
        ("f32", DataType::Float32),
        ("d32", DataType::Date32),
        ("ts", DataType::Timestamp(TimeUnit::Microsecond, None)),
        ("flag", DataType::Boolean),
        ("s", DataType::Utf8),
        ("dense", DataType::Int64),
    ];
    let schema = Schema::new(
        fields
            .map(|(name, data_type)| Field::new(name, data_type, name != "dense"))
            .to_vec(),
    );
    let columns: Vec<ArrayRef> = vec![
        Arc::new(TimestampMicrosecondArray::from(nullable(
            &rolls,
            3,
            13,
            offset(70, 10),
        ))),
        Arc::new(Int32Array::from(nullable(&rolls, 5, 9, |r| {
            i32::try_from(r % 4).unwrap()
        }))),
        Arc::new(StringArray::from(nullable(&rolls, 7, 9, |r| {
            ["x", "y", "z"][usize::try_from(r % 3).unwrap()]
        }))),
        Arc::new(Int8Array::from(nullable(&rolls, 9, 5, |r| {
            i8::try_from(offset(256, 128)(r)).unwrap()
        }))),
        Arc::new(Int32Array::from(nullable(&rolls, 11, 6, |r| {
            i32::try_from(offset(100_000, 50_000)(r)).unwrap()
        }))),
        Arc::new(UInt16Array::from(nullable(&rolls, 13, 7, |r| {
            u16::try_from(r % 60_000).unwrap()
        }))),
        Arc::new(UInt32Array::from(nullable(&rolls, 15, 4, |r| {
            u32::try_from(r % 4_000_000_000).unwrap()
        }))),
        Arc::new(Float32Array::from(nullable(&rolls, 17, 5, |r| {
            small_floats[usize::try_from(r % 7).unwrap()]
        }))),
        Arc::new(Date32Array::from(nullable(&rolls, 19, 5, |r| {
            i32::try_from(offset(1000, 500)(r)).unwrap()
        }))),
        Arc::new(TimestampMicrosecondArray::from(nullable(
            &rolls,
            21,
            5,
            offset(1000, 500),
        ))),
        Arc::new(BooleanArray::from(nullable(&rolls, 23, 5, |r| r % 2 == 0))),
        Arc::new(StringArray::from(nullable(&rolls, 25, 5, |r| {
            ["p", "q"][usize::try_from(r % 2).unwrap()]
        }))),
        Arc::new(Int64Array::from(
            rolls
                .iter()
                .map(|&r| offset(2001, 1000)(r >> 27))
                .collect::<Vec<_>>(),
        )),
    ];
    RecordBatch::try_new(Arc::new(schema), columns).unwrap()
}

fn wide_equivalence_spec(groups: &[&str]) -> WindowSpec {
    let spec =
        WindowSpec::hopping("time", Duration::from_micros(20), Duration::from_micros(5)).unwrap();
    let spec = if groups.is_empty() {
        spec
    } else {
        spec.group_by(groups.iter().copied()).unwrap()
    };
    [
        (AggregateFunction::Sum, "i8"),
        (AggregateFunction::Sum, "u32"),
        (AggregateFunction::Sum, "f32"),
        (AggregateFunction::Avg, "i32"),
        (AggregateFunction::Avg, "u16"),
        (AggregateFunction::Avg, "i8"),
        (AggregateFunction::Min, "d32"),
        (AggregateFunction::Max, "ts"),
        (AggregateFunction::Min, "flag"),
        (AggregateFunction::Max, "u16"),
        (AggregateFunction::Count, "s"),
        (AggregateFunction::Min, "f32"),
        (AggregateFunction::Sum, "dense"),
        (AggregateFunction::Max, "dense"),
        (AggregateFunction::Avg, "dense"),
        (AggregateFunction::Count, "dense"),
    ]
    .into_iter()
    .enumerate()
    .fold(spec, |spec, (ordinal, (function, column))| {
        spec.aggregate(function, column, &format!("result_{ordinal}"))
            .unwrap()
    })
}

/// Small integer, unsigned average, temporal and boolean extreme kernels
/// under no-group, integer-group, and composite-group keys, across an
/// overlap-4 hopping geometry.
#[test]
fn columnar_fast_path_matches_row_reference_across_types_and_keys() {
    let record = wide_equivalence_record();
    for groups in [&[][..], &["g_int"][..], &["g_int", "g_str"][..]] {
        assert_columnar_matches_reference(&record, wide_equivalence_spec(groups), 250, 15);
    }
}

fn two_sum_batch(a: [i64; 3], b: [i64; 3], times: [i64; 3]) -> Batch {
    let schema = Schema::new(vec![
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
    ]);
    let record = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(TimestampMicrosecondArray::from(times.to_vec())),
            Arc::new(Int64Array::from(a.to_vec())),
            Arc::new(Int64Array::from(b.to_vec())),
        ],
    )
    .unwrap();
    Batch::table(vec![record], crate::BatchMetadata::default()).unwrap()
}

/// The columnar passes visit rows aggregate by aggregate, yet a failing batch
/// must report the row-wise reference error: the first failing row, and
/// within it the first failing aggregate, ahead of later budget checks.
#[test]
fn columnar_failures_report_the_row_reference_error() {
    let job = equivalence_job();
    let context = StreamOperatorContext::new(&job, "window", None);
    let spec = WindowSpec::tumbling("time", Duration::from_micros(10))
        .unwrap()
        .aggregate(AggregateFunction::Sum, "a", "sum_a")
        .unwrap()
        .aggregate(AggregateFunction::Sum, "b", "sum_b")
        .unwrap();
    // Row 1 overflows aggregates[1] before row 2 overflows aggregates[0];
    // row 1 overflows aggregates[0] before row 2 opens a key past a one-row
    // budget.
    for (batch, max_rows) in [
        (
            two_sum_batch([0, i64::MAX, 1], [i64::MAX, 1, 0], [0, 0, 0]),
            8,
        ),
        (two_sum_batch([i64::MAX, 1, 0], [0, 0, 0], [0, 0, 20]), 1),
    ] {
        let schema = batch.table_payload().unwrap().batches()[0].schema();
        let (mut fast, mut rowwise) = operator_pair(schema, spec.clone());
        for operator in [&mut fast, &mut rowwise] {
            operator
                .set_state_budget(StateBudget::new(max_rows, 1 << 20).unwrap())
                .unwrap();
        }
        let expected = rowwise
            .prepare_input_batch(&batch, &context)
            .err()
            .expect("the reference rejects the batch");
        let actual = fast
            .prepare_input_batch(&batch, &context)
            .err()
            .expect("the columnar path rejects the batch");
        assert_eq!(actual.to_string(), expected.to_string());
    }
}
