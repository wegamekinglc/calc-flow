use super::*;

thread_local! {
    pub(super) static GROUP_ENCODINGS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn spec(names: &[&str]) -> WindowSpec {
    WindowSpec::tumbling("time", Duration::from_micros(10))
        .unwrap()
        .group_by(names.iter().copied())
        .unwrap()
        .aggregate(AggregateFunction::Sum, "value", "total")
        .unwrap()
}

fn record(groups: Vec<ArrayRef>) -> RecordBatch {
    let rows = groups[0].len();
    let mut fields = vec![Field::new(
        "time",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )];
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(TimestampMicrosecondArray::from(vec![0; rows]))];
    for (index, group) in groups.into_iter().enumerate() {
        fields.push(Field::new(
            format!("key{index}"),
            group.data_type().clone(),
            true,
        ));
        arrays.push(group);
    }
    fields.push(Field::new("value", DataType::Int64, false));
    arrays.push(Arc::new(Int64Array::from(vec![1; rows])));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

fn validate_update(record: &RecordBatch, spec: WindowSpec) -> (usize, usize) {
    let operator = WindowAggregateOperator::new("window", record.schema(), spec.clone()).unwrap();
    let mut expected = BTreeMap::<Vec<u8>, i128>::new();
    for row in 0..record.num_rows() {
        let mut key = Vec::new();
        for column in &operator.compiled.group_columns {
            let scalar = scalar_at(
                record.column(column.index).as_ref(),
                &column.data_type,
                row,
                "window",
            )
            .unwrap();
            encode_group_scalar(&mut key, &column.data_type, scalar.as_ref()).unwrap();
        }
        *expected.entry(key).or_default() += 1;
    }
    let job = crate::StreamJobContext::new(
        1,
        "fixed-window-keys",
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "window", None);
    let batch = Batch::table(vec![record.clone()], crate::BatchMetadata::default()).unwrap();
    GROUP_ENCODINGS.with(|calls| calls.set(0));
    let update = operator.prepare_input_batch(&batch, &context).unwrap();
    let calls = GROUP_ENCODINGS.with(std::cell::Cell::get);
    assert_eq!(update.accumulators.len(), expected.len());
    assert_eq!(update.usage.rows, expected.len() as u64);
    for (key, entry) in &update.accumulators {
        assert_eq!(key.start.as_micros(), 0);
        assert_eq!(key.end.as_micros(), 10);
        assert!(matches!(entry.aggregates.as_slice(),
            [AccumulatorValue::SignedSum(Some(total))] if *total == expected[key.stable_group_key.as_ref()]));
    }
    assert_eq!(
        update.usage.bytes,
        window_state_usage(&update.accumulators).bytes
    );
    (calls, expected.len())
}

fn integer_arrays() -> Vec<ArrayRef> {
    const ROWS: usize = 8192;
    macro_rules! signed {
        ($array:ident, $value:ty) => {
            Arc::new($array::from(
                (0..ROWS)
                    .map(|row| match row % 4 {
                        0 => None,
                        1 => Some(<$value>::MIN),
                        2 => Some(0),
                        _ => Some(<$value>::MAX),
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef
        };
    }
    macro_rules! unsigned {
        ($array:ident, $value:ty) => {
            Arc::new($array::from(
                (0..ROWS)
                    .map(|row| match row % 4 {
                        0 => None,
                        1 => Some(0),
                        2 => Some(1),
                        _ => Some(<$value>::MAX),
                    })
                    .collect::<Vec<_>>(),
            )) as ArrayRef
        };
    }
    vec![
        signed!(Int8Array, i8),
        signed!(Int16Array, i16),
        signed!(Int32Array, i32),
        signed!(Int64Array, i64),
        unsigned!(UInt8Array, u8),
        unsigned!(UInt16Array, u16),
        unsigned!(UInt32Array, u32),
        unsigned!(UInt64Array, u64),
    ]
}

#[test]
fn interned_integer_groups_encode_only_distinct_keys() {
    let counts = integer_arrays()
        .into_iter()
        .map(|array| {
            let kind = array.data_type().clone();
            let (calls, groups) = validate_update(&record(vec![array]), spec(&["key0"]));
            (kind, calls, groups)
        })
        .collect::<Vec<_>>();
    assert!(
        counts.iter().all(|(_, calls, groups)| calls == groups),
        "{counts:?}"
    );
}

#[test]
fn interned_composite_integer_groups_preserve_nulls_and_full_width_values() {
    let pairs = [
        (None, None),
        (None, Some(0)),
        (Some(0), None),
        (Some(0), Some(0)),
        (Some(i64::MIN), Some(u64::MAX)),
        (Some(i64::MAX), Some(0)),
    ];
    let first = Arc::new(Int64Array::from(
        (0..1536)
            .map(|row| pairs[row % pairs.len()].0)
            .collect::<Vec<_>>(),
    ));
    let second = Arc::new(UInt64Array::from(
        (0..1536)
            .map(|row| pairs[row % pairs.len()].1)
            .collect::<Vec<_>>(),
    ));
    let (calls, groups) = validate_update(&record(vec![first, second]), spec(&["key0", "key1"]));
    assert_eq!(groups, pairs.len());
    assert_eq!(calls, groups);
}

#[tokio::test]
async fn interned_integer_groups_cross_records_restore_and_continuation() {
    let pairs = [
        (None, None),
        (None, Some(0)),
        (Some(0), None),
        (Some(0), Some(0)),
        (Some(i64::MIN), Some(u64::MAX)),
        (Some(i64::MAX), Some(0)),
    ];
    let make_record = |reverse: bool| {
        let indices = (0..12).map(|index| {
            if reverse {
                pairs.len() - 1 - index % pairs.len()
            } else {
                index % pairs.len()
            }
        });
        let selected = indices.map(|index| pairs[index]).collect::<Vec<_>>();
        record(vec![
            Arc::new(Int64Array::from(
                selected.iter().map(|pair| pair.0).collect::<Vec<_>>(),
            )),
            Arc::new(UInt64Array::from(
                selected.iter().map(|pair| pair.1).collect::<Vec<_>>(),
            )),
        ])
    };
    let first = make_record(false);
    let second = make_record(true);
    let input = Batch::table(
        vec![first.clone(), second.clone()],
        crate::BatchMetadata::default(),
    )
    .unwrap();
    let mut operator =
        WindowAggregateOperator::new("window", first.schema(), spec(&["key0", "key1"])).unwrap();
    let job = crate::StreamJobContext::new(
        1,
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "window", None);
    let mut output = crate::EdgeCollector::new(operator.output_ports().to_vec());
    GROUP_ENCODINGS.with(|calls| calls.set(0));
    operator
        .process_data("input", input.clone(), &context, &mut output)
        .await
        .unwrap();
    let calls = GROUP_ENCODINGS.with(std::cell::Cell::get);
    assert_eq!(
        input.table_payload().unwrap().batches(),
        &[first.clone(), second.clone()]
    );
    let snapshot = operator.checkpoint(crate::Epoch::new(1).unwrap()).unwrap();
    let mut restored =
        WindowAggregateOperator::new("window", first.schema(), spec(&["key0", "key1"])).unwrap();
    restored.restore(&snapshot).unwrap();
    let mut resumed = crate::EdgeCollector::new(restored.output_ports().to_vec());
    let continuation = Batch::table(vec![second], crate::BatchMetadata::default()).unwrap();
    operator
        .process_data("input", continuation.clone(), &context, &mut output)
        .await
        .unwrap();
    restored
        .process_data("input", continuation, &context, &mut resumed)
        .await
        .unwrap();
    assert_eq!(
        operator.state.accumulator_bytes,
        restored.state.accumulator_bytes
    );
    operator.on_end(&context, &mut output).await.unwrap();
    restored.on_end(&context, &mut resumed).await.unwrap();
    let output = output.drain("output");
    let resumed = resumed.drain("output");
    assert_eq!(output.len(), 1);
    assert_eq!(resumed.len(), 1);
    let output = output[0].as_data().unwrap();
    let resumed = resumed[0].as_data().unwrap();
    assert_eq!(output.metadata(), resumed.metadata());
    assert_eq!(
        output.table_payload().unwrap().batches(),
        resumed.table_payload().unwrap().batches()
    );
    let output = &output.table_payload().unwrap().batches()[0];
    assert_eq!(output.num_rows(), pairs.len());
    let totals = output
        .column_by_name("total")
        .unwrap()
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(totals.values().as_ref(), &[6; 6]);
    assert_eq!(calls, pairs.len());
}
