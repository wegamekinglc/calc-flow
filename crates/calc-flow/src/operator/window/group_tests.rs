use super::*;

thread_local! {
    pub(super) static SLOT_LOOKUPS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static CACHE_KEY_COMPARISONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn prepare_group_update(record: RecordBatch, spec: WindowSpec) -> InputBatchUpdate {
    let operator = WindowAggregateOperator::new("window", record.schema(), spec).unwrap();
    let job = crate::StreamJobContext::new(
        1,
        "window-slot-cache",
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "window", None);
    let batch = Batch::table(vec![record], crate::BatchMetadata::default()).unwrap();
    operator.prepare_input_batch(&batch, &context).unwrap()
}

#[test]
fn repeated_tumbling_groups_reuse_accumulator_slots() {
    let keys = [None, Some(""), Some("same"), Some("a\0b")];
    for large in [false, true] {
        for composite in [false, true] {
            let record = string_record(
                large,
                vec![0; 8192],
                (0..8192).map(|row| keys[row % keys.len()]).collect(),
            );
            SLOT_LOOKUPS.with(|calls| calls.set(0));
            let update = prepare_group_update(record, spec(composite));
            assert_eq!(update.accumulators.len(), keys.len());
            assert_eq!(update.usage.rows, 4);
            for entry in update.accumulators.values() {
                assert!(matches!(
                    entry.aggregates.as_slice(),
                    [AccumulatorValue::SignedSum(Some(2048))]
                ));
            }
            assert_eq!(SLOT_LOOKUPS.with(std::cell::Cell::get), 4);
        }
    }
}

#[test]
fn repeated_two_window_groups_reuse_all_accumulator_slots() {
    assert_hopping_slot_reuse(2);
}

#[test]
fn repeated_three_window_groups_reuse_all_accumulator_slots() {
    assert_hopping_slot_reuse(3);
}

fn assert_hopping_slot_reuse(overlap: usize) {
    let keys = [None, Some(""), Some("same"), Some("a\0b")];
    for large in [false, true] {
        for composite in [false, true] {
            let record = string_record(
                large,
                vec![0; 8192],
                (0..8192).map(|row| keys[row % keys.len()]).collect(),
            );
            SLOT_LOOKUPS.with(|calls| calls.set(0));
            let update = prepare_group_update(record, hopping_spec(overlap, composite));
            assert_eq!(update.accumulators.len(), overlap * keys.len());
            assert_eq!(update.usage.rows, (overlap * keys.len()) as u64);
            for entry in update.accumulators.values() {
                assert!(matches!(
                    entry.aggregates.as_slice(),
                    [AccumulatorValue::SignedSum(Some(2048))]
                ));
            }
            assert_eq!(
                SLOT_LOOKUPS.with(std::cell::Cell::get),
                overlap * keys.len()
            );
        }
    }
}

#[test]
fn high_overlap_preserves_all_windows_and_map_lookups() {
    let record = string_record(false, vec![0; 128], vec![Some("same"); 128]);
    CACHE_KEY_COMPARISONS.with(|calls| calls.set(0));
    prepare_group_update(record.clone(), hopping_spec(2, false));
    assert!(CACHE_KEY_COMPARISONS.with(std::cell::Cell::get) > 0);
    SLOT_LOOKUPS.with(|calls| calls.set(0));
    CACHE_KEY_COMPARISONS.with(|calls| calls.set(0));
    let update = prepare_group_update(record, hopping_spec(8, false));
    assert_eq!(update.accumulators.len(), 8);
    assert_eq!(update.usage.rows, 8);
    let starts = update
        .accumulators
        .keys()
        .map(|key| key.start.as_micros())
        .collect::<Vec<_>>();
    assert_eq!(starts, [-70, -60, -50, -40, -30, -20, -10, 0]);
    for entry in update.accumulators.values() {
        assert!(matches!(
            entry.aggregates.as_slice(),
            [AccumulatorValue::SignedSum(Some(128))]
        ));
    }
    assert_eq!(SLOT_LOOKUPS.with(std::cell::Cell::get), 1024);
    assert_eq!(CACHE_KEY_COMPARISONS.with(std::cell::Cell::get), 0);
}

#[test]
fn hopping_slot_cache_keeps_colliding_groups_separate() {
    let groups = (0..65)
        .map(|index| format!("group{index}"))
        .collect::<Vec<_>>();
    for overlap in [2, 3, 4] {
        let record = string_record(
            false,
            vec![0; 130],
            groups
                .iter()
                .cycle()
                .take(130)
                .map(|key| Some(key.as_str()))
                .collect(),
        );
        let update = prepare_group_update(record, hopping_spec(overlap, false));
        assert_eq!(update.accumulators.len(), overlap * groups.len());
        assert_eq!(update.usage.rows, (overlap * groups.len()) as u64);
        for entry in update.accumulators.values() {
            assert!(matches!(
                entry.aggregates.as_slice(),
                [AccumulatorValue::SignedSum(Some(2))]
            ));
        }
    }
}

fn hopping_spec(overlap: usize, composite: bool) -> WindowSpec {
    let mut spec = spec(composite);
    spec.geometry = WindowGeometry::Hopping {
        size_micros: (overlap * 10) as u64,
        slide_micros: 10,
    };
    spec
}

#[test]
fn slot_cache_keeps_colliding_groups_separate() {
    let groups = (0..65)
        .map(|index| format!("group{index}"))
        .collect::<Vec<_>>();
    let record = string_record(
        false,
        vec![0; 130],
        groups
            .iter()
            .cycle()
            .take(130)
            .map(|key| Some(key.as_str()))
            .collect(),
    );
    let update = prepare_group_update(record, spec(false));
    assert_eq!(update.accumulators.len(), 65);
    assert_eq!(update.usage.rows, 65);
    for entry in update.accumulators.values() {
        assert!(matches!(
            entry.aggregates.as_slice(),
            [AccumulatorValue::SignedSum(Some(2))]
        ));
    }
}

#[test]
fn slot_cache_keeps_out_of_order_and_hopping_windows_separate() {
    let hopping = WindowSpec::hopping("time", Duration::from_micros(20), Duration::from_micros(10))
        .unwrap()
        .group_by(["group"])
        .unwrap()
        .aggregate(AggregateFunction::Sum, "value", "total")
        .unwrap();
    for (spec, expected) in [
        (spec(false), BTreeMap::from([(0, 3), (10, 2)])),
        (hopping, BTreeMap::from([(-10, 3), (0, 5), (10, 2)])),
        (
            hopping_spec(3, false),
            BTreeMap::from([(-20, 3), (-10, 5), (0, 5), (10, 2)]),
        ),
        (
            hopping_spec(4, false),
            BTreeMap::from([(-30, 3), (-20, 5), (-10, 5), (0, 5), (10, 2)]),
        ),
    ] {
        let record = string_record(false, vec![0, 10, 0, 10, 0], vec![Some("same"); 5]);
        let update = prepare_group_update(record, spec);
        let totals = update
            .accumulators
            .iter()
            .map(|(key, entry)| {
                let [AccumulatorValue::SignedSum(Some(total))] = entry.aggregates.as_slice() else {
                    panic!("expected a signed sum");
                };
                (key.start.as_micros(), *total)
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(totals, expected);
    }
}

fn string_record(large: bool, times: Vec<i64>, groups: Vec<Option<&str>>) -> RecordBatch {
    let len = times.len();
    let schema = Arc::new(Schema::new(vec![
        Field::new(
            "time",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new(
            "group",
            if large {
                DataType::LargeUtf8
            } else {
                DataType::Utf8
            },
            true,
        ),
        Field::new("partition", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let group: ArrayRef = if large {
        Arc::new(LargeStringArray::from(groups))
    } else {
        Arc::new(StringArray::from(groups))
    };
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(TimestampMicrosecondArray::from(times)),
            group,
            Arc::new(Int64Array::from(vec![7; len])),
            Arc::new(Int64Array::from(vec![1; len])),
        ],
    )
    .unwrap()
}

fn spec(composite: bool) -> WindowSpec {
    WindowSpec::tumbling("time", Duration::from_micros(10))
        .unwrap()
        .group_by(if composite {
            vec!["group", "partition"]
        } else {
            vec!["group"]
        })
        .unwrap()
        .aggregate(AggregateFunction::Sum, "value", "total")
        .unwrap()
}

#[test]
fn repeated_string_groups_do_not_allocate_owned_scalars() {
    for large in [false, true] {
        for composite in [false, true] {
            let record = string_record(large, vec![0; 64], vec![Some("key\0with escape"); 64]);
            let spec = spec(composite);
            let operator =
                WindowAggregateOperator::new("window", record.schema(), spec.clone()).unwrap();
            let columns = RecordColumns::new(&record, &spec, &operator.compiled, "window").unwrap();
            let mut scratch = BatchScratch::new(WindowStateUsage { rows: 0, bytes: 0 }, 1);
            scratch
                .intern_group(&columns, 0, "window", &spec.group_by)
                .unwrap();
            let allocation = allocation_counter::measure(|| {
                for row in 1..64 {
                    assert_eq!(
                        scratch
                            .intern_group(&columns, row, "window", &spec.group_by)
                            .unwrap(),
                        0
                    );
                }
            });
            assert_eq!(
                allocation.count_total, 0,
                "large={large}, composite={composite}: {allocation:?}"
            );
        }
    }
}

#[tokio::test]
async fn borrowed_string_groups_keep_values_across_records_windows_and_restore() {
    for large in [false, true] {
        for composite in [false, true] {
            let first = string_record(large, vec![0, 0, 0], vec![None, Some(""), Some("a\0b")]);
            let second = string_record(
                large,
                vec![10, 10, 10, 11],
                vec![Some("a\0b"), None, Some(""), Some("a\0b")],
            );
            let schema = first.schema();
            let spec = spec(composite);
            let mut operator =
                WindowAggregateOperator::new("window", schema.clone(), spec.clone()).unwrap();
            let job = crate::StreamJobContext::new(
                1,
                "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                JsonMap::new(),
                None,
                crate::CancellationToken::new(),
            );
            let context = StreamOperatorContext::new(&job, "window", None);
            let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
            operator
                .process_data(
                    "input",
                    Batch::table(vec![first, second], crate::BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
            let snapshot = operator.checkpoint(crate::Epoch::new(1).unwrap()).unwrap();
            let mut restored = WindowAggregateOperator::new("window", schema, spec).unwrap();
            restored.restore(&snapshot).unwrap();
            let mut restored_collector =
                crate::EdgeCollector::new(restored.output_ports().to_vec());
            operator.on_end(&context, &mut collector).await.unwrap();
            restored
                .on_end(&context, &mut restored_collector)
                .await
                .unwrap();
            let output = collector.drain("output");
            let resumed = restored_collector.drain("output");
            let record = &output[0]
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()[0];
            let resumed = &resumed[0]
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()[0];
            assert_eq!(record, resumed);
            assert_eq!(record.num_rows(), 6);
            let totals = record
                .column_by_name("total")
                .unwrap()
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            assert_eq!(totals.values().as_ref(), &[1, 1, 1, 1, 1, 2]);
        }
    }
}
