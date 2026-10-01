use super::*;

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
            let mut scratch = BatchScratch::new(WindowStateUsage { rows: 0, bytes: 0 });
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
