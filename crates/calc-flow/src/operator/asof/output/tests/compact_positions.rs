use super::*;

fn payload(key: BatchKey, values: &[i64]) -> RowPayload {
    let record = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from(values.to_vec())) as ArrayRef,
    )])
    .unwrap();
    record_payload(key, record)
}

fn record_payload(key: BatchKey, record: RecordBatch) -> RowPayload {
    RowPayload {
        batch: Arc::new(PayloadBatch {
            key,
            record: Arc::new(record),
            body_bytes: 0,
            encoded_charge_bytes: 0,
            encoded: std::sync::OnceLock::new(),
        }),
        row: 0,
    }
}

fn compact_plan(
    rows: &[(PayloadView<'_>, Option<PayloadView<'_>>)],
    schema: &SchemaRef,
) -> (OutputPlan, MemoryReservation) {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(32 << 20));
    let mut credit = MemoryConsumer::new("compact-gather-control").register(&pool);
    let mut builder = OutputPlanBuilder::new(rows.len(), None, &mut credit, "asof").unwrap();
    for &(left, right) in rows {
        builder.push(left, right, &mut credit, "asof").unwrap();
    }
    let plan = builder.finish(schema, &mut credit, "asof").unwrap();
    (plan, credit)
}

fn assert_ranges_match_legacy(plan: &OutputPlan, expected: &ArrayRef) {
    for range in [0..2, 2..4, 3..5, 4..6, 0..6] {
        let mut gather = GatherPlan::new(&plan.right, true);
        gather.positions = gather
            .positions
            .map(|positions| positions.slice(range.clone()));
        let column = gather.column(0, &DataType::Int64, range.len()).unwrap();
        let expected = expected.slice(range.start, range.len());
        assert_eq!(column.to_data(), expected.to_data(), "range {range:?}");
    }
}

#[test]
fn compact_indices_and_promoted_positions_match_legacy_output_and_row_slices() {
    let left = payload((0, 0), &[10, 11, 12]);
    let right = payload((1, 0), &[20, 21]);
    let second = payload((1, 1), &[30]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("left", DataType::Int64, false),
        Field::new("right", DataType::Int64, true),
    ]));
    let left_row = |row| PayloadView {
        batch: &left.batch,
        row,
    };
    let right_row = |row| PayloadView {
        batch: &right.batch,
        row,
    };
    for multiple in [false, true] {
        let alternate = if multiple {
            second.view()
        } else {
            right_row(0)
        };
        let rows = [
            (left_row(0), None),
            (left_row(1), Some(right_row(0))),
            (left_row(2), Some(right_row(1))),
            (left_row(0), None),
            (left_row(1), Some(alternate)),
            (left_row(2), Some(right_row(0))),
        ];
        let legacy = materialize_rows(&rows, &schema).unwrap();
        let expected = &legacy.table_payload().unwrap().batches()[0];
        let (plan, credit) = compact_plan(&rows, &right.batch.record.schema());
        let actual = materialize_plans(
            &GatherPlan::new(&plan.left, false),
            &GatherPlan::new(&plan.right, true),
            rows.len(),
            &schema,
            None,
        )
        .unwrap();
        assert_eq!(&actual.table_payload().unwrap().batches()[0], expected);
        assert_ranges_match_legacy(&plan, expected.column(1));
        drop((plan, credit));
    }
}

#[test]
fn compact_take_drops_unselected_utf8_backing_and_preserves_source_nulls() {
    let giant = "x".repeat(1 << 20);
    let right = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(StringArray::from(vec![
            Some(giant.as_str()),
            Some("a"),
            Some(""),
            None,
        ])) as ArrayRef,
    )])
    .unwrap();
    let right = record_payload((1, 0), right);
    let retained = Arc::downgrade(right.batch.record.column(0));
    let left = payload((0, 0), &[10, 11, 12]);
    let rows = (0..3)
        .map(|row| {
            (
                PayloadView {
                    batch: &left.batch,
                    row,
                },
                Some(PayloadView {
                    batch: &right.batch,
                    row: row + 1,
                }),
            )
        })
        .collect::<Vec<_>>();
    let schema = Arc::new(Schema::new(vec![
        Field::new("left", DataType::Int64, false),
        Field::new("right", DataType::Utf8, true),
    ]));
    let expected = materialize_rows(&rows, &schema).unwrap();
    let (plan, credit) = compact_plan(&rows, &right.batch.record.schema());
    let actual = materialize_plans(
        &GatherPlan::new(&plan.left, false),
        &GatherPlan::new(&plan.right, true),
        rows.len(),
        &schema,
        None,
    )
    .unwrap();
    assert_eq!(
        actual.table_payload().unwrap().batches(),
        expected.table_payload().unwrap().batches()
    );
    assert!(
        actual.table_payload().unwrap().batches()[0]
            .column(1)
            .get_array_memory_size()
            < 4_096
    );
    drop(rows);
    drop((plan, credit, right));
    assert!(
        retained.upgrade().is_none(),
        "output retained an unselected giant UTF8 parent"
    );
}

#[test]
fn promoted_right_positions_keep_one_output_allocation_peak() {
    let count = 65_536;
    let values = (0..i64::try_from(count).unwrap()).collect::<Vec<_>>();
    let left = payload((0, 0), &values);
    let first = payload((1, 0), &[-1]);
    let second = payload((1, 1), &values);
    let rows = (0..count)
        .map(|row| {
            let right = if row == 0 {
                first.view()
            } else {
                PayloadView {
                    batch: &second.batch,
                    row: row - 1,
                }
            };
            (
                PayloadView {
                    batch: &left.batch,
                    row,
                },
                Some(right),
            )
        })
        .collect::<Vec<_>>();
    let (plan, credit) = compact_plan(&rows, &second.batch.record.schema());
    let legacy = OutputRows::new(&rows);
    let expected = GatherPlan::new(&legacy.right, true)
        .column(0, &DataType::Int64, count)
        .unwrap();
    let mut actual = None;
    let measured = allocation_counter::measure(|| {
        actual = Some(
            GatherPlan::new(&plan.right, true)
                .column(0, &DataType::Int64, count)
                .unwrap(),
        );
    });
    assert_eq!(actual.unwrap().to_data(), expected.to_data());
    assert!(
        measured.bytes_max < count as u64 * 12 + 16_384,
        "multiple owners must retain one-pass output allocation: {measured:?}"
    );
    drop((plan, credit));
}
