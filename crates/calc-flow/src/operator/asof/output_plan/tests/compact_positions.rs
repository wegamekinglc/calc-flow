use super::*;

fn rows_payload(rows: usize, side: u8) -> RowPayload {
    let values = Int64Array::from((0..i64::try_from(rows).unwrap()).collect::<Vec<_>>());
    let record = RecordBatch::try_from_iter(vec![("value", Arc::new(values) as ArrayRef)]).unwrap();
    RowPayload {
        batch: Arc::new(PayloadBatch {
            key: (side, 0),
            record: Arc::new(record),
            body_bytes: 0,
            encoded_charge_bytes: 0,
            encoded: OnceLock::new(),
        }),
        row: 0,
    }
}

#[test]
fn test_single_source_right_positions_avoid_tuple_sized_peak_allocations() {
    let rows = 65_536;
    let left = rows_payload(rows, 0);
    let right = rows_payload(rows, 1);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(32 << 20));
    let mut credit = MemoryConsumer::new("compact-right-positions").register(&pool);
    let measured = allocation_counter::measure(|| {
        let mut builder = OutputPlanBuilder::new(rows, None, &mut credit, "asof").unwrap();
        let left_source = builder
            .left_source(left.view(), &mut credit, "asof")
            .unwrap();
        let right_source = builder
            .right_source(right.view(), &mut credit, "asof")
            .unwrap();
        for row in 0..rows {
            builder.left_position(left_source, row, "asof").unwrap();
            builder
                .push_right_source(Some((right_source, row)), "asof")
                .unwrap();
        }
        let plan = builder
            .finish(&right.batch.record.schema(), &mut credit, "asof")
            .unwrap();
        assert_eq!(plan.len, rows);
        assert_eq!(plan.matched, rows as u64);
    });
    assert!(
        measured.bytes_max < rows as u64 * 14 + 16_384,
        "one right owner needs compact indices; tuple-sized allocation peak: {measured:?}"
    );
    drop(credit);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_single_source_fixed_right_positions_skip_range_bookkeeping() {
    let rows = 2;
    let left = rows_payload(rows, 0);
    let right = rows_payload(rows, 1);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut credit = MemoryConsumer::new("compact-fixed-right-positions").register(&pool);
    let mut builder = OutputPlanBuilder::new(rows, None, &mut credit, "asof").unwrap();
    let left_source = builder
        .left_source(left.view(), &mut credit, "asof")
        .unwrap();
    let right_source = builder
        .right_source(right.view(), &mut credit, "asof")
        .unwrap();
    for row in 0..rows {
        builder.left_position(left_source, row, "asof").unwrap();
        builder
            .push_right_source(Some((right_source, row)), "asof")
            .unwrap();
    }
    assert!(builder.right.run.is_none());
    let plan = builder
        .finish(&right.batch.record.schema(), &mut credit, "asof")
        .unwrap();
    assert_eq!(plan.raw_bytes, 32);
    drop((plan, credit));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_late_second_right_source_uses_funded_tuple_promotion() {
    let rows = 4_096;
    let left = rows_payload(rows, 0);
    let right = rows_payload(rows, 1);
    let second = rows_payload(1, 2);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(32 << 20));
    let mut credit = MemoryConsumer::new("late-second-right-source").register(&pool);
    let mut builder = OutputPlanBuilder::new(rows, None, &mut credit, "asof").unwrap();
    let left_source = builder
        .left_source(left.view(), &mut credit, "asof")
        .unwrap();
    let first_source = builder
        .right_source(right.view(), &mut credit, "asof")
        .unwrap();
    let second_source = builder
        .right_source(second.view(), &mut credit, "asof")
        .unwrap();
    builder.left_position(left_source, 0, "asof").unwrap();
    builder.push_right_source(None, "asof").unwrap();
    for row in 1..rows - 1 {
        builder.left_position(left_source, row, "asof").unwrap();
        builder
            .push_right_source(Some((first_source, row - 1)), "asof")
            .unwrap();
    }
    builder
        .left_position(left_source, rows - 1, "asof")
        .unwrap();
    let promoted = allocation_counter::measure(|| {
        builder
            .push_right_source(Some((second_source, 0)), "asof")
            .unwrap();
    });
    assert!(
        promoted.bytes_max <= (rows * size_of::<(usize, usize)>() + 4_096) as u64,
        "one-time tuple promotion must fit its existing metadata credit: {promoted:?}"
    );
    let plan = builder
        .finish(&right.batch.record.schema(), &mut credit, "asof")
        .unwrap();
    let expected = std::iter::once((0, 0))
        .chain((0..rows - 2).map(|row| (first_source + 1, row)))
        .chain(std::iter::once((second_source + 1, 0)))
        .collect::<Vec<_>>();
    assert_eq!(plan.right.positions, expected);
    assert_eq!(plan.len, rows);
    assert_eq!(plan.matched, rows as u64 - 1);
    drop((plan, credit));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_mixed_source_fixed_positions_preserve_legacy_raw_accounting() {
    fn raw_bytes(compact: bool) -> u64 {
        let left = rows_payload(3, 0);
        let first = rows_payload(3, 1);
        let second = rows_payload(1, 2);
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let mut credit = MemoryConsumer::new("mixed-source-fixed-positions").register(&pool);
        let mut builder =
            OutputPlanBuilder::new_for_owner(3, None, &mut credit, "asof", compact).unwrap();
        let left_source = builder
            .left_source(left.view(), &mut credit, "asof")
            .unwrap();
        let first_source = builder
            .right_source(first.view(), &mut credit, "asof")
            .unwrap();
        let second_source = builder
            .right_source(second.view(), &mut credit, "asof")
            .unwrap();
        for row in 0..2 {
            builder.left_position(left_source, row, "asof").unwrap();
            builder
                .push_right_source(Some((first_source, row)), "asof")
                .unwrap();
        }
        builder.left_position(left_source, 2, "asof").unwrap();
        builder
            .push_right_source(Some((second_source, 0)), "asof")
            .unwrap();
        let plan = builder
            .finish(&first.batch.record.schema(), &mut credit, "asof")
            .unwrap();
        let bytes = plan.raw_bytes;
        drop((plan, credit));
        assert_eq!(pool.reserved(), 0);
        bytes
    }

    assert_eq!(raw_bytes(true), raw_bytes(false));
}

fn differently_nullable_source_credit(
    compact: bool,
    first_has_nulls: bool,
    promote: bool,
) -> (u64, usize) {
    fn right_payload(has_nulls: bool, ordinal: u64) -> RowPayload {
        let values = if has_nulls {
            Int64Array::from(vec![None, Some(1)])
        } else {
            Int64Array::from(vec![0, 1])
        };
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));
        RowPayload {
            batch: Arc::new(PayloadBatch {
                key: (1, ordinal),
                record: Arc::new(RecordBatch::try_new(schema, vec![Arc::new(values)]).unwrap()),
                body_bytes: 0,
                encoded_charge_bytes: 0,
                encoded: OnceLock::new(),
            }),
            row: 0,
        }
    }

    let left = rows_payload(4, 0);
    let first = right_payload(first_has_nulls, 0);
    let second = right_payload(!first_has_nulls, 1);
    assert_eq!(first.batch.record.schema(), second.batch.record.schema());
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let mut credit = MemoryConsumer::new("source-validity-credit").register(&pool);
    let mut builder =
        OutputPlanBuilder::new_for_owner(4, None, &mut credit, "asof", compact).unwrap();
    let left_source = builder
        .left_source(left.view(), &mut credit, "asof")
        .unwrap();
    let first_source = builder
        .right_source(first.view(), &mut credit, "asof")
        .unwrap();
    builder.left_position(left_source, 0, "asof").unwrap();
    builder.push_right_source(None, "asof").unwrap();
    for row in 0..2 {
        builder.left_position(left_source, row + 1, "asof").unwrap();
        builder
            .push_right_source(Some((first_source, row)), "asof")
            .unwrap();
    }
    let second_source = builder
        .right_source(second.view(), &mut credit, "asof")
        .unwrap();
    if promote {
        builder.left_position(left_source, 3, "asof").unwrap();
        builder
            .push_right_source(Some((second_source, 0)), "asof")
            .unwrap();
    }
    let plan = builder
        .finish(&first.batch.record.schema(), &mut credit, "asof")
        .unwrap();
    assert_eq!(plan.matched, 2 + u64::from(promote));
    assert_eq!(plan.len, 3 + usize::from(promote));
    let result = (plan.raw_bytes, credit.size());
    drop((plan, credit));
    assert_eq!(pool.reserved(), 0);
    result
}

#[test]
fn test_fixed_right_promotion_charges_original_source_validity() {
    for first_has_nulls in [true, false] {
        assert_eq!(
            differently_nullable_source_credit(true, first_has_nulls, true),
            differently_nullable_source_credit(false, first_has_nulls, true),
            "compact prefix must charge its original source; first_has_nulls={first_has_nulls}"
        );
    }
}

#[test]
fn test_fixed_right_finish_ignores_unused_registered_source() {
    for first_has_nulls in [true, false] {
        assert_eq!(
            differently_nullable_source_credit(true, first_has_nulls, false),
            differently_nullable_source_credit(false, first_has_nulls, false),
            "unused registered source must not change output credit; first_has_nulls={first_has_nulls}"
        );
    }
}

fn pending_promotion(
    rows: usize,
) -> (
    OutputPlanBuilder<'static>,
    MemoryReservation,
    Arc<dyn MemoryPool>,
    usize,
) {
    let left = rows_payload(rows, 0);
    let first = rows_payload(rows, 1);
    let second = rows_payload(1, 2);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(32 << 20));
    let mut credit = MemoryConsumer::new("pending-position-promotion").register(&pool);
    let mut builder = OutputPlanBuilder::new(rows, None, &mut credit, "asof").unwrap();
    let left_source = builder
        .left_source(left.view(), &mut credit, "asof")
        .unwrap();
    let first_source = builder
        .right_source(first.view(), &mut credit, "asof")
        .unwrap();
    let second_source = builder
        .right_source(second.view(), &mut credit, "asof")
        .unwrap();
    for row in 0..rows - 1 {
        builder.left_position(left_source, row, "asof").unwrap();
        builder
            .push_right_source(Some((first_source, row)), "asof")
            .unwrap();
    }
    builder
        .left_position(left_source, rows - 1, "asof")
        .unwrap();
    (builder, credit, pool, second_source)
}

#[test]
fn cancelled_promotion_allocates_no_tuple_buffer_and_refunds_plan_credit() {
    let (builder, credit, pool, second_source) = pending_promotion(4_096);
    let cancellation = crate::CancellationToken::new();
    let job = crate::StreamJobContext::new(
        924,
        "cancelled-position-promotion",
        crate::JsonMap::new(),
        None,
        cancellation.clone(),
    );
    let mut builder = builder.with_stop(GatherStop::from_job(&job));
    cancellation.cancel();
    let mut result = None;
    let measured = allocation_counter::measure(|| {
        result = Some(builder.push_right_source(Some((second_source, 0)), "asof"));
    });
    assert!(matches!(
        result.unwrap(),
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert!(
        measured.bytes_total < 4_096,
        "cancelled promotion allocated a tuple buffer: {measured:?}"
    );
    drop((builder, credit));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn promotion_needs_no_additional_workspace_when_the_pool_is_full() {
    let (mut builder, credit, pool, second_source) = pending_promotion(4_096);
    let reserved = credit.size();
    let filler = MemoryConsumer::new("promotion-pool-filler").register(&pool);
    filler.try_grow((32 << 20) - pool.reserved()).unwrap();
    builder
        .push_right_source(Some((second_source, 0)), "asof")
        .unwrap();
    assert_eq!(credit.size(), reserved);
    assert_eq!(pool.reserved(), 32 << 20);
    drop((builder, credit, filler));
    assert_eq!(pool.reserved(), 0);
}
