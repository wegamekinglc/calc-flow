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
