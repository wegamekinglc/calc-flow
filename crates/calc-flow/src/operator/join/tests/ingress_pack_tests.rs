use super::*;

#[tokio::test]
async fn test_scaling_ingress_packs_masks_across_word_boundaries() {
    let operator = StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let plan = operator.ingress_plan("left").unwrap();
    for lengths in [vec![4_097], vec![0, 63, 64, 65, 1_023, 1_024, 1_025, 0]] {
        let expected = lengths
            .iter()
            .map(|rows: &usize| rows.div_ceil(1_024))
            .sum::<usize>();
        let records = lengths
            .iter()
            .map(|&rows| {
                left_batch(vec![10; rows])
                    .table_payload()
                    .unwrap()
                    .batches()[0]
                    .clone()
            })
            .collect();
        let batch = Batch::table(records, BatchMetadata::default()).unwrap();
        let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
        let producer = async {
            plan.prepare(&batch, &sender).await.unwrap();
            drop(sender);
        };
        let consumer = async {
            let mut messages = 0;
            while let Some(facts) = receiver.recv().await {
                facts.unwrap();
                messages += 1;
            }
            messages
        };
        let ((), messages) = tokio::join!(producer, consumer);
        assert_eq!(messages, expected, "transport should group 16 mask words");
    }
}

fn nullable_input(unit: TimeUnit, lengths: &[usize], bad_time: bool) -> Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::Int64, true),
        Field::new(
            "authorized_at",
            DataType::Timestamp(unit, Some("UTC".into())),
            true,
        ),
        Field::new("amount", DataType::Int64, true),
    ]));
    let records = lengths
        .iter()
        .map(|&rows| {
            let mut keys = vec![Some(7); rows + 5];
            let mut times = vec![Some(10); rows + 5];
            for offset in [0, 63, 64, 1_023, 1_024, 1_025] {
                if offset < rows {
                    keys[offset + 5] = None;
                    times[offset + 5] = Some(9);
                }
            }
            for offset in [62, 1_022, 1_026] {
                if offset < rows {
                    times[offset + 5] = Some(9);
                }
            }
            for offset in [1, 1_027] {
                if offset < rows {
                    times[offset + 5] = Some(20);
                }
            }
            if rows > 1_024 {
                times[1_024 + 5] = if bad_time { Some(i64::MAX) } else { None };
            }
            let times = datafusion::arrow::compute::cast(
                &Int64Array::from(times),
                schema.field(1).data_type(),
            )
            .unwrap();
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(keys)),
                    times,
                    Arc::new(Int64Array::from(vec![42; rows + 5])),
                ],
            )
            .unwrap()
            .slice(5, rows)
        })
        .collect();
    Batch::table(records, BatchMetadata::default()).unwrap()
}

async fn admission_result(
    batch: &Batch,
    packed: bool,
    opposite: (IngressState, Option<i64>),
) -> (
    std::result::Result<(), String>,
    StreamJoinStatus,
    Vec<(u64, EventTime)>,
) {
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(20_000, 10_000_000, 1_000).unwrap();
    let mut operator = StreamJoinOperator::new(
        "match",
        batch.table_payload().unwrap().schema().clone(),
        right_schema(),
        declaration,
    )
    .unwrap();
    let plan = operator.ingress_plan("left").unwrap();
    let (sender, preparation) = crate::operator::ingress::Preparation::join_channel();
    let crate::operator::ingress::Sender::Join(sender) = sender else {
        unreachable!()
    };
    let job = job();
    let context = progress_context(&job, (IngressState::Active, Some(10)), opposite)
        .with_ingress_preparation(packed.then_some(&preparation));
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let result = if packed {
        tokio::try_join!(
            plan.prepare(batch, &sender),
            operator.process_data("left", batch.clone(), &context, &mut collector),
        )
        .map(|_| ())
    } else {
        operator
            .process_data("left", batch.clone(), &context, &mut collector)
            .await
    };
    let observed = (
        result.map_err(|error| error.to_string()),
        operator.status(),
        operator
            .state
            .left
            .iter()
            .map(|row| (row.row_id, row.event_time))
            .collect(),
    );
    assert!(collector.drain("output").is_empty());
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((operator, context, collector));
    drop((sender, preparation));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
    observed
}

#[tokio::test]
async fn test_scaling_ingress_packed_and_inline_preserve_sliced_nulls_progress_and_identity() {
    let batch = nullable_input(TimeUnit::Microsecond, &[0, 1_025, 65, 4_097, 0], false);
    let original = batch.table_payload().unwrap().batches().to_vec();
    for opposite in [
        (IngressState::Active, None),
        (IngressState::Active, Some(60_000_015)),
        (IngressState::Ended, None),
    ] {
        let inline = admission_result(&batch, false, opposite).await;
        let packed = admission_result(&batch, true, opposite).await;
        assert!(inline.0.is_ok());
        assert!(inline.1.left.late_rows > 0);
        assert_eq!(packed, inline);
        match opposite {
            (IngressState::Ended, _) => assert!(packed.2.is_empty()),
            (_, Some(_)) => {
                assert!(!packed.2.is_empty());
                assert!(packed.2.iter().all(|(_, time)| time.as_micros() == 20));
            }
            _ => assert!(packed.2.len() > 1_000),
        }
    }
    assert_eq!(batch.table_payload().unwrap().batches(), original);
}

#[tokio::test]
async fn test_scaling_ingress_later_pack_time_error_keeps_precedence_and_no_commit() {
    let batch = nullable_input(TimeUnit::Second, &[1_025, 65], true);
    let opposite = (IngressState::Active, None);
    let inline = admission_result(&batch, false, opposite).await;
    let packed = admission_result(&batch, true, opposite).await;
    assert!(
        inline
            .0
            .as_ref()
            .unwrap_err()
            .contains("event time cannot be represented")
    );
    assert_eq!(packed, inline);
    assert_eq!(packed.1.left.retained_rows, 0);
    assert!(packed.2.is_empty());
}

#[tokio::test]
async fn test_scaling_ingress_dropped_receiver_stops_blocked_preparation() {
    use futures::poll;
    let operator = StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let plan = operator.ingress_plan("left").unwrap();
    let batch = left_batch(vec![10; 8_192]);
    let (sender, preparation) = crate::operator::ingress::Preparation::join_channel();
    let crate::operator::ingress::Sender::Join(sender) = sender else {
        unreachable!()
    };
    let mut producing = Box::pin(plan.prepare(&batch, &sender));
    assert!(poll!(&mut producing).is_pending());
    drop(preparation);
    tokio::time::timeout(Duration::from_secs(1), producing)
        .await
        .unwrap()
        .unwrap();
}
