use super::*;

fn assert_null_row(row: &StoredRow) {
    assert_eq!(
        (row.row_id, row.event_time.as_micros(), row.charge),
        (0, 95, 136)
    );
    assert_eq!(row.encoded_key.as_slice(), KEY);
    let record = row.record.view();
    assert_eq!(record.schema(), schema());
    assert_eq!(record.num_rows(), 1);
    assert!(record.column(2).is_null(0));
    let dictionary = record
        .column(3)
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .unwrap();
    let values = dictionary
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        values.value(usize::try_from(dictionary.keys().value(0)).unwrap()),
        "red"
    );
    let nested = record
        .column(4)
        .as_any()
        .downcast_ref::<ListArray>()
        .unwrap();
    assert!(!nested.is_null(0));
    let child = nested.value(0);
    let child = child.as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(child.len(), 2);
    assert_eq!(child.value(0), 1);
    assert!(!child.is_null(0));
    assert!(child.is_null(1));
}

fn assert_writer_rows(operator: &StreamJoinOperator) {
    assert_eq!(
        (operator.state.left.len(), operator.state.right.len()),
        (2, 1)
    );
    assert_null_row(&operator.state.left[0]);
    assert_row(
        &operator.state.left[1],
        1,
        96,
        148,
        "猫",
        "blue",
        Some(&[2, 3]),
    );
    assert_row(
        &operator.state.right[0],
        0,
        100,
        141,
        "ok",
        "red",
        Some(&[9]),
    );
    assert_eq!(
        (
            operator.state.next_left_row_id,
            operator.state.next_right_row_id,
            operator.state.next_output_sequence,
        ),
        (2, 1, 1)
    );
    assert_eq!(
        operator.state.metrics,
        JoinMetrics {
            left: SideMetrics {
                retained_rows: 2,
                retained_bytes: 284,
                ..SideMetrics::default()
            },
            right: SideMetrics {
                retained_rows: 1,
                retained_bytes: 141,
                ..SideMetrics::default()
            },
            emitted_match_rows: 2,
            ..JoinMetrics::default()
        }
    );
    assert!(!operator.state.ended);
}

async fn prepared_capture(left: &RecordBatch, right: &RecordBatch) -> OperatorStateSnapshot {
    let mut operator = operator();
    let job = job();
    let context = StreamOperatorContext::new(&job, "v2-match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            "left",
            Batch::table(vec![left.clone()], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    operator
        .process_data(
            "right",
            Batch::table(vec![right.clone()], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    assert_writer_rows(&operator);
    assert_eq!(operator.state.last_checkpoint_epoch, None);
    operator.prepare_checkpoint_async(&context).await.unwrap();
    operator.checkpoint(Epoch::INITIAL).unwrap()
}

fn assert_inventory(snapshot: &OperatorStateSnapshot) {
    let inventory = &snapshot.inline_metadata["v2_inventory"];
    assert_eq!(inventory["codec_version"], 2);
    assert_eq!(inventory["base_epoch"], 1);
    assert_eq!(inventory["deltas"], serde_json::json!([]));
    for (side, rows) in [("left", 2_u64), ("right", 1_u64)] {
        let base = snapshot.segments[&format!("{side}-base")].bytes();
        assert_eq!(&base[..8], b"CFJIDX2\0");
        assert_eq!(&base[16..24], rows.to_le_bytes());
        assert_eq!(&base[24..32], 0_u64.to_le_bytes());
        let payloads = inventory["payloads"].as_array().unwrap();
        let mut payload_rows = 0;
        for payload in payloads.iter().filter(|payload| payload["side"] == side) {
            let digest = payload["sha256"].as_str().unwrap();
            let segment = &snapshot.segments[&format!("{side}-payload-{digest}")];
            assert_eq!(&segment.bytes()[..8], b"CFJPAY2\0");
            assert_eq!(hex::encode(Sha256::digest(segment.bytes())), digest);
            assert_eq!(
                payload["bytes"].as_u64().unwrap(),
                u64::try_from(segment.bytes().len()).unwrap()
            );
            payload_rows += payload["rows"].as_u64().unwrap();
        }
        assert_eq!(payload_rows, rows);
    }
}

fn assert_same_segments(left: &OperatorStateSnapshot, right: &OperatorStateSnapshot) {
    assert_eq!(left.inline_metadata, right.inline_metadata);
    assert_eq!(left.segments.len(), right.segments.len());
    for ((left_name, left), (right_name, right)) in left.segments.iter().zip(&right.segments) {
        assert_eq!(left_name, right_name);
        assert_eq!(left.bytes(), right.bytes());
    }
}

#[tokio::test]
async fn test_join_prepared_capture_writes_v2_payload_locators_deterministically() {
    assert_bulk_capture_fits_default_runtime_credit().await;
    let left = record(
        &[95, 96],
        &[None, Some("猫")],
        &["red", "blue"],
        vec![Some(vec![Some(1), None]), Some(vec![Some(2), Some(3)])],
    );
    let right = record(&[100], &[Some("ok")], &["red"], vec![Some(vec![Some(9)])]);
    let captured = prepared_capture(&left, &right).await;
    assert_eq!(captured.inline_metadata["layout_version"], 2);
    assert_eq!(captured.inline_metadata["epoch"], 1);
    assert_inventory(&captured);

    let mut restored = operator();
    restored.restore(&captured).unwrap();
    assert_writer_rows(&restored);
    assert_eq!(restored.state.last_checkpoint_epoch, Some(Epoch::INITIAL));

    let repeated = prepared_capture(&left, &right).await;
    assert_same_segments(&captured, &repeated);
}

async fn assert_bulk_capture_fits_default_runtime_credit() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("amount", DataType::Int64, false),
    ]));
    let spec = StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "ts",
        "ts",
        JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
        JoinStateLimits::new(4_000_000, 4 << 30, 100_000_000).unwrap(),
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new(
        "bulk-credit",
        Arc::clone(&schema),
        Arc::clone(&schema),
        spec,
    )
    .unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "bulk-credit", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    for (side, prefix) in [("left", "L"), ("right", "R")] {
        let keys = (0..30_625)
            .map(|row| format!("{prefix}{row:07}"))
            .collect::<Vec<_>>();
        let record = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(keys)),
                Arc::new(TimestampMicrosecondArray::from(vec![100_000_000; 30_625])),
                Arc::new(Int64Array::from(vec![7; 30_625])),
            ],
        )
        .unwrap();
        operator
            .process_data(
                side,
                Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
    }
    assert_eq!(
        (operator.state.left.len(), operator.state.right.len()),
        (30_625, 30_625)
    );
    assert!(collector.drain("output").is_empty());
    let pool = operator
        .runtime
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    eprintln!("bulk checkpoint pre-capture credit={}", pool.reserved());
    let captured = operator
        .checkpoint(Epoch::INITIAL)
        .expect("61,250 flat rows must checkpoint within the unchanged default runtime pool");
    assert_eq!(captured.inline_metadata["layout_version"], 2);
    assert_eq!(operator.state.last_checkpoint_epoch, Some(Epoch::INITIAL));
    drop((captured, operator));
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}
