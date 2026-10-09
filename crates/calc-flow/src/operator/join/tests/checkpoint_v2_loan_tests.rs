use super::*;
use datafusion::arrow::array::Int8Array;

const INT8_KEY: [u8; 10] = [2, 0, 0, 0, 0, 1, 0, 0, 0, 7];
const ROW_CHARGE: u64 = 101;

fn int8_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int8, false),
        Field::new(
            "at",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
    ]))
}

fn int8_operator() -> StreamJoinOperator {
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "at",
        "at",
        JoinTimeBounds::new(Duration::ZERO, Duration::from_micros(10)).unwrap(),
        JoinStateLimits::new(100, 1_000_000, 100).unwrap(),
    )
    .unwrap();
    StreamJoinOperator::new("v2-loan", int8_schema(), int8_schema(), spec).unwrap()
}

fn int8_record(time: i64) -> RecordBatch {
    RecordBatch::try_new(
        int8_schema(),
        vec![
            Arc::new(Int8Array::from(vec![7_i8])),
            Arc::new(TimestampMicrosecondArray::from(vec![time]).with_timezone("UTC")),
        ],
    )
    .unwrap()
}

fn int8_batch(time: i64) -> Batch {
    Batch::table(vec![int8_record(time)], BatchMetadata::default()).unwrap()
}

fn int8_payload(side: u8, time: i64) -> (StateSegment, [u8; 32]) {
    let record = int8_record(time);
    assert_eq!(encode_join_key_v1(&record, 0, &[0]).unwrap(), INT8_KEY);
    assert_eq!(
        state_row_charge(&record, 0, &[0], "v2-loan").unwrap(),
        ROW_CHARGE
    );
    let options = IpcWriteOptions::try_new(8, false, MetadataVersion::V5).unwrap();
    let mut ipc = Vec::new();
    let mut writer = StreamWriter::try_new_with_options(&mut ipc, &int8_schema(), options).unwrap();
    writer.write(&record).unwrap();
    writer.finish().unwrap();
    drop(writer);
    let mut bytes = header(*b"CFJPAY2\0", side);
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&u64::try_from(ipc.len()).unwrap().to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&ipc);
    let digest = Sha256::digest(&bytes).into();
    (StateSegment::new(bytes), digest)
}

fn int8_base(side: u8, time: i64, digest: &[u8; 32]) -> StateSegment {
    let mut bytes = header(*b"CFJIDX2\0", side);
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&time.to_le_bytes());
    bytes.extend_from_slice(&ROW_CHARGE.to_le_bytes());
    bytes.extend_from_slice(digest);
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&10_u64.to_le_bytes());
    bytes.extend_from_slice(&INT8_KEY);
    StateSegment::new(bytes)
}

fn int8_snapshot(operator: &StreamJoinOperator) -> OperatorStateSnapshot {
    let metadata = JoinCheckpointMetadata {
        layout_version: 2,
        spec: operator.spec.clone(),
        next_left_row_id: 1,
        next_right_row_id: 1,
        next_output_sequence: 1,
        ended: false,
        epoch: 1,
        metrics: JoinMetrics {
            left: SideMetrics {
                retained_rows: 1,
                retained_bytes: ROW_CHARGE,
                ..SideMetrics::default()
            },
            right: SideMetrics {
                retained_rows: 1,
                retained_bytes: ROW_CHARGE,
                ..SideMetrics::default()
            },
            emitted_match_rows: 1,
            ..JoinMetrics::default()
        },
    };
    let Value::Object(mut metadata) = serde_json::to_value(metadata).unwrap() else {
        panic!("object metadata");
    };
    let mut segments = BTreeMap::new();
    let mut inventory = Vec::new();
    for (side, name, time) in [(0, "left", 95), (1, "right", 100)] {
        let (payload, digest) = int8_payload(side, time);
        let sha256 = hex::encode(digest);
        inventory.push(serde_json::json!({
            "side": name, "sha256": sha256, "rows": 1, "bytes": payload.bytes().len(),
        }));
        segments.insert(format!("{name}-base"), int8_base(side, time, &digest));
        segments.insert(format!("{name}-payload-{sha256}"), payload);
    }
    metadata.insert(
        "v2_inventory".into(),
        serde_json::json!({
            "codec_version": 2, "base_epoch": 1, "deltas": [], "payloads": inventory,
        }),
    );
    OperatorStateSnapshot {
        inline_metadata: metadata.into_iter().collect(),
        segments,
    }
}

fn assert_continued(operator: &StreamJoinOperator) {
    assert_eq!(
        (operator.state.left.len(), operator.state.right.len()),
        (1, 2)
    );
    assert_eq!(
        (
            operator.state.next_left_row_id,
            operator.state.next_right_row_id,
            operator.state.next_output_sequence
        ),
        (1, 2, 2)
    );
    assert_eq!(operator.state.last_checkpoint_epoch, Epoch::new(1));
    assert_eq!(operator.status().emitted_match_rows, 2);
    assert_eq!(operator.status().left.retained_bytes, 101);
    assert_eq!(operator.status().right.retained_bytes, 202);
    assert_eq!(
        operator
            .state
            .right
            .iter()
            .map(|row| (row.row_id, row.event_time.as_micros(), row.charge))
            .collect::<Vec<_>>(),
        [(0, 100, 101), (1, 101, 101)]
    );
}

async fn assert_sql_success(
    operator: &mut StreamJoinOperator,
    context: &StreamOperatorContext<'_>,
) {
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("right", int8_batch(101), context, &mut output)
        .await
        .unwrap();
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    let batch = messages[0].as_data().unwrap();
    assert_eq!(batch.metadata().sequence(), 1);
    assert_eq!(batch.num_rows(), 1);
    let record = &batch.table_payload().unwrap().batches()[0];
    assert_eq!(record.num_columns(), 4);
    assert_eq!(
        record
            .column(0)
            .as_any()
            .downcast_ref::<Int8Array>()
            .unwrap()
            .value(0),
        7
    );
    assert_eq!(
        record
            .column(1)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(0),
        95
    );
    assert_eq!(
        record
            .column(2)
            .as_any()
            .downcast_ref::<Int8Array>()
            .unwrap()
            .value(0),
        7
    );
    assert_eq!(
        record
            .column(3)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(0),
        101
    );
    assert_continued(operator);
}

async fn assert_sql_error(operator: &mut StreamJoinOperator, context: &StreamOperatorContext<'_>) {
    let original = operator.compiled.equality_query.clone();
    operator.compiled.equality_query = parse_select_query(&format!(
        "SELECT \"missing_ResourcesExhausted\" FROM {PROBE_TABLE}"
    ))
    .unwrap();
    let before = operator.status();
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    crate::datafusion::owned::observe_legacy_inputs();
    let error = operator
        .process_data("right", int8_batch(102), context, &mut output)
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("missing_ResourcesExhausted"),
        "{error}"
    );
    assert!(
        crate::datafusion::owned::take_legacy_inputs().is_empty(),
        "a nonbudget failure must not retry Legacy SQL"
    );
    assert_eq!(operator.status(), before);
    assert!(output.drain("output").is_empty());
    operator.compiled.equality_query = original;
    assert_continued(operator);
}

async fn assert_pending_loan(
    operator: &mut StreamJoinOperator,
    job: &StreamJobContext,
    context: &StreamOperatorContext<'_>,
) {
    let containers = Arc::downgrade(operator.v2_containers.as_ref().unwrap());
    let pool = operator.checkpoint_preload_test_pool().unwrap();
    let before_bytes = pool.reserved();
    let before = operator.status();
    let lock = operator.runtime.runtime().unwrap().owned_test_query_lock();
    let _guard = lock.lock_owned().await;
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let input = int8_batch(102);
    let source = Arc::downgrade(input.table_payload().unwrap().batches()[0].column(0));
    let mut future = Box::pin(operator.process_data("right", input, context, &mut output));
    tokio::time::timeout(Duration::from_secs(2), async {
        while containers.strong_count() != 2 {
            assert!(futures::poll!(future.as_mut()).is_pending());
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    job.cancellation().cancel();
    assert_eq!(
        containers.strong_count(),
        2,
        "the queued SQL input still borrows the container after a cancel request"
    );
    assert!(source.upgrade().is_some());
    drop(future);
    assert!(source.upgrade().is_none());
    assert_eq!(
        containers.strong_count(),
        1,
        "abandoning the actual process_data future releases its SQL loan"
    );
    assert_eq!(pool.reserved(), before_bytes);
    assert_eq!(operator.status(), before);
    assert!(output.drain("output").is_empty());
    assert_continued(operator);
}

#[tokio::test]
async fn test_v2_reader_behavior_keeps_sql_loans_until_actual_release() {
    let mut operator = int8_operator();
    operator.restore(&int8_snapshot(&operator)).unwrap();
    let containers = Arc::downgrade(operator.v2_containers.as_ref().unwrap());
    let pool = operator.checkpoint_preload_test_pool().unwrap();
    assert!(pool.reserved() > 0);
    assert_eq!(containers.strong_count(), 1);
    let job = job();
    let context = StreamOperatorContext::new(&job, "v2-loan", None);
    assert_sql_success(&mut operator, &context).await;
    assert_sql_error(&mut operator, &context).await;
    assert_pending_loan(&mut operator, &job, &context).await;
    assert_eq!(containers.strong_count(), 1);
    let last_buffer = operator.state.left[0].record.column(0).to_data().buffers()[0].clone();
    operator.reset().unwrap();
    assert!(containers.upgrade().is_none());
    drop(operator);
    assert_eq!(last_buffer.as_slice(), &[7]);
    assert!(
        pool.reserved() > 0,
        "the actual last payload Buffer still owns its paid backing"
    );
    drop(last_buffer);
    drop(context);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert_eq!(pool.reserved(), 0);
}
