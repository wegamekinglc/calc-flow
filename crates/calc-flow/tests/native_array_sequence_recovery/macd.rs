use super::*;

fn kernel() -> serde_json::Value {
    serde_json::json!({"function":"macd_components","fast_span":1,"slow_span":3,"signal_span":3})
}
fn replay(
    root: &Path,
    resumed: Arc<Mutex<Vec<Vec<u8>>>>,
    events: tokio::sync::mpsc::UnboundedReceiver<()>,
    output: tokio::sync::mpsc::UnboundedSender<Batch>,
    opens: Arc<AtomicUsize>,
) -> StreamingRunner {
    let plan = sequence_plan(kernel(), 256 * 1024 * 1024);
    let source = plan.source_binding_ids()[0].to_owned();
    let sink = plan.sink_binding_ids()[0].to_owned();
    StreamingRunner::new(
        plan,
        BTreeMap::from([(
            source.clone(),
            SourceBinding::new(Replay {
                open_calls: Some(opens),
                values: vec![input(vec![1.0, 3.0]), input(vec![5.0, 7.0])],
                position: 0,
                events,
                resumed,
                source,
            })
            .with_watermark_policy(calc_flow::WatermarkPolicy::Disabled { idle_timeout: None }),
        )]),
        BTreeMap::from([(
            sink,
            vec![SinkBinding::ordinary("output", Collected(output)).unwrap()],
        )]),
        ManagedCheckpointRuntime::new(root).unwrap(),
    )
    .unwrap()
}
fn packed(batch: &Batch) -> Vec<Option<u64>> {
    let payload = batch.external_payload().unwrap();
    let array = payload.as_any().downcast_ref::<NativeArray>().unwrap();
    assert_eq!(array.shape(), &[2, 3]);
    let arrow = array.as_arrow();
    let values = arrow
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Float64Array>()
        .unwrap();
    values
        .values()
        .iter()
        .enumerate()
        .map(|(index, value)| {
            arrow
                .nulls()
                .unwrap()
                .is_valid(index)
                .then(|| value.to_bits())
        })
        .collect()
}
type Prefix = (
    tempfile::TempDir,
    Arc<Mutex<Vec<Vec<u8>>>>,
    Arc<AtomicUsize>,
);
async fn prefix() -> Prefix {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let resumed = Arc::new(Mutex::new(Vec::new()));
    let opens = Arc::new(AtomicUsize::new(0));
    let (send, events) = tokio::sync::mpsc::unbounded_channel();
    let (output, mut received) = tokio::sync::mpsc::unbounded_channel();
    let first = replay(
        &root,
        Arc::clone(&resumed),
        events,
        output,
        Arc::clone(&opens),
    )
    .start()
    .await
    .unwrap();
    send.send(()).unwrap();
    assert_eq!(
        packed(&received.recv().await.unwrap()),
        [0.0_f64, 0.0, 0.0, 1.0, 0.5, 0.5].map(|value| Some(value.to_bits()))
    );
    first.trigger_checkpoint().await.unwrap();
    assert_eq!(first.cancel().await.state, JobState::Cancelled);
    drop(send);
    assert_eq!(opens.load(Ordering::SeqCst), 1);
    (directory, resumed, opens)
}
#[tokio::test]
async fn test_macd_sequence_new_job_replays_packed_components_without_reseeding() {
    tokio::time::timeout(Duration::from_secs(5), restart())
        .await
        .unwrap();
}
async fn restart() {
    let (directory, resumed, opens) = prefix().await;
    let (send, events) = tokio::sync::mpsc::unbounded_channel();
    let (output, mut received) = tokio::sync::mpsc::unbounded_channel();
    let next = replay(
        &directory.path().join("managed"),
        Arc::clone(&resumed),
        events,
        output,
        Arc::clone(&opens),
    )
    .start()
    .await
    .unwrap();
    assert_eq!(*resumed.lock().unwrap(), [vec![1]]);
    assert_eq!(opens.load(Ordering::SeqCst), 2);
    send.send(()).unwrap();
    assert_eq!(
        packed(&received.recv().await.unwrap()),
        [1.5_f64, 1.0, 0.5, 1.75, 1.375, 0.375].map(|value| Some(value.to_bits()))
    );
    drop(send);
    assert_eq!(next.wait().await.state, JobState::Completed);
}
#[tokio::test]
async fn test_macd_sequence_normal_restore_rejects_bad_counts_with_valid_checksum_before_open() {
    let (directory, resumed, opens) = prefix().await;
    let root = directory.path().join("managed");
    let (path, manifest, mut bytes) = saved(&root).await;
    assert_eq!(bytes.len(), 156);
    bytes[108..116].copy_from_slice(&3_u64.to_le_bytes());
    replace_state(&root, path, &manifest, bytes).await;
    let (_send, events) = tokio::sync::mpsc::unbounded_channel();
    let (output, mut received) = tokio::sync::mpsc::unbounded_channel();
    native_error(
        replay(
            &root,
            Arc::clone(&resumed),
            events,
            output,
            Arc::clone(&opens),
        )
        .start()
        .await,
        "state.fast_count",
    );
    assert_eq!(opens.load(Ordering::SeqCst), 1);
    assert!(resumed.lock().unwrap().is_empty());
    assert!(received.try_recv().is_err());
}
#[tokio::test]
async fn test_macd_sequence_terminal_valid_and_corrupt_restore_run_codec_before_source_open() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let opens = Opens::default();
    complete(runner_with(
        &root,
        opens.clone(),
        kernel(),
        256 * 1024 * 1024,
    ))
    .await;
    complete(runner_with(
        &root,
        opens.clone(),
        kernel(),
        256 * 1024 * 1024,
    ))
    .await;
    assert_eq!(opens.sources.load(Ordering::SeqCst), 1);
    assert_eq!(opens.sinks.load(Ordering::SeqCst), 2);
    let (path, manifest, mut bytes) = saved(&root).await;
    assert_eq!(&bytes[8..12], &3_u32.to_le_bytes());
    bytes[116..124].copy_from_slice(&(-0.0_f64).to_bits().to_le_bytes());
    replace_state(&root, path, &manifest, bytes).await;
    native_error(
        runner_with(&root, opens.clone(), kernel(), 256 * 1024 * 1024)
            .start()
            .await,
        "state.fast_value",
    );
    assert_eq!(opens.sources.load(Ordering::SeqCst), 1);
    assert_eq!(opens.sinks.load(Ordering::SeqCst), 2);
}
