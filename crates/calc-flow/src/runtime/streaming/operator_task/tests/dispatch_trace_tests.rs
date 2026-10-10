use super::*;

#[tokio::test]
async fn test_benchmark_frontier_waits_for_full_watermark_dispatch() {
    let release = Arc::new(tokio::sync::Notify::new());
    let entered = Arc::new(AtomicBool::new(false));
    let output = Port::new("output", BatchKind::Table, false, None).unwrap();
    let operator = PendingHandlerOperator {
        input_ports: vec![Port::new("input", BatchKind::Table, true, None).unwrap()],
        output_ports: vec![output.clone()],
        handler: PendingHandler::Watermark,
        resolution: HandlerResolution::Release(release.clone()),
        entered: entered.clone(),
        poll_dropped: Arc::default(),
    };
    let mut harness = harness_with_operator(
        &["input"],
        1,
        CompiledStreamOperator::External(Box::new(operator)),
        output,
        None,
        None,
    );
    let trace = harness.dispatch_trace.clone();
    trace.start(7, "fingerprint", "asof", false).unwrap();
    trace.register_nodes(&["node".into()]).unwrap();
    start(&mut harness).await;
    harness
        .inputs
        .get_mut("input")
        .unwrap()
        .send(StreamMessage::watermark(EventTime::from_micros(9)))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        while !entered.load(Ordering::SeqCst) {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(trace.frontier("node"), None);
    release.notify_one();
    tokio::time::timeout(Duration::from_secs(5), async {
        while trace.frontier("node").is_none() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(trace.frontier("node"), Some(9));
    assert_eq!(trace.take().unwrap()["completed_frontiers"]["node"], 9);
    harness.cancellation.cancel();
    assert!(harness.supervisor.join_all().await.errors.is_empty());
}
