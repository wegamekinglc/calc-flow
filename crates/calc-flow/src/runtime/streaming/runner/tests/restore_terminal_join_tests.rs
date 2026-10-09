use super::*;

#[derive(Default)]
struct SourceCalls {
    opened: AtomicUsize,
    polled: AtomicUsize,
    closed: AtomicUsize,
    delivered: Mutex<Vec<(i64, usize)>>,
}

struct TerminalSource {
    inner: BaseRowsSource,
    calls: Arc<SourceCalls>,
}

#[async_trait]
impl StreamSource for TerminalSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.calls.opened.fetch_add(1, Ordering::SeqCst);
        self.inner.open(cursor).await
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        self.calls.polled.fetch_add(1, Ordering::SeqCst);
        self.inner.next().await
    }

    async fn close(&mut self) -> Result<()> {
        self.calls.closed.fetch_add(1, Ordering::SeqCst);
        self.calls
            .delivered
            .lock()
            .push((self.inner.timestamp, self.inner.delivered));
        self.inner.close().await
    }

    fn capabilities(&self) -> SourceCapabilities {
        self.inner.capabilities()
    }
}

fn terminal_source(
    timestamp: i64,
    released: &Arc<AtomicBool>,
    calls: &Arc<SourceCalls>,
) -> SourceBinding {
    SourceBinding::new(
        Box::new(TerminalSource {
            inner: BaseRowsSource {
                permitted: Arc::new(AtomicUsize::new(1)),
                released: Arc::clone(released),
                count: 1,
                delivered: 0,
                timestamp,
                watermark_delivered: false,
            },
            calls: Arc::clone(calls),
        }),
        None,
        0,
    )
    .unwrap()
}

struct Recovered {
    manifest: crate::CheckpointManifest,
    parses: usize,
    readers: usize,
    wire: usize,
}

struct TerminalSink {
    inner: PeriodicCheckpointSink,
    records: Arc<Mutex<Vec<RecordBatch>>>,
    recovered: Arc<Mutex<Option<Recovered>>>,
    observations: Arc<Mutex<RestoreObservations>>,
    parses: Arc<AtomicUsize>,
}

#[async_trait]
impl TransactionalStreamSink for TerminalSink {
    async fn open(&mut self) -> Result<()> {
        self.inner.open().await
    }
    async fn begin_epoch(&mut self, epoch: crate::Epoch) -> Result<()> {
        self.inner.begin_epoch(epoch).await
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.records
            .lock()
            .extend(batch.table_payload()?.batches().iter().cloned());
        self.inner.write(batch).await
    }
    async fn pre_commit(&mut self, epoch: crate::Epoch) -> Result<JsonMap> {
        self.inner.pre_commit(epoch).await
    }
    async fn commit(&mut self, epoch: crate::Epoch, state: &JsonMap) -> Result<()> {
        self.inner.commit(epoch, state).await
    }
    async fn abort(&mut self, epoch: crate::Epoch, state: Option<&JsonMap>) -> Result<()> {
        self.inner.abort(epoch, state).await
    }
    async fn recover(&mut self, manifest: &crate::CheckpointManifest) -> Result<()> {
        let recovered = {
            let observed = self.observations.lock();
            Recovered {
                manifest: manifest.clone(),
                parses: self.parses.load(Ordering::SeqCst),
                readers: observed.readers.len(),
                wire: observed.wire.len(),
            }
        };
        *self.recovered.lock() = Some(recovered);
        self.inner.recover(manifest).await
    }
    async fn close(&mut self) -> Result<()> {
        self.inner.close().await
    }
}

#[derive(Default)]
struct Fixture {
    observations: Arc<Mutex<RestoreObservations>>,
    parses: Arc<AtomicUsize>,
    records: Arc<Mutex<Vec<RecordBatch>>>,
    log: Arc<Mutex<Vec<String>>>,
    sink_closed: Arc<AtomicUsize>,
    recovered: Arc<Mutex<Option<Recovered>>>,
}

fn terminal_plan(fixture: &Fixture) -> crate::StreamExecutionPlan {
    let schema = fixed_schema();
    let mut join = StreamJoinOperator::new(
        "match",
        Arc::clone(&schema),
        schema,
        StreamJoinSpec::inner(
            ["key"],
            ["key"],
            "ts",
            "ts",
            JoinTimeBounds::new(StdDuration::ZERO, StdDuration::from_micros(10)).unwrap(),
            JoinStateLimits::new(100_000, 134_217_728, 1_000_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let parses = Arc::clone(&fixture.parses);
    join.set_checkpoint_metadata_test_hook(Arc::new(move |_, parsing| {
        if parsing {
            parses.fetch_add(1, Ordering::SeqCst);
        }
    }));
    let schemas = Arc::clone(&fixture.observations);
    join.set_checkpoint_schema_test_hook(Arc::new(move |credit, building| {
        if building {
            schemas.lock().descriptor = credit.map(|credit| std::ptr::from_ref(credit) as usize);
        }
    }));
    let observed = Arc::clone(&fixture.observations);
    join.set_checkpoint_decoded_row_test_hook(Arc::new(move |credit, payload, copied, owned| {
        let mut observed = observed.lock();
        if copied {
            observed.payloads.push(payload);
        } else {
            observed.readers.push((
                credit.map(|credit| (std::ptr::from_ref(credit) as usize, credit.size())),
                owned && std::thread::current().name() == Some("calc-flow-gather"),
            ));
        }
    }));
    PipelineBuilder::new("terminal-join")
        .unwrap()
        .add_node("match", Box::new(join))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

fn terminal_spec(
    fixture: &Fixture,
    released: &Arc<AtomicBool>,
    calls: &Arc<SourceCalls>,
) -> ContinuousJobSpec {
    let plan = terminal_plan(fixture);
    ContinuousJobSpec {
        context: StreamJobContext::new(
            92,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources: [("left", 95), ("right", 100)]
            .into_iter()
            .map(|(binding_id, timestamp)| NamedSourceBinding {
                binding_id: binding_id.into(),
                binding: terminal_source(timestamp, released, calls),
            })
            .collect(),
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "pairs".into(),
            binding: OrdinarySinkBinding::new_transactional(Box::new(TerminalSink {
                inner: PeriodicCheckpointSink {
                    log: Arc::clone(&fixture.log),
                    closed: Arc::clone(&fixture.sink_closed),
                },
                records: Arc::clone(&fixture.records),
                recovered: Arc::clone(&fixture.recovered),
                observations: Arc::clone(&fixture.observations),
                parses: Arc::clone(&fixture.parses),
            })),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    }
}

fn terminal_checkpoint(root: &Path, fixture: &Fixture) -> CheckpointRuntimeSpec {
    let observed = Arc::clone(&fixture.observations);
    CheckpointRuntimeSpec::managed(
        ManagedCheckpointRuntime::new(root).unwrap(),
        StreamRuntimeConfig {
            checkpoint_interval: StdDuration::from_secs(3_600),
            checkpoint_timeout: StdDuration::from_secs(10),
            ..StreamRuntimeConfig::default()
        },
    )
    .unwrap()
    .with_join_preload_read_hook(Arc::new(move |bytes, _, credit, _| {
        assert!(bytes.starts_with(b"CFJOIN1\0") || bytes.starts_with(b"CFJDLT1\0"));
        assert!(credit.size() >= bytes.len());
        observed.lock().wire.push(Arc::as_ptr(credit) as usize);
        Ok(())
    }))
}

fn assert_literal_pair(records: &[RecordBatch]) {
    let mut pairs = Vec::new();
    for record in records {
        for row in 0..record.num_rows() {
            let key = |side| {
                record
                    .column_by_name(side)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .value(row)
            };
            let time = |side| {
                record
                    .column_by_name(side)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap()
                    .value(row)
            };
            pairs.push((
                key("left__key"),
                time("left__ts"),
                key("right__key"),
                time("right__ts"),
            ));
        }
    }
    assert_eq!(pairs, [(7, 95, 7, 100)]);
}

async fn wait_pair(fixture: &Fixture) {
    tokio::time::timeout(StdDuration::from_secs(10), async {
        while fixture
            .records
            .lock()
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>()
            != 1
        {
            tokio::time::sleep(StdDuration::from_millis(5)).await;
        }
    })
    .await
    .expect("literal first pair must reach the transactional sink before the dirty cut");
}

async fn natural_history(
    root: &Path,
    manifest: &crate::CheckpointManifest,
) -> Vec<(String, Vec<u8>)> {
    let backend = LocalStateBackend::new(root.join("state")).await.unwrap();
    let key =
        StateLineageKey::new(manifest.pipeline_name(), manifest.pipeline_fingerprint()).unwrap();
    let lineage = backend.open_lineage(&key).await.unwrap();
    let mut history = Vec::new();
    for handle in &manifest.operators()["match"].segments {
        history.push((
            handle.segment_id().to_owned(),
            lineage.load_segment(handle).await.unwrap(),
        ));
    }
    history
}

fn history_readers(history: &[(String, Vec<u8>)]) -> usize {
    let mut readers = 0;
    for (id, bytes) in history {
        let count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        eprintln!(
            "natural terminal segment {id}: magic={:?}, count={count}, first_tag={:?}",
            &bytes[..8],
            bytes.get(16)
        );
        if bytes.starts_with(b"CFJOIN1\0") {
            readers += usize::try_from(count).unwrap();
        } else {
            assert!(bytes.starts_with(b"CFJDLT1\0"));
            assert_eq!(
                count, 1,
                "the natural one-row fixture emits one operation per dirty side cut"
            );
            match bytes[16] {
                1 => readers += 1,
                2 => {}
                tag => panic!("unexpected original delta tag {tag}"),
            }
        }
    }
    assert!(
        readers > 0,
        "the retained dirty cut must supply real historical IPC rows"
    );
    readers
}

fn assert_terminal_manifest(manifest: &crate::CheckpointManifest) {
    assert_eq!(manifest.sources().len(), 2);
    for source in manifest.sources().values() {
        assert!(source.ended);
        assert_eq!(
            source.cursor.as_ref().unwrap().order,
            hex::encode(1_u64.to_be_bytes())
        );
    }
    let entry = &manifest.operators()["match"];
    assert_eq!(entry.progress.len(), 2);
    assert!(
        entry
            .progress
            .values()
            .all(|input| input.state == ManifestIngressState::Ended)
    );
    let frontier = &entry.inline_metadata[crate::pipeline::OUTPUT_FRONTIER_METADATA_KEY_V1];
    eprintln!(
        "actual terminal epoch={}, frontier={frontier}, ingress={:?}",
        manifest.epoch().as_u64(),
        entry.progress
    );
}

fn assert_paid_terminal(fixture: &Fixture, recovered: &Recovered, history: &[(String, Vec<u8>)]) {
    let expected_readers = history_readers(history);
    let observed = fixture.observations.lock();
    eprintln!(
        "terminal Join parse={}, readers={:?}, wire={:?}, postcopy={:?}",
        fixture.parses.load(Ordering::SeqCst),
        observed.readers,
        observed.wire,
        observed.payloads
    );
    assert_eq!(
        observed.readers.len(),
        expected_readers,
        "terminal history must reach the authoritative Join row readers"
    );
    assert_eq!(fixture.parses.load(Ordering::SeqCst), 1);
    assert_eq!(observed.wire.len(), history.len());
    assert_eq!(
        (recovered.parses, recovered.readers, recovered.wire),
        (1, expected_readers, history.len())
    );
    assert!(
        observed
            .readers
            .iter()
            .all(
                |(funding, native)| funding.is_some_and(|(id, bytes)| bytes > 0
                    && Some(id) != observed.descriptor
                    && !observed.wire.contains(&id)
                    && *native)
            )
    );
    assert!(
        observed.payloads.is_empty(),
        "tombstones remove the transient upserts before final resident copies"
    );
}

#[tokio::test]
async fn test_terminal_managed_join_restores_paid_v1_history_before_sink_recovery() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("managed");
    let fixture = Fixture::default();
    let released = Arc::new(AtomicBool::new(false));
    let calls = Arc::new(SourceCalls::default());
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(
            terminal_spec(&fixture, &released, &calls),
            terminal_checkpoint(&root, &fixture),
        )
        .await
        .unwrap();
    wait_retained(&job, 1, 1).await;
    wait_for_join_emission(&job, 1).await;
    wait_pair(&fixture).await;
    assert_literal_pair(&fixture.records.lock());
    assert!(!released.load(Ordering::SeqCst));
    job.trigger_checkpoint().await.unwrap();
    released.store(true, Ordering::SeqCst);
    let outcome = tokio::time::timeout(StdDuration::from_secs(30), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Completed, "{outcome:?}");
    assert_eq!(outcome.cause, TerminalCause::NaturalEnd);
    assert_eq!(job.stream_join_status()["match"].emitted_match_rows, 1);
    drop(job);
    runner.shutdown().await.unwrap();
    let mut delivered = calls.delivered.lock().clone();
    delivered.sort_unstable();
    assert_eq!(delivered, [(95, 1), (100, 1)]);
    assert_eq!(calls.closed.load(Ordering::SeqCst), 2);
    fixture.log.lock().clear();
    fixture.sink_closed.store(0, Ordering::SeqCst);
    let no_source_calls = Arc::new(SourceCalls::default());
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(
            terminal_spec(&fixture, &released, &no_source_calls),
            terminal_checkpoint(&root, &fixture),
        )
        .await
        .unwrap();
    let outcome = tokio::time::timeout(StdDuration::from_secs(30), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Completed, "{outcome:?}");
    let restored_status = job.stream_join_status();
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(no_source_calls.opened.load(Ordering::SeqCst), 0);
    assert_eq!(no_source_calls.polled.load(Ordering::SeqCst), 0);
    assert_eq!(no_source_calls.closed.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.sink_closed.load(Ordering::SeqCst), 1);
    assert_literal_pair(&fixture.records.lock());
    let recovered = fixture.recovered.lock().take().unwrap();
    assert_terminal_manifest(&recovered.manifest);
    assert_eq!(
        *fixture.log.lock(),
        [
            "sink-open".to_owned(),
            format!("sink-recover:{}", recovered.manifest.epoch().as_u64()),
            "sink-close".to_owned()
        ]
    );
    let history = natural_history(&root, &recovered.manifest).await;
    assert_paid_terminal(&fixture, &recovered, &history);
    let status = &restored_status["match"];
    assert_eq!(
        (status.left.retained_rows, status.right.retained_rows),
        (0, 0)
    );
    assert_eq!(
        (status.left.retained_bytes, status.right.retained_bytes),
        (0, 0)
    );
    assert!(status.left.ended && status.right.ended);
    assert_eq!(status.emitted_match_rows, 1);
}
