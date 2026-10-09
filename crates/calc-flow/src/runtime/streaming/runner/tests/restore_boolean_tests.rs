use super::*;
use datafusion::arrow::array::BooleanArray;

type BooleanPair = (i64, i64, bool, bool);

struct BooleanRowsSource {
    inner: BaseRowsSource,
    reopened: Arc<Mutex<Vec<(i64, usize)>>>,
}

#[async_trait]
impl StreamSource for BooleanRowsSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.inner.open(cursor).await?;
        self.reopened
            .lock()
            .push((self.inner.timestamp, self.inner.delivered));
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        Ok(match self.inner.next().await? {
            Some(SourceEvent::Data { cursor, .. }) => Some(SourceEvent::Data {
                batch: boolean_row(self.inner.timestamp, self.inner.delivered - 1),
                cursor,
            }),
            event => event,
        })
    }

    async fn close(&mut self) -> Result<()> {
        self.inner.close().await
    }

    fn capabilities(&self) -> SourceCapabilities {
        self.inner.capabilities()
    }
}

fn boolean_source(
    permitted: &Arc<AtomicUsize>,
    released: &Arc<AtomicBool>,
    reopened: &Arc<Mutex<Vec<(i64, usize)>>>,
    timestamp: i64,
) -> SourceBinding {
    SourceBinding::new(
        Box::new(BooleanRowsSource {
            inner: BaseRowsSource {
                permitted: permitted.clone(),
                released: released.clone(),
                count: 2,
                delivered: 0,
                timestamp,
                watermark_delivered: false,
            },
            reopened: reopened.clone(),
        }),
        None,
        0,
    )
    .unwrap()
}

fn boolean_schema() -> Arc<Schema> {
    let mut fields = fixed_schema().fields().to_vec();
    fields.push(Arc::new(Field::new("flag", DataType::Boolean, false)));
    Arc::new(Schema::new(fields))
}

fn boolean_value(timestamp: i64, offset: usize) -> bool {
    match (timestamp, offset) {
        (95, 0) | (100, 1) => false,
        (95, 1) | (100, 0) => true,
        _ => panic!("the two-row source has fixed Boolean values"),
    }
}

fn boolean_row(timestamp: i64, offset: usize) -> Batch {
    let value = boolean_value(timestamp, offset);
    let record = RecordBatch::try_new(
        boolean_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                timestamp + i64::try_from(offset).unwrap(),
            ])),
            Arc::new(BooleanArray::from(vec![value])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

struct BooleanSink(Arc<Mutex<Vec<RecordBatch>>>);

#[async_trait]
impl OrdinaryStreamSink for BooleanSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.0
            .lock()
            .extend(batch.table_payload()?.batches().iter().cloned());
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

fn boolean_plan(
    observations: &Arc<Mutex<RestoreObservations>>,
    parses: &Arc<AtomicUsize>,
) -> crate::StreamExecutionPlan {
    let schema = boolean_schema();
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
    join.set_checkpoint_v1_test_producer();
    let observed_parses = parses.clone();
    join.set_checkpoint_metadata_test_hook(Arc::new(move |_, parsing| {
        if parsing {
            observed_parses.fetch_add(1, Ordering::SeqCst);
        }
    }));
    let schemas = observations.clone();
    join.set_checkpoint_schema_test_hook(Arc::new(move |credit, constructing| {
        if constructing {
            schemas.lock().descriptor = credit.map(|credit| std::ptr::from_ref(credit) as usize);
        }
    }));
    let observed = observations.clone();
    join.set_checkpoint_decoded_row_test_hook(Arc::new(
        move |credit, payload, decoded, owned_work| {
            let mut observed = observed.lock();
            if decoded {
                observed.payloads.push(payload);
            } else {
                observed.readers.push((
                    credit.map(|credit| (std::ptr::from_ref(credit) as usize, credit.size())),
                    owned_work && std::thread::current().name() == Some("calc-flow-gather"),
                ));
            }
        },
    ));
    PipelineBuilder::new("restore-booleans")
        .unwrap()
        .add_node("match", Box::new(join))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

fn boolean_job_spec(
    plan: crate::StreamExecutionPlan,
    records: &Arc<Mutex<Vec<RecordBatch>>>,
    sources: Vec<NamedSourceBinding>,
) -> ContinuousJobSpec {
    ContinuousJobSpec {
        context: StreamJobContext::new(
            91,
            plan.fingerprint(),
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        plan,
        sources,
        sinks: vec![NamedSinkBinding {
            output_id: "output".into(),
            sink_id: "booleans".into(),
            binding: OrdinarySinkBinding::new(Box::new(BooleanSink(records.clone()))),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    }
}

fn checkpoint(
    root: &Path,
    observations: &Arc<Mutex<RestoreObservations>>,
) -> CheckpointRuntimeSpec {
    let wire_owners = observations.clone();
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
        assert!(bytes.starts_with(b"CFJDLT1\0"));
        assert!(credit.size() >= bytes.len());
        wire_owners.lock().wire.push(Arc::as_ptr(credit) as usize);
        Ok(())
    }))
}

fn pair_at(record: &RecordBatch, index: usize) -> BooleanPair {
    let timestamp = |name| {
        record
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(index)
    };
    let flag = |name| {
        record
            .column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .value(index)
    };
    (
        timestamp("left__ts"),
        timestamp("right__ts"),
        flag("left__flag"),
        flag("right__flag"),
    )
}

fn assert_boolean_pairs(records: &[RecordBatch]) {
    let mut actual = records
        .iter()
        .flat_map(|record| (0..record.num_rows()).map(|index| pair_at(record, index)))
        .collect::<Vec<_>>();
    actual.sort_unstable();
    let expected = [
        (95, 101, false, false),
        (96, 100, true, true),
        (96, 101, true, false),
    ];
    assert_eq!(actual, expected);
}

fn assert_boolean_owners(observations: &Mutex<RestoreObservations>) {
    let observed = observations.lock();
    assert_eq!(observed.wire.len(), 2);
    assert_eq!(observed.readers.len(), 2);
    eprintln!(
        "boolean restored readers: {:?}; resident: {:?}",
        observed.readers, observed.payloads
    );
    assert!(
        observed.readers.iter().all(|(funding, native)| {
            funding.is_some_and(|(identity, paid)| {
                paid > 0
                    && Some(identity) != observed.descriptor
                    && !observed.wire.contains(&identity)
                    && *native
            })
        }),
        "both real boolean readers need independent prepaid native workspace"
    );
    assert_eq!(observed.payloads.len(), 2);
    assert!(
        observed
            .payloads
            .iter()
            .all(|owner| owner.is_some_and(|(_, paid)| paid > 0))
    );
}

async fn wait_collected(records: &Mutex<Vec<RecordBatch>>) {
    tokio::time::timeout(StdDuration::from_secs(10), async {
        loop {
            if records
                .lock()
                .iter()
                .map(RecordBatch::num_rows)
                .sum::<usize>()
                >= 3
            {
                return;
            }
            tokio::time::sleep(StdDuration::from_millis(5)).await;
        }
    })
    .await
    .expect("three continuation pairs must reach the actual sink before watermark");
}

#[tokio::test]
async fn test_managed_checkpoint_restart_funds_boolean_payloads() {
    let directory = tempfile::tempdir().unwrap();
    let managed_root = directory.path().join("managed");
    let observations = Arc::new(Mutex::new(RestoreObservations::default()));
    let parses = Arc::new(AtomicUsize::new(0));
    let left = Arc::new(AtomicUsize::new(0));
    let right = Arc::new(AtomicUsize::new(0));
    let released = Arc::new(AtomicBool::new(false));
    let reopened = Arc::new(Mutex::new(Vec::new()));
    let records = Arc::new(Mutex::new(Vec::new()));
    let spec = || {
        boolean_job_spec(
            boolean_plan(&observations, &parses),
            &records,
            vec![
                NamedSourceBinding {
                    binding_id: "left".into(),
                    binding: boolean_source(&left, &released, &reopened, 95),
                },
                NamedSourceBinding {
                    binding_id: "right".into(),
                    binding: boolean_source(&right, &released, &reopened, 100),
                },
            ],
        )
    };
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(spec(), checkpoint(&managed_root, &observations))
        .await
        .unwrap();
    left.store(1, Ordering::SeqCst);
    right.store(1, Ordering::SeqCst);
    wait_retained(&job, 1, 1).await;
    wait_for_join_emission(&job, 1).await;
    job.trigger_checkpoint().await.unwrap();
    assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
    drop(job);
    runner.shutdown().await.unwrap();
    records.lock().clear();
    reopened.lock().clear();

    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(spec(), checkpoint(&managed_root, &observations))
        .await
        .unwrap();
    wait_retained(&job, 1, 1).await;
    let mut cursors = reopened.lock().clone();
    cursors.sort_unstable();
    assert_eq!(cursors, [(95, 1), (100, 1)]);
    assert!(!released.load(Ordering::SeqCst));
    left.store(2, Ordering::SeqCst);
    right.store(2, Ordering::SeqCst);
    wait_retained(&job, 2, 2).await;
    wait_collected(&records).await;
    assert!(!released.load(Ordering::SeqCst));
    released.store(true, Ordering::SeqCst);
    let outcome = tokio::time::timeout(StdDuration::from_secs(30), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Completed, "{outcome:?}");
    drop(job);
    runner.shutdown().await.unwrap();
    assert_boolean_pairs(&records.lock());
    assert_eq!(parses.load(Ordering::SeqCst), 1);
    assert_boolean_owners(&observations);
}
