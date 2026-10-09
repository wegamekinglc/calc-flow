use super::*;
use datafusion::arrow::{
    array::{Float32Array, Float64Array, PrimitiveArray},
    datatypes::{ArrowPrimitiveType, Float32Type, Float64Type, TimestampMicrosecondType},
};

type FloatPair = (i64, i64, u32, u64, u32, u64);

struct FloatRowsSource {
    inner: BaseRowsSource,
    reopened: Arc<Mutex<Vec<(i64, usize)>>>,
}

#[async_trait]
impl StreamSource for FloatRowsSource {
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
                batch: float_row(self.inner.timestamp, self.inner.delivered - 1),
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

fn float_source(
    permitted: &Arc<AtomicUsize>,
    released: &Arc<AtomicBool>,
    reopened: &Arc<Mutex<Vec<(i64, usize)>>>,
    timestamp: i64,
) -> SourceBinding {
    SourceBinding::new(
        Box::new(FloatRowsSource {
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

fn float_schema() -> Arc<Schema> {
    let mut fields = fixed_schema().fields().to_vec();
    fields.push(Arc::new(Field::new("f32", DataType::Float32, false)));
    fields.push(Arc::new(Field::new("f64", DataType::Float64, false)));
    Arc::new(Schema::new(fields))
}

fn float_bits(timestamp: i64, offset: usize) -> (u32, u64) {
    match (timestamp, offset) {
        (95, 0) => (0x8000_0000, 0x7ff8_0000_0000_006d),
        (100, 0) => (0x7fc0_1234, 0x8000_0000_0000_0000),
        (95, 1) => (0x3f40_0000, 0x4004_0000_0000_0000),
        (100, 1) => (0x4040_0000, 0x4012_0000_0000_0000),
        _ => panic!("the two-row source has fixed float bit patterns"),
    }
}

fn float_row(timestamp: i64, offset: usize) -> Batch {
    let (single, double) = float_bits(timestamp, offset);
    let record = RecordBatch::try_new(
        float_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                timestamp + i64::try_from(offset).unwrap(),
            ])),
            Arc::new(Float32Array::from(vec![f32::from_bits(single)])),
            Arc::new(Float64Array::from(vec![f64::from_bits(double)])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

struct FloatSink(Arc<Mutex<Vec<RecordBatch>>>);

#[async_trait]
impl OrdinaryStreamSink for FloatSink {
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

fn float_plan(
    observations: &Arc<Mutex<RestoreObservations>>,
    parses: &Arc<AtomicUsize>,
) -> crate::StreamExecutionPlan {
    let schema = float_schema();
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
    PipelineBuilder::new("restore-floats")
        .unwrap()
        .add_node("match", Box::new(join))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

fn float_job_spec(
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
            sink_id: "floats".into(),
            binding: OrdinarySinkBinding::new(Box::new(FloatSink(records.clone()))),
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

fn column<'a, T: ArrowPrimitiveType>(record: &'a RecordBatch, name: &str) -> &'a PrimitiveArray<T> {
    record
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .unwrap()
}

fn pair_at(record: &RecordBatch, index: usize) -> FloatPair {
    (
        column::<TimestampMicrosecondType>(record, "left__ts").value(index),
        column::<TimestampMicrosecondType>(record, "right__ts").value(index),
        column::<Float32Type>(record, "left__f32")
            .value(index)
            .to_bits(),
        column::<Float64Type>(record, "left__f64")
            .value(index)
            .to_bits(),
        column::<Float32Type>(record, "right__f32")
            .value(index)
            .to_bits(),
        column::<Float64Type>(record, "right__f64")
            .value(index)
            .to_bits(),
    )
}

fn assert_float_pairs(records: &[RecordBatch]) {
    let mut actual = records
        .iter()
        .flat_map(|record| (0..record.num_rows()).map(|index| pair_at(record, index)))
        .collect::<Vec<_>>();
    actual.sort_unstable();
    let expected = [(95, 101), (96, 100), (96, 101)].map(|(left, right)| {
        let single_left = float_bits(95, usize::try_from(left - 95).unwrap());
        let single_right = float_bits(100, usize::try_from(right - 100).unwrap());
        (
            left,
            right,
            single_left.0,
            single_left.1,
            single_right.0,
            single_right.1,
        )
    });
    assert_eq!(actual, expected);
}

fn assert_float_owners(observations: &Mutex<RestoreObservations>) {
    let observed = observations.lock();
    assert_eq!(observed.wire.len(), 2);
    assert_eq!(observed.readers.len(), 2);
    eprintln!(
        "float restored readers: {:?}; resident: {:?}",
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
        "both real float readers need independent prepaid native workspace"
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
async fn test_managed_join_float_payload_rows_have_independent_credit() {
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
        float_job_spec(
            float_plan(&observations, &parses),
            &records,
            vec![
                NamedSourceBinding {
                    binding_id: "left".into(),
                    binding: float_source(&left, &released, &reopened, 95),
                },
                NamedSourceBinding {
                    binding_id: "right".into(),
                    binding: float_source(&right, &released, &reopened, 100),
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
    assert_float_pairs(&records.lock());
    assert_eq!(parses.load(Ordering::SeqCst), 1);
    assert_float_owners(&observations);
}
