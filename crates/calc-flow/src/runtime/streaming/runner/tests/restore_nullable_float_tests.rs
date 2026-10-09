use super::*;
use datafusion::arrow::array::{Array, Float64Array, StringArray, UInt64Array};

type QuotePair = (String, u64, u64, i64, i64, Option<u64>, Option<u64>);

struct QuoteRowsSource {
    inner: BaseRowsSource,
    reopened: Arc<Mutex<Vec<(i64, usize)>>>,
}

#[async_trait]
impl StreamSource for QuoteRowsSource {
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
                batch: quote_row(self.inner.timestamp, self.inner.delivered - 1),
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

fn quote_source(
    permitted: &Arc<AtomicUsize>,
    released: &Arc<AtomicBool>,
    reopened: &Arc<Mutex<Vec<(i64, usize)>>>,
    timestamp: i64,
) -> SourceBinding {
    SourceBinding::new(
        Box::new(QuoteRowsSource {
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

fn quote_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
            false,
        ),
        Field::new("sequence", DataType::UInt64, false),
        Field::new("symbol", DataType::Utf8, false),
        Field::new("price", DataType::Float64, true),
    ]))
}

fn quote_price(timestamp: i64, offset: usize) -> Option<f64> {
    match (timestamp, offset) {
        (95, 0) | (100, 1) => None,
        (100, 0) => Some(-0.0),
        (95, 1) => Some(f64::from_bits(0x7ff8_0000_0000_0042)),
        _ => panic!("fixed nullable quote source position"),
    }
}

fn quote_row(timestamp: i64, offset: usize) -> Batch {
    let record = RecordBatch::try_new(
        quote_schema(),
        vec![
            Arc::new(
                TimestampMicrosecondArray::from(vec![timestamp + i64::try_from(offset).unwrap()])
                    .with_timezone("UTC"),
            ),
            Arc::new(UInt64Array::from(vec![u64::try_from(offset + 1).unwrap()])),
            Arc::new(StringArray::from(vec!["猫🙂"])),
            Arc::new(Float64Array::from(vec![quote_price(timestamp, offset)])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

struct QuoteSink(Arc<Mutex<Vec<RecordBatch>>>);

#[async_trait]
impl OrdinaryStreamSink for QuoteSink {
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

fn quote_plan(
    observations: &Arc<Mutex<RestoreObservations>>,
    parses: &Arc<AtomicUsize>,
) -> crate::StreamExecutionPlan {
    let schema = quote_schema();
    let mut join = StreamJoinOperator::new(
        "match",
        Arc::clone(&schema),
        schema,
        StreamJoinSpec::inner(
            ["symbol"],
            ["symbol"],
            "event_time",
            "event_time",
            JoinTimeBounds::new(StdDuration::ZERO, StdDuration::from_micros(10)).unwrap(),
            JoinStateLimits::new(100_000, 134_217_728, 1_000_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
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
    PipelineBuilder::new("restore-nullable-float-quotes")
        .unwrap()
        .add_node("match", Box::new(join))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

fn quote_job_spec(
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
            sink_id: "quotes".into(),
            binding: OrdinarySinkBinding::new(Box::new(QuoteSink(records.clone()))),
        }],
        edge_budget: EdgeBudget {
            max_rows: 1,
            max_bytes: 1 << 20,
        },
        delivery_mode: M2DeliveryMode::ProcessLocalOrdered,
        static_inputs: crate::static_input::PreparedStaticInputs::default(),
    }
}

fn quote_checkpoint(
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

fn quote_column<'a, T: 'static>(record: &'a RecordBatch, name: &str) -> &'a T {
    record
        .column_by_name(name)
        .unwrap()
        .as_any()
        .downcast_ref::<T>()
        .unwrap()
}

fn quote_pair(record: &RecordBatch, index: usize) -> QuotePair {
    let symbol = quote_column::<StringArray>(record, "left__symbol").value(index);
    assert_eq!(
        symbol,
        quote_column::<StringArray>(record, "right__symbol").value(index)
    );
    (
        symbol.to_owned(),
        quote_column::<UInt64Array>(record, "left__sequence").value(index),
        quote_column::<UInt64Array>(record, "right__sequence").value(index),
        quote_column::<TimestampMicrosecondArray>(record, "left__event_time").value(index),
        quote_column::<TimestampMicrosecondArray>(record, "right__event_time").value(index),
        quote_price_bits(record, "left__price", index),
        quote_price_bits(record, "right__price", index),
    )
}

fn quote_price_bits(record: &RecordBatch, name: &str, index: usize) -> Option<u64> {
    let values = quote_column::<Float64Array>(record, name);
    (!values.is_null(index)).then(|| values.value(index).to_bits())
}

fn assert_quote_pairs(records: &[RecordBatch]) {
    let mut actual = records
        .iter()
        .flat_map(|record| (0..record.num_rows()).map(|index| quote_pair(record, index)))
        .collect::<Vec<_>>();
    actual.sort_unstable();
    let expected = [
        ("猫🙂".to_owned(), 1, 2, 95, 101, None, None),
        (
            "猫🙂".to_owned(),
            2,
            1,
            96,
            100,
            Some(0x7ff8_0000_0000_0042),
            Some(0x8000_0000_0000_0000),
        ),
        (
            "猫🙂".to_owned(),
            2,
            2,
            96,
            101,
            Some(0x7ff8_0000_0000_0042),
            None,
        ),
    ];
    assert_eq!(actual, expected);
}

async fn wait_quote_pairs(records: &Mutex<Vec<RecordBatch>>) {
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
    .expect("three continuation pairs reach the sink before watermark");
}

fn assert_quote_owners(observations: &Mutex<RestoreObservations>) {
    let observed = observations.lock();
    assert_eq!(observed.wire.len(), 2);
    assert_eq!(observed.readers.len(), 2);
    eprintln!(
        "Nullable Float64 readers: {:?}; resident: {:?}",
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
        "both real nullable Float64 readers need independent prepaid native workspace"
    );
    assert_eq!(observed.payloads.len(), 2);
    assert!(
        observed
            .payloads
            .iter()
            .all(|owner| owner.is_some_and(|(_, paid)| paid > 0))
    );
}

#[tokio::test]
async fn test_managed_checkpoint_restart_funds_nullable_float_payloads() {
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
        quote_job_spec(
            quote_plan(&observations, &parses),
            &records,
            vec![
                NamedSourceBinding {
                    binding_id: "left".into(),
                    binding: quote_source(&left, &released, &reopened, 95),
                },
                NamedSourceBinding {
                    binding_id: "right".into(),
                    binding: quote_source(&right, &released, &reopened, 100),
                },
            ],
        )
    };
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(spec(), quote_checkpoint(&managed_root, &observations))
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
        .start_checkpointed(spec(), quote_checkpoint(&managed_root, &observations))
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
    wait_for_join_emission(&job, 4).await;
    wait_quote_pairs(&records).await;
    assert!(!released.load(Ordering::SeqCst));
    released.store(true, Ordering::SeqCst);
    let outcome = tokio::time::timeout(StdDuration::from_secs(30), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Completed, "{outcome:?}");
    drop(job);
    runner.shutdown().await.unwrap();
    assert_quote_pairs(&records.lock());
    assert_eq!(parses.load(Ordering::SeqCst), 1);
    assert_quote_owners(&observations);
}
