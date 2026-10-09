use super::*;
use datafusion::arrow::{
    array::TimestampMicrosecondArray,
    datatypes::{DataType, Field, Schema, TimeUnit},
};

#[path = "restore_segments_tests.rs"]
mod restore_segments_tests;

#[path = "restore_v2_tests.rs"]
mod restore_v2_tests;

#[path = "restore_writer_v2_tests.rs"]
mod restore_writer_v2_tests;

#[path = "restore_float_tests.rs"]
mod restore_float_tests;

#[path = "restore_boolean_tests.rs"]
mod restore_boolean_tests;

#[path = "restore_utf8_tests.rs"]
mod restore_utf8_tests;

#[path = "restore_nullable_float_tests.rs"]
mod restore_nullable_float_tests;

#[path = "restore_terminal_join_tests.rs"]
mod restore_terminal_join_tests;

#[derive(Default)]
struct RestoreObservations {
    readers: Vec<(Option<(usize, usize)>, bool)>,
    payloads: Vec<Option<(usize, usize)>>,
    descriptor: Option<usize>,
    wire: Vec<usize>,
}

struct BaseRowsSource {
    permitted: Arc<AtomicUsize>,
    released: Arc<AtomicBool>,
    count: usize,
    delivered: usize,
    timestamp: i64,
    watermark_delivered: bool,
}

#[async_trait]
impl StreamSource for BaseRowsSource {
    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        if let Some(cursor) = cursor {
            self.delivered =
                usize::try_from(u64::from_be_bytes(cursor.order().try_into().unwrap())).unwrap();
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        if self.delivered < self.count {
            while self.permitted.load(Ordering::SeqCst) <= self.delivered {
                tokio::time::sleep(StdDuration::from_millis(5)).await;
            }
            self.delivered += 1;
            return Ok(Some(SourceEvent::Data {
                batch: fixed_row(self.timestamp),
                cursor: Cursor::unbound(
                    u64::try_from(self.delivered)
                        .unwrap()
                        .to_be_bytes()
                        .to_vec(),
                    JsonMap::new(),
                )
                .unwrap(),
            }));
        }
        if self.watermark_delivered {
            return Ok(None);
        }
        while !self.released.load(Ordering::SeqCst) {
            tokio::time::sleep(StdDuration::from_millis(5)).await;
        }
        self.watermark_delivered = true;
        Ok(Some(SourceEvent::Watermark(EventTime::from_micros(120))))
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }

    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replayable: true,
            max_batch_rows: 1,
            max_batch_bytes: 1 << 20,
        }
    }
}

fn fixed_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
    ]))
}

fn fixed_row(timestamp: i64) -> Batch {
    let record = RecordBatch::try_new(
        fixed_schema(),
        vec![
            Arc::new(Int64Array::from(vec![7])),
            Arc::new(TimestampMicrosecondArray::from(vec![timestamp])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn restore_plan(
    observations: &Arc<Mutex<RestoreObservations>>,
    parses: &Arc<AtomicUsize>,
) -> crate::StreamExecutionPlan {
    restore_plan_with_producer(observations, parses, false)
}

fn v1_restore_plan(
    observations: &Arc<Mutex<RestoreObservations>>,
    parses: &Arc<AtomicUsize>,
) -> crate::StreamExecutionPlan {
    restore_plan_with_producer(observations, parses, true)
}

fn restore_plan_with_producer(
    observations: &Arc<Mutex<RestoreObservations>>,
    parses: &Arc<AtomicUsize>,
    v1_producer: bool,
) -> crate::StreamExecutionPlan {
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
    if v1_producer {
        join.set_checkpoint_v1_test_producer();
    }
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
                let funding =
                    credit.map(|credit| (std::ptr::from_ref(credit) as usize, credit.size()));
                observed.readers.push((
                    funding,
                    owned_work && std::thread::current().name() == Some("calc-flow-gather"),
                ));
            }
        },
    ));
    let join_output = join.output_ports()[0].schema().unwrap().clone();
    let mut window = WindowSpec::tumbling("left__ts", StdDuration::from_micros(10)).unwrap();
    window.aggregates = vec![AggregateSpec {
        function: AggregateFunction::Count,
        column: "left__ts".into(),
        output: "pairs".into(),
    }];
    let window = WindowAggregateOperator::new("agg", join_output, window).unwrap();
    PipelineBuilder::new("restore-bases")
        .unwrap()
        .add_node("match", Box::new(join))
        .unwrap()
        .add_node("agg", Box::new(window))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("match", "output").unwrap(),
            PortEndpoint::new("agg", "input").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap()
}

fn source(
    permitted: &Arc<AtomicUsize>,
    released: &Arc<AtomicBool>,
    count: usize,
    timestamp: i64,
) -> SourceBinding {
    SourceBinding::new(
        Box::new(BaseRowsSource {
            permitted: permitted.clone(),
            released: released.clone(),
            count,
            delivered: 0,
            timestamp,
            watermark_delivered: false,
        }),
        None,
        0,
    )
    .unwrap()
}

async fn wait_retained(job: &super::super::ContinuousJob, left: u64, right: u64) {
    tokio::time::timeout(StdDuration::from_secs(10), async {
        loop {
            let status = job.stream_join_status();
            if status["match"].left.retained_rows == left
                && status["match"].right.retained_rows == right
            {
                return;
            }
            tokio::time::sleep(StdDuration::from_millis(5)).await;
        }
    })
    .await
    .expect("fixed Join rows must reach the checkpoint cut");
}

#[tokio::test]
async fn test_managed_join_decoded_arrow_rows_have_independent_credit() {
    let directory = tempfile::tempdir().unwrap();
    let managed_root = directory.path().join("managed");
    let observations = Arc::new(Mutex::new(RestoreObservations::default()));
    let parses = Arc::new(AtomicUsize::new(0));
    let left = Arc::new(AtomicUsize::new(0));
    let right = Arc::new(AtomicUsize::new(0));
    let released = Arc::new(AtomicBool::new(false));
    let rows = Arc::new(Mutex::new(Vec::new()));
    let reads = Arc::new(AtomicUsize::new(0));
    let observed_reads = reads.clone();
    let wire_owners = observations.clone();
    let wire_hook: super::super::super::checkpoint_runtime::CheckpointPrepaidReadHook =
        Arc::new(move |bytes, _, credit, _| {
            assert!(bytes.starts_with(b"CFJOIN1\0"));
            assert!(credit.size() >= bytes.len());
            wire_owners.lock().wire.push(Arc::as_ptr(credit) as usize);
            observed_reads.fetch_add(1, Ordering::SeqCst);
            Ok(())
        });
    let checkpoint = || {
        CheckpointRuntimeSpec::managed(
            ManagedCheckpointRuntime::new(&managed_root).unwrap(),
            StreamRuntimeConfig {
                checkpoint_interval: StdDuration::from_secs(3_600),
                checkpoint_timeout: StdDuration::from_secs(10),
                ..StreamRuntimeConfig::default()
            },
        )
        .unwrap()
        .with_join_preload_read_hook(wire_hook.clone())
    };
    let spec = || {
        let mut spec = ac5_job_spec(v1_restore_plan(&observations, &parses), &rows);
        spec.sources = vec![
            NamedSourceBinding {
                binding_id: "left".into(),
                binding: source(&left, &released, 3, 95),
            },
            NamedSourceBinding {
                binding_id: "right".into(),
                binding: source(&right, &released, 1, 100),
            },
        ];
        spec
    };
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(spec(), checkpoint())
        .await
        .unwrap();
    for expected in 1..=3 {
        left.store(expected, Ordering::SeqCst);
        wait_retained(&job, u64::try_from(expected).unwrap(), 0).await;
        job.trigger_checkpoint().await.unwrap();
    }
    right.store(1, Ordering::SeqCst);
    wait_retained(&job, 3, 1).await;
    wait_for_join_emission(&job, 3).await;
    job.trigger_checkpoint().await.unwrap();
    job.trigger_checkpoint().await.unwrap();
    assert_eq!(job.cancel().await.state, ContinuousJobState::Cancelled);
    drop(job);
    runner.shutdown().await.unwrap();

    released.store(true, Ordering::SeqCst);
    let mut runner = ContinuousRunner::new();
    let job = runner
        .start_checkpointed(spec(), checkpoint())
        .await
        .unwrap();
    let outcome = tokio::time::timeout(StdDuration::from_secs(30), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, ContinuousJobState::Completed, "{outcome:?}");
    drop(job);
    runner.shutdown().await.unwrap();
    assert_eq!(rows.lock().as_slice(), [3]);
    assert_eq!(reads.load(Ordering::SeqCst), 2);
    assert_eq!(parses.load(Ordering::SeqCst), 1);

    let observed = observations.lock();
    assert_eq!(observed.readers.len(), 4);
    assert_eq!(observed.payloads.len(), 4);
    assert!(
        observed.readers.iter().all(|(funding, native)| {
            funding.is_some_and(|(identity, paid)| {
                paid > 0
                    && Some(identity) != observed.descriptor
                    && !observed.wire.contains(&identity)
                    && *native
            })
        }),
        "all four real decoded rows need independent prepaid workspace on the native worker: {:?}",
        observed.readers
    );
    assert!(
        observed
            .payloads
            .iter()
            .all(|owner| owner.is_some_and(|(_, paid)| paid > 0)),
        "restored Arrow payloads must keep independent resident credit"
    );
}
