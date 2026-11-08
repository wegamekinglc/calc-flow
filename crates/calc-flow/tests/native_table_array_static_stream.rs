use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, Cursor, JobState, NativeWatermarkCapability, ProjectSpec,
    ProviderRegistry, ReplayPositioning, Result, SinkBinding, SourceBinding, SourceCapabilities,
    SourceDeliveryCapability, SourceEvent, SourceSchema, StreamRequirements, StreamSink,
    StreamSource, StreamingRunner, UdfRegistrySnapshot,
    array::{NativeArray, NativeTableArrayFactory},
    compile_stream_project_graph,
};
use datafusion::arrow::{
    array::{Float32Array, Float64Array},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use serde_json::json;
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

struct Rows {
    batches: VecDeque<Batch>,
    opened: Arc<AtomicUsize>,
}
#[async_trait]
impl StreamSource for Rows {
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities {
            replay_positioning: ReplayPositioning::Unsupported,
            delivery: SourceDeliveryCapability::Lossy,
            max_batch_rows: 2,
            max_batch_bytes: 1024,
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: NativeWatermarkCapability::NeverEmits,
        }
    }
    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        self.opened.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        self.batches
            .pop_front()
            .map(|batch| {
                Ok(SourceEvent::Data {
                    batch,
                    cursor: Cursor::new("observations", vec![0], BTreeMap::new())?,
                })
            })
            .transpose()
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
struct Output(Arc<Mutex<Vec<Batch>>>);
#[async_trait]
impl StreamSink for Output {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.0.lock().unwrap().push(batch.clone());
        Ok(())
    }
    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
fn project() -> ProjectSpec {
    let fields = json!([{"name":"a","data_type":"float32","nullable":false},{"name":"b","data_type":"float32","nullable":false}]);
    let mut output = fields.as_array().unwrap().clone();
    output.push(json!({"name":"score","data_type":"float64","nullable":true}));
    let options = json!({"ir_version":2,"inputs":[{"name":"observations","kind":"table","columns":["a","b"]},{"name":"coefficients","kind":"array","binding":"static","dtype":"float64","shape":[2,1]}],"nodes":[{"id":"x","op":"input","name":"observations"},{"id":"w","op":"input","name":"coefficients"},{"id":"y","op":"matmul","left":"x","right":"w"}],"outputs":[{"name":"result","kind":"table","node":"y","attach_to":"observations","names":["score"]}]});
    serde_json::from_value(json!({"format_version":3,"id":"native-static-stream","name":"native-static-stream","runtime":{"mode":"stream","options":{}},"static_inputs":[{"name":"coefficients","kind":"array","mutability":"static","backend":"native","dtype":"float64","shape":[2,1]}],"graph":{"name":"native-static-stream","nodes":[{"id":"calc","operator":{"kind":"external","provider":"native","name":"table_array","version":"1","options":options},"input_ports":[{"name":"observations","kind":"table","schema":fields},{"name":"coefficients","kind":"array"}],"output_ports":[{"name":"result","kind":"table","schema":output}]}]}})).unwrap()
}
fn input() -> Batch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Float32, false),
        Field::new("b", DataType::Float32, false),
    ]));
    let rows = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float32Array::from(vec![1.0, 3.0])),
            Arc::new(Float32Array::from(vec![2.0, 4.0])),
        ],
    )
    .unwrap();
    Batch::table(vec![rows], BatchMetadata::default()).unwrap()
}
#[tokio::test]
async fn test_native_table_stream_prepares_actual_native_static_owner_before_source_open() {
    let registry = ProviderRegistry::default();
    registry
        .register_stream(
            "native",
            "table_array",
            "1",
            Arc::new(NativeTableArrayFactory::default()),
        )
        .unwrap();
    let plan = compile_stream_project_graph(
        &project(),
        &registry,
        &UdfRegistrySnapshot::default(),
        &StreamRequirements::default(),
    )
    .unwrap();
    let opened = Arc::new(AtomicUsize::new(0));
    let output = Arc::new(Mutex::new(Vec::new()));
    let sources = BTreeMap::from([(
        "observations".into(),
        SourceBinding::new(Rows {
            batches: VecDeque::from([input()]),
            opened: Arc::clone(&opened),
        })
        .with_watermark_policy(calc_flow::WatermarkPolicy::Disabled { idle_timeout: None }),
    )]);
    let sinks = BTreeMap::from([(
        "result".into(),
        vec![SinkBinding::ordinary("sink", Output(Arc::clone(&output))).unwrap()],
    )]);
    let weights = NativeArray::from_f64(vec![2.0, 3.0], vec![2, 1], None).unwrap();
    let statics = BTreeMap::from([(
        "coefficients".into(),
        Batch::external(Arc::new(weights), BatchMetadata::default()).unwrap(),
    )]);
    let runner = StreamingRunner::without_checkpoints(plan, sources, sinks)
        .unwrap()
        .with_static_inputs(statics)
        .unwrap();
    let job = runner.start().await.unwrap();
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(3), job.wait())
        .await
        .unwrap();
    assert_eq!(outcome.state, JobState::Completed);
    assert!(outcome.errors.is_empty());
    assert_eq!(opened.load(Ordering::SeqCst), 1);
    let results = output.lock().unwrap();
    let values = results[0].table_payload().unwrap().batches()[0]
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(
        values
            .values()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        [8.0_f64.to_bits(), 18.0_f64.to_bits()]
    );
}

#[tokio::test]
async fn test_direct_builder_rejects_dynamic_static_rhs_before_any_source_opens() {
    use calc_flow::{
        BatchKind, ExternalOperatorSpec, PipelineBuilder, Port, StreamOperatorFactory,
    };
    let options =
        serde_json::to_value(project()).unwrap()["graph"]["nodes"][0]["operator"]["options"]
            .clone();
    let fields = vec![
        Field::new("a", DataType::Float32, false),
        Field::new("b", DataType::Float32, false),
    ];
    let mut attached = fields.clone();
    attached.push(Field::new("score", DataType::Float64, true));
    let spec = ExternalOperatorSpec::new(
        "native",
        "table_array",
        "1",
        serde_json::from_value(options).unwrap(),
    )
    .unwrap();
    let operator = StreamOperatorFactory::create(
        &NativeTableArrayFactory::default(),
        &spec,
        vec![
            Port::new("observations", BatchKind::Table, true, Some(fields)).unwrap(),
            Port::new("coefficients", BatchKind::Array, true, None).unwrap(),
        ],
        vec![Port::new("result", BatchKind::Table, true, Some(attached)).unwrap()],
    )
    .unwrap();
    let plan = PipelineBuilder::new("direct-dynamic-rhs")
        .unwrap()
        .add_node("calc", operator)
        .unwrap()
        .compile_stream(
            &UdfRegistrySnapshot::default(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let opened = Arc::new(AtomicUsize::new(0));
    let sources = ["observations", "coefficients"]
        .map(|name| {
            (
                name.into(),
                SourceBinding::new(Rows {
                    batches: VecDeque::new(),
                    opened: Arc::clone(&opened),
                })
                .with_watermark_policy(calc_flow::WatermarkPolicy::Disabled { idle_timeout: None }),
            )
        })
        .into_iter()
        .collect();
    let sinks = BTreeMap::from([(
        "result".into(),
        vec![SinkBinding::ordinary("sink", Output(Arc::default())).unwrap()],
    )]);
    let runner = StreamingRunner::without_checkpoints(plan, sources, sinks).unwrap();
    let error = match runner.start().await {
        Ok(job) => {
            job.cancel().await;
            panic!("a static RHS was accepted as a dynamic source");
        }
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("static_inputs.coefficients"),
        "{error}"
    );
    assert_eq!(opened.load(Ordering::SeqCst), 0);
}
