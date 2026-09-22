use std::time::Duration;

use calc_flow::{
    BatchKind, BatchOperator, BatchOperatorContext, CancellationToken, ConnectorIdentity,
    EnvironmentSecretResolver, ExpressionOperator, Port, RunContext, SourceEvent, SourceSchema,
    StreamSource,
};
use calc_flow_connectors::KAFKA_CONNECTOR_VERSION;
use rdkafka::{
    config::ClientConfig,
    mocking::MockCluster,
    producer::{FutureProducer, FutureRecord},
};

use super::*;

async fn next_data(source: &mut dyn StreamSource) -> calc_flow::Result<SourceEvent> {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            match source.next().await? {
                Some(SourceEvent::Idle) => {}
                Some(event) => return Ok(event),
                None => panic!("Kafka source ended before the expected message"),
            }
        }
    })
    .await
    .expect("Kafka fixture must produce the next message within ten seconds")
}

fn runtime_with_decoder(tokio: &Arc<tokio::runtime::Runtime>) -> PyRuntime {
    Python::attach(|py| {
        let runtime = PyRuntime::from_tokio(Arc::clone(tokio));
        let module = PyModule::from_code(
            py,
            c"import pyarrow as pa\n\ndef decode(payload):\n    schema = pa.schema([pa.field('value', pa.int64())])\n    if payload == b'empty':\n        return pa.Table.from_batches([], schema=schema)\n    values = [None] if payload == b'null' else [int(payload)]\n    return pa.record_batch([values], schema=schema)\n",
            c"source_decode.py",
            c"source_decode",
        ).unwrap();
        runtime
            .register_kafka_decoder(
                py,
                "source-values",
                "1",
                module.getattr("decode").unwrap().unbind(),
            )
            .unwrap();
        runtime
    })
}

fn source_options(bootstrap: &str) -> calc_flow::JsonMap {
    BTreeMap::from([
        ("bootstrap_servers".into(), serde_json::json!(bootstrap)),
        ("topic".into(), serde_json::json!("decoder-values")),
        ("partitions".into(), serde_json::json!([0])),
        ("format".into(), serde_json::json!("custom")),
        (
            "decoder".into(),
            serde_json::json!({"name": "source-values", "version": "1"}),
        ),
        (
            "schema".into(),
            serde_json::json!([{"name": "value", "data_type": "int64", "nullable": false}]),
        ),
    ])
}

fn exact_operator(schema: &datafusion::arrow::datatypes::Schema) -> ExpressionOperator {
    let input = Port::new(
        "input",
        BatchKind::Table,
        true,
        Some(
            schema
                .fields()
                .iter()
                .map(|field| field.as_ref().clone())
                .collect(),
        ),
    )
    .unwrap();
    let output = Port::new("output", BatchKind::Table, true, None).unwrap();
    ExpressionOperator::new("increment", "result = value + 1", vec![], None, vec![])
        .unwrap()
        .with_ports(input, output)
        .unwrap()
}

#[test]
fn test_registered_kafka_decoder_empty_then_data_reaches_exact_operator() {
    Python::initialize();
    let tokio = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let runtime = runtime_with_decoder(&tokio);
    let factory = runtime
        .snapshot()
        .unwrap()
        .connectors
        .resolve_source(
            &ConnectorIdentity::new("calc-flow-connectors", "kafka", KAFKA_CONNECTOR_VERSION)
                .unwrap(),
        )
        .unwrap();

    tokio.block_on(async {
        let cluster = MockCluster::new(1).unwrap();
        cluster.create_topic("decoder-values", 1, 1).unwrap();
        let producer: FutureProducer = ClientConfig::new()
            .set("bootstrap.servers", cluster.bootstrap_servers())
            .set("message.timeout.ms", "5000")
            .create()
            .unwrap();
        for payload in ["empty", "41", "null"] {
            producer
                .send(
                    FutureRecord::to("decoder-values")
                        .partition(0)
                        .key("key")
                        .payload(payload),
                    Duration::from_secs(5),
                )
                .await
                .unwrap();
        }
        let options = source_options(&cluster.bootstrap_servers());
        let mut source = factory
            .open(&options, &EnvironmentSecretResolver)
            .await
            .unwrap();
        source.open(None).await.unwrap();
        let SourceSchema::Exact(schema) = source.capabilities().schema else {
            panic!("explicit source schema must be exact");
        };
        let mut operator = exact_operator(&schema);
        let run = RunContext::new(BTreeMap::new(), None, CancellationToken::new()).unwrap();
        let context = BatchOperatorContext { run: &run };

        let SourceEvent::Data { batch, cursor } = next_data(source.as_mut()).await.unwrap() else {
            panic!("empty output must still consume one source message");
        };
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.table_payload().unwrap().schema(), &schema);
        assert_eq!(batch.metadata().sequence(), 1);
        let result = operator
            .process(&BTreeMap::from([("input".into(), batch)]), &context)
            .await
            .unwrap();
        assert_eq!(result["output"].num_rows(), 0);
        source.close().await.unwrap();
        drop(source);

        let mut source = factory
            .open(&options, &EnvironmentSecretResolver)
            .await
            .unwrap();
        source.open(Some(cursor)).await.unwrap();
        let SourceEvent::Data { batch, .. } = next_data(source.as_mut()).await.unwrap() else {
            panic!("normal output must follow the consumed empty output");
        };
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.metadata().sequence(), 2);
        assert_eq!(
            batch.metadata().attributes()["offset"],
            serde_json::json!(1)
        );
        let result = operator
            .process(&BTreeMap::from([("input".into(), batch)]), &context)
            .await
            .unwrap();
        let values = result["output"].table_payload().unwrap().batches()[0]
            .column_by_name("result")
            .unwrap()
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .unwrap();
        assert_eq!(values.value(0), 42);
        let error = next_data(source.as_mut()).await.unwrap_err();
        assert!(
            matches!(&error, calc_flow::CalcFlowError::Connector(inner)
            if inner.identity.provider.as_ref() == "calc-flow-python"),
            "{error}"
        );
        assert!(error.to_string().contains("offset 2"), "{error}");
        source.close().await.unwrap();
    });
}
