//! Decode protobuf Kafka payloads through a runtime-loaded descriptor set.
//!
//! The offline demo decodes two sample orders with the public codec API.
//! Set `CALC_FLOW_KAFKA_BOOTSTRAP` (default `127.0.0.1:9092`) to also
//! consume the prepared `calc-flow-example-orders-proto` topic with the
//! managed source; see docs/connectors/kafka.md for topic preparation.

mod support;
use std::path::PathBuf;
use std::time::Duration;
use support::ExampleResult;

use calc_flow::{ArrowFieldSpec, DecodeBounds, FormatDecoder as _, JsonMap};
fn arguments() -> ExampleResult<Vec<String>> {
    let arguments = support::arguments()?;
    if !arguments.is_empty() && arguments != ["--consume"] && arguments != ["--produce"] {
        return Err("usage: kafka_protobuf_source [--produce|--consume]".into());
    }
    Ok(arguments)
}

#[cfg(test)]
use calc_flow_connectors::kafka::KafkaSourceConfig;
use calc_flow_connectors::protobuf::ProtobufCodec;
use prost::Message as _;
use prost_reflect::{DescriptorPool, DynamicMessage, Value};

const MESSAGE: &str = "calcflow.examples.Order";

fn descriptor_set_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/data/orders.pb")
        .canonicalize()
        .expect("examples/data/orders.pb exists")
}

fn order_schema() -> Vec<ArrowFieldSpec> {
    vec![
        ArrowFieldSpec {
            name: "id".into(),
            data_type: "int64".into(),
            nullable: false,
        },
        ArrowFieldSpec {
            name: "quantity".into(),
            data_type: "int64".into(),
            nullable: false,
        },
        ArrowFieldSpec {
            name: "price".into(),
            data_type: "float64".into(),
            nullable: false,
        },
    ]
}

/// Encodes one sample order with the same descriptor set the source loads.
fn encode_order(id: i64, quantity: i64, price: f64) -> ExampleResult<Vec<u8>> {
    let set =
        prost_types::FileDescriptorSet::decode(std::fs::read(descriptor_set_path())?.as_slice())?;
    let pool = DescriptorPool::from_file_descriptor_set(set)?;
    let descriptor = pool
        .get_message_by_name(MESSAGE)
        .ok_or("sample message missing from the descriptor set")?;
    let mut message = DynamicMessage::new(descriptor.clone());
    for (name, value) in [
        ("id", Value::I64(id)),
        ("quantity", Value::I64(quantity)),
        ("price", Value::F64(price)),
    ] {
        let field = descriptor
            .get_field_by_name(name)
            .ok_or("sample field missing")?;
        message.set_field(&field, value);
    }
    Ok(message.encode_to_vec())
}

fn source_options(bootstrap: &str) -> JsonMap {
    JsonMap::from([
        (
            "bootstrap_servers".to_string(),
            serde_json::Value::String(bootstrap.to_string()),
        ),
        (
            "topic".to_string(),
            serde_json::Value::String("calc-flow-example-orders-proto".into()),
        ),
        ("partitions".to_string(), serde_json::json!([0])),
        (
            "auto_offset_reset".to_string(),
            serde_json::Value::String("earliest".into()),
        ),
        (
            "format".to_string(),
            serde_json::Value::String("protobuf".into()),
        ),
        (
            "descriptor_set".to_string(),
            serde_json::Value::String(descriptor_set_path().to_string_lossy().into_owned()),
        ),
        (
            "message".to_string(),
            serde_json::Value::String(MESSAGE.into()),
        ),
        (
            "schema".to_string(),
            serde_json::json!([
                {"name": "id", "data_type": "int64", "nullable": false},
                {"name": "quantity", "data_type": "int64", "nullable": false},
                {"name": "price", "data_type": "float64", "nullable": false},
            ]),
        ),
    ])
}

fn decode_samples() -> ExampleResult {
    let codec = ProtobufCodec::new("1", &descriptor_set_path(), MESSAGE)?;
    let bounds = DecodeBounds::new(1024, 1 << 20)?;
    for (id, quantity, price) in [(1, 2, 10.0), (2, 3, 20.0)] {
        let payload = encode_order(id, quantity, price)?;
        let batch = codec.decode(&payload, &bounds, &order_schema())?;
        println!("decoded order {id}: {} row(s)", batch.num_rows());
    }
    Ok(())
}

async fn consume_topic(bootstrap: &str) -> ExampleResult {
    let mut registry = support::registry()?;
    calc_flow_connectors::register_kafka_connectors(&mut registry)?;
    let source = support::source(
        "kafka",
        serde_json::to_value(source_options(bootstrap))?,
        Some("protobuf"),
        None,
    );
    support::read_source(source, &registry, true, "at_least_once").await
}

async fn produce_orders(bootstrap: &str) -> ExampleResult {
    use rdkafka::{
        ClientConfig,
        producer::{FutureProducer, FutureRecord},
    };
    let producer: FutureProducer = ClientConfig::new()
        .set("bootstrap.servers", bootstrap)
        .create()?;
    for (id, quantity, price) in [(1, 2, 10.0), (2, 3, 20.0)] {
        let bytes = encode_order(id, quantity, price)?;
        producer
            .send(
                FutureRecord::<(), _>::to("calc-flow-example-orders-proto")
                    .partition(0)
                    .payload(&bytes),
                Duration::from_secs(10),
            )
            .await
            .map_err(|(error, _)| error)?;
    }
    println!("produced 2 protobuf orders to calc-flow-example-orders-proto");
    Ok(())
}

#[tokio::main]
async fn main() -> ExampleResult {
    let arguments = arguments()?;
    let bootstrap =
        std::env::var("CALC_FLOW_KAFKA_BOOTSTRAP").unwrap_or_else(|_| "127.0.0.1:9092".into());
    if arguments == ["--produce"] {
        return produce_orders(&bootstrap).await;
    }

    decode_samples()?;
    if arguments == ["--consume"] || std::env::var_os("CALC_FLOW_KAFKA_BOOTSTRAP").is_some() {
        consume_topic(&bootstrap).await?;
    } else {
        println!("set CALC_FLOW_KAFKA_BOOTSTRAP to consume a prepared topic");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_orders_decode_against_the_example_schema() {
        let codec = ProtobufCodec::new("1", &descriptor_set_path(), MESSAGE).expect("codec");
        let bounds = DecodeBounds::new(1024, 1 << 20).expect("bounds");
        let payload = encode_order(2, 3, 20.0).expect("payload encodes");
        let batch = codec
            .decode(&payload, &bounds, &order_schema())
            .expect("payload decodes");
        assert_eq!(batch.num_rows(), 1);
    }

    #[test]
    fn example_options_parse() {
        KafkaSourceConfig::from_options(&source_options("127.0.0.1:1")).expect("options parse");
    }

    #[test]
    fn managed_project_compiles_without_connecting_to_kafka() {
        let root = tempfile::tempdir().expect("temporary root");
        let mut registry = support::registry().expect("file registry");
        calc_flow_connectors::register_kafka_connectors(&mut registry).expect("Kafka registry");
        let source = support::source(
            "kafka",
            serde_json::to_value(source_options("127.0.0.1:1")).unwrap(),
            Some("protobuf"),
            None,
        );
        let project = support::project(
            root.path(),
            source,
            support::file_sink(root.path(), "at_least_once"),
        )
        .unwrap();
        let plan = calc_flow::compile_stream_project(
            &project,
            &calc_flow::ProviderRegistry::default(),
            &calc_flow::UdfRegistry::new().snapshot(),
            &registry.snapshot(),
            &calc_flow::StreamRequirements::default(),
        )
        .expect("managed source and sink compile");
        assert_eq!(plan.source_binding_ids(), ["input"]);
        assert_eq!(plan.sink_binding_ids(), ["output"]);
    }
}
