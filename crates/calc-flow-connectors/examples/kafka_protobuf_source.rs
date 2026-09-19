//! Decode protobuf Kafka payloads through a runtime-loaded descriptor set.
//!
//! The offline demo decodes two sample orders with the public codec API.
//! Set `CALC_FLOW_KAFKA_BOOTSTRAP` (default `127.0.0.1:9092`) to also
//! consume the prepared `calc-flow-example-orders-proto` topic with the
//! managed source; see docs/connectors/kafka.md for topic preparation.

use std::error::Error;
use std::path::PathBuf;
use std::time::Duration;

use calc_flow::{
    ArrowFieldSpec, DecodeBounds, FormatDecoder as _, JsonMap, SourceEvent, StreamSource as _,
};
use calc_flow_connectors::kafka::{KafkaSource, KafkaSourceConfig};
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
fn encode_order(id: i64, quantity: i64, price: f64) -> Result<Vec<u8>, Box<dyn Error>> {
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

fn decode_samples() -> Result<(), Box<dyn Error>> {
    let codec = ProtobufCodec::new("1", &descriptor_set_path(), MESSAGE)?;
    let bounds = DecodeBounds::new(1024, 1 << 20)?;
    for (id, quantity, price) in [(1, 2, 10.0), (2, 3, 20.0)] {
        let payload = encode_order(id, quantity, price)?;
        let batch = codec.decode(&payload, &bounds, &order_schema())?;
        println!("decoded order {id}: {} row(s)", batch.num_rows());
    }
    Ok(())
}

async fn consume_topic(bootstrap: &str) -> Result<(), Box<dyn Error>> {
    let config = KafkaSourceConfig::from_options(&source_options(bootstrap))?;
    let mut source = KafkaSource::new(config)?;
    source.open(None).await?;
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let mut delivered = 0_u64;
    while delivered < 2 && std::time::Instant::now() < deadline {
        match source.next().await? {
            Some(SourceEvent::Data { batch, .. }) => {
                delivered += batch.num_rows() as u64;
                println!("kafka batch: {} row(s)", batch.num_rows());
            }
            _ => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
    println!("consumed {delivered} protobuf row(s) from the broker");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    decode_samples()?;
    let bootstrap =
        std::env::var("CALC_FLOW_KAFKA_BOOTSTRAP").unwrap_or_else(|_| "127.0.0.1:9092".into());
    if std::env::var_os("CALC_FLOW_KAFKA_BOOTSTRAP").is_some() {
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
}
