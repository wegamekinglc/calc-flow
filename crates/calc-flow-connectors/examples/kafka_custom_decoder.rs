//! Register a custom Kafka payload decoder and select it from data-only
//! project options.
//!
//! The offline demo decodes one pipe-delimited `id|quantity|price` record
//! through a user-written [`FormatDecoder`], then wires the decoder into
//! the trusted connector registry. Set `CALC_FLOW_KAFKA_BOOTSTRAP`
//! (default `127.0.0.1:9092`) to also consume the prepared
//! `calc-flow-example-orders-pipe` topic; see docs/connectors/kafka.md.

use std::collections::BTreeMap;
use std::error::Error;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{ArrayRef, Float64Array, Int64Array};
use arrow::record_batch::RecordBatch;
use calc_flow::{
    ArrowFieldSpec, Batch, BatchMetadata, CalcFlowError, ConnectorRegistry,
    ConnectorSourceFactory as _, DecodeBounds, FormatDecoder, FormatIdentity, JsonMap, SourceEvent,
    StreamSource as _,
};
use calc_flow_connectors::arrow_schema::schema_from_spec;
use calc_flow_connectors::kafka::{
    KafkaDecoderRegistry, KafkaSource, KafkaSourceConfig, KafkaSourceFactory,
};
use calc_flow_connectors::register_kafka_connectors_with_decoders;

/// The pipe-delimited orders decoder identity.
const IDENTITY: &str = "pipe-orders";

/// Decodes one `id|quantity|price` record per payload.
struct PipeOrdersDecoder;

impl PipeOrdersDecoder {
    fn identity() -> FormatIdentity {
        FormatIdentity::new(IDENTITY, "1").expect("the decoder identity is valid")
    }
}

impl FormatDecoder for PipeOrdersDecoder {
    fn identity(&self) -> FormatIdentity {
        Self::identity()
    }

    fn decode(
        &self,
        bytes: &[u8],
        bounds: &DecodeBounds,
        schema: &[ArrowFieldSpec],
    ) -> calc_flow::Result<Batch> {
        let values = parse_fields(bytes)?;
        let columns = schema
            .iter()
            .map(|field| {
                values
                    .iter()
                    .find(|(name, _)| *name == field.name)
                    .map(|(_, column)| Arc::clone(column))
                    .ok_or_else(|| invalid(format!("no pipe field named {:?}", field.name)))
            })
            .collect::<calc_flow::Result<Vec<_>>>()?;
        let batch = RecordBatch::try_new(schema_from_spec(schema)?, columns)
            .map_err(|error| invalid(error.to_string()))?;
        bounds.check(&Self::identity(), 1, batch.get_array_memory_size() as u64)?;
        Batch::table(
            vec![batch],
            BatchMetadata::new(IDENTITY, 0, BTreeMap::new())?,
        )
    }
}

fn parse_fields(bytes: &[u8]) -> calc_flow::Result<[(&'static str, ArrayRef); 3]> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| invalid(format!("payload is not UTF-8: {error}")))?;
    let fields: Vec<&str> = text.trim_end().split('|').collect();
    let [id, quantity, price] = fields.as_slice() else {
        return Err(invalid(format!(
            "expected id|quantity|price, found {} fields",
            fields.len()
        )));
    };
    Ok([
        ("id", Arc::new(Int64Array::from(vec![parse::<i64>(id)?]))),
        (
            "quantity",
            Arc::new(Int64Array::from(vec![parse::<i64>(quantity)?])),
        ),
        (
            "price",
            Arc::new(Float64Array::from(vec![parse::<f64>(price)?])),
        ),
    ])
}

fn parse<T: std::str::FromStr>(value: &str) -> calc_flow::Result<T> {
    value
        .parse()
        .map_err(|_| invalid(format!("{value:?} is not a number")))
}

fn invalid(detail: String) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "payload".into(),
        message: detail,
    }
}

fn source_options(bootstrap: &str) -> JsonMap {
    JsonMap::from([
        (
            "bootstrap_servers".to_string(),
            serde_json::Value::String(bootstrap.to_string()),
        ),
        (
            "topic".to_string(),
            serde_json::Value::String("calc-flow-example-orders-pipe".into()),
        ),
        ("partitions".to_string(), serde_json::json!([0])),
        (
            "auto_offset_reset".to_string(),
            serde_json::Value::String("earliest".into()),
        ),
        (
            "format".to_string(),
            serde_json::Value::String("custom".into()),
        ),
        (
            "decoder".to_string(),
            serde_json::json!({"name": "pipe-orders", "version": "1"}),
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

fn order_schema() -> Vec<ArrowFieldSpec> {
    serde_json::from_value(source_options("127.0.0.1:1")["schema"].clone())
        .expect("the example schema is a field list")
}

async fn consume_topic(
    bootstrap: &str,
    decoders: &KafkaDecoderRegistry,
) -> Result<(), Box<dyn Error>> {
    let config = KafkaSourceConfig::from_options(&source_options(bootstrap))?;
    let mut source = KafkaSource::with_decoders(config, decoders)?;
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
    println!("consumed {delivered} custom-decoded row(s) from the broker");
    Ok(())
}

fn offline_demo(decoders: &KafkaDecoderRegistry) -> Result<(), Box<dyn Error>> {
    let batch = decoders.resolve(&PipeOrdersDecoder::identity())?.decode(
        b"1|2|10.0",
        &DecodeBounds::new(1024, 1 << 20)?,
        &order_schema(),
    )?;
    println!("decoded order: {} row(s)", batch.num_rows());

    // The shared registry flows into the trusted connector registry; the
    // data-only `decoder` option selects `pipe-orders/1` at open time.
    let factory = KafkaSourceFactory::new().with_decoders(decoders.clone());
    factory.validate(&source_options("127.0.0.1:1"))?;
    let mut connectors = ConnectorRegistry::new();
    register_kafka_connectors_with_decoders(&mut connectors, decoders.clone())?;
    println!("registered pipe-orders/1 with the kafka connector");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let decoders = KafkaDecoderRegistry::default();
    decoders.register(Arc::new(PipeOrdersDecoder))?;
    offline_demo(&decoders)?;
    if let Ok(bootstrap) = std::env::var("CALC_FLOW_KAFKA_BOOTSTRAP") {
        consume_topic(&bootstrap, &decoders).await?;
    } else {
        println!("set CALC_FLOW_KAFKA_BOOTSTRAP to consume a prepared topic");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipe_payloads_decode_and_malformed_input_fails() {
        let decoder = PipeOrdersDecoder;
        let bounds = DecodeBounds::new(1024, 1 << 20).expect("bounds");
        let batch = decoder
            .decode(b"2|3|20.0", &bounds, &order_schema())
            .expect("payload decodes");
        assert_eq!(batch.num_rows(), 1);
        assert!(decoder.decode(b"2|3", &bounds, &order_schema()).is_err());
        assert!(
            decoder
                .decode(b"2|x|20.0", &bounds, &order_schema())
                .is_err()
        );
    }

    #[test]
    fn example_options_parse() {
        KafkaSourceConfig::from_options(&source_options("127.0.0.1:1")).expect("options parse");
    }
}
