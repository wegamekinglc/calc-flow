//! Register a custom Kafka payload decoder and select it from data-only
//! project options.
//!
//! The offline demo decodes one pipe-delimited `id|quantity|price` record
//! through a user-written [`FormatDecoder`], then wires the decoder into
//! the trusted connector registry. Set `CALC_FLOW_KAFKA_BOOTSTRAP`
//! (default `127.0.0.1:9092`) to also consume the prepared
//! `calc-flow-example-orders-pipe` topic; see docs/connectors/kafka.md.

use std::collections::BTreeMap;
mod support;
use std::sync::Arc;
use support::ExampleResult;

use arrow::array::{ArrayRef, Float64Array, Int64Array};
use arrow::record_batch::RecordBatch;
use calc_flow::{
    ArrowFieldSpec, Batch, BatchMetadata, CalcFlowError, ConnectorRegistry,
    ConnectorSourceFactory as _, DecodeBounds, FormatDecoder, FormatIdentity, JsonMap,
};
use calc_flow_connectors::arrow_schema::schema_from_spec;
use calc_flow_connectors::kafka::{KafkaDecoderRegistry, KafkaSourceFactory};
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

async fn consume_topic(bootstrap: &str, decoders: &KafkaDecoderRegistry) -> ExampleResult {
    let mut registry = support::registry()?;
    register_kafka_connectors_with_decoders(&mut registry, decoders.clone())?;
    let source = support::source(
        "kafka",
        serde_json::to_value(source_options(bootstrap))?,
        Some("custom"),
        None,
    );
    support::read_source(source, &registry, true, "at_least_once").await
}

fn offline_demo(decoders: &KafkaDecoderRegistry) -> ExampleResult {
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
async fn main() -> ExampleResult {
    let arguments = arguments()?;

    let decoders = KafkaDecoderRegistry::default();
    decoders.register(Arc::new(PipeOrdersDecoder))?;
    offline_demo(&decoders)?;
    if arguments == ["--consume"] || std::env::var_os("CALC_FLOW_KAFKA_BOOTSTRAP").is_some() {
        let bootstrap =
            std::env::var("CALC_FLOW_KAFKA_BOOTSTRAP").unwrap_or_else(|_| "127.0.0.1:9092".into());
        consume_topic(&bootstrap, &decoders).await?;
    } else {
        println!("set CALC_FLOW_KAFKA_BOOTSTRAP to consume a prepared topic");
    }
    Ok(())
}

fn arguments() -> ExampleResult<Vec<String>> {
    let arguments = support::arguments()?;
    if !arguments.is_empty() && arguments != ["--consume"] {
        return Err("usage: kafka_custom_decoder [--consume]".into());
    }
    Ok(arguments)
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
        calc_flow_connectors::kafka::KafkaSourceConfig::from_options(&source_options(
            "127.0.0.1:1",
        ))
        .expect("options parse");
    }

    #[test]
    fn managed_project_compiles_with_the_registered_decoder() {
        let root = tempfile::tempdir().expect("temporary root");
        let decoders = KafkaDecoderRegistry::default();
        decoders
            .register(Arc::new(PipeOrdersDecoder))
            .expect("decoder registers");
        let mut registry = support::registry().expect("file registry");
        register_kafka_connectors_with_decoders(&mut registry, decoders).expect("Kafka registry");
        let source = support::source(
            "kafka",
            serde_json::to_value(source_options("127.0.0.1:1")).unwrap(),
            Some("custom"),
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
        .expect("custom decoder graph compiles");
        assert_eq!(plan.source_binding_ids(), ["input"]);
        assert_eq!(plan.sink_binding_ids(), ["output"]);
    }
}
