//! Integration tests for the M6.6 HTTP polling and WebSocket
//! connectors: offline configuration, capability, TLS, and backpressure
//! contracts.

#![cfg(feature = "http-websocket")]

use std::collections::BTreeMap;

use calc_flow::StreamSource as _;
use calc_flow_connectors::http::{HttpSourceConfig, resolve_auth_header, resolve_http_url};
use calc_flow_connectors::websocket::{BackpressureMode, WebSocketSourceConfig};
use serde_json::{Value, json};

fn http_options() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("url_key".to_string(), json!("HTTP_TEST_URL")),
        ("poll_interval_ms".to_string(), json!(500)),
    ])
}

fn ws_options() -> BTreeMap<String, Value> {
    BTreeMap::from([
        ("url_key".to_string(), json!("WS_TEST_URL")),
        ("backpressure".to_string(), json!("block")),
    ])
}

#[test]
fn http_config_parses_and_defaults_tls_on() {
    let config = HttpSourceConfig::from_options(&http_options()).expect("parses");
    assert!(!config.insecure, "TLS verification defaults to on");
    assert!(config.conditional, "conditional requests default to on");
    assert_eq!(config.poll_interval, std::time::Duration::from_millis(500));

    let mut insecure = http_options();
    insecure.insert("insecure".to_string(), json!(true));
    let config = HttpSourceConfig::from_options(&insecure).expect("insecure parses");
    assert!(config.insecure, "explicit insecure accepted");

    let mut bad_key = http_options();
    bad_key.insert("url_key".to_string(), json!(42));
    let error = HttpSourceConfig::from_options(&bad_key).expect_err("key must be string");
    assert!(error.to_string().contains("url_key"), "{error}");

    let mut bad_bound = http_options();
    bad_bound.insert("max_response_bytes".to_string(), json!("large"));
    let error = HttpSourceConfig::from_options(&bad_bound).expect_err("bound type");
    assert!(error.to_string().contains("max_response_bytes"), "{error}");
}

#[test]
fn http_url_and_auth_only_from_secrets() {
    struct OneUrl;
    impl calc_flow::SecretResolver for OneUrl {
        fn resolve(
            &self,
            reference: &calc_flow::SecretReference,
        ) -> calc_flow::Result<calc_flow::SecretHandle> {
            match reference.key.as_str() {
                "HTTP_URL" => Ok(calc_flow::SecretHandle::from_bytes(
                    b"https://user:pass@api.example.com/data",
                )),
                "HTTP_AUTH" => Ok(calc_flow::SecretHandle::from_bytes(b"Bearer secret-token")),
                _ => Err(calc_flow::CalcFlowError::NotFound {
                    resource: "secret".into(),
                    key: reference.key.clone(),
                }),
            }
        }
    }

    let url = resolve_http_url(&OneUrl, "HTTP_URL").expect("resolves");
    assert!(url.starts_with("https://"), "{url}");

    let auth = resolve_auth_header(&OneUrl, "HTTP_AUTH")
        .expect("auth resolves")
        .expect("auth present");
    assert!(auth.starts_with("Bearer "), "{auth}");

    let error = resolve_http_url(&OneUrl, "WRONG").expect_err("missing key");
    assert!(
        error.to_string().contains("could not be resolved"),
        "{error}"
    );
    assert!(
        !error.to_string().contains("pass") && !error.to_string().contains("secret-token"),
        "credentials never enter errors"
    );

    let none = resolve_auth_header(&OneUrl, "ABSENT").expect("absent auth is optional");
    assert!(none.is_none());
}

#[test]
fn websocket_config_parses_backpressure_modes() {
    let config = WebSocketSourceConfig::from_options(&ws_options()).expect("parses");
    assert_eq!(config.backpressure, BackpressureMode::Block);
    assert!(!config.insecure, "TLS verification defaults to on");

    let mut drop = ws_options();
    drop.insert("backpressure".to_string(), json!("drop_oldest"));
    let config = WebSocketSourceConfig::from_options(&drop).expect("drop parses");
    assert_eq!(config.backpressure, BackpressureMode::DropOldest);

    let mut bad = ws_options();
    bad.insert("backpressure".to_string(), json!("discard"));
    let error = WebSocketSourceConfig::from_options(&bad).expect_err("mode vocabulary");
    assert!(error.to_string().contains("backpressure"), "{error}");
}

#[test]
fn drop_oldest_rejects_exactly_once() {
    let mut conflict = ws_options();
    conflict.insert("backpressure".to_string(), json!("drop_oldest"));
    conflict.insert("delivery".to_string(), json!("exactly_once"));
    let error = WebSocketSourceConfig::from_options(&conflict)
        .expect_err("drop_oldest + exactly_once fails closed");
    assert!(error.to_string().contains("incompatible"), "{error}");

    // Block mode coexists with exactly-once.
    let mut compatible = ws_options();
    compatible.insert("delivery".to_string(), json!("exactly_once"));
    let config = WebSocketSourceConfig::from_options(&compatible).expect("block + exactly_once ok");
    assert_eq!(config.backpressure, BackpressureMode::Block);
}

#[test]
fn http_capabilities_distinguish_replayability() {
    use calc_flow::ReplayPositioning;

    let mut conditional = http_options();
    conditional.insert("conditional".to_string(), json!(true));
    let config = HttpSourceConfig::from_options(&conditional).expect("parses");
    let source = calc_flow_connectors::http::HttpSource::new(config).expect("builds");
    assert_eq!(
        source.capabilities().replay_positioning,
        ReplayPositioning::ExactPauseReportAndSeek,
        "conditional HTTP is replayable through ETag cursors"
    );
    assert_eq!(
        source.capabilities().delivery,
        calc_flow::SourceDeliveryCapability::Lossless
    );

    let mut unconditional = http_options();
    unconditional.insert("conditional".to_string(), json!(false));
    let config = HttpSourceConfig::from_options(&unconditional).expect("parses");
    let source = calc_flow_connectors::http::HttpSource::new(config).expect("builds");
    assert_eq!(
        source.capabilities().replay_positioning,
        ReplayPositioning::Unsupported,
        "unconditional HTTP is unreplayable"
    );
}

#[test]
fn websocket_capabilities_distinguish_lossiness() {
    use calc_flow::{ReplayPositioning, SourceDeliveryCapability};

    let config = WebSocketSourceConfig::from_options(&ws_options()).expect("block parses");
    let source = calc_flow_connectors::websocket::WebSocketSource::new(config).expect("builds");
    assert_eq!(
        source.capabilities().replay_positioning,
        ReplayPositioning::Unsupported,
        "WebSocket is always unreplayable"
    );
    assert_eq!(
        source.capabilities().delivery,
        SourceDeliveryCapability::Lossless,
        "Block mode pauses reads and loses nothing"
    );

    let mut drop = ws_options();
    drop.insert("backpressure".to_string(), json!("drop_oldest"));
    let config = WebSocketSourceConfig::from_options(&drop).expect("drop parses");
    let source = calc_flow_connectors::websocket::WebSocketSource::new(config).expect("builds");
    assert_eq!(
        source.capabilities().delivery,
        SourceDeliveryCapability::Lossy,
        "DropOldest mode is lossy and observable"
    );
}

#[test]
fn factories_register_through_the_trusted_registry() {
    let mut registry = calc_flow::ConnectorRegistry::new();
    calc_flow_connectors::http::register_http_connectors(&mut registry).expect("http registers");
    calc_flow_connectors::websocket::register_websocket_connectors(&mut registry)
        .expect("websocket registers");
    let snapshot = registry.snapshot();
    let names: Vec<String> = snapshot
        .identities()
        .iter()
        .map(|identity| identity.name.to_string())
        .collect();
    assert!(names.contains(&"http".to_string()), "{names:?}");
    assert!(names.contains(&"websocket".to_string()), "{names:?}");

    let http_identity = calc_flow::ConnectorIdentity::new(
        "calc-flow-connectors",
        "http",
        calc_flow_connectors::http::IDENTITY_VERSION,
    )
    .expect("identity");
    let source = snapshot.resolve_source(&http_identity).expect("resolves");
    assert!(!source.descriptor().capabilities.snapshot);
    assert!(source.descriptor().capabilities.polling);

    let ws_identity = calc_flow::ConnectorIdentity::new(
        "calc-flow-connectors",
        "websocket",
        calc_flow_connectors::websocket::IDENTITY_VERSION,
    )
    .expect("identity");
    let source = snapshot.resolve_source(&ws_identity).expect("resolves");
    assert_eq!(
        source.descriptor().capabilities.delivery,
        calc_flow::DeliveryCapability::BestEffort,
        "the WebSocket descriptor declares best-effort delivery"
    );

    let error =
        calc_flow_connectors::http::register_http_connectors(&mut registry).expect_err("occupied");
    assert!(matches!(error, calc_flow::CalcFlowError::Conflict { .. }));
}
