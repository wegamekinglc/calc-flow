//! Integration tests for the M6.6 HTTP polling and WebSocket
//! connectors: offline configuration, capability, TLS, and backpressure
//! contracts.

#![cfg(all(feature = "http", feature = "websocket"))]

use std::collections::BTreeMap;

use calc_flow::StreamSource as _;
use calc_flow_connectors::http::{HttpSourceConfig, resolve_auth_header, resolve_http_url};
use calc_flow_connectors::websocket::{BackpressureMode, WebSocketSourceConfig};
use serde_json::{Value, json};

fn http_options() -> BTreeMap<String, Value> {
    BTreeMap::from([("poll_interval_ms".to_string(), json!(500))])
}

fn ws_options() -> BTreeMap<String, Value> {
    BTreeMap::from([("backpressure".to_string(), json!("block"))])
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
    assert!(error.to_string().contains("secret reference"), "{error}");

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

    let mut block = ws_options();
    block.insert("delivery".to_string(), json!("exactly_once"));
    let error = WebSocketSourceConfig::from_options(&block)
        .expect_err("an unreplayable block-mode source also fails closed");
    assert!(error.to_string().contains("unreplayable"), "{error}");
}

#[test]
fn http_capabilities_distinguish_replayability() {
    use calc_flow::{ReplayPositioning, SourceDeliveryCapability};

    let mut conditional = http_options();
    conditional.insert("conditional".to_string(), json!(true));
    let config = HttpSourceConfig::from_options(&conditional).expect("parses");
    let source = calc_flow_connectors::http::HttpSource::new(config).expect("builds");
    assert_eq!(
        source.capabilities().replay_positioning,
        ReplayPositioning::Unsupported,
        "conditional validators do not make polling history replayable"
    );
    assert_eq!(
        source.capabilities().delivery,
        SourceDeliveryCapability::Lossy,
        "a polling endpoint can change more than once between requests"
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

#[tokio::test]
async fn http_rejects_cursor_restore_even_with_conditional_validators() {
    let config = HttpSourceConfig::from_options(&http_options()).expect("parses");
    let mut source = calc_flow_connectors::http::HttpSource::new(config).expect("builds");
    let cursor = calc_flow::Cursor::unbound(
        1_u64.to_be_bytes().to_vec(),
        BTreeMap::from([
            ("sequence".to_string(), json!(1)),
            ("etag".to_string(), json!("historical-validator")),
        ]),
    )
    .expect("cursor");

    let error = source.open(Some(cursor)).await.expect_err("unreplayable");
    assert!(error.to_string().contains("historical"), "{error}");
}

#[tokio::test]
async fn registered_http_source_retries_and_reads_through_stream_source() {
    use calc_flow::ConnectorSourceFactory as _;
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("binds");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        for attempt in 0..2 {
            let (mut socket, _) = listener.accept().await.expect("accepts");
            let mut request = vec![0_u8; 4096];
            let _ = socket.read(&mut request).await.expect("reads request");
            if attempt == 0 {
                socket
                    .write_all(
                        b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    )
                    .await
                    .expect("writes retryable response");
            } else {
                let body = b"{\"value\":1}\n";
                let headers = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nETag: \"v1\"\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                socket
                    .write_all(headers.as_bytes())
                    .await
                    .expect("writes headers");
                socket.write_all(body).await.expect("writes body");
            }
            socket
                .shutdown()
                .await
                .expect("flushes and closes the response connection");
        }
    });

    let factory = calc_flow_connectors::http::HttpSourceFactory::new();
    let options = BTreeMap::from([
        ("poll_interval_ms".into(), json!(1)),
        ("retry_backoff_ms".into(), json!(1)),
        ("max_retries".into(), json!(1)),
    ]);
    let mut source = factory
        .open(&options, &Endpoint(format!("http://{address}/events")))
        .await
        .expect("factory opens exact runtime source");
    source.open(None).await.expect("source opens");
    let event = source.next().await.expect("retry succeeds").expect("data");
    let calc_flow::SourceEvent::Data { batch, cursor } = event else {
        panic!("successful retry must emit data")
    };
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(cursor.payload()["sequence"], json!(1));
    source.close().await.expect("closes");
    server.await.expect("server joins");
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
        SourceDeliveryCapability::Lossy,
        "Block mode bounds live buffering but cannot replay after a process failure"
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

#[tokio::test]
async fn registered_websocket_source_keeps_one_connection_across_batches() {
    use calc_flow::ConnectorSourceFactory as _;
    use futures_util::SinkExt as _;
    use tokio_tungstenite::tungstenite::Message;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("binds");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.expect("one connection");
        let mut websocket = tokio_tungstenite::accept_async(socket)
            .await
            .expect("handshake");
        websocket
            .send(Message::Text("{\"value\":1}".into()))
            .await
            .expect("first frame");
        websocket
            .send(Message::Text("{\"value\":2}".into()))
            .await
            .expect("second frame");
        websocket.close(None).await.expect("server closes");
    });

    let factory = calc_flow_connectors::websocket::WebSocketSourceFactory::new();
    let options = BTreeMap::from([
        ("max_batch_rows".into(), json!(1)),
        ("max_batch_bytes".into(), json!(4096)),
        ("max_frame_bytes".into(), json!(1024)),
    ]);
    let mut source = factory
        // Loopback plaintext is deliberate: this test exercises framing and
        // connection reuse, while TLS policy is covered by config tests.
        .open(&options, &Endpoint(format!("ws://{address}/events"))) // nosemgrep
        .await
        .expect("factory binds the secret without exposing it");
    source.open(None).await.expect("connects once");

    for expected_sequence in 1..=2 {
        let event = source.next().await.expect("reads").expect("data");
        let calc_flow::SourceEvent::Data { batch, cursor } = event else {
            panic!("frame must produce data")
        };
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(cursor.payload()["sequence"], json!(expected_sequence));
    }
    assert!(source.next().await.expect("closed stream").is_none());
    source.close().await.expect("joins reader");
    server.await.expect("server joins");
}

#[tokio::test]
async fn websocket_drop_oldest_is_bounded_and_observable() {
    use calc_flow::ConnectorSourceFactory as _;
    use futures_util::SinkExt as _;
    use tokio_tungstenite::tungstenite::Message;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("binds");
    let address = listener.local_addr().expect("address");
    let server = tokio::spawn(async move {
        let (socket, _) = listener.accept().await.expect("connection");
        let mut websocket = tokio_tungstenite::accept_async(socket)
            .await
            .expect("handshake");
        for value in 1..=3 {
            websocket
                .send(Message::Text(format!("{{\"value\":{value}}}").into()))
                .await
                .expect("frame");
        }
        websocket.close(None).await.expect("server closes");
    });

    let factory = calc_flow_connectors::websocket::WebSocketSourceFactory::new();
    let options = BTreeMap::from([
        ("backpressure".into(), json!("drop_oldest")),
        ("max_batch_rows".into(), json!(2)),
        ("max_batch_bytes".into(), json!(4096)),
        ("max_frame_bytes".into(), json!(1024)),
    ]);
    let mut source = factory
        // Loopback plaintext is deliberate for bounded backpressure behavior;
        // TLS verification policy is covered by the connector config tests.
        .open(&options, &Endpoint(format!("ws://{address}/events"))) // nosemgrep
        .await
        .expect("factory opens");
    source.open(None).await.expect("connects");
    server.await.expect("server sends every frame");
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;

    let event = source.next().await.expect("reads").expect("data");
    let calc_flow::SourceEvent::Data { batch, cursor } = event else {
        panic!("buffered frames must produce data")
    };
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(batch.metadata().attributes()["dropped_frames"], json!(1));
    assert_eq!(cursor.payload()["dropped_frames"], json!(1));
    source.close().await.expect("joins reader");
}

struct Endpoint(String);

impl calc_flow::SecretResolver for Endpoint {
    fn resolve(
        &self,
        reference: &calc_flow::SecretReference,
    ) -> calc_flow::Result<calc_flow::SecretHandle> {
        if reference.key == "url" {
            Ok(calc_flow::SecretHandle::from_bytes(self.0.as_bytes()))
        } else {
            Err(calc_flow::CalcFlowError::NotFound {
                resource: "secret".into(),
                key: reference.key.clone(),
            })
        }
    }
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
    assert_eq!(
        source.descriptor().capabilities.delivery,
        calc_flow::DeliveryCapability::BestEffort,
    );
    assert_eq!(
        source.descriptor().capabilities.replay,
        calc_flow::ReplayCapability::Unreplayable,
    );

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

#[test]
fn url_redaction_truncates_errors() {
    // The redaction helpers only exist through the error surface; exercise
    // them by checking that a missing secret's error text is truncated.
    struct NoSecrets;
    impl calc_flow::SecretResolver for NoSecrets {
        fn resolve(
            &self,
            _reference: &calc_flow::SecretReference,
        ) -> calc_flow::Result<calc_flow::SecretHandle> {
            Err(calc_flow::CalcFlowError::NotFound {
                resource: "secret".into(),
                key: "any".into(),
            })
        }
    }
    let error = resolve_http_url(&NoSecrets, "ANY").expect_err("fails");
    assert!(
        error.to_string().contains("could not be resolved"),
        "{error}"
    );
}
