use super::*;
use crate::tls_test_support::{bounded, listener, roots};

async fn tls_peer(listener: tokio::net::TcpListener, acceptor: tokio_rustls::TlsAcceptor) -> bool {
    let (tcp, _) = listener.accept().await.unwrap();
    match acceptor.accept(tcp).await {
        Ok(tls) => tokio_tungstenite::accept_async(tls).await.is_ok(),
        Err(_) => false,
    }
}

#[tokio::test]
async fn tls_default_rejects_untrusted_websocket_server() {
    let (listener, acceptor) = listener().await;
    let port = listener.local_addr().unwrap().port();
    let peer = tokio::spawn(tls_peer(listener, acceptor));
    let result = bounded(Box::pin(connect_websocket(
        &format!("wss://localhost:{port}"),
        false,
    )))
    .await;
    let accepted = bounded(peer).await.unwrap();
    assert!(result.is_err());
    assert!(!accepted);
}

#[tokio::test]
async fn tls_explicit_insecure_accepts_untrusted_websocket_server() {
    let (listener, acceptor) = listener().await;
    let port = listener.local_addr().unwrap().port();
    let peer = tokio::spawn(tls_peer(listener, acceptor));
    let result = bounded(Box::pin(connect_websocket(
        &format!("wss://localhost:{port}"),
        true,
    )))
    .await;
    let accepted = bounded(peer).await.unwrap();
    assert!(result.is_ok());
    assert!(accepted);
}

#[tokio::test]
async fn tls_trusted_websocket_ca_still_checks_hostname() {
    for (host, expected) in [("localhost", true), ("127.0.0.1", false)] {
        let (listener, acceptor) = listener().await;
        let port = listener.local_addr().unwrap().port();
        let peer = tokio::spawn(tls_peer(listener, acceptor));
        let tls = websocket_tls_config(roots(), false);
        let result = bounded(Box::pin(tokio_tungstenite::connect_async_tls_with_config(
            format!("wss://{host}:{port}"),
            None,
            false,
            Some(tokio_tungstenite::Connector::Rustls(Arc::new(tls))),
        )))
        .await;
        let accepted = bounded(peer).await.unwrap();
        assert_eq!(result.is_ok(), expected, "host={host}");
        assert_eq!(accepted, expected, "host={host}");
    }
}

#[tokio::test]
async fn tls_configuration_keeps_plain_websocket_support() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let peer = tokio::spawn(async move {
        let (tcp, _) = listener.accept().await.unwrap();
        tokio_tungstenite::accept_async(tcp).await.is_ok()
    });
    let result = bounded(Box::pin(connect_websocket(
        &format!("ws://localhost:{port}"),
        false,
    )))
    .await;
    let accepted = bounded(peer).await.unwrap();
    assert!(result.is_ok());
    assert!(accepted);
}
