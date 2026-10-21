use super::*;
use crate::tls_test_support::{bounded, listener};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

fn config(port: u16, mode: SslMode) -> ReplicationConfig {
    let mut config = ReplicationConfig::new("localhost", "user", "password", "db", "slot", "pub");
    config.port = port;
    config.tls.mode = mode;
    config
}

async fn tls_peer(listener: tokio::net::TcpListener, acceptor: tokio_rustls::TlsAcceptor) -> bool {
    let (mut tcp, _) = listener.accept().await.unwrap();
    let mut request = [0; 8];
    tcp.read_exact(&mut request).await.unwrap();
    assert_eq!(request, [0, 0, 0, 8, 4, 210, 22, 47]);
    tcp.write_all(b"S").await.unwrap();
    match acceptor.accept(tcp).await {
        Ok(mut tls) => tls.write_all(b"ok").await.is_ok(),
        Err(_) => false,
    }
}

async fn reads_marker(config: &ReplicationConfig) -> bool {
    let Ok(mut stream) = connect_bootstrap_stream(config).await else {
        return false;
    };
    let mut marker = [0; 2];
    stream.read_exact(&mut marker).await.is_ok() && marker == *b"ok"
}

#[tokio::test]
async fn tls_bootstrap_preserves_ca_and_hostname_verification() {
    for (host, trust_ca, expected) in [
        ("localhost", true, true),
        ("localhost", false, false),
        ("127.0.0.1", true, false),
    ] {
        let (listener, acceptor) = listener().await;
        let mut config = config(listener.local_addr().unwrap().port(), SslMode::VerifyFull);
        config.host = host.into();
        if trust_ca {
            config.tls.ca_pem_path = Some(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tls/ca.pem"),
            );
        }
        let peer = tokio::spawn(tls_peer(listener, acceptor));
        let accepted = bounded(reads_marker(&config)).await;
        let peer_accepted = bounded(peer).await.unwrap();
        assert_eq!(accepted, expected, "host={host}, trust_ca={trust_ca}");
        assert_eq!(peer_accepted, expected, "host={host}, trust_ca={trust_ca}");
    }
}

#[tokio::test]
async fn tls_bootstrap_require_preserves_encryption_without_ca_verification() {
    let (listener, acceptor) = listener().await;
    let config = config(listener.local_addr().unwrap().port(), SslMode::Require);
    let peer = tokio::spawn(tls_peer(listener, acceptor));
    let accepted = bounded(reads_marker(&config)).await;
    let peer_accepted = bounded(peer).await.unwrap();
    assert!(accepted);
    assert!(peer_accepted);
}

#[tokio::test]
async fn tls_bootstrap_preserves_disable_prefer_and_required_negotiation() {
    for (mode, expected) in [
        (SslMode::Disable, true),
        (SslMode::Prefer, true),
        (SslMode::Require, false),
    ] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let config = config(listener.local_addr().unwrap().port(), mode);
        let peer = tokio::spawn(async move {
            let (mut tcp, _) = listener.accept().await.unwrap();
            if mode != SslMode::Disable {
                let mut request = [0; 8];
                tcp.read_exact(&mut request).await.unwrap();
                assert_eq!(request, [0, 0, 0, 8, 4, 210, 22, 47]);
                tcp.write_all(b"N").await.unwrap();
            }
            let _ = tcp.write_all(b"ok").await;
        });
        let accepted = bounded(reads_marker(&config)).await;
        bounded(peer).await.unwrap();
        assert_eq!(accepted, expected, "mode={mode:?}");
    }
}
