use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

pub(crate) fn roots() -> rustls::RootCertStore {
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(CertificateDer::from(
            include_bytes!("../tests/fixtures/tls/ca.der").to_vec(),
        ))
        .unwrap();
    roots
}

pub(crate) async fn listener() -> (TcpListener, TlsAcceptor) {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let cert = CertificateDer::from(include_bytes!("../tests/fixtures/tls/server.der").to_vec());
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
        include_bytes!("../tests/fixtures/tls/server-key.der").to_vec(),
    ));
    let server = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert], key)
        .unwrap();
    (
        TcpListener::bind("127.0.0.1:0").await.unwrap(),
        TlsAcceptor::from(Arc::new(server)),
    )
}

pub(crate) async fn bounded<F: Future>(future: F) -> F::Output {
    tokio::time::timeout(Duration::from_secs(10), future)
        .await
        .expect("local TLS exchange timed out")
}
