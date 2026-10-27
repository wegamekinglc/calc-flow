//! Rust equivalent of `examples/services/websocket_orders.py` (Ctrl-C to stop).
use futures_util::{SinkExt, StreamExt};
use std::{error::Error, path::PathBuf};
use tokio::net::TcpListener;
use tokio_tungstenite::{accept_async, tungstenite::Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/data/orders.jsonl");
    let payload = tokio::fs::read_to_string(path).await?;
    let listener = TcpListener::bind("127.0.0.1:8765").await?;
    println!("Sample orders: ws://127.0.0.1:8765 (Ctrl-C to stop)");
    loop {
        let (socket, _) = listener.accept().await?;
        let payload = payload.clone();
        tokio::spawn(async move {
            let result = async {
                let mut socket = accept_async(socket).await?;
                for line in payload.lines() {
                    socket.send(Message::Text(line.into())).await?;
                }
                while let Some(message) = socket.next().await {
                    if matches!(message?, Message::Close(_)) {
                        break;
                    }
                }
                Ok::<_, tokio_tungstenite::tungstenite::Error>(())
            }
            .await;
            if let Err(error) = result {
                eprintln!("WebSocket client: {error}");
            }
        });
    }
}
