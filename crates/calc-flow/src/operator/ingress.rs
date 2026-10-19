//! Read-only ingress facts. No dictionary, identity lease, or state is shared
//! with the producer. Only the operator consumes facts and publishes changes.

use tokio::sync::{Mutex, mpsc};

use super::{asof::admission::IngressFacts as AsofFacts, join::admission_masks::PureMasksChunk};
use crate::{CalcFlowError, Result};

pub(crate) enum Sender {
    Join(mpsc::Sender<Result<PureMasksChunk>>),
    Asof(mpsc::Sender<Result<AsofFacts>>),
}

impl Sender {
    pub(crate) async fn error(&self, error: CalcFlowError) {
        match self {
            Self::Join(sender) => {
                let _ = sender.send(Err(error)).await;
            }
            Self::Asof(sender) => {
                let _ = sender.send(Err(error)).await;
            }
        }
    }
}

pub(crate) enum Preparation {
    Join(Mutex<mpsc::Receiver<Result<PureMasksChunk>>>),
    Asof {
        receiver: Mutex<mpsc::Receiver<Result<AsofFacts>>>,
        facts: std::sync::OnceLock<AsofFacts>,
    },
}

impl Preparation {
    pub(crate) fn join_channel() -> (Sender, Self) {
        // Producer, queue, and owner each hold at most one 16-mask pack.
        let (sender, receiver) = mpsc::channel(1);
        (Sender::Join(sender), Self::Join(Mutex::new(receiver)))
    }

    pub(crate) fn asof_channel() -> (Sender, Self) {
        let (sender, receiver) = mpsc::channel(1);
        (
            Sender::Asof(sender),
            Self::Asof {
                receiver: Mutex::new(receiver),
                facts: std::sync::OnceLock::new(),
            },
        )
    }

    pub(crate) async fn join(&self) -> Result<PureMasksChunk> {
        match self {
            Self::Join(receiver) => receive(receiver).await,
            Self::Asof { .. } => Err(kind_mismatch()),
        }
    }

    pub(crate) async fn asof(&self) -> Result<AsofFacts> {
        match self {
            Self::Asof { receiver, facts } => {
                // Replay pressure can retry admission with the same context.
                // Immutable facts belong to the envelope, not to one attempt.
                if let Some(prepared) = facts.get() {
                    return Ok(*prepared);
                }
                let prepared = receive(receiver).await?;
                let _ = facts.set(prepared);
                Ok(prepared)
            }
            Self::Join(_) => Err(kind_mismatch()),
        }
    }

    pub(crate) fn asof_order(&self, watermark: Option<i64>) -> Option<bool> {
        let Self::Asof { facts, .. } = self else {
            return None;
        };
        facts
            .get()
            .filter(|prepared| prepared.watermark() == watermark)
            .and_then(|prepared| prepared.ordered)
    }
}

async fn receive<T>(receiver: &Mutex<mpsc::Receiver<Result<T>>>) -> Result<T> {
    receiver.lock().await.recv().await.unwrap_or_else(|| {
        Err(CalcFlowError::Internal {
            message: "ingress admission producer closed before its facts arrived".into(),
        })
    })
}

fn kind_mismatch() -> CalcFlowError {
    CalcFlowError::Internal {
        message: "ingress admission facts do not match their operator".into(),
    }
}
