//! Capacity-one adapters for timing a public runner with one outstanding input.

use async_trait::async_trait;
use calc_flow::{
    Batch, CalcFlowError, Cursor, JsonMap, Result, SourceCapabilities, SourceEvent, StreamSink,
};

pub(super) struct RunnerBenchSource {
    capabilities: SourceCapabilities,
    input: tokio::sync::mpsc::Receiver<Batch>,
    sequence: u64,
}

impl RunnerBenchSource {
    pub(super) fn channel(
        capabilities: SourceCapabilities,
    ) -> (tokio::sync::mpsc::Sender<Batch>, Self) {
        let (sender, input) = tokio::sync::mpsc::channel(1);
        (
            sender,
            Self {
                input,
                sequence: 0,
                capabilities,
            },
        )
    }
}

#[async_trait]
impl calc_flow::StreamSource for RunnerBenchSource {
    fn capabilities(&self) -> SourceCapabilities {
        self.capabilities.clone()
    }

    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let Some(batch) = self.input.recv().await else {
            return Ok(None);
        };
        let cursor = Cursor::unbound(self.sequence.to_be_bytes().to_vec(), JsonMap::new())?;
        self.sequence = self
            .sequence
            .checked_add(1)
            .expect("benchmark source sequence exhausted");
        Ok(Some(SourceEvent::Data { batch, cursor }))
    }

    async fn close(&mut self) -> Result<()> {
        self.input.close();
        Ok(())
    }
}

pub(super) struct RunnerBenchSink {
    output: tokio::sync::mpsc::Sender<Batch>,
}

impl RunnerBenchSink {
    pub(super) fn channel() -> (Self, tokio::sync::mpsc::Receiver<Batch>) {
        let (output, receiver) = tokio::sync::mpsc::channel(1);
        (Self { output }, receiver)
    }
}

#[async_trait]
impl StreamSink for RunnerBenchSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.output
            .send(batch.clone())
            .await
            .map_err(|_| CalcFlowError::Internal {
                message: "benchmark output receiver closed".into(),
            })
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
