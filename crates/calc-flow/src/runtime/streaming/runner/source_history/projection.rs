use crate::{
    Cursor, ExpressionOperator, OperatorMetadata, Result, SourceCapabilities, SourceEvent,
    SourceHistoryContext, SourceHistoryReplayFactory, SourceSchema, StreamSource,
};
use async_trait::async_trait;
use std::sync::Arc;

pub(super) fn factory(
    inner: Arc<dyn SourceHistoryReplayFactory>,
    steps: Vec<ExpressionOperator>,
) -> Arc<dyn SourceHistoryReplayFactory> {
    if steps.is_empty() {
        return inner;
    }
    Arc::new(Factory {
        inner,
        steps: steps.into(),
    })
}

struct Factory {
    inner: Arc<dyn SourceHistoryReplayFactory>,
    steps: Arc<[ExpressionOperator]>,
}

impl SourceHistoryReplayFactory for Factory {
    fn create(&self, history: SourceHistoryContext) -> Result<Box<dyn StreamSource>> {
        Ok(Box::new(Reader {
            inner: self.inner.create(history)?,
            steps: self.steps.clone(),
        }))
    }
}

struct Reader {
    inner: Box<dyn StreamSource>,
    steps: Arc<[ExpressionOperator]>,
}

#[async_trait]
impl StreamSource for Reader {
    fn capabilities(&self) -> SourceCapabilities {
        let mut capabilities = self.inner.capabilities();
        let last = self.steps.last().expect("projection chain is nonempty");
        capabilities.schema = SourceSchema::Exact(
            last.output_ports()[0]
                .schema()
                .expect("projection schema is proven")
                .clone(),
        );
        capabilities
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        self.inner.open(cursor).await
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        match self.inner.next().await? {
            Some(SourceEvent::Data { mut batch, cursor }) => {
                for step in self.steps.iter() {
                    batch = step.project_replay_batch(&batch)?;
                }
                Ok(Some(SourceEvent::Data { batch, cursor }))
            }
            event => Ok(event),
        }
    }

    async fn close(&mut self) -> Result<()> {
        self.inner.close().await
    }
}
