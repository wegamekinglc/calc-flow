mod micro_batch;
mod streaming;

use std::collections::BTreeMap;

use crate::{CalcFlowError, Result, RunResult, Sink};

pub use micro_batch::MicroBatchRunner;
pub use streaming::StreamingRunner;

/// Ordered sink lists keyed by external graph output name.
#[derive(Default)]
pub struct SinkRouter {
    routes: BTreeMap<String, Vec<Box<dyn Sink>>>,
}

impl SinkRouter {
    /// Creates an empty router.
    pub const fn new() -> Self {
        Self {
            routes: BTreeMap::new(),
        }
    }

    /// Appends a sink to an output's delivery order.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for an empty output name.
    pub fn add(&mut self, output: &str, sink: Box<dyn Sink>) -> Result<()> {
        if output.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "sink.output".into(),
                message: "must not be empty".into(),
            });
        }
        self.routes.entry(output.into()).or_default().push(sink);
        Ok(())
    }

    /// Writes configured outputs in sorted-output and insertion order.
    ///
    /// All route names are validated before the first write and delivery stops
    /// on the first sink error.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for an unknown routed output,
    /// or the first error returned by a sink.
    pub async fn write_all(&mut self, result: &RunResult) -> Result<()> {
        if let Some(unknown) = self
            .routes
            .keys()
            .find(|output| !result.outputs.contains_key(*output))
        {
            return Err(CalcFlowError::InvalidArgument {
                field: "sinks".into(),
                message: format!("sink configured for unknown graph output {unknown:?}"),
            });
        }
        for (output, sinks) in &mut self.routes {
            let batch = &result.outputs[output];
            for sink in sinks {
                sink.write(batch, result.context()).await?;
            }
        }
        Ok(())
    }
}
