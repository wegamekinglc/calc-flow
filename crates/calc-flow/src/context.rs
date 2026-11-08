use std::{collections::BTreeMap, sync::Arc};

use chrono::{DateTime, Utc};
use serde_json::Value;
use tokio_util::sync::CancellationToken as TokioCancellationToken;
use uuid::Uuid;

use crate::{CalcFlowError, JsonMap, Result};

#[derive(Clone, Debug, Default)]
pub struct CancellationToken(TokioCancellationToken);

impl CancellationToken {
    pub fn new() -> Self {
        Self(TokioCancellationToken::new())
    }

    pub fn cancel(&self) {
        self.0.cancel();
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.is_cancelled()
    }

    /// Waits until cancellation is requested.
    #[allow(
        dead_code,
        reason = "the crate-private M2 runtime owns cancellation observation"
    )]
    pub(crate) async fn cancelled(&self) {
        self.0.cancelled().await;
    }
}

#[derive(Clone)]
pub struct RunContext {
    run_id: Arc<str>,
    node_id: Option<Arc<str>>,
    settings: Arc<JsonMap>,
    deadline: Option<DateTime<Utc>>,
    cancellation: CancellationToken,
    native_statics: Option<Arc<crate::array::NativeScope>>,
}

impl std::fmt::Debug for RunContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RunContext")
            .field("run_id", &self.run_id)
            .field("node_id", &self.node_id)
            .field("settings", &self.settings)
            .field("deadline", &self.deadline)
            .field("cancellation", &self.cancellation)
            .field("native_static_scope", &self.native_statics)
            .finish()
    }
}

impl RunContext {
    /// Creates a context for a new run.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the deadline does not
    /// use UTC.
    pub fn new(
        settings: BTreeMap<String, Value>,
        deadline: Option<DateTime<Utc>>,
        cancellation: CancellationToken,
    ) -> Result<Self> {
        if deadline.is_some_and(|value| value.timezone() != Utc) {
            return Err(CalcFlowError::InvalidArgument {
                field: "deadline".into(),
                message: "must use UTC".into(),
            });
        }
        Ok(Self {
            run_id: Uuid::new_v4().to_string().into(),
            node_id: None,
            settings: Arc::new(settings),
            deadline,
            cancellation,
            native_statics: None,
        })
    }

    /// Creates a context scoped to an operator node.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when `node_id` is empty or
    /// contains only whitespace.
    pub fn for_node(&self, node_id: &str) -> Result<Self> {
        if node_id.trim().is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "node_id".into(),
                message: "must not be empty".into(),
            });
        }
        let mut context = self.clone();
        context.node_id = Some(node_id.into());
        Ok(context)
    }

    pub(crate) fn with_native_static_scope(
        mut self,
        scope: Arc<crate::array::NativeScope>,
    ) -> Self {
        self.native_statics = Some(scope);
        self
    }
    pub(crate) fn native_static_inputs(
        &self,
        inputs: &BTreeMap<String, crate::Batch>,
    ) -> Result<(BTreeMap<String, crate::Batch>, usize)> {
        match (&self.native_statics, self.node_id()) {
            (Some(scope), Some(node)) => scope.resolve(node, inputs),
            _ => Ok((inputs.clone(), 0)),
        }
    }
    pub(crate) fn native_static_input(&self, port: &str) -> Option<&crate::Batch> {
        self.native_statics.as_ref()?.input(self.node_id()?, port)
    }

    /// Verifies that the run remains active.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Cancelled`] when cancellation was requested or
    /// the deadline has passed.
    pub fn check_cancelled(&self) -> Result<()> {
        if self.cancellation.is_cancelled()
            || self.deadline.is_some_and(|deadline| Utc::now() >= deadline)
        {
            return Err(CalcFlowError::Cancelled {
                run_id: self.run_id.to_string(),
            });
        }
        Ok(())
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    pub fn node_id(&self) -> Option<&str> {
        self.node_id.as_deref()
    }

    pub fn settings(&self) -> &JsonMap {
        &self.settings
    }

    /// Returns the absolute UTC deadline for this run, if configured.
    pub const fn deadline(&self) -> Option<&DateTime<Utc>> {
        self.deadline.as_ref()
    }
}
