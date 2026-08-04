use std::collections::BTreeMap;

use async_trait::async_trait;
use serde_json::Value;

use crate::{Batch, CalcFlowError, Result, RunContext};

use super::OperatorMetadata;

/// The execution context a batch run hands to one operator invocation.
pub struct BatchOperatorContext<'a> {
    pub run: &'a RunContext,
}

/// A finite-input operator: one call maps a complete, immutable input map to
/// a new output map (plan section 2.2).
///
/// The five metadata accessors live on the [`OperatorMetadata`] supertrait so
/// the batch and stream compilers can never drift on port, schema, or UDF
/// validation (API note A1).
#[async_trait]
pub trait BatchOperator: OperatorMetadata {
    /// Processes borrowed inputs into a new output map.
    ///
    /// # Errors
    ///
    /// Returns an error when input validation, cancellation, or calculation
    /// fails.
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>>;

    /// Captures JSON-compatible operator state.
    ///
    /// # Errors
    ///
    /// Stateful implementations may reject state that cannot be captured.
    fn snapshot(&self) -> Result<Value> {
        Ok(Value::Null)
    }

    /// Restores JSON-compatible operator state.
    ///
    /// # Errors
    ///
    /// The default stateless lifecycle rejects non-null state.
    fn restore(&mut self, state: &Value) -> Result<()> {
        if state.is_null() {
            Ok(())
        } else {
            Err(CalcFlowError::Format {
                message: "stateless operator state must be null".into(),
            })
        }
    }

    /// Resets operator state.
    ///
    /// # Errors
    ///
    /// Stateful implementations may fail while releasing or recreating their
    /// owned state.
    fn reset(&mut self) -> Result<()> {
        self.restore(&Value::Null)
    }
}
