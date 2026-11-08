use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;

use super::invalid;
use crate::{
    Batch, BatchOperator, BatchOperatorContext, BatchOperatorFactory, ExternalOperatorSpec,
    JsonMap, OperatorMetadata, Port, ProviderRegistry, Result,
};

mod budget;
mod document;
mod execute;
mod infer;
mod ir;
mod schema;
mod shape;
mod validate;
mod worker;

/// Factory for the bounded, batch-only `native:array_expr@1` provider.
///
/// Operators from one factory share two admitted blocking worker slots.
///
/// ```
/// use calc_flow::{ProviderRegistry, Result, array};
/// # fn main() -> Result<()> {
/// let registry = ProviderRegistry::default();
/// array::register_native(&registry)?;
/// assert!(registry.resolve_batch("native", "array_expr", "1").is_ok());
/// assert!(registry.resolve_stream("native", "array_expr", "1").is_err());
/// # Ok(())
/// # }
/// ```
#[derive(Default)]
pub struct NativeArrayFactory {
    workers: Arc<worker::Workers>,
}

impl NativeArrayFactory {
    /// Infers output descriptors for bounded descriptor-only IR1/IR2 queries.
    ///
    /// Query outputs contain only `name` and `node`. No input rows, numerical
    /// buffers, or provider registration are needed.
    ///
    /// ```
    /// use calc_flow::{JsonMap, array::NativeArrayFactory};
    /// # fn main() -> calc_flow::Result<()> {
    /// let query: JsonMap = serde_json::from_value(serde_json::json!({
    ///     "ir_version": 2,
    ///     "inputs": [{"name": "x", "dtype": "float32", "shape": [null, 2]}],
    ///     "nodes": [{"id": "x", "op": "input", "name": "x"},
    ///               {"id": "y", "op": "reduce", "function": "mean", "input": "x", "axis": 1}],
    ///     "outputs": [{"name": "output", "node": "y"}]
    /// })).expect("query is a JSON object");
    /// let inferred = NativeArrayFactory::infer_program(&query)?;
    /// assert_eq!(inferred["outputs"][0]["dtype"], "float64");
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::CalcFlowError::InvalidArgument`] for malformed programs,
    /// incompatible descriptors, or exceeded declaration bounds.
    pub fn infer_program(options: &JsonMap) -> Result<JsonMap> {
        let plan = infer::plan(validate::parse_query(options)?)?;
        infer::query_outputs(&plan)
    }

    /// Returns the data-only native provider capability and strict options schema.
    pub fn capability_descriptor() -> JsonMap {
        schema::descriptor()
    }
}

/// Registers the native array batch factory under its exact versioned identity.
///
/// # Errors
///
/// Returns [`crate::CalcFlowError::InvalidArgument`] on duplicate registration;
/// the existing registry entry remains unchanged.
pub fn register_native(registry: &ProviderRegistry) -> Result<()> {
    registry.register_batch(
        "native",
        "array_expr",
        "1",
        Arc::new(NativeArrayFactory::default()),
    )
}

impl BatchOperatorFactory for NativeArrayFactory {
    fn create(
        &self,
        spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Box<dyn BatchOperator>> {
        if (spec.provider(), spec.name(), spec.version()) != ("native", "array_expr", "1") {
            return Err(invalid("provider", "requires native:array_expr@1"));
        }
        let program = validate::parse(spec.options())?;
        validate::ports(
            &inputs,
            program.inputs.iter().map(|input| input.name.as_str()),
            "inputs",
        )?;
        validate::ports(
            &outputs,
            program.outputs.iter().map(|output| output.name.as_str()),
            "outputs",
        )?;
        Ok(Box::new(NativeOperator {
            program: Arc::new(infer::plan(program)?),
            workers: Arc::clone(&self.workers),
            inputs,
            outputs,
            options: spec.options().clone(),
        }))
    }
}

struct NativeOperator {
    program: Arc<infer::Plan>,
    workers: Arc<worker::Workers>,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    options: JsonMap,
}

impl OperatorMetadata for NativeOperator {
    fn name(&self) -> &'static str {
        "native:array_expr@1"
    }
    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }
    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }
    fn configuration(&self) -> JsonMap {
        BTreeMap::from([
            ("provider".into(), "native".into()),
            ("name".into(), "array_expr".into()),
            ("version".into(), "1".into()),
            (
                "options".into(),
                serde_json::to_value(&self.options).expect("JSON options are serializable"),
            ),
        ])
    }
}

#[async_trait]
impl BatchOperator for NativeOperator {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        context.run.check_cancelled()?;
        let prepared = execute::prepare(&self.program, inputs)?;
        let plan = Arc::clone(&self.program);
        let inputs = inputs.clone();
        let outputs = self
            .workers
            .run(context.run.clone(), move |check| {
                execute::run(&plan, &inputs, &prepared, check)
            })
            .await?;
        context.run.check_cancelled()?;
        Ok(outputs)
    }
}

#[cfg(test)]
mod monitor_tests;
