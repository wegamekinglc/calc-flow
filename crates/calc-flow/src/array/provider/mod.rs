use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;

use super::invalid;
use crate::{
    Batch, BatchOperator, BatchOperatorContext, BatchOperatorFactory, ExternalOperatorSpec,
    JsonMap, OperatorMetadata, Port, ProviderRegistry, Result,
};

mod budget;
mod execute;
mod infer;
mod ir;
mod schema;
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
