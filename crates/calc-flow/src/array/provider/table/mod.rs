use std::{collections::BTreeMap, sync::Arc};

use super::{infer, worker};
use crate::{
    Batch, BatchOperator, BatchOperatorContext, BatchOperatorFactory, ExternalOperatorSpec,
    JsonMap, OperatorMetadata, Port, Result, array::invalid,
};
use async_trait::async_trait;

mod attachment;
mod document;
mod lineage;
mod outlet;
mod pack;
mod plan;
mod rhs;
mod run;
mod source;

/// Factory for row-preserving native table/array programs.
#[derive(Default)]
pub struct NativeTableArrayFactory {
    workers: Arc<worker::Workers>,
}

struct TablePlan {
    numeric: Arc<infer::Plan>,
    table: String,
    columns: Vec<usize>,
    dtype: super::ir::DType,
    output: OutputMode,
}
enum OutputMode {
    Table(Vec<String>),
    Array,
}
struct TableOperator {
    plan: Arc<TablePlan>,
    workers: Arc<worker::Workers>,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    options: JsonMap,
}

pub(super) fn template(
    node: &str,
    operator: &dyn std::any::Any,
) -> Option<(super::preparation::record::Template, Arc<worker::Workers>)> {
    let operator = operator.downcast_ref::<TableOperator>()?;
    Some((
        super::preparation::record::Template::table(
            node,
            Arc::clone(&operator.plan.numeric),
            &operator.plan.table,
        ),
        Arc::clone(&operator.workers),
    ))
}

impl NativeTableArrayFactory {
    /// Returns the native table bridge capability and strict IR2 wrapper schema.
    ///
    /// Stream programs require proven table row lineage and stateless operations.
    ///
    /// ```
    /// use calc_flow::array::NativeTableArrayFactory;
    /// let capability = NativeTableArrayFactory::capability_descriptor();
    /// assert_eq!(capability["checkpoint_support"], "stateless");
    /// assert_eq!(capability["supports_static_inputs"], true);
    /// ```
    pub fn capability_descriptor() -> JsonMap {
        super::schema::table_descriptor()
    }
    pub(super) fn with_workers(workers: Arc<worker::Workers>) -> Self {
        Self { workers }
    }
    fn operator(
        &self,
        spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<TableOperator> {
        if (spec.provider(), spec.name(), spec.version()) != ("native", "table_array", "1") {
            return Err(invalid("provider", "requires native:table_array@1"));
        }
        let plan = plan::build(spec.options(), &inputs, &outputs)?;
        Ok(TableOperator {
            plan: Arc::new(plan),
            workers: Arc::clone(&self.workers),
            inputs,
            outputs,
            options: spec.options().clone(),
        })
    }
}
impl BatchOperatorFactory for NativeTableArrayFactory {
    fn create(
        &self,
        spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Box<dyn BatchOperator>> {
        Ok(Box::new(self.operator(spec, inputs, outputs)?))
    }
}
impl crate::StreamOperatorFactory for NativeTableArrayFactory {
    fn create(
        &self,
        spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Box<dyn crate::StreamOperator>> {
        let operator = self.operator(spec, inputs, outputs)?;
        lineage::stream(&operator.plan.numeric)?;
        Ok(Box::new(operator))
    }
}
impl OperatorMetadata for TableOperator {
    fn name(&self) -> &'static str {
        "native:table_array@1"
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
            ("name".into(), "table_array".into()),
            ("version".into(), "1".into()),
            (
                "options".into(),
                serde_json::to_value(&self.options).expect("JSON options serialize"),
            ),
        ])
    }
}
fn stream_inputs(
    operator: &TableOperator,
    context: &crate::StreamOperatorContext<'_>,
    batch: Batch,
) -> Result<(BTreeMap<String, Batch>, usize)> {
    require_static(&operator.plan, |port| context.native_static_input(port))?;
    let supplied = BTreeMap::from([(operator.plan.table.clone(), batch)]);
    let (inputs, reserved) = context.native_static_inputs(&supplied)?;
    validate_inputs(&operator.inputs, &inputs)?;
    Ok((inputs, reserved))
}
async fn stream_run(
    workers: &worker::Workers,
    job: &crate::StreamJobContext,
    plan: Arc<TablePlan>,
    inputs: BTreeMap<String, Batch>,
    reserved: usize,
) -> Result<BTreeMap<String, Batch>> {
    workers
        .run_stream(job, move |check| {
            run::execute(&plan, &inputs, reserved, check)
                .map_err(|error| super::execute::reserved_error(error, reserved))
        })
        .await
}
fn require_static<'a>(plan: &TablePlan, input: impl Fn(&str) -> Option<&'a Batch>) -> Result<()> {
    for port in plan
        .numeric
        .program
        .inputs
        .iter()
        .filter(|port| port.name != plan.table)
    {
        if input(&port.name).is_none() {
            return Err(invalid(
                &format!("inputs.{}.binding", port.name),
                "requires a prepared native static input",
            ));
        }
    }
    Ok(())
}
fn validate_inputs(ports: &[Port], inputs: &BTreeMap<String, Batch>) -> Result<()> {
    if inputs.len() != ports.len() {
        return Err(invalid("inputs", "input names differ from declarations"));
    }
    for port in ports {
        let batch = inputs
            .get(port.name())
            .ok_or_else(|| invalid("inputs", "input names differ from declarations"))?;
        port.validate(batch, &format!("inputs.{}", port.name()))?;
    }
    Ok(())
}

#[async_trait]
impl BatchOperator for TableOperator {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        context.run.check_cancelled()?;
        require_static(&self.plan, |port| context.run.native_static_input(port))?;
        let (inputs, reserved) = context.run.native_static_inputs(inputs)?;
        validate_inputs(&self.inputs, &inputs)?;
        let plan = Arc::clone(&self.plan);
        self.workers
            .run_table(context.run.clone(), move |check| {
                run::execute(&plan, &inputs, reserved, check)
                    .map_err(|error| super::execute::reserved_error(error, reserved))
            })
            .await
    }
}

#[async_trait]
impl crate::StreamOperator for TableOperator {
    fn lifecycle(&self) -> crate::StreamOperatorLifecycle {
        crate::StreamOperatorLifecycle::Stateless {
            microbatch_invariant: true,
            deterministic: true,
            replay_safe: true,
        }
    }
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &crate::StreamOperatorContext<'_>,
        output: &mut dyn crate::StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()?;
        if ingress != self.plan.table {
            return Err(invalid(
                "ingress",
                "only the table port accepts stream data",
            ));
        }
        let (inputs, reserved) = stream_inputs(self, context, batch)?;
        let plan = Arc::clone(&self.plan);
        let results = stream_run(&self.workers, context.job(), plan, inputs, reserved).await?;
        context.check_cancelled()?;
        let (name, batch) = results
            .into_iter()
            .next()
            .expect("table programs have exactly one output");
        output.emit(&name, batch).await
    }
    async fn on_watermark(
        &mut self,
        _watermark: crate::EventTime,
        context: &crate::StreamOperatorContext<'_>,
        _output: &mut dyn crate::StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()
    }
    async fn on_end(
        &mut self,
        context: &crate::StreamOperatorContext<'_>,
        _output: &mut dyn crate::StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()
    }
}

#[cfg(test)]
mod tests;
