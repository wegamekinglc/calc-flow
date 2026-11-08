use super::super::worker::{self, Workers};
use super::settlement::Preparation;
use super::{
    build,
    record::{self, Template},
    retention::Mode,
    scope::{Consumer, NativeScope},
};
use crate::{
    Batch, OperatorMetadata, PortEndpoint, Result, RunContext, StaticInputSpec, StreamJobContext,
};
use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Default)]
pub(crate) struct NativeStaticPlan {
    templates: Vec<Template>,
    external: BTreeMap<String, PortEndpoint>,
    workers: Option<Arc<Workers>>,
    specs: Arc<BTreeMap<String, StaticInputSpec>>,
    consumers: Arc<Vec<Consumer>>,
    #[cfg(test)]
    probe: Arc<super::probe::Probe>,
}
impl NativeStaticPlan {
    #[cfg(test)]
    pub(crate) fn test_permits(&self) -> Arc<tokio::sync::Semaphore> {
        self.workers.as_ref().unwrap().test_permits()
    }
    #[cfg(test)]
    pub(crate) fn test_admission(&self) -> tokio::sync::oneshot::Receiver<()> {
        self.probe.observe_admission()
    }
    #[cfg(test)]
    pub(crate) fn test_work(
        &self,
        work: impl FnOnce(&mut dyn FnMut() -> Result<()>) -> Result<()> + Send + 'static,
    ) {
        self.probe.install(Box::new(work));
    }
    pub(crate) fn validate_bindings(&self) -> Result<()> {
        record::validate_required(&self.templates, &self.external, &self.specs)
    }
    pub(crate) fn needs_preparation(&self) -> bool {
        !self.specs.is_empty()
    }
    pub(crate) async fn prepare_stream(
        &self,
        pending: Preparation<Arc<BTreeMap<String, Batch>>>,
        context: &StreamJobContext,
        context_copies: usize,
    ) -> Result<Preparation<NativeScope>> {
        let specs = Arc::clone(&self.specs);
        let consumers = Arc::clone(&self.consumers);
        let mode = Mode::Managed {
            checkpointing: context.checkpointing(),
            context_copies,
        };
        let workers = self
            .workers
            .as_ref()
            .expect("native consumers retain their factory worker owner");
        #[cfg(test)]
        let probe = Arc::clone(&self.probe);
        let pending = pending.map(move |supplied| {
            move |check: &mut dyn FnMut() -> Result<()>| {
                #[cfg(test)]
                probe.run(check)?;
                build::prepare(&specs, &supplied, consumers, mode, check)
            }
        });
        #[cfg(test)]
        self.probe.entered();
        let scope = workers
            .run_preparation(worker::Scope::stream(context), "static_inputs", pending)
            .await?;
        self.validate_scope(scope.value())?;
        context.check_cancelled()?;
        Ok(scope)
    }
    pub(crate) fn set_endpoints(&mut self, external: &BTreeMap<String, PortEndpoint>) {
        self.external = external.clone();
    }
    pub(crate) fn push(&mut self, node: &str, operator: &dyn OperatorMetadata) {
        if let Some((template, workers)) = template(node, operator) {
            self.templates.push(template);
            if self.workers.is_none() {
                self.workers = Some(workers);
            }
        }
    }
    pub(crate) fn configure(
        &mut self,
        external: &BTreeMap<String, PortEndpoint>,
        specs: &BTreeMap<String, StaticInputSpec>,
    ) -> Result<()> {
        let consumers = record::bind(&self.templates, external, specs)?;
        self.set_endpoints(external);
        let names = consumers
            .iter()
            .flat_map(|consumer| consumer.ports.values().map(AsRef::as_ref))
            .collect::<BTreeSet<&str>>();
        self.specs = Arc::new(
            specs
                .iter()
                .filter(|(name, _)| names.contains(name.as_str()))
                .map(|(name, spec)| (name.clone(), spec.clone()))
                .collect(),
        );
        self.consumers = Arc::new(consumers);
        Ok(())
    }
    pub(crate) async fn prepare_batch(
        &self,
        supplied: Arc<BTreeMap<String, Batch>>,
        context: RunContext,
    ) -> Result<RunContext> {
        let _ = record::bind(&self.templates, &self.external, &self.specs)?;
        if self.specs.is_empty() {
            return Ok(context);
        }
        let specs = Arc::clone(&self.specs);
        let consumers = Arc::clone(&self.consumers);
        let workers = self
            .workers
            .as_ref()
            .expect("native consumers retain their factory worker owner");
        let scope = workers
            .run_scoped(
                worker::Scope::Batch(context.clone()),
                "static_inputs",
                move |check| build::prepare(&specs, &supplied, consumers, Mode::Finite, check),
            )
            .await?;
        self.validate_scope(&scope)?;
        context.check_cancelled()?;
        Ok(context.with_native_static_scope(Arc::new(scope)))
    }
    fn validate_scope(&self, scope: &NativeScope) -> Result<()> {
        for template in &self.templates {
            template.validate_scope(scope)?;
        }
        Ok(())
    }
}
fn template(node: &str, operator: &dyn OperatorMetadata) -> Option<(Template, Arc<Workers>)> {
    let operator = operator as &dyn Any;
    if let Some(operator) = operator.downcast_ref::<super::super::NativeOperator>() {
        return Some((
            Template::array(node, Arc::clone(&operator.program)),
            Arc::clone(&operator.workers),
        ));
    }
    super::super::table::template(node, operator)
}
