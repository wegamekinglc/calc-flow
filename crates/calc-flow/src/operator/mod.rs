//! Operator metadata, the batch/stream trait split, and provider factories.

mod batch;
mod expression;
mod sql;
mod stream;
mod union;
mod window;

pub use batch::{BatchOperator, BatchOperatorContext};
pub use expression::ExpressionOperator;
pub use sql::SqlOperator;
pub use stream::{
    EdgeCollector, OperatorStateSnapshot, StreamCollector, StreamOperator, StreamOperatorContext,
};
pub use union::UnionOperator;
pub use window::{
    AggregateFunction, AggregateSpec, MAX_WINDOW_OVERLAP, WindowAggregateOperator, WindowGeometry,
    WindowSpec,
};

pub(crate) use expression::expression_query;
pub(crate) use stream::{LateMetricDelta, LateMetricSink, accumulate_late_metrics};

use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::Arc,
};

use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
use parking_lot::RwLock;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::{Value, json};

use crate::{
    Batch, BatchKind, CalcFlowError, DataFusionConfig, DataFusionRuntime, JsonMap, Result,
    UdfReference, UdfRegistrySnapshot, json::validate_portable_identifier,
};

/// Lazily built operator-scoped `DataFusion` resources for stream execution.
struct StreamRuntimeState {
    resources: Option<(DataFusionConfig, UdfRegistrySnapshot, Vec<UdfReference>)>,
    runtime: Option<DataFusionRuntime>,
}

impl StreamRuntimeState {
    const fn new() -> Self {
        Self {
            resources: None,
            runtime: None,
        }
    }

    fn set_resources(
        &mut self,
        config: DataFusionConfig,
        udfs: UdfRegistrySnapshot,
        selected: Vec<UdfReference>,
    ) {
        self.resources = Some((config, udfs, selected));
    }

    fn runtime(&mut self) -> Result<&DataFusionRuntime> {
        if self.runtime.is_none() {
            let (config, udfs, selected) = self.resources.clone().unwrap_or_else(|| {
                (
                    DataFusionConfig::default(),
                    UdfRegistrySnapshot::default(),
                    Vec::new(),
                )
            });
            let mut runtime = DataFusionRuntime::new(config)?;
            runtime.register_udfs(&udfs, &selected)?;
            self.runtime = Some(runtime);
        }
        Ok(self.runtime.as_ref().expect("runtime initialized above"))
    }

    const fn is_initialized(&self) -> bool {
        self.runtime.is_some()
    }
}

impl Clone for StreamRuntimeState {
    fn clone(&self) -> Self {
        Self {
            resources: self.resources.clone(),
            runtime: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Port {
    name: String,
    kind: BatchKind,
    required: bool,
    schema: Option<SchemaRef>,
}

impl Port {
    /// Creates a named, typed operator boundary.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the name is not a
    /// portable identifier or an array port declares an Arrow schema.
    pub fn new(
        name: &str,
        kind: BatchKind,
        required: bool,
        fields: Option<Vec<Field>>,
    ) -> Result<Self> {
        Self::with_schema_ref(
            name,
            kind,
            required,
            fields.map(|fields| Arc::new(Schema::new(fields))),
        )
    }

    pub(crate) fn with_schema_ref(
        name: &str,
        kind: BatchKind,
        required: bool,
        schema: Option<SchemaRef>,
    ) -> Result<Self> {
        if !is_identifier(name) {
            return Err(CalcFlowError::InvalidArgument {
                field: "port.name".into(),
                message: "must be a non-empty portable identifier".into(),
            });
        }
        if kind == BatchKind::Array && schema.is_some() {
            return Err(CalcFlowError::InvalidArgument {
                field: "port.schema".into(),
                message: "array ports cannot declare Arrow schemas".into(),
            });
        }
        Ok(Self {
            name: name.into(),
            kind,
            required,
            schema,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn kind(&self) -> BatchKind {
        self.kind
    }

    pub const fn required(&self) -> bool {
        self.required
    }

    pub const fn schema(&self) -> Option<&SchemaRef> {
        self.schema.as_ref()
    }

    /// Validates a batch against this port's kind and optional exact schema.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Compile`] when the kind or schema differs.
    pub fn validate(&self, batch: &Batch, endpoint: &str) -> Result<()> {
        if batch.kind() != self.kind {
            return Err(CalcFlowError::Compile {
                message: format!(
                    "{endpoint} expects a {:?} batch, received {:?}",
                    self.kind,
                    batch.kind()
                ),
            });
        }
        if let Some(expected) = &self.schema {
            let actual = batch.table_payload()?.schema();
            if actual != expected {
                return Err(CalcFlowError::Compile {
                    message: format!("{endpoint} schema mismatch"),
                });
            }
        }
        Ok(())
    }
}

fn is_identifier(value: &str) -> bool {
    let mut characters = value.chars();
    characters.next().is_some_and(|first| {
        (first == '_' || first.is_ascii_alphabetic())
            && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
    })
}

/// Metadata every compiled node exposes to the graph compiler (API note A1).
///
/// The supertrait guarantees the batch and stream compilers can never drift
/// on metadata semantics; method signatures are identical to the retired v2
/// `Operator` trait's.
pub trait OperatorMetadata: Send + Sync {
    fn name(&self) -> &str;
    fn input_ports(&self) -> &[Port];
    fn output_ports(&self) -> &[Port];
    fn configuration(&self) -> JsonMap;

    fn udf_references(&self) -> Vec<UdfReference> {
        Vec::new()
    }
}

/// One graph node's operator as accepted by [`crate::PipelineBuilder`].
///
/// `Expression`, `Sql`, `Union`, and `Window` are built-in operators. `Batch`
/// and `Stream` carry external operators through their respective trait
/// objects; `compile_batch` rejects stream-only nodes and `compile_stream`
/// rejects batch-only nodes (plan section 2.2).
pub enum NodeOperator {
    Expression(ExpressionOperator),
    Sql(SqlOperator),
    Union(UnionOperator),
    Window(WindowAggregateOperator),
    Batch(Box<dyn BatchOperator>),
    Stream(Box<dyn StreamOperator>),
}

impl NodeOperator {
    pub(crate) fn metadata(&self) -> &dyn OperatorMetadata {
        match self {
            Self::Expression(operator) => operator,
            Self::Sql(operator) => operator,
            Self::Union(operator) => operator,
            Self::Window(operator) => operator,
            Self::Batch(operator) => operator.as_ref(),
            Self::Stream(operator) => operator.as_ref(),
        }
    }

    pub(crate) fn input_ports(&self) -> &[Port] {
        self.metadata().input_ports()
    }

    pub(crate) fn output_ports(&self) -> &[Port] {
        self.metadata().output_ports()
    }

    pub(crate) fn configuration(&self) -> JsonMap {
        self.metadata().configuration()
    }

    pub(crate) fn udf_references(&self) -> Vec<UdfReference> {
        self.metadata().udf_references()
    }

    pub(crate) const fn requires_datafusion(&self) -> bool {
        matches!(self, Self::Expression(_) | Self::Sql(_))
    }
}

impl From<Box<ExpressionOperator>> for NodeOperator {
    fn from(value: Box<ExpressionOperator>) -> Self {
        Self::Expression(*value)
    }
}

impl From<Box<SqlOperator>> for NodeOperator {
    fn from(value: Box<SqlOperator>) -> Self {
        Self::Sql(*value)
    }
}

impl From<Box<UnionOperator>> for NodeOperator {
    fn from(value: Box<UnionOperator>) -> Self {
        Self::Union(*value)
    }
}

impl From<WindowAggregateOperator> for NodeOperator {
    fn from(value: WindowAggregateOperator) -> Self {
        Self::Window(value)
    }
}

impl From<Box<WindowAggregateOperator>> for NodeOperator {
    fn from(value: Box<WindowAggregateOperator>) -> Self {
        Self::Window(*value)
    }
}

impl From<Box<dyn BatchOperator>> for NodeOperator {
    fn from(value: Box<dyn BatchOperator>) -> Self {
        Self::Batch(value)
    }
}

impl From<Box<dyn StreamOperator>> for NodeOperator {
    fn from(value: Box<dyn StreamOperator>) -> Self {
        Self::Stream(value)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct ExternalOperatorSpec {
    provider: String,
    name: String,
    version: String,
    options: JsonMap,
}

impl ExternalOperatorSpec {
    /// Creates a validated, data-only external operator specification.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when an identity component
    /// is not a non-empty portable identifier.
    pub fn new(provider: &str, name: &str, version: &str, options: JsonMap) -> Result<Self> {
        validate_provider_identity(provider, name, version)?;
        Ok(Self {
            provider: provider.into(),
            name: name.into(),
            version: version.into(),
            options,
        })
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub const fn options(&self) -> &JsonMap {
        &self.options
    }
}

impl<'de> Deserialize<'de> for ExternalOperatorSpec {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            provider: String,
            name: String,
            version: String,
            options: JsonMap,
        }

        let fields = Fields::deserialize(deserializer)?;
        Self::new(
            &fields.provider,
            &fields.name,
            &fields.version,
            fields.options,
        )
        .map_err(D::Error::custom)
    }
}

/// Creates external batch operators for validated data-only specifications
/// (API note A1.3).
pub trait BatchOperatorFactory: Send + Sync {
    /// Creates an external batch operator for a validated specification.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider rejects the specification or ports.
    fn create(
        &self,
        spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Box<dyn BatchOperator>>;
}

/// Creates external stream operators for validated data-only specifications
/// (API note A1.3).
pub trait StreamOperatorFactory: Send + Sync {
    /// Creates an external stream operator for a validated specification.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider rejects the specification or ports.
    fn create(
        &self,
        spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Box<dyn StreamOperator>>;
}

type ProviderKey = (String, String, String);

#[derive(Default)]
pub struct ProviderRegistry {
    batch_factories: RwLock<BTreeMap<ProviderKey, Arc<dyn BatchOperatorFactory>>>,
    stream_factories: RwLock<BTreeMap<ProviderKey, Arc<dyn StreamOperatorFactory>>>,
}

impl ProviderRegistry {
    /// Registers a batch factory under an exact provider, name, and version.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when an identity component
    /// is invalid or the key is already registered. A rejected registration
    /// leaves the registry unchanged.
    pub fn register_batch(
        &self,
        provider: &str,
        name: &str,
        version: &str,
        factory: Arc<dyn BatchOperatorFactory>,
    ) -> Result<()> {
        validate_provider_identity(provider, name, version)?;
        let key = (provider.into(), name.into(), version.into());
        match self.batch_factories.write().entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(factory);
                Ok(())
            }
            Entry::Occupied(_) => Err(CalcFlowError::InvalidArgument {
                field: "provider".into(),
                message: "duplicate provider/name/version".into(),
            }),
        }
    }

    /// Resolves an exact external batch factory.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when an identity component
    /// is invalid, or [`CalcFlowError::Compile`] when the provider is
    /// unavailable.
    pub fn resolve_batch(
        &self,
        provider: &str,
        name: &str,
        version: &str,
    ) -> Result<Arc<dyn BatchOperatorFactory>> {
        validate_provider_identity(provider, name, version)?;
        self.batch_factories
            .read()
            .get(&(provider.into(), name.into(), version.into()))
            .cloned()
            .ok_or_else(|| CalcFlowError::Compile {
                message: format!("provider {provider}:{name}@{version} is unavailable"),
            })
    }

    /// Registers a stream factory under an exact provider, name, and version.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when an identity component
    /// is invalid or the key is already registered. A rejected registration
    /// leaves the registry unchanged.
    pub fn register_stream(
        &self,
        provider: &str,
        name: &str,
        version: &str,
        factory: Arc<dyn StreamOperatorFactory>,
    ) -> Result<()> {
        validate_provider_identity(provider, name, version)?;
        let key = (provider.into(), name.into(), version.into());
        match self.stream_factories.write().entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(factory);
                Ok(())
            }
            Entry::Occupied(_) => Err(CalcFlowError::InvalidArgument {
                field: "provider".into(),
                message: "duplicate provider/name/version".into(),
            }),
        }
    }

    /// Resolves an exact external stream factory.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when an identity component
    /// is invalid, or [`CalcFlowError::Compile`] when the provider offers no
    /// stream factory (plan section 2.2).
    pub fn resolve_stream(
        &self,
        provider: &str,
        name: &str,
        version: &str,
    ) -> Result<Arc<dyn StreamOperatorFactory>> {
        validate_provider_identity(provider, name, version)?;
        self.stream_factories
            .read()
            .get(&(provider.into(), name.into(), version.into()))
            .cloned()
            .ok_or_else(|| CalcFlowError::Compile {
                message: format!(
                    "provider {provider}:{name}@{version} offers no stream operator factory"
                ),
            })
    }
}

fn validate_provider_identity(provider: &str, name: &str, version: &str) -> Result<()> {
    for (field, value) in [("provider", provider), ("name", name), ("version", version)] {
        validate_portable_identifier(field, value)?;
    }
    Ok(())
}

pub(crate) fn validate_operator_name(name: &str) -> Result<()> {
    if name.trim().is_empty() {
        Err(CalcFlowError::InvalidArgument {
            field: "operator.name".into(),
            message: "must not be empty".into(),
        })
    } else {
        Ok(())
    }
}

pub(crate) fn validate_builtin_port(port: &Port, name: &str, field: &str) -> Result<()> {
    if port.name() != name || port.kind() != BatchKind::Table {
        return Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: format!("must contain the {name:?} table port"),
        });
    }
    Ok(())
}

pub(crate) fn validate_required_input(port: &Port, field: &str) -> Result<()> {
    if !port.required() {
        return Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: "built-in input ports must be required".into(),
        });
    }
    Ok(())
}

pub(crate) fn table_port(name: &str) -> Result<Port> {
    Port::new(name, BatchKind::Table, true, None)
}

pub(crate) fn udf_configuration(reference: &UdfReference) -> Value {
    json!({
        "provider": reference.provider(),
        "name": reference.name(),
        "version": reference.version(),
        "kind": reference.kind(),
    })
}
