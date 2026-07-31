use std::{
    collections::{BTreeMap, BTreeSet, btree_map::Entry},
    sync::Arc,
};

use async_trait::async_trait;
use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
use parking_lot::RwLock;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::{Value, json};

use crate::{
    Batch, BatchKind, CalcFlowError, DataFusionRuntime, JsonMap, Result, RunContext, UdfReference,
    expression::{sql_projection, validate_select_query},
    json::validate_portable_identifier,
};

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
        if !is_identifier(name) {
            return Err(CalcFlowError::InvalidArgument {
                field: "port.name".into(),
                message: "must be a non-empty portable identifier".into(),
            });
        }
        if kind == BatchKind::Array && fields.is_some() {
            return Err(CalcFlowError::InvalidArgument {
                field: "port.schema".into(),
                message: "array ports cannot declare Arrow schemas".into(),
            });
        }
        Ok(Self {
            name: name.into(),
            kind,
            required,
            schema: fields.map(|fields| Arc::new(Schema::new(fields))),
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

pub struct OperatorContext<'a> {
    pub run: &'a RunContext,
}

#[async_trait]
pub trait Operator: Send + Sync {
    fn name(&self) -> &str;
    fn input_ports(&self) -> &[Port];
    fn output_ports(&self) -> &[Port];
    fn configuration(&self) -> JsonMap;

    fn udf_references(&self) -> Vec<UdfReference> {
        Vec::new()
    }

    /// Processes borrowed inputs into a new output map.
    ///
    /// # Errors
    ///
    /// Returns an error when input validation, cancellation, or calculation
    /// fails.
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &OperatorContext<'_>,
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

#[derive(Clone, Debug)]
pub struct ExpressionOperator {
    name: String,
    expression: Option<String>,
    select: Vec<String>,
    filter_expression: Option<String>,
    query: String,
    udfs: Vec<UdfReference>,
    input_ports: [Port; 1],
    output_ports: [Port; 1],
}

impl ExpressionOperator {
    /// Creates a `DataFusion` expression or projection operator.
    ///
    /// A non-empty `expression` and a non-empty `select` list are the two
    /// calculation modes. Exactly one must be supplied.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the name is empty,
    /// exactly one calculation mode is not supplied, a projection is empty,
    /// or the generated read-only query is malformed.
    pub fn new(
        name: &str,
        expression: &str,
        select: Vec<String>,
        filter_expression: Option<String>,
        udfs: Vec<UdfReference>,
    ) -> Result<Self> {
        validate_operator_name(name)?;
        let has_expression = !expression.trim().is_empty();
        let has_select = !select.is_empty();
        if has_expression == has_select {
            return Err(CalcFlowError::InvalidArgument {
                field: "operator.calculation".into(),
                message: "provide exactly one expression or non-empty select list".into(),
            });
        }
        if select.iter().any(|projection| projection.trim().is_empty()) {
            return Err(CalcFlowError::InvalidArgument {
                field: "operator.select".into(),
                message: "projection expressions must not be empty".into(),
            });
        }
        let expression = has_expression.then(|| expression.into());
        let query = expression_query(expression.as_deref(), &select, filter_expression.as_deref())?;
        Ok(Self {
            name: name.into(),
            expression,
            select,
            filter_expression,
            query,
            udfs,
            input_ports: [table_port("input")?],
            output_ports: [table_port("output")?],
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn input_ports(&self) -> &[Port] {
        &self.input_ports
    }

    pub const fn output_ports(&self) -> &[Port] {
        &self.output_ports
    }

    /// Returns this operator with exact configuration-defined table ports.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] unless the ports are the
    /// built-in `input` and `output` table boundaries.
    pub fn with_ports(mut self, input: Port, output: Port) -> Result<Self> {
        validate_builtin_port(&input, "input", "operator.input_ports")?;
        validate_required_input(&input, "operator.input_ports")?;
        validate_builtin_port(&output, "output", "operator.output_ports")?;
        self.input_ports = [input];
        self.output_ports = [output];
        Ok(self)
    }
}

impl ExpressionOperator {
    pub fn configuration(&self) -> JsonMap {
        BTreeMap::from([
            (
                "expression".into(),
                self.expression
                    .as_ref()
                    .map_or(Value::Null, |value| Value::String(value.clone())),
            ),
            (
                "filter_expression".into(),
                self.filter_expression
                    .as_ref()
                    .map_or(Value::Null, |value| Value::String(value.clone())),
            ),
            (
                "select".into(),
                Value::Array(self.select.iter().cloned().map(Value::String).collect()),
            ),
            (
                "udfs".into(),
                Value::Array(self.udfs.iter().map(udf_configuration).collect()),
            ),
        ])
    }

    pub fn udf_references(&self) -> Vec<UdfReference> {
        self.udfs.clone()
    }

    #[doc(hidden)]
    pub(crate) async fn process_table(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        run: &RunContext,
        datafusion: &DataFusionRuntime,
    ) -> Result<BTreeMap<String, Batch>> {
        run.check_cancelled()?;
        let input = required_input(inputs, "input", self.name(), run.node_id())?;
        self.input_ports[0].validate(input, &format!("{}.input", self.name))?;
        let tables = BTreeMap::from([("input".into(), input.clone())]);
        let output = datafusion.sql(&self.query, &tables, run.node_id()).await?;
        run.check_cancelled()?;
        Ok(BTreeMap::from([("output".into(), output)]))
    }
}

#[derive(Clone, Debug)]
pub struct SqlOperator {
    name: String,
    query: String,
    aliases: Vec<String>,
    udfs: Vec<UdfReference>,
    input_ports: Vec<Port>,
    output_ports: [Port; 1],
}

impl SqlOperator {
    /// Creates a multi-input `DataFusion` SQL operator.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the operator name is
    /// empty, the query is not one valid `SELECT`/CTE, or the aliases are
    /// empty, duplicate, or invalid port names.
    pub fn new(
        name: &str,
        query: &str,
        aliases: Vec<String>,
        udfs: Vec<UdfReference>,
    ) -> Result<Self> {
        validate_operator_name(name)?;
        if aliases.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "operator.inputs".into(),
                message: "SQL operators require at least one input alias".into(),
            });
        }
        let mut unique = BTreeSet::new();
        if aliases.iter().any(|alias| !unique.insert(alias.as_str())) {
            return Err(CalcFlowError::InvalidArgument {
                field: "operator.inputs".into(),
                message: "SQL operator input aliases must be unique".into(),
            });
        }
        let input_ports = aliases
            .iter()
            .map(|alias| table_port(alias))
            .collect::<Result<Vec<_>>>()?;
        validate_select_query(query)?;
        Ok(Self {
            name: name.into(),
            query: query.into(),
            aliases,
            udfs,
            input_ports,
            output_ports: [table_port("output")?],
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn input_ports(&self) -> &[Port] {
        &self.input_ports
    }

    pub const fn output_ports(&self) -> &[Port] {
        &self.output_ports
    }

    /// Returns this operator with exact configuration-defined table ports.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] unless inputs match the SQL
    /// aliases in order and the output is the built-in `output` port.
    pub fn with_ports(mut self, inputs: Vec<Port>, output: Port) -> Result<Self> {
        if inputs.len() != self.aliases.len()
            || inputs.iter().zip(&self.aliases).any(|(port, alias)| {
                port.name() != alias || port.kind() != BatchKind::Table || !port.required()
            })
        {
            return Err(CalcFlowError::InvalidArgument {
                field: "operator.input_ports".into(),
                message: "ports must be table ports matching SQL aliases in order".into(),
            });
        }
        validate_builtin_port(&output, "output", "operator.output_ports")?;
        self.input_ports = inputs;
        self.output_ports = [output];
        Ok(self)
    }
}

impl SqlOperator {
    pub fn configuration(&self) -> JsonMap {
        BTreeMap::from([
            ("query".into(), Value::String(self.query.clone())),
            (
                "inputs".into(),
                Value::Array(self.aliases.iter().cloned().map(Value::String).collect()),
            ),
            (
                "udfs".into(),
                Value::Array(self.udfs.iter().map(udf_configuration).collect()),
            ),
        ])
    }

    pub fn udf_references(&self) -> Vec<UdfReference> {
        self.udfs.clone()
    }

    #[doc(hidden)]
    pub(crate) async fn process_table(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        run: &RunContext,
        datafusion: &DataFusionRuntime,
    ) -> Result<BTreeMap<String, Batch>> {
        run.check_cancelled()?;
        let tables = self
            .aliases
            .iter()
            .zip(&self.input_ports)
            .map(|(alias, port)| {
                let batch = required_input(inputs, alias, self.name(), run.node_id())?;
                port.validate(batch, &format!("{}.{alias}", self.name))?;
                Ok((alias.clone(), batch.clone()))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let output = datafusion.sql(&self.query, &tables, run.node_id()).await?;
        run.check_cancelled()?;
        Ok(BTreeMap::from([("output".into(), output)]))
    }
}

/// Compile-time operator classification used to keep table and external engines separate.
pub enum OperatorDefinition {
    /// An operator implemented by an explicitly registered external provider.
    External(Box<dyn Operator>),
    /// A built-in `DataFusion` expression operator.
    Expression(ExpressionOperator),
    /// A built-in `DataFusion` SQL operator.
    Sql(SqlOperator),
}

impl<T> From<Box<T>> for OperatorDefinition
where
    T: Operator + 'static,
{
    fn from(value: Box<T>) -> Self {
        Self::External(value)
    }
}

impl From<Box<dyn Operator>> for OperatorDefinition {
    fn from(value: Box<dyn Operator>) -> Self {
        Self::External(value)
    }
}

impl From<Box<ExpressionOperator>> for OperatorDefinition {
    fn from(value: Box<ExpressionOperator>) -> Self {
        Self::Expression(*value)
    }
}

impl From<Box<SqlOperator>> for OperatorDefinition {
    fn from(value: Box<SqlOperator>) -> Self {
        Self::Sql(*value)
    }
}

impl OperatorDefinition {
    pub(crate) fn input_ports(&self) -> &[Port] {
        match self {
            Self::External(operator) => operator.input_ports(),
            Self::Expression(operator) => operator.input_ports(),
            Self::Sql(operator) => operator.input_ports(),
        }
    }

    pub(crate) fn output_ports(&self) -> &[Port] {
        match self {
            Self::External(operator) => operator.output_ports(),
            Self::Expression(operator) => operator.output_ports(),
            Self::Sql(operator) => operator.output_ports(),
        }
    }

    pub(crate) fn configuration(&self) -> JsonMap {
        match self {
            Self::External(operator) => operator.configuration(),
            Self::Expression(operator) => operator.configuration(),
            Self::Sql(operator) => operator.configuration(),
        }
    }

    pub(crate) fn udf_references(&self) -> Vec<UdfReference> {
        match self {
            Self::External(operator) => operator.udf_references(),
            Self::Expression(operator) => operator.udf_references(),
            Self::Sql(operator) => operator.udf_references(),
        }
    }

    pub(crate) const fn requires_datafusion(&self) -> bool {
        matches!(self, Self::Expression(_) | Self::Sql(_))
    }

    pub(crate) fn snapshot(&self) -> Result<Value> {
        match self {
            Self::External(operator) => operator.snapshot(),
            Self::Expression(_) | Self::Sql(_) => Ok(Value::Null),
        }
    }

    pub(crate) fn restore(&mut self, state: &Value) -> Result<()> {
        match self {
            Self::External(operator) => operator.restore(state),
            Self::Expression(_) | Self::Sql(_) if state.is_null() => Ok(()),
            Self::Expression(_) | Self::Sql(_) => Err(CalcFlowError::Format {
                message: "stateless operator state must be null".into(),
            }),
        }
    }

    pub(crate) fn reset(&mut self) -> Result<()> {
        match self {
            Self::External(operator) => operator.reset(),
            Self::Expression(_) | Self::Sql(_) => Ok(()),
        }
    }

    pub(crate) async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        run: &RunContext,
        datafusion: Option<&DataFusionRuntime>,
    ) -> Result<BTreeMap<String, Batch>> {
        match self {
            Self::External(operator) => operator.process(inputs, &OperatorContext { run }).await,
            Self::Expression(operator) => {
                let datafusion = required_datafusion(datafusion, operator.name())?;
                operator.process_table(inputs, run, datafusion).await
            }
            Self::Sql(operator) => {
                let datafusion = required_datafusion(datafusion, operator.name())?;
                operator.process_table(inputs, run, datafusion).await
            }
        }
    }
}

pub(crate) enum CompiledOperator {
    ExistingData(OperatorDefinition),
}

impl CompiledOperator {
    pub(crate) async fn process_data(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        run: &RunContext,
        datafusion: Option<&DataFusionRuntime>,
    ) -> Result<BTreeMap<String, Batch>> {
        match self {
            Self::ExistingData(operator) => operator.process(inputs, run, datafusion).await,
        }
    }

    pub(crate) fn snapshot(&self) -> Result<Value> {
        match self {
            Self::ExistingData(operator) => operator.snapshot(),
        }
    }

    pub(crate) fn restore(&mut self, state: &Value) -> Result<()> {
        match self {
            Self::ExistingData(operator) => operator.restore(state),
        }
    }

    pub(crate) fn reset(&mut self) -> Result<()> {
        match self {
            Self::ExistingData(operator) => operator.reset(),
        }
    }
}

fn required_datafusion<'a>(
    datafusion: Option<&'a DataFusionRuntime>,
    operator: &str,
) -> Result<&'a DataFusionRuntime> {
    datafusion.ok_or_else(|| CalcFlowError::Internal {
        message: format!("table operator {operator:?} has no run-scoped DataFusion runtime"),
    })
}

fn validate_operator_name(name: &str) -> Result<()> {
    if name.trim().is_empty() {
        Err(CalcFlowError::InvalidArgument {
            field: "operator.name".into(),
            message: "must not be empty".into(),
        })
    } else {
        Ok(())
    }
}

fn validate_builtin_port(port: &Port, name: &str, field: &str) -> Result<()> {
    if port.name() != name || port.kind() != BatchKind::Table {
        return Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: format!("must contain the {name:?} table port"),
        });
    }
    Ok(())
}

fn validate_required_input(port: &Port, field: &str) -> Result<()> {
    if !port.required() {
        return Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: "built-in input ports must be required".into(),
        });
    }
    Ok(())
}

pub(crate) fn expression_query(
    expression: Option<&str>,
    select: &[String],
    filter_expression: Option<&str>,
) -> Result<String> {
    let mut query = if let Some(expression) = expression {
        sql_projection(expression, "input")?
    } else {
        format!("SELECT {} FROM input", select.join(", "))
    };
    if let Some(filter) = filter_expression {
        query.push_str(" WHERE (");
        query.push_str(filter);
        query.push(')');
    }
    validate_select_query(&query)
}

fn table_port(name: &str) -> Result<Port> {
    Port::new(name, BatchKind::Table, true, None)
}

fn required_input<'a>(
    inputs: &'a BTreeMap<String, Batch>,
    input: &str,
    operator: &str,
    node_id: Option<&str>,
) -> Result<&'a Batch> {
    inputs.get(input).ok_or_else(|| CalcFlowError::Operator {
        node_id: node_id.unwrap_or(operator).into(),
        message: format!("missing required input {input}"),
    })
}

fn udf_configuration(reference: &UdfReference) -> Value {
    json!({
        "provider": reference.provider(),
        "name": reference.name(),
        "version": reference.version(),
        "kind": reference.kind(),
    })
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

pub trait ExternalOperatorFactory: Send + Sync {
    /// Creates an external operator for a validated data-only specification.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider rejects the specification or ports.
    fn create(
        &self,
        spec: &ExternalOperatorSpec,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
    ) -> Result<Box<dyn Operator>>;
}

type ProviderKey = (String, String, String);
type ProviderFactories = BTreeMap<ProviderKey, Arc<dyn ExternalOperatorFactory>>;

#[derive(Default)]
pub struct ProviderRegistry {
    factories: RwLock<ProviderFactories>,
}

impl ProviderRegistry {
    /// Registers an external factory under an exact provider, name, and version.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when an identity component
    /// is invalid or the key is already registered. A rejected registration
    /// leaves the registry unchanged.
    pub fn register(
        &self,
        provider: &str,
        name: &str,
        version: &str,
        factory: Arc<dyn ExternalOperatorFactory>,
    ) -> Result<()> {
        validate_provider_identity(provider, name, version)?;
        let key = (provider.into(), name.into(), version.into());
        match self.factories.write().entry(key) {
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

    /// Resolves an exact external provider factory.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when an identity component
    /// is invalid, or [`CalcFlowError::Compile`] when the provider is
    /// unavailable.
    pub fn resolve(
        &self,
        provider: &str,
        name: &str,
        version: &str,
    ) -> Result<Arc<dyn ExternalOperatorFactory>> {
        validate_provider_identity(provider, name, version)?;
        self.factories
            .read()
            .get(&(provider.into(), name.into(), version.into()))
            .cloned()
            .ok_or_else(|| CalcFlowError::Compile {
                message: format!("provider {provider}:{name}@{version} is unavailable"),
            })
    }
}

fn validate_provider_identity(provider: &str, name: &str, version: &str) -> Result<()> {
    for (field, value) in [("provider", provider), ("name", name), ("version", version)] {
        validate_portable_identifier(field, value)?;
    }
    Ok(())
}

#[cfg(test)]
mod compiled_operator_tests {
    use serde_json::Value;

    use super::{CompiledOperator, ExpressionOperator, OperatorDefinition};

    #[test]
    fn compiled_operator_existing_data_delegates_lifecycle() {
        let definition = OperatorDefinition::Expression(
            ExpressionOperator::new(
                "expression",
                "plus_one = value + 1",
                Vec::new(),
                None,
                Vec::new(),
            )
            .unwrap(),
        );
        let mut operator = CompiledOperator::ExistingData(definition);

        assert_eq!(operator.snapshot().unwrap(), Value::Null);
        operator.restore(&Value::Null).unwrap();
        operator.reset().unwrap();
    }
}
