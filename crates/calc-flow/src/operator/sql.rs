use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
};

use async_trait::async_trait;
use serde_json::Value;

use crate::{
    Batch, BatchKind, CalcFlowError, DataFusionConfig, DataFusionRuntime, EventTime, JsonMap, Port,
    Result, RunContext, UdfReference, UdfRegistrySnapshot, expression::validate_select_query,
};

use super::{
    BatchOperator, BatchOperatorContext, OperatorMetadata, StreamCollector, StreamOperator,
    StreamOperatorContext, table_port, udf_configuration, validate_builtin_port,
    validate_operator_name,
};

use super::expression::required_input;

/// Lazily built operator-scoped `DataFusion` resources for the stream and
/// standalone-trait paths.
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
}

/// A multi-input `DataFusion` SQL operator.
///
/// Batch graphs may use several input aliases. Stream graphs accept exactly
/// one alias (spec NG6: incremental multi-input joins are undefined); the
/// single-alias form implements [`StreamOperator`] with per-batch stateless
/// processing (API note A2.4).
pub struct SqlOperator {
    name: String,
    query: String,
    aliases: Vec<String>,
    udfs: Vec<UdfReference>,
    input_ports: Vec<Port>,
    output_ports: [Port; 1],
    stream_state: StreamRuntimeState,
}

impl Clone for SqlOperator {
    /// A clone carries the declaration but not the built runtime; the lazy
    /// operator-scoped session is rebuilt on first use.
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            query: self.query.clone(),
            aliases: self.aliases.clone(),
            udfs: self.udfs.clone(),
            input_ports: self.input_ports.clone(),
            output_ports: self.output_ports.clone(),
            stream_state: StreamRuntimeState {
                resources: self.stream_state.resources.clone(),
                runtime: None,
            },
        }
    }
}

impl fmt::Debug for SqlOperator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SqlOperator")
            .field("name", &self.name)
            .field("query", &self.query)
            .field("aliases", &self.aliases)
            .field("udfs", &self.udfs)
            .field("input_ports", &self.input_ports)
            .field("output_ports", &self.output_ports)
            .finish_non_exhaustive()
    }
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
            stream_state: StreamRuntimeState::new(),
        })
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

    /// Attaches the plan's `DataFusion` resources for the stream path.
    pub(crate) fn set_stream_resources(
        &mut self,
        config: DataFusionConfig,
        udfs: UdfRegistrySnapshot,
        selected_udfs: Vec<UdfReference>,
    ) {
        self.stream_state.resources = Some((config, udfs, selected_udfs));
    }

    #[doc(hidden)]
    pub(crate) async fn process_table(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        run: &RunContext,
        datafusion: &DataFusionRuntime,
    ) -> Result<BTreeMap<String, Batch>> {
        run.check_cancelled()?;
        let tables = self.collect_tables(inputs, run.node_id())?;
        let output = datafusion.sql(&self.query, &tables, run.node_id()).await?;
        run.check_cancelled()?;
        Ok(BTreeMap::from([("output".into(), output)]))
    }

    fn collect_tables(
        &self,
        inputs: &BTreeMap<String, Batch>,
        node_id: Option<&str>,
    ) -> Result<BTreeMap<String, Batch>> {
        self.aliases
            .iter()
            .zip(&self.input_ports)
            .map(|(alias, port)| {
                let batch = required_input(inputs, alias, &self.name, node_id)?;
                port.validate(batch, &format!("{}.{alias}", self.name))?;
                Ok((alias.clone(), batch.clone()))
            })
            .collect()
    }
}

impl OperatorMetadata for SqlOperator {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_ports(&self) -> &[Port] {
        &self.input_ports
    }

    fn output_ports(&self) -> &[Port] {
        &self.output_ports
    }

    fn configuration(&self) -> JsonMap {
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

    fn udf_references(&self) -> Vec<UdfReference> {
        self.udfs.clone()
    }
}

#[async_trait]
impl BatchOperator for SqlOperator {
    /// Standalone batch processing through the operator-scoped session. The
    /// batch executor instead drives this operator through the run-scoped
    /// session (v2 invariant), so this trait path is for direct use.
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        let tables = self.collect_tables(inputs, None)?;
        let runtime = self.stream_state.runtime()?;
        let output = runtime.sql(&self.query, &tables, Some(&self.name)).await?;
        Ok(BTreeMap::from([("output".into(), output)]))
    }
}

#[async_trait]
impl StreamOperator for SqlOperator {
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let [alias] = self.aliases.as_slice() else {
            return Err(CalcFlowError::Operator {
                node_id: self.name.clone(),
                message: "multi-input SQL has no incremental stream semantics".into(),
            });
        };
        if ingress != alias {
            return Err(CalcFlowError::Operator {
                node_id: self.name.clone(),
                message: format!("unknown ingress {ingress:?}; expected {alias:?}"),
            });
        }
        context.check_cancelled()?;
        self.input_ports[0].validate(&batch, &format!("{}.{alias}", self.name))?;
        let tables = BTreeMap::from([(alias.clone(), batch)]);
        let runtime = self.stream_state.runtime()?;
        let produced = runtime.sql(&self.query, &tables, Some(&self.name)).await?;
        context.check_cancelled()?;
        output.emit("output", produced).await
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }
}
