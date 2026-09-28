use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    io::Cursor,
};

use async_trait::async_trait;
use datafusion::arrow::{
    datatypes::SchemaRef,
    ipc::{reader::FileReader, writer::FileWriter},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, DataFusionConfig, DataFusionRuntime, Epoch,
    EventTime, JsonMap, Port, Result, RunContext, UdfReference, UdfRegistrySnapshot,
    expression::{ValidatedQuery, parse_select_query},
};

use super::{
    BatchOperator, BatchOperatorContext, OperatorMetadata, OperatorStateSnapshot, StateBudget,
    StateSegment, StreamCollector, StreamOperator, StreamOperatorContext, StreamRuntimeState,
    table_port, udf_configuration, validate_builtin_port, validate_operator_name,
};

use super::expression::required_input;

/// A multi-input `DataFusion` SQL operator.
///
/// Batch graphs may use several input aliases. Stream graphs accept exactly
/// one alias (spec NG6: incremental multi-input joins are undefined); the
/// single-alias form retains bounded input for cumulative aggregate snapshots
/// and processes row-level SQL independently for each batch. Each emitted
/// aggregate snapshot reruns the query over all retained input. Call
/// [`Self::set_state_budget`] to enforce an application-chosen state limit.
pub struct SqlOperator {
    name: String,
    query: String,
    validated: ValidatedQuery,
    aliases: Vec<String>,
    udfs: Vec<UdfReference>,
    input_ports: Vec<Port>,
    output_ports: [Port; 1],
    stream_state: StreamRuntimeState,
    stream_aggregate: bool,
    retained: Option<RetainedSqlInput>,
    state_budget: Option<StateBudget>,
}

#[derive(Clone)]
struct RetainedSqlInput {
    batch: Batch,
    segment: StateSegment,
    rows: u64,
    bytes: u64,
}

impl Clone for SqlOperator {
    /// A clone carries the declaration but not the built runtime; the lazy
    /// operator-scoped session is rebuilt on first use.
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            query: self.query.clone(),
            validated: self.validated.clone(),
            aliases: self.aliases.clone(),
            udfs: self.udfs.clone(),
            input_ports: self.input_ports.clone(),
            output_ports: self.output_ports.clone(),
            stream_state: self.stream_state.clone(),
            stream_aggregate: self.stream_aggregate,
            retained: None,
            state_budget: self.state_budget,
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
        let validated = parse_select_query(query)?;
        let stream_aggregate = validated.has_stream_aggregate();
        Ok(Self {
            name: name.into(),
            query: query.into(),
            validated,
            aliases,
            udfs,
            input_ports,
            output_ports: [table_port("output")?],
            stream_state: StreamRuntimeState::new(),
            stream_aggregate,
            retained: None,
            state_budget: None,
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

    /// The normalized read-only query executed by this operator.
    pub(crate) fn query_text(&self) -> &str {
        &self.query
    }

    /// Plans the exact SQL output schema without processing rows.
    ///
    /// This internal adapter seam uses built-in SQL functions and named Arrow
    /// schemas. It does not execute batches or select registered UDFs.
    ///
    /// # Errors
    ///
    /// Returns an error for selected UDFs, mismatched aliases, unresolved SQL,
    /// or duplicate output field names.
    #[doc(hidden)]
    pub async fn infer_schema(&self, schemas: &BTreeMap<String, SchemaRef>) -> Result<SchemaRef> {
        if !self.udfs.is_empty()
            || !schemas
                .keys()
                .eq(self.aliases.iter().collect::<BTreeSet<_>>())
        {
            return Err(CalcFlowError::InvalidArgument {
                field: "sql.tables".into(),
                message: "schema planning requires the declared aliases and built-in SQL functions"
                    .into(),
            });
        }
        let schema = DataFusionRuntime::new(DataFusionConfig::default())?
            .infer_query_schema(&self.query, schemas, &self.name)
            .await?;
        let mut names = BTreeSet::new();
        for field in schema.fields() {
            if !names.insert(field.name()) {
                return Err(CalcFlowError::InvalidArgument {
                    field: "sql.output.schema".into(),
                    message: format!(
                        "duplicate output field {:?}; use unique aliases",
                        field.name()
                    ),
                });
            }
        }
        Ok(schema)
    }

    /// Attaches the plan's `DataFusion` resources for the stream path.
    pub(crate) fn set_stream_resources(
        &mut self,
        config: DataFusionConfig,
        udfs: UdfRegistrySnapshot,
        selected_udfs: Vec<UdfReference>,
    ) {
        self.stream_state.set_resources(config, udfs, selected_udfs);
    }

    pub(crate) const fn stream_runtime_initialized(&self) -> bool {
        self.stream_state.is_initialized()
    }

    pub(crate) const fn has_stream_aggregate(&self) -> bool {
        self.stream_aggregate
    }

    /// Sets an optional row and logical-byte limit for retained stream SQL
    /// aggregate input. A newly constructed operator has no fixed state limit.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] if existing state exceeds
    /// the requested budget.
    ///
    /// # Examples
    ///
    /// ```
    /// use calc_flow::{SqlOperator, StateBudget};
    /// fn configure(operator: &mut SqlOperator) -> calc_flow::Result<()> {
    ///     operator.set_state_budget(StateBudget::new(20_000_000, 8 << 30)?)
    /// }
    /// ```
    pub fn set_state_budget(&mut self, budget: StateBudget) -> Result<()> {
        self.set_stream_state_budget(Some(budget))
    }

    pub(crate) fn set_stream_state_budget(&mut self, budget: Option<StateBudget>) -> Result<()> {
        if budget.is_some_and(|budget| {
            self.retained
                .as_ref()
                .is_some_and(|state| !budget.allows(state.rows, state.bytes))
        }) {
            return Err(CalcFlowError::InvalidArgument {
                field: "sql.state_budget".into(),
                message: "existing SQL aggregate state exceeds the requested budget".into(),
            });
        }
        self.state_budget = budget;
        Ok(())
    }

    fn query_digest(&self) -> String {
        hex::encode(Sha256::digest(self.query.as_bytes()))
    }

    async fn accumulate(&self, batch: &Batch) -> Result<RetainedSqlInput> {
        let rows = u64::try_from(batch.num_rows())
            .map_err(|_| sql_state_error("input row count exceeds u64"))?;
        let bytes = if batch.num_rows() == 0 && self.retained.is_some() {
            0
        } else {
            u64::try_from(batch.estimated_bytes()?)
                .map_err(|_| sql_state_error("input byte count exceeds u64"))?
        };
        let previous_rows = self.retained.as_ref().map_or(0, |state| state.rows);
        let previous_bytes = self.retained.as_ref().map_or(0, |state| state.bytes);
        let rows = previous_rows
            .checked_add(rows)
            .ok_or_else(|| sql_state_error("retained row count overflowed"))?;
        let bytes = previous_bytes
            .checked_add(bytes)
            .ok_or_else(|| sql_state_error("retained byte count overflowed"))?;
        if self
            .state_budget
            .is_some_and(|budget| !budget.allows(rows, bytes))
        {
            return Err(CalcFlowError::Operator {
                node_id: self.name.clone(),
                message: "SQL aggregate retained input exceeds the configured state budget".into(),
            });
        }
        let mut records = self
            .retained
            .as_ref()
            .map(|state| {
                state
                    .batch
                    .table_payload()
                    .expect("retained table")
                    .batches()
                    .to_vec()
            })
            .unwrap_or_default();
        if batch.num_rows() > 0 || records.is_empty() {
            records.extend_from_slice(batch.table_payload().expect("validated table").batches());
        }
        let combined = Batch::table(records, batch.metadata().clone())?;
        let (combined, segment) = tokio::task::spawn_blocking(move || {
            let segment = StateSegment::new(encode_sql_state(&combined)?);
            Ok::<_, CalcFlowError>((combined, segment))
        })
        .await
        .map_err(|error| CalcFlowError::Internal {
            message: format!("SQL state encoder task failed: {error}"),
        })??;
        Ok(RetainedSqlInput {
            batch: combined,
            segment,
            rows,
            bytes,
        })
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
        let output = datafusion
            .sql_validated(&self.validated, &tables, run.node_id())
            .await?;
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
        let output = runtime
            .sql_validated(&self.validated, &tables, Some(&self.name))
            .await?;
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
        let next = if self.stream_aggregate {
            Some(self.accumulate(&batch).await?)
        } else {
            None
        };
        let tables = BTreeMap::from([(
            alias.clone(),
            next.as_ref().map_or(batch, |state| state.batch.clone()),
        )]);
        let runtime = self.stream_state.runtime()?;
        let produced = runtime
            .sql_validated(&self.validated, &tables, Some(&self.name))
            .await?;
        context.check_cancelled()?;
        output.emit("output", produced).await?;
        self.retained = next;
        Ok(())
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

    fn checkpoint(&mut self, _epoch: Epoch) -> Result<OperatorStateSnapshot> {
        let Some(state) = &self.retained else {
            return Ok(OperatorStateSnapshot::default());
        };
        Ok(OperatorStateSnapshot {
            inline_metadata: BTreeMap::from([
                ("query_sha256".into(), json!(self.query_digest())),
                ("rows".into(), json!(state.rows)),
                ("bytes".into(), json!(state.bytes)),
            ]),
            segments: BTreeMap::from([("input".into(), state.segment.clone())]),
        })
    }

    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        if snapshot.inline_metadata.is_empty() && snapshot.segments.is_empty() {
            self.retained = None;
            return Ok(());
        }
        if !self.stream_aggregate
            || snapshot.inline_metadata.len() != 3
            || snapshot.segments.len() != 1
            || snapshot
                .inline_metadata
                .get("query_sha256")
                .and_then(Value::as_str)
                != Some(self.query_digest().as_str())
        {
            return Err(sql_state_error(
                "SQL aggregate checkpoint does not match this operator",
            ));
        }
        let segment = snapshot
            .segments
            .get("input")
            .ok_or_else(|| sql_state_error("SQL aggregate checkpoint has no input segment"))?;
        let rows = snapshot
            .inline_metadata
            .get("rows")
            .and_then(Value::as_u64)
            .ok_or_else(|| sql_state_error("SQL aggregate checkpoint has no row count"))?;
        let bytes = snapshot
            .inline_metadata
            .get("bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| sql_state_error("SQL aggregate checkpoint has no byte count"))?;
        let batch = decode_sql_state(segment.bytes())?;
        let actual_rows = u64::try_from(batch.num_rows())
            .map_err(|_| sql_state_error("restored row count exceeds u64"))?;
        let actual_bytes = u64::try_from(batch.estimated_bytes()?)
            .map_err(|_| sql_state_error("restored byte count exceeds u64"))?;
        if rows != actual_rows
            || bytes != actual_bytes
            || self
                .state_budget
                .is_some_and(|budget| !budget.allows(rows, bytes))
        {
            return Err(sql_state_error(
                "SQL aggregate checkpoint charge is invalid",
            ));
        }
        self.retained = Some(RetainedSqlInput {
            batch,
            segment: segment.clone(),
            rows,
            bytes,
        });
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.retained = None;
        Ok(())
    }
}

fn sql_state_error(message: &str) -> CalcFlowError {
    CalcFlowError::Format {
        message: message.into(),
    }
}

fn encode_sql_state(batch: &Batch) -> Result<Vec<u8>> {
    let table = batch.table_payload().expect("SQL state is a table");
    let mut bytes = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut bytes, table.schema())
            .map_err(|error| sql_state_error(&format!("SQL state IPC header failed: {error}")))?;
        for record in table.batches() {
            writer.write(record).map_err(|error| {
                sql_state_error(&format!("SQL state IPC write failed: {error}"))
            })?;
        }
        writer
            .finish()
            .map_err(|error| sql_state_error(&format!("SQL state IPC finish failed: {error}")))?;
    }
    Ok(bytes)
}

fn decode_sql_state(bytes: &[u8]) -> Result<Batch> {
    if !bytes.starts_with(b"ARROW1") || !bytes.ends_with(b"ARROW1") {
        return Err(sql_state_error(
            "SQL state segment has no Arrow IPC file magic",
        ));
    }
    let reader = FileReader::try_new(Cursor::new(bytes), None)
        .map_err(|error| sql_state_error(&format!("SQL state IPC is invalid: {error}")))?;
    let records = reader
        .collect::<std::result::Result<Vec<RecordBatch>, _>>()
        .map_err(|error| sql_state_error(&format!("SQL state records are invalid: {error}")))?;
    Batch::table(records, BatchMetadata::default())
}
