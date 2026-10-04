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
use serde_json::Value;
#[cfg(test)]
use serde_json::json;
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

#[cfg(test)]
mod checkpoint_test_hooks;
mod compact;
#[cfg(test)]
mod current_checkpoint_tests;
mod current_retained;
mod incremental;
mod ipc;
mod metadata;
#[cfg(test)]
mod recovery_test_hooks;
mod restore;
pub(crate) use restore::PreparedSqlRestore;
mod retention;

/// A multi-input `DataFusion` SQL operator.
///
/// Batch graphs may use several input aliases. Stream graphs accept one alias,
/// emit cumulative aggregate snapshots and process row-level SQL per batch.
/// Use [`Self::set_state_budget`] to enforce an application-chosen state limit.
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
    compact: Option<Box<compact::CompactSqlState>>,
    retained_capture: Option<std::sync::Arc<current_retained::RetainedCapture>>,
    state_budget: Option<StateBudget>,
    incremental: Option<Box<incremental::IncrementalSql>>,
    incremental_checked: bool,
    #[cfg(test)]
    incremental_work: (usize, usize),
    #[cfg(test)]
    retained_handles_copied: std::sync::atomic::AtomicUsize,
    #[cfg(test)]
    checkpoint_test_owner: std::sync::OnceLock<std::sync::Arc<()>>,
}

struct RetainedSqlInput {
    records: Vec<RecordBatch>,
    metadata: BatchMetadata,
    projection: Option<std::sync::Arc<retention::SqlProjection>>,
    projection_checked: bool,
    backing_reservations:
        Vec<std::sync::Arc<datafusion::execution::memory_pool::MemoryReservation>>,
    reservation: Option<datafusion::execution::memory_pool::MemoryReservation>,
    segment: Option<std::sync::Arc<ipc::SqlInputSegment>>,
    metadata_segment: Option<std::sync::Arc<metadata::SqlMetadata>>,
    rows: u64,
    bytes: u64,
}

type RetentionPlans = (
    datafusion::logical_expr::LogicalPlan,
    datafusion::logical_expr::LogicalPlan,
);

struct SqlCheckpointInput {
    records: Vec<RecordBatch>,
    metadata: BatchMetadata,
    native: Option<incremental::compact_state::PaidNativeStateRecords>,
    #[cfg(test)]
    native_name: Option<String>,
    reservation: datafusion::execution::memory_pool::MemoryReservation,
    input_reservation: datafusion::execution::memory_pool::MemoryReservation,
}

type EncodedSqlState = (
    Option<std::sync::Arc<ipc::SqlInputSegment>>,
    Option<std::sync::Arc<metadata::SqlMetadata>>,
    Option<incremental::compact_state::PaidNativeStateRecords>,
);

struct PreparedRetention {
    batch: Batch,
    plans: Option<RetentionPlans>,
    backing_reservation:
        Option<std::sync::Arc<datafusion::execution::memory_pool::MemoryReservation>>,
    projection: Option<std::sync::Arc<retention::SqlProjection>>,
    _reservation: Option<datafusion::execution::memory_pool::MemoryReservation>,
}

pub(crate) struct PreparedSqlCheckpoint {
    compact: Option<std::sync::Arc<compact::CompactCapture>>,
    retained: Option<std::sync::Arc<current_retained::RetainedCapture>>,
}

struct MaterializedSqlInput {
    batch: Batch,
    native: Option<incremental::compact_state::PaidNativeStateRecords>,
    #[cfg(test)]
    native_name: Option<String>,
    _reservation: datafusion::execution::memory_pool::MemoryReservation,
}

impl MaterializedSqlInput {
    fn batch(&self) -> &Batch {
        &self.batch
    }
}

impl PreparedSqlCheckpoint {
    pub(crate) fn snapshot(&self) -> Result<OperatorStateSnapshot> {
        match (&self.compact, &self.retained) {
            (Some(_), Some(_)) => Err(sql_state_error("SQL checkpoint contains both state shapes")),
            (Some(capture), None) => Ok(capture.snapshot.clone()),
            (None, Some(capture)) => Ok(capture.snapshot.clone()),
            (None, None) => Ok(OperatorStateSnapshot::default()),
        }
    }
}

fn record_copy_reservation(
    runtime: &DataFusionRuntime,
    name: &str,
    records: usize,
    columns: usize,
    copies: usize,
) -> Result<datafusion::execution::memory_pool::MemoryReservation> {
    let width = incremental::checked_bytes(
        0,
        [
            (copies, size_of::<RecordBatch>()),
            (
                2,
                size_of::<std::sync::Arc<datafusion::execution::memory_pool::MemoryReservation>>(),
            ),
            (columns, size_of::<datafusion::arrow::array::ArrayRef>()),
        ],
        name,
    )?;
    let bytes = incremental::checked_bytes(256, [(records, width)], name)?;
    let reservation = runtime.incremental_reservation(name);
    reservation
        .try_grow(bytes)
        .map_err(|error| CalcFlowError::DataFusion {
            node_id: Some(name.into()),
            message: error.to_string(),
        })?;
    Ok(reservation)
}

impl RetainedSqlInput {
    fn materialize(&self, runtime: &DataFusionRuntime, name: &str) -> Result<MaterializedSqlInput> {
        let reservation = record_copy_reservation(
            runtime,
            name,
            self.records.len(),
            self.records[0].num_columns(),
            2,
        )?;
        let records = self.records.clone();
        #[cfg(test)]
        incremental_tests::after_record_copies(&self.metadata, records.len());
        let batch = Batch::table(records, self.metadata.clone())?;
        Ok(MaterializedSqlInput {
            batch,
            native: None,
            #[cfg(test)]
            native_name: None,
            _reservation: reservation,
        })
    }

    async fn checkpoint_records(
        &self,
        runtime: &DataFusionRuntime,
        name: &str,
        context: &StreamOperatorContext<'_>,
    ) -> Result<(
        Vec<RecordBatch>,
        BatchMetadata,
        datafusion::execution::memory_pool::MemoryReservation,
    )> {
        let reservation = record_copy_reservation(
            runtime,
            name,
            self.records.len(),
            self.records[0].num_columns(),
            2,
        )?;
        let mut records = Vec::with_capacity(self.records.len());
        for chunk in self.records.chunks(8192) {
            context.check_cancelled()?;
            records.extend_from_slice(chunk);
            #[cfg(test)]
            incremental_tests::after_record_copies(&self.metadata, chunk.len());
            tokio::task::yield_now().await;
        }
        context.check_cancelled()?;
        Ok((records, self.metadata.clone(), reservation))
    }

    fn needs_checkpoint(&self) -> bool {
        self.segment.is_none() || self.metadata_segment.is_none()
    }

    async fn prepare_checkpoint_parts(
        &self,
        runtime: &DataFusionRuntime,
        name: &str,
        context: &StreamOperatorContext<'_>,
    ) -> Result<(
        Option<SqlCheckpointInput>,
        Option<(
            BatchMetadata,
            datafusion::execution::memory_pool::MemoryReservation,
        )>,
    )> {
        let prepared = if self.segment.is_none() {
            let (records, metadata, reservation) =
                self.checkpoint_records(runtime, name, context).await?;
            Some(SqlCheckpointInput {
                records,
                metadata,
                reservation,
                input_reservation: runtime.incremental_reservation(name),
                native: None,
                #[cfg(test)]
                native_name: None,
            })
        } else {
            None
        };
        let metadata = if self.metadata_segment.is_none() {
            let reservation = metadata::reserve(runtime, &self.metadata, name)?;
            Some((self.metadata.clone(), reservation))
        } else {
            None
        };
        Ok((prepared, metadata))
    }

    fn reserve_append(
        &mut self,
        additional: usize,
        runtime: &DataFusionRuntime,
        name: &str,
        columns: usize,
    ) -> Result<()> {
        let (capacity, bytes) = self.append_capacity(additional, columns)?;
        let reservation = self
            .reservation
            .get_or_insert_with(|| runtime.incremental_reservation(name));
        incremental::ensure_reservation(reservation, bytes, name)?;
        self.backing_reservations
            .try_reserve_exact(capacity - self.backing_reservations.len())
            .map_err(|error| CalcFlowError::Internal {
                message: format!("SQL backing lease allocation failed: {error}"),
            })?;
        self.records
            .try_reserve_exact(capacity - self.records.len())
            .map_err(|error| CalcFlowError::Internal {
                message: format!("SQL retained record allocation failed: {error}"),
            })?;
        Ok(())
    }

    fn append_capacity(&self, additional: usize, columns: usize) -> Result<(usize, usize)> {
        let count = self
            .records
            .len()
            .checked_add(additional)
            .ok_or_else(|| sql_state_error("retained record count overflowed"))?;
        let capacity = count
            .max(1)
            .checked_next_power_of_two()
            .ok_or_else(|| sql_state_error("retained record capacity overflowed"))?
            .max(self.records.capacity());
        let width = retained_record_width(columns)?;
        let bytes = capacity
            .checked_mul(width)
            .ok_or_else(|| sql_state_error("retained capacity charge overflowed"))?;
        Ok((capacity, bytes))
    }
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
            compact: None,
            retained_capture: None,
            state_budget: self.state_budget,
            incremental: None,
            incremental_checked: false,
            #[cfg(test)]
            incremental_work: (0, 0),
            #[cfg(test)]
            retained_handles_copied: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            checkpoint_test_owner: std::sync::OnceLock::new(),
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
            compact: None,
            retained_capture: None,
            state_budget: None,
            incremental: None,
            incremental_checked: false,
            #[cfg(test)]
            incremental_work: (0, 0),
            #[cfg(test)]
            retained_handles_copied: std::sync::atomic::AtomicUsize::new(0),
            #[cfg(test)]
            checkpoint_test_owner: std::sync::OnceLock::new(),
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
                || self
                    .compact
                    .as_ref()
                    .is_some_and(|state| !budget.allows(state.ledger.rows, state.ledger.bytes))
        }) {
            return Err(CalcFlowError::InvalidArgument {
                field: "sql.state_budget".into(),
                message: "existing SQL aggregate state exceeds the requested budget".into(),
            });
        }
        self.state_budget = budget;
        Ok(())
    }

    fn retention_runtime(&self) -> Result<&DataFusionRuntime> {
        self.stream_state
            .runtime
            .as_ref()
            .ok_or_else(|| CalcFlowError::Internal {
                message: "SQL retention runtime has not been initialized".into(),
            })
    }

    async fn prepare_retention(
        &self,
        batch: Batch,
        alias: &str,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedRetention> {
        if let Some(state) = &self.compact {
            state.check_input(self, &batch)?;
        }
        if !self.can_project_retention() {
            return Ok(PreparedRetention {
                batch,
                plans: None,
                backing_reservation: None,
                projection: None,
                _reservation: None,
            });
        }
        let runtime = self.retention_runtime()?;
        let (plans, projection) = self
            .prepare_retention_projection(&batch, alias, runtime, context)
            .await?;
        let Some(projection) = projection else {
            return Ok(PreparedRetention {
                batch,
                plans: None,
                backing_reservation: None,
                projection: None,
                _reservation: None,
            });
        };
        let (reservation, projected) = self.project_retained_records(
            runtime,
            batch.table_payload()?.batches(),
            batch.metadata(),
            &projection,
        )?;
        let (projected, backing_reservation) = self.detach_backing(projected)?;
        Ok(PreparedRetention {
            batch: projected,
            plans,
            backing_reservation,
            projection: Some(projection),
            _reservation: Some(reservation),
        })
    }

    fn can_project_retention(&self) -> bool {
        self.stream_aggregate
            && self.udfs.is_empty()
            && self
                .compact
                .as_ref()
                .is_none_or(|state| state.projection().is_some())
            && !self
                .retained
                .as_ref()
                .is_some_and(|state| state.projection_checked && state.projection.is_none())
    }

    async fn prepare_retention_projection(
        &self,
        batch: &Batch,
        alias: &str,
        runtime: &DataFusionRuntime,
        context: &StreamOperatorContext<'_>,
    ) -> Result<(
        Option<RetentionPlans>,
        Option<std::sync::Arc<retention::SqlProjection>>,
    )> {
        let existing = self
            .retained
            .as_ref()
            .and_then(|state| state.projection.clone())
            .or_else(|| {
                self.compact
                    .as_deref()
                    .and_then(compact::CompactSqlState::projection)
            });
        let mut plans = None;
        let projection = if let Some(projection) = existing {
            Some(projection)
        } else {
            let schema = self.retained.as_ref().map_or_else(
                || batch.table_payload().map(|table| table.schema().clone()),
                |state| Ok(state.records[0].schema()),
            )?;
            let projection = retention::SqlProjection::resolve(
                runtime,
                &self.validated,
                alias,
                schema,
                &self.name,
            )?;
            if let Some(projection) = projection {
                plans = projection
                    .prepare_plan(runtime, &self.validated, alias, &self.name)
                    .await?;
                plans.as_ref().map(|_| projection)
            } else {
                None
            }
        };
        context.check_cancelled()?;
        Ok((plans, projection))
    }

    fn project_retained_records(
        &self,
        runtime: &DataFusionRuntime,
        records: &[RecordBatch],
        metadata: &BatchMetadata,
        projection: &retention::SqlProjection,
    ) -> Result<(datafusion::execution::memory_pool::MemoryReservation, Batch)> {
        let reservation = record_copy_reservation(
            runtime,
            &self.name,
            records.len(),
            projection.columns.ordinals().len(),
            2,
        )?;
        let records = records
            .iter()
            .map(|record| projection.columns.project(record))
            .collect::<Result<Vec<_>>>()?;
        let projected = Batch::table(records, metadata.clone())?;
        Ok((reservation, projected))
    }

    fn detach_backing(
        &self,
        batch: Batch,
    ) -> Result<(
        Batch,
        Option<std::sync::Arc<datafusion::execution::memory_pool::MemoryReservation>>,
    )> {
        let table = batch.table_payload()?;
        let visible = batch.estimated_bytes()?;
        let backing = sql_array_bytes(
            table.batches(),
            |array| Ok(array.get_buffer_memory_size()),
            "SQL backing charge overflowed",
        )?;
        if backing <= visible.saturating_mul(4).max(64 << 10) {
            return Ok((batch, None));
        }
        let (reservation, bound) = self.detachment_workspace(table, visible)?;
        let (_encoded, _decoded, detached) = detach_sql_batch(&batch, bound)?;
        Ok((detached, Some(std::sync::Arc::new(reservation))))
    }

    fn detachment_workspace(
        &self,
        table: &crate::batch::TableBatch,
        visible: usize,
    ) -> Result<(datafusion::execution::memory_pool::MemoryReservation, usize)> {
        let wire_bytes = sql_array_bytes(
            table.batches(),
            |array| sql_ipc_buffer_bytes(&array.to_data()),
            "SQL IPC buffer bound overflowed",
        )?;
        let schema = ipc::schema_bytes(table.schema())?;
        let reservation = self
            .retention_runtime()?
            .incremental_reservation(&self.name);
        let bound = incremental::checked_bytes(
            8192,
            [
                (wire_bytes.max(visible), 16),
                (schema, 16),
                (table.batches().len(), 8192),
            ],
            &self.name,
        )?;
        incremental::ensure_reservation(&reservation, bound, &self.name)?;
        Ok((reservation, bound))
    }

    pub(crate) fn prepare_checkpoint_work(
        &self,
        check_cancelled: &dyn Fn() -> Result<()>,
    ) -> Result<PreparedSqlCheckpoint> {
        #[cfg(test)]
        let _test_exit = checkpoint_test_hooks::WorkExit::new(self);
        check_cancelled()?;
        let compact = self
            .compact
            .as_ref()
            .map(|state| {
                compact::prepare_capture(
                    self,
                    state,
                    self.incremental.as_ref().expect("compact native plan"),
                    check_cancelled,
                )
            })
            .transpose()?;
        let retained = self
            .retained
            .as_ref()
            .map(|state| {
                self.retained_capture.as_ref().map_or_else(
                    || current_retained::capture(self, state, check_cancelled),
                    |capture| Ok(capture.clone()),
                )
            })
            .transpose()?;
        #[cfg(test)]
        if let Some(capture) = &compact {
            checkpoint_test_hooks::prepared(self, capture)?;
        }
        check_cancelled()?;
        Ok(PreparedSqlCheckpoint { compact, retained })
    }

    pub(crate) fn install_checkpoint_work(&mut self, prepared: PreparedSqlCheckpoint) {
        #[cfg(test)]
        let observed_compact = prepared.compact.is_some();
        if let Some(capture) = prepared.compact {
            self.compact
                .as_mut()
                .expect("capture retains its original operator")
                .capture = Some(capture);
        }
        self.retained_capture = prepared.retained;
        #[cfg(test)]
        if observed_compact {
            checkpoint_test_hooks::installed(self);
        }
    }

    fn query_digest(&self) -> String {
        hex::encode(Sha256::digest(self.query.as_bytes()))
    }

    fn incoming_charge(batch: &Batch, previous: Option<&RetainedSqlInput>) -> Result<(u64, u64)> {
        let incoming_rows = u64::try_from(batch.num_rows())
            .map_err(|_| sql_state_error("input row count exceeds u64"))?;
        let incoming_bytes = if batch.num_rows() == 0 && previous.is_some() {
            0
        } else {
            u64::try_from(batch.estimated_bytes()?)
                .map_err(|_| sql_state_error("input byte count exceeds u64"))?
        };
        Ok((incoming_rows, incoming_bytes))
    }

    fn accumulated_charge(
        &self,
        batch: &Batch,
        previous: Option<&RetainedSqlInput>,
    ) -> Result<(u64, u64)> {
        let (incoming_rows, incoming_bytes) = Self::incoming_charge(batch, previous)?;
        let previous_rows = previous.map_or(0, |state| state.rows);
        let previous_bytes = previous.map_or(0, |state| state.bytes);
        let rows = previous_rows
            .checked_add(incoming_rows)
            .ok_or_else(|| sql_state_error("retained row count overflowed"))?;
        let bytes = previous_bytes
            .checked_add(incoming_bytes)
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
        Ok((rows, bytes))
    }

    fn merged_records(batch: &Batch, previous: Option<&RetainedSqlInput>) -> Vec<RecordBatch> {
        let mut records = previous
            .map(|state| state.records.clone())
            .unwrap_or_default();
        if batch.num_rows() > 0 || records.is_empty() {
            records.extend_from_slice(batch.table_payload().expect("validated table").batches());
        }
        records
    }

    fn accumulate(&self, prepared: &PreparedRetention) -> Result<RetainedSqlInput> {
        let batch = &prepared.batch;
        let previous = self.retained.as_ref();
        let (rows, bytes) = self.accumulated_charge(batch, previous)?;
        let table = batch.table_payload()?;
        let count = previous
            .map_or(0, |state| state.records.len())
            .checked_add(table.batches().len())
            .ok_or_else(|| sql_state_error("retained record count overflowed"))?;
        let reservation = record_copy_reservation(
            self.retention_runtime()?,
            &self.name,
            count,
            table.schema().fields().len(),
            2,
        )?;
        #[cfg(test)]
        self.retained_handles_copied.fetch_add(
            previous.map_or(0, |state| state.records.len()),
            std::sync::atomic::Ordering::SeqCst,
        );
        let records = Self::merged_records(batch, previous);
        let schema = records[0].schema();
        if records.iter().any(|record| record.schema() != schema) {
            return Err(CalcFlowError::InvalidArgument {
                field: "batches".into(),
                message: "schemas must match".into(),
            });
        }
        let segment = previous
            .filter(|_| batch.num_rows() == 0)
            .and_then(|state| state.segment.clone());
        Ok(RetainedSqlInput {
            records,
            metadata: batch.metadata().clone(),
            projection: prepared.projection.clone(),
            projection_checked: true,
            backing_reservations: previous
                .map_or_else(Vec::new, |state| state.backing_reservations.clone())
                .into_iter()
                .chain(prepared.backing_reservation.iter().cloned())
                .collect(),
            reservation: Some(reservation),
            segment,
            metadata_segment: previous
                .filter(|state| state.metadata == *batch.metadata())
                .and_then(|state| state.metadata_segment.clone()),
            rows,
            bytes,
        })
    }

    fn validate_checkpoint_charge(&self, batch: &Batch, rows: u64, bytes: u64) -> Result<()> {
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
        Ok(())
    }

    async fn initialize_incremental(
        &mut self,
        alias: &str,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
        plans: Option<(
            datafusion::logical_expr::LogicalPlan,
            datafusion::logical_expr::LogicalPlan,
        )>,
    ) -> Result<Option<Box<incremental::IncrementalSql>>> {
        if self.incremental_checked {
            return Ok(None);
        }
        let mut initialized = self.plan_incremental(alias, batch, plans).await?;
        context.check_cancelled()?;
        if initialized
            .as_ref()
            .is_some_and(|plan| plan.requires_grouped_float_proof())
            && !self
                .stream_state
                .runtime()?
                .prove_grouped_float_plan(&self.validated, alias, batch, &self.name)
                .await?
        {
            self.incremental_checked = batch.num_rows() != 0;
            return Ok(None);
        }
        context.check_cancelled()?;
        if initialized
            .as_ref()
            .is_some_and(|plan| plan.requires_global_record_proof())
            && !self
                .stream_state
                .runtime()?
                .prove_global_record_plan(&self.validated, alias, batch, &self.name)
                .await?
        {
            self.incremental_checked = batch.num_rows() != 0;
            return Ok(None);
        }
        context.check_cancelled()?;
        if let Some(incremental) = initialized.as_mut() {
            #[cfg(test)]
            {
                self.incremental_work.1 += 1;
            }
            self.rebuild_incremental(incremental, context).await?;
        } else {
            self.incremental_checked = true;
        }
        Ok(initialized)
    }

    async fn plan_incremental(
        &mut self,
        alias: &str,
        batch: &Batch,
        plans: Option<(
            datafusion::logical_expr::LogicalPlan,
            datafusion::logical_expr::LogicalPlan,
        )>,
    ) -> Result<Option<Box<incremental::IncrementalSql>>> {
        let schema = self.retained.as_ref().map_or_else(
            || batch.table_payload().map(|table| table.schema().clone()),
            |state| Ok(state.records[0].schema()),
        )?;
        if let Some((raw, analyzed)) = plans {
            return incremental::IncrementalSql::from_plan(
                self.stream_state.runtime()?,
                &self.validated,
                schema,
                &raw,
                &analyzed,
                &self.name,
            )
            .map(|plan| plan.map(Box::new));
        }
        incremental::IncrementalSql::plan(
            self.stream_state.runtime()?,
            &self.validated,
            alias,
            schema,
            &self.name,
        )
        .await
        .map(|plan| plan.map(Box::new))
    }

    async fn rebuild_incremental(
        &mut self,
        incremental: &mut incremental::IncrementalSql,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        if let Some(retained) = self.retained.as_ref().filter(|state| state.rows != 0) {
            let runtime = self.stream_state.runtime()?;
            let materialized = retained.materialize(runtime, &self.name)?;
            let transaction = incremental
                .update(&materialized.batch, context, &self.name)
                .await?;
            #[cfg(test)]
            {
                self.incremental_work.0 += transaction.rows;
            }
            incremental.commit(transaction);
        }
        Ok(())
    }

    async fn process_incremental(
        &mut self,
        prepared: PreparedRetention,
        initialized: Option<Box<incremental::IncrementalSql>>,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.process_compact(prepared, initialized, context, output)
            .await
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
        let alias = alias.clone();
        if self.stream_aggregate {
            self.stream_state.runtime()?;
        }
        let mut prepared = self.prepare_retention(batch, &alias, context).await?;
        if self.stream_aggregate && self.udfs.is_empty() {
            let initialized = self
                .initialize_incremental(&alias, &prepared.batch, context, prepared.plans.take())
                .await?;
            if self.incremental.is_some() || initialized.is_some() {
                return self
                    .process_incremental(prepared, initialized, context, output)
                    .await;
            }
        }
        let next = if self.stream_aggregate {
            Some(self.accumulate(&prepared)?)
        } else {
            None
        };
        let runtime = self.stream_state.runtime()?;
        let materialized = next
            .as_ref()
            .map(|state| state.materialize(runtime, &self.name))
            .transpose()?;
        let tables = BTreeMap::from([(
            alias.clone(),
            materialized
                .as_ref()
                .map_or(prepared.batch, |state| state.batch.clone()),
        )]);
        #[cfg(test)]
        {
            self.incremental_work.0 += tables.values().map(Batch::num_rows).sum::<usize>();
            self.incremental_work.1 += 1;
        }
        let runtime = self.stream_state.runtime()?;
        let produced = runtime
            .sql_validated(&self.validated, &tables, Some(&self.name))
            .await?;
        context.check_cancelled()?;
        output.emit("output", produced).await?;
        self.retained = next;
        self.retained_capture = None;
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

    async fn prepare_checkpoint_async(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        context.check_cancelled()?;
        if self.compact.is_some() {
            return self.prepare_compact_capture_async(context).await;
        }
        let Some(state) = self
            .retained
            .as_ref()
            .filter(|state| state.needs_checkpoint())
        else {
            return Ok(());
        };
        let runtime = self.stream_state.runtime()?;
        let (prepared, metadata) = state
            .prepare_checkpoint_parts(runtime, &self.name, context)
            .await?;
        let job = context.job().clone();
        let attempt = tokio_util::sync::CancellationToken::new();
        let _cancel_on_drop = attempt.clone().drop_guard();
        let (segment, metadata, native) =
            encode_sql_state_async(prepared, metadata, job, attempt).await?;
        drop(native);
        context.check_cancelled()?;
        let state = self.retained.as_mut().expect("retained during preparation");
        if segment.is_some() {
            state.segment = segment;
        }
        if metadata.is_some() {
            state.metadata_segment = metadata;
        }
        Ok(())
    }

    fn checkpoint(&mut self, _epoch: Epoch) -> Result<OperatorStateSnapshot> {
        if self.compact.is_some() {
            return self.compact_checkpoint();
        }
        let prepared = self.prepare_checkpoint_work(&|| Ok(()))?;
        self.install_checkpoint_work(prepared);
        Ok(self
            .retained_capture
            .as_ref()
            .map_or_else(OperatorStateSnapshot::default, |capture| {
                capture.snapshot.clone()
            }))
    }

    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        let prepared = self.prepare_restore(snapshot, &|| Ok(()))?;
        self.install_restore(prepared);
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.compact = None;
        self.retained = None;
        self.retained_capture = None;
        self.incremental = None;
        self.incremental_checked = false;
        Ok(())
    }
}

fn sql_ipc_buffer_bytes(data: &datafusion::arrow::array::ArrayData) -> Result<usize> {
    let buffers = data
        .buffers()
        .iter()
        .try_fold(0usize, |bytes, buffer| bytes.checked_add(buffer.len()))
        .ok_or_else(|| sql_state_error("SQL IPC buffer bound overflowed"))?;
    let buffers = buffers
        .checked_add(data.nulls().map_or(0, |nulls| nulls.buffer().len()))
        .ok_or_else(|| sql_state_error("SQL IPC null bound overflowed"))?;
    data.child_data().iter().try_fold(buffers, |bytes, child| {
        bytes
            .checked_add(sql_ipc_buffer_bytes(child)?)
            .ok_or_else(|| sql_state_error("SQL IPC child bound overflowed"))
    })
}

fn retained_record_width(columns: usize) -> Result<usize> {
    let columns = columns
        .checked_mul(size_of::<datafusion::arrow::array::ArrayRef>())
        .ok_or_else(|| sql_state_error("retained column charge overflowed"))?;
    size_of::<RecordBatch>()
        .checked_add(columns)
        .and_then(|bytes| {
            bytes.checked_add(size_of::<
                std::sync::Arc<datafusion::execution::memory_pool::MemoryReservation>,
            >())
        })
        .ok_or_else(|| sql_state_error("retained record charge overflowed"))
}

fn sql_state_error(message: &str) -> CalcFlowError {
    CalcFlowError::Format {
        message: message.into(),
    }
}

async fn encode_sql_state_async(
    prepared: Option<SqlCheckpointInput>,
    metadata: Option<(
        BatchMetadata,
        datafusion::execution::memory_pool::MemoryReservation,
    )>,
    job: crate::StreamJobContext,
    attempt: tokio_util::sync::CancellationToken,
) -> Result<EncodedSqlState> {
    tokio::task::spawn_blocking(move || {
        let prepared = prepared
            .map(|input| encode_checkpoint_input(input, &job, &attempt))
            .transpose()?;
        let metadata = metadata
            .map(|(metadata, reservation)| {
                metadata::encode(&metadata, reservation, || {
                    check_encoding_cancelled(&job, &attempt)
                })
            })
            .transpose()?;
        let (segment, native) =
            prepared.map_or((None, None), |(segment, native)| (Some(segment), native));
        Ok((segment, metadata, native))
    })
    .await
    .map_err(|error| CalcFlowError::Internal {
        message: format!("SQL state encoder task failed: {error}"),
    })?
}

fn encode_checkpoint_input(
    input: SqlCheckpointInput,
    job: &crate::StreamJobContext,
    attempt: &tokio_util::sync::CancellationToken,
) -> Result<(
    std::sync::Arc<ipc::SqlInputSegment>,
    Option<incremental::compact_state::PaidNativeStateRecords>,
)> {
    check_encoding_cancelled(job, attempt)?;
    let SqlCheckpointInput {
        records,
        metadata,
        reservation,
        input_reservation,
        native,
        #[cfg(test)]
        native_name,
    } = input;
    let mut materialized = MaterializedSqlInput {
        batch: Batch::table(records, metadata)?,
        native,
        #[cfg(test)]
        native_name,
        _reservation: reservation,
    };
    #[cfg(test)]
    if let (Some(name), Some(export)) = (&materialized.native_name, &materialized.native) {
        compact::direct_async_tests::before_ipc(name, export.records(), export.reserved_bytes());
    }
    let result = ipc::encode(materialized.batch(), input_reservation, || {
        check_encoding_cancelled(job, attempt)
    });
    #[cfg(test)]
    tests::after_encode(materialized.batch());
    result.map(|segment| (segment, materialized.native.take()))
}

fn check_encoding_cancelled(
    job: &crate::StreamJobContext,
    attempt: &tokio_util::sync::CancellationToken,
) -> Result<()> {
    job.check_cancelled()?;
    if attempt.is_cancelled() {
        return Err(CalcFlowError::Cancelled {
            run_id: job.job_id().to_string(),
        });
    }
    Ok(())
}

fn sql_array_bytes(
    records: &[RecordBatch],
    charge: impl Fn(&datafusion::arrow::array::ArrayRef) -> Result<usize>,
    overflow: &str,
) -> Result<usize> {
    records
        .iter()
        .flat_map(RecordBatch::columns)
        .try_fold(0usize, |bytes, array| {
            bytes
                .checked_add(charge(array)?)
                .ok_or_else(|| sql_state_error(overflow))
        })
}

fn detach_sql_batch(batch: &Batch, bound: usize) -> Result<(Vec<u8>, Batch, Batch)> {
    let encoded = encode_sql_state(batch)?;
    if encoded.capacity() > bound / 4 {
        return Err(sql_state_error(
            "SQL detachment IPC exceeded its prepaid bound",
        ));
    }
    let decoded = decode_sql_state(&encoded)?;
    let detached = Batch::table(
        decoded.table_payload()?.batches().to_vec(),
        batch.metadata().clone(),
    )?;
    validate_detached_bytes(&detached, bound)?;
    Ok((encoded, decoded, detached))
}

fn validate_detached_bytes(detached: &Batch, bound: usize) -> Result<()> {
    let actual = sql_array_bytes(
        detached.table_payload()?.batches(),
        |array| Ok(array.get_array_memory_size()),
        "SQL detached allocation charge overflowed",
    )?;
    if actual > bound / 2 {
        return Err(sql_state_error(
            "SQL detached arrays exceeded their prepaid bound",
        ));
    }
    Ok(())
}

fn encode_sql_state(batch: &Batch) -> Result<Vec<u8>> {
    encode_sql_state_checked(batch, || Ok(()))
}

fn encode_sql_state_checked(
    batch: &Batch,
    check_cancelled: impl FnMut() -> Result<()>,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    encode_sql_state_into(batch, &mut bytes, check_cancelled)?;
    Ok(bytes)
}

fn encode_sql_state_into<W: std::io::Write>(
    batch: &Batch,
    output: &mut W,
    mut check_cancelled: impl FnMut() -> Result<()>,
) -> Result<()> {
    #[cfg(test)]
    tests::before_checked_encode(batch)?;
    check_cancelled()?;
    write_sql_state_file(batch, output, &mut check_cancelled)?;
    check_cancelled()?;
    Ok(())
}

fn write_sql_state_file<W: std::io::Write>(
    batch: &Batch,
    output: &mut W,
    check_cancelled: &mut impl FnMut() -> Result<()>,
) -> Result<()> {
    let table = batch.table_payload().expect("SQL state is a table");
    #[cfg(test)]
    tests::before_schema_allocation(batch)?;
    let mut writer = FileWriter::try_new(output, table.schema())
        .map_err(|error| sql_state_error(&format!("SQL state IPC header failed: {error}")))?;
    write_sql_records(&mut writer, batch, check_cancelled)?;
    check_cancelled()?;
    writer
        .finish()
        .map_err(|error| sql_state_error(&format!("SQL state IPC finish failed: {error}")))?;
    Ok(())
}

fn write_sql_records<W: std::io::Write>(
    writer: &mut FileWriter<W>,
    batch: &Batch,
    check_cancelled: &mut impl FnMut() -> Result<()>,
) -> Result<()> {
    let table = batch.table_payload().expect("SQL state is a table");
    for record in table.batches() {
        check_cancelled()?;
        #[cfg(test)]
        tests::after_checked_cancel(batch);
        writer
            .write(record)
            .map_err(|error| sql_state_error(&format!("SQL state IPC write failed: {error}")))?;
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, LazyLock, Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    use datafusion::arrow::array::{Array, Int64Array};

    use super::*;
    use crate::{CancellationToken, EdgeCollector, StreamJobContext};

    pub(super) const ENCODE_SOURCE: &str = "sql-lazy-ipc-test";
    pub(super) static ENCODE_CALLS: AtomicUsize = AtomicUsize::new(0);
    pub(super) const CANCEL_SOURCE: &str = "sql-cancel-encoding";
    pub(super) static CANCEL_WRITES: AtomicUsize = AtomicUsize::new(0);
    pub(super) const DROP_SOURCE: &str = "dropped-prepare";
    pub(super) static DROP_WRITES: AtomicUsize = AtomicUsize::new(0);
    static ENCODER_FINISHED: LazyLock<Mutex<BTreeMap<String, tokio::sync::oneshot::Sender<()>>>> =
        LazyLock::new(|| Mutex::new(BTreeMap::new()));

    pub(super) fn after_encode(batch: &Batch) {
        if let Some(finished) = ENCODER_FINISHED
            .lock()
            .unwrap()
            .remove(batch.metadata().source())
        {
            let _ = finished.send(());
        }
    }
    type EncoderHook = Box<dyn FnOnce() -> Result<()> + Send>;
    static ENCODER_HOOKS: LazyLock<Mutex<BTreeMap<String, EncoderHook>>> =
        LazyLock::new(|| Mutex::new(BTreeMap::new()));
    static SCHEMA_ALLOCATION_HOOKS: LazyLock<Mutex<BTreeMap<String, EncoderHook>>> =
        LazyLock::new(|| Mutex::new(BTreeMap::new()));

    pub(super) fn before_schema_allocation(batch: &Batch) -> Result<()> {
        let hook = SCHEMA_ALLOCATION_HOOKS
            .lock()
            .unwrap()
            .remove(batch.metadata().source());
        hook.map_or(Ok(()), |hook| hook())
    }

    pub(super) fn on_schema_allocation(
        source: &str,
        hook: impl FnOnce() -> Result<()> + Send + 'static,
    ) {
        assert!(
            SCHEMA_ALLOCATION_HOOKS
                .lock()
                .unwrap()
                .insert(source.into(), Box::new(hook))
                .is_none()
        );
    }

    pub(super) fn before_checked_encode(batch: &Batch) -> Result<()> {
        if batch.metadata().source() == ENCODE_SOURCE {
            ENCODE_CALLS.fetch_add(1, Ordering::SeqCst);
        }
        before_encode(batch)
    }

    pub(super) fn after_checked_cancel(batch: &Batch) {
        if batch.metadata().source() == CANCEL_SOURCE {
            CANCEL_WRITES.fetch_add(1, Ordering::SeqCst);
        }
        if batch.metadata().source() == DROP_SOURCE {
            DROP_WRITES.fetch_add(1, Ordering::SeqCst);
        }
    }

    pub(super) fn before_encode(batch: &Batch) -> Result<()> {
        let hook = ENCODER_HOOKS
            .lock()
            .unwrap()
            .remove(batch.metadata().source());
        hook.map_or(Ok(()), |hook| hook())
    }

    fn on_encode(source: &str, hook: impl FnOnce() -> Result<()> + Send + 'static) {
        assert!(
            ENCODER_HOOKS
                .lock()
                .unwrap()
                .insert(source.into(), Box::new(hook))
                .is_none()
        );
    }

    fn operator() -> SqlOperator {
        SqlOperator::new(
            "totals",
            "SELECT SUM(value) AS total FROM events WHERE value IS NOT NULL",
            vec!["events".into()],
            Vec::new(),
        )
        .unwrap()
    }

    fn batch(source: &str, values: &[i64]) -> Batch {
        Batch::table(
            vec![
                RecordBatch::try_from_iter(vec![(
                    "value",
                    Arc::new(Int64Array::from(values.to_vec())) as Arc<dyn Array>,
                )])
                .unwrap(),
            ],
            BatchMetadata::new(source, 0, JsonMap::new()).unwrap(),
        )
        .unwrap()
    }

    fn job() -> StreamJobContext {
        StreamJobContext::new(
            1,
            "sql-test",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
    }

    fn total(collector: &mut EdgeCollector) -> i64 {
        let output = collector.drain("output");
        assert_eq!(output.len(), 1);
        let table = output[0].as_data().unwrap().table_payload().unwrap();
        table.batches()[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0)
    }

    #[tokio::test]
    async fn test_sql_ipc_encoding_is_deferred_until_checkpoint() {
        ENCODE_CALLS.store(0, Ordering::SeqCst);
        let mut operator = operator();
        let job = job();
        let context = StreamOperatorContext::new(&job, "totals", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        for (values, expected) in [(&[1][..], 1), (&[2, 3][..], 6)] {
            operator
                .process_data(
                    "events",
                    batch(ENCODE_SOURCE, values),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
            assert_eq!(total(&mut collector), expected);
        }
        assert_eq!(ENCODE_CALLS.load(Ordering::SeqCst), 0);
        operator.prepare_checkpoint_async(&context).await.unwrap();
        assert_eq!(ENCODE_CALLS.load(Ordering::SeqCst), 1);
        let first = operator.checkpoint(Epoch::INITIAL).unwrap();
        operator.prepare_checkpoint_async(&context).await.unwrap();
        let second = operator.checkpoint(Epoch::INITIAL).unwrap();
        assert!(Arc::ptr_eq(
            &first.segments["input-retained"].bytes_arc(),
            &second.segments["input-retained"].bytes_arc()
        ));
        assert_eq!(ENCODE_CALLS.load(Ordering::SeqCst), 1);
        operator
            .process_data(
                "events",
                batch(ENCODE_SOURCE, &[4]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        assert_eq!(total(&mut collector), 10);
        assert_eq!(ENCODE_CALLS.load(Ordering::SeqCst), 1);
        operator.prepare_checkpoint_async(&context).await.unwrap();
        let third = operator.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(ENCODE_CALLS.load(Ordering::SeqCst), 2);
        assert!(!Arc::ptr_eq(
            &first.segments["input-retained"].bytes_arc(),
            &third.segments["input-retained"].bytes_arc()
        ));
        let mut restored = self::operator();
        StreamOperator::restore(&mut restored, &third).unwrap();
        restored
            .process_data("events", batch("restored", &[5]), &context, &mut collector)
            .await
            .unwrap();
        assert_eq!(total(&mut collector), 15);
    }

    #[tokio::test]
    async fn test_sql_checkpoint_encoder_stops_when_cancelled() {
        CANCEL_WRITES.store(0, Ordering::SeqCst);
        let mut operator = operator();
        let job = job();
        let context = StreamOperatorContext::new(&job, "totals", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        for value in [1, 2] {
            operator
                .process_data(
                    "events",
                    batch(CANCEL_SOURCE, &[value]),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
            collector.drain("output");
        }
        let cancellation = job.cancellation().clone();
        on_encode(CANCEL_SOURCE, move || {
            cancellation.cancel();
            Ok(())
        });
        assert!(matches!(
            operator.prepare_checkpoint_async(&context).await,
            Err(CalcFlowError::Cancelled { .. })
        ));
        assert_eq!(CANCEL_WRITES.load(Ordering::SeqCst), 0);
        assert!(operator.retained.as_ref().unwrap().segment.is_none());
        assert_eq!(operator.retained.as_ref().unwrap().rows, 2);
        let active = self::job();
        operator
            .prepare_checkpoint_async(&StreamOperatorContext::new(&active, "totals", None))
            .await
            .unwrap();
        let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
        let mut restored = self::operator();
        StreamOperator::restore(&mut restored, &snapshot).unwrap();
        restored
            .process_data(
                "events",
                batch("restored-cancel", &[3]),
                &StreamOperatorContext::new(&active, "totals", None),
                &mut collector,
            )
            .await
            .unwrap();
        assert_eq!(total(&mut collector), 6);
    }

    fn same_segment(left: &OperatorStateSnapshot, right: &OperatorStateSnapshot) -> bool {
        let name = if left.inline_metadata["state_layout"] == json!(3) {
            "group-state"
        } else {
            "input-retained"
        };
        Arc::ptr_eq(
            &left.segments[name].bytes_arc(),
            &right.segments[name].bytes_arc(),
        )
    }

    #[tokio::test]
    async fn test_sql_empty_batch_reuses_prepared_checkpoint() {
        let mut operator = operator();
        let job = job();
        let context = StreamOperatorContext::new(&job, "totals", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "events",
                batch("empty-initial", &[]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        let empty = collector.drain("output");
        assert!(
            empty[0]
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()[0]
                .column(0)
                .is_null(0)
        );
        let empty_snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(empty_snapshot.inline_metadata["rows"], json!(0));
        let mut restored = self::operator();
        StreamOperator::restore(&mut restored, &empty_snapshot).unwrap();
        restored
            .process_data(
                "events",
                batch("after-empty", &[1, 2]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        assert_eq!(total(&mut collector), 3);
        restored.prepare_checkpoint_async(&context).await.unwrap();
        let first = restored.checkpoint(Epoch::INITIAL).unwrap();
        restored
            .process_data(
                "events",
                batch("different-empty-metadata", &[]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        assert_eq!(total(&mut collector), 3);
        restored.prepare_checkpoint_async(&context).await.unwrap();
        let second = restored.checkpoint(Epoch::INITIAL).unwrap();
        assert!(same_segment(&first, &second));
        assert_eq!(
            first.inline_metadata["rows"],
            second.inline_metadata["rows"]
        );
        assert_eq!(
            first.inline_metadata["bytes"],
            second.inline_metadata["bytes"]
        );
        assert_ne!(
            first.segments["batch-metadata"].bytes(),
            second.segments["batch-metadata"].bytes()
        );
        assert_eq!(
            restored.retained.as_ref().unwrap().metadata.source(),
            "different-empty-metadata"
        );
    }

    struct RejectOutput;

    #[async_trait]
    impl StreamCollector for RejectOutput {
        async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
            Err(CalcFlowError::Operator {
                node_id: "sink".into(),
                message: "injected emit failure".into(),
            })
        }
    }

    fn rollback_state(operator: &SqlOperator) -> ((u64, u64), BatchMetadata, bool) {
        if let Some(state) = &operator.compact {
            assert!(operator.retained.is_none());
            assert!(operator.incremental.is_some());
            (
                (state.ledger.rows, state.ledger.bytes),
                state.metadata.clone(),
                state.capture.is_some(),
            )
        } else {
            let state = operator.retained.as_ref().unwrap();
            assert!(operator.incremental.is_none());
            (
                (state.rows, state.bytes),
                state.metadata.clone(),
                operator.retained_capture.is_some(),
            )
        }
    }

    async fn assert_retry_total(query: &str, collector: &mut EdgeCollector) {
        let expected = DataFusionRuntime::new(DataFusionConfig::default())
            .unwrap()
            .sql(
                query,
                &BTreeMap::from([("events".into(), batch("emit-next", &[1, 3]))]),
                Some("rollback-oracle"),
            )
            .await
            .unwrap();
        let output = collector.drain("output");
        assert_eq!(output.len(), 1);
        let actual = output[0].as_data().unwrap();
        assert_eq!(actual.metadata(), expected.metadata());
        let actual_table = actual.table_payload().unwrap();
        let expected_table = expected.table_payload().unwrap();
        assert_eq!(actual_table.schema(), expected_table.schema());
        assert_eq!(actual_table.batches(), expected_table.batches());
        assert_eq!(
            actual_table.batches()[0]
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            4
        );
    }

    #[tokio::test]
    async fn test_sql_rejected_output_preserves_clean_and_dirty_state() {
        let job = job();
        let context = StreamOperatorContext::new(&job, "totals", None);
        for (query, native) in [
            (
                "SELECT SUM(value) AS total FROM events WHERE value IS NOT NULL",
                false,
            ),
            ("SELECT SUM(value) AS total FROM events", true),
        ] {
            for prepared in [false, true] {
                let mut operator =
                    SqlOperator::new("totals", query, vec!["events".into()], vec![]).unwrap();
                let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
                operator
                    .process_data(
                        "events",
                        batch("emit-initial", &[1]),
                        &context,
                        &mut collector,
                    )
                    .await
                    .unwrap();
                collector.drain("output");
                assert_eq!(operator.compact.is_some(), native);
                let before = prepared.then(|| operator.checkpoint(Epoch::INITIAL).unwrap());
                let state = rollback_state(&operator);
                assert_eq!(state.2, prepared);
                let error = operator
                    .process_data(
                        "events",
                        batch("emit-rejected", &[2]),
                        &context,
                        &mut RejectOutput,
                    )
                    .await
                    .unwrap_err();
                assert!(error.to_string().contains("injected emit failure"));
                assert_eq!(rollback_state(&operator), state);
                if let Some(before) = before {
                    let after = operator.checkpoint(Epoch::INITIAL).unwrap();
                    assert_eq!(
                        before.inline_metadata["state_layout"],
                        json!(if native { 3 } else { 4 })
                    );
                    assert_eq!(before.inline_metadata, after.inline_metadata);
                    assert!(same_segment(&before, &after));
                    assert_eq!(
                        before.segments.keys().collect::<Vec<_>>(),
                        after.segments.keys().collect::<Vec<_>>()
                    );
                    for (name, segment) in &before.segments {
                        assert_eq!(segment.bytes(), after.segments[name].bytes());
                    }
                }
                operator
                    .process_data("events", batch("emit-next", &[3]), &context, &mut collector)
                    .await
                    .unwrap();
                assert_retry_total(query, &mut collector).await;
            }
        }
    }

    #[tokio::test]
    async fn test_sql_query_and_byte_budget_errors_preserve_checkpoint() {
        let mut operator = SqlOperator::new(
            "totals",
            "SELECT SUM(CAST(value AS INT)) AS total FROM events",
            vec!["events".into()],
            Vec::new(),
        )
        .unwrap();
        let job = job();
        let context = StreamOperatorContext::new(&job, "totals", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        let input = batch("budget-initial", &[1]);
        let bytes = u64::try_from(input.estimated_bytes().unwrap()).unwrap();
        operator
            .process_data("events", input, &context, &mut collector)
            .await
            .unwrap();
        assert_eq!(total(&mut collector), 1);
        let first = operator.checkpoint(Epoch::INITIAL).unwrap();
        assert!(
            operator
                .process_data(
                    "events",
                    batch("query-failure", &[i64::MAX]),
                    &context,
                    &mut collector
                )
                .await
                .is_err()
        );
        assert!(collector.drain("output").is_empty());
        assert!(same_segment(
            &first,
            &operator.checkpoint(Epoch::INITIAL).unwrap()
        ));
        operator
            .set_state_budget(StateBudget::new(100, bytes).unwrap())
            .unwrap();
        let error = operator
            .process_data(
                "events",
                batch("budget-failure", &[2]),
                &context,
                &mut collector,
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("configured state budget"));
        assert!(collector.drain("output").is_empty());
        assert!(same_segment(
            &first,
            &operator.checkpoint(Epoch::INITIAL).unwrap()
        ));
        operator
            .set_state_budget(StateBudget::new(100, bytes * 2).unwrap())
            .unwrap();
        operator
            .process_data(
                "events",
                batch("budget-next", &[2]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        assert_eq!(total(&mut collector), 3);
    }

    #[tokio::test]
    async fn test_sql_checkpoint_encoder_failure_can_be_retried() {
        let mut operator = operator();
        let job = job();
        let context = StreamOperatorContext::new(&job, "totals", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "events",
                batch("encode-failure", &[1, 2]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        assert_eq!(total(&mut collector), 3);
        on_encode("encode-failure", || {
            Err(sql_state_error("injected encoder failure"))
        });
        let error = operator
            .prepare_checkpoint_async(&context)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("injected encoder failure"));
        assert!(operator.retained.as_ref().unwrap().segment.is_none());
        on_encode("encode-failure", || {
            Err(sql_state_error("injected sync failure"))
        });
        assert!(operator.checkpoint(Epoch::INITIAL).is_err());
        assert!(operator.retained.as_ref().unwrap().segment.is_none());
        operator.prepare_checkpoint_async(&context).await.unwrap();
        let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(snapshot.inline_metadata["rows"], json!(2));
        let mut restored = self::operator();
        StreamOperator::restore(&mut restored, &snapshot).unwrap();
        restored
            .process_data(
                "events",
                batch("retry-next", &[3]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        assert_eq!(total(&mut collector), 6);
    }

    #[tokio::test]
    async fn test_sql_dropped_worker_retains_materialized_record_credits() {
        let source = "sql-worker-materialized-credit";
        let record = batch(source, &[]).table_payload().unwrap().batches()[0].clone();
        let input = Batch::table(
            vec![record; 12_000],
            BatchMetadata::new(source, 0, JsonMap::new()).unwrap(),
        )
        .unwrap();
        let mut operator = operator();
        let job = job();
        let context = StreamOperatorContext::new(&job, "totals", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("events", input, &context, &mut collector)
            .await
            .unwrap();
        collector.drain("output");
        let pressure = operator
            .stream_state
            .runtime()
            .unwrap()
            .incremental_reservation("pressure");
        pressure.try_grow((1 << 30) - (3 << 20)).unwrap();
        let probe = operator
            .stream_state
            .runtime()
            .unwrap()
            .incremental_reservation("probe");
        let (entered, started) = tokio::sync::oneshot::channel();
        let (finished, completion) = tokio::sync::oneshot::channel();
        ENCODER_FINISHED
            .lock()
            .unwrap()
            .insert(source.into(), finished);
        let (resume, paused) = std::sync::mpsc::channel();
        on_encode(source, move || {
            entered.send(()).unwrap();
            paused
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
            Ok(())
        });
        {
            let prepare = operator.prepare_checkpoint_async(&context);
            tokio::pin!(prepare);
            tokio::select! {
                result = &mut prepare => panic!("encoder finished before release: {result:?}"),
                result = started => result.unwrap(),
            }
        }
        let result = probe.try_grow(3 << 19);
        resume.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(10), completion)
            .await
            .unwrap()
            .unwrap();
        assert!(
            result.is_err(),
            "record clone credits must remain with the paused worker after dropping preparation"
        );
        assert!(operator.retained.as_ref().unwrap().segment.is_none());
        assert!(!job.cancellation().is_cancelled());
    }

    #[tokio::test]
    async fn test_sql_dropped_preparation_cannot_install_stale_state() {
        DROP_WRITES.store(0, Ordering::SeqCst);
        let mut operator = operator();
        let job = job();
        let context = StreamOperatorContext::new(&job, "totals", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("events", batch(DROP_SOURCE, &[1]), &context, &mut collector)
            .await
            .unwrap();
        collector.drain("output");
        let (entered, started) = tokio::sync::oneshot::channel();
        let (finished, completion) = tokio::sync::oneshot::channel();
        ENCODER_FINISHED
            .lock()
            .unwrap()
            .insert(DROP_SOURCE.into(), finished);
        let (resume, paused) = std::sync::mpsc::channel();
        on_encode(DROP_SOURCE, move || {
            entered.send(()).unwrap();
            paused
                .recv_timeout(std::time::Duration::from_secs(10))
                .unwrap();
            Ok(())
        });
        {
            let prepare = operator.prepare_checkpoint_async(&context);
            tokio::pin!(prepare);
            tokio::select! {
                result = &mut prepare => panic!("encoder finished before release: {result:?}"),
                result = started => result.unwrap(),
            }
        }
        assert!(operator.retained.as_ref().unwrap().segment.is_none());
        resume.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(10), completion)
            .await
            .unwrap()
            .unwrap();
        assert!(!job.cancellation().is_cancelled());
        assert_eq!(DROP_WRITES.load(Ordering::SeqCst), 0);
        operator
            .process_data(
                "events",
                batch("after-dropped-prepare", &[2]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        assert_eq!(total(&mut collector), 3);
        let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
        assert!(same_segment(
            &snapshot,
            &operator.checkpoint(Epoch::INITIAL).unwrap()
        ));
        let mut restored = self::operator();
        StreamOperator::restore(&mut restored, &snapshot).unwrap();
        let active = self::job();
        restored
            .process_data(
                "events",
                batch("drop-restored", &[3]),
                &StreamOperatorContext::new(&active, "totals", None),
                &mut collector,
            )
            .await
            .unwrap();
        assert_eq!(total(&mut collector), 6);
    }
}

#[cfg(test)]
#[path = "sql/incremental_tests.rs"]
mod incremental_tests;
