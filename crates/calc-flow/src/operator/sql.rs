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

mod incremental;
mod ipc;
mod metadata;
mod restore;
mod retention;

/// A multi-input `DataFusion` SQL operator.
///
/// Batch graphs may use several input aliases. Stream graphs accept exactly
/// one alias (spec NG6: incremental multi-input joins are undefined); the
/// single-alias form retains input for cumulative aggregate snapshots
/// and processes row-level SQL independently for each batch. Call
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
    incremental: Option<Box<incremental::IncrementalSql>>,
    incremental_checked: bool,
    #[cfg(test)]
    incremental_work: (usize, usize),
    #[cfg(test)]
    retained_handles_copied: std::sync::atomic::AtomicUsize,
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

struct PendingSqlInput {
    first: Option<RetainedSqlInput>,
    records: Vec<RecordBatch>,
    metadata: BatchMetadata,
    rows: u64,
    bytes: u64,
    clear_segment: bool,
    backing_reservation:
        Option<std::sync::Arc<datafusion::execution::memory_pool::MemoryReservation>>,
    _reservation: Option<datafusion::execution::memory_pool::MemoryReservation>,
}

type RetentionPlans = (
    datafusion::logical_expr::LogicalPlan,
    datafusion::logical_expr::LogicalPlan,
);

type SqlCheckpointInput = (
    Vec<RecordBatch>,
    BatchMetadata,
    datafusion::execution::memory_pool::MemoryReservation,
    datafusion::execution::memory_pool::MemoryReservation,
);

struct PreparedRetention {
    batch: Batch,
    plans: Option<RetentionPlans>,
    backing_reservation:
        Option<std::sync::Arc<datafusion::execution::memory_pool::MemoryReservation>>,
    migrated: Option<RetainedSqlInput>,
    projection: Option<std::sync::Arc<retention::SqlProjection>>,
    _reservation: Option<datafusion::execution::memory_pool::MemoryReservation>,
}

struct MaterializedSqlInput {
    batch: Batch,
    _reservation: datafusion::execution::memory_pool::MemoryReservation,
}

impl MaterializedSqlInput {
    fn batch(&self) -> &Batch {
        &self.batch
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
            Some((
                records,
                metadata,
                reservation,
                runtime.incremental_reservation(name),
            ))
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
            state_budget: self.state_budget,
            incremental: None,
            incremental_checked: false,
            #[cfg(test)]
            incremental_work: (0, 0),
            #[cfg(test)]
            retained_handles_copied: std::sync::atomic::AtomicUsize::new(0),
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
            incremental: None,
            incremental_checked: false,
            #[cfg(test)]
            incremental_work: (0, 0),
            #[cfg(test)]
            retained_handles_copied: std::sync::atomic::AtomicUsize::new(0),
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
        if !self.can_project_retention() {
            return Ok(PreparedRetention {
                batch,
                plans: None,
                backing_reservation: None,
                migrated: None,
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
                migrated: None,
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
        let migrated = self
            .retained
            .as_ref()
            .filter(|state| state.projection.is_none())
            .map(|state| self.project_legacy(state, &projection))
            .transpose()?;
        Ok(PreparedRetention {
            batch: projected,
            plans,
            backing_reservation,
            migrated,
            projection: Some(projection),
            _reservation: Some(reservation),
        })
    }

    fn can_project_retention(&self) -> bool {
        self.stream_aggregate
            && self.udfs.is_empty()
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
            .and_then(|state| state.projection.clone());
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

    fn project_legacy(
        &self,
        state: &RetainedSqlInput,
        projection: &std::sync::Arc<retention::SqlProjection>,
    ) -> Result<RetainedSqlInput> {
        let runtime = self.retention_runtime()?;
        let (copies, projected) =
            self.project_retained_records(runtime, &state.records, &state.metadata, projection)?;
        let (projected, backing) = self.detach_backing(projected)?;
        let bytes = self.projected_legacy_charge(&projected, state.rows)?;
        let mut migrated = RetainedSqlInput {
            records: Vec::new(),
            metadata: state.metadata.clone(),
            projection: Some(projection.clone()),
            projection_checked: true,
            backing_reservations: backing.map_or_else(
                || state.backing_reservations.clone(),
                |reservation| vec![reservation],
            ),
            reservation: None,
            segment: None,
            metadata_segment: None,
            rows: state.rows,
            bytes,
        };
        let records = projected.table_payload()?.batches();
        migrated.reserve_append(
            records.len(),
            runtime,
            &self.name,
            projection.columns.ordinals().len(),
        )?;
        migrated.records.extend_from_slice(records);
        drop(copies);
        Ok(migrated)
    }

    fn projected_legacy_charge(&self, projected: &Batch, rows: u64) -> Result<u64> {
        let bytes = u64::try_from(projected.estimated_bytes()?)
            .map_err(|_| sql_state_error("projected byte charge exceeds u64"))?;
        if self
            .state_budget
            .is_some_and(|budget| !budget.allows(rows, bytes))
        {
            return Err(sql_state_error(
                "projected legacy input exceeds the state budget",
            ));
        }
        Ok(bytes)
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
        let previous = prepared.migrated.as_ref().or(self.retained.as_ref());
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

    fn read_projection(
        &self,
        snapshot: &OperatorStateSnapshot,
    ) -> Result<std::sync::Arc<retention::SqlProjection>> {
        if !self.projection_checkpoint_matches(snapshot) {
            return Err(sql_state_error(
                "SQL pruned checkpoint identity or inventory is invalid",
            ));
        }
        let runtime = self.retention_runtime()?;
        let (_schema_workspace, schema) = self.read_projection_schema(snapshot, runtime)?;
        let [alias] = self.aliases.as_slice() else {
            return Err(sql_state_error("SQL pruned restore requires one alias"));
        };
        let projection = retention::SqlProjection::resolve(
            self.retention_runtime()?,
            &self.validated,
            alias,
            schema,
            &self.name,
        )?;
        validate_projection_descriptor(snapshot, projection)
    }

    fn projection_checkpoint_matches(&self, snapshot: &OperatorStateSnapshot) -> bool {
        let fields = [
            "query_sha256",
            "rows",
            "bytes",
            "state_layout",
            "state_accounting",
            "retained_ordinals",
            "dependency_sha256",
            "physical_schema_sha256",
            "batch_metadata_sha256",
        ];
        self.stream_aggregate
            && self.udfs.is_empty()
            && snapshot.inline_metadata.len() == fields.len()
            && fields
                .iter()
                .all(|field| snapshot.inline_metadata.contains_key(*field))
            && snapshot.inline_metadata["state_layout"] == json!(2)
            && snapshot.inline_metadata["state_accounting"] == json!(2)
            && snapshot.inline_metadata["query_sha256"] == json!(self.query_digest())
            && projection_checkpoint_segments_match(snapshot)
    }

    fn read_projection_schema(
        &self,
        snapshot: &OperatorStateSnapshot,
        runtime: &DataFusionRuntime,
    ) -> Result<(
        datafusion::execution::memory_pool::MemoryReservation,
        SchemaRef,
    )> {
        let schema_workspace = runtime.incremental_reservation(&self.name);
        incremental::ensure_reservation(
            &schema_workspace,
            incremental::checked_bytes(
                8192,
                [(snapshot.segments["logical-schema"].bytes().len(), 8)],
                &self.name,
            )?,
            &self.name,
        )?;
        let schema = retention::decode_schema(&snapshot.segments["logical-schema"])?;
        if self.input_ports[0]
            .schema()
            .is_some_and(|expected| *expected != schema)
        {
            return Err(sql_state_error(
                "SQL checkpoint logical schema differs from the declared input",
            ));
        }
        Ok((schema_workspace, schema))
    }

    fn checkpoint_matches(&self, snapshot: &OperatorStateSnapshot) -> bool {
        let fields = [
            "query_sha256",
            "rows",
            "bytes",
            "state_layout",
            "state_accounting",
            "batch_metadata_sha256",
        ];
        self.stream_aggregate
            && snapshot.inline_metadata.len() == fields.len()
            && fields
                .iter()
                .all(|field| snapshot.inline_metadata.contains_key(*field))
            && snapshot.inline_metadata["state_layout"] == json!(2)
            && snapshot.inline_metadata["state_accounting"] == json!(2)
            && snapshot.inline_metadata["query_sha256"] == json!(self.query_digest())
            && snapshot.segments.len() == 2
            && snapshot.segments.contains_key("input")
            && snapshot.segments.contains_key("batch-metadata")
            && snapshot.inline_metadata["batch_metadata_sha256"]
                == json!(snapshot.segments["batch-metadata"].sha256())
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
        migrated: Option<&RetainedSqlInput>,
        plans: Option<(
            datafusion::logical_expr::LogicalPlan,
            datafusion::logical_expr::LogicalPlan,
        )>,
    ) -> Result<Option<Box<incremental::IncrementalSql>>> {
        if self.incremental_checked {
            return Ok(None);
        }
        let mut initialized = self.plan_incremental(alias, batch, migrated, plans).await?;
        context.check_cancelled()?;
        if let Some(incremental) = initialized.as_mut() {
            #[cfg(test)]
            {
                self.incremental_work.1 += 1;
            }
            self.rebuild_incremental(incremental, context, migrated)
                .await?;
        } else {
            self.incremental_checked = true;
        }
        Ok(initialized)
    }

    async fn plan_incremental(
        &mut self,
        alias: &str,
        batch: &Batch,
        migrated: Option<&RetainedSqlInput>,
        plans: Option<(
            datafusion::logical_expr::LogicalPlan,
            datafusion::logical_expr::LogicalPlan,
        )>,
    ) -> Result<Option<Box<incremental::IncrementalSql>>> {
        let schema = migrated.or(self.retained.as_ref()).map_or_else(
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
        migrated: Option<&RetainedSqlInput>,
    ) -> Result<()> {
        if let Some(retained) = migrated.or(self.retained.as_ref()) {
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
        mut prepared: PreparedRetention,
        mut initialized: Option<Box<incremental::IncrementalSql>>,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let pending = self.prepare_incremental_input(&mut prepared)?;
        let batch = prepared.batch;
        let incremental = initialized
            .as_mut()
            .or(self.incremental.as_mut())
            .expect("checked incremental plan");
        let runtime = self.stream_state.runtime()?;
        let transaction = incremental.update(&batch, context, &self.name).await?;
        #[cfg(test)]
        {
            self.incremental_work.0 += transaction.rows;
        }
        let produced =
            runtime.incremental_output(transaction.records.clone(), batch.metadata().clone())?;
        context.check_cancelled()?;
        output.emit("output", produced).await?;
        incremental.commit(transaction);
        self.commit_incremental_input(initialized, pending);
        Ok(())
    }

    fn prepare_incremental_input(
        &mut self,
        prepared: &mut PreparedRetention,
    ) -> Result<PendingSqlInput> {
        let batch = &prepared.batch;
        let previous = prepared.migrated.as_ref().or(self.retained.as_ref());
        let (rows, bytes) = self.accumulated_charge(batch, previous)?;
        let (append, additional) = validate_incremental_append(batch, previous)?;
        let metadata = batch.metadata().clone();
        let mut first = initial_incremental_input(
            prepared.migrated.take(),
            self.retained.as_ref(),
            &metadata,
            prepared.projection.as_ref(),
        );
        let runtime = self.stream_state.runtime()?;
        let columns = batch.table_payload()?.schema().fields().len();
        let reservation = reserve_incremental_records(
            &mut first,
            &mut self.retained,
            additional,
            columns,
            runtime,
            &self.name,
        )?;
        let records = append_batches(batch, append)?.to_vec();
        Ok(PendingSqlInput {
            first,
            records,
            metadata,
            rows,
            bytes,
            clear_segment: batch.num_rows() != 0,
            backing_reservation: prepared.backing_reservation.clone(),
            _reservation: reservation,
        })
    }

    fn commit_incremental_input(
        &mut self,
        initialized: Option<Box<incremental::IncrementalSql>>,
        pending: PendingSqlInput,
    ) {
        if initialized.is_some() {
            self.incremental = initialized;
        }
        self.incremental_checked = true;
        if let Some(first) = pending.first {
            self.retained = Some(first);
        }
        let state = self.retained.as_mut().expect("retained after emission");
        state.records.extend(pending.records);
        state
            .backing_reservations
            .extend(pending.backing_reservation);
        let metadata_changed = state.metadata != pending.metadata;
        state.metadata = pending.metadata;
        if metadata_changed {
            state.metadata_segment = None;
        }
        state.rows = pending.rows;
        state.bytes = pending.bytes;
        if pending.clear_segment {
            state.segment = None;
        }
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
                .initialize_incremental(
                    &alias,
                    &prepared.batch,
                    context,
                    prepared.migrated.as_ref(),
                    prepared.plans.take(),
                )
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
        let (segment, metadata) = encode_sql_state_async(prepared, metadata, job, attempt).await?;
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
        if let Some(state) = self
            .retained
            .as_mut()
            .filter(|state| state.segment.is_none() || state.metadata_segment.is_none())
        {
            let runtime = self.stream_state.runtime()?;
            let segment = if state.segment.is_none() {
                let materialized = state.materialize(runtime, &self.name)?;
                Some(ipc::encode(
                    &materialized.batch,
                    runtime.incremental_reservation(&self.name),
                    || Ok(()),
                )?)
            } else {
                None
            };
            let metadata = if state.metadata_segment.is_none() {
                Some(metadata::encode(
                    &state.metadata,
                    metadata::reserve(runtime, &state.metadata, &self.name)?,
                    || Ok(()),
                )?)
            } else {
                None
            };
            if segment.is_some() {
                state.segment = segment;
            }
            if metadata.is_some() {
                state.metadata_segment = metadata;
            }
        }
        let Some(state) = &self.retained else {
            return Ok(OperatorStateSnapshot::default());
        };
        let mut inline_metadata = BTreeMap::from([
            ("query_sha256".into(), json!(self.query_digest())),
            ("rows".into(), json!(state.rows)),
            ("bytes".into(), json!(state.bytes)),
            ("state_layout".into(), json!(2)),
            ("state_accounting".into(), json!(2)),
            (
                "batch_metadata_sha256".into(),
                json!(
                    state
                        .metadata_segment
                        .as_ref()
                        .expect("prepared metadata")
                        .segment
                        .sha256()
                ),
            ),
        ]);
        let mut segments = BTreeMap::from([(
            "batch-metadata".into(),
            state
                .metadata_segment
                .as_ref()
                .expect("prepared metadata")
                .segment
                .clone(),
        )]);
        if let Some(projection) = &state.projection {
            inline_metadata.extend([
                (
                    "retained_ordinals".into(),
                    json!(projection.columns.ordinals()),
                ),
                (
                    "dependency_sha256".into(),
                    json!(hex::encode(projection.columns.dependency_digest())),
                ),
                (
                    "physical_schema_sha256".into(),
                    json!(retention::schema_digest(
                        projection.columns.physical_schema()
                    )?),
                ),
            ]);
            segments.insert("logical-schema".into(), projection.logical_segment.clone());
        }
        let input_name = if state.projection.is_some() {
            "input-projected"
        } else {
            "input"
        };
        segments.insert(
            input_name.into(),
            state
                .segment
                .as_ref()
                .expect("prepared segment")
                .segment
                .clone(),
        );
        Ok(OperatorStateSnapshot {
            inline_metadata,
            segments,
        })
    }

    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        let prepared = self.prepare_restore(snapshot, &|| Ok(()))?;
        self.install_restore(prepared);
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.retained = None;
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
) -> Result<(
    Option<std::sync::Arc<ipc::SqlInputSegment>>,
    Option<std::sync::Arc<metadata::SqlMetadata>>,
)> {
    tokio::task::spawn_blocking(move || {
        let segment = prepared
            .map(|(records, metadata, reservation, input_reservation)| {
                let materialized = MaterializedSqlInput {
                    batch: Batch::table(records, metadata)?,
                    _reservation: reservation,
                };
                let batch = materialized.batch();
                let result = ipc::encode(batch, input_reservation, || {
                    check_encoding_cancelled(&job, &attempt)
                });
                #[cfg(test)]
                tests::after_encode(batch);
                result
            })
            .transpose()?;
        let metadata = metadata
            .map(|(metadata, reservation)| {
                metadata::encode(&metadata, reservation, || {
                    check_encoding_cancelled(&job, &attempt)
                })
            })
            .transpose()?;
        Ok((segment, metadata))
    })
    .await
    .map_err(|error| CalcFlowError::Internal {
        message: format!("SQL state encoder task failed: {error}"),
    })?
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

fn projection_checkpoint_segments_match(snapshot: &OperatorStateSnapshot) -> bool {
    snapshot.segments.len() == 3
        && snapshot.segments.contains_key("input-projected")
        && snapshot.segments.contains_key("logical-schema")
        && snapshot.segments.contains_key("batch-metadata")
        && snapshot.inline_metadata["batch_metadata_sha256"]
            == json!(snapshot.segments["batch-metadata"].sha256())
}

fn validate_projection_descriptor(
    snapshot: &OperatorStateSnapshot,
    projection: Option<std::sync::Arc<retention::SqlProjection>>,
) -> Result<std::sync::Arc<retention::SqlProjection>> {
    let projection = projection.ok_or_else(|| {
        sql_state_error("SQL pruned checkpoint cannot prove complete dependencies")
    })?;
    if snapshot.inline_metadata["retained_ordinals"] != json!(projection.columns.ordinals())
        || snapshot.inline_metadata["dependency_sha256"]
            != json!(hex::encode(projection.columns.dependency_digest()))
        || snapshot.inline_metadata["physical_schema_sha256"]
            != json!(retention::schema_digest(
                projection.columns.physical_schema()
            )?)
        || projection.logical_segment.bytes() != snapshot.segments["logical-schema"].bytes()
    {
        return Err(sql_state_error(
            "SQL pruned checkpoint descriptor differs from trusted dependencies",
        ));
    }
    Ok(projection)
}

fn validate_incremental_append(
    batch: &Batch,
    previous: Option<&RetainedSqlInput>,
) -> Result<(bool, usize)> {
    let append = batch.num_rows() > 0 || previous.is_none();
    if append
        && previous.is_some_and(|state| {
            state.records[0].schema()
                != batch
                    .table_payload()
                    .expect("validated table")
                    .schema()
                    .clone()
        })
    {
        return Err(CalcFlowError::InvalidArgument {
            field: "batches".into(),
            message: "schemas must match".into(),
        });
    }
    let additional = append_batches(batch, append)?.len();
    Ok((append, additional))
}

fn initial_incremental_input(
    migrated: Option<RetainedSqlInput>,
    retained: Option<&RetainedSqlInput>,
    metadata: &BatchMetadata,
    projection: Option<&std::sync::Arc<retention::SqlProjection>>,
) -> Option<RetainedSqlInput> {
    migrated.or_else(|| {
        retained.is_none().then(|| RetainedSqlInput {
            records: Vec::new(),
            metadata: metadata.clone(),
            projection: projection.cloned(),
            projection_checked: true,
            backing_reservations: Vec::new(),
            reservation: None,
            segment: None,
            metadata_segment: None,
            rows: 0,
            bytes: 0,
        })
    })
}

fn reserve_incremental_records(
    first: &mut Option<RetainedSqlInput>,
    retained: &mut Option<RetainedSqlInput>,
    additional: usize,
    columns: usize,
    runtime: &DataFusionRuntime,
    name: &str,
) -> Result<Option<datafusion::execution::memory_pool::MemoryReservation>> {
    let reservation = if additional == 0 {
        None
    } else {
        Some(record_copy_reservation(
            runtime, name, additional, columns, 1,
        )?)
    };
    if additional != 0 {
        first
            .as_mut()
            .or(retained.as_mut())
            .expect("retained preflight")
            .reserve_append(additional, runtime, name, columns)?;
    }
    Ok(reservation)
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

fn append_batches(batch: &Batch, append: bool) -> Result<&[RecordBatch]> {
    if append {
        Ok(batch.table_payload()?.batches())
    } else {
        Ok(&[])
    }
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
            "SELECT SUM(value) AS total FROM events",
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
            &first.segments["input"].bytes_arc(),
            &second.segments["input"].bytes_arc()
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
            &first.segments["input"].bytes_arc(),
            &third.segments["input"].bytes_arc()
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
        Arc::ptr_eq(
            &left.segments["input"].bytes_arc(),
            &right.segments["input"].bytes_arc(),
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
        let empty = batch("different-empty-metadata", &[]);
        let metadata = empty.metadata().clone();
        restored
            .process_data("events", empty, &context, &mut collector)
            .await
            .unwrap();
        assert_eq!(total(&mut collector), 3);
        restored.prepare_checkpoint_async(&context).await.unwrap();
        let second = restored.checkpoint(Epoch::INITIAL).unwrap();
        assert!(same_segment(&first, &second));
        for field in [
            "query_sha256",
            "rows",
            "bytes",
            "state_layout",
            "state_accounting",
        ] {
            assert_eq!(first.inline_metadata[field], second.inline_metadata[field]);
        }
        assert_ne!(
            first.segments["batch-metadata"].bytes(),
            second.segments["batch-metadata"].bytes()
        );
        let mut recovered = self::operator();
        StreamOperator::restore(&mut recovered, &second).unwrap();
        assert_eq!(recovered.retained.as_ref().unwrap().metadata, metadata);
    }

    #[tokio::test]
    async fn test_sql_current_checkpoint_restore_and_clone() {
        let job = job();
        let context = StreamOperatorContext::new(&job, "totals", None);
        let mut source = operator();
        let mut initial = EdgeCollector::new(source.output_ports().to_vec());
        source
            .process_data("events", batch("current", &[1, 2]), &context, &mut initial)
            .await
            .unwrap();
        source.prepare_checkpoint_async(&context).await.unwrap();
        let captured = source.checkpoint(Epoch::INITIAL).unwrap();
        let mut operator = operator();
        StreamOperator::restore(&mut operator, &captured).unwrap();
        let mut invalid = captured.clone();
        invalid.inline_metadata.insert("rows".into(), json!(3));
        assert!(StreamOperator::restore(&mut operator, &invalid).is_err());
        assert!(same_segment(
            &captured,
            &operator.checkpoint(Epoch::INITIAL).unwrap()
        ));
        let mut cloned = operator.clone();
        assert!(
            cloned
                .checkpoint(Epoch::INITIAL)
                .unwrap()
                .segments
                .is_empty()
        );
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("events", batch("continued", &[3]), &context, &mut collector)
            .await
            .unwrap();
        assert_eq!(total(&mut collector), 6);
        cloned
            .process_data("events", batch("cloned", &[4]), &context, &mut collector)
            .await
            .unwrap();
        assert_eq!(total(&mut collector), 4);
        StreamOperator::reset(&mut operator).unwrap();
        assert!(
            operator
                .checkpoint(Epoch::INITIAL)
                .unwrap()
                .segments
                .is_empty()
        );
        StreamOperator::restore(&mut cloned, &OperatorStateSnapshot::default()).unwrap();
        assert!(
            cloned
                .checkpoint(Epoch::INITIAL)
                .unwrap()
                .segments
                .is_empty()
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

    #[tokio::test]
    async fn test_sql_rejected_output_preserves_clean_and_dirty_state() {
        let job = job();
        let context = StreamOperatorContext::new(&job, "totals", None);
        for prepared in [false, true] {
            let mut operator = operator();
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
            let before = if prepared {
                Some(operator.checkpoint(Epoch::INITIAL).unwrap())
            } else {
                None
            };
            let charge = operator
                .retained
                .as_ref()
                .map(|state| (state.rows, state.bytes));
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
            assert_eq!(
                operator
                    .retained
                    .as_ref()
                    .map(|state| (state.rows, state.bytes)),
                charge
            );
            assert_eq!(
                operator.retained.as_ref().unwrap().segment.is_some(),
                prepared
            );
            if let Some(before) = before {
                assert!(same_segment(
                    &before,
                    &operator.checkpoint(Epoch::INITIAL).unwrap()
                ));
            }
            operator
                .process_data("events", batch("emit-next", &[3]), &context, &mut collector)
                .await
                .unwrap();
            assert_eq!(total(&mut collector), 4);
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
