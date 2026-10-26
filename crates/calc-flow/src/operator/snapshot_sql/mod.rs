//! Complete-relation replacement SQL. Every data message is one snapshot.
use std::{collections::BTreeMap, sync::Arc};

use async_trait::async_trait;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};

use crate::{
    Batch, CalcFlowError, DataFusionConfig, Epoch, JsonMap, OperatorMetadata,
    OperatorStateSnapshot, Port, Result, SqlOperator, StateBudget, StreamCollector, StreamOperator,
    StreamOperatorContext, UdfReference, UdfRegistrySnapshot,
    expression::{ValidatedQuery, parse_select_query},
    state_memory_pool::OwnedStatePool,
};

use super::{
    StreamRuntimeState,
    sql::{ipc, metadata},
};

mod checkpoint;
#[cfg(test)]
mod tests;
pub(crate) use checkpoint::{PreparedSnapshotCapture, PreparedSnapshotRestore};

struct Relation {
    batch: Batch,
    metadata: Arc<metadata::SqlMetadata>,
    backing: Arc<MemoryReservation>,
}

/// A stream-only SQL operator whose inputs replace complete named relations.
///
/// Every alias must have an exact schema before graph compilation. No result
/// is emitted until all aliases have received a batch (an empty table counts).
/// Each subsequent update emits a complete result using the latest relations.
/// Inputs and results must fit their edge budgets as whole batches. Outputs
/// must connect directly to sinks; exactly-once requests are unsupported.
///
/// # Examples
///
/// ```
/// use calc_flow::{BatchKind, Port, SnapshotSqlOperator};
/// use datafusion::arrow::datatypes::{DataType, Field};
/// let fields = vec![Field::new("value", DataType::Int64, false)];
/// let operator = SnapshotSqlOperator::new(
///     "latest", "SELECT * FROM p", vec!["p".into()], vec![],
/// )?.with_ports(
///     vec![Port::new("p", BatchKind::Table, true, Some(fields.clone()))?],
///     Port::new("output", BatchKind::Table, true, Some(fields))?,
/// )?;
/// # Ok::<(), calc_flow::CalcFlowError>(())
/// ```
pub struct SnapshotSqlOperator {
    declaration: Box<SqlOperator>,
    validated: ValidatedQuery,
    runtime: StreamRuntimeState,
    pool: Option<Arc<OwnedStatePool>>,
    budget: Option<StateBudget>,
    relations: Vec<Option<Arc<Relation>>>,
    capture: Option<PreparedSnapshotCapture>,
}

impl std::fmt::Debug for SnapshotSqlOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SnapshotSqlOperator")
            .field("declaration", &self.declaration)
            .finish_non_exhaustive()
    }
}

impl SnapshotSqlOperator {
    /// Declares a read-only query over one or more complete replacement inputs.
    ///
    /// # Errors
    /// Returns an error for an invalid query/name or ambiguous SQL aliases.
    /// Alias names must start with an ASCII letter/underscore and contain only
    /// ASCII letters, digits and underscores; case-insensitive duplicates fail.
    pub fn new(
        name: &str,
        query: &str,
        aliases: Vec<String>,
        udfs: Vec<UdfReference>,
    ) -> Result<Self> {
        let mut normalized = std::collections::BTreeSet::new();
        for alias in &aliases {
            if !alias
                .as_bytes()
                .first()
                .is_some_and(|c| c.is_ascii_alphabetic() || *c == b'_')
                || !alias
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'_')
                || !normalized.insert(alias.to_ascii_lowercase())
            {
                return Err(CalcFlowError::InvalidArgument {
                    field: "snapshot_sql.aliases".into(),
                    message: "aliases must be unambiguous SQL identifiers".into(),
                });
            }
        }
        let relations = (0..aliases.len()).map(|_| None).collect();
        Ok(Self {
            declaration: Box::new(SqlOperator::new(name, query, aliases, udfs)?),
            validated: parse_select_query(query)?,
            runtime: StreamRuntimeState::new(),
            pool: None,
            budget: None,
            relations,
            capture: None,
        })
    }

    /// Supplies required, exact input and result schemas.
    ///
    /// # Errors
    /// Returns an error unless ports match the aliases and have exact schemas.
    pub fn with_ports(mut self, inputs: Vec<Port>, output: Port) -> Result<Self> {
        if inputs
            .iter()
            .chain(std::iter::once(&output))
            .any(|p| p.schema().is_none())
        {
            return Err(state_error(
                "snapshot_sql requires exact input and output schemas",
            ));
        }
        self.declaration = Box::new((*self.declaration).with_ports(inputs, output)?);
        Ok(self)
    }

    /// Limits latest-relation rows and all owned state bytes for this node.
    ///
    /// Candidate replacements and cached encoded state pay while the old state
    /// remains live. Rows count all aliases after replacement, including shared
    /// relations. Byte accounting conservatively charges reachable Arrow backing,
    /// schema, metadata and handles. Reusing an owned relation shares its lease.
    /// Execution and serialization scratch use the engine pool separately.
    ///
    /// # Errors
    /// Returns an error if the existing state exceeds the new budget.
    pub fn set_state_budget(&mut self, budget: StateBudget) -> Result<()> {
        self.set_budget(Some(budget))
    }

    pub(crate) fn set_budget(&mut self, budget: Option<StateBudget>) -> Result<()> {
        check_rows(&self.relations, budget)?;
        self.ensure_pool()?;
        self.pool
            .as_ref()
            .expect("initialized pool")
            .set_limit(budget.map(StateBudget::max_bytes))
            .map_err(engine_error)?;
        self.budget = budget;
        Ok(())
    }

    pub(crate) fn query_text(&self) -> &str {
        self.declaration.query_text()
    }

    pub(crate) fn set_stream_resources(
        &mut self,
        config: DataFusionConfig,
        udfs: UdfRegistrySnapshot,
        selected: Vec<UdfReference>,
    ) {
        self.runtime.set_resources(config, udfs, selected);
    }

    pub(crate) const fn stream_runtime_initialized(&self) -> bool {
        self.runtime.is_initialized()
    }

    #[cfg(test)]
    pub(crate) fn state_pool_for_test(&self) -> Arc<OwnedStatePool> {
        self.pool.as_ref().unwrap().clone()
    }

    #[cfg(test)]
    pub(crate) fn input_rows_for_test(&self) -> usize {
        self.relations
            .iter()
            .flatten()
            .map(|r| r.batch.num_rows())
            .sum()
    }

    fn ensure_pool(&mut self) -> Result<()> {
        if self.pool.is_none() {
            let pool = self.runtime.runtime()?.snapshot_state_pool();
            pool.set_limit(self.budget.map(StateBudget::max_bytes))
                .map_err(engine_error)?;
            self.pool = Some(pool);
        }
        Ok(())
    }

    fn reservation(&self) -> MemoryReservation {
        let pool: Arc<dyn MemoryPool> = self.pool.as_ref().expect("initialized pool").clone();
        MemoryConsumer::new(format!("snapshot-sql:{}", self.name())).register(&pool)
    }

    fn scratch(&mut self, bound: usize) -> Result<MemoryReservation> {
        let name = self.name().to_owned();
        let reservation = self.runtime.runtime()?.incremental_reservation(&name);
        reservation.try_grow(bound).map_err(engine_error)?;
        Ok(reservation)
    }

    fn relation(&self, batch: Batch, check: &dyn Fn() -> Result<()>) -> Result<Arc<Relation>> {
        check()?;
        if let Some(owned) = self.relations.iter().flatten().find(|owned| {
            std::ptr::eq(
                owned.batch.table_payload().expect("table").batches(),
                batch.table_payload().expect("validated table").batches(),
            ) && owned.batch.metadata() == batch.metadata()
        }) {
            return Ok(owned.clone());
        }
        let backing = if let Some(owned) = self.relations.iter().flatten().find(|owned| {
            std::ptr::eq(
                owned.batch.table_payload().expect("table").batches(),
                batch.table_payload().expect("table").batches(),
            )
        }) {
            owned.backing.clone()
        } else {
            let reservation = self.reservation();
            reservation
                .try_grow(backing_bytes(&batch)?)
                .map_err(engine_error)?;
            Arc::new(reservation)
        };
        let reservation =
            metadata::reserve_with(batch.metadata(), self.name(), self.reservation())?;
        let metadata = metadata::encode(batch.metadata(), reservation, check)?;
        check()?;
        Ok(Arc::new(Relation {
            batch,
            metadata,
            backing,
        }))
    }
}

impl OperatorMetadata for SnapshotSqlOperator {
    fn name(&self) -> &str {
        self.declaration.name()
    }
    fn input_ports(&self) -> &[Port] {
        self.declaration.input_ports()
    }
    fn output_ports(&self) -> &[Port] {
        self.declaration.output_ports()
    }
    fn udf_references(&self) -> Vec<UdfReference> {
        self.declaration.udf_references()
    }
    fn configuration(&self) -> JsonMap {
        let mut configuration = self.declaration.configuration();
        configuration.insert("snapshot_policy".into(), "replace_snapshot_v1".into());
        configuration
    }
}

#[async_trait]
impl StreamOperator for SnapshotSqlOperator {
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()?;
        let index = self
            .input_ports()
            .iter()
            .position(|p| p.name() == ingress)
            .ok_or_else(|| state_error("unknown snapshot_sql ingress"))?;
        self.input_ports()[index].validate(&batch, ingress)?;
        self.ensure_pool()?;
        let mut next = self.relations.clone();
        // Check rows before retaining or allocating the candidate.
        let rows =
            self.relations
                .iter()
                .enumerate()
                .try_fold(batch.num_rows(), |rows, (i, r)| {
                    if i == index {
                        Ok(rows)
                    } else {
                        add(rows, r.as_ref().map_or(0, |r| r.batch.num_rows()))
                    }
                })?;
        if self.budget.is_some_and(|b| rows as u64 > b.max_rows()) {
            return Err(state_error("snapshot_sql latest relations exceed max_rows"));
        }
        next[index] = Some(self.relation(batch, &|| context.check_cancelled())?);
        if next.iter().all(Option::is_some) {
            let tables = self
                .input_ports()
                .iter()
                .zip(&next)
                .map(|(p, r)| {
                    (
                        p.name().to_owned(),
                        r.as_ref().expect("initialized").batch.clone(),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            let name = self.name().to_owned();
            let query = &self.validated;
            let runtime = self.runtime.runtime()?;
            let produced = runtime
                .snapshot_sql(query, &tables, &name, context.output_budget())
                .await?;
            context.check_cancelled()?;
            self.output_ports()[0].validate(&produced, "snapshot_sql.output")?;
            output.emit("output", produced).await?;
        }
        self.relations = next;
        self.capture = None;
        Ok(())
    }

    async fn on_watermark(
        &mut self,
        _watermark: crate::EventTime,
        context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()
    }

    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()
    }

    fn checkpoint(&mut self, _epoch: Epoch) -> Result<OperatorStateSnapshot> {
        let prepared = self.prepare_checkpoint_work(&|| Ok(()))?;
        let snapshot = prepared.snapshot.clone();
        self.capture = Some(prepared);
        Ok(snapshot)
    }

    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        let prepared = self.prepare_restore(snapshot, &|| Ok(()))?;
        self.install_restore(prepared);
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.relations.iter_mut().for_each(|r| *r = None);
        self.capture = None;
        Ok(())
    }
}

fn backing_bytes(batch: &Batch) -> Result<usize> {
    let table = batch.table_payload()?;
    let schema = ipc::schema_bytes(table.schema())?;
    table
        .batches()
        .iter()
        .try_fold(add(512, schema)?, |bytes, record| {
            record
                .columns()
                .iter()
                .try_fold(add(bytes, 256)?, |bytes, array| {
                    add(bytes, array.get_array_memory_size())
                })
        })
}

fn check_rows(relations: &[Option<Arc<Relation>>], budget: Option<StateBudget>) -> Result<()> {
    let rows = relations
        .iter()
        .flatten()
        .try_fold(0, |rows, r| add(rows, r.batch.num_rows()))?;
    if budget.is_some_and(|b| rows as u64 > b.max_rows()) {
        return Err(state_error("snapshot_sql latest relations exceed max_rows"));
    }
    Ok(())
}

fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| state_error("snapshot_sql accounting overflow"))
}
fn engine_error(error: impl std::fmt::Display) -> CalcFlowError {
    state_error(&error.to_string())
}
fn state_error(message: &str) -> CalcFlowError {
    CalcFlowError::Format {
        message: message.into(),
    }
}
