use std::{mem::size_of, sync::Arc};

use ahash::RandomState;
use datafusion::{
    arrow::{
        array::ArrayRef,
        datatypes::{DataType, Field, Schema, SchemaRef},
        record_batch::RecordBatch,
        row::{RowConverter, SortField},
    },
    common::ScalarValue,
    execution::memory_pool::MemoryReservation,
    logical_expr::{
        Accumulator, EmitTo, Expr, GroupsAccumulator, LogicalPlan, execution_props::ExecutionProps,
    },
    physical_expr::{
        PhysicalExpr,
        aggregate::{AggregateFunctionExpr, LoweredAggregateBuilder},
        create_physical_expr,
    },
    physical_plan::aggregates::{
        group_values::{GroupValues, new_group_values},
        order::GroupOrdering,
    },
};
use hashbrown::HashMap;

use crate::{
    Batch, CalcFlowError, DataFusionRuntime, Result, StreamOperatorContext,
    expression::ValidatedQuery,
};

const CHUNK_ROWS: usize = 8192;

pub(super) struct IncrementalSql {
    schema: SchemaRef,
    aggregate_schema: SchemaRef,
    output_schema: SchemaRef,
    aggregates: Vec<Arc<AggregateFunctionExpr>>,
    aggregate_bytes: usize,
    plan_bytes: usize,
    projection: Vec<Arc<dyn PhysicalExpr>>,
    keys: Vec<usize>,
    variable_columns: Vec<usize>,
    converter: Option<RowConverter>,
    groups: Vec<Group>,
    index: HashMap<Arc<[u8]>, usize, RandomState>,
    reservation: MemoryReservation,
    #[cfg(test)]
    pub(super) finalized_groups: Arc<std::sync::atomic::AtomicUsize>,
    #[cfg(test)]
    pub(super) partial_groups: Arc<std::sync::atomic::AtomicUsize>,
    #[cfg(test)]
    pub(super) historical_key_lookups: Arc<std::sync::atomic::AtomicUsize>,
    #[cfg(test)]
    pub(super) encoded_rows: Arc<std::sync::atomic::AtomicUsize>,
}

struct Group {
    key: Arc<[u8]>,
    values: Arc<[ScalarValue]>,
    states: Vec<Vec<ScalarValue>>,
    results: Vec<ScalarValue>,
    _reservation: MemoryReservation,
}

struct Candidate {
    group: Group,
    accumulators: Vec<Box<dyn Accumulator>>,
}

struct InputCandidates {
    groups: CandidateMap,
    touched: KeyIndex,
    new_count: usize,
    native: Option<NativeKeys>,
    partial: Option<PartialGroups>,
}

struct NativeKeys {
    groups: Box<dyn GroupValues>,
    indices: Vec<usize>,
    width: usize,
    reservation: MemoryReservation,
}

impl NativeKeys {
    fn new(
        field: &Field,
        rows: usize,
        width: usize,
        reservation: MemoryReservation,
        name: &str,
    ) -> Result<Self> {
        let bytes = native_key_bytes(0, rows, width, name)?;
        reservation
            .try_grow(bytes)
            .map_err(|error| df_error(name, error))?;
        let mut indices = Vec::new();
        indices
            .try_reserve_exact(rows)
            .map_err(|error| df_error(name, error))?;
        let schema = Arc::new(Schema::new(vec![Field::new(
            "key",
            field.data_type().clone(),
            field.is_nullable(),
        )]));
        let groups = new_group_values(schema, &GroupOrdering::None)
            .map_err(|error| df_error(name, error))?;
        Ok(Self {
            groups,
            indices,
            width,
            reservation,
        })
    }

    fn intern(&mut self, array: ArrayRef, name: &str) -> Result<()> {
        let count = self
            .groups
            .len()
            .checked_add(array.len())
            .ok_or_else(|| df_error(name, "native group count overflowed"))?;
        let bytes = native_key_bytes(count, self.indices.capacity(), self.width, name)?;
        if bytes > self.reservation.size() {
            self.reservation
                .try_grow(bytes - self.reservation.size())
                .map_err(|error| df_error(name, error))?;
        }
        self.groups
            .intern(&[array], &mut self.indices)
            .map_err(|error| df_error(name, error))?;
        let actual = checked_bytes(
            self.groups.size(),
            [(self.indices.capacity(), size_of::<usize>())],
            name,
        )?;
        if actual > self.reservation.size() {
            return Err(df_error(name, "native keys exceeded reserved capacity"));
        }
        Ok(())
    }
}

fn native_key_width(data_type: &DataType) -> Option<usize> {
    match data_type {
        DataType::Boolean => Some(0),
        DataType::Int8 | DataType::UInt8 => Some(1),
        DataType::Int16 | DataType::UInt16 => Some(2),
        DataType::Int32 | DataType::UInt32 => Some(4),
        DataType::Int64 | DataType::UInt64 => Some(8),
        _ => None,
    }
}

fn native_key_bytes(count: usize, rows: usize, width: usize, name: &str) -> Result<usize> {
    let indices = checked_bytes(4096 + 128, [(rows, 2 * size_of::<usize>())], name)?;
    if width == 0 {
        return Ok(indices);
    }
    let values = count
        .max(128)
        .checked_next_power_of_two()
        .ok_or_else(|| df_error(name, "native value capacity overflowed"))?;
    let buckets = count
        .checked_mul(2)
        .and_then(|count| count.max(256).checked_next_power_of_two())
        .ok_or_else(|| df_error(name, "native bucket capacity overflowed"))?;
    checked_bytes(
        indices,
        [
            (64, 1),
            (buckets, 2 * (size_of::<(usize, u64)>() + 1)),
            (values, width),
            (values, width),
            (count.div_ceil(8), 1),
        ],
        name,
    )
}

struct PartialGroups {
    accumulators: Vec<Box<dyn GroupsAccumulator>>,
    slots: Vec<usize>,
    base_bytes: usize,
    reservation: MemoryReservation,
}

impl PartialGroups {
    fn new(
        aggregates: &[Arc<AggregateFunctionExpr>],
        reservation: MemoryReservation,
        name: &str,
    ) -> Result<Self> {
        let base_bytes = checked_bytes(4096, [(aggregates.len(), 512)], name)?;
        reservation
            .try_grow(base_bytes)
            .map_err(|error| df_error(name, error))?;
        let accumulators = aggregates
            .iter()
            .map(|expr| {
                expr.create_groups_accumulator()
                    .map_err(|error| df_error(name, error))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            accumulators,
            slots: Vec::new(),
            base_bytes,
            reservation,
        })
    }

    fn add_slot(&mut self, slot: usize, name: &str) -> Result<usize> {
        let count = self
            .slots
            .len()
            .checked_add(1)
            .ok_or_else(|| df_error(name, "partial group count overflowed"))?;
        self.reserve(count, name)?;
        let rank = self.slots.len();
        self.slots.push(slot);
        Ok(rank)
    }

    fn reserve(&mut self, count: usize, name: &str) -> Result<()> {
        let capacity = count
            .max(4)
            .checked_next_power_of_two()
            .ok_or_else(|| df_error(name, "partial group capacity overflowed"))?;
        let width = checked_bytes(size_of::<usize>(), [(self.accumulators.len(), 40)], name)?;
        let bytes = checked_bytes(self.base_bytes, [(capacity, width)], name)?;
        if bytes > self.reservation.size() {
            self.reservation
                .try_grow(bytes - self.reservation.size())
                .map_err(|error| df_error(name, error))?;
        }
        self.slots
            .try_reserve_exact(capacity - self.slots.len())
            .map_err(|error| df_error(name, error))?;
        Ok(())
    }

    fn update(
        &mut self,
        arguments: &[Vec<ArrayRef>],
        indices: &[usize],
        count: usize,
        name: &str,
    ) -> Result<()> {
        for (accumulator, arguments) in self.accumulators.iter_mut().zip(arguments) {
            accumulator
                .update_batch(arguments, indices, None, count)
                .map_err(|error| df_error(name, error))?;
        }
        let actual = self.accumulators.iter().try_fold(
            checked_bytes(
                self.base_bytes,
                [(self.slots.capacity(), size_of::<usize>())],
                name,
            )?,
            |total, accumulator| checked_bytes(total, [(accumulator.size(), 1)], name),
        )?;
        if actual > self.reservation.size() {
            return Err(df_error(
                name,
                "partial accumulator exceeded reserved capacity",
            ));
        }
        Ok(())
    }

    async fn merge(
        &mut self,
        candidates: &mut CandidateMap,
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<()> {
        context.check_cancelled()?;
        let states = self
            .accumulators
            .iter_mut()
            .map(|accumulator| {
                accumulator
                    .state(EmitTo::All)
                    .map_err(|error| df_error(name, error))
            })
            .collect::<Result<Vec<_>>>()?;
        for (rank, slot) in self.slots.iter().enumerate() {
            if rank % 128 == 0 {
                context.check_cancelled()?;
                tokio::task::yield_now().await;
            }
            let candidate = candidates
                .get_mut(slot)
                .expect("candidate for partial group");
            for (accumulator, states) in candidate.accumulators.iter_mut().zip(&states) {
                let arrays = states
                    .iter()
                    .map(|array| array.slice(rank, 1))
                    .collect::<Vec<_>>();
                accumulator
                    .merge_batch(&arrays)
                    .map_err(|error| df_error(name, error))?;
            }
        }
        Ok(())
    }
}

type CandidateMap = HashMap<usize, Candidate, RandomState>;
type KeyIndex = HashMap<Arc<[u8]>, usize, RandomState>;
type PreparedGroups = (Vec<(usize, Group)>, Vec<Option<Group>>);

pub(super) struct Transaction {
    pub records: Vec<RecordBatch>,
    #[cfg(test)]
    pub rows: usize,
    groups: Vec<(usize, Group)>,
    new_groups: Vec<Option<Group>>,
    _reservation: MemoryReservation,
}

impl IncrementalSql {
    pub async fn plan(
        runtime: &DataFusionRuntime,
        query: &ValidatedQuery,
        alias: &str,
        schema: SchemaRef,
        name: &str,
    ) -> Result<Option<Self>> {
        let (raw, analyzed) = runtime
            .incremental_sql_plan(query, alias, schema.clone(), name)
            .await?;
        let Some((_, raw_aggregate)) = shape(&raw) else {
            return Ok(None);
        };
        if raw_aggregate.aggr_expr.is_empty()
            || !raw_aggregate
                .aggr_expr
                .iter()
                .all(|expr| eligible(expr, &schema))
        {
            return Ok(None);
        }
        let keys = raw_aggregate
            .group_expr
            .iter()
            .map(|expr| {
                let Expr::Column(column) = expr else {
                    return None;
                };
                let index = schema.index_of(&column.name).ok()?;
                key_type(schema.field(index).data_type()).then_some(index)
            })
            .collect::<Option<Vec<_>>>();
        let Some(keys) = keys else { return Ok(None) };
        let variable_columns = variable_columns(&keys, &raw_aggregate.aggr_expr, &schema, name)?;
        let Some((projection, aggregate)) = shape(&analyzed) else {
            return Ok(None);
        };
        let reservation = runtime.incremental_reservation(name);
        let plan_bytes = checked_bytes(
            4096,
            [
                (keys.len(), 256),
                (aggregate.aggr_expr.len(), 1024),
                (projection.expr.len(), 512),
                (query.text().len(), 8),
            ],
            name,
        )?;
        reservation
            .try_grow(plan_bytes)
            .map_err(|error| df_error(name, error))?;
        let props = ExecutionProps::new();
        let Ok(aggregates) = aggregate
            .aggr_expr
            .iter()
            .map(|expr| {
                LoweredAggregateBuilder::new(expr, aggregate.input.schema(), &schema, &props)
                    .build()
                    .map(|lowered| lowered.aggregate)
            })
            .collect::<datafusion::error::Result<Vec<_>>>()
        else {
            return Ok(None);
        };
        let Ok(projection) = projection
            .expr
            .iter()
            .map(|expr| create_physical_expr(expr, &aggregate.schema, &props))
            .collect::<datafusion::error::Result<Vec<_>>>()
        else {
            return Ok(None);
        };
        let Some(aggregate_bytes) = aggregate_bytes(&aggregates, name)? else {
            return Ok(None);
        };
        if !keys.is_empty() && !native_groups_supported(&aggregates) {
            return Ok(None);
        }
        let converter = row_converter(&keys, &schema, name)?;
        Ok(Some(Self {
            schema,
            aggregate_schema: Arc::new(aggregate.schema.as_arrow().clone()),
            output_schema: Arc::new(analyzed.schema().as_arrow().clone()),
            aggregates,
            aggregate_bytes,
            plan_bytes,
            projection,
            keys,
            variable_columns,
            converter,
            groups: Vec::new(),
            index: HashMap::with_hasher(RandomState::new()),
            reservation,
            #[cfg(test)]
            finalized_groups: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            #[cfg(test)]
            partial_groups: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            #[cfg(test)]
            historical_key_lookups: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            #[cfg(test)]
            encoded_rows: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        }))
    }

    fn candidate(
        &self,
        previous: Option<&Group>,
        key: &[u8],
        record: Option<(&[ArrayRef], usize)>,
        name: &str,
    ) -> Result<Candidate> {
        let reservation = self.reservation.new_empty();
        let bytes = checked_bytes(
            self.aggregate_bytes,
            [
                (key.len(), 4),
                (self.keys.len(), size_of::<ScalarValue>()),
                (1, size_of::<Group>()),
                (1, size_of::<Candidate>()),
                (4, size_of::<(usize, Candidate)>()),
                (4, size_of::<(Arc<[u8]>, usize)>()),
                (1, size_of::<(usize, Group)>()),
                (8, size_of::<usize>()),
            ],
            name,
        )?;
        reservation
            .try_grow(bytes)
            .map_err(|error| df_error(name, error))?;
        let (key, values) = if let Some(previous) = previous {
            (previous.key.clone(), previous.values.clone())
        } else {
            let values = self
                .keys
                .iter()
                .enumerate()
                .map(|(index, _)| {
                    let (record, row) = record.expect("grouped candidate has a row");
                    ScalarValue::try_from_array(&record[index], row)
                        .map_err(|error| df_error(name, error))
                })
                .collect::<Result<Vec<_>>>()?;
            (Arc::from(key), Arc::from(values))
        };
        let accumulators = self
            .aggregates
            .iter()
            .enumerate()
            .map(|(index, expr)| {
                let mut accumulator = expr
                    .create_accumulator()
                    .map_err(|error| df_error(name, error))?;
                if let Some(previous) = previous {
                    let arrays = previous.states[index]
                        .iter()
                        .map(ScalarValue::to_array)
                        .collect::<datafusion::error::Result<Vec<_>>>()
                        .map_err(|error| df_error(name, error))?;
                    accumulator
                        .merge_batch(&arrays)
                        .map_err(|error| df_error(name, error))?;
                }
                Ok(accumulator)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Candidate {
            group: Group {
                key,
                values,
                states: Vec::new(),
                results: Vec::new(),
                _reservation: reservation,
            },
            accumulators,
        })
    }

    pub async fn update(
        &mut self,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<Transaction> {
        let table = batch.table_payload()?;
        if batch.num_rows() != 0 && table.schema() != &self.schema {
            return Err(CalcFlowError::InvalidArgument {
                field: "batches".into(),
                message: "schemas must match".into(),
            });
        }
        let reservation = self.reservation.new_empty();
        let width = checked_bytes(
            if self.keys.is_empty() { 0 } else { 128 },
            [(self.aggregates.len(), 32), (self.keys.len(), 24)],
            name,
        )?;
        let workspace = checked_bytes(4096, [(batch.num_rows().min(CHUNK_ROWS), width)], name)?;
        reservation
            .try_grow(workspace)
            .map_err(|error| df_error(name, error))?;
        let partial = if self.keys.is_empty() {
            None
        } else {
            Some(PartialGroups::new(
                &self.aggregates,
                self.reservation.new_empty(),
                name,
            )?)
        };
        let native = self.native_keys(batch.num_rows(), name)?;
        let mut candidates = InputCandidates {
            groups: HashMap::with_hasher(RandomState::new()),
            touched: HashMap::with_hasher(RandomState::new()),
            new_count: 0,
            native,
            partial,
        };
        if self.keys.is_empty() {
            candidates
                .groups
                .insert(0, self.candidate(self.groups.first(), &[], None, name)?);
        }
        #[cfg(test)]
        let mut rows = 0;
        for record in table.batches() {
            for offset in (0..record.num_rows()).step_by(CHUNK_ROWS) {
                context.check_cancelled()?;
                let chunk = record.slice(offset, CHUNK_ROWS.min(record.num_rows() - offset));
                self.update_chunk(&chunk, &mut candidates, &reservation, workspace, name)?;
                #[cfg(test)]
                {
                    rows += chunk.num_rows();
                }
                tokio::task::yield_now().await;
            }
        }
        let mut native = candidates.native.take();
        if let Some(native) = native.as_mut() {
            self.native_candidates(native, &mut candidates, context, name)
                .await?;
        }
        if let Some(mut partial) = candidates.partial.take() {
            partial.merge(&mut candidates.groups, context, name).await?;
        }
        drop(native);
        self.finalize_candidates(&mut candidates.groups, context, name)
            .await?;
        let new_count = candidates
            .new_count
            .max(usize::from(self.keys.is_empty() && self.groups.is_empty()));
        let count = self
            .groups
            .len()
            .checked_add(new_count)
            .ok_or_else(|| df_error(name, "incremental group count overflowed"))?;
        let records = self
            .output_records(count, &candidates.groups, &reservation, context, name)
            .await?;
        let new_count = count - self.groups.len();
        self.reserve_groups(new_count, count, name)?;
        let (groups, new_groups) = self
            .prepare_groups(candidates.groups, new_count, context)
            .await?;
        Ok(Transaction {
            records,
            #[cfg(test)]
            rows,
            groups,
            new_groups,
            _reservation: reservation,
        })
    }

    fn native_keys(&self, rows: usize, name: &str) -> Result<Option<NativeKeys>> {
        self.keys
            .first()
            .filter(|_| self.keys.len() == 1)
            .and_then(|&index| {
                native_key_width(self.schema.field(index).data_type()).map(|width| (index, width))
            })
            .map(|(index, width)| {
                NativeKeys::new(
                    self.schema.field(index),
                    rows.min(CHUNK_ROWS),
                    width,
                    self.reservation.new_empty(),
                    name,
                )
            })
            .transpose()
    }

    fn update_chunk(
        &self,
        chunk: &RecordBatch,
        candidates: &mut InputCandidates,
        reservation: &MemoryReservation,
        workspace: usize,
        name: &str,
    ) -> Result<()> {
        let variable_bytes = self.variable_columns.iter().try_fold(0, |total, &index| {
            let bytes = chunk
                .column(index)
                .to_data()
                .get_slice_memory_size()
                .map_err(|error| df_error(name, error))?;
            checked_bytes(total, [(bytes, 4)], name)
        })?;
        let needed = checked_bytes(workspace, [(variable_bytes, 1)], name)?;
        if needed > reservation.size() {
            reservation
                .try_grow(needed - reservation.size())
                .map_err(|error| df_error(name, error))?;
        }
        let arguments = self.arguments(chunk, name)?;
        if self.keys.is_empty() {
            let candidate = candidates.groups.get_mut(&0).expect("global candidate");
            for (arguments, accumulator) in arguments.iter().zip(&mut candidate.accumulators) {
                accumulator
                    .update_batch(arguments)
                    .map_err(|error| df_error(name, error))?;
            }
            return Ok(());
        }
        if let Some(native) = candidates.native.as_mut() {
            native.intern(chunk.column(self.keys[0]).clone(), name)?;
            let count = native.groups.len();
            let partial = candidates
                .partial
                .as_mut()
                .expect("grouped partial accumulators");
            partial.reserve(count, name)?;
            partial.update(&arguments, &native.indices, count, name)?;
            #[cfg(test)]
            self.partial_groups
                .fetch_max(count, std::sync::atomic::Ordering::SeqCst);
            return Ok(());
        }
        self.update_grouped_chunk(chunk, &arguments, candidates, name)
    }

    fn update_grouped_chunk(
        &self,
        chunk: &RecordBatch,
        arguments: &[Vec<ArrayRef>],
        candidates: &mut InputCandidates,
        name: &str,
    ) -> Result<()> {
        let key_arrays = self
            .keys
            .iter()
            .map(|&index| chunk.column(index).clone())
            .collect::<Vec<_>>();
        #[cfg(test)]
        self.encoded_rows
            .fetch_add(chunk.num_rows(), std::sync::atomic::Ordering::SeqCst);
        let encoded = self
            .converter
            .as_ref()
            .expect("grouped key converter")
            .convert_columns(&key_arrays)
            .map_err(|error| df_error(name, error))?;
        let partial = candidates
            .partial
            .as_mut()
            .expect("grouped partial accumulators");
        let mut indices = Vec::with_capacity(chunk.num_rows());
        for row in 0..chunk.num_rows() {
            let encoded_row = encoded.row(row);
            let key = encoded_row.as_ref();
            if let Some(&rank) = candidates.touched.get(key) {
                indices.push(rank);
                continue;
            }
            #[cfg(test)]
            self.historical_key_lookups
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let (slot, previous, new_count) = if let Some(&slot) = self.index.get(key) {
                (slot, self.groups.get(slot), candidates.new_count)
            } else {
                let slot = self
                    .groups
                    .len()
                    .checked_add(candidates.new_count)
                    .ok_or_else(|| df_error(name, "incremental group count overflowed"))?;
                let count = candidates
                    .new_count
                    .checked_add(1)
                    .ok_or_else(|| df_error(name, "new group count overflowed"))?;
                (slot, None, count)
            };
            let candidate = self.candidate(previous, key, Some((&key_arrays, row)), name)?;
            let rank = partial.add_slot(slot, name)?;
            candidates.touched.insert(candidate.group.key.clone(), rank);
            candidates.groups.insert(slot, candidate);
            candidates.new_count = new_count;
            indices.push(rank);
        }
        partial.update(arguments, &indices, partial.slots.len(), name)?;
        #[cfg(test)]
        self.partial_groups
            .fetch_max(partial.slots.len(), std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    async fn native_candidates(
        &self,
        native: &mut NativeKeys,
        candidates: &mut InputCandidates,
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<()> {
        let keys = native
            .groups
            .emit(EmitTo::All)
            .map_err(|error| df_error(name, error))?;
        let count = keys[0].len();
        for start in (0..count).step_by(CHUNK_ROWS) {
            context.check_cancelled()?;
            let rows = CHUNK_ROWS.min(count - start);
            let arrays = keys
                .iter()
                .map(|array| array.slice(start, rows))
                .collect::<Vec<_>>();
            #[cfg(test)]
            self.encoded_rows
                .fetch_add(rows, std::sync::atomic::Ordering::SeqCst);
            let encoded = self
                .converter
                .as_ref()
                .expect("grouped key converter")
                .convert_columns(&arrays)
                .map_err(|error| df_error(name, error))?;
            for row in 0..rows {
                let encoded_row = encoded.row(row);
                let key = encoded_row.as_ref();
                #[cfg(test)]
                self.historical_key_lookups
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let (slot, previous, new_count) = if let Some(&slot) = self.index.get(key) {
                    (slot, self.groups.get(slot), candidates.new_count)
                } else {
                    let slot = self
                        .groups
                        .len()
                        .checked_add(candidates.new_count)
                        .ok_or_else(|| df_error(name, "incremental group count overflowed"))?;
                    let count = candidates
                        .new_count
                        .checked_add(1)
                        .ok_or_else(|| df_error(name, "new group count overflowed"))?;
                    (slot, None, count)
                };
                let candidate = self.candidate(previous, key, Some((&arrays, row)), name)?;
                let rank = candidates
                    .partial
                    .as_mut()
                    .expect("grouped partial accumulators")
                    .add_slot(slot, name)?;
                debug_assert_eq!(rank, start + row);
                candidates.groups.insert(slot, candidate);
                candidates.new_count = new_count;
            }
            tokio::task::yield_now().await;
        }
        Ok(())
    }

    fn arguments(&self, chunk: &RecordBatch, name: &str) -> Result<Vec<Vec<ArrayRef>>> {
        self.aggregates
            .iter()
            .map(|expr| {
                expr.expressions()
                    .iter()
                    .map(|arg| {
                        arg.evaluate(chunk)
                            .and_then(|value| value.into_array(chunk.num_rows()))
                    })
                    .collect::<datafusion::error::Result<Vec<_>>>()
                    .map_err(|error| df_error(name, error))
            })
            .collect::<Result<Vec<_>>>()
    }

    async fn finalize_candidates(
        &self,
        candidates: &mut CandidateMap,
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<()> {
        for (index, candidate) in candidates.values_mut().enumerate() {
            if index % 128 == 0 {
                if index != 0 {
                    tokio::task::yield_now().await;
                }
                context.check_cancelled()?;
            }
            candidate.group.states = candidate
                .accumulators
                .iter_mut()
                .map(|accumulator| accumulator.state().map_err(|error| df_error(name, error)))
                .collect::<Result<Vec<_>>>()?;
            candidate.group.results = candidate
                .accumulators
                .iter_mut()
                .map(|accumulator| {
                    accumulator
                        .evaluate()
                        .map_err(|error| df_error(name, error))
                })
                .collect::<Result<Vec<_>>>()?;
            #[cfg(test)]
            self.finalized_groups
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        Ok(())
    }

    async fn output_records(
        &self,
        count: usize,
        candidates: &CandidateMap,
        reservation: &MemoryReservation,
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<Vec<RecordBatch>> {
        let width = checked_bytes(
            0,
            [
                (self.keys.len(), 128),
                (self.aggregates.len(), 128),
                (self.projection.len(), 128),
            ],
            name,
        )?;
        let output_charge = checked_bytes(0, [(count, width)], name)?;
        let mut output_charge = output_charge;
        for (index, group) in self.groups.iter().enumerate() {
            if index % CHUNK_ROWS == 0 {
                context.check_cancelled()?;
                tokio::task::yield_now().await;
            }
            output_charge = checked_bytes(output_charge, [(group.key.len(), 4)], name)?;
        }
        for (index, candidate) in candidates.values().enumerate() {
            if index % CHUNK_ROWS == 0 {
                context.check_cancelled()?;
                tokio::task::yield_now().await;
            }
            output_charge = checked_bytes(output_charge, [(candidate.group.key.len(), 4)], name)?;
        }
        reservation
            .try_grow(output_charge)
            .map_err(|error| df_error(name, error))?;
        let mut records = Vec::new();
        for start in (0..count).step_by(CHUNK_ROWS) {
            context.check_cancelled()?;
            let end = count.min(start.saturating_add(CHUNK_ROWS));
            let columns = (0..self.keys.len() + self.aggregates.len())
                .map(|column| {
                    ScalarValue::iter_to_array((start..end).map(|slot| {
                        let group = candidates
                            .get(&slot)
                            .map_or_else(|| &self.groups[slot], |candidate| &candidate.group);
                        if column < self.keys.len() {
                            group.values[column].clone()
                        } else {
                            group.results[column - self.keys.len()].clone()
                        }
                    }))
                    .map_err(|error| df_error(name, error))
                })
                .collect::<Result<Vec<ArrayRef>>>()?;
            let aggregate = RecordBatch::try_new(self.aggregate_schema.clone(), columns)
                .map_err(|error| df_error(name, error))?;
            let output = self
                .projection
                .iter()
                .map(|expr| {
                    expr.evaluate(&aggregate)
                        .and_then(|value| value.into_array(end - start))
                        .map_err(|error| df_error(name, error))
                })
                .collect::<Result<Vec<_>>>()?;
            records.push(
                RecordBatch::try_new(self.output_schema.clone(), output)
                    .map_err(|error| df_error(name, error))?,
            );
            tokio::task::yield_now().await;
        }
        if records.is_empty() {
            records.push(RecordBatch::new_empty(self.output_schema.clone()));
        }
        Ok(records)
    }

    fn reserve_groups(&mut self, new_count: usize, count: usize, name: &str) -> Result<()> {
        if new_count != 0 {
            let capacity = count
                .checked_next_power_of_two()
                .ok_or_else(|| df_error(name, "incremental capacity overflowed"))?;
            let charge = checked_bytes(
                self.plan_bytes,
                [
                    (capacity, size_of::<Group>()),
                    (
                        capacity,
                        4 * (size_of::<Arc<[u8]>>() + size_of::<usize>() + 1),
                    ),
                ],
                name,
            )?;
            if charge > self.reservation.size() {
                self.reservation
                    .try_grow(charge - self.reservation.size())
                    .map_err(|error| df_error(name, error))?;
            }
            self.groups
                .try_reserve_exact(new_count)
                .map_err(|error| df_error(name, error))?;
            self.index
                .try_reserve(new_count)
                .map_err(|error| df_error(name, error))?;
        }
        Ok(())
    }

    async fn prepare_groups(
        &self,
        candidates: CandidateMap,
        new_count: usize,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedGroups> {
        let mut groups = Vec::with_capacity(candidates.len() - new_count);
        let mut new_groups = (0..new_count).map(|_| None).collect::<Vec<_>>();
        for (index, (slot, candidate)) in candidates.into_iter().enumerate() {
            if index % CHUNK_ROWS == 0 {
                context.check_cancelled()?;
                tokio::task::yield_now().await;
            }
            if slot < self.groups.len() {
                groups.push((slot, candidate.group));
            } else {
                new_groups[slot - self.groups.len()] = Some(candidate.group);
            }
        }
        Ok((groups, new_groups))
    }

    #[cfg(test)]
    pub(super) fn capacity_charge(&self) -> (usize, usize, usize) {
        (
            self.groups.capacity(),
            self.index.capacity(),
            self.reservation.size(),
        )
    }

    pub fn commit(&mut self, transaction: Transaction) {
        for (slot, group) in transaction.groups {
            self.groups[slot] = group;
        }
        for group in transaction.new_groups.into_iter().flatten() {
            let slot = self.groups.len();
            self.index.insert(group.key.clone(), slot);
            self.groups.push(group);
        }
    }
}

fn row_converter(keys: &[usize], schema: &SchemaRef, name: &str) -> Result<Option<RowConverter>> {
    if keys.is_empty() {
        return Ok(None);
    }
    RowConverter::new(
        keys.iter()
            .map(|&index| SortField::new(schema.field(index).data_type().clone()))
            .collect(),
    )
    .map(Some)
    .map_err(|error| df_error(name, error))
}

fn native_groups_supported(aggregates: &[Arc<AggregateFunctionExpr>]) -> bool {
    aggregates
        .iter()
        .all(|expr| expr.groups_accumulator_supported() && expr.create_groups_accumulator().is_ok())
}

fn variable_columns(
    keys: &[usize],
    aggregates: &[Expr],
    schema: &SchemaRef,
    name: &str,
) -> Result<Vec<usize>> {
    let mut variable_columns = keys
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>();
    for expr in aggregates {
        if let Expr::AggregateFunction(function) = unalias(expr) {
            for argument in &function.params.args {
                if let Expr::Column(column) = argument {
                    variable_columns.insert(
                        schema
                            .index_of(&column.name)
                            .map_err(|error| df_error(name, error))?,
                    );
                }
            }
        }
    }
    Ok(variable_columns
        .into_iter()
        .filter(|&index| {
            matches!(
                schema.field(index).data_type(),
                DataType::Utf8 | DataType::LargeUtf8
            )
        })
        .collect::<Vec<_>>())
}

fn aggregate_bytes(aggregates: &[Arc<AggregateFunctionExpr>], name: &str) -> Result<Option<usize>> {
    let mut total = 0;
    for expr in aggregates {
        let (Ok(accumulator), Ok(fields)) = (expr.create_accumulator(), expr.state_fields()) else {
            return Ok(None);
        };
        total = checked_bytes(
            total,
            [
                (1, accumulator.size()),
                (1, size_of::<Box<dyn Accumulator>>()),
                (1, size_of::<Vec<ScalarValue>>()),
                (fields.len(), size_of::<ScalarValue>()),
                (1, size_of::<ScalarValue>()),
            ],
            name,
        )?;
    }
    Ok(Some(total))
}

fn shape(
    plan: &LogicalPlan,
) -> Option<(
    &datafusion::logical_expr::Projection,
    &datafusion::logical_expr::Aggregate,
)> {
    let LogicalPlan::Projection(projection) = plan else {
        return None;
    };
    let LogicalPlan::Aggregate(aggregate) = projection.input.as_ref() else {
        return None;
    };
    if !matches!(aggregate.input.as_ref(), LogicalPlan::TableScan(_)) {
        return None;
    }
    if !projection
        .expr
        .iter()
        .all(|expr| matches!(unalias(expr), Expr::Column(_)))
    {
        return None;
    }
    Some((projection, aggregate))
}

fn unalias(mut expr: &Expr) -> &Expr {
    while let Expr::Alias(alias) = expr {
        expr = &alias.expr;
    }
    expr
}

fn key_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Boolean
            | DataType::Utf8
            | DataType::LargeUtf8
    )
}

fn eligible(expr: &Expr, schema: &SchemaRef) -> bool {
    let Expr::AggregateFunction(function) = unalias(expr) else {
        return false;
    };
    let params = &function.params;
    if params.distinct
        || params.filter.is_some()
        || !params.order_by.is_empty()
        || params.null_treatment.is_some()
        || params.args.len() != 1
    {
        return false;
    }
    let builtin = datafusion::functions_aggregate::all_default_aggregate_functions()
        .into_iter()
        .any(|candidate| candidate.name() == function.func.name() && candidate == function.func);
    if !builtin {
        return false;
    }
    let count = function.func.name() == "count";
    if count
        && matches!(
            &params.args[0],
            Expr::Literal(ScalarValue::Int64(Some(1)), _)
        )
    {
        return true;
    }
    let Expr::Column(column) = &params.args[0] else {
        return false;
    };
    let Ok(field) = schema.field_with_name(&column.name) else {
        return false;
    };
    if count {
        key_type(field.data_type())
    } else {
        matches!(
            field.data_type(),
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
        ) && matches!(function.func.name(), "sum" | "min" | "max")
    }
}

fn df_error(name: &str, error: impl std::fmt::Display) -> CalcFlowError {
    CalcFlowError::DataFusion {
        node_id: Some(name.into()),
        message: error.to_string(),
    }
}

pub(super) fn checked_bytes(
    base: usize,
    terms: impl IntoIterator<Item = (usize, usize)>,
    name: &str,
) -> Result<usize> {
    terms.into_iter().try_fold(base, |total, (count, width)| {
        count
            .checked_mul(width)
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| df_error(name, "incremental memory charge overflowed"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sql_incremental_touched_map_growth_fits_prepaid_containers() {
        use std::mem::MaybeUninit;
        let mut touched: KeyIndex = HashMap::with_hasher(RandomState::new());
        let mut groups: HashMap<usize, MaybeUninit<Candidate>, RandomState> =
            HashMap::with_hasher(RandomState::new());
        for count in 1..=16_384_usize {
            let old_touched = touched.allocation_size();
            let old_groups = groups.allocation_size();
            touched.insert(Arc::from(count.to_le_bytes()), count);
            groups.insert(count, MaybeUninit::uninit());
            let key_bytes = touched.allocation_size();
            let group_bytes = groups.allocation_size();
            let key_peak = if key_bytes == old_touched {
                key_bytes
            } else {
                key_bytes + old_touched
            };
            let group_peak = if group_bytes == old_groups {
                group_bytes
            } else {
                group_bytes + old_groups
            };
            let prepaid = checked_bytes(
                4096,
                [
                    (count, 4 * size_of::<(Arc<[u8]>, usize)>()),
                    (count, 4 * size_of::<(usize, Candidate)>()),
                    (count, size_of::<[usize; 8]>()),
                ],
                "totals",
            )
            .unwrap();
            assert!(
                key_peak + group_peak <= prepaid,
                "table growth exceeded reserved containers at {count} groups"
            );
        }
    }

    #[tokio::test]
    async fn test_sql_incremental_native_snapshots_preserve_accumulators() {
        let values = [
            ScalarValue::Int8(Some(1)),
            ScalarValue::Int16(Some(1)),
            ScalarValue::Int32(Some(1)),
            ScalarValue::Int64(Some(1)),
            ScalarValue::UInt8(Some(1)),
            ScalarValue::UInt16(Some(1)),
            ScalarValue::UInt32(Some(1)),
            ScalarValue::UInt64(Some(1)),
            ScalarValue::Boolean(Some(true)),
            ScalarValue::Utf8(Some("a".into())),
            ScalarValue::LargeUtf8(Some("a".into())),
        ];
        let record =
            RecordBatch::try_from_iter(values.iter().enumerate().map(|(index, value)| {
                (format!("value_{index}"), value.to_array_of_size(2).unwrap())
            }))
            .unwrap();
        let mut expressions = vec!["COUNT(*)".into()];
        for index in 0..values.len() {
            expressions.push(format!("COUNT(value_{index})"));
            if index < 8 {
                expressions.extend(
                    ["SUM", "MIN", "MAX"].map(|function| format!("{function}(value_{index})")),
                );
            }
        }
        let query = crate::expression::parse_select_query(&format!(
            "SELECT {} FROM events",
            expressions.join(", ")
        ))
        .unwrap();
        let runtime = DataFusionRuntime::new(crate::DataFusionConfig::default()).unwrap();
        let plan = IncrementalSql::plan(&runtime, &query, "events", record.schema(), "totals")
            .await
            .unwrap()
            .unwrap();
        let arguments = plan.arguments(&record, "totals").unwrap();
        for (expression, arguments) in plan.aggregates.iter().zip(arguments) {
            let mut accumulator = expression.create_accumulator().unwrap();
            accumulator.update_batch(&arguments).unwrap();
            let state = accumulator.state().unwrap();
            let value = accumulator.evaluate().unwrap();
            assert_eq!(accumulator.state().unwrap(), state);
            assert_eq!(accumulator.evaluate().unwrap(), value);
            accumulator.update_batch(&arguments).unwrap();
            let updated = accumulator.state().unwrap();
            let value = accumulator.evaluate().unwrap();
            assert_eq!(accumulator.state().unwrap(), updated);
            assert_eq!(accumulator.evaluate().unwrap(), value);
        }
    }

    fn partial_group_fixture() -> (RecordBatch, String) {
        let maximums = [
            ScalarValue::Int8(Some(i8::MAX)),
            ScalarValue::Int16(Some(i16::MAX)),
            ScalarValue::Int32(Some(i32::MAX)),
            ScalarValue::Int64(Some(i64::MAX)),
            ScalarValue::UInt8(Some(u8::MAX)),
            ScalarValue::UInt16(Some(u16::MAX)),
            ScalarValue::UInt32(Some(u32::MAX)),
            ScalarValue::UInt64(Some(u64::MAX)),
            ScalarValue::Boolean(Some(true)),
            ScalarValue::Utf8(Some("a".into())),
            ScalarValue::LargeUtf8(Some("a".into())),
        ];
        let mut columns = vec![(
            "key".into(),
            Arc::new(datafusion::arrow::array::Int64Array::from(vec![0, 1, 0, 2])) as ArrayRef,
        )];
        let mut expressions = vec!["key".into(), "COUNT(*)".into()];
        for (index, value) in maximums.iter().enumerate() {
            let null = ScalarValue::try_from(&value.data_type()).unwrap();
            let one = if index < 8 {
                ScalarValue::Int64(Some(1))
                    .cast_to(&value.data_type())
                    .unwrap()
            } else {
                value.clone()
            };
            columns.push((
                format!("value_{index}"),
                ScalarValue::iter_to_array([value.clone(), null.clone(), one, null]).unwrap(),
            ));
            expressions.push(format!("COUNT(value_{index})"));
            if index < 8 {
                expressions.extend(
                    ["SUM", "MIN", "MAX"].map(|function| format!("{function}(value_{index})")),
                );
            }
        }
        let record = RecordBatch::try_from_iter(columns).unwrap();
        let query_text = format!("SELECT {} FROM events GROUP BY key", expressions.join(", "));
        (record, query_text)
    }

    #[test]
    fn test_sql_incremental_native_key_charge_overflow_returns_error() {
        let rows = (usize::MAX - 4224) / (2 * size_of::<usize>());
        assert!(native_key_bytes(0, rows, 8, "totals").is_err());
        assert!(native_key_bytes(usize::MAX, 0, 8, "totals").is_err());
    }

    #[test]
    fn test_sql_incremental_native_key_credit_covers_primitive_allocation_peaks() {
        let mut map = hashbrown::HashTable::<(usize, u64)>::with_capacity(128);
        let mut values = Vec::<u64>::with_capacity(128);
        for count in 1..=16384 {
            let old_map = map.allocation_size();
            let old_values = values.capacity() * size_of::<u64>();
            map.insert_unique(count as u64, (count - 1, count as u64), |entry| entry.1);
            values.push(count as u64);
            let new_map = map.allocation_size();
            let new_values = values.capacity() * size_of::<u64>();
            let map_peak = if new_map == old_map {
                new_map
            } else {
                old_map + new_map
            };
            let values_peak = if new_values == old_values {
                new_values
            } else {
                old_values + new_values
            };
            let peak = map_peak + values_peak;
            assert!(
                peak <= native_key_bytes(count, 1, 8, "totals").unwrap(),
                "unfunded native key capacity at {count}: {peak}"
            );
        }
    }

    #[test]
    fn test_sql_incremental_native_key_factory_preserves_dense_nullable_ranks() {
        use datafusion::{
            arrow::datatypes::{Field, Schema},
            physical_plan::aggregates::{group_values::new_group_values, order::GroupOrdering},
        };
        let cases = vec![
            (
                ScalarValue::Int8(Some(i8::MIN)),
                ScalarValue::Int8(Some(i8::MAX)),
                ScalarValue::Int8(Some(0)),
            ),
            (
                ScalarValue::Int16(Some(i16::MIN)),
                ScalarValue::Int16(Some(i16::MAX)),
                ScalarValue::Int16(Some(0)),
            ),
            (
                ScalarValue::Int32(Some(i32::MIN)),
                ScalarValue::Int32(Some(i32::MAX)),
                ScalarValue::Int32(Some(0)),
            ),
            (
                ScalarValue::Int64(Some(i64::MIN)),
                ScalarValue::Int64(Some(i64::MAX)),
                ScalarValue::Int64(Some(0)),
            ),
            (
                ScalarValue::UInt8(Some(0)),
                ScalarValue::UInt8(Some(u8::MAX)),
                ScalarValue::UInt8(Some(1)),
            ),
            (
                ScalarValue::UInt16(Some(0)),
                ScalarValue::UInt16(Some(u16::MAX)),
                ScalarValue::UInt16(Some(1)),
            ),
            (
                ScalarValue::UInt32(Some(0)),
                ScalarValue::UInt32(Some(u32::MAX)),
                ScalarValue::UInt32(Some(1)),
            ),
            (
                ScalarValue::UInt64(Some(0)),
                ScalarValue::UInt64(Some(u64::MAX)),
                ScalarValue::UInt64(Some(1)),
            ),
            (
                ScalarValue::Boolean(Some(false)),
                ScalarValue::Boolean(Some(true)),
                ScalarValue::Boolean(Some(false)),
            ),
        ];
        for (low, high, other) in cases {
            let data_type = low.data_type();
            let null = ScalarValue::try_from(&data_type).unwrap();
            let first =
                ScalarValue::iter_to_array([null.clone(), low.clone(), low.clone(), high.clone()])
                    .unwrap();
            let second = ScalarValue::iter_to_array([
                high.clone(),
                other.clone(),
                null.clone(),
                low.clone(),
            ])
            .unwrap();
            let schema = Arc::new(Schema::new(vec![Field::new(
                "key",
                data_type.clone(),
                true,
            )]));
            let mut keys = new_group_values(schema, &GroupOrdering::None).unwrap();
            let mut indices = Vec::new();
            keys.intern(&[first], &mut indices).unwrap();
            assert_eq!(indices, [0, 1, 1, 2]);
            assert_eq!(keys.len(), 3);
            keys.intern(&[second.slice(0, 0)], &mut indices).unwrap();
            assert!(indices.is_empty());
            keys.intern(&[second], &mut indices).unwrap();
            let boolean = data_type == DataType::Boolean;
            assert_eq!(indices, [2, if boolean { 1 } else { 3 }, 0, 1]);
            assert_eq!(keys.len(), if boolean { 3 } else { 4 });
            let emitted = keys.emit(EmitTo::All).unwrap();
            assert_eq!(emitted.len(), 1);
            assert_eq!(emitted[0].data_type(), &data_type);
            let mut expected = vec![null, low, high];
            if !boolean {
                expected.push(other);
            }
            let actual = (0..emitted[0].len())
                .map(|row| ScalarValue::try_from_array(&emitted[0], row).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
            assert_eq!(keys.len(), 0);
        }
    }

    #[tokio::test]
    async fn test_sql_incremental_native_partial_groups_merge_scalar_state() {
        use datafusion::logical_expr::EmitTo;
        let (record, query_text) = partial_group_fixture();
        let query = crate::expression::parse_select_query(&query_text).unwrap();
        let runtime = DataFusionRuntime::new(crate::DataFusionConfig::default()).unwrap();
        let plan = IncrementalSql::plan(&runtime, &query, "events", record.schema(), "totals")
            .await
            .unwrap()
            .unwrap();
        let seed = plan.arguments(&record.slice(0, 1), "totals").unwrap();
        let first = plan.arguments(&record.slice(1, 1), "totals").unwrap();
        let second = plan.arguments(&record.slice(2, 2), "totals").unwrap();
        let mut results = [Vec::new(), Vec::new(), Vec::new()];
        for (index, expression) in plan.aggregates.iter().enumerate() {
            assert!(expression.groups_accumulator_supported());
            let mut partial = expression.create_groups_accumulator().unwrap();
            partial.update_batch(&first[index], &[1], None, 2).unwrap();
            partial
                .update_batch(&second[index], &[0, 2], None, 3)
                .unwrap();
            let states = partial.state(EmitTo::All).unwrap();
            let fields = expression.state_fields().unwrap();
            assert_eq!(states.len(), fields.len());
            for (state, field) in states.iter().zip(fields) {
                assert_eq!(state.data_type(), field.data_type());
                assert_eq!(state.len(), 3);
            }
            for (rank, result) in results.iter_mut().enumerate() {
                let mut scalar = expression.create_accumulator().unwrap();
                if rank == 0 {
                    scalar.update_batch(&seed[index]).unwrap();
                }
                scalar
                    .merge_batch(
                        &states
                            .iter()
                            .map(|array| array.slice(rank, 1))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                result.push(scalar.evaluate().unwrap());
            }
        }
        let mut columns =
            vec![Arc::new(datafusion::arrow::array::Int64Array::from(vec![0, 1, 2])) as ArrayRef];
        for column in 0..plan.aggregates.len() {
            columns.push(
                ScalarValue::iter_to_array(results.iter().map(|row| row[column].clone())).unwrap(),
            );
        }
        let aggregate = RecordBatch::try_new(plan.aggregate_schema.clone(), columns).unwrap();
        let actual = RecordBatch::try_new(
            plan.output_schema.clone(),
            plan.projection
                .iter()
                .map(|expr| expr.evaluate(&aggregate).unwrap().into_array(3).unwrap())
                .collect(),
        )
        .unwrap();
        let expected = runtime
            .sql(
                &query_text,
                &std::collections::BTreeMap::from([(
                    "events".into(),
                    Batch::table(vec![record], crate::BatchMetadata::default()).unwrap(),
                )]),
                None,
            )
            .await
            .unwrap();
        let mut expected_rows = (0..3)
            .map(|row| {
                expected.table_payload().unwrap().batches()[0]
                    .columns()
                    .iter()
                    .map(|column| ScalarValue::try_from_array(column, row).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        expected_rows.sort_by_key(|row| row[0].to_string());
        let actual_rows = (0..3)
            .map(|row| {
                actual
                    .columns()
                    .iter()
                    .map(|column| ScalarValue::try_from_array(column, row).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual.schema(),
            expected.table_payload().unwrap().schema().clone()
        );
        assert_eq!(actual_rows, expected_rows);
    }
}
