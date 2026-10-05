use std::{mem::size_of, sync::Arc};

use ahash::RandomState;
use datafusion::{
    arrow::{
        array::{ArrayRef, BooleanArray},
        datatypes::{DataType, Field, Schema, SchemaRef},
        record_batch::RecordBatch,
        row::{RowConverter, SortField},
    },
    common::{DFSchema, ScalarValue},
    execution::memory_pool::MemoryReservation,
    logical_expr::{
        Accumulator, EmitTo, Expr, GroupsAccumulator, LogicalPlan, execution_props::ExecutionProps,
    },
    physical_expr::{
        PhysicalExpr,
        aggregate::{AggregateFunctionExpr, LoweredAggregateBuilder},
        create_physical_expr,
        expressions::Column,
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

#[path = "compact_state.rs"]
pub(super) mod compact_state;

#[path = "grouped_float.rs"]
pub(in crate::operator::sql) mod grouped_float;

#[path = "grouped_sum.rs"]
mod grouped_sum;

#[path = "predicate.rs"]
mod predicate;

#[path = "dirty.rs"]
mod dirty;

#[path = "variable_extrema.rs"]
mod variable_extrema;

#[path = "global_record.rs"]
pub(in crate::operator::sql) mod global_record;

pub(super) struct IncrementalSql {
    schema: SchemaRef,
    aggregate_schema: SchemaRef,
    output_schema: SchemaRef,
    aggregates: Vec<Arc<AggregateFunctionExpr>>,
    filter_columns: Vec<Option<usize>>,
    predicate: Option<predicate::InputPredicate>,
    aggregate_bytes: usize,
    finalizer_bytes: usize,
    plan_bytes: usize,
    projection: Vec<Arc<dyn PhysicalExpr>>,
    keys: Vec<usize>,
    variable_columns: Vec<usize>,
    variable_extrema: Vec<usize>,
    converter: Option<RowConverter>,
    index: HashMap<Arc<[u8]>, usize, RandomState>,
    groups: Vec<Group>,
    dirty: dirty::DirtyGroups,
    sequential: Option<grouped_float::Proof>,
    global_records: Option<global_record::Proof>,
    container_fee: Option<MemoryReservation>,
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
    reservation: MemoryReservation,
}

struct Candidate {
    accumulators: Vec<Option<Box<dyn Accumulator>>>,
    group: Group,
    variable: Option<Box<variable_extrema::Bounds>>,
}

struct InputCandidates {
    touched: KeyIndex,
    groups: CandidateMap,
    new_count: usize,
    native: Option<NativeKeys>,
    partial: Option<PartialGroups>,
    proof: Option<grouped_float::Proof>,
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
        ensure_reservation(&self.reservation, bytes, name)?;
        self.groups
            .intern(&[array], &mut self.indices)
            .map_err(|error| df_error(name, error))?;
        self.validate_capacity(name)
    }

    fn validate_capacity(&self, name: &str) -> Result<()> {
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
    accumulators: Vec<grouped_sum::PartialAccumulator>,
    slots: Vec<usize>,
    sequential: Vec<bool>,
    seeded: usize,
    base_bytes: usize,
    group_bytes: usize,
    state_fields: usize,
    variable_bytes: usize,
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
        let (group_bytes, state_fields) = native_state_charge(aggregates, name)?;
        let accumulators = aggregates
            .iter()
            .map(|expr| grouped_sum::PartialAccumulator::new(expr, name))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            accumulators,
            slots: Vec::new(),
            sequential: vec![false; aggregates.len()],
            seeded: 0,
            base_bytes,
            group_bytes,
            state_fields,
            variable_bytes: 0,
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
        let capacity = if self.sequential.iter().any(|selected| *selected) {
            grouped_float::capacity(count, name)?
        } else {
            count
                .max(4)
                .checked_next_power_of_two()
                .ok_or_else(|| df_error(name, "partial group capacity overflowed"))?
        };
        let bitmap_bytes = checked_bytes(0, [(capacity.div_ceil(512), 64)], name)?;
        let bitmap_copies = checked_bytes(0, [(self.state_fields, 3)], name)?;
        let bytes = checked_bytes(
            self.base_bytes,
            [
                (capacity, self.group_bytes),
                (bitmap_bytes, bitmap_copies),
                (self.variable_bytes, variable_extrema::STATE_STRING_FACTOR),
            ],
            name,
        )?;
        ensure_reservation(&self.reservation, bytes, name)?;
        self.slots
            .try_reserve_exact(capacity - self.slots.len())
            .map_err(|error| df_error(name, error))?;
        Ok(())
    }

    fn reserve_variable(&mut self, growth: usize, name: &str) -> Result<()> {
        if growth != 0 {
            self.variable_bytes = checked_bytes(self.variable_bytes, [(growth, 1)], name)?;
            self.reserve(self.slots.len(), name)?;
        }
        Ok(())
    }

    fn seed(
        &mut self,
        candidates: &CandidateMap,
        aggregates: &[Arc<AggregateFunctionExpr>],
        name: &str,
    ) -> Result<()> {
        if !self.sequential.iter().any(|selected| *selected) {
            return Ok(());
        }
        let slots = &self.slots;
        for (index, _) in aggregates.iter().enumerate() {
            if self.sequential[index] {
                let saved = (self.seeded..slots.len()).map(|rank| {
                    (
                        rank,
                        candidates[&slots[rank]].group.states[index].as_slice(),
                    )
                });
                self.accumulators[index].seed_batch(saved, slots.len(), name)?;
            }
        }
        self.seeded = self.slots.len();
        Ok(())
    }

    fn update(
        &mut self,
        arguments: &[Vec<ArrayRef>],
        filters: &[Option<&BooleanArray>],
        indices: &[usize],
        count: usize,
        name: &str,
    ) -> Result<()> {
        for (index, (accumulator, arguments)) in
            self.accumulators.iter_mut().zip(arguments).enumerate()
        {
            accumulator
                .update_batch(
                    arguments,
                    indices,
                    filters.get(index).copied().flatten(),
                    count,
                )
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
                format!(
                    "partial accumulator needs {actual} bytes; reserved {}",
                    self.reservation.size()
                ),
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
            for (index, arrays) in states.iter().enumerate() {
                if self.sequential[index] {
                    let state = arrays
                        .iter()
                        .map(|array| {
                            ScalarValue::try_from_array(array, rank)
                                .map_err(|error| df_error(name, error))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    candidate.group.results[index] = grouped_sum::result(&state, name)?;
                    candidate.group.states[index] = state;
                } else {
                    let arrays = arrays
                        .iter()
                        .map(|array| array.slice(rank, 1))
                        .collect::<Vec<_>>();
                    candidate.accumulators[index]
                        .as_mut()
                        .expect("summary accumulator")
                        .merge_batch(&arrays)
                        .map_err(|error| df_error(name, error))?;
                }
            }
        }
        Ok(())
    }
}

fn grouped_finalizer_charge(
    keys: &[usize],
    aggregates: &[Arc<AggregateFunctionExpr>],
    name: &str,
) -> Result<usize> {
    if keys.is_empty() {
        return Ok(0);
    }
    aggregates
        .iter()
        .filter(|expr| expr.fun().name() == "avg")
        .try_fold(0_usize, |maximum, expr| {
            let fields = expr.state_fields().map_err(|error| df_error(name, error))?;
            let buffers = fields.iter().try_fold(4096, |bytes, field| {
                let width = primitive_buffer_charge(field.data_type(), name)?;
                checked_bytes(bytes, [(3, width), (3, 64)], name)
            })?;
            let result = primitive_buffer_charge(expr.field().data_type(), name)?;
            Ok(maximum.max(checked_bytes(buffers, [(3, result), (3, 64)], name)?))
        })
}

fn primitive_buffer_charge(data_type: &DataType, name: &str) -> Result<usize> {
    let width = data_type
        .primitive_width()
        .ok_or_else(|| df_error(name, "native SQL result has variable-width state"))?;
    Ok(checked_bytes(0, [(4, width)], name)?.max(64))
}

fn native_grouped_result(
    expression: &AggregateFunctionExpr,
    state: &[ScalarValue],
    name: &str,
) -> Result<ScalarValue> {
    let count = if state[0] == ScalarValue::UInt64(Some(0)) && state[1].is_null() {
        ScalarValue::UInt64(None)
    } else {
        state[0].clone()
    };
    let arrays = vec![
        count.to_array().map_err(|error| df_error(name, error))?,
        state[1].to_array().map_err(|error| df_error(name, error))?,
    ];
    let mut native = expression
        .create_groups_accumulator()
        .map_err(|error| df_error(name, error))?;
    native
        .merge_batch(&arrays, &[0], None, 1)
        .map_err(|error| df_error(name, error))?;
    let result = native
        .evaluate(EmitTo::All)
        .map_err(|error| df_error(name, error))?;
    ScalarValue::try_from_array(&result, 0).map_err(|error| df_error(name, error))
}

fn native_state_charge(
    aggregates: &[Arc<AggregateFunctionExpr>],
    name: &str,
) -> Result<(usize, usize)> {
    aggregates
        .iter()
        .try_fold((2 * size_of::<usize>(), 0), |(bytes, count), aggregate| {
            let fields = aggregate
                .state_fields()
                .map_err(|error| df_error(name, error))?;
            let count = checked_bytes(count, [(fields.len(), 1)], name)?;
            let bytes = fields.iter().try_fold(bytes, |bytes, field| {
                let width = variable_extrema::state_width(field.data_type(), name)?;
                checked_bytes(bytes, [(3, width)], name)
            })?;
            Ok((bytes, count))
        })
}

type CandidateMap = HashMap<usize, Candidate, RandomState>;
type KeyIndex = HashMap<Arc<[u8]>, usize, RandomState>;
type PreparedGroups = (Vec<(usize, Group)>, Vec<Option<Group>>);
type PreparedTransaction = (
    Vec<RecordBatch>,
    Vec<(usize, Group)>,
    Vec<Option<Group>>,
    Option<GroupContainer>,
);

struct GroupContainer {
    groups: Vec<Group>,
    index: KeyIndex,
    reservation: MemoryReservation,
}

pub(super) struct Transaction {
    container: Option<GroupContainer>,
    dirty: Option<dirty::DirtyGroups>,
    pub records: Vec<RecordBatch>,
    #[cfg(test)]
    pub rows: usize,
    groups: Vec<(usize, Group)>,
    track_updates: bool,
    new_groups: Vec<Option<Group>>,
    _reservation: MemoryReservation,
    proof: Option<grouped_float::Proof>,
}

impl Transaction {
    pub(super) fn changes_state(&self) -> bool {
        (self.track_updates && !self.groups.is_empty())
            || self.new_groups.iter().any(Option::is_some)
    }
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
        Self::from_plan(runtime, query, schema, &raw, &analyzed, name)
    }

    pub(super) fn plan_sync(
        runtime: &DataFusionRuntime,
        query: &ValidatedQuery,
        alias: &str,
        logical_schema: SchemaRef,
        physical_schema: SchemaRef,
        name: &str,
    ) -> Result<Option<Self>> {
        let plans = prepare_sync_plan(runtime, query, alias, logical_schema, name)?;
        Self::from_plan(
            runtime,
            query,
            physical_schema,
            &plans.raw,
            &plans.analyzed,
            name,
        )
    }

    pub(super) fn from_plan(
        runtime: &DataFusionRuntime,
        query: &ValidatedQuery,
        schema: SchemaRef,
        raw: &LogicalPlan,
        analyzed: &LogicalPlan,
        name: &str,
    ) -> Result<Option<Self>> {
        let Some((keys, variable_columns)) = plan_inputs(raw, &schema, name)? else {
            return Ok(None);
        };
        let Some((projection, aggregate)) = shape(analyzed) else {
            return Ok(None);
        };
        let reservation = runtime.incremental_reservation(name);
        let rebound_fields = if aggregate.input.schema().as_arrow() == schema.as_ref() {
            0
        } else {
            schema.fields().len()
        };
        let plan_bytes = checked_bytes(
            4096,
            [
                (keys.len(), 256),
                (aggregate.aggr_expr.len(), 1024),
                (projection.expr.len(), 512),
                (predicate::plan_nodes(&aggregate.input, &schema), 512),
                (query.text().len(), 8),
                (rebound_fields, 512),
            ],
            name,
        )?;
        reservation
            .try_grow(plan_bytes)
            .map_err(|error| df_error(name, error))?;
        let Some((aggregates, projection, filter_columns, predicate)) =
            physical_plan(projection, aggregate, &schema)
        else {
            return Ok(None);
        };
        let Some(aggregate_bytes) = aggregate_bytes(&aggregates, name)? else {
            return Ok(None);
        };
        if !grouped_aggregates_supported(&keys, &aggregates) {
            return Ok(None);
        }
        let global_records = if global_record::raw_selected(raw, &schema) {
            global_record::Proof::new(runtime, &aggregates, name)?
        } else {
            None
        };
        let sequential = if global_records.is_some() {
            None
        } else {
            match initial_grouped_proof(runtime, &reservation, &keys, &schema, &aggregates, name)? {
                GroupStrategy::Unsupported => return Ok(None),
                GroupStrategy::Exact => None,
                GroupStrategy::Sequential(proof) => Some(proof),
            }
        };
        let (converter, finalizer_bytes) = grouped_layout(&keys, &aggregates, &schema, name)?;
        let variable_extrema = aggregates
            .iter()
            .enumerate()
            .filter_map(|(index, expression)| {
                matches!(
                    expression.field().data_type(),
                    DataType::Utf8 | DataType::LargeUtf8
                )
                .then_some(index)
            })
            .collect();
        Ok(Some(Self {
            schema,
            aggregate_schema: Arc::new(aggregate.schema.as_arrow().clone()),
            output_schema: Arc::new(analyzed.schema().as_arrow().clone()),
            aggregates,
            filter_columns,
            predicate,
            aggregate_bytes,
            finalizer_bytes,
            plan_bytes,
            projection,
            keys,
            variable_columns,
            variable_extrema,
            converter,
            groups: Vec::new(),
            dirty: dirty::DirtyGroups::empty(reservation.new_empty()),
            sequential,
            global_records,
            container_fee: None,
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

    pub(super) fn requires_grouped_float_proof(&self) -> bool {
        self.sequential.is_some()
    }

    pub(super) fn requires_global_record_proof(&self) -> bool {
        self.global_records.is_some()
    }

    fn native_policy(&self) -> &'static str {
        if self.requires_global_record_proof() {
            "global-record-float-v1"
        } else if self.requires_grouped_float_proof() {
            "sequential-grouped-float-v1"
        } else {
            "exact-numeric-v1"
        }
    }

    pub(in crate::operator::sql) fn checkpoint_policy(&self) -> grouped_float::Policy {
        if let Some(global) = &self.global_records {
            return grouped_float::Policy::GlobalRecordFloatV1(global.policy.clone());
        }
        self.sequential
            .as_ref()
            .map_or(grouped_float::Policy::ExactNumericV1, |proof| {
                grouped_float::Policy::SequentialGroupedFloatV1(proof.policy.clone())
            })
    }

    pub(in crate::operator::sql) fn restore_grouped_proof(
        &mut self,
        policy: &grouped_float::Policy,
        groups: usize,
        rows: u64,
        name: &str,
    ) -> Result<()> {
        policy.validate(&self.checkpoint_policy(), rows, name)?;
        if !self.groups.is_empty() || !self.index.is_empty() {
            return Err(df_error(name, "restored proof requires an empty candidate"));
        }
        if let grouped_float::Policy::SequentialGroupedFloatV1(policy) = policy {
            let layout = grouped_float::group_layout(&self.keys, &self.schema)
                .expect("certified grouped key");
            let rows =
                usize::try_from(policy.max_record_rows).map_err(|error| df_error(name, error))?;
            self.sequential = Some(grouped_float::Proof::new(
                self.reservation.new_empty(),
                policy.config,
                groups,
                rows,
                layout,
                &self.aggregates,
                name,
            )?);
        }
        Ok(())
    }

    fn prepare_grouped_proof(
        &self,
        records: &[RecordBatch],
        name: &str,
    ) -> Result<Option<grouped_float::Proof>> {
        self.sequential
            .as_ref()
            .map(|previous| {
                let previous_rows = usize::try_from(previous.policy.max_record_rows)
                    .map_err(|error| df_error(name, error))?;
                let rows = records
                    .iter()
                    .map(RecordBatch::num_rows)
                    .fold(previous_rows, usize::max);
                let layout = grouped_float::group_layout(&self.keys, &self.schema)
                    .expect("certified grouped key");
                grouped_float::Proof::new(
                    self.reservation.new_empty(),
                    previous.policy.config,
                    self.groups.len(),
                    rows,
                    layout,
                    &self.aggregates,
                    name,
                )
            })
            .transpose()
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
        let variable = variable_extrema::Bounds::new(
            &self.variable_extrema,
            previous,
            bytes,
            &reservation,
            name,
        )?;
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
        let accumulators = self.candidate_accumulators(previous, name)?;
        Ok(Candidate {
            group: Group {
                key,
                values,
                states: previous.map_or_else(
                    || vec![Vec::new(); self.aggregates.len()],
                    |previous| previous.states.clone(),
                ),
                results: vec![ScalarValue::Null; self.aggregates.len()],
                reservation,
            },
            accumulators,
            variable,
        })
    }

    fn candidate_accumulators(
        &self,
        previous: Option<&Group>,
        name: &str,
    ) -> Result<Vec<Option<Box<dyn Accumulator>>>> {
        let accumulators = self
            .aggregates
            .iter()
            .enumerate()
            .map(|(index, expr)| {
                if self.global_records.is_some()
                    || (self.sequential.is_some() && grouped_float::selected(expr))
                {
                    return Ok(None);
                }
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
                Ok(Some(accumulator))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(accumulators)
    }

    pub async fn update(
        &mut self,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<Transaction> {
        self.update_with_input_owner(batch, None, context, name)
            .await
    }

    pub(super) async fn update_with_input_owner(
        &mut self,
        batch: &Batch,
        input_owner: Option<Arc<MemoryReservation>>,
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<Transaction> {
        let table = batch.table_payload()?;
        self.validate_input_schema(batch.num_rows(), table.schema())?;
        let proof = self.prepare_grouped_proof(table.batches(), name)?;
        let (reservation, workspace, mut candidates) =
            self.input_candidates(batch.num_rows(), name)?;
        candidates.proof = proof;
        let rows_processed = if let Some(global) = &self.global_records {
            let candidate = candidates.groups.get_mut(&0).expect("global candidate");
            let values = global
                .update(
                    (table.batches(), input_owner),
                    (&self.aggregates, &candidate.group.states),
                    self.reservation.new_empty(),
                    context,
                    name,
                )
                .await?;
            for (index, state) in values.into_iter().enumerate() {
                candidate.group.results[index] = global.result(index, &state, name)?;
                candidate.group.states[index] = state;
            }
            batch.num_rows()
        } else {
            self.update_records(
                table.batches(),
                &mut candidates,
                (&reservation, workspace),
                context,
                name,
            )
            .await?
        };
        #[cfg(not(test))]
        let _ = rows_processed;
        self.finish_candidates(&mut candidates, context, name)
            .await?;
        let proof = candidates.proof.take();
        let (records, groups, new_groups, container) = self
            .prepare_transaction(candidates, &reservation, context, name)
            .await?;
        let dirty = self
            .dirty
            .prepare(self.groups.len() + new_groups.len(), name)?;
        Ok(Transaction {
            records,
            track_updates: batch.num_rows() != 0,
            dirty,
            #[cfg(test)]
            rows: rows_processed,
            groups,
            new_groups,
            _reservation: reservation,
            proof,
            container,
        })
    }

    async fn prepare_transaction(
        &mut self,
        candidates: InputCandidates,
        reservation: &MemoryReservation,
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<PreparedTransaction> {
        let count = self.output_count(&candidates, name)?;
        let records = self
            .output_records(count, &candidates.groups, reservation, context, name)
            .await?;
        let new_count = count - self.groups.len();
        let grow =
            new_count != 0 && (count > self.groups.capacity() || count > self.index.capacity());
        let container = grow
            .then(|| self.prepare_container(&candidates.groups, count, name))
            .transpose()?;
        let (groups, new_groups) = self
            .prepare_groups(candidates.groups, new_count, context)
            .await?;
        Ok((records, groups, new_groups, container))
    }

    fn validate_input_schema(&self, rows: usize, schema: &SchemaRef) -> Result<()> {
        if rows != 0 && schema != &self.schema {
            return Err(CalcFlowError::InvalidArgument {
                field: "batches".into(),
                message: "schemas must match".into(),
            });
        }
        Ok(())
    }

    fn input_candidates(
        &self,
        rows: usize,
        name: &str,
    ) -> Result<(MemoryReservation, usize, InputCandidates)> {
        let (reservation, workspace) = self.input_workspace(rows, name)?;
        let mut partial = if self.keys.is_empty() {
            None
        } else {
            Some(PartialGroups::new(
                &self.aggregates,
                self.reservation.new_empty(),
                name,
            )?)
        };
        if self.sequential.is_some() {
            if let Some(partial) = &mut partial {
                partial.sequential = self
                    .aggregates
                    .iter()
                    .map(|expression| grouped_float::selected(expression))
                    .collect();
            }
        }
        let native = if self.sequential.is_some() {
            None
        } else {
            self.native_keys(rows, name)?
        };
        let mut candidates = InputCandidates {
            groups: HashMap::with_hasher(RandomState::new()),
            touched: HashMap::with_hasher(RandomState::new()),
            new_count: 0,
            native,
            partial,
            proof: None,
        };
        if self.keys.is_empty() {
            candidates
                .groups
                .insert(0, self.candidate(self.groups.first(), &[], None, name)?);
        }
        Ok((reservation, workspace, candidates))
    }

    fn input_workspace(&self, rows: usize, name: &str) -> Result<(MemoryReservation, usize)> {
        let reservation = self.reservation.new_empty();
        let width = checked_bytes(
            if self.keys.is_empty() { 0 } else { 128 },
            [(self.aggregates.len(), 32), (self.keys.len(), 24)],
            name,
        )?;
        let workspace = checked_bytes(
            4096,
            [(self.finalizer_bytes, 1), (rows.min(CHUNK_ROWS), width)],
            name,
        )?;
        let workspace = match &self.predicate {
            Some(predicate) => checked_bytes(
                workspace,
                [(
                    predicate.workspace(rows.min(CHUNK_ROWS), self.aggregates.len(), name)?,
                    1,
                )],
                name,
            )?,
            None => workspace,
        };
        reservation
            .try_grow(workspace)
            .map_err(|error| df_error(name, error))?;
        Ok((reservation, workspace))
    }

    async fn update_records(
        &self,
        records: &[RecordBatch],
        candidates: &mut InputCandidates,
        workspace: (&MemoryReservation, usize),
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<usize> {
        #[cfg(test)]
        let mut rows = 0;
        #[cfg(not(test))]
        let rows = 0;
        for record in records {
            for offset in (0..record.num_rows()).step_by(CHUNK_ROWS) {
                context.check_cancelled()?;
                let chunk = record.slice(offset, CHUNK_ROWS.min(record.num_rows() - offset));
                self.update_chunk(&chunk, candidates, workspace.0, workspace.1, name)?;
                #[cfg(test)]
                {
                    rows += chunk.num_rows();
                }
                tokio::task::yield_now().await;
            }
        }
        Ok(rows)
    }

    async fn finish_candidates(
        &self,
        candidates: &mut InputCandidates,
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<()> {
        let mut native = candidates.native.take();
        if let Some(native) = native.as_mut() {
            self.native_candidates(native, candidates, context, name)
                .await?;
        }
        if let Some(mut partial) = candidates.partial.take() {
            partial.merge(&mut candidates.groups, context, name).await?;
        }
        drop(native);
        self.finalize_candidates(&mut candidates.groups, context, name)
            .await?;
        Ok(())
    }

    fn output_count(&self, candidates: &InputCandidates, name: &str) -> Result<usize> {
        let new_count = candidates
            .new_count
            .max(usize::from(self.keys.is_empty() && self.groups.is_empty()));
        let count = self
            .groups
            .len()
            .checked_add(new_count)
            .ok_or_else(|| df_error(name, "incremental group count overflowed"))?;
        Ok(count)
    }

    fn native_keys(&self, rows: usize, name: &str) -> Result<Option<NativeKeys>> {
        if !self.variable_extrema.is_empty() {
            return Ok(None);
        }
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
        self.reserve_chunk(chunk, reservation, workspace, name)?;
        let arguments = self.arguments(chunk, name)?;
        if self.keys.is_empty() {
            let candidate = candidates.groups.get_mut(&0).expect("global candidate");
            if let Some(variable) = candidate.variable.as_mut() {
                for row in 0..chunk.num_rows() {
                    variable.observe(&arguments, &[], row, &candidate.group.reservation, name)?;
                }
            }
            return update_global(&arguments, &mut candidates.groups, name);
        }
        let selection = self
            .predicate
            .as_ref()
            .map(|predicate| predicate.evaluate(chunk, name))
            .transpose()?;
        let filters = self.filters(chunk);
        let combined =
            predicate::combine(selection.as_ref(), &filters, self.aggregates.len(), name)?;
        let filters = if selection.is_some() {
            combined.iter().map(Option::as_ref).collect()
        } else {
            filters
        };
        if let Some(native) = candidates.native.as_mut() {
            native.intern_selected(chunk.column(self.keys[0]).clone(), selection.as_ref(), name)?;
            let count = native.groups.len();
            let partial = candidates
                .partial
                .as_mut()
                .expect("grouped partial accumulators");
            partial.reserve(count, name)?;
            partial.update(&arguments, &filters, &native.indices, count, name)?;
            #[cfg(test)]
            self.partial_groups
                .fetch_max(count, std::sync::atomic::Ordering::SeqCst);
            return Ok(());
        }
        self.update_grouped_chunk(
            chunk,
            &arguments,
            &filters,
            selection.as_ref(),
            candidates,
            name,
        )
    }

    fn filters<'a>(&self, chunk: &'a RecordBatch) -> Vec<Option<&'a BooleanArray>> {
        self.filter_columns
            .iter()
            .map(|column| {
                column.map(|column| {
                    chunk
                        .column(column)
                        .as_any()
                        .downcast_ref::<BooleanArray>()
                        .expect("validated Boolean aggregate filter")
                })
            })
            .collect()
    }

    fn reserve_chunk(
        &self,
        chunk: &RecordBatch,
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
        ensure_reservation(reservation, needed, name)?;
        Ok(())
    }

    fn update_grouped_chunk(
        &self,
        chunk: &RecordBatch,
        arguments: &[Vec<ArrayRef>],
        filters: &[Option<&BooleanArray>],
        selection: Option<&BooleanArray>,
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
            if selection.is_some_and(|selection| !predicate::selected(selection, row)) {
                indices.push(0);
                continue;
            }
            let encoded_row = encoded.row(row);
            let key = encoded_row.as_ref();
            if let Some(&rank) = candidates.touched.get(key) {
                indices.push(rank);
                continue;
            }
            #[cfg(test)]
            self.historical_key_lookups
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let (slot, previous, new_count) =
                self.candidate_slot(key, candidates.new_count, name)?;
            if let Some(proof) = &candidates.proof {
                let count = self
                    .groups
                    .len()
                    .checked_add(new_count)
                    .ok_or_else(|| df_error(name, "sequential group count overflowed"))?;
                let (_, width) = grouped_float::group_layout(&self.keys, &self.schema)
                    .expect("certified grouped key");
                proof.grow(count, width, &self.aggregates, name)?;
            }
            let candidate = self.candidate(previous, key, Some((&key_arrays, row)), name)?;
            let rank = partial.add_slot(slot, name)?;
            candidates.touched.insert(candidate.group.key.clone(), rank);
            candidates.groups.insert(slot, candidate);
            candidates.new_count = new_count;
            indices.push(rank);
        }
        if !self.variable_extrema.is_empty() {
            let growth = variable_extrema::prepare_grouped(
                arguments,
                filters,
                selection,
                &indices,
                &partial.slots,
                &mut candidates.groups,
                name,
            )?;
            partial.reserve_variable(growth, name)?;
        }
        partial.seed(&candidates.groups, &self.aggregates, name)?;
        partial.update(arguments, filters, &indices, partial.slots.len(), name)?;
        #[cfg(test)]
        self.partial_groups
            .fetch_max(partial.slots.len(), std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    fn candidate_slot(
        &self,
        key: &[u8],
        new_count: usize,
        name: &str,
    ) -> Result<(usize, Option<&Group>, usize)> {
        let (slot, previous, new_count) = if let Some(&slot) = self.index.get(key) {
            (slot, self.groups.get(slot), new_count)
        } else {
            let slot = self
                .groups
                .len()
                .checked_add(new_count)
                .ok_or_else(|| df_error(name, "incremental group count overflowed"))?;
            let count = new_count
                .checked_add(1)
                .ok_or_else(|| df_error(name, "new group count overflowed"))?;
            (slot, None, count)
        };
        Ok((slot, previous, new_count))
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
            self.native_candidate_rows(&arrays, &encoded, start, rows, candidates, name)?;
            tokio::task::yield_now().await;
        }
        Ok(())
    }

    fn native_candidate_rows(
        &self,
        arrays: &[ArrayRef],
        encoded: &datafusion::arrow::row::Rows,
        start: usize,
        rows: usize,
        candidates: &mut InputCandidates,
        name: &str,
    ) -> Result<()> {
        for row in 0..rows {
            let encoded_row = encoded.row(row);
            let key = encoded_row.as_ref();
            #[cfg(test)]
            self.historical_key_lookups
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let (slot, previous, new_count) =
                self.candidate_slot(key, candidates.new_count, name)?;
            let candidate = self.candidate(previous, key, Some((arrays, row)), name)?;
            let rank = candidates
                .partial
                .as_mut()
                .expect("grouped partial accumulators")
                .add_slot(slot, name)?;
            debug_assert_eq!(rank, start + row);
            candidates.groups.insert(slot, candidate);
            candidates.new_count = new_count;
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

    fn candidate_result(
        &self,
        expression: &AggregateFunctionExpr,
        state: &[ScalarValue],
        accumulator: &mut dyn Accumulator,
        name: &str,
    ) -> Result<ScalarValue> {
        if !self.keys.is_empty() && expression.fun().name() == "avg" {
            native_grouped_result(expression, state, name)
        } else {
            accumulator
                .evaluate()
                .map_err(|error| df_error(name, error))
        }
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
            for (index, accumulator) in candidate.accumulators.iter_mut().enumerate() {
                if let Some(accumulator) = accumulator {
                    let state = accumulator.state().map_err(|error| df_error(name, error))?;
                    let result = self.candidate_result(
                        &self.aggregates[index],
                        &state,
                        accumulator.as_mut(),
                        name,
                    )?;
                    candidate.group.states[index] = state;
                    candidate.group.results[index] = result;
                }
            }
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
        let output_charge = self.output_charge(count, candidates, context, name).await?;
        reservation
            .try_grow(output_charge)
            .map_err(|error| df_error(name, error))?;
        let mut records = Vec::new();
        for start in (0..count).step_by(CHUNK_ROWS) {
            context.check_cancelled()?;
            let end = count.min(start.saturating_add(CHUNK_ROWS));
            records.push(self.output_chunk(start, end, candidates, name)?);
            tokio::task::yield_now().await;
        }
        if records.is_empty() {
            records.push(RecordBatch::new_empty(self.output_schema.clone()));
        }
        Ok(records)
    }

    async fn output_charge(
        &self,
        count: usize,
        candidates: &CandidateMap,
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<usize> {
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
        let variable = !self.variable_extrema.is_empty();
        let output_charge =
            key_output_charge(output_charge, self.groups.iter(), variable, context, name).await?;
        key_output_charge(
            output_charge,
            candidates.values().map(|candidate| &candidate.group),
            variable,
            context,
            name,
        )
        .await
    }

    fn output_chunk(
        &self,
        start: usize,
        end: usize,
        candidates: &CandidateMap,
        name: &str,
    ) -> Result<RecordBatch> {
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
        RecordBatch::try_new(self.output_schema.clone(), output)
            .map_err(|error| df_error(name, error))
    }

    fn prepare_container(
        &self,
        candidates: &CandidateMap,
        count: usize,
        name: &str,
    ) -> Result<GroupContainer> {
        let capacity = count
            .max(4)
            .checked_next_power_of_two()
            .ok_or_else(|| df_error(name, "sequential container capacity overflowed"))?;
        let bytes = checked_bytes(
            4096,
            [
                (capacity, size_of::<Group>()),
                (
                    capacity,
                    4 * (size_of::<Arc<[u8]>>() + size_of::<usize>() + 1),
                ),
            ],
            name,
        )?;
        let reservation = self.reservation.new_empty();
        ensure_reservation(&reservation, bytes, name)?;
        let mut groups = Vec::new();
        groups
            .try_reserve_exact(capacity)
            .map_err(|error| df_error(name, error))?;
        let mut index = KeyIndex::with_hasher(RandomState::new());
        index
            .try_reserve(count)
            .map_err(|error| df_error(name, error))?;
        index.extend(self.index.iter().map(|(key, &slot)| (key.clone(), slot)));
        index.extend(
            candidates
                .iter()
                .map(|(&slot, candidate)| (candidate.group.key.clone(), slot)),
        );
        Ok(GroupContainer {
            groups,
            index,
            reservation,
        })
    }

    fn install_container(&mut self, mut container: GroupContainer) {
        container.groups.append(&mut self.groups);
        self.groups = container.groups;
        self.index = container.index;
        self.container_fee = Some(container.reservation);
    }

    fn reserve_groups(&mut self, new_count: usize, count: usize, name: &str) -> Result<()> {
        if self.sequential.is_some() {
            let container = self.prepare_container(
                &CandidateMap::with_hasher(RandomState::new()),
                count,
                name,
            )?;
            self.install_container(container);
            return Ok(());
        }
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
            ensure_reservation(&self.reservation, charge, name)?;
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
        if let Some(dirty) = transaction.dirty {
            self.dirty = dirty;
        }
        if let Some(container) = transaction.container {
            self.install_container(container);
        }
        if let Some(proof) = transaction.proof {
            self.sequential = Some(proof);
        }
        for (slot, group) in transaction.groups {
            if transaction.track_updates {
                self.dirty.mark(slot);
            }
            self.groups[slot] = group;
        }
        for group in transaction.new_groups.into_iter().flatten() {
            let slot = self.groups.len();
            self.dirty.mark(slot);
            self.index.insert(group.key.clone(), slot);
            self.groups.push(group);
        }
    }

    pub(super) fn clear_checkpoint_changes(&mut self) {
        self.dirty.clear();
    }
}

fn prepare_sync_plan(
    runtime: &DataFusionRuntime,
    query: &ValidatedQuery,
    alias: &str,
    schema: SchemaRef,
    name: &str,
) -> Result<crate::datafusion::compact::PaidSqlPlan> {
    let fields = super::ipc::schema_bytes(&schema).map_err(|error| df_error(name, error))?;
    let charge = checked_bytes(
        8192,
        [
            (query.text().len(), 16),
            (alias.len(), 16),
            (fields, 16),
            (schema.fields().len(), 512),
        ],
        name,
    )?;
    let reservation = runtime.incremental_reservation(name);
    ensure_reservation(&reservation, charge, name)?;
    runtime.incremental_sql_plan_sync(query, alias, schema, name, reservation)
}

fn grouped_layout(
    keys: &[usize],
    aggregates: &[Arc<AggregateFunctionExpr>],
    schema: &SchemaRef,
    name: &str,
) -> Result<(Option<RowConverter>, usize)> {
    Ok((
        row_converter(keys, schema, name)?,
        grouped_finalizer_charge(keys, aggregates, name)?,
    ))
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
    predicate: Option<&Expr>,
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
    for column in predicate.into_iter().flat_map(Expr::column_refs) {
        variable_columns.insert(
            schema
                .index_of(&column.name)
                .map_err(|error| df_error(name, error))?,
        );
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
    if !matches!(aggregate.input.as_ref(), LogicalPlan::TableScan(_))
        && !matches!(aggregate.input.as_ref(), LogicalPlan::Filter(filter)
            if !aggregate.group_expr.is_empty()
                && matches!(filter.input.as_ref(), LogicalPlan::TableScan(_)))
    {
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

fn eligible(expr: &Expr, schema: &SchemaRef, floating_extrema: bool, global: bool) -> bool {
    let Expr::AggregateFunction(function) = unalias(expr) else {
        return false;
    };
    let params = &function.params;
    if !aggregate_parameters_supported(params) {
        return false;
    }
    if !params.filter.as_deref().is_none_or(|filter| {
        let Expr::Column(column) = filter else {
            return false;
        };
        !global
            && schema
                .field_with_name(&column.name)
                .is_ok_and(|field| field.data_type() == &DataType::Boolean)
    }) {
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
        count_argument_supported(field.data_type())
    } else {
        aggregate_argument_supported(
            field.data_type(),
            function.func.name(),
            floating_extrema,
            global,
        )
    }
}

fn aggregate_argument_supported(
    data_type: &DataType,
    function: &str,
    floating_extrema: bool,
    global: bool,
) -> bool {
    match function {
        "sum" => {
            exact_numeric(data_type) || matches!(data_type, DataType::Float32 | DataType::Float64)
        }
        "min" | "max" => extrema_argument_supported(data_type, floating_extrema),
        "avg" => {
            (global && data_type.is_integer())
                || matches!(
                    data_type,
                    DataType::Int64
                        | DataType::UInt64
                        | DataType::Float32
                        | DataType::Float64
                        | DataType::Decimal32(_, 0..)
                        | DataType::Decimal64(_, 0..)
                        | DataType::Decimal128(_, 0..)
                        | DataType::Decimal256(_, 0..)
                )
        }
        _ => false,
    }
}

fn extrema_argument_supported(data_type: &DataType, global: bool) -> bool {
    exact_numeric(data_type)
        || matches!(data_type, DataType::Utf8 | DataType::LargeUtf8)
        || (global && matches!(data_type, DataType::Float32 | DataType::Float64))
}

fn count_argument_supported(data_type: &DataType) -> bool {
    key_type(data_type)
        || exact_numeric(data_type)
        || matches!(data_type, DataType::Float32 | DataType::Float64)
}

fn exact_numeric(data_type: &DataType) -> bool {
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
            | DataType::Decimal32(..)
            | DataType::Decimal64(..)
            | DataType::Decimal128(..)
            | DataType::Decimal256(..)
    )
}

fn grouped_aggregates_supported(keys: &[usize], aggregates: &[Arc<AggregateFunctionExpr>]) -> bool {
    keys.is_empty() || native_groups_supported(aggregates)
}

async fn key_output_charge<'a>(
    mut charge: usize,
    groups: impl Iterator<Item = &'a Group>,
    variable: bool,
    context: &StreamOperatorContext<'_>,
    name: &str,
) -> Result<usize> {
    for (index, group) in groups.enumerate() {
        if index % CHUNK_ROWS == 0 {
            context.check_cancelled()?;
            tokio::task::yield_now().await;
        }
        charge = checked_bytes(charge, [(group.key.len(), 4)], name)?;
        if variable {
            charge = checked_bytes(
                charge,
                [(variable_extrema::result_bytes(group, name)?, 4)],
                name,
            )?;
        }
    }
    Ok(charge)
}

fn aggregate_parameters_supported(
    params: &datafusion::logical_expr::expr::AggregateFunctionParams,
) -> bool {
    !params.distinct
        && params.order_by.is_empty()
        && params.null_treatment.is_none()
        && params.args.len() == 1
}

pub(super) fn ensure_reservation(
    reservation: &MemoryReservation,
    bytes: usize,
    name: &str,
) -> Result<()> {
    if bytes > reservation.size() {
        reservation
            .try_grow(bytes - reservation.size())
            .map_err(|error| df_error(name, error))?;
    }
    Ok(())
}

fn update_global(
    arguments: &[Vec<ArrayRef>],
    candidates: &mut CandidateMap,
    name: &str,
) -> Result<()> {
    let candidate = candidates.get_mut(&0).expect("global candidate");
    for (arguments, accumulator) in arguments.iter().zip(&mut candidate.accumulators) {
        accumulator
            .as_mut()
            .expect("global summary accumulator")
            .update_batch(arguments)
            .map_err(|error| df_error(name, error))?;
    }
    Ok(())
}

enum GroupStrategy {
    Unsupported,
    Exact,
    Sequential(grouped_float::Proof),
}

fn initial_grouped_proof(
    runtime: &DataFusionRuntime,
    reservation: &MemoryReservation,
    keys: &[usize],
    schema: &SchemaRef,
    aggregates: &[Arc<AggregateFunctionExpr>],
    name: &str,
) -> Result<GroupStrategy> {
    if aggregates
        .iter()
        .any(|expression| grouped_sum::selected(expression))
        && keys.is_empty()
    {
        return Ok(GroupStrategy::Unsupported);
    }
    if keys.is_empty()
        || !aggregates
            .iter()
            .any(|expression| grouped_float::selected(expression))
    {
        return Ok(GroupStrategy::Exact);
    }
    if !runtime.grouped_float_model_supported(name)? {
        return Ok(GroupStrategy::Unsupported);
    }
    let Some(layout) = grouped_float::group_layout(keys, schema) else {
        return Ok(GroupStrategy::Unsupported);
    };
    grouped_float::Proof::new(
        reservation.new_empty(),
        runtime.compact_runtime_config(),
        0,
        0,
        layout,
        aggregates,
        name,
    )
    .map(GroupStrategy::Sequential)
}

fn sequential_group_key(
    aggregate: &datafusion::logical_expr::Aggregate,
    schema: &SchemaRef,
) -> bool {
    !aggregate.group_expr.is_empty()
        && aggregate.group_expr.iter().all(|expression| {
            let Expr::Column(column) = expression else {
                return false;
            };
            schema
                .field_with_name(&column.name)
                .ok()
                .and_then(|field| grouped_float::key_layout(field.data_type()))
                .is_some()
        })
}

fn plan_inputs(
    raw: &LogicalPlan,
    schema: &SchemaRef,
    name: &str,
) -> Result<Option<(Vec<usize>, Vec<usize>)>> {
    let Some((_, raw_aggregate)) = shape(raw) else {
        return Ok(None);
    };
    if let LogicalPlan::Filter(filter) = raw_aggregate.input.as_ref() {
        if !predicate::InputPredicate::supported(&filter.predicate, schema) {
            return Ok(None);
        }
    }
    let global = raw_aggregate.group_expr.is_empty();
    let floating_extrema = global || sequential_group_key(raw_aggregate, schema);
    if raw_aggregate.aggr_expr.is_empty()
        || !raw_aggregate
            .aggr_expr
            .iter()
            .all(|expr| eligible(expr, schema, floating_extrema, global))
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
    let predicate = match raw_aggregate.input.as_ref() {
        LogicalPlan::Filter(filter) => Some(&filter.predicate),
        _ => None,
    };
    let variable_columns =
        variable_columns(&keys, &raw_aggregate.aggr_expr, predicate, schema, name)?;
    Ok(Some((keys, variable_columns)))
}

type PhysicalSqlPlan = (
    Vec<Arc<AggregateFunctionExpr>>,
    Vec<Arc<dyn PhysicalExpr>>,
    Vec<Option<usize>>,
    Option<predicate::InputPredicate>,
);

fn physical_plan(
    projection: &datafusion::logical_expr::Projection,
    aggregate: &datafusion::logical_expr::Aggregate,
    schema: &SchemaRef,
) -> Option<PhysicalSqlPlan> {
    let props = ExecutionProps::new();
    let logical = aggregate.input.schema();
    let rebound;
    let input = if logical.as_arrow() == schema.as_ref() {
        logical.as_ref()
    } else {
        let qualifiers = schema
            .fields()
            .iter()
            .map(|field| {
                let index = logical.as_arrow().index_of(field.name()).ok()?;
                let (qualifier, original) = logical.qualified_field(index);
                (original == field).then(|| qualifier.cloned())
            })
            .collect::<Option<Vec<_>>>()?;
        rebound = DFSchema::from_field_specific_qualified_schema(qualifiers, schema).ok()?;
        &rebound
    };
    let Ok(lowered) = aggregate
        .aggr_expr
        .iter()
        .map(|expr| LoweredAggregateBuilder::new(expr, input, schema, &props).build())
        .collect::<datafusion::error::Result<Vec<_>>>()
    else {
        return None;
    };
    let filter_columns = lowered
        .iter()
        .map(|lowered| match &lowered.filter {
            None => Some(None),
            Some(filter) => {
                let column = filter.downcast_ref::<Column>()?;
                (schema.field(column.index()).data_type() == &DataType::Boolean)
                    .then_some(Some(column.index()))
            }
        })
        .collect::<Option<Vec<_>>>()?;
    let filter_columns = if filter_columns.iter().all(Option::is_none) {
        Vec::new()
    } else {
        filter_columns
    };
    let aggregates = lowered
        .into_iter()
        .map(|lowered| lowered.aggregate)
        .collect();
    let Ok(projection) = projection
        .expr
        .iter()
        .map(|expr| create_physical_expr(expr, &aggregate.schema, &props))
        .collect::<datafusion::error::Result<Vec<_>>>()
    else {
        return None;
    };
    let predicate = match aggregate.input.as_ref() {
        LogicalPlan::Filter(filter) => Some(predicate::InputPredicate::new(
            &filter.predicate,
            input,
            schema,
            &props,
        )?),
        _ => None,
    };
    Some((aggregates, projection, filter_columns, predicate))
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

    async fn decimal_sum_expression() -> Arc<AggregateFunctionExpr> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, false),
            Field::new("value", DataType::Decimal256(60, 0), true),
        ]));
        let query = crate::expression::parse_select_query(
            "SELECT key, SUM(value) AS total FROM events GROUP BY key",
        )
        .unwrap();
        let runtime = DataFusionRuntime::new(crate::DataFusionConfig::default()).unwrap();
        let (_, analyzed) = runtime
            .incremental_sql_plan(&query, "events", schema.clone(), "totals")
            .await
            .unwrap();
        let (projection, aggregate) = shape(&analyzed).unwrap();
        physical_plan(projection, aggregate, &schema)
            .unwrap()
            .0
            .remove(0)
    }

    #[tokio::test]
    async fn test_sql_decimal_partial_growth_is_prepaid() {
        use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
        let expression = decimal_sum_expression().await;
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let reservation = MemoryConsumer::new("decimal-partial").register(&pool);
        let mut partial = PartialGroups::new(&[expression], reservation, "totals").unwrap();
        for slot in 0..1024 {
            partial.add_slot(slot, "totals").unwrap();
        }
        let values = ScalarValue::Decimal256(
            Some(datafusion::arrow::datatypes::i256::from_i128(1)),
            60,
            0,
        );
        partial
            .update(
                &[vec![values.to_array_of_size(1024).unwrap()]],
                &[],
                &(0..1024).collect::<Vec<_>>(),
                1024,
                "totals",
            )
            .unwrap();
        let previous = partial.accumulators[0].size();
        partial.add_slot(1024, "totals").unwrap();
        let prepaid = partial.reservation.size();
        partial
            .update(
                &[vec![values.to_array().unwrap()]],
                &[],
                &[1024],
                1025,
                "totals",
            )
            .unwrap();
        let peak = previous
            + partial.accumulators[0].size()
            + partial.slots.capacity() * size_of::<usize>();
        assert!(
            peak <= prepaid,
            "native capacity growth needs {peak} bytes; prepaid only {prepaid}"
        );
    }

    #[tokio::test]
    async fn test_sql_decimal_partial_pool_failure_precedes_capacity_growth() {
        use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
        let expression = decimal_sum_expression().await;
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let reservation = MemoryConsumer::new("decimal-partial").register(&pool);
        let mut partial = PartialGroups::new(&[expression], reservation, "totals").unwrap();
        for slot in 0..4 {
            partial.add_slot(slot, "totals").unwrap();
        }
        let before = (
            partial.slots.clone(),
            partial.slots.capacity(),
            partial.reservation.size(),
        );
        let pressure = MemoryConsumer::new("pressure").register(&pool);
        pressure.try_grow((1 << 20) - pool.reserved()).unwrap();
        assert!(partial.add_slot(4, "totals").is_err());
        assert_eq!(
            (
                partial.slots.clone(),
                partial.slots.capacity(),
                partial.reservation.size()
            ),
            before
        );
        drop(pressure);
        assert_eq!(partial.add_slot(4, "totals").unwrap(), 4);
        drop(partial);
        assert_eq!(pool.reserved(), 0);
    }

    async fn decimal_avg_expression() -> Arc<AggregateFunctionExpr> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, false),
            Field::new("value", DataType::Decimal256(60, 2), true),
        ]));
        let query = crate::expression::parse_select_query(
            "SELECT key, AVG(value) AS mean FROM events GROUP BY key",
        )
        .unwrap();
        let runtime = DataFusionRuntime::new(crate::DataFusionConfig::default()).unwrap();
        let (_, analyzed) = runtime
            .incremental_sql_plan(&query, "events", schema.clone(), "totals")
            .await
            .unwrap();
        let (projection, aggregate) = shape(&analyzed).unwrap();
        physical_plan(projection, aggregate, &schema)
            .unwrap()
            .0
            .remove(0)
    }

    fn native_avg_buffer_bytes(expression: &AggregateFunctionExpr, values: ArrayRef) -> usize {
        let count = values.len();
        let mut native = expression.create_groups_accumulator().unwrap();
        native
            .update_batch(&[values], &(0..count).collect::<Vec<_>>(), None, count)
            .unwrap();
        native
            .state(EmitTo::All)
            .unwrap()
            .iter()
            .map(|array| array.get_buffer_memory_size())
            .sum()
    }

    #[tokio::test]
    async fn test_sql_decimal_avg_partial_growth_and_extraction_are_prepaid() {
        use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
        let expression = decimal_avg_expression().await;
        let fields = expression.state_fields().unwrap();
        assert_eq!(fields[0].data_type(), &DataType::UInt64);
        assert_eq!(fields[1].data_type(), &DataType::Decimal256(60, 2));
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let reservation = MemoryConsumer::new("decimal-avg-partial").register(&pool);
        let mut partial = PartialGroups::new(&[expression.clone()], reservation, "totals").unwrap();
        for slot in 0..1024 {
            partial.add_slot(slot, "totals").unwrap();
        }
        let value = ScalarValue::Decimal256(
            Some(datafusion::arrow::datatypes::i256::from_i128(1)),
            60,
            2,
        );
        partial
            .update(
                &[vec![value.to_array_of_size(1024).unwrap()]],
                &[],
                &(0..1024).collect::<Vec<_>>(),
                1024,
                "totals",
            )
            .unwrap();
        let before = partial.accumulators[0].size();
        let previous_buffers =
            native_avg_buffer_bytes(&expression, value.to_array_of_size(1024).unwrap());
        let capacity = partial.slots.capacity();
        let prepaid = partial.reservation.size();
        let pressure = MemoryConsumer::new("pressure").register(&pool);
        pressure.try_grow((1 << 20) - pool.reserved()).unwrap();
        assert!(partial.add_slot(1024, "totals").is_err());
        assert_eq!(partial.slots.capacity(), capacity);
        assert_eq!(partial.accumulators[0].size(), before);
        assert_eq!(partial.reservation.size(), prepaid);
        drop(pressure);
        partial.add_slot(1024, "totals").unwrap();
        partial
            .update(
                &[vec![
                    ScalarValue::Decimal256(None, 60, 2).to_array().unwrap(),
                ]],
                &[],
                &[1024],
                1025,
                "totals",
            )
            .unwrap();
        let states = partial.accumulators[0].state(EmitTo::All).unwrap();
        assert_eq!(states[0].len(), 1025);
        assert_eq!(states[1].len(), 1025);
        assert!(states[0].is_null(1024));
        assert!(states[1].is_null(1024));
        assert!(!states[0].is_null(0));
        assert!(!states[1].is_null(0));
        let extracted = states
            .iter()
            .map(|array| array.get_buffer_memory_size())
            .sum::<usize>();
        let peak = previous_buffers + 2 * extracted + partial.slots.capacity() * size_of::<usize>();
        assert!(peak <= partial.reservation.size());
        assert_eq!(pool.reserved(), partial.reservation.size());
        drop(states);
        drop(partial);
        assert_eq!(pool.reserved(), 0);
    }

    #[derive(Debug, PartialEq)]
    enum BoundaryResult {
        Value(ScalarValue),
        Error,
        Panic,
    }

    fn boundary_outcome(result: std::thread::Result<Result<ScalarValue>>) -> BoundaryResult {
        match result {
            Ok(Ok(value)) => BoundaryResult::Value(value),
            Ok(Err(_)) => BoundaryResult::Error,
            Err(_) => BoundaryResult::Panic,
        }
    }

    async fn boundary_plan(value: &ScalarValue, grouped: bool) -> (IncrementalSql, RecordBatch) {
        let record = RecordBatch::try_from_iter_with_nullable([
            (
                "key",
                ScalarValue::Int64(Some(0)).to_array().unwrap(),
                false,
            ),
            ("value", value.to_array().unwrap(), true),
        ])
        .unwrap();
        let text = if grouped {
            "SELECT key, AVG(value) AS mean FROM events GROUP BY key"
        } else {
            "SELECT AVG(value) AS mean FROM events"
        };
        let query = crate::expression::parse_select_query(text).unwrap();
        let runtime = DataFusionRuntime::new(crate::DataFusionConfig::default()).unwrap();
        let plan = IncrementalSql::plan(&runtime, &query, "events", record.schema(), "totals")
            .await
            .unwrap()
            .unwrap();
        (plan, record)
    }

    fn reference_boundary(
        plan: &IncrementalSql,
        count: Option<u64>,
        sum: &ScalarValue,
    ) -> BoundaryResult {
        let expression = &plan.aggregates[0];
        let arrays = vec![
            ScalarValue::UInt64(count).to_array().unwrap(),
            sum.to_array().unwrap(),
        ];
        boundary_outcome(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
            || {
                if plan.keys.is_empty() {
                    let mut scalar = expression.create_accumulator().unwrap();
                    scalar.merge_batch(&arrays).unwrap();
                    scalar.evaluate().map_err(|error| df_error("totals", error))
                } else {
                    let mut native = expression.create_groups_accumulator().unwrap();
                    native.merge_batch(&arrays, &[0], None, 1).unwrap();
                    let result = native
                        .evaluate(EmitTo::All)
                        .map_err(|error| df_error("totals", error))?;
                    ScalarValue::try_from_array(&result, 0)
                        .map_err(|error| df_error("totals", error))
                }
            },
        )))
    }

    async fn candidate_boundary(
        plan: &IncrementalSql,
        record: &RecordBatch,
        count: Option<u64>,
        sum: &ScalarValue,
    ) -> BoundaryResult {
        use futures::FutureExt;
        let arrays = vec![record.column(0).clone()];
        let keys = if plan.keys.is_empty() {
            None
        } else {
            Some((arrays.as_slice(), 0))
        };
        let mut candidate = plan.candidate(None, &[0], keys, "totals").unwrap();
        candidate.accumulators[0]
            .as_mut()
            .unwrap()
            .merge_batch(&[
                ScalarValue::UInt64(Some(count.unwrap_or(0)))
                    .to_array()
                    .unwrap(),
                sum.to_array().unwrap(),
            ])
            .unwrap();
        let mut candidates = CandidateMap::with_hasher(RandomState::new());
        candidates.insert(0, candidate);
        let job = crate::StreamJobContext::new(
            1,
            "boundary",
            crate::JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "totals", None);
        let (_reservation, _) = plan.input_workspace(0, "totals").unwrap();
        let result = std::panic::AssertUnwindSafe(async {
            plan.finalize_candidates(&mut candidates, &context, "totals")
                .await?;
            Ok(candidates[&0].group.results[0].clone())
        })
        .catch_unwind()
        .await;
        boundary_outcome(result)
    }

    async fn assert_boundary_counts(value: ScalarValue, counts: &[u64], grouped: bool) {
        let (plan, record) = boundary_plan(&value, grouped).await;
        for count in counts {
            let reference = reference_boundary(&plan, Some(*count), &value);
            let candidate = candidate_boundary(&plan, &record, Some(*count), &value).await;
            assert_eq!(
                candidate, reference,
                "count={count}, grouped={grouped}, value={value:?}"
            );
        }
    }

    #[tokio::test]
    async fn test_sql_decimal_avg_grouped_count_conversion_boundary_matches_native() {
        assert_boundary_counts(
            ScalarValue::Decimal32(Some(8), 9, 0),
            &[
                1,
                i32::MAX as u64,
                (i32::MAX as u64) + 1,
                u64::from(u32::MAX),
                u64::from(u32::MAX) + 2,
            ],
            true,
        )
        .await;
        assert_boundary_counts(
            ScalarValue::Decimal64(Some(8), 18, 0),
            &[1, i64::MAX as u64, (i64::MAX as u64) + 1, u64::MAX],
            true,
        )
        .await;
    }

    #[tokio::test]
    async fn test_sql_decimal_avg_wrapped_zero_count_preserves_native_failure() {
        assert_boundary_counts(
            ScalarValue::Decimal32(Some(8), 9, 0),
            &[0, u64::from(u32::MAX) + 1],
            true,
        )
        .await;
        assert_boundary_counts(ScalarValue::Decimal64(Some(8), 18, 0), &[0], true).await;
    }

    #[tokio::test]
    async fn test_sql_decimal_avg_global_count_conversion_preserves_scalar() {
        assert_boundary_counts(
            ScalarValue::Decimal32(Some(8), 9, 0),
            &[0, 1, (i32::MAX as u64) + 1],
            false,
        )
        .await;
        assert_boundary_counts(
            ScalarValue::Decimal64(Some(8), 18, 0),
            &[0, 1, (i64::MAX as u64) + 1],
            false,
        )
        .await;
    }

    #[tokio::test]
    async fn test_sql_decimal_avg_unseen_seed_and_null_sum_match_native() {
        for value in [
            ScalarValue::Decimal32(None, 9, 0),
            ScalarValue::Decimal64(None, 18, 0),
            ScalarValue::Decimal128(None, 38, 0),
            ScalarValue::Decimal256(None, 76, 0),
        ] {
            let (plan, record) = boundary_plan(&value, true).await;
            for count in [None, Some(1), Some((i32::MAX as u64) + 1)] {
                assert_eq!(
                    candidate_boundary(&plan, &record, count, &value).await,
                    reference_boundary(&plan, count, &value)
                );
            }
        }
    }

    struct BoundaryReject;

    #[async_trait::async_trait]
    impl crate::StreamCollector for BoundaryReject {
        async fn emit(&mut self, _port: &str, _batch: Batch) -> Result<()> {
            Err(CalcFlowError::Operator {
                node_id: "boundary-reject".into(),
                message: "injected emit rejection".into(),
            })
        }
    }

    async fn boundary_rejected_update(value: ScalarValue, count: u64) {
        use crate::{OperatorMetadata, StreamOperator};
        use futures::FutureExt;
        let (plan, record) = boundary_plan(&value, true).await;
        let input = Batch::table(vec![record], crate::BatchMetadata::default()).unwrap();
        let mut operator = crate::SqlOperator::new(
            "totals",
            "SELECT key, AVG(value) AS mean FROM events GROUP BY key",
            vec!["events".into()],
            vec![],
        )
        .unwrap();
        let job = crate::StreamJobContext::new(
            1,
            "boundary",
            crate::JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "totals", None);
        let mut collector = crate::EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("events", input.clone(), &context, &mut collector)
            .await
            .unwrap();
        collector.drain("output");
        let incremental = operator.incremental.as_mut().unwrap();
        incremental.groups[0].states[0][0] = ScalarValue::UInt64(Some(count));
        let before = operator.checkpoint(crate::Epoch::INITIAL).unwrap();
        let failed = std::panic::AssertUnwindSafe(operator.process_data(
            "events",
            input.clone(),
            &context,
            &mut BoundaryReject,
        ))
        .catch_unwind()
        .await;
        assert!(
            matches!(failed, Ok(Err(CalcFlowError::Operator {node_id, ..})) if node_id == "boundary-reject")
        );
        let after = operator.checkpoint(crate::Epoch::INITIAL).unwrap();
        assert_eq!(before.inline_metadata, after.inline_metadata);
        assert!(Arc::ptr_eq(
            &before.segments["group-state"].bytes_arc(),
            &after.segments["group-state"].bytes_arc()
        ));
        assert_eq!(
            operator.incremental.as_ref().unwrap().groups[0].states[0][0],
            ScalarValue::UInt64(Some(count))
        );
        operator
            .process_data("events", input, &context, &mut collector)
            .await
            .unwrap();
        let output = collector.drain("output");
        assert_eq!(output.len(), 1);
        let record = &output[0]
            .as_data()
            .unwrap()
            .table_payload()
            .unwrap()
            .batches()[0];
        let result = ScalarValue::try_from_array(record.column(1), 0).unwrap();
        assert_eq!(
            BoundaryResult::Value(result),
            reference_boundary(&plan, Some(count + 1), &value)
        );
    }

    #[tokio::test]
    async fn test_sql_decimal_avg_grouped_boundary_rejected_output_rolls_back_and_retries() {
        boundary_rejected_update(ScalarValue::Decimal32(Some(0), 9, 0), (i32::MAX as u64) + 1)
            .await;
        boundary_rejected_update(
            ScalarValue::Decimal64(Some(0), 18, 0),
            (i64::MAX as u64) + 1,
        )
        .await;
    }

    fn native_finalizer_peak(expression: &AggregateFunctionExpr, sum: &ScalarValue) -> usize {
        let count = ScalarValue::UInt64(if sum.is_null() { None } else { Some(1) });
        let arrays = vec![count.to_array().unwrap(), sum.to_array().unwrap()];
        let seed_bytes = arrays
            .iter()
            .map(|array| array.get_array_memory_size())
            .sum::<usize>()
            + arrays.capacity() * size_of::<ArrayRef>();
        let mut native = expression.create_groups_accumulator().unwrap();
        let owner_bytes = size_of_val(native.as_ref());
        native.merge_batch(&arrays, &[0], None, 1).unwrap();
        let state = native.state(EmitTo::All).unwrap();
        let state_bytes = state
            .iter()
            .map(|array| array.get_array_memory_size())
            .sum::<usize>()
            + state.capacity() * size_of::<ArrayRef>();
        let mut result_owner = expression.create_groups_accumulator().unwrap();
        result_owner.merge_batch(&arrays, &[0], None, 1).unwrap();
        let result = result_owner.evaluate(EmitTo::All).unwrap();
        seed_bytes + state_bytes + 2 * owner_bytes + result.get_array_memory_size()
    }

    async fn assert_native_finalizer_lease(value: ScalarValue) {
        use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
        let (mut plan, record) = boundary_plan(&value, true).await;
        let peak = native_finalizer_peak(&plan.aggregates[0], &value);
        assert!(peak <= plan.finalizer_bytes);
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        plan.reservation = MemoryConsumer::new("finalizer-test").register(&pool);
        plan.reservation.try_grow(plan.plan_bytes).unwrap();
        let job = crate::StreamJobContext::new(
            1,
            "finalizer-test",
            crate::JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "totals", None);
        let input = Batch::table(vec![record], crate::BatchMetadata::default()).unwrap();
        let transaction = plan.update(&input, &context, "totals").await.unwrap();
        let Transaction {
            _reservation: lease,
            ..
        } = &transaction;
        assert!(lease.size() >= 4096 + plan.finalizer_bytes);
        let pressure = MemoryConsumer::new("pressure").register(&pool);
        pressure.try_grow((1 << 20) - pool.reserved()).unwrap();
        assert_eq!(pool.reserved(), 1 << 20);
        drop(transaction);
        assert!(pool.reserved() < 1 << 20);
        assert!(plan.groups.is_empty());
        drop(pressure);
        drop(plan);
        assert_eq!(pool.reserved(), 0);
    }

    #[tokio::test]
    async fn test_sql_decimal_avg_native_finalizer_peak_is_prepaid_until_transaction_drops() {
        for value in [
            ScalarValue::Decimal32(Some(8), 9, 0),
            ScalarValue::Decimal64(Some(8), 18, 0),
            ScalarValue::Decimal128(Some(8), 38, 0),
            ScalarValue::Decimal256(
                Some(datafusion::arrow::datatypes::i256::from_i128(8)),
                76,
                0,
            ),
        ] {
            let null = ScalarValue::try_from(&value.data_type()).unwrap();
            assert_native_finalizer_lease(value).await;
            assert_native_finalizer_lease(null).await;
        }
    }

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

#[cfg(test)]
#[path = "compact_checkpoint_tests.rs"]
mod compact_checkpoint_tests;

#[cfg(test)]
#[path = "compact_plan_tests.rs"]
mod compact_plan_tests;
