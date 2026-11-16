use super::{
    AggregateFunctionExpr, ArrayRef, BooleanArray, CHUNK_ROWS, GroupKey, PartialGroups, RecordBatch,
    Result, ScalarValue, SchemaRef, df_error, global_record, grouped_float, grouped_sum, native_grouped_result, ordered_last, predicate, variable_extrema,
};
use crate::StateBudget;
use crate::operator::keyed::{GroupId, KeyCut, KeySchema, KeyStore, MORSEL_ROWS};
use datafusion::arrow::{
    array::Array,
    datatypes::FieldRef,
    execution::memory_pool::MemoryReservation,
};
use sha2::{Digest, Sha256};
use std::{
    mem::size_of,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

#[derive(Clone)]
pub(super) struct KeyedState {
    state: Arc<Mutex<State>>,
    reservation: MemoryReservation,
}

struct State {
    keys: Option<KeyStore>,
    groups: usize,
    aggregates: Vec<AggregateState>,
    marks: Vec<u64>,
    generation: u64,
    active: bool,
    fingerprint: [u8; 32],
    global_key: Arc<[u8]>,
    reservation: MemoryReservation,
}

struct AggregateState {
    states: Vec<StateColumn>,
    result: StateColumn,
    initialized: Vec<bool>,
}

struct StateColumn {
    null: ScalarValue,
    values: Vec<ScalarValue>,
}

pub(super) struct Transaction {
    state: Arc<Mutex<State>>,
    cut: Option<KeyCut>,
    charge_cut: usize,
    old_groups: usize,
    generation: u64,
    limit: usize,
    affected: Vec<GroupId>,
    undo: Vec<GroupUndo>,
    leased: Vec<GroupId>,
    ranks: Vec<Option<usize>>,
    partial: Option<PartialGroups>,
    undo_reservation: MemoryReservation,
    active: bool,
}

struct GroupUndo {
    group: GroupId,
    states: Vec<Vec<ScalarValue>>,
    results: Vec<ScalarValue>,
    initialized: Vec<bool>,
}

pub(super) struct Committed {
    pub affected: Vec<GroupId>,
}

impl KeyedState {
    pub(super) fn new(
        keys: &[GroupKey],
        aggregates: &[Arc<AggregateFunctionExpr>],
        reservation: MemoryReservation,
    ) -> Result<Self> {
        let grouped = !keys.is_empty();
        let transaction_reservation = reservation.new_empty();
        let state_reservation = reservation.new_empty();
        let key_store = grouped
            .then(|| {
                let fields = keys.iter().map(|key| key.field.clone()).collect();
                KeySchema::groupable(fields).map(|schema| KeyStore::new(schema, reservation))
            })
            .transpose()?;
        let aggregates = aggregates
            .iter()
            .map(|expression| AggregateState::new(expression.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        let fingerprint = key_store.as_ref().map_or_else(global_fingerprint, KeyStore::layout_fingerprint);
        Ok(Self {
            state: Arc::new(Mutex::new(State {
                keys: key_store,
                groups: 0,
                aggregates,
                marks: Vec::new(),
                generation: 0,
                active: false,
                fingerprint,
                global_key: Arc::from([]),
                reservation: state_reservation,
            })),
            reservation: transaction_reservation,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.lock().groups
    }

    pub(super) fn is_empty(&self) -> bool {
        self.lock().groups == 0
    }

    pub(super) fn layout_fingerprint(&self) -> [u8; 32] {
        self.lock().fingerprint
    }

    pub(super) fn canonical_slot(&self, slot: usize) -> Result<Arc<[u8]>> {
        let state = self.lock();
        match &state.keys {
            Some(keys) => Ok(keys.canonical(GroupId::new(slot)?)),
            None if slot == 0 && state.groups == 1 => Ok(state.global_key.clone()),
            None => Err(df_error("keyed aggregate", "global group slot is invalid")),
        }
    }

    pub(super) fn begin(&self, budget: Option<StateBudget>) -> Result<Transaction> {
        let mut state = self.lock();
        if state.active {
            return Err(df_error(
                "keyed aggregate",
                "a keyed aggregate transaction is already active",
            ));
        }
        let cut = state.keys.as_ref().map(KeyStore::cut);
        let charge_cut = state.reservation.size();
        let old_groups = state.groups;
        let generation = state
            .generation
            .checked_add(1)
            .ok_or_else(|| df_error("keyed aggregate", "transaction generation overflowed"))?;
        state.generation = generation;
        state.active = true;
        let limit = budget.map_or(usize::MAX, |budget| {
            usize::try_from(budget.max_rows()).unwrap_or(usize::MAX)
        });
        let undo_reservation = self.reservation.new_empty();
        let mut transaction = Transaction {
            state: self.state.clone(),
            cut,
            charge_cut,
            old_groups,
            generation,
            limit,
            affected: Vec::new(),
            undo: Vec::new(),
            leased: Vec::new(),
            ranks: Vec::new(),
            partial: None,
            undo_reservation,
            active: true,
        };
        let initialized = if state.keys.is_none() {
            transaction.ensure_global(&mut state)
        } else {
            Ok(())
        };
        drop(state);
        initialized?;
        Ok(transaction)
    }

    pub(super) fn export(
        &self,
        slots: Option<&[usize]>,
        schema: SchemaRef,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<RecordBatch>> {
        self.lock().records(slots, schema, true, check)
    }

    pub(super) fn output_charge(&self, slots: Option<&[usize]>) -> Result<usize> {
        self.lock().output_charge(slots)
    }

    pub(super) fn history_counts(
        &self,
        functions: &[String],
        historical_rows: u64,
        check: &dyn Fn() -> Result<()>,
        name: &str,
    ) -> Result<Vec<u64>> {
        let state = self.lock();
        let mut counts = vec![0u64; functions.len()];
        for row in 0..state.groups {
            if row % 128 == 0 {
                check()?;
            }
            for (index, function) in functions.iter().enumerate() {
                let Some(column) = state.aggregates[index].states.first() else {
                    continue;
                };
                let count = match (function.as_str(), &column.values[row]) {
                    ("count", ScalarValue::Int64(value)) => u64::try_from(
                        value.ok_or_else(|| df_error(name, "native COUNT state is null"))?,
                    )
                    .map_err(|_| df_error(name, "native COUNT state is negative"))?,
                    ("avg", ScalarValue::UInt64(value)) => value.unwrap_or(0),
                    _ => continue,
                };
                counts[index] = counts[index]
                    .checked_add(count)
                    .filter(|total| *total <= historical_rows)
                    .ok_or_else(|| df_error(name, "native aggregate count exceeds historical rows"))?;
            }
        }
        Ok(counts)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn restore_records(
        &self,
        records: &[RecordBatch],
        key_count: usize,
        replace: bool,
        budget: Option<StateBudget>,
        aggregates: &[Arc<AggregateFunctionExpr>],
        global: Option<&global_record::Proof>,
        sequential: bool,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        if records.iter().all(|record| record.num_rows() == 0) {
            check()?;
            return Ok(());
        }
        let mut transaction = self.begin(budget)?;
        let mut seen = std::collections::BTreeSet::new();
        for record in records {
            for start in (0..record.num_rows()).step_by(MORSEL_ROWS) {
                check()?;
                let rows = MORSEL_ROWS.min(record.num_rows() - start);
                let chunk = record.slice(start, rows);
                transaction.restore_chunk(
                    &chunk, key_count, replace, aggregates, global, sequential, &mut seen, check,
                )?;
            }
            if record.num_rows() == 0 {
                check()?;
            }
        }
        transaction.commit();
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn snapshot(
        &self,
    ) -> Vec<(Vec<ScalarValue>, Vec<Vec<ScalarValue>>, Vec<ScalarValue>)> {
        let state = self.lock();
        (0..state.groups)
            .map(|slot| {
                let group = GroupId::new(slot).expect("dense test group");
                let keys = match &state.keys {
                    Some(keys) => keys
                        .columns(std::slice::from_ref(&group))
                        .unwrap()
                        .iter()
                        .map(|array| ScalarValue::try_from_array(array, 0).unwrap())
                        .collect(),
                    None => Vec::new(),
                };
                let states = state
                    .aggregates
                    .iter()
                    .map(|aggregate| {
                        aggregate
                            .states
                            .iter()
                            .map(|column| column.values[slot].clone())
                            .collect()
                    })
                    .collect();
                let results = state
                    .aggregates
                    .iter()
                    .map(|aggregate| aggregate.result.values[slot].clone())
                    .collect();
                (keys, states, results)
            })
            .collect()
    }

    #[cfg(test)]
    pub(super) fn record_for_test(&self, schema: SchemaRef) -> RecordBatch {
        let state = self.lock();
        state
            .record(0, state.groups, None, schema, false)
            .unwrap()
    }

    #[cfg(test)]
    pub(super) fn set_state_field_for_test(
        &self,
        aggregate: usize,
        field: usize,
        group: GroupId,
        value: ScalarValue,
    ) {
        let mut state = self.lock();
        state.aggregates[aggregate].states[field].values[group.index()] = value;
    }

    #[cfg(test)]
    pub(super) fn state_field_for_test(
        &self,
        aggregate: usize,
        field: usize,
        group: GroupId,
    ) -> ScalarValue {
        self.lock().aggregates[aggregate].states[field].values[group.index()].clone()
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Transaction {
    pub(super) fn init_partial(
        &mut self,
        aggregates: &[Arc<AggregateFunctionExpr>],
        ordered_last: bool,
        sequential: bool,
        reservation: MemoryReservation,
        name: &str,
    ) -> Result<()> {
        let mut partial = PartialGroups::new(aggregates, reservation, name)?;
        partial.sequential = if ordered_last {
            vec![true; aggregates.len()]
        } else if sequential {
            aggregates
                .iter()
                .map(|expression| grouped_float::selected(expression))
                .collect()
        } else {
            vec![false; aggregates.len()]
        };
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.keys.is_none() {
            self.rank(GroupId::new(0)?, &mut state, &mut partial, name)?;
        }
        self.partial = Some(partial);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_chunk(
        &mut self,
        keys: &[ArrayRef],
        rows: usize,
        arguments: &[Vec<ArrayRef>],
        filters: &[Option<&BooleanArray>],
        selection: Option<&BooleanArray>,
        variable: &[usize],
        check: &mut dyn FnMut() -> Result<()>,
        name: &str,
    ) -> Result<()> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let groups = match &mut state.keys {
            Some(store) => store.route_selected_with_limit(keys, selection, self.limit, check)?,
            None => vec![GroupId::new(0)?; rows],
        };
        self.grow(&mut state)?;
        let mut partial = self.partial.take().expect("partial aggregate state");
        let mut indices = Vec::with_capacity(rows);
        for (row, group) in groups.into_iter().enumerate() {
            if selection.is_some_and(|selection| !predicate::selected(selection, row)) {
                indices.push(0);
                continue;
            }
            indices.push(self.rank(group, &mut state, &mut partial, name)?);
        }
        self.observe_variable(&mut partial, arguments, filters, selection, variable, name)?;
        self.seed(&state, &mut partial, name)?;
        partial.update(arguments, filters, &indices, partial.slots.len(), name)?;
        self.partial = Some(partial);
        Ok(())
    }

    pub(super) fn global_states(&self) -> Vec<Vec<ScalarValue>> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        (0..state.aggregates.len())
            .map(|aggregate| state.state(aggregate, GroupId::new(0).expect("global slot"))
                .unwrap_or_default())
            .collect()
    }

    pub(super) fn touch_global(&mut self) -> Result<()> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.touch(&mut state, GroupId::new(0)?)
    }

    pub(super) fn set_global(
        &mut self,
        aggregate: usize,
        values: Vec<ScalarValue>,
        result: ScalarValue,
    ) -> Result<()> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.set_state(aggregate, GroupId::new(0)?, values, result)
    }

    pub(super) fn finish(
        &mut self,
        aggregates: &[Arc<AggregateFunctionExpr>],
        check: &mut dyn FnMut() -> Result<()>,
        name: &str,
    ) -> Result<()> {
        let Some(mut partial) = self.partial.take() else {
            return Ok(());
        };
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        self.seed_integer_history(&state, &mut partial, name)?;
        let states = partial.emit_states(name)?;
        for (rank, slot) in partial.slots.iter().enumerate() {
            if rank % 128 == 0 {
                check()?;
            }
            let group = GroupId::new(*slot)?;
            for (index, arrays) in states.iter().enumerate() {
                self.merge_rank(
                    &mut state,
                    &partial,
                    aggregates[index].as_ref(),
                    group,
                    index,
                    arrays,
                    rank,
                    name,
                )?;
            }
        }
        Ok(())
    }

    pub(super) fn group_count(&self) -> usize {
        self.state.lock().unwrap_or_else(PoisonError::into_inner).groups
    }

    pub(super) fn first_key(&self, slot: usize) -> Result<Arc<[u8]>> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        match &state.keys {
            Some(keys) => Ok(keys.canonical(GroupId::new(slot)?)),
            None if slot == 0 && state.groups == 1 => Ok(state.global_key.clone()),
            None => Err(df_error("keyed aggregate", "global group slot is invalid")),
        }
    }

    pub(super) fn record(
        &self,
        start: usize,
        end: usize,
        schema: SchemaRef,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<RecordBatch> {
        check()?;
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .record(start, end, None, schema, false)
    }

    pub(super) fn dirty_output_chunks(&self) -> std::collections::BTreeSet<usize> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let mut chunks = std::collections::BTreeSet::new();
        for previous in &self.undo {
            if previous.changed(&state) {
                chunks.insert(previous.group.index() / CHUNK_ROWS);
            }
        }
        for group in self
            .affected
            .iter()
            .filter(|group| group.index() >= self.old_groups)
        {
            chunks.insert(group.index() / CHUNK_ROWS);
        }
        chunks
    }

    pub(super) fn output_charge(&self) -> Result<usize> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .output_charge(None)
    }

    pub(super) fn changes_state(&self) -> bool {
        !self.affected.is_empty()
    }

    pub(super) fn commit(mut self) -> Committed {
        self.settle(true);
        Committed {
            affected: std::mem::take(&mut self.affected),
        }
    }

    fn ensure_global(&mut self, state: &mut State) -> Result<()> {
        if state.groups == 0 {
            state.grow(1)?;
            state.groups = 1;
            self.ranks.resize(1, None);
            self.touch(state, GroupId::new(0)?)?;
        }
        Ok(())
    }

    fn grow(&mut self, state: &mut State) -> Result<()> {
        let groups = state.keys.as_ref().map_or(state.groups, KeyStore::len);
        if groups != state.groups {
            state.grow(groups)?;
            state.groups = groups;
        }
        self.ranks.resize(groups, None);
        Ok(())
    }

    fn rank(
        &mut self,
        group: GroupId,
        state: &mut State,
        partial: &mut PartialGroups,
        name: &str,
    ) -> Result<usize> {
        if let Some(rank) = self.ranks[group.index()] {
            return Ok(rank);
        }
        let rank = partial.add_slot(group.index(), name)?;
        self.ranks[group.index()] = Some(rank);
        self.touch(state, group)?;
        Ok(rank)
    }

    fn touch(&mut self, state: &mut State, group: GroupId) -> Result<()> {
        if state.marks[group.index()] == self.generation {
            return Ok(());
        }
        let existing = group.index() < self.old_groups;
        let charge = if existing {
            GroupUndo::charge(state, group)
        } else {
            size_of::<GroupId>()
        };
        self.undo_reservation
            .try_grow(charge)
            .map_err(|error| df_error("keyed aggregate", error))?;
        state.marks[group.index()] = self.generation;
        self.affected.push(group);
        if existing {
            if let Some(keys) = &mut state.keys {
                keys.acquire(group)?;
                self.leased.push(group);
            }
            self.undo.push(GroupUndo::new(state, group));
        }
        Ok(())
    }

    fn observe_variable(
        &self,
        partial: &mut PartialGroups,
        arguments: &[Vec<ArrayRef>],
        filters: &[Option<&BooleanArray>],
        selection: Option<&BooleanArray>,
        variable: &[usize],
        name: &str,
    ) -> Result<()> {
        let mut growth = 0;
        for &aggregate in variable {
            for row in 0..arguments[aggregate][0].len() {
                let selected = selection.is_none_or(|selection| predicate::selected(selection, row));
                let filtered = filters
                    .get(aggregate)
                    .copied()
                    .flatten()
                    .is_none_or(|filter| predicate::selected(filter, row));
                if selected && filtered {
                    let width = variable_extrema::array_bytes(&arguments[aggregate][0], row);
                    growth = growth.checked_add(width).ok_or_else(|| {
                        df_error(name, "variable aggregate charge overflowed")
                    })?;
                }
            }
        }
        partial.reserve_variable(growth, name)
    }

    fn seed(&self, state: &State, partial: &mut PartialGroups, name: &str) -> Result<()> {
        for index in 0..partial.accumulators.len() {
            if partial.sequential[index] {
                let saved = (partial.seeded..partial.slots.len()).map(|rank| {
                    let group = GroupId::new(partial.slots[rank]).expect("dense group");
                    (rank, state.seed_state(index, group))
                });
                partial.accumulators[index].seed_batch(saved, partial.slots.len(), name)?;
            }
        }
        partial.seeded = partial.slots.len();
        Ok(())
    }

    fn seed_integer_history(&self, state: &State, partial: &mut PartialGroups, name: &str) -> Result<()> {
        for index in 0..partial.accumulators.len() {
            if partial.accumulators[index].integer_sum() {
                let saved = partial.slots.iter().enumerate().map(|(rank, slot)| {
                    let group = GroupId::new(*slot).expect("dense group");
                    (rank, state.seed_state(index, group))
                });
                partial.accumulators[index].seed_batch(saved, partial.slots.len(), name)?;
            }
        }
        partial.check_size(name)
    }

    #[allow(clippy::too_many_arguments)]
    fn merge_rank(
        &self,
        state: &mut State,
        partial: &PartialGroups,
        expression: &AggregateFunctionExpr,
        group: GroupId,
        index: usize,
        arrays: &[ArrayRef],
        rank: usize,
        name: &str,
    ) -> Result<()> {
        if partial.sequential[index] || partial.accumulators[index].integer_sum() {
            let values = scalars(arrays, rank, name)?;
            let result = partial.rank_result(index, &values, name)?;
            return state.set_state(index, group, values, result);
        }
        let mut accumulator = expression
            .create_accumulator()
            .map_err(|error| df_error(name, error))?;
        if state.initialized(index, group) {
            let previous = state.state(index, group)?;
            merge(&mut *accumulator, &previous, name)?;
        }
        let arrays = arrays
            .iter()
            .map(|array| array.slice(rank, 1))
            .collect::<Vec<_>>();
        accumulator
            .merge_batch(&arrays)
            .map_err(|error| df_error(name, error))?;
        let values = accumulator.state().map_err(|error| df_error(name, error))?;
        let result = if state.keys.is_some() && expression.fun().name() == "avg" {
            native_grouped_result(expression, &values, name)?
        } else {
            accumulator.evaluate().map_err(|error| df_error(name, error))?
        };
        state.set_state(index, group, values, result)
    }

    #[allow(clippy::too_many_arguments)]
    fn restore_chunk(
        &mut self,
        chunk: &RecordBatch,
        key_count: usize,
        replace: bool,
        aggregates: &[Arc<AggregateFunctionExpr>],
        global: Option<&global_record::Proof>,
        sequential: bool,
        seen: &mut std::collections::BTreeSet<GroupId>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        let keys = chunk.columns()[..key_count].to_vec();
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let groups = match &mut state.keys {
            Some(store) => store.route_with_limit(&keys, self.limit, check)?,
            None => vec![GroupId::new(0)?; chunk.num_rows()],
        };
        self.grow(&mut state)?;
        for (row, group) in groups.into_iter().enumerate() {
            if !seen.insert(group) || (group.index() < self.old_groups && !replace) {
                return Err(df_error(
                    "keyed aggregate",
                    "native state contains duplicate keys",
                ));
            }
            self.touch(&mut state, group)?;
            let mut column = key_count;
            for (index, expression) in aggregates.iter().enumerate() {
                let width = expression
                    .state_fields()
                    .map_err(|error| df_error("keyed aggregate", error))?
                    .len();
                let values = scalars(&chunk.columns()[column..column + width], row, "keyed aggregate")?;
                let result = restored_result(
                    index,
                    expression.as_ref(),
                    &values,
                    state.keys.is_some(),
                    global,
                    sequential,
                    "keyed aggregate",
                )?;
                state.set_state(index, group, values, result)?;
                column += width;
            }
        }
        Ok(())
    }

    fn settle(&mut self, commit: bool) {
        if !self.active {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if !commit {
            for previous in self.undo.drain(..).rev() {
                previous.restore(&mut state);
            }
            state.truncate(self.old_groups);
            state.reservation.resize(self.charge_cut);
            if let (Some(keys), Some(cut)) = (&mut state.keys, self.cut) {
                keys.rollback(cut);
            }
        }
        for group in self.leased.drain(..) {
            if let Some(keys) = &mut state.keys {
                let _ = keys.release(group);
            }
        }
        state.active = false;
        self.active = false;
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        self.settle(false);
    }
}

impl AggregateState {
    fn new(expression: &AggregateFunctionExpr) -> Result<Self> {
        let fields = expression
            .state_fields()
            .map_err(|error| df_error("keyed aggregate", error))?;
        let states = fields
            .iter()
            .map(StateColumn::new)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            states,
            result: StateColumn::new(&expression.field())?,
            initialized: Vec::new(),
        })
    }

    fn grow(&mut self, groups: usize) {
        for column in &mut self.states {
            column.grow(groups);
        }
        self.result.grow(groups);
        self.initialized.resize(groups, false);
    }

    fn truncate(&mut self, groups: usize) {
        for column in &mut self.states {
            column.values.truncate(groups);
        }
        self.result.values.truncate(groups);
        self.initialized.truncate(groups);
    }
}

impl StateColumn {
    fn new(field: &FieldRef) -> Result<Self> {
        Ok(Self {
            null: ScalarValue::try_from(field.data_type())
                .map_err(|error| df_error("keyed aggregate", error))?,
            values: Vec::new(),
        })
    }

    fn grow(&mut self, groups: usize) {
        self.values.resize(groups, self.null.clone());
    }

    fn array(&self, groups: &[GroupId]) -> Result<ArrayRef> {
        ScalarValue::iter_to_array(groups.iter().map(|group| self.values[group.index()].clone()))
            .map_err(|error| df_error("keyed aggregate", error))
    }
}

impl State {
    fn grow(&mut self, groups: usize) -> Result<()> {
        let new = groups.saturating_sub(self.groups);
        let width = self.aggregates.iter().try_fold(size_of::<u64>(), |width, aggregate| {
            let fields = aggregate.states.len() + 1;
            width
                .checked_add(fields * size_of::<ScalarValue>() + size_of::<bool>())
                .ok_or_else(|| df_error("keyed aggregate", "typed state width overflowed"))
        })?;
        let charge = new
            .checked_mul(width)
            .ok_or_else(|| df_error("keyed aggregate", "typed state charge overflowed"))?;
        self.reservation
            .try_grow(charge)
            .map_err(|error| df_error("keyed aggregate", error))?;
        for aggregate in &mut self.aggregates {
            aggregate.grow(groups);
        }
        self.marks.resize(groups, 0);
        Ok(())
    }

    fn truncate(&mut self, groups: usize) {
        self.groups = groups;
        for aggregate in &mut self.aggregates {
            aggregate.truncate(groups);
        }
        self.marks.truncate(groups);
    }

    fn initialized(&self, aggregate: usize, group: GroupId) -> bool {
        self.aggregates[aggregate].initialized[group.index()]
    }

    fn state(&self, aggregate: usize, group: GroupId) -> Result<Vec<ScalarValue>> {
        if !self.initialized(aggregate, group) {
            return Ok(Vec::new());
        }
        Ok(self.aggregates[aggregate]
            .states
            .iter()
            .map(|column| column.values[group.index()].clone())
            .collect())
    }

    fn seed_state(&self, aggregate: usize, group: GroupId) -> Vec<ScalarValue> {
        self.state(aggregate, group).unwrap_or_default()
    }

    fn result(&self, aggregate: usize, group: GroupId) -> ScalarValue {
        self.aggregates[aggregate].result.values[group.index()].clone()
    }

    fn set_state(
        &mut self,
        aggregate: usize,
        group: GroupId,
        values: Vec<ScalarValue>,
        result: ScalarValue,
    ) -> Result<()> {
        let target = &mut self.aggregates[aggregate];
        if target.states.len() != values.len() {
            return Err(df_error("keyed aggregate", "aggregate state width differs"));
        }
        for (column, value) in target.states.iter_mut().zip(values) {
            if value.data_type() != column.null.data_type() {
                return Err(df_error("keyed aggregate", "aggregate state type differs"));
            }
            column.values[group.index()] = value;
        }
        if result.data_type() != target.result.null.data_type() {
            return Err(df_error("keyed aggregate", "aggregate result type differs"));
        }
        target.result.values[group.index()] = result;
        target.initialized[group.index()] = true;
        Ok(())
    }

    fn record(
        &self,
        start: usize,
        end: usize,
        slots: Option<&[usize]>,
        schema: SchemaRef,
        state: bool,
    ) -> Result<RecordBatch> {
        if start == end {
            return Ok(RecordBatch::new_empty(schema));
        }
        let groups = (start..end)
            .map(|row| GroupId::new(slots.map_or(row, |slots| slots[row])))
            .collect::<Result<Vec<_>>>()?;
        let mut columns = match &self.keys {
            Some(keys) => keys.columns(&groups)?,
            None => Vec::new(),
        };
        if state {
            for aggregate in &self.aggregates {
                for column in &aggregate.states {
                    columns.push(column.array(&groups)?);
                }
            }
        } else {
            for (index, _) in self.aggregates.iter().enumerate() {
                columns.push(
                    ScalarValue::iter_to_array(
                        groups.iter().map(|group| self.result(index, *group)),
                    )
                    .map_err(|error| df_error("keyed aggregate", error))?,
                );
            }
        }
        RecordBatch::try_new(schema, columns).map_err(|error| df_error("keyed aggregate", error))
    }

    fn records(
        &self,
        slots: Option<&[usize]>,
        schema: SchemaRef,
        state: bool,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<RecordBatch>> {
        let count = slots.map_or(self.groups, <[usize]>::len);
        if count == 0 {
            check()?;
            return Ok(vec![RecordBatch::new_empty(schema)]);
        }
        let mut records = Vec::with_capacity(count.div_ceil(CHUNK_ROWS));
        for start in (0..count).step_by(CHUNK_ROWS) {
            check()?;
            let end = count.min(start + CHUNK_ROWS);
            records.push(self.record(start, end, slots, schema.clone(), state)?);
        }
        Ok(records)
    }

    fn output_charge(&self, slots: Option<&[usize]>) -> Result<usize> {
        let count = slots.map_or(self.groups, <[usize]>::len);
        let groups = (0..count)
            .map(|row| GroupId::new(slots.map_or(row, |slots| slots[row])))
            .collect::<Result<Vec<_>>>()?;
        let keys = match &self.keys {
            Some(keys) => keys.output_charge(&groups)?,
            None => 0,
        };
        groups.iter().try_fold(keys, |total, group| {
            let values = self.aggregates.iter().flat_map(|aggregate| {
                aggregate
                    .states
                    .iter()
                    .map(|column| &column.values[group.index()])
                    .chain(std::iter::once(&aggregate.result.values[group.index()]))
            });
            values.try_fold(total, |total, value| {
                total
                    .checked_add(value.size())
                    .and_then(|total| total.checked_add(64))
                    .ok_or_else(|| df_error("keyed aggregate", "output charge overflowed"))
            })
        })
    }
}

impl GroupUndo {
    fn charge(state: &State, group: GroupId) -> usize {
        state.aggregates.iter().fold(
            size_of::<Self>() + size_of::<GroupId>(),
            |charge, aggregate| {
            aggregate
                .states
                .iter()
                .map(|column| column.values[group.index()].size())
                .chain(std::iter::once(aggregate.result.values[group.index()].size()))
                .fold(charge, |charge, bytes| charge.saturating_add(bytes))
        })
    }

    fn new(state: &State, group: GroupId) -> Self {
        let states = state
            .aggregates
            .iter()
            .map(|aggregate| {
                aggregate
                    .states
                    .iter()
                    .map(|column| column.values[group.index()].clone())
                    .collect()
            })
            .collect();
        let results = state
            .aggregates
            .iter()
            .map(|aggregate| aggregate.result.values[group.index()].clone())
            .collect();
        let initialized = state
            .aggregates
            .iter()
            .map(|aggregate| aggregate.initialized[group.index()])
            .collect();
        Self {
            group,
            states,
            results,
            initialized,
        }
    }

    fn changed(&self, state: &State) -> bool {
        self.initialized.iter().enumerate().any(|(index, initialized)| {
            let aggregate = &state.aggregates[index];
            if aggregate.initialized[self.group.index()] != *initialized {
                return true;
            }
            if aggregate.result.values[self.group.index()] != self.results[index] {
                return true;
            }
            aggregate.states.iter().enumerate().any(|(field, column)| {
                column.values[self.group.index()] != self.states[index][field]
            })
        })
    }

    fn restore(self, state: &mut State) {
        for (index, aggregate) in state.aggregates.iter_mut().enumerate() {
            for (field, column) in aggregate.states.iter_mut().enumerate() {
                column.values[self.group.index()] = self.states[index][field].clone();
            }
            aggregate.result.values[self.group.index()] = self.results[index].clone();
            aggregate.initialized[self.group.index()] = self.initialized[index];
        }
    }
}

fn scalars(arrays: &[ArrayRef], row: usize, name: &str) -> Result<Vec<ScalarValue>> {
    arrays
        .iter()
        .map(|array| ScalarValue::try_from_array(array, row).map_err(|error| df_error(name, error)))
        .collect()
}

fn merge(accumulator: &mut dyn super::Accumulator, values: &[ScalarValue], name: &str) -> Result<()> {
    let arrays = values
        .iter()
        .map(ScalarValue::to_array)
        .collect::<datafusion::error::Result<Vec<_>>>()
        .map_err(|error| df_error(name, error))?;
    accumulator
        .merge_batch(&arrays)
        .map_err(|error| df_error(name, error))
}

fn restored_result(
    aggregate: usize,
    expression: &AggregateFunctionExpr,
    values: &[ScalarValue],
    grouped: bool,
    global: Option<&global_record::Proof>,
    sequential: bool,
    name: &str,
) -> Result<ScalarValue> {
    if ordered_last::selected(expression) {
        return ordered_last::result(values, expression.ignore_nulls(), name);
    }
    if let Some(global) = global {
        return global.result(aggregate, values, name);
    }
    if sequential && grouped_sum::selected(expression) {
        return grouped_sum::result(values, name);
    }
    if grouped && expression.fun().name() == "avg" {
        return native_grouped_result(expression, values, name);
    }
    let mut accumulator = expression
        .create_accumulator()
        .map_err(|error| df_error(name, error))?;
    merge(&mut *accumulator, values, name)?;
    accumulator.evaluate().map_err(|error| df_error(name, error))
}

fn global_fingerprint() -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"shared-keyed-columns-v1/global-reserved-slot-zero-v1");
    digest.update(datafusion::DATAFUSION_VERSION.as_bytes());
    digest.finalize().into()
}
