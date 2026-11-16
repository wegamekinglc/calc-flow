use super::{
    AggregateFunctionExpr, ArrayRef, BooleanArray, CHUNK_ROWS, GroupKey, Int64Array, RecordBatch,
    Result, SchemaRef, UInt64Array, grouped_sum, predicate,
};
use crate::StateBudget;
use crate::operator::keyed::{GroupId, KeyCut, KeySchema, KeyStore, MORSEL_ROWS};
use datafusion::arrow::{array::Array, datatypes::DataType};
use datafusion::execution::memory_pool::MemoryReservation;
use std::{
    mem::size_of,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

#[derive(Clone)]
pub(super) struct KeyedSum {
    state: Arc<Mutex<KeyedSumState>>,
}

struct KeyedSumState {
    keys: KeyStore,
    sums: SumColumn,
    marks: Vec<u64>,
    generation: u64,
    active: bool,
}

pub(super) struct Transaction {
    state: Arc<Mutex<KeyedSumState>>,
    cut: KeyCut,
    old_groups: usize,
    generation: u64,
    limit: usize,
    undo: Vec<(GroupId, SumValue)>,
    affected: Vec<GroupId>,
    leased: Vec<GroupId>,
    undo_reservation: MemoryReservation,
    active: bool,
}

pub(super) struct Committed {
    pub affected: Vec<GroupId>,
}

#[derive(Clone, Copy, PartialEq)]
enum SumValue {
    Int64(Option<i64>),
    UInt64(Option<u64>),
}

enum SumColumn {
    Int64(Vec<Option<i64>>),
    UInt64(Vec<Option<u64>>),
}

enum SumInput<'a> {
    Int64(&'a Int64Array),
    UInt64(&'a UInt64Array),
}

impl KeyedSum {
    pub(super) fn plan(
        keys: &[GroupKey],
        aggregates: &[Arc<AggregateFunctionExpr>],
        reservation: MemoryReservation,
    ) -> Result<Option<Self>> {
        let [expression] = aggregates else {
            return Ok(None);
        };
        if keys.is_empty() || !grouped_sum::integer_sum(expression) {
            return Ok(None);
        }
        let fields = keys.iter().map(|key| key.field.clone()).collect();
        let schema = KeySchema::groupable(fields)?;
        let sums = match expression.field().data_type() {
            DataType::Int64 => SumColumn::Int64(Vec::new()),
            DataType::UInt64 => SumColumn::UInt64(Vec::new()),
            _ => return Ok(None),
        };
        Ok(Some(Self {
            state: Arc::new(Mutex::new(KeyedSumState {
                keys: KeyStore::new(schema, reservation),
                sums,
                marks: Vec::new(),
                generation: 0,
                active: false,
            })),
        }))
    }

    pub(super) fn len(&self) -> usize {
        self.lock().keys.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.lock().keys.is_empty()
    }

    pub(super) fn layout_fingerprint(&self) -> [u8; 32] {
        self.lock().keys.layout_fingerprint()
    }

    pub(super) fn canonical_slot(&self, slot: usize) -> Result<Arc<[u8]>> {
        let group = GroupId::new(slot)?;
        Ok(self.lock().keys.canonical(group))
    }

    pub(super) fn begin(&self, budget: Option<StateBudget>) -> Result<Transaction> {
        let mut state = self.lock();
        if state.active {
            return Err(super::df_error(
                "keyed SUM",
                "a keyed SUM transaction is already active",
            ));
        }
        let cut = state.keys.cut();
        let old_groups = state.keys.len();
        let generation = state
            .generation
            .checked_add(1)
            .ok_or_else(|| super::df_error("keyed SUM", "transaction generation overflowed"))?;
        state.generation = generation;
        state.active = true;
        let undo_reservation = state.keys.transaction_reservation();
        let limit = budget.map_or(usize::MAX, |budget| {
            usize::try_from(budget.max_rows()).unwrap_or(usize::MAX)
        });
        Ok(Transaction {
            state: self.state.clone(),
            cut,
            old_groups,
            generation,
            limit,
            undo: Vec::new(),
            affected: Vec::new(),
            leased: Vec::new(),
            undo_reservation,
            active: true,
        })
    }

    pub(super) fn export(
        &self,
        slots: Option<&[usize]>,
        schema: SchemaRef,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<RecordBatch>> {
        self.lock().records(slots, schema, check)
    }

    pub(super) fn output_charge(&self, slots: Option<&[usize]>) -> Result<usize> {
        self.lock().output_charge(slots)
    }

    pub(super) fn restore_records(
        &self,
        records: &[RecordBatch],
        key_count: usize,
        replace: bool,
        budget: Option<StateBudget>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        let mut transaction = self.begin(budget)?;
        let mut seen = std::collections::BTreeSet::new();
        for record in records {
            for start in (0..record.num_rows()).step_by(MORSEL_ROWS) {
                check()?;
                let rows = MORSEL_ROWS.min(record.num_rows() - start);
                let chunk = record.slice(start, rows);
                transaction.restore_chunk(&chunk, key_count, replace, &mut seen, check)?;
            }
            if record.num_rows() == 0 {
                check()?;
            }
        }
        transaction.commit();
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn canonical_births(&self) -> usize {
        self.lock().keys.canonical_births()
    }

    #[cfg(test)]
    pub(super) fn next_id(&self) -> usize {
        self.lock().keys.next_id()
    }

    #[cfg(test)]
    pub(super) fn reserved_bytes(&self) -> usize {
        self.lock().keys.reserved_bytes()
    }

    #[cfg(test)]
    pub(super) fn canonical_bytes(&self) -> Vec<Arc<[u8]>> {
        self.lock().keys.canonical_bytes()
    }

    fn lock(&self) -> MutexGuard<'_, KeyedSumState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Transaction {
    pub(super) fn apply(
        &mut self,
        keys: &[ArrayRef],
        values: &ArrayRef,
        selection: Option<&BooleanArray>,
        filter: Option<&BooleanArray>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        let input = SumInput::new(values)?;
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let groups = state
            .keys
            .route_selected_with_limit(keys, selection, self.limit, check)?;
        grow(&mut state)?;
        for (row, group) in groups.into_iter().enumerate() {
            if selection.is_some_and(|selection| !predicate::selected(selection, row)) {
                continue;
            }
            touch(
                &mut state,
                group,
                self.generation,
                self.old_groups,
                &mut TouchUndo {
                    affected: &mut self.affected,
                    leased: &mut self.leased,
                    undo: &mut self.undo,
                    reservation: &mut self.undo_reservation,
                },
            )?;
            if filter.is_some_and(|filter| !predicate::selected(filter, row)) {
                continue;
            }
            if let Some(value) = input.value(row) {
                state.sums.add(group, value);
            }
        }
        Ok(())
    }

    pub(super) fn records(
        &self,
        schema: SchemaRef,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<RecordBatch>> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .records(None, schema, check)
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
            .record(start, end, None, schema)
    }

    pub(super) fn dirty_output_chunks(&self) -> std::collections::BTreeSet<usize> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let mut chunks = std::collections::BTreeSet::new();
        for (group, previous) in &self.undo {
            if state.sums.value(*group) != *previous {
                chunks.insert(group.index() / CHUNK_ROWS);
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

    fn restore_chunk(
        &mut self,
        chunk: &RecordBatch,
        key_count: usize,
        replace: bool,
        seen: &mut std::collections::BTreeSet<GroupId>,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        let keys = chunk.columns()[..key_count].to_vec();
        let input = SumInput::new(chunk.column(key_count))?;
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let groups = state.keys.route_with_limit(&keys, self.limit, check)?;
        grow(&mut state)?;
        for (row, group) in groups.into_iter().enumerate() {
            if !seen.insert(group) || (group.index() < self.old_groups && !replace) {
                return Err(super::df_error(
                    "keyed SUM",
                    "native state contains duplicate keys",
                ));
            }
            touch(
                &mut state,
                group,
                self.generation,
                self.old_groups,
                &mut TouchUndo {
                    affected: &mut self.affected,
                    leased: &mut self.leased,
                    undo: &mut self.undo,
                    reservation: &mut self.undo_reservation,
                },
            )?;
            state.sums.set(group, input.value(row));
        }
        Ok(())
    }

    fn settle(&mut self, commit: bool) {
        if !self.active {
            return;
        }
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if !commit {
            for (group, value) in self.undo.drain(..).rev() {
                state.sums.set(group, Some(value));
            }
            state.sums.truncate(self.old_groups);
            state.marks.truncate(self.old_groups);
            state.keys.rollback(self.cut);
        }
        for group in self.leased.drain(..) {
            let _ = state.keys.release(group);
        }
        state.active = false;
        self.active = false;
    }
}

struct TouchUndo<'a> {
    affected: &'a mut Vec<GroupId>,
    leased: &'a mut Vec<GroupId>,
    undo: &'a mut Vec<(GroupId, SumValue)>,
    reservation: &'a mut MemoryReservation,
}

fn grow(state: &mut KeyedSumState) -> Result<()> {
    let groups = state.keys.len();
    let new = groups.saturating_sub(state.sums.len());
    if new == 0 {
        return Ok(());
    }
    let width = state.sums.width() + size_of::<u64>();
    let bytes = new
        .checked_mul(width)
        .ok_or_else(|| super::df_error("keyed SUM", "typed state charge overflowed"))?;
    state.keys.charge(bytes)?;
    state.sums.grow(groups);
    state.marks.resize(groups, 0);
    Ok(())
}

fn touch(
    state: &mut KeyedSumState,
    group: GroupId,
    generation: u64,
    old_groups: usize,
    undo: &mut TouchUndo<'_>,
) -> Result<()> {
    if state.marks[group.index()] == generation {
        return Ok(());
    }
    let existing = usize::from(group.index() < old_groups);
    let charge =
        size_of::<GroupId>() + existing * (size_of::<GroupId>() + size_of::<(GroupId, SumValue)>());
    undo.reservation
        .try_grow(charge)
        .map_err(|error| super::df_error("keyed SUM", error))?;
    state.marks[group.index()] = generation;
    undo.affected.push(group);
    if group.index() < old_groups {
        state.keys.acquire(group)?;
        undo.leased.push(group);
        undo.undo.push((group, state.sums.value(group)));
    }
    Ok(())
}

impl Drop for Transaction {
    fn drop(&mut self) {
        self.settle(false);
    }
}

impl KeyedSumState {
    fn records(
        &self,
        slots: Option<&[usize]>,
        schema: SchemaRef,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<RecordBatch>> {
        let count = slots.map_or(self.keys.len(), <[usize]>::len);
        if count == 0 {
            check()?;
            return Ok(vec![RecordBatch::new_empty(schema)]);
        }
        let mut records = Vec::with_capacity(count.div_ceil(CHUNK_ROWS));
        for start in (0..count).step_by(CHUNK_ROWS) {
            check()?;
            let end = count.min(start + CHUNK_ROWS);
            records.push(self.record(start, end, slots, schema.clone())?);
        }
        Ok(records)
    }

    fn record(
        &self,
        start: usize,
        end: usize,
        slots: Option<&[usize]>,
        schema: SchemaRef,
    ) -> Result<RecordBatch> {
        let groups = (start..end)
            .map(|row| {
                let slot = slots.map_or(row, |slots| slots[row]);
                GroupId::new(slot)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut columns = self.keys.columns(&groups)?;
        columns.push(self.sums.array(&groups));
        RecordBatch::try_new(schema, columns).map_err(|error| super::df_error("keyed SUM", error))
    }

    fn output_charge(&self, slots: Option<&[usize]>) -> Result<usize> {
        let count = slots.map_or(self.keys.len(), <[usize]>::len);
        let groups = (0..count)
            .map(|row| GroupId::new(slots.map_or(row, |slots| slots[row])))
            .collect::<Result<Vec<_>>>()?;
        self.keys.output_charge(&groups)
    }
}

impl<'a> SumInput<'a> {
    fn new(array: &'a ArrayRef) -> Result<Self> {
        match array.data_type() {
            DataType::Int64 => array
                .as_any()
                .downcast_ref::<Int64Array>()
                .map(SumInput::Int64)
                .ok_or_else(|| super::df_error("keyed SUM", "Int64 input array differs")),
            DataType::UInt64 => array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .map(SumInput::UInt64)
                .ok_or_else(|| super::df_error("keyed SUM", "UInt64 input array differs")),
            _ => Err(super::df_error(
                "keyed SUM",
                "integer SUM input type differs",
            )),
        }
    }

    fn value(&self, row: usize) -> Option<SumValue> {
        match self {
            Self::Int64(values) if values.is_valid(row) => {
                Some(SumValue::Int64(Some(values.value(row))))
            }
            Self::UInt64(values) if values.is_valid(row) => {
                Some(SumValue::UInt64(Some(values.value(row))))
            }
            Self::Int64(_) | Self::UInt64(_) => None,
        }
    }
}

impl SumColumn {
    fn len(&self) -> usize {
        match self {
            Self::Int64(values) => values.len(),
            Self::UInt64(values) => values.len(),
        }
    }

    fn width(&self) -> usize {
        match self {
            Self::Int64(_) => size_of::<Option<i64>>(),
            Self::UInt64(_) => size_of::<Option<u64>>(),
        }
    }

    fn grow(&mut self, groups: usize) {
        match self {
            Self::Int64(values) => values.resize(groups, None),
            Self::UInt64(values) => values.resize(groups, None),
        }
    }

    fn truncate(&mut self, groups: usize) {
        match self {
            Self::Int64(values) => values.truncate(groups),
            Self::UInt64(values) => values.truncate(groups),
        }
    }

    fn value(&self, group: GroupId) -> SumValue {
        match self {
            Self::Int64(values) => SumValue::Int64(values[group.index()]),
            Self::UInt64(values) => SumValue::UInt64(values[group.index()]),
        }
    }

    fn set(&mut self, group: GroupId, value: Option<SumValue>) {
        match self {
            Self::Int64(values) => {
                values[group.index()] = match value {
                    Some(SumValue::Int64(value)) => value,
                    None => None,
                    _ => unreachable!("typed SUM state and input were compiled together"),
                };
            }
            Self::UInt64(values) => {
                values[group.index()] = match value {
                    Some(SumValue::UInt64(value)) => value,
                    None => None,
                    _ => unreachable!("typed SUM state and input were compiled together"),
                };
            }
        }
    }

    fn add(&mut self, group: GroupId, value: SumValue) {
        match (self, value) {
            (Self::Int64(values), SumValue::Int64(value)) => {
                let current = &mut values[group.index()];
                *current = value.map(|value| current.unwrap_or(0).wrapping_add(value));
            }
            (Self::UInt64(values), SumValue::UInt64(value)) => {
                let current = &mut values[group.index()];
                *current = value.map(|value| current.unwrap_or(0).wrapping_add(value));
            }
            _ => unreachable!("typed SUM state and input were compiled together"),
        }
    }

    fn array(&self, groups: &[GroupId]) -> ArrayRef {
        match self {
            Self::Int64(values) => Arc::new(Int64Array::from(
                groups
                    .iter()
                    .map(|group| values[group.index()])
                    .collect::<Vec<_>>(),
            )),
            Self::UInt64(values) => Arc::new(UInt64Array::from(
                groups
                    .iter()
                    .map(|group| values[group.index()])
                    .collect::<Vec<_>>(),
            )),
        }
    }
}
