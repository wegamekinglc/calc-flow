use super::{
    AggregateFunctionExpr, ArrayRef, BooleanArray, CHUNK_ROWS, GroupKey, Int64Array, RecordBatch,
    Result, ScalarValue, SchemaRef, UInt64Array, grouped_sum, predicate,
};
use crate::StateBudget;
use crate::operator::keyed::{ExportValue, GroupId, KeyCut, KeySchema, KeyStore, MORSEL_ROWS};
use datafusion::arrow::{array::Array, datatypes::DataType};
use datafusion::execution::memory_pool::MemoryReservation;
use sha2::{Digest, Sha256};
use std::{
    mem::size_of,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

#[derive(Clone)]
pub(super) struct KeyedSum {
    state: Arc<Mutex<KeyedSumState>>,
}

pub(super) struct KeyedSumState {
    keys: KeyStore,
    sums: PartitionedSums,
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

#[derive(Clone, Copy)]
enum SumKind {
    Int64,
    UInt64,
}

struct PartitionedSums {
    kind: SumKind,
    partitions: Vec<std::collections::BTreeMap<GroupId, SumValue>>,
    groups: usize,
}

enum SumInput<'a> {
    Int64(&'a Int64Array),
    UInt64(&'a UInt64Array),
}

impl SumValue {
    fn scalar(self) -> ScalarValue {
        match self {
            Self::Int64(value) => ScalarValue::Int64(value),
            Self::UInt64(value) => ScalarValue::UInt64(value),
        }
    }
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
        let kind = match expression.field().data_type() {
            DataType::Int64 => SumKind::Int64,
            DataType::UInt64 => SumKind::UInt64,
            _ => return Ok(None),
        };
        let sums = PartitionedSums::new(kind, default_partitions());
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
        let state = self.lock();
        state.sums.layout_fingerprint(&state.keys)
    }

    #[cfg(test)]
    pub(super) fn with_partition_count_for_test(self, partitions: usize) -> Self {
        assert!(self.is_empty());
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .sums
            .set_partition_count(partitions);
        self
    }

    #[cfg(test)]
    pub(super) fn partition_count_for_test(&self) -> usize {
        self.lock().sums.partition_count()
    }

    #[cfg(test)]
    pub(super) fn owner_for_test(&self, group: GroupId) -> usize {
        self.lock().sums.owner(group)
    }

    pub(super) fn group_count(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .keys
            .len()
    }

    pub(super) fn first_key(&self, slot: usize) -> Result<Arc<[u8]>> {
        let group = GroupId::new(slot)?;
        Ok(self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .keys
            .canonical(group))
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

    pub(super) fn export_rows(&self, slots: Option<&[usize]>) -> Result<Vec<Vec<ExportValue>>> {
        let state = self.lock();
        let count = slots.map_or(state.keys.len(), <[usize]>::len);
        (0..count)
            .map(|row| {
                let slot = slots.map_or(row, |slots| slots[row]);
                let group = GroupId::new(slot)?;
                let mut values = (0..state.keys.fields_len())
                    .map(|column| state.keys.export_value(group, column))
                    .collect::<Result<Vec<_>>>()?;
                values.push(ExportValue::Scalar(state.sums.value(group).scalar()));
                Ok(values)
            })
            .collect()
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
        let state = self.state.clone();
        let mut state = state.lock().unwrap_or_else(PoisonError::into_inner);
        let groups = state
            .keys
            .route_selected_with_limit(keys, selection, self.limit, check)?;
        grow(&mut state)?;
        for (row, group) in groups.into_iter().enumerate() {
            self.apply_row(&mut state, &input, selection, filter, row, group)?;
        }
        Ok(())
    }

    fn undo_parts(&mut self) -> TouchUndo<'_> {
        TouchUndo {
            affected: &mut self.affected,
            leased: &mut self.leased,
            undo: &mut self.undo,
            reservation: &mut self.undo_reservation,
        }
    }

    fn apply_row(
        &mut self,
        state: &mut KeyedSumState,
        input: &SumInput,
        selection: Option<&BooleanArray>,
        filter: Option<&BooleanArray>,
        row: usize,
        group: GroupId,
    ) -> Result<()> {
        if selection.is_some_and(|selection| !predicate::selected(selection, row)) {
            return Ok(());
        }
        touch(
            state,
            group,
            self.generation,
            self.old_groups,
            &mut self.undo_parts(),
        )?;
        if filter.is_some_and(|filter| !predicate::selected(filter, row)) {
            return Ok(());
        }
        if let Some(value) = input.value(row) {
            state.sums.add(group, value);
        }
        Ok(())
    }

    #[cfg(test)]
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

    pub(super) fn group_count(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .keys
            .len()
    }

    pub(super) fn first_key(&self, slot: usize) -> Result<Arc<[u8]>> {
        let group = GroupId::new(slot)?;
        Ok(self
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .keys
            .canonical(group))
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
        let state = self.state.clone();
        let mut state = state.lock().unwrap_or_else(PoisonError::into_inner);
        let groups = state.keys.route_with_limit(&keys, self.limit, check)?;
        grow(&mut state)?;
        for (row, group) in groups.into_iter().enumerate() {
            self.restore_row(&mut state, &input, replace, seen, row, group)?;
        }
        Ok(())
    }

    fn restore_row(
        &mut self,
        state: &mut KeyedSumState,
        input: &SumInput,
        replace: bool,
        seen: &mut std::collections::BTreeSet<GroupId>,
        row: usize,
        group: GroupId,
    ) -> Result<()> {
        if !seen.insert(group) || (group.index() < self.old_groups && !replace) {
            return Err(super::df_error(
                "keyed SUM",
                "native state contains duplicate keys",
            ));
        }
        touch(
            state,
            group,
            self.generation,
            self.old_groups,
            &mut self.undo_parts(),
        )?;
        state.sums.set(group, input.value(row));
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

impl PartitionedSums {
    fn new(kind: SumKind, partitions: usize) -> Self {
        assert!(partitions > 0);
        Self {
            kind,
            partitions: (0..partitions)
                .map(|_| std::collections::BTreeMap::new())
                .collect(),
            groups: 0,
        }
    }

    fn layout_fingerprint(&self, keys: &KeyStore) -> [u8; 32] {
        let mut identity = keys.layout_fingerprint().to_vec();
        identity.extend_from_slice(b":fixed-group-id-modulo-v1:");
        identity.extend_from_slice(self.partitions.len().to_string().as_bytes());
        Sha256::digest(identity).into()
    }

    #[cfg(test)]
    fn set_partition_count(&mut self, partitions: usize) {
        assert!(partitions > 0);
        assert_eq!(self.groups, 0);
        assert!(
            self.partitions
                .iter()
                .all(std::collections::BTreeMap::is_empty)
        );
        self.partitions = (0..partitions)
            .map(|_| std::collections::BTreeMap::new())
            .collect();
    }

    #[cfg(test)]
    fn partition_count(&self) -> usize {
        self.partitions.len()
    }

    fn owner(&self, group: GroupId) -> usize {
        group.index() % self.partitions.len()
    }

    fn len(&self) -> usize {
        self.groups
    }

    fn width(&self) -> usize {
        let scalar = match self.kind {
            SumKind::Int64 => size_of::<Option<i64>>(),
            SumKind::UInt64 => size_of::<Option<u64>>(),
        };
        scalar + size_of::<(GroupId, SumValue)>() + 6 * size_of::<usize>()
    }

    fn grow(&mut self, groups: usize) {
        self.groups = self.groups.max(groups);
    }

    fn truncate(&mut self, groups: usize) {
        self.groups = groups;
        for partition in &mut self.partitions {
            partition.retain(|group, _| group.index() < groups);
        }
    }

    fn value(&self, group: GroupId) -> SumValue {
        self.partitions[self.owner(group)]
            .get(&group)
            .copied()
            .unwrap_or_else(|| self.null())
    }

    fn set(&mut self, group: GroupId, value: Option<SumValue>) {
        let null = self.null();
        let owner = self.owner(group);
        self.partitions[owner].insert(group, value.unwrap_or(null));
    }

    fn add(&mut self, group: GroupId, value: SumValue) {
        let null = self.null();
        let owner = self.owner(group);
        let current = self.partitions[owner].entry(group).or_insert(null);
        match current {
            SumValue::Int64(current_value) => {
                let SumValue::Int64(value) = value else {
                    unreachable!("typed SUM state and input were compiled together")
                };
                *current_value = value.map(|value| current_value.unwrap_or(0).wrapping_add(value));
            }
            SumValue::UInt64(current_value) => {
                let SumValue::UInt64(value) = value else {
                    unreachable!("typed SUM state and input were compiled together")
                };
                *current_value = value.map(|value| current_value.unwrap_or(0).wrapping_add(value));
            }
        }
    }

    fn array(&self, groups: &[GroupId]) -> ArrayRef {
        match self.kind {
            SumKind::Int64 => Arc::new(Int64Array::from(
                groups
                    .iter()
                    .map(|group| match self.value(*group) {
                        SumValue::Int64(value) => value,
                        SumValue::UInt64(_) => unreachable!(),
                    })
                    .collect::<Vec<_>>(),
            )),
            SumKind::UInt64 => Arc::new(UInt64Array::from(
                groups
                    .iter()
                    .map(|group| match self.value(*group) {
                        SumValue::Int64(_) => unreachable!(),
                        SumValue::UInt64(value) => value,
                    })
                    .collect::<Vec<_>>(),
            )),
        }
    }

    fn null(&self) -> SumValue {
        match self.kind {
            SumKind::Int64 => SumValue::Int64(None),
            SumKind::UInt64 => SumValue::UInt64(None),
        }
    }
}

fn default_partitions() -> usize {
    std::thread::available_parallelism().map_or(1, |available| available.get().min(4))
}
