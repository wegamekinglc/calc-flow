use super::{
    AggregateFunctionExpr, ArrayRef, BooleanArray, CHUNK_ROWS, GroupKey, Int64Array, RecordBatch,
    Result, ScalarValue, SchemaRef, UInt64Array, grouped_sum, predicate,
};
use crate::StateBudget;
use crate::operator::keyed::{ExportValue, GroupId, KeyCut, KeySchema, KeyStore, MORSEL_ROWS};
use datafusion::arrow::{array::Array, datatypes::DataType};
use datafusion::execution::memory_pool::MemoryReservation;
use sha2::{Digest, Sha256};
#[cfg(feature = "benchmark-internals")]
use std::time::Instant;
use std::{
    mem::size_of,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

#[derive(Clone)]
pub(super) struct KeyedSum {
    state: Arc<Mutex<KeyedSumState>>,
}

pub(super) struct KeyedSumState {
    keys: KeyStore,
    sums: PartitionedSums,
    generation: u64,
    active: bool,
    #[cfg(test)]
    panic_group: Option<GroupId>,
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
    undo_reservations: Vec<MemoryReservation>,
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
    partitions: Vec<Vec<SumValue>>,
    marks: Vec<Vec<u64>>,
    groups: usize,
    parallel_min_rows: usize,
    #[cfg(test)]
    worker_ids: Arc<Mutex<std::collections::HashSet<std::thread::ThreadId>>>,
}

enum SumInput<'a> {
    Int64(&'a Int64Array),
    UInt64(&'a UInt64Array),
}

impl SumValue {
    fn record(self, keys: &[ArrayRef], schema: &SchemaRef) -> Result<RecordBatch> {
        let mut columns = keys.to_vec();
        columns.push(
            self.scalar()
                .to_array_of_size(1)
                .map_err(|error| super::df_error("keyed SUM", error))?,
        );
        RecordBatch::try_new(schema.clone(), columns)
            .map_err(|error| super::df_error("keyed SUM", error))
    }

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
                generation: 0,
                active: false,
                #[cfg(test)]
                panic_group: None,
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

    #[cfg(test)]
    pub(super) fn force_parallel_for_test(&self) {
        self.lock().sums.parallel_min_rows = 1;
    }

    #[cfg(test)]
    pub(super) fn worker_count_for_test(&self) -> usize {
        self.lock().sums.worker_ids.lock().unwrap().len()
    }

    #[cfg(test)]
    pub(super) fn panic_group_for_test(&self, group: GroupId) {
        self.lock().panic_group = Some(group);
    }

    #[cfg(test)]
    pub(super) fn clear_panic_for_test(&self) {
        self.lock().panic_group = None;
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
            undo_reservations: Vec::new(),
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
        #[cfg(feature = "benchmark-internals")]
        let route_started = Instant::now();
        let input = SumInput::new(values)?;
        let state = self.state.clone();
        let mut guard = state.lock().unwrap_or_else(PoisonError::into_inner);
        let groups = guard
            .keys
            .route_selected_with_limit(keys, selection, self.limit, check)?;
        grow(&mut guard)?;
        let count = guard.sums.partitions.len();
        self.undo_reservations = (0..count)
            .map(|_| guard.keys.transaction_reservation())
            .collect();
        let kind = guard.sums.kind;
        let parallel_min_rows = guard.sums.parallel_min_rows;
        let mut sums = std::mem::replace(&mut guard.sums, PartitionedSums::new(kind, count));
        drop(guard);
        #[cfg(feature = "benchmark-internals")]
        crate::benchmark_internals::note_keyed_sum_route_ns(
            u64::try_from(route_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        );
        #[cfg(feature = "benchmark-internals")]
        let apply_started = Instant::now();
        let parallel = count > 1 && groups.len() >= parallel_min_rows;
        let result = if parallel {
            self.apply_parallel(&state, &input, selection, filter, &groups, &mut sums)
        } else {
            self.apply_serial(&state, &input, selection, filter, &groups, &mut sums)
        };
        #[cfg(feature = "benchmark-internals")]
        crate::benchmark_internals::note_keyed_sum_apply_ns(
            u64::try_from(apply_started.elapsed().as_nanos()).unwrap_or(u64::MAX),
        );
        state.lock().unwrap_or_else(PoisonError::into_inner).sums = sums;
        result
    }

    fn apply_serial(
        &mut self,
        state: &Arc<Mutex<KeyedSumState>>,
        input: &SumInput,
        selection: Option<&BooleanArray>,
        filter: Option<&BooleanArray>,
        groups: &[GroupId],
        sums: &mut PartitionedSums,
    ) -> Result<()> {
        let kind = sums.kind;
        let (mut partitions, mut marks) = (
            std::mem::take(&mut sums.partitions),
            std::mem::take(&mut sums.marks),
        );
        for (row, group) in groups.iter().copied().enumerate() {
            let owner = group.index() % partitions.len();
            if let Err(error) = apply_row(
                state,
                RowContext {
                    generation: self.generation,
                    old_groups: self.old_groups,
                    partitions: partitions.len(),
                    kind,
                    input,
                    selection,
                    filter,
                    row,
                    group,
                },
                &mut partitions[owner],
                &mut marks[owner],
                &mut self.undo_parts(owner),
            ) {
                sums.partitions = partitions;
                sums.marks = marks;
                return Err(error);
            }
        }
        sums.partitions = partitions;
        sums.marks = marks;
        Ok(())
    }

    fn apply_parallel(
        &mut self,
        state: &Arc<Mutex<KeyedSumState>>,
        input: &SumInput,
        selection: Option<&BooleanArray>,
        filter: Option<&BooleanArray>,
        groups: &[GroupId],
        sums: &mut PartitionedSums,
    ) -> Result<()> {
        let kind = sums.kind;
        let mut work = sums
            .partitions
            .iter_mut()
            .zip(std::mem::take(&mut sums.marks))
            .map(|(values, marks)| PartitionWork {
                values: std::mem::take(values),
                marks,
                rows: Vec::new(),
            })
            .collect::<Vec<_>>();
        let partition_count = work.len();
        for (row, group) in groups.iter().copied().enumerate() {
            work[group.index() % partition_count]
                .rows
                .push((row, group));
        }
        let generation = self.generation;
        let old_groups = self.old_groups;
        let execution = PartitionExecution {
            state,
            generation,
            old_groups,
            partitions: partition_count,
            kind,
            input,
            selection,
            filter,
        };
        #[cfg(test)]
        let worker_ids = Arc::clone(&sums.worker_ids);
        let outcomes = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(work.len());
            for work in work {
                #[cfg(test)]
                let worker_ids = Arc::clone(&worker_ids);
                handles.push(scope.spawn(move || {
                    #[cfg(test)]
                    worker_ids
                        .lock()
                        .unwrap()
                        .insert(std::thread::current().id());
                    Self::run_partition_worker(execution, work)
                }));
            }
            handles
                .into_iter()
                .map(|handle| {
                    handle.join().map_err(|payload| {
                        let message: String = payload.downcast_ref::<&str>().map_or_else(
                            || "keyed SUM partition worker panicked".to_owned(),
                            |value| (*value).to_owned(),
                        );
                        super::df_error("keyed SUM", message)
                    })
                })
                .collect::<Result<Vec<_>>>()
        });
        let outcomes = outcomes?;
        sums.marks = (0..outcomes.len()).map(|_| Vec::new()).collect();
        let mut first_error = None;
        for (index, outcome) in outcomes.into_iter().enumerate() {
            sums.partitions[index] = outcome.values;
            sums.marks[index] = outcome.marks;
            self.affected.extend(outcome.affected);
            self.leased.extend(outcome.leased);
            self.undo.extend(outcome.undo);
            self.undo_reservations[index] = outcome.reservation;
            if let Err(error) = outcome.result
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn run_partition_worker(
        execution: PartitionExecution<'_>,
        work: PartitionWork,
    ) -> ParallelOutcome {
        let PartitionWork {
            mut values,
            mut marks,
            rows,
        } = work;
        let mut affected = Vec::new();
        let mut leased = Vec::new();
        let mut undo = Vec::new();
        let mut reservation = execution
            .state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .keys
            .transaction_reservation();
        let mut result = Ok(());
        for (row, group) in rows {
            let attempt = catch_unwind(AssertUnwindSafe(|| {
                apply_row(
                    execution.state,
                    RowContext {
                        generation: execution.generation,
                        old_groups: execution.old_groups,
                        partitions: execution.partitions,
                        kind: execution.kind,
                        input: execution.input,
                        selection: execution.selection,
                        filter: execution.filter,
                        row,
                        group,
                    },
                    &mut values,
                    &mut marks,
                    &mut TouchUndo {
                        affected: &mut affected,
                        leased: &mut leased,
                        undo: &mut undo,
                        reservation: &mut reservation,
                    },
                )
            }));
            match attempt {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    result = Err(error);
                    break;
                }
                Err(payload) => {
                    result = Err(super::df_error(
                        "keyed SUM",
                        format!(
                            "partition worker panicked: {}",
                            payload
                                .downcast_ref::<&str>()
                                .copied()
                                .unwrap_or("non-string panic payload")
                        ),
                    ));
                    break;
                }
            }
        }
        ParallelOutcome {
            values,
            marks,
            affected,
            leased,
            undo,
            reservation,
            result,
        }
    }

    fn undo_parts(&mut self, partition: usize) -> TouchUndo<'_> {
        TouchUndo {
            affected: &mut self.affected,
            leased: &mut self.leased,
            undo: &mut self.undo,
            reservation: &mut self.undo_reservations[partition],
        }
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

    pub(super) fn visit_changes(
        &mut self,
        schema: &SchemaRef,
        check: impl FnMut() -> Result<()>,
        reserve: impl FnMut(usize, usize) -> Result<()>,
        mut visit: impl FnMut(Option<RecordBatch>, RecordBatch) -> Result<()>,
    ) -> Result<()> {
        self.changed_groups(check, reserve, |state, group, previous, current| {
            let keys = state.keys.columns(&[group])?;
            let before = previous
                .map(|value| value.record(&keys, schema))
                .transpose()?;
            visit(before, current.record(&keys, schema)?)
        })
    }

    pub(super) fn change_record(
        &mut self,
        schema: &SchemaRef,
        mut check: impl FnMut() -> Result<()>,
        reserve: impl FnMut(usize, usize) -> Result<()>,
    ) -> Result<(RecordBatch, ArrayRef)> {
        let mut versions = Vec::new();
        let mut records = Vec::new();
        let mut operations = super::StringBuilder::with_capacity(0, 0);
        self.changed_groups(&mut check, reserve, |state, group, previous, current| {
            if versions.len() + 1 + usize::from(previous.is_some()) > MORSEL_ROWS {
                state.flush_versions(schema, &mut versions, &mut records)?;
            }
            if let Some(previous) = previous {
                versions.push((group, previous));
                operations.append_value("retract");
            }
            versions.push((group, current));
            operations.append_value("upsert");
            Ok(())
        })?;
        check()?;
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .flush_versions(schema, &mut versions, &mut records)?;
        let record = match records.as_slice() {
            [] => RecordBatch::new_empty(schema.clone()),
            [only] => only.clone(),
            _ => datafusion::arrow::compute::concat_batches(schema, &records)
                .map_err(|error| super::df_error("keyed SUM", error))?,
        };
        Ok((record, Arc::new(operations.finish())))
    }

    fn changed_groups(
        &mut self,
        mut check: impl FnMut() -> Result<()>,
        mut reserve: impl FnMut(usize, usize) -> Result<()>,
        mut visit: impl FnMut(&KeyedSumState, GroupId, Option<SumValue>, SumValue) -> Result<()>,
    ) -> Result<()> {
        check()?;
        self.undo.sort_unstable_by_key(|(group, _)| *group);
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let existing = self
            .undo
            .iter()
            .map(|&(group, previous)| Result::Ok((group, Some(previous))));
        // New groups occupy the suffix after every pre-existing group ID.
        let added = (self.old_groups..state.keys.len()).map(|slot| Ok((GroupId::new(slot)?, None)));
        for (index, item) in existing.chain(added).enumerate() {
            if index % 128 == 0 {
                check()?;
            }
            let (group, previous) = item?;
            state.visit_change(group, previous, &mut reserve, &mut visit)?;
        }
        Ok(())
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
        let mut guard = state.lock().unwrap_or_else(PoisonError::into_inner);
        let groups = guard.keys.route_with_limit(&keys, self.limit, check)?;
        grow(&mut guard)?;
        let kind = guard.sums.kind;
        let count = guard.sums.partitions.len();
        self.undo_reservations = (0..count)
            .map(|_| guard.keys.transaction_reservation())
            .collect();
        let mut sums = std::mem::replace(&mut guard.sums, PartitionedSums::new(kind, count));
        drop(guard);
        let mut result = Ok(());
        let generation = self.generation;
        let old_groups = self.old_groups;
        for (row, group) in groups.into_iter().enumerate() {
            let owner = group.index() % sums.partitions.len();
            if let Err(error) = Self::restore_row(
                &state,
                RowContext {
                    generation,
                    old_groups,
                    partitions: sums.partitions.len(),
                    kind,
                    input: &input,
                    selection: None,
                    filter: None,
                    row,
                    group,
                },
                replace,
                seen,
                &mut sums.partitions[owner],
                &mut sums.marks[owner],
                &mut self.undo_parts(owner),
            ) {
                result = Err(error);
                break;
            }
        }
        state.lock().unwrap_or_else(PoisonError::into_inner).sums = sums;
        result
    }

    fn restore_row(
        state: &Arc<Mutex<KeyedSumState>>,
        context: RowContext<'_>,
        replace: bool,
        seen: &mut std::collections::BTreeSet<GroupId>,
        values: &mut [SumValue],
        marks: &mut [u64],
        undo: &mut TouchUndo<'_>,
    ) -> Result<()> {
        let group = context.group;
        if !seen.insert(group) || (group.index() < context.old_groups && !replace) {
            return Err(super::df_error(
                "keyed SUM",
                "native state contains duplicate keys",
            ));
        }
        touch(state, &context, values, marks, undo)?;
        values[group.index() / context.partitions] = context
            .input
            .value(context.row)
            .unwrap_or_else(|| null_value(context.kind));
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
    Ok(())
}

#[derive(Clone, Copy)]
struct RowContext<'a> {
    generation: u64,
    old_groups: usize,
    partitions: usize,
    kind: SumKind,
    input: &'a SumInput<'a>,
    selection: Option<&'a BooleanArray>,
    filter: Option<&'a BooleanArray>,
    row: usize,
    group: GroupId,
}

#[derive(Clone, Copy)]
struct PartitionExecution<'a> {
    state: &'a Arc<Mutex<KeyedSumState>>,
    generation: u64,
    old_groups: usize,
    partitions: usize,
    kind: SumKind,
    input: &'a SumInput<'a>,
    selection: Option<&'a BooleanArray>,
    filter: Option<&'a BooleanArray>,
}

struct PartitionWork {
    values: Vec<SumValue>,
    marks: Vec<u64>,
    rows: Vec<(usize, GroupId)>,
}

struct ParallelOutcome {
    values: Vec<SumValue>,
    marks: Vec<u64>,
    affected: Vec<GroupId>,
    leased: Vec<GroupId>,
    undo: Vec<(GroupId, SumValue)>,
    reservation: MemoryReservation,
    result: Result<()>,
}

fn apply_row(
    state: &Arc<Mutex<KeyedSumState>>,
    context: RowContext<'_>,
    values: &mut [SumValue],
    marks: &mut [u64],
    undo: &mut TouchUndo<'_>,
) -> Result<()> {
    #[cfg(test)]
    if state
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .panic_group
        == Some(context.group)
    {
        panic!("injected keyed SUM partition panic");
    }
    if context
        .selection
        .is_some_and(|selection| !predicate::selected(selection, context.row))
    {
        return Ok(());
    }
    touch(state, &context, values, marks, undo)?;
    if context
        .filter
        .is_some_and(|filter| !predicate::selected(filter, context.row))
    {
        return Ok(());
    }
    if let Some(value) = context.input.value(context.row) {
        add_value(values, context.group.index() / context.partitions, value);
    }
    Ok(())
}

fn touch(
    state: &Arc<Mutex<KeyedSumState>>,
    context: &RowContext<'_>,
    values: &[SumValue],
    marks: &mut [u64],
    undo: &mut TouchUndo<'_>,
) -> Result<()> {
    let group = context.group;
    let local = group.index() / context.partitions;
    if marks[local] == context.generation {
        return Ok(());
    }
    let existing = usize::from(group.index() < context.old_groups);
    let charge =
        size_of::<GroupId>() + existing * (size_of::<GroupId>() + size_of::<(GroupId, SumValue)>());
    undo.reservation
        .try_grow(charge)
        .map_err(|error| super::df_error("keyed SUM", error))?;
    marks[local] = context.generation;
    undo.affected.push(group);
    if group.index() < context.old_groups {
        let previous = values
            .get(local)
            .copied()
            .unwrap_or_else(|| null_value(context.kind));
        state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .keys
            .acquire(group)?;
        undo.leased.push(group);
        undo.undo.push((group, previous));
    }
    Ok(())
}

fn add_value(values: &mut [SumValue], local: usize, value: SumValue) {
    let current = &mut values[local];
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

fn null_value(kind: SumKind) -> SumValue {
    match kind {
        SumKind::Int64 => SumValue::Int64(None),
        SumKind::UInt64 => SumValue::UInt64(None),
    }
}

impl Drop for Transaction {
    fn drop(&mut self) {
        self.settle(false);
    }
}

impl KeyedSumState {
    fn flush_versions(
        &self,
        schema: &SchemaRef,
        versions: &mut Vec<(GroupId, SumValue)>,
        records: &mut Vec<RecordBatch>,
    ) -> Result<()> {
        if versions.is_empty() {
            return Ok(());
        }
        let groups = versions.iter().map(|&(group, _)| group).collect::<Vec<_>>();
        let mut columns = self.keys.columns(&groups)?;
        columns.push(
            self.sums
                .values_array(versions.drain(..).map(|(_, value)| value)),
        );
        records.push(
            RecordBatch::try_new(schema.clone(), columns)
                .map_err(|error| super::df_error("keyed SUM", error))?,
        );
        Ok(())
    }

    fn visit_change(
        &self,
        group: GroupId,
        previous: Option<SumValue>,
        reserve: &mut impl FnMut(usize, usize) -> Result<()>,
        visit: &mut impl FnMut(&KeyedSumState, GroupId, Option<SumValue>, SumValue) -> Result<()>,
    ) -> Result<()> {
        let current = self.sums.value(group);
        if previous != Some(current) {
            reserve(
                self.keys.output_charge(&[group])?,
                1 + usize::from(previous.is_some()),
            )?;
            visit(self, group, previous, current)?;
        }
        Ok(())
    }

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
            partitions: (0..partitions).map(|_| Vec::new()).collect(),
            marks: (0..partitions).map(|_| Vec::new()).collect(),
            groups: 0,
            parallel_min_rows: 64_000,
            #[cfg(test)]
            worker_ids: Arc::new(Mutex::new(std::collections::HashSet::new())),
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
        assert!(self.partitions.iter().all(Vec::is_empty));
        self.partitions = (0..partitions).map(|_| Vec::new()).collect();
        self.marks = (0..partitions).map(|_| Vec::new()).collect();
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

    fn resize_marks(&mut self) {
        let groups = self.groups;
        let partitions = self.partitions.len();
        for (partition, marks) in self.marks.iter_mut().enumerate() {
            let len = if partition < groups {
                (groups - 1 - partition) / partitions + 1
            } else {
                0
            };
            marks.resize(len, 0);
        }
    }

    fn resize_values(&mut self) {
        let groups = self.groups;
        let partitions = self.partitions.len();
        let null = self.null();
        for (partition, values) in self.partitions.iter_mut().enumerate() {
            let len = if partition < groups {
                (groups - 1 - partition) / partitions + 1
            } else {
                0
            };
            values.resize(len, null);
        }
    }

    fn grow(&mut self, groups: usize) {
        self.groups = self.groups.max(groups);
        self.resize_values();
        self.resize_marks();
    }

    fn truncate(&mut self, groups: usize) {
        self.groups = groups;
        self.resize_values();
        for marks in &mut self.marks {
            marks.fill(0);
        }
        self.resize_marks();
    }

    fn value(&self, group: GroupId) -> SumValue {
        let local = group.index() / self.partitions.len();
        self.partitions[self.owner(group)]
            .get(local)
            .copied()
            .unwrap_or_else(|| self.null())
    }

    fn set(&mut self, group: GroupId, value: Option<SumValue>) {
        let null = self.null();
        let owner = self.owner(group);
        let local = group.index() / self.partitions.len();
        self.partitions[owner][local] = value.unwrap_or(null);
    }

    fn array(&self, groups: &[GroupId]) -> ArrayRef {
        self.values_array(groups.iter().map(|&group| self.value(group)))
    }

    fn values_array(&self, values: impl Iterator<Item = SumValue>) -> ArrayRef {
        match self.kind {
            SumKind::Int64 => Arc::new(Int64Array::from_iter(values.map(|value| match value {
                SumValue::Int64(value) => value,
                SumValue::UInt64(_) => unreachable!(),
            }))),
            SumKind::UInt64 => Arc::new(UInt64Array::from_iter(values.map(|value| match value {
                SumValue::UInt64(value) => value,
                SumValue::Int64(_) => unreachable!(),
            }))),
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
