use std::collections::{BTreeMap, VecDeque};

use super::super::{
    BufferedRow, CompiledRollingSpec, KeyValue, Result, RollingStreamState, StateBudget,
    TableBatch, history_event_time, operator_error,
};
use super::{controlled, resources};

mod bytes;

type Entity = Vec<Option<KeyValue>>;
type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

pub(super) fn scratch(
    input: &TableBatch,
    compiled: &CompiledRollingSpec,
    budget: StateBudget,
    node: &str,
) -> Result<u64> {
    let rows = input.batches().iter().try_fold(0_usize, |rows, batch| {
        rows.checked_add(batch.num_rows())
            .ok_or_else(|| overflow(node))
    })?;
    bytes::plan::<&Entity>(rows, capacity(rows, budget), retains_rows(compiled), node)
}

pub(super) fn finite(
    rows: &[BufferedRow],
    compiled: &CompiledRollingSpec,
    budget: StateBudget,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    check()?;
    let mut simulation = Simulation::new(compiled, budget, node);
    for (index, row) in rows.iter().enumerate() {
        controlled::tick(index, check)?;
        simulation.observe(row, check)?;
    }
    check()
}

pub(super) fn stream_scratch(
    source: &RollingStreamState,
    compiled: &CompiledRollingSpec,
    closing_count: usize,
    budget: StateBudget,
    node: &str,
    check: Check<'_>,
) -> Result<u64> {
    let counts = source_counts(source, node, check)?;
    check_limit(counts, budget, node)?;
    let entities = counts
        .entities
        .checked_add(closing_count)
        .ok_or_else(|| overflow(node))?;
    let pushes = closing_pushes(counts.retained, closing_count, compiled, node)?;
    bytes::plan::<&Entity>(pushes, capacity(entities, budget), pushes > 0, node)
}

pub(super) fn stream(
    rows: &[BufferedRow],
    source: &RollingStreamState,
    compiled: &CompiledRollingSpec,
    budget: StateBudget,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    validate_closing(rows, source, node, check)?;
    let mut simulation = Simulation::new(compiled, budget, node);
    simulation.seed(source, check)?;
    for (index, row) in rows.iter().enumerate() {
        controlled::tick(index, check)?;
        simulation.pending = simulation
            .pending
            .checked_sub(1)
            .ok_or_else(|| overflow(node))?;
        simulation.observe(row, check)?;
    }
    check()
}

#[derive(Clone, Copy)]
struct Counts {
    entities: usize,
    retained: usize,
    pending: usize,
}
fn source_counts(source: &RollingStreamState, node: &str, check: Check<'_>) -> Result<Counts> {
    check()?;
    let mut retained = 0_usize;
    for (index, entity) in source.histories.by_entity.values().enumerate() {
        controlled::tick(index, check)?;
        retained = retained
            .checked_add(entity.rows.len())
            .ok_or_else(|| overflow(node))?;
    }
    check()?;
    Ok(Counts {
        entities: source.histories.by_entity.len(),
        retained,
        pending: source.buffer.len(),
    })
}
fn closing_pushes(
    history: usize,
    closing: usize,
    compiled: &CompiledRollingSpec,
    node: &str,
) -> Result<usize> {
    if retains_rows(compiled) {
        history.checked_add(closing).ok_or_else(|| overflow(node))
    } else {
        Ok(history)
    }
}
fn validate_closing(
    rows: &[BufferedRow],
    source: &RollingStreamState,
    node: &str,
    check: Check<'_>,
) -> Result<()> {
    check()?;
    let mut previous = None;
    for (index, row) in rows.iter().enumerate() {
        controlled::tick(index, check)?;
        if previous.is_some_and(|previous| previous >= &row.identity)
            || !source.buffer.contains_key(&row.identity)
        {
            return Err(operator_error(
                node,
                "rolling closing row prefix is not a pending canonical selection",
            ));
        }
        previous = Some(&row.identity);
    }
    check()
}
struct Simulation<'a, 'c> {
    tails: BTreeMap<&'a Entity, VecDeque<i64>>,
    retained: usize,
    pending: usize,
    work: usize,
    compiled: &'c CompiledRollingSpec,
    budget: StateBudget,
    node: &'c str,
}
impl<'a, 'c> Simulation<'a, 'c> {
    fn new(compiled: &'c CompiledRollingSpec, budget: StateBudget, node: &'c str) -> Self {
        Self {
            tails: BTreeMap::new(),
            retained: 0,
            pending: 0,
            work: 0,
            compiled,
            budget,
            node,
        }
    }
    fn seed(&mut self, source: &'a RollingStreamState, check: Check<'_>) -> Result<()> {
        let counts = source_counts(source, self.node, check)?;
        check_limit(counts, self.budget, self.node)?;
        self.pending = counts.pending;
        self.retained = counts.retained;
        for (key, entity) in &source.histories.by_entity {
            bytes::tick(&mut self.work, check)?;
            let tail = self.tails.entry(key).or_default();
            for values in &entity.rows {
                bytes::tick(&mut self.work, check)?;
                tail.push_back(history_event_time(values, self.compiled));
            }
        }
        check()
    }
    fn observe(&mut self, row: &'a BufferedRow, check: Check<'_>) -> Result<()> {
        bytes::tick(&mut self.work, check)?;
        self.ensure_entity(&row.identity.entity)?;
        let history = self
            .tails
            .get_mut(&row.identity.entity)
            .expect("initialized entity");
        let previous = history.len();
        trim(
            history,
            row.identity.event_time,
            self.compiled,
            &mut self.work,
            check,
        )?;
        let retains = retains_rows(self.compiled);
        let retained = replacement(self.retained, previous - history.len(), retains, self.node)?;
        check_limit(
            Counts {
                entities: self.tails.len(),
                retained,
                pending: self.pending,
            },
            self.budget,
            self.node,
        )?;
        if retains {
            self.tails
                .get_mut(&row.identity.entity)
                .expect("initialized entity")
                .push_back(row.identity.event_time);
        }
        self.retained = retained;
        Ok(())
    }
    fn ensure_entity(&mut self, entity: &'a Entity) -> Result<()> {
        if self.tails.contains_key(entity) {
            return Ok(());
        }
        let entities = self
            .tails
            .len()
            .checked_add(1)
            .ok_or_else(|| overflow(self.node))?;
        check_limit(
            Counts {
                entities,
                retained: self.retained,
                pending: self.pending,
            },
            self.budget,
            self.node,
        )?;
        self.tails.insert(entity, VecDeque::new());
        Ok(())
    }
}
fn replacement(retained: usize, removed: usize, append: bool, node: &str) -> Result<usize> {
    retained
        .checked_sub(removed)
        .and_then(|retained| retained.checked_add(usize::from(append)))
        .ok_or_else(|| overflow(node))
}
fn trim(
    history: &mut VecDeque<i64>,
    event_time: i64,
    compiled: &CompiledRollingSpec,
    position: &mut usize,
    check: Check<'_>,
) -> Result<()> {
    let count = usize::try_from(compiled.max_row_retention).unwrap_or(usize::MAX);
    let bound = compiled
        .max_duration_micros
        .map(|duration| i128::from(event_time) - i128::from(duration));
    while history.len() >= count {
        bytes::tick(position, check)?;
        let Some(&first) = history.front() else {
            break;
        };
        if bound.is_some_and(|bound| i128::from(first) > bound) {
            break;
        }
        history.pop_front();
    }
    Ok(())
}
fn check_limit(counts: Counts, budget: StateBudget, node: &str) -> Result<()> {
    let rows = counts
        .entities
        .checked_add(counts.retained)
        .and_then(|rows| rows.checked_add(counts.pending))
        .ok_or_else(|| overflow(node))?;
    if resources::size(rows, node)? > budget.max_rows() {
        return Err(operator_error(
            node,
            "rolling.state_budget.max_rows exceeded at a retained row prefix",
        ));
    }
    Ok(())
}
fn capacity(rows: usize, budget: StateBudget) -> usize {
    rows.min(usize::try_from(budget.max_rows()).unwrap_or(usize::MAX))
}
fn retains_rows(compiled: &CompiledRollingSpec) -> bool {
    compiled.max_row_retention > 0 || compiled.max_duration_micros.is_some()
}
fn overflow(node: &str) -> crate::CalcFlowError {
    operator_error(node, "rolling retained row prefix count overflowed")
}

#[cfg(test)]
mod state_tests;
