use datafusion::common::ScalarValue;

use super::{
    Check,
    bytes::{add, growing, invalid, size, slots, total},
    clones::scalar_heap,
};
use crate::Result;
use crate::operator::rolling::{
    BTreeMap, BufferedRow, CompiledRollingSpec, EntityRollingState, KeyValue, RollingHistories,
    RollingStreamState, RowIdentity, WindowState,
};

pub(super) struct Source<'a> {
    pending: Pending<'a>,
    histories: Option<&'a RollingHistories>,
    pub compiled: &'a CompiledRollingSpec,
}
enum Pending<'a> {
    Buffered(&'a BTreeMap<RowIdentity, BufferedRow>),
    Finite(&'a [BufferedRow]),
}
impl Pending<'_> {
    fn rows(&self) -> impl Iterator<Item = &BufferedRow> {
        let buffer = match self {
            Self::Buffered(buffer) => Some(*buffer),
            Self::Finite(_) => None,
        };
        let finite = match self {
            Self::Finite(rows) => *rows,
            Self::Buffered(_) => &[],
        };
        buffer
            .into_iter()
            .flat_map(|buffer| buffer.values())
            .chain(finite)
    }
}
impl<'a> Source<'a> {
    pub fn stream(state: &'a RollingStreamState, compiled: &'a CompiledRollingSpec) -> Self {
        Self {
            pending: Pending::Buffered(&state.buffer),
            histories: Some(&state.histories),
            compiled,
        }
    }
    pub fn finite(rows: &'a [BufferedRow], compiled: &'a CompiledRollingSpec) -> Self {
        Self {
            pending: Pending::Finite(rows),
            histories: None,
            compiled,
        }
    }
    fn histories(&self) -> impl Iterator<Item = (&Vec<Option<KeyValue>>, &EntityRollingState)> {
        self.histories
            .into_iter()
            .flat_map(|history| history.by_entity.iter())
    }
}
#[derive(Default)]
pub(super) struct Summary {
    pub history_rows: usize,
    pub row_width: usize,
    pub row_heap: u64,
    pub entity_heap: u64,
    pub sequence_heap: u64,
    pub prior_copies: u64,
}
impl Source<'_> {
    pub fn summary(&self, check: Check<'_>) -> Result<Summary> {
        let mut summary = Summary::default();
        for (entity, state) in self.histories() {
            summary.include_entity(entity, state, check)?;
        }

        for row in self.pending.rows() {
            check()?;
            summary.entity_heap = summary
                .entity_heap
                .max(key_heap(row.identity.entity.iter().flatten(), check)?);
        }
        self.row_summary(&mut summary, check)?;
        Ok(summary)
    }
    fn row_summary(&self, summary: &mut Summary, check: Check<'_>) -> Result<()> {
        for row in self.rows() {
            check()?;
            summary.row_width = summary.row_width.max(row.len());
            summary.row_heap = summary.row_heap.max(row_heap(row, check)?);
            summary.sequence_heap = summary.sequence_heap.max(self.sequence_heap(row, check)?);
        }
        Ok(())
    }
    fn sequence_heap(&self, row: &[ScalarValue], check: Check<'_>) -> Result<u64> {
        self.compiled
            .sequence_columns
            .iter()
            .try_fold(0, |sum, column| {
                let value = row.get(column.index).ok_or_else(invalid)?;
                add(sum, scalar_heap(value, check)?)
            })
    }
    pub fn column_heap(&self, index: usize, check: Check<'_>) -> Result<u64> {
        self.rows().try_fold(0, |maximum, row| {
            check()?;
            let value = row.get(index).ok_or_else(invalid)?;
            Ok(maximum.max(scalar_heap(value, check)?))
        })
    }
    fn rows(&self) -> impl Iterator<Item = &[ScalarValue]> {
        self.pending.rows().map(|row| row.values.as_slice()).chain(
            self.histories()
                .flat_map(|(_, entity)| entity.rows.iter().map(Vec::as_slice)),
        )
    }
}
impl Summary {
    fn include_entity(
        &mut self,
        entity: &[Option<KeyValue>],
        state: &EntityRollingState,
        check: Check<'_>,
    ) -> Result<()> {
        check()?;
        self.history_rows = self
            .history_rows
            .checked_add(state.rows.len())
            .ok_or_else(invalid)?;
        self.entity_heap = self
            .entity_heap
            .max(key_heap(entity.iter().flatten(), check)?);
        self.prior_copies = add(self.prior_copies, entity_copy(state, check)?)?;
        Ok(())
    }
}
fn entity_copy(state: &EntityRollingState, check: Check<'_>) -> Result<u64> {
    let rows = state.rows.iter().try_fold(0, |sum, row| {
        check()?;
        add(
            sum,
            total([slots::<ScalarValue>(row.len()), row_heap(row, check)])?,
        )
    });
    let windows = state
        .windows
        .iter()
        .try_fold(0, |sum, window| add(sum, extrema_copy(window, check)?));
    total([
        slots::<Vec<ScalarValue>>(state.rows.len()),
        growing::<WindowState>(state.windows.len()),
        growing::<super::super::State>(state.indicators.len()),
        rows,
        windows,
    ])
}
fn extrema_copy(window: &WindowState, check: Check<'_>) -> Result<u64> {
    let WindowState::Extrema(state) = window else {
        return Ok(0);
    };
    let values = state.queue.iter().try_fold(0, |sum, (key, value)| {
        check()?;
        add(
            sum,
            total([
                slots::<KeyValue>(key.sequence.len()),
                key_heap(key.sequence.iter(), check),
                scalar_heap(value, check),
            ])?,
        )
    });
    total([
        slots::<(super::super::super::ExtremaKey, ScalarValue)>(state.queue.len()),
        values,
    ])
}
pub(super) fn row_heap(row: &[ScalarValue], check: Check<'_>) -> Result<u64> {
    row.iter()
        .try_fold(0, |sum, value| add(sum, scalar_heap(value, check)?))
}
fn key_heap<'a>(mut keys: impl Iterator<Item = &'a KeyValue>, check: Check<'_>) -> Result<u64> {
    keys.try_fold(0, |sum, key| {
        check()?;
        let bytes = match key {
            KeyValue::String(value) => size(value.len())?,
            _ => 0,
        };
        add(sum, bytes)
    })
}
