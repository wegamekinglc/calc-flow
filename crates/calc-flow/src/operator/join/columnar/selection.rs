use super::SelectedRow;
use crate::operator::join::{StreamJoinOperator, admitted::AdmittedGroup, operator_error};
use crate::{EventTime, Result};
use datafusion::execution::memory_pool::MemoryReservation;

pub(in crate::operator::join) struct OwnedSelection {
    sources: Vec<usize>,
    ids: Vec<u64>,
    times: Vec<EventTime>,
    _credit: MemoryReservation,
}

impl OwnedSelection {
    pub(in crate::operator::join) async fn from_group(
        operator: &mut StreamJoinOperator,
        group: AdmittedGroup<'_>,
        context: &crate::StreamOperatorContext<'_>,
        quantum: &mut super::Quantum,
    ) -> Result<Self> {
        let count = group.len();
        let credit = selection_credit(operator, count)?;
        let mut value = Self {
            sources: Vec::with_capacity(count),
            ids: Vec::with_capacity(count),
            times: Vec::with_capacity(count),
            _credit: credit,
        };
        for row in group.iter() {
            quantum.step(context, 8, 24).await?;
            value.sources.push(row.record.offset());
            value.ids.push(row.row_id);
            value.times.push(row.event_time);
        }
        Ok(value)
    }

    pub(in crate::operator::join) fn rows(&self) -> SelectedRows<'_> {
        SelectedRows::Columns(self)
    }
}

fn selection_credit(operator: &mut StreamJoinOperator, count: usize) -> Result<MemoryReservation> {
    operator
        .optional_credit(selection_bytes(count)?)?
        .ok_or_else(|| operator_error(&operator.name, "retained selection credit admission failed"))
}

fn selection_bytes(count: usize) -> Result<usize> {
    let bytes = count
        .checked_mul(size_of::<usize>() + size_of::<u64>() + size_of::<EventTime>())
        .ok_or_else(|| super::super::charge_overflow("retained selection"))?;
    let control = super::reservation_control()
        .ok_or_else(|| super::super::charge_overflow("retained selection"))?;
    bytes
        .checked_add(control)
        .ok_or_else(|| super::super::charge_overflow("retained selection"))
}

#[derive(Clone, Copy)]
pub(in crate::operator::join) enum SelectedRows<'a> {
    Slice(&'a [SelectedRow]),
    Admitted(AdmittedGroup<'a>),
    Columns(&'a OwnedSelection),
}

impl SelectedRows<'_> {
    pub(in crate::operator::join) fn len(&self) -> usize {
        match self {
            Self::Slice(rows) => rows.len(),
            Self::Admitted(rows) => rows.len(),
            Self::Columns(rows) => rows.sources.len(),
        }
    }
    pub(in crate::operator::join) fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(in crate::operator::join) fn at(&self, index: usize) -> SelectedRow {
        match self {
            Self::Slice(rows) => rows[index],
            Self::Admitted(rows) => {
                let row = rows.row(index);
                SelectedRow {
                    source: row.record.offset(),
                    row_id: row.row_id,
                    time: row.event_time,
                }
            }
            Self::Columns(rows) => SelectedRow {
                source: rows.sources[index],
                row_id: rows.ids[index],
                time: rows.times[index],
            },
        }
    }
    pub(in crate::operator::join) fn iter(
        &self,
    ) -> impl ExactSizeIterator<Item = SelectedRow> + '_ {
        (0..self.len()).map(|index| self.at(index))
    }
}

impl<'a> From<&'a [SelectedRow]> for SelectedRows<'a> {
    fn from(rows: &'a [SelectedRow]) -> Self {
        Self::Slice(rows)
    }
}
impl<'a> From<&'a Vec<SelectedRow>> for SelectedRows<'a> {
    fn from(rows: &'a Vec<SelectedRow>) -> Self {
        Self::Slice(rows)
    }
}
impl<'a, const N: usize> From<&'a [SelectedRow; N]> for SelectedRows<'a> {
    fn from(rows: &'a [SelectedRow; N]) -> Self {
        Self::Slice(rows)
    }
}
