use super::{
    checked, reason,
    state::{BatchKey, PayloadView},
    workspace::{ColumnWorkspace, OutputColumns},
};
use crate::{Result, StreamingFailureReason};
use ahash::RandomState;
use datafusion::{
    arrow::{array::ArrayRef, datatypes::Schema, record_batch::RecordBatch},
    execution::memory_pool::MemoryReservation,
};
use std::{collections::HashMap, sync::Arc};

mod backing;

#[derive(Clone, Copy)]
pub(super) struct Span {
    pub source: usize,
    pub start: usize,
    pub end: usize,
}

pub(super) struct OutputSide {
    pub batches: Vec<Vec<ArrayRef>>,
    pub positions: Vec<(usize, usize)>,
    pub spans: Vec<Span>,
    pub has_nulls: bool,
}

pub(super) struct OutputPlan {
    pub left: OutputSide,
    pub right: OutputSide,
    pub len: usize,
    pub matched: u64,
    #[cfg(test)]
    pub raw_bytes: u64,
}

pub(super) struct OutputPlanBuilder<'a> {
    left: SideBuilder<'a>,
    right: SideBuilder<'a>,
    len: usize,
    matched: u64,
}

struct SideBuilder<'a> {
    rows: OutputSide,
    columns: Vec<OutputColumns>,
    sources: HashMap<BatchKey, usize, RandomState>,
    selected: Option<&'a [usize]>,
    recent: Option<(BatchKey, usize)>,
    span: Option<Span>,
    run: Option<SourceRun>,
    raw: u64,
    right: bool,
}

struct SourceRun {
    source: usize,
    start: usize,
    end: usize,
    count: usize,
}

impl<'a> OutputPlanBuilder<'a> {
    pub fn new(
        capacity: usize,
        selected: Option<&'a [Vec<usize>; 2]>,
        workspace: &mut MemoryReservation,
        name: &str,
    ) -> Result<Self> {
        let rows = (capacity as u64)
            .checked_mul(256)
            .ok_or_else(|| overflow(name))?;
        grow_workspace(workspace, checked(name, rows, 16 * 1024)?, name)?;
        Ok(Self {
            left: SideBuilder::new(
                capacity,
                selected.map(|columns| columns[0].as_slice()),
                false,
            ),
            right: SideBuilder::new(
                capacity,
                selected.map(|columns| columns[1].as_slice()),
                true,
            ),
            len: 0,
            matched: 0,
        })
    }

    pub fn left_source(
        &mut self,
        left: PayloadView<'_>,
        workspace: &mut MemoryReservation,
        name: &str,
    ) -> Result<usize> {
        self.left.source(left, workspace, name)
    }

    pub fn push_reference(
        &mut self,
        source: usize,
        row: usize,
        right: Option<PayloadView<'_>>,
        workspace: &mut MemoryReservation,
        name: &str,
    ) -> Result<()> {
        self.left.record_span(source, row, name)?;
        if let Some(right) = right {
            self.right.push(right, workspace, name)?;
            self.matched += 1;
        } else {
            self.right.rows.positions.push((0, 0));
            self.right.rows.has_nulls = true;
        }
        self.len += 1;
        Ok(())
    }

    #[cfg(test)]
    pub fn push(
        &mut self,
        left: PayloadView<'_>,
        right: Option<PayloadView<'_>>,
        workspace: &mut MemoryReservation,
        name: &str,
    ) -> Result<()> {
        let source = self.left_source(left, workspace, name)?;
        self.push_reference(source, left.row, right, workspace, name)
    }

    pub fn finish(
        mut self,
        right_schema: &Schema,
        workspace: &mut MemoryReservation,
        name: &str,
    ) -> Result<OutputPlan> {
        self.left.flush_span(name)?;
        self.left.flush_run(name)?;
        self.right.flush_run(name)?;
        self.reserve_buffers(right_schema, workspace, name)?;
        Ok(OutputPlan {
            left: self.left.rows,
            right: self.right.rows,
            len: self.len,
            matched: self.matched,
            #[cfg(test)]
            raw_bytes: self.left.raw + self.right.raw,
        })
    }
    fn reserve_buffers(
        &self,
        schema: &Schema,
        workspace: &mut MemoryReservation,
        name: &str,
    ) -> Result<()> {
        self.reserve_null_source(schema, workspace, name)?;
        let raw = checked(name, self.left.raw, self.right.raw)?;
        let null = self.null_bytes(schema, name)?;
        let buffers = output_buffer_bytes(raw, null, name)?;
        grow_workspace(workspace, buffers, name)
    }

    fn null_bytes(&self, schema: &Schema, name: &str) -> Result<u64> {
        null_row_bytes(schema, self.right.selected, name)?
            .checked_mul(self.len as u64 - self.matched)
            .ok_or_else(|| overflow(name))
    }

    fn reserve_null_source(
        &self,
        schema: &Schema,
        workspace: &mut MemoryReservation,
        name: &str,
    ) -> Result<()> {
        let fields = self
            .right
            .selected
            .map_or(schema.fields().len(), <[usize]>::len);
        let bytes = (fields as u64)
            .checked_mul(512)
            .ok_or_else(|| overflow(name))?;
        grow_workspace(workspace, bytes, name)
    }
}

impl<'a> SideBuilder<'a> {
    fn new(capacity: usize, selected: Option<&'a [usize]>, right: bool) -> Self {
        Self {
            rows: OutputSide {
                batches: Vec::new(),
                positions: if right {
                    Vec::with_capacity(capacity)
                } else {
                    Vec::new()
                },
                has_nulls: false,
                spans: Vec::new(),
            },
            columns: Vec::new(),
            sources: HashMap::with_hasher(RandomState::new()),
            selected,
            recent: None,
            span: None,
            run: None,
            raw: 0,
            right,
        }
    }

    fn source(
        &mut self,
        row: PayloadView<'_>,
        workspace: &mut MemoryReservation,
        name: &str,
    ) -> Result<usize> {
        #[cfg(test)]
        if !self.right {
            super::workspace::record_left_output_source_visit();
        }
        if let Some((key, source)) = self.recent
            && key == row.batch.key
        {
            return Ok(source);
        }
        let source = if let Some(&source) = self.sources.get(&row.batch.key) {
            source
        } else {
            self.register(row, workspace, name)?
        };
        self.recent = Some((row.batch.key, source));
        Ok(source)
    }

    fn register(
        &mut self,
        row: PayloadView<'_>,
        workspace: &mut MemoryReservation,
        name: &str,
    ) -> Result<usize> {
        grow_workspace(workspace, self.source_credit(row, workspace, name)?, name)?;
        let descriptors = selected_columns(&row.batch.record, self.selected)
            .map(ColumnWorkspace::new)
            .collect::<Result<Vec<_>>>()?;
        let descriptors = OutputColumns::new(descriptors, name)?;
        let source = self.rows.batches.len();
        self.rows.batches.push(row.batch.record.columns().to_vec());
        self.columns.push(descriptors);
        self.sources.insert(row.batch.key, source);
        #[cfg(test)]
        super::workspace::record_output_source_registration();
        Ok(source)
    }

    fn source_credit(
        &self,
        row: PayloadView<'_>,
        workspace: &MemoryReservation,
        name: &str,
    ) -> Result<u64> {
        let fields = self
            .selected
            .map_or(row.batch.record.num_columns(), <[usize]>::len);
        let metadata = (fields as u64)
            .checked_mul(512)
            .and_then(|bytes| bytes.checked_add(512))
            .ok_or_else(|| overflow(name))?;
        let (types, heads) = source_headers(&row.batch.record, name)?;
        let backing = checked(
            name,
            backing::source_bytes(&row.batch.record, workspace, name)?,
            types,
        )?;
        checked(name, metadata, checked(name, heads, backing)?)
    }

    fn push(
        &mut self,
        row: PayloadView<'_>,
        workspace: &mut MemoryReservation,
        name: &str,
    ) -> Result<()> {
        let source = self.source(row, workspace, name)?;
        let end = row.row + 1;
        self.record_range(source, row.row..end, name)?;
        self.rows
            .positions
            .push((source + usize::from(self.right), row.row));
        Ok(())
    }

    fn record_span(&mut self, source: usize, row: usize, name: &str) -> Result<()> {
        if let Some(span) = self.span.as_mut()
            && span.source == source
            && span.end == row
        {
            span.end += 1;
        } else {
            self.flush_span(name)?;
            self.span = Some(Span {
                source,
                start: row,
                end: row + 1,
            });
        }
        Ok(())
    }

    fn flush_span(&mut self, name: &str) -> Result<()> {
        if let Some(span) = self.span.take() {
            self.record_range(span.source, span.start..span.end, name)?;
            self.rows.spans.push(span);
        }
        Ok(())
    }

    fn record_range(
        &mut self,
        source: usize,
        range: std::ops::Range<usize>,
        name: &str,
    ) -> Result<()> {
        #[cfg(test)]
        if !self.right {
            super::workspace::record_left_output_range_visit();
        }
        if let Some(run) = self.run.as_mut()
            && run.source == source
            && (self.columns[source].is_fixed_width() || run.end == range.start)
        {
            run.end = range.end;
            run.count += range.len();
            return Ok(());
        }
        self.flush_run(name)?;
        self.run = Some(SourceRun {
            source,
            start: range.start,
            end: range.end,
            count: range.len(),
        });
        Ok(())
    }

    fn flush_run(&mut self, name: &str) -> Result<()> {
        if let Some(run) = self.run.take() {
            let columns = &self.columns[run.source];
            let bytes = if columns.is_fixed_width() {
                columns.fixed_bytes(run.count, name)?
            } else {
                columns.range_bytes(run.start..run.end, name)?
            };
            self.raw = checked(name, self.raw, bytes)?;
        }
        Ok(())
    }
}

fn output_buffer_bytes(raw: u64, null: u64, name: &str) -> Result<u64> {
    checked(name, raw, null)?
        .checked_mul(4)
        .ok_or_else(|| overflow(name))
}

fn source_headers(record: &RecordBatch, name: &str) -> Result<(u64, u64)> {
    let columns = record.columns();
    let types = columns.iter().try_fold(0, |total, column| {
        checked(name, total, column.data_type().size() as u64)
    })?;
    let heads = (columns.len() as u64)
        .checked_mul(256)
        .ok_or_else(|| overflow(name))?;
    Ok((types, heads))
}

pub(super) fn selected_columns<'a>(
    batch: &'a RecordBatch,
    selected: Option<&'a [usize]>,
) -> impl Iterator<Item = ArrayRef> + 'a {
    (0..selected.map_or(batch.num_columns(), <[usize]>::len))
        .map(move |position| selected.map_or(position, |indices| indices[position]))
        .map(move |index| Arc::clone(batch.column(index)))
}

pub(super) fn null_row_bytes(
    schema: &Schema,
    selected: Option<&[usize]>,
    name: &str,
) -> Result<u64> {
    let mut indices = (0..selected.map_or(schema.fields().len(), <[usize]>::len))
        .map(|position| selected.map_or(position, |indices| indices[position]));
    indices.try_fold(0, |total, index| {
        use datafusion::arrow::datatypes::DataType;
        let bytes = match schema.field(index).data_type() {
            DataType::Null => 0,
            DataType::Boolean => 1,
            DataType::Utf8 | DataType::Binary => 4,
            DataType::LargeUtf8 | DataType::LargeBinary => 8,
            DataType::FixedSizeBinary(width) => {
                u64::try_from(*width).expect("validated ASOF fixed binary width")
            }
            data_type => data_type
                .primitive_width()
                .expect("validated flat ASOF type") as u64,
        };
        checked(name, total, checked(name, bytes, 1)?)
    })
}

pub(super) fn grow_workspace(
    workspace: &mut MemoryReservation,
    bytes: u64,
    name: &str,
) -> Result<()> {
    let bytes = usize::try_from(bytes).map_err(|_| {
        reason(
            name,
            StreamingFailureReason::AsofWorkspaceLimitExceeded,
            "ASOF output workspace exceeds address domain",
        )
    })?;
    workspace.try_grow(bytes).map_err(|_| {
        reason(
            name,
            StreamingFailureReason::AsofWorkspaceLimitExceeded,
            "ASOF output workspace exceeds max_state_bytes",
        )
    })
}

fn overflow(name: &str) -> crate::CalcFlowError {
    reason(
        name,
        StreamingFailureReason::AsofCounterOverflow,
        "ASOF output workspace arithmetic overflowed",
    )
}

#[cfg(test)]
mod tests;
