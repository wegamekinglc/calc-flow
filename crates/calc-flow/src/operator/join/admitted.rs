use super::{StreamJoinOperator, columnar::RowPayload, operator_error};
use crate::{Batch, EventTime, Result, batch::TableBatch};
use datafusion::arrow::{array::ArrayRef, datatypes::SchemaRef, record_batch::RecordBatch};
use datafusion::execution::memory_pool::MemoryReservation;
use std::{ops::Range, sync::Arc};

#[cfg(test)]
thread_local! {
    static SPARSE_PROMOTION_OBSERVER: std::cell::RefCell<Option<Arc<tokio::sync::Notify>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(super) struct PromotionObservation(Option<Arc<tokio::sync::Notify>>);
#[cfg(test)]
impl Drop for PromotionObservation {
    fn drop(&mut self) {
        SPARSE_PROMOTION_OBSERVER.with(|observer| {
            observer.replace(self.0.take());
        });
    }
}
#[cfg(test)]
pub(super) fn observe_sparse_promotion(observer: Arc<tokio::sync::Notify>) -> PromotionObservation {
    PromotionObservation(SPARSE_PROMOTION_OBSERVER.with(|current| current.replace(Some(observer))))
}

const INLINE_ROWS: usize = 64;
const INLINE_RECORDS: usize = 4;

mod metadata;
use metadata::{RetainBits, SparseIndices, Values};

struct Segment {
    record: usize,
    source_base: usize,
    row_id_base: u64,
    range: Range<usize>,
    first: usize,
    sparse_start: Option<usize>,
    payload: Option<RowPayload>,
    payload_base: Option<usize>,
}

impl Segment {
    fn new(record: usize, source_base: usize) -> Self {
        Self {
            record,
            source_base,
            row_id_base: 0,
            range: 0..0,
            first: 0,
            sparse_start: None,
            payload: None,
            payload_base: None,
        }
    }
}

struct Segments {
    inline: [Option<Segment>; INLINE_RECORDS],
    heap: Option<Vec<Segment>>,
    len: usize,
}
impl Segments {
    fn new(capacity: usize) -> Self {
        Self {
            inline: std::array::from_fn(|_| None),
            heap: (capacity > INLINE_RECORDS).then(|| Vec::with_capacity(capacity)),
            len: 0,
        }
    }
    fn push(&mut self, value: Segment) {
        if let Some(values) = &mut self.heap {
            assert!(values.len() < values.capacity(), "prepaid segment capacity");
            values.push(value);
        } else {
            self.inline[self.len] = Some(value);
        }
        self.len += 1;
    }
    fn len(&self) -> usize {
        self.len
    }
    fn get(&self, index: usize) -> &Segment {
        self.heap.as_ref().map_or_else(
            || self.inline[index].as_ref().expect("record segment"),
            |values| &values[index],
        )
    }
    fn get_mut(&mut self, index: usize) -> &mut Segment {
        self.heap.as_mut().map_or_else(
            || self.inline[index].as_mut().expect("record segment"),
            |values| &mut values[index],
        )
    }
}

pub(super) struct AdmittedBatch {
    table: Option<TableBatch>,
    segments: Segments,
    indices: SparseIndices,
    times: Values<i64, INLINE_ROWS>,
    retained: RetainBits,
    len: usize,
    current: usize,
    total_records: usize,
    workspace: Option<MemoryReservation>,
    parents: Option<MemoryReservation>,
}

impl Default for AdmittedBatch {
    fn default() -> Self {
        Self {
            table: None,
            segments: Segments::new(0),
            indices: SparseIndices::new(0, true),
            times: Values::new(0, true),
            retained: RetainBits::new(0, true),
            len: 0,
            current: 0,
            total_records: 0,
            workspace: None,
            parents: None,
        }
    }
}

impl AdmittedBatch {
    pub(super) fn reserve(operator: &mut StreamJoinOperator, batch: &Batch) -> Result<Self> {
        let table = batch.table_payload()?;
        let rows = batch.num_rows();
        let records = table.batches().len();
        let inline = rows <= INLINE_ROWS && records <= INLINE_RECORDS;
        let workspace = reserve_metadata(operator, rows, records, inline)?;
        let value = Self {
            table: Some(table.clone()),
            segments: Segments::new(records),
            indices: SparseIndices::new(rows, inline),
            times: Values::new(rows, inline),
            retained: RetainBits::new(rows, inline),
            workspace,
            total_records: records,
            ..Self::default()
        };
        Ok(value)
    }

    #[cfg(test)]
    pub(super) fn sparse_index_capacity(&self) -> usize {
        self.indices.capacity()
    }
    #[cfg(test)]
    pub(super) fn sparse_index_len(&self) -> usize {
        self.indices.len()
    }
    #[cfg(test)]
    pub(super) fn retention_metadata_bytes(&self) -> usize {
        self.retained.allocated_bytes()
    }
    #[cfg(test)]
    pub(super) fn workspace_bytes(&self) -> usize {
        self.workspace.as_ref().map_or(0, MemoryReservation::size)
    }

    pub(super) fn len(&self) -> usize {
        self.len
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(super) fn iter(&self) -> RowIter<'_> {
        RowIter::new(self, 0..self.len)
    }

    pub(super) fn begin_record(&mut self, row_id: u64, source_base: usize) -> usize {
        let index = self.current;
        let mut segment = Segment::new(index, source_base);
        segment.row_id_base = row_id;
        segment.range = self.len..self.len;
        self.segments.push(segment);
        self.current += 1;
        index
    }

    pub(super) async fn ensure_selection(
        &mut self,
        index: usize,
        physical: usize,
        context: &crate::StreamOperatorContext<'_>,
        quantum: &mut super::columnar::Quantum,
    ) -> Result<()> {
        debug_assert!(
            matches!(self.times, Values::Inline { .. }) || self.workspace.is_some(),
            "prepaid admission metadata"
        );
        let segment = self.segments.get(index);
        if segment.range.is_empty()
            || segment.sparse_start.is_some()
            || physical == segment.first + segment.range.len()
        {
            return Ok(());
        }
        let start = self
            .indices
            .promote(segment.first, segment.range.len(), context, quantum)
            .await?;
        self.segments.get_mut(index).sparse_start = Some(start);
        Ok(())
    }

    pub(super) fn push_ready(
        &mut self,
        index: usize,
        physical: usize,
        time: EventTime,
        retain: bool,
    ) {
        let segment = self.segments.get_mut(index);
        if segment.range.is_empty() {
            segment.first = physical;
        }
        if segment.sparse_start.is_some() {
            self.indices.push(physical);
        } else {
            assert_eq!(
                physical,
                segment.first + segment.range.len(),
                "prepared dense selection"
            );
        }
        segment.range.end += 1;
        self.times.push(time.as_micros());
        self.retained.push(retain);
        self.len += 1;
    }

    fn segment_index(&self, position: usize) -> usize {
        assert!(position < self.len);
        let mut start = 0;
        let mut end = self.current;
        while start < end {
            #[cfg(test)]
            super::note_join_work(|work| work.admission_segment_lookups += 1);
            let middle = start + (end - start) / 2;
            if self.segments.get(middle).range.end <= position {
                start = middle + 1;
            } else {
                end = middle;
            }
        }
        start
    }

    fn physical(&self, segment: &Segment, ordinal: usize) -> usize {
        segment.sparse_start.map_or_else(
            || segment.first + ordinal - segment.range.start,
            |start| self.indices.get(start + ordinal - segment.range.start),
        )
    }

    pub(super) fn row(&self, position: usize) -> AdmittedRef<'_> {
        let segment = self.segments.get(self.segment_index(position));
        let physical = self.physical(segment, position);
        let offset = segment
            .payload_base
            .map_or(physical, |base| base + position - segment.range.start);
        let record = match &segment.payload {
            Some(payload) => PayloadRef::Payload(payload, offset),
            None => PayloadRef::Record(
                &self
                    .table
                    .as_ref()
                    .expect("borrowed admission table")
                    .batches()[segment.record],
                offset,
            ),
        };
        AdmittedRef {
            record,
            row_id: segment.row_id_base + physical as u64,
            event_time: EventTime::from_micros(self.times.get(position)),
            retain: self.retained.get(position),
        }
    }

    pub(super) fn group_range(&self, index: usize) -> Range<usize> {
        self.segments.get(index).range.clone()
    }

    pub(super) fn record_count(&self) -> usize {
        self.segments.len()
    }

    pub(super) fn replace(&mut self, index: usize, payload: RowPayload) {
        let segment = self.segments.get_mut(index);
        segment.payload_base = Some(payload.offset());
        segment.payload = Some(payload);
        self.release_table();
    }

    /// Installs a zero-copy projected parent indexed by physical row. The
    /// group keeps no retained row, so nothing enters state from it and the
    /// projected columns can share the incoming buffers.
    pub(super) fn replace_view(&mut self, index: usize, parent: Arc<RecordBatch>) {
        let segment = self.segments.get_mut(index);
        segment.payload_base = None;
        segment.payload = Some(RowPayload::Rowed { parent, row: 0 });
        self.release_table();
    }

    #[cfg(test)]
    pub(super) fn projection_is_view(&self, index: usize) -> bool {
        let segment = self.segments.get(index);
        segment.payload_base.is_none()
            && matches!(segment.payload.as_ref(), Some(RowPayload::Rowed { .. }))
    }

    pub(super) fn complete_record(&mut self) {
        self.release_table();
    }

    fn release_table(&mut self) {
        if self.current < self.total_records {
            return;
        }
        let borrowed = (0..self.segments.len()).any(|index| {
            let segment = self.segments.get(index);
            !segment.range.is_empty() && segment.payload.is_none()
        });
        if !borrowed {
            self.table = None;
        }
    }

    pub(super) fn prepare_retained_parents(&mut self) {
        for index in 0..self.segments.len() {
            let segment = self.segments.get_mut(index);
            let retained = self.retained.any(segment.range.clone());
            if !retained || segment.payload.is_some() {
                continue;
            }
            let record = &self
                .table
                .as_ref()
                .expect("temporary admission table")
                .batches()[segment.record];
            segment.payload = Some(RowPayload::Rowed {
                parent: Arc::new(record.clone()),
                row: 0,
            });
        }
        self.release_table();
    }

    fn parent_bytes(&self) -> Result<usize> {
        let mut bytes = 0_usize;
        for index in 0..self.segments.len() {
            let segment = self.segments.get(index);
            if segment.range.is_empty() || segment.payload.is_some() {
                continue;
            }
            let record = &self.table.as_ref().expect("temporary table").batches()[segment.record];
            bytes = bytes
                .checked_add(record_parent_bytes(record)?)
                .ok_or_else(|| super::charge_overflow("Join parents"))?;
        }
        if bytes == 0 {
            return Ok(0);
        }
        parent_workspace_bytes(bytes)
    }

    pub(super) fn detach(&mut self, operator: &mut StreamJoinOperator) -> Result<bool> {
        let bytes = self.parent_bytes()?;
        if bytes == 0 {
            self.release_table();
            return Ok(true);
        }
        let Some(credit) = operator.optional_credit(bytes)? else {
            return Ok(false);
        };
        for index in 0..self.segments.len() {
            let segment = self.segments.get_mut(index);
            if segment.range.is_empty() || segment.payload.is_some() {
                continue;
            }
            let record = &self.table.as_ref().expect("temporary table").batches()[segment.record];
            segment.payload = Some(RowPayload::Rowed {
                parent: Arc::new(record.clone()),
                row: 0,
            });
        }
        self.parents = Some(credit);
        self.release_table();
        Ok(true)
    }
}

fn record_parent_bytes(record: &RecordBatch) -> Result<usize> {
    record
        .num_columns()
        .checked_mul(size_of::<ArrayRef>())
        .and_then(|columns| columns.checked_add(size_of::<RecordBatch>() + 2 * size_of::<usize>()))
        .ok_or_else(|| super::charge_overflow("Join parents"))
}

fn parent_workspace_bytes(bytes: usize) -> Result<usize> {
    let control = super::columnar::reservation_control()
        .ok_or_else(|| super::charge_overflow("Join parents"))?;
    bytes
        .checked_add(control)
        .ok_or_else(|| super::charge_overflow("Join parents"))
}

fn reserve_metadata(
    operator: &mut StreamJoinOperator,
    rows: usize,
    records: usize,
    inline: bool,
) -> Result<Option<MemoryReservation>> {
    if inline {
        return Ok(None);
    }
    let bytes = workspace_bytes(rows, records)
        .ok_or_else(|| operator_error(&operator.name, "Join admission workspace size overflow"))?;
    operator.optional_credit(bytes)?.map(Some).ok_or_else(|| {
        operator_error(
            &operator.name,
            "Join admission workspace credit admission failed",
        )
    })
}

fn workspace_bytes(rows: usize, records: usize) -> Option<usize> {
    let segments = if records > INLINE_RECORDS {
        records.checked_mul(size_of::<Segment>())?
    } else {
        0
    };
    metadata::metadata_bytes(rows)?
        .checked_add(segments)?
        .checked_add(super::columnar::reservation_control()?)
}

#[derive(Clone, Copy)]
pub(super) enum PayloadRef<'a> {
    Record(&'a RecordBatch, usize),
    Payload(&'a RowPayload, usize),
}

impl<'a> PayloadRef<'a> {
    pub(super) fn columns(self) -> &'a [ArrayRef] {
        match self {
            Self::Record(record, _) => record.columns(),
            Self::Payload(payload, _) => payload.columns(),
        }
    }
    pub(super) fn shared_columns(self) -> Option<&'a [ArrayRef]> {
        match self {
            Self::Record(record, _) => Some(record.columns()),
            Self::Payload(payload, _) => payload.shared_columns(),
        }
    }
    pub(super) fn schema_ref(self) -> &'a SchemaRef {
        match self {
            Self::Record(record, _) => record.schema_ref(),
            Self::Payload(payload, _) => payload.schema_ref(),
        }
    }
    pub(super) fn offset(self) -> usize {
        match self {
            Self::Record(_, row) | Self::Payload(_, row) => row,
        }
    }
    pub(super) fn column(self, index: usize) -> &'a ArrayRef {
        &self.columns()[index]
    }
    pub(super) fn num_columns(self) -> usize {
        self.columns().len()
    }
    #[cfg(test)]
    pub(super) fn view(self) -> RecordBatch {
        #[cfg(test)]
        super::note_join_work(|work| work.output_row_batch_views += 1);
        RecordBatch::try_new_with_options(
            Arc::clone(self.schema_ref()),
            self.columns()
                .iter()
                .map(|column| column.slice(self.offset(), 1))
                .collect(),
            &datafusion::arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(1)),
        )
        .expect("validated payload position")
    }
    pub(super) fn owned(self) -> RowPayload {
        self.owner()
            .expect("prepared admission parent")
            .with_offset(self.offset())
    }
    pub(super) fn owner(self) -> Option<&'a RowPayload> {
        match self {
            Self::Record(..) => None,
            Self::Payload(payload, _) => Some(payload),
        }
    }
    #[cfg(test)]
    pub(super) fn funded_owner(self) -> Option<(usize, usize)> {
        self.owner().and_then(RowPayload::funded_owner)
    }
    #[cfg(test)]
    pub(super) fn owner_count(self) -> usize {
        match self.owner() {
            Some(RowPayload::Rowed { parent, .. }) => Arc::strong_count(parent),
            Some(RowPayload::Shared { chunk, .. }) => Arc::strong_count(chunk),
            Some(RowPayload::RestoredV2 { chunk, .. }) => Arc::strong_count(chunk),
            _ => 0,
        }
    }
}

impl<'a> From<&'a RowPayload> for PayloadRef<'a> {
    fn from(payload: &'a RowPayload) -> Self {
        Self::Payload(payload, payload.offset())
    }
}

pub(super) struct AdmittedRef<'a> {
    pub(super) record: PayloadRef<'a>,
    pub(super) event_time: EventTime,
    pub(super) row_id: u64,
    pub(super) retain: bool,
}

pub(super) trait AdmittedRows: Sync {
    fn len(&self) -> usize;
    fn row(&self, index: usize) -> AdmittedRef<'_>;
    fn group_end(&self, start: usize) -> usize;
    fn source_index(&self, index: usize) -> usize;
    /// Retained bits for a window of at most 64 rows starting at `start`;
    /// bit `i` marks the row at `start + i`.
    fn retained_bits(&self, start: usize, length: usize) -> u64 {
        debug_assert!(length <= 64);
        (0..length).fold(0, |bits, offset| {
            bits | (u64::from(self.row(start + offset).retain) << offset)
        })
    }
    fn any_retained(&self, range: Range<usize>) -> bool {
        (range.start..range.end).any(|index| self.row(index).retain)
    }
    /// why: key identity reads only (offset, retain), so a group can resolve
    /// its segment once instead of re-walking `row` per row
    fn key_offsets(
        &self,
        range: Range<usize>,
    ) -> Box<dyn Iterator<Item = (usize, bool)> + Send + '_> {
        Box::new(range.map(|index| {
            let row = self.row(index);
            (row.record.offset(), row.retain)
        }))
    }
}

impl AdmittedRows for AdmittedBatch {
    fn len(&self) -> usize {
        self.len
    }
    fn row(&self, index: usize) -> AdmittedRef<'_> {
        self.row(index)
    }
    fn group_end(&self, start: usize) -> usize {
        self.segments.get(self.segment_index(start)).range.end
    }
    fn source_index(&self, index: usize) -> usize {
        let segment = self.segments.get(self.segment_index(index));
        segment.source_base + self.physical(segment, index)
    }
    fn retained_bits(&self, start: usize, length: usize) -> u64 {
        self.retained.bits(start, length)
    }
    fn any_retained(&self, range: Range<usize>) -> bool {
        self.retained.any(range)
    }
    fn key_offsets(
        &self,
        range: Range<usize>,
    ) -> Box<dyn Iterator<Item = (usize, bool)> + Send + '_> {
        if range.is_empty() {
            return Box::new(std::iter::empty());
        }
        let segment = self.segments.get(self.segment_index(range.start));
        let base = segment.range.start;
        let dense = segment.first;
        let sparse = segment.sparse_start;
        let payload = segment.payload_base;
        Box::new(range.map(move |position| {
            let ordinal = position - base;
            let physical = sparse.map_or_else(
                || dense + ordinal,
                |start| self.indices.get(start + ordinal),
            );
            let offset = payload.map_or(physical, |origin| origin + ordinal);
            (offset, self.retained.get(position))
        }))
    }
}

impl dyn AdmittedRows + '_ {
    #[cfg(test)]
    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(super) fn iter(&self) -> RowIter<'_> {
        RowIter::new(self, 0..self.len())
    }
    pub(super) fn groups(&self) -> GroupIter<'_> {
        GroupIter {
            rows: self,
            next: 0,
        }
    }
}

pub(super) struct RowIter<'a> {
    rows: &'a dyn AdmittedRows,
    range: Range<usize>,
}
impl<'a> RowIter<'a> {
    fn new(rows: &'a dyn AdmittedRows, range: Range<usize>) -> Self {
        Self { rows, range }
    }
}
impl<'a> Iterator for RowIter<'a> {
    type Item = AdmittedRef<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|index| self.rows.row(index))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}
impl ExactSizeIterator for RowIter<'_> {}
impl DoubleEndedIterator for RowIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.range.next_back().map(|index| self.rows.row(index))
    }
}
impl<'a> IntoIterator for &'a AdmittedBatch {
    type Item = AdmittedRef<'a>;
    type IntoIter = RowIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a> IntoIterator for &'a dyn AdmittedRows {
    type Item = AdmittedRef<'a>;
    type IntoIter = RowIter<'a>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[derive(Clone, Copy)]
pub(super) struct AdmittedGroup<'a> {
    rows: &'a dyn AdmittedRows,
    start: usize,
    end: usize,
}
impl<'a> AdmittedGroup<'a> {
    pub(super) fn new(rows: &'a dyn AdmittedRows, range: Range<usize>) -> Self {
        Self {
            rows,
            start: range.start,
            end: range.end,
        }
    }
    pub(super) fn len(&self) -> usize {
        self.end - self.start
    }
    pub(super) fn row(&self, index: usize) -> AdmittedRef<'a> {
        self.rows.row(self.start + index)
    }
    pub(super) fn iter(&self) -> RowIter<'a> {
        RowIter::new(self.rows, self.start..self.end)
    }
    pub(super) fn key_offsets(&self) -> Box<dyn Iterator<Item = (usize, bool)> + Send + '_> {
        self.rows.key_offsets(self.start..self.end)
    }
    pub(super) fn block(&self, start: usize, end: usize) -> AdmittedGroup<'a> {
        AdmittedGroup::new(self.rows, self.start + start..self.start + end)
    }
    pub(super) fn retained_bits(&self, start: usize, length: usize) -> u64 {
        self.rows.retained_bits(self.start + start, length)
    }
    pub(super) fn any_retained(&self) -> bool {
        self.rows.any_retained(self.start..self.end)
    }
}
pub(super) struct GroupIter<'a> {
    rows: &'a dyn AdmittedRows,
    next: usize,
}
impl<'a> Iterator for GroupIter<'a> {
    type Item = AdmittedGroup<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.rows.len() {
            return None;
        }
        let end = self.rows.group_end(self.next);
        let group = AdmittedGroup::new(self.rows, self.next..end);
        self.next = end;
        Some(group)
    }
}

#[cfg(test)]
mod fixtures {
    use super::super::AdmittedRow;
    use super::*;
    impl AdmittedRows for Vec<AdmittedRow> {
        fn len(&self) -> usize {
            Vec::len(self)
        }
        fn row(&self, index: usize) -> AdmittedRef<'_> {
            let row = &self[index];
            AdmittedRef {
                record: (&row.record).into(),
                event_time: row.event_time,
                row_id: row.row_id,
                retain: row.retain,
            }
        }
        fn group_end(&self, start: usize) -> usize {
            let columns = self[start].record.columns();
            (start + 1..self.len())
                .find(|&index| !std::ptr::eq(columns, self[index].record.columns()))
                .unwrap_or(self.len())
        }
        fn source_index(&self, index: usize) -> usize {
            index
        }
    }
    impl<const N: usize> AdmittedRows for [AdmittedRow; N] {
        fn len(&self) -> usize {
            N
        }
        fn row(&self, index: usize) -> AdmittedRef<'_> {
            let row = &self[index];
            AdmittedRef {
                record: (&row.record).into(),
                event_time: row.event_time,
                row_id: row.row_id,
                retain: row.retain,
            }
        }
        fn group_end(&self, start: usize) -> usize {
            start + 1
        }
        fn source_index(&self, index: usize) -> usize {
            index
        }
    }
    impl AdmittedBatch {
        pub(in crate::operator::join) fn push_fixture(&mut self, row: AdmittedRow) {
            let mut rows = self
                .iter()
                .map(|row| AdmittedRow {
                    record: row.record.owned(),
                    event_time: row.event_time,
                    row_id: row.row_id,
                    retain: row.retain,
                })
                .collect::<Vec<_>>();
            rows.push(row);
            *self = rows.into();
        }
    }
    impl From<Vec<AdmittedRow>> for AdmittedBatch {
        fn from(rows: Vec<AdmittedRow>) -> Self {
            let count = rows.len();
            let mut batch = Self {
                segments: Segments::new(count),
                indices: SparseIndices::new(count, count <= INLINE_ROWS),
                times: Values::new(count, count <= INLINE_ROWS),
                retained: RetainBits::new(count, count <= INLINE_ROWS),
                ..Self::default()
            };
            for (index, row) in rows.into_iter().enumerate() {
                let offset = row.record.offset();
                let mut segment = Segment::new(index, index);
                segment.row_id_base = row.row_id;
                segment.range = index..index;
                segment.payload = Some(row.record);
                segment.payload_base = Some(offset);
                batch.segments.push(segment);
                batch.push_ready(index, 0, row.event_time, row.retain);
            }
            batch.current = batch.segments.len();
            batch.total_records = batch.current;
            batch
        }
    }
}
