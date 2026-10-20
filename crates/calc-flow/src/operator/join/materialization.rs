//! Preflight matched-pair ranges before allocating independently owned output chunks.
use crate::operator::join::retained::Rows;

mod emission;

use std::{ops::Range, sync::Arc};

use datafusion::arrow::{array::ArrayRef, datatypes::SchemaRef, record_batch::RecordBatch};

#[cfg(test)]
use super::{AdmittedRow, StoredRow};
use super::{AdmittedRows, MatchedPair, PayloadRef};
use crate::batch::checked_accumulate;
use crate::operator::output_chunk::OutputChunkErrors;
use crate::operator::row_cost::{Offsets, fixed_width, terminal_offset_width, variable_offsets};
use crate::{Batch, BatchMetadata, CalcFlowError, EdgeBudget, Result};

pub(super) struct JoinOutput<'a> {
    pub(super) schema: &'a SchemaRef,
    pub(super) projection: Option<&'a Arc<[usize]>>,
    pub(super) admitted: &'a dyn AdmittedRows,
    pub(super) opposite: &'a dyn Rows,
    pub(super) matched: &'a dyn super::matches::Matches,
    pub(super) incoming_is_left: bool,
    pub(super) operator_id: &'a str,
    /// Conservative per-admitted-row logical byte charges aligned with
    /// `admitted`. `None` when the schema keeps the generic measured path.
    pub(super) admitted_charges: Option<&'a [u64]>,
}

struct FlatRow {
    bytes: usize,
    nulls: Vec<usize>,
    terminal: usize,
}

pub(super) fn supports_flat_ranges(schema: &SchemaRef) -> bool {
    schema
        .fields()
        .iter()
        .all(|field| fixed_width(field.data_type()).is_some())
}

#[derive(Default)]
struct FlatChunk {
    rows: usize,
    bytes: usize,
    nulls: Vec<bool>,
    null_count: usize,
}

impl FlatChunk {
    fn fits(&self, incoming: &FlatRow, opposite: &FlatRow, budget: EdgeBudget) -> bool {
        self.rows < budget.max_rows
            && self
                .bytes_with(incoming, opposite)
                .is_some_and(|bytes| bytes <= budget.max_bytes)
    }

    fn bytes_with(&self, incoming: &FlatRow, opposite: &FlatRow) -> Option<usize> {
        let bitmap_count = self.null_count
            + incoming
                .nulls
                .iter()
                .chain(&opposite.nulls)
                .filter(|index| !self.nulls.get(**index).copied().unwrap_or(false))
                .count();
        incoming
            .bytes
            .checked_add(opposite.bytes)
            .and_then(|bytes| self.bytes.checked_add(bytes))
            .and_then(|bytes| bytes.checked_add(incoming.terminal))
            .and_then(|bytes| bytes.checked_add(opposite.terminal))
            .and_then(|bytes| {
                bitmap_count
                    .checked_mul((self.rows + 1).div_ceil(8))
                    .and_then(|bitmap| bytes.checked_add(bitmap))
            })
    }

    fn push(&mut self, incoming: &FlatRow, opposite: &FlatRow) {
        self.rows += 1;
        self.bytes += incoming.bytes + opposite.bytes;
        for &index in incoming.nulls.iter().chain(&opposite.nulls) {
            if index >= self.nulls.len() {
                self.nulls.resize(index + 1, false);
            }
            if !self.nulls[index] {
                self.nulls[index] = true;
                self.null_count += 1;
            }
        }
    }
}

/// Allocation-free width cache for one immutable source-column run.
#[derive(Default)]
struct RowWidth<'a> {
    bytes: Option<usize>,
    columns: Option<&'a [ArrayRef]>,
}

impl<'a> RowWidth<'a> {
    fn agree(
        &mut self,
        record: PayloadRef<'a>,
        projection: Option<&[(usize, usize)]>,
    ) -> Option<()> {
        let columns = record.columns();
        if self
            .columns
            .is_some_and(|previous| std::ptr::eq(previous, columns))
        {
            return Some(());
        }
        agrees_with(&mut self.bytes, side_row_bytes(record, projection)?)?;
        self.columns = Some(columns);
        Some(())
    }
}

/// Running per-pair scan state for the flat-range walk.
struct Scan {
    ranges: Vec<Range<usize>>,
    start: usize,
    chunk: FlatChunk,
}

impl Scan {
    fn step(
        &mut self,
        index: usize,
        left: &FlatRow,
        right: &FlatRow,
        budget: EdgeBudget,
        oversized: impl Fn(usize) -> CalcFlowError,
    ) -> Result<()> {
        if !self.chunk.fits(left, right, budget) {
            if self.start == index {
                return Err(oversized(
                    self.chunk.bytes_with(left, right).unwrap_or(usize::MAX),
                ));
            }
            self.ranges.push(self.start..index);
            self.start = index;
            self.chunk = FlatChunk::default();
            if !self.chunk.fits(left, right, budget) {
                return Err(oversized(
                    self.chunk.bytes_with(left, right).unwrap_or(usize::MAX),
                ));
            }
        }
        self.chunk.push(left, right);
        Ok(())
    }

    fn finish(&mut self, total: usize) {
        if self.start < total {
            self.ranges.push(self.start..total);
        }
    }
}

/// A dense seen-bit table over a bounded span of opposite coordinates.
struct SeenRows {
    low: usize,
    seen: Vec<u64>,
}

impl SeenRows {
    fn new(low: usize, high: usize) -> Self {
        let bits = u64::BITS as usize;
        Self {
            low,
            seen: vec![0_u64; (high - low + 1).div_ceil(bits)],
        }
    }

    /// Records the coordinate, reporting whether this was its first visit.
    fn first_visit(&mut self, index: usize) -> bool {
        let index = index - self.low;
        let bits = u64::BITS as usize;
        let word = &mut self.seen[index / bits];
        let mask = 1_u64 << (index % bits);
        let first = *word & mask == 0;
        *word |= mask;
        first
    }
}

impl JoinOutput<'_> {
    pub(super) fn materialize(&self, range: Range<usize>) -> Result<RecordBatch> {
        let record = super::output_projection::materialize(self, range)?;
        #[cfg(test)]
        super::note_join_work(|work| {
            work.output_peak_bytes = work.output_peak_bytes.max(record.get_array_memory_size());
        });
        Ok(record)
    }

    pub(super) fn logical_column(&self, ordinal: usize) -> usize {
        self.projection.map_or(ordinal, |columns| columns[ordinal])
    }

    pub(super) fn records(&self, pair: &MatchedPair) -> (PayloadRef<'_>, PayloadRef<'_>) {
        let incoming = self.admitted.payload(pair.pos);
        let opposite = self.opposite.payload(pair.opposite_index);
        if self.incoming_is_left {
            (incoming, opposite)
        } else {
            (opposite, incoming)
        }
    }

    /// Resolve only the selected side, reusing its immutable source/chunk cursor.
    pub(super) fn side_payloads<'a>(
        &'a self,
        matched: &'a dyn super::matches::Matches,
        left: bool,
    ) -> impl ExactSizeIterator<Item = PayloadRef<'a>> {
        let mut admitted = self.admitted.cursor();
        let mut retained = self.opposite.cursor();
        matched.iter().map(move |pair| {
            if left == self.incoming_is_left {
                admitted.payload(pair.pos)
            } else {
                retained.payload(pair.opposite_index)
            }
        })
    }

    fn side_columns(&self, incoming: bool) -> Option<Vec<(usize, usize)>> {
        let projection = self.projection?;
        let left_width = self
            .matched
            .first()
            .map_or(0, |pair| self.records(&pair).0.num_columns());
        let left = incoming == self.incoming_is_left;
        Some(
            projection
                .iter()
                .enumerate()
                .filter(|(_, index)| (**index < left_width) == left)
                .map(|(ordinal, &index)| (if left { index } else { index - left_width }, ordinal))
                .collect(),
        )
    }

    pub(super) fn ranges(&self, budget: EdgeBudget) -> Result<Vec<Range<usize>>> {
        if let Some(charges) = self.admitted_charges
            && self.single_chunk_fits(charges, budget)
        {
            return Ok(std::iter::once(0..self.matched.len()).collect());
        }
        if supports_flat_ranges(self.schema) {
            self.flat_ranges(budget)
        } else {
            self.nested_ranges(budget)
        }
    }

    /// Conservative whole-output fit check using cached logical charges.
    ///
    /// Each per-row charge includes the state-row overhead and a null-bitmap
    /// allowance, so the summed bound never under-estimates the detailed flat
    /// model. Only the row cap and byte sum decide; no per-row schema work.
    fn single_chunk_fits(&self, charges: &[u64], budget: EdgeBudget) -> bool {
        if self.matched.is_empty() || self.matched.len() > budget.max_rows {
            return false;
        }
        let limit = budget.max_bytes as u128;
        let mut total = 0_u128;
        for pair in self.matched.iter() {
            let incoming = charges[pair.pos];
            let opposite = self.opposite.charge(pair.opposite_index);
            total += u128::from(incoming) + u128::from(opposite);
            if total > limit {
                return false;
            }
        }
        true
    }

    fn flat_ranges(&self, budget: EdgeBudget) -> Result<Vec<Range<usize>>> {
        #[cfg(test)]
        if tests::FORCE_SLOW.with(std::cell::Cell::get) {
            return self.per_pair_ranges(budget);
        }
        if let Some(ranges) = self.fixed_width_ranges(budget)? {
            return Ok(ranges);
        }
        self.per_pair_ranges(budget)
    }

    /// Equal-length ranges for selections whose every matched row costs the
    /// same flat bytes, or `None` while the per-pair scan is required.
    ///
    /// Fixed-width columns without variable payloads carry no terminal offset,
    /// and null-free source arrays keep the shared null-bitmap term at zero,
    /// so `FlatChunk::bytes_with` charges exactly
    /// `rows * (incoming_bytes + opposite_bytes)` for such a selection. Under
    /// that linear cost the greedy per-pair scan fills every chunk to
    /// `min(max_rows, max_bytes / row_bytes)` rows, which is all this path
    /// computes.
    fn fixed_width_ranges(&self, budget: EdgeBudget) -> Result<Option<Vec<Range<usize>>>> {
        let total = self.matched.len();
        if total == 0 {
            return Ok(Some(Vec::new()));
        }
        let Some(row_bytes) = self.constant_row_bytes() else {
            return Ok(None);
        };
        let per_chunk = if row_bytes == 0 {
            budget.max_rows
        } else {
            budget.max_bytes / row_bytes
        }
        .min(budget.max_rows);
        if per_chunk == 0 {
            // The per-pair scan rejects this same single row first.
            return Err(oversized_row(row_bytes, budget));
        }
        Ok(Some(
            (0..total)
                .step_by(per_chunk)
                .map(|start| start..(start + per_chunk).min(total))
                .collect(),
        ))
    }

    /// Constant per-pair flat bytes when every matched row costs the same, or
    /// `None` when any selected column varies by row.
    ///
    /// Distinct row coordinates are filtered as before; source-column runs
    /// reuse their immutable width inspection without allocating a source map.
    fn constant_row_bytes(&self) -> Option<usize> {
        let incoming_columns = self.side_columns(true);
        let opposite_columns = self.side_columns(false);
        let (low, high) = self
            .matched
            .iter()
            .fold((usize::MAX, 0), |(low, high), pair| {
                (low.min(pair.opposite_index), high.max(pair.opposite_index))
            });
        if high - low + 1 > self.matched.len().saturating_mul(2) {
            return None;
        }
        let mut seen = SeenRows::new(low, high);
        let mut incoming = RowWidth::default();
        let mut opposite = RowWidth::default();
        let mut previous_incoming = None;
        let mut admitted = self.admitted.cursor();
        let mut retained = self.opposite.cursor();
        for pair in self.matched.iter() {
            if previous_incoming != Some(pair.pos) {
                incoming.agree(admitted.payload(pair.pos), incoming_columns.as_deref())?;
                previous_incoming = Some(pair.pos);
            }
            if seen.first_visit(pair.opposite_index) {
                opposite.agree(
                    retained.payload(pair.opposite_index),
                    opposite_columns.as_deref(),
                )?;
            }
        }
        Some(incoming.bytes? + opposite.bytes?)
    }

    fn per_pair_ranges(&self, budget: EdgeBudget) -> Result<Vec<Range<usize>>> {
        let incoming_columns = self.side_columns(true);
        let opposite_columns = self.side_columns(false);
        let mut incoming = RowCache::new(self.matched.iter().map(|pair| pair.pos));
        let mut opposite = RowCache::new(self.matched.iter().map(|pair| pair.opposite_index));
        let mut scan = Scan {
            ranges: Vec::new(),
            start: 0,
            chunk: FlatChunk::default(),
        };
        let oversized = |bytes: usize| oversized_row(bytes, budget);
        let mut admitted = self.admitted.cursor();
        let mut retained = self.opposite.cursor();
        for (index, pair) in self.matched.iter().enumerate() {
            // Payloads are read only for rows whose costs are not cached yet.
            let mut record = || admitted.payload(pair.pos);
            let left = incoming.get_or_try_insert_with(pair.pos, || {
                flat_row(record(), 0, incoming_columns.as_deref())
            })?;
            let right = opposite.get_or_try_insert_with(pair.opposite_index, || {
                flat_row(
                    retained.payload(pair.opposite_index),
                    record().num_columns(),
                    opposite_columns.as_deref(),
                )
            })?;
            scan.step(index, left, right, budget, oversized)?;
        }
        scan.finish(self.matched.len());
        Ok(scan.ranges)
    }

    fn nested_ranges(&self, budget: EdgeBudget) -> Result<Vec<Range<usize>>> {
        let incoming_columns = self.side_columns(true);
        let opposite_columns = self.side_columns(false);
        let mut admitted = self.admitted.cursor();
        let mut retained = self.opposite.cursor();
        let incoming = estimated_rows(self.matched.iter().map(|pair| pair.pos), |pos| {
            estimated_row(admitted.payload(pos), incoming_columns.as_deref())
        })?;
        let opposite = estimated_rows(
            self.matched.iter().map(|pair| pair.opposite_index),
            |index| estimated_row(retained.payload(index), opposite_columns.as_deref()),
        )?;
        let mut ranges = Vec::new();
        let mut start = 0;
        let mut bytes = 0_usize;
        for (index, pair) in self.matched.iter().enumerate() {
            let row_bytes = estimated_cost(&incoming, pair.pos)
                .saturating_add(estimated_cost(&opposite, pair.opposite_index));
            if nested_range_is_full(start..index, bytes, row_bytes, budget) {
                self.validate_nested_range(start..index, budget, &mut ranges)?;
                start = index;
                bytes = 0;
            }
            bytes = bytes.saturating_add(row_bytes);
        }
        if start < self.matched.len() {
            self.validate_nested_range(start..self.matched.len(), budget, &mut ranges)?;
        }
        Ok(ranges)
    }

    fn validate_nested_range(
        &self,
        range: Range<usize>,
        budget: EdgeBudget,
        ranges: &mut Vec<Range<usize>>,
    ) -> Result<()> {
        // Drop each bounded preflight allocation before validating another range.
        let bytes = estimated(&self.materialize(range.clone())?)?;
        if bytes <= budget.max_bytes {
            ranges.push(range);
        } else if range.len() == 1 {
            return Err(oversized_row(bytes, budget));
        } else {
            let mid = range.start + range.len() / 2;
            self.validate_nested_range(range.start..mid, budget, ranges)?;
            self.validate_nested_range(mid..range.end, budget, ranges)?;
        }
        Ok(())
    }
}

fn nested_range_is_full(
    range: Range<usize>,
    bytes: usize,
    row_bytes: usize,
    budget: EdgeBudget,
) -> bool {
    range.start < range.end
        && (range.end - range.start >= budget.max_rows
            || bytes.saturating_add(row_bytes) > budget.max_bytes)
}

/// Per-preflight row costs indexed by matched row coordinates.
///
/// Retained chunk coordinates can have gaps. A dense table spans the matched
/// coordinate range only while it stays within twice the matched pairs, which
/// the per-match scratch allowance already funds; wider ranges index sorted
/// distinct coordinates instead.
struct RowCache<T> {
    keys: CacheKeys,
    rows: Vec<Option<T>>,
}

enum CacheKeys {
    Offset(usize),
    Sorted(Vec<usize>),
}

impl<T> RowCache<T> {
    fn new(coordinates: impl ExactSizeIterator<Item = usize> + Clone) -> Self {
        let limit = coordinates.len().saturating_mul(2);
        let (low, high) = coordinates
            .clone()
            .fold((usize::MAX, 0), |(low, high), coordinate| {
                (low.min(coordinate), high.max(coordinate))
            });
        let (keys, slots) = if low > high {
            (CacheKeys::Offset(0), 0)
        } else if high - low < limit {
            (CacheKeys::Offset(low), high - low + 1)
        } else {
            let mut keys = coordinates.collect::<Vec<_>>();
            keys.sort_unstable();
            keys.dedup();
            let slots = keys.len();
            (CacheKeys::Sorted(keys), slots)
        };
        Self {
            keys,
            rows: std::iter::repeat_with(|| None).take(slots).collect(),
        }
    }

    fn slot(&self, coordinate: usize) -> usize {
        match &self.keys {
            CacheKeys::Offset(low) => coordinate - low,
            CacheKeys::Sorted(keys) => keys
                .binary_search(&coordinate)
                .expect("row cache covers every matched coordinate"),
        }
    }

    fn get(&self, coordinate: usize) -> Option<&T> {
        self.rows[self.slot(coordinate)].as_ref()
    }

    fn get_or_try_insert_with(
        &mut self,
        coordinate: usize,
        compute: impl FnOnce() -> Result<T>,
    ) -> Result<&T> {
        let slot = self.slot(coordinate);
        match &mut self.rows[slot] {
            Some(row) => Ok(&*row),
            empty => Ok(&*empty.insert(compute()?)),
        }
    }
}

fn flat_row(
    record: PayloadRef<'_>,
    column_offset: usize,
    projection: Option<&[(usize, usize)]>,
) -> Result<FlatRow> {
    #[cfg(test)]
    tests::FLAT_VISITS.with(|count| count.set(count.get() + 1));
    let mut bytes = 0;
    let mut nulls = Vec::new();
    let mut terminal = 0;
    let row = record.offset();
    for (column_index, column) in row_columns(record, column_offset, projection) {
        let width = fixed_width(column.data_type()).expect("flat schema was checked");
        let offset = terminal_offset_width(column.data_type());
        terminal = checked_accumulate(terminal, offset, "batch")?;
        bytes = checked_accumulate(bytes, width - offset, "batch")?;
        if let Some(offsets) = variable_offsets(column.as_ref()) {
            bytes = checked_accumulate(bytes, selected_width(&offsets, row), "batch")?;
        }
        if column.nulls().is_some_and(|nulls| nulls.is_null(row)) {
            nulls.push(column_index);
        }
    }
    Ok(FlatRow {
        bytes,
        nulls,
        terminal,
    })
}

/// One side's constant per-row byte charge over its selected columns, or
/// `None` when any of them varies by row.
///
/// Mirrors the widths `flat_row` accumulates: a fixed-width column without
/// variable offsets contributes its full width and no terminal offset, and a
/// null-free source array contributes no null entry for any of its rows.
fn side_row_bytes(record: PayloadRef<'_>, projection: Option<&[(usize, usize)]>) -> Option<usize> {
    #[cfg(test)]
    tests::SIDE_VISITS.with(|count| count.set(count.get() + 1));
    let mut bytes = 0_usize;
    for (_, column) in row_columns(record, 0, projection) {
        let width = fixed_width(column.data_type())?;
        if variable_offsets(column.as_ref()).is_some() || column.null_count() != 0 {
            return None;
        }
        bytes = bytes.checked_add(width)?;
    }
    Some(bytes)
}

/// Records `bytes` in `model` or confirms every row of a side charges alike.
fn agrees_with(model: &mut Option<usize>, bytes: usize) -> Option<()> {
    if let Some(known) = *model {
        (known == bytes).then_some(())
    } else {
        *model = Some(bytes);
        Some(())
    }
}

fn row_columns<'a>(
    record: PayloadRef<'a>,
    offset: usize,
    projection: Option<&'a [(usize, usize)]>,
) -> impl Iterator<Item = (usize, &'a ArrayRef)> {
    let full = record
        .columns()
        .iter()
        .enumerate()
        .take(if projection.is_some() {
            0
        } else {
            record.num_columns()
        })
        .map(move |(index, column)| (offset + index, column));
    let selected = projection
        .into_iter()
        .flatten()
        .map(move |&(index, ordinal)| (ordinal, record.column(index)));
    full.chain(selected)
}

fn selected_width(offsets: &Offsets<'_>, row: usize) -> usize {
    match offsets {
        Offsets::Narrow(offsets) => usize::try_from(offsets[row + 1] - offsets[row])
            .expect("validated Arrow offsets fit usize"),
        Offsets::Wide(offsets) => usize::try_from(offsets[row + 1] - offsets[row])
            .expect("validated Arrow offsets fit usize"),
    }
}

fn estimated_rows(
    coordinates: impl ExactSizeIterator<Item = usize> + Clone,
    mut estimate: impl FnMut(usize) -> Result<usize>,
) -> Result<RowCache<usize>> {
    let mut rows = RowCache::new(coordinates.clone());
    for coordinate in coordinates {
        rows.get_or_try_insert_with(coordinate, || estimate(coordinate))?;
    }
    Ok(rows)
}

fn estimated_cost(rows: &RowCache<usize>, coordinate: usize) -> usize {
    *rows
        .get(coordinate)
        .expect("nested preflight estimated every matched row")
}

fn estimated_row(record: PayloadRef<'_>, projection: Option<&[(usize, usize)]>) -> Result<usize> {
    #[cfg(test)]
    tests::NESTED_VISITS.with(|count| count.set(count.get() + 1));
    row_columns(record, 0, projection).try_fold(0_usize, |total, (_, column)| {
        let bytes = column
            .to_data()
            .slice(record.offset(), 1)
            .get_slice_memory_size()
            .map_err(|error| CalcFlowError::InvalidArgument {
                field: "batch".into(),
                message: format!("Arrow slice memory could not be measured: {error}"),
            })?;
        total
            .checked_add(bytes)
            .ok_or_else(|| CalcFlowError::InvalidArgument {
                field: "batch".into(),
                message: "size sum overflowed usize".into(),
            })
    })
}

fn estimated(record: &RecordBatch) -> Result<usize> {
    #[cfg(test)]
    tests::NESTED_VISITS.with(|count| count.set(count.get() + 1));
    Batch::table(vec![record.clone()], BatchMetadata::default())?.estimated_bytes()
}

fn oversized_row(bytes: usize, budget: EdgeBudget) -> CalcFlowError {
    OutputChunkErrors::STREAM_JOIN.over_budget_row(bytes, budget.max_bytes)
}

/// Conservative logical byte charges for every row of one incoming record.
///
/// Returns `None` when any column lacks the flat width model, so the charge
/// fast path never runs for nested schemas. Each row carries the state-row
/// overhead plus a whole-schema null-bitmap allowance, keeping the sum an
/// upper bound of the detailed flat model for single-chunk decisions.
pub(super) fn flat_row_charges(record: &RecordBatch, overhead: u64) -> Result<Option<Vec<u64>>> {
    let bitmap_per_row = record.num_columns().div_ceil(8);
    let overhead = usize::try_from(overhead).map_err(|_| overflow_charge())?;
    let mut constant = checked_accumulate(overhead, bitmap_per_row, "batch")?;
    for column in record.columns() {
        let Some(width) = fixed_width(column.data_type()) else {
            return Ok(None);
        };
        constant = checked_accumulate(constant, width, "batch")?;
    }
    let mut charges = vec![constant; record.num_rows()];
    for column in record.columns() {
        if let Some(offsets) = variable_offsets(column.as_ref()) {
            offsets.add_charges(&mut charges)?;
        }
    }
    let charges = charges
        .into_iter()
        .map(|charge| u64::try_from(charge).map_err(|_| overflow_charge()))
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(charges))
}

fn overflow_charge() -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "batch".into(),
        message: "row charge overflowed".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EventTime;
    use datafusion::arrow::{
        array::{ArrayRef, Int64Array, ListArray, StringArray},
        datatypes::{DataType, Field, Int64Type, Schema},
    };
    use std::{cell::Cell, sync::Arc};

    thread_local! {
        pub(super) static FLAT_VISITS: Cell<usize> = const { Cell::new(0) };
        pub(super) static NESTED_VISITS: Cell<usize> = const { Cell::new(0) };
        pub(super) static SIDE_VISITS: Cell<usize> = const { Cell::new(0) };
        pub(super) static FORCE_SLOW: Cell<bool> = const { Cell::new(false) };
    }

    /// Forces `flat_ranges` onto the per-pair scan while alive.
    pub(super) struct PerPairScan;
    impl PerPairScan {
        pub(super) fn force() -> Self {
            FORCE_SLOW.with(|forced| forced.set(true));
            PerPairScan
        }
    }
    impl Drop for PerPairScan {
        fn drop(&mut self) {
            FORCE_SLOW.with(|forced| forced.set(false));
        }
    }

    fn sparse_preflight(nested: bool, retained: usize) {
        let column: ArrayRef = if nested {
            Arc::new(ListArray::from_iter_primitive::<Int64Type, _, _>([Some(
                vec![Some(7_i64)],
            )]))
        } else {
            Arc::new(Int64Array::from(vec![7]))
        };
        let field = Field::new("value", column.data_type().clone(), true);
        let record =
            RecordBatch::try_new(Arc::new(Schema::new(vec![field.clone()])), vec![column]).unwrap();
        let schema = Arc::new(Schema::new(vec![field.clone(), field.with_name("other")]));
        let admitted = vec![AdmittedRow {
            record: record.clone().into(),
            event_time: EventTime::from_micros(100),
            row_id: 0,
            retain: true,
        }];
        let opposite = vec![
            StoredRow {
                record: record.into(),
                event_time: EventTime::from_micros(100),
                row_id: 0,
                charge: 0,
                encoded_key: Arc::new(vec![].into()),
            };
            retained
        ];
        let matched = [
            MatchedPair {
                pos: 0,
                opposite_index: retained - 1,
            },
            MatchedPair {
                pos: 0,
                opposite_index: 0,
            },
            MatchedPair {
                pos: 0,
                opposite_index: retained - 1,
            },
        ];
        FLAT_VISITS.with(|count| count.set(0));
        NESTED_VISITS.with(|count| count.set(0));
        let output = JoinOutput {
            schema: &schema,
            projection: None,
            admitted: &admitted,
            opposite: &opposite,
            matched: &matched,
            incoming_is_left: true,
            operator_id: "sparse",
            admitted_charges: None,
        };
        let ranges = output.ranges(EdgeBudget::new(2, 4096).unwrap()).unwrap();
        assert_eq!(ranges, vec![0..2, 2..3]);
        for range in ranges {
            let rows = range.len();
            let batch = output.materialize(range).unwrap();
            assert_eq!(batch.num_rows(), rows);
            assert_eq!(batch.column(0).data_type() == &DataType::Int64, !nested);
        }
        assert!(
            FLAT_VISITS.with(Cell::get) <= 3,
            "flat costs must visit only distinct matched rows"
        );
        assert!(
            NESTED_VISITS.with(Cell::get) <= 5,
            "nested costs must visit only matched rows and bounded ranges"
        );
    }

    #[test]
    fn test_sparse_flat_preflight_cost_is_independent_of_retained_rows() {
        for retained in [1_000, 100_000] {
            sparse_preflight(false, retained);
        }
    }

    #[test]
    fn test_sparse_nested_preflight_cost_is_independent_of_retained_rows() {
        for retained in [1_000, 100_000] {
            sparse_preflight(true, retained);
        }
    }

    #[test]
    fn test_flat_preflight_avoids_per_row_scratch_allocations() {
        let record = RecordBatch::try_from_iter(vec![
            (
                "key",
                Arc::new(StringArray::from(vec!["K0000000"])) as ArrayRef,
            ),
            ("value", Arc::new(Int64Array::from(vec![7])) as ArrayRef),
        ])
        .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("left_key", DataType::Utf8, false),
            Field::new("left_value", DataType::Int64, false),
            Field::new("right_key", DataType::Utf8, false),
            Field::new("right_value", DataType::Int64, false),
        ]));
        let admitted = (0..1_000)
            .map(|_| AdmittedRow {
                record: record.clone().into(),
                event_time: EventTime::from_micros(100),
                row_id: 0,
                retain: true,
            })
            .collect::<Vec<_>>();
        let opposite = vec![
            StoredRow {
                record: record.into(),
                event_time: EventTime::from_micros(100),
                row_id: 0,
                charge: 0,
                encoded_key: Arc::new(vec![].into()),
            };
            1_000
        ];
        for fanout in [1, 10] {
            let matched = (0..1_000)
                .flat_map(|pos| {
                    (0..fanout).map(move |offset| MatchedPair {
                        pos,
                        opposite_index: (pos + offset) % 1_000,
                    })
                })
                .collect::<Vec<_>>();
            let output = JoinOutput {
                schema: &schema,
                projection: None,
                admitted: &admitted,
                opposite: &opposite,
                matched: &matched,
                incoming_is_left: true,
                operator_id: "allocation",
                admitted_charges: None,
            };
            let bytes = Batch::table(
                vec![output.materialize(0..1_000).unwrap()],
                BatchMetadata::default(),
            )
            .unwrap()
            .estimated_bytes()
            .unwrap();
            let measured = allocation_counter::measure(|| {
                let ranges = output
                    .ranges(EdgeBudget::new(1_000, bytes).unwrap())
                    .unwrap();
                assert_eq!(ranges.len(), fanout);
                assert_eq!(ranges.iter().map(Range::len).sum::<usize>(), matched.len());
            });
            assert!(
                measured.count_total < 1_000,
                "fanout {fanout}: preflight allocated {} times; flat row scratch must be reused",
                measured.count_total
            );
        }
    }

    fn int64_row() -> RecordBatch {
        RecordBatch::try_from_iter(vec![(
            "value",
            Arc::new(Int64Array::from(vec![7])) as ArrayRef,
        )])
        .unwrap()
    }

    fn admitted_row(record: &RecordBatch) -> AdmittedRow {
        AdmittedRow {
            record: record.clone().into(),
            event_time: EventTime::from_micros(100),
            row_id: 0,
            retain: true,
        }
    }

    fn stored_row(record: &RecordBatch) -> StoredRow {
        StoredRow {
            record: record.clone().into(),
            event_time: EventTime::from_micros(100),
            row_id: 0,
            charge: 0,
            encoded_key: Arc::new(vec![].into()),
        }
    }

    fn int64_pair_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("left_value", DataType::Int64, false),
            Field::new("right_value", DataType::Int64, false),
        ]))
    }

    #[test]
    fn test_flat_preflight_cache_allocations_are_independent_of_distinct_rows() {
        let (rows, fanout) = (20_000, 4);
        let record = int64_row();
        let schema = int64_pair_schema();
        let admitted = (0..rows).map(|_| admitted_row(&record)).collect::<Vec<_>>();
        let opposite = vec![stored_row(&record); rows];
        let matched: super::super::matches::Selection = (0..rows)
            .flat_map(|pos| {
                (0..fanout).map(move |offset| MatchedPair {
                    pos,
                    opposite_index: (pos + offset * 5_000) % rows,
                })
            })
            .collect::<Vec<_>>()
            .into();
        let output = JoinOutput {
            schema: &schema,
            projection: None,
            admitted: &admitted,
            opposite: &opposite,
            matched: &matched,
            incoming_is_left: true,
            operator_id: "distinct",
            admitted_charges: None,
        };
        FLAT_VISITS.with(|count| count.set(0));
        let measured = allocation_counter::measure(|| {
            let _forced = PerPairScan::force();
            let ranges = output
                .ranges(EdgeBudget::new(1_000, 1 << 20).unwrap())
                .unwrap();
            assert_eq!(ranges.len(), rows * fanout / 1_000);
        });
        assert_eq!(FLAT_VISITS.with(Cell::get), 2 * rows);
        assert!(
            measured.count_total < 64,
            "flat preflight cached {} distinct rows with {} allocations; per-row cost \
             caches must be dense",
            2 * rows,
            measured.count_total
        );
    }

    #[test]
    fn test_interval_preflight_checks_each_incoming_run_once() {
        let (rows, fanout) = (16, 11);
        let record = int64_row();
        let schema = int64_pair_schema();
        let admitted = (0..rows).map(|_| admitted_row(&record)).collect::<Vec<_>>();
        let opposite = vec![stored_row(&record); rows];
        let matched = (0..rows)
            .flat_map(|pos| {
                (0..fanout).map(move |offset| MatchedPair {
                    pos,
                    opposite_index: (pos + offset) % rows,
                })
            })
            .collect::<Vec<_>>();
        let output = JoinOutput {
            schema: &schema,
            projection: None,
            admitted: &admitted,
            opposite: &opposite,
            matched: &matched,
            incoming_is_left: true,
            operator_id: "interval-widths",
            admitted_charges: None,
        };
        let budget = EdgeBudget::new(17, 160).unwrap();
        SIDE_VISITS.with(|count| count.set(0));
        let fast = output.ranges(budget).unwrap();
        let visits = SIDE_VISITS.with(Cell::get);
        let slow = {
            let _scan = PerPairScan::force();
            output.ranges(budget).unwrap()
        };
        assert_eq!(fast, slow);
        assert_eq!(fast.iter().map(Range::len).sum::<usize>(), rows * fanout);
        assert_eq!(
            visits,
            rows * 2,
            "fanout must not repeat incoming width checks"
        );
    }

    #[test]
    fn test_fixed_width_ranges_visit_a_constant_number_of_flat_rows() {
        let (rows, fanout) = (25_000, 4);
        let record = int64_row();
        let schema = int64_pair_schema();
        let admitted = (0..rows).map(|_| admitted_row(&record)).collect::<Vec<_>>();
        let opposite = vec![stored_row(&record); rows];
        let matched: super::super::matches::Selection = (0..rows)
            .flat_map(|pos| {
                (0..fanout).map(move |offset| MatchedPair {
                    pos,
                    opposite_index: (pos + offset * 5_000) % rows,
                })
            })
            .collect::<Vec<_>>()
            .into();
        let output = JoinOutput {
            schema: &schema,
            projection: None,
            admitted: &admitted,
            opposite: &opposite,
            matched: &matched,
            incoming_is_left: true,
            operator_id: "fixed-width",
            admitted_charges: None,
        };
        FLAT_VISITS.with(|count| count.set(0));
        let ranges = output
            .ranges(EdgeBudget::new(1_000, 1 << 20).unwrap())
            .unwrap();
        assert_eq!(ranges.iter().map(Range::len).sum::<usize>(), rows * fanout);
        let visits = FLAT_VISITS.with(Cell::get);
        assert!(
            visits <= 4,
            "constant-row-bytes preflight visited {visits} rows for {} pairs; \
             fixed-width null-free ranges need no per-row costs",
            rows * fanout
        );
    }

    #[test]
    fn test_fixed_width_ranges_match_the_per_pair_scan() {
        let (rows, fanout) = (500, 3);
        let record = int64_row();
        let schema = int64_pair_schema();
        let admitted = (0..rows).map(|_| admitted_row(&record)).collect::<Vec<_>>();
        let opposite = vec![stored_row(&record); rows];
        let matched: Vec<MatchedPair> = (0..rows)
            .flat_map(|pos| {
                (0..fanout).map(move |offset| MatchedPair {
                    pos,
                    opposite_index: (pos + offset * 100) % rows,
                })
            })
            .collect();
        let output = JoinOutput {
            schema: &schema,
            projection: None,
            admitted: &admitted,
            opposite: &opposite,
            matched: &matched,
            incoming_is_left: true,
            operator_id: "equivalent",
            admitted_charges: None,
        };
        // 16-byte rows: a 500-byte budget fits 31 rows while 64 would fit the
        // row cap, and a 10-row cap leaves the byte budget idle.
        for (max_rows, max_bytes) in [(64, 500), (10, 100_000)] {
            let budget = EdgeBudget::new(max_rows, max_bytes).unwrap();
            FLAT_VISITS.with(|count| count.set(0));
            let fast = output.ranges(budget).unwrap();
            assert_eq!(
                FLAT_VISITS.with(Cell::get),
                0,
                "fixed-width null-free preflight must take the arithmetic path"
            );
            let slow = {
                let _forced = PerPairScan::force();
                output.ranges(budget).unwrap()
            };
            // Constant row bytes make the greedy scan fill every chunk to
            // min(max_rows, max_bytes / row_bytes), so boundaries agree.
            assert_eq!(fast, slow, "budget {max_rows}x{max_bytes}");
            let mut end = 0;
            for range in &fast {
                assert_eq!(range.start, end);
                assert!(range.len() <= max_rows, "{range:?} rows over {max_rows}");
                assert!(
                    range.len() * 16 <= max_bytes,
                    "{range:?} bytes over {max_bytes}"
                );
                assert_eq!(
                    output.materialize(range.clone()).unwrap().num_rows(),
                    range.len()
                );
                end = range.end;
            }
            assert_eq!(end, matched.len());
        }
    }

    #[test]
    fn test_oversized_fixed_width_row_reports_the_same_error_in_both_paths() {
        let record = int64_row();
        let schema = int64_pair_schema();
        let admitted = vec![admitted_row(&record)];
        let opposite = vec![stored_row(&record)];
        let matched = vec![MatchedPair {
            pos: 0,
            opposite_index: 0,
        }];
        let output = JoinOutput {
            schema: &schema,
            projection: None,
            admitted: &admitted,
            opposite: &opposite,
            matched: &matched,
            incoming_is_left: true,
            operator_id: "oversized",
            admitted_charges: None,
        };
        // One int64 pair costs 16 bytes; the 8-byte budget rejects the first
        // row of either path with the same charge.
        let budget = EdgeBudget::new(4, 8).unwrap();
        let fast = output.ranges(budget).unwrap_err();
        let slow = {
            let _forced = PerPairScan::force();
            output.ranges(budget).unwrap_err()
        };
        assert_eq!(fast.to_string(), slow.to_string());
    }

    #[test]
    fn test_nullable_or_variable_columns_keep_the_per_pair_scan() {
        let null_column: ArrayRef = Arc::new(Int64Array::from(vec![Some(7), None]));
        let null_record = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "value",
                DataType::Int64,
                true,
            )])),
            vec![null_column],
        )
        .unwrap();
        let text_record = RecordBatch::try_from_iter(vec![
            (
                "key",
                Arc::new(StringArray::from(vec!["K0000000", "K0000001"])) as ArrayRef,
            ),
            ("value", Arc::new(Int64Array::from(vec![7, 8])) as ArrayRef),
        ])
        .unwrap();
        let text_schema = Arc::new(Schema::new(vec![
            Field::new("left_key", DataType::Utf8, false),
            Field::new("left_value", DataType::Int64, false),
            Field::new("right_key", DataType::Utf8, false),
            Field::new("right_value", DataType::Int64, false),
        ]));
        let nullable_schema = Arc::new(Schema::new(vec![
            Field::new("left_value", DataType::Int64, true),
            Field::new("right_value", DataType::Int64, true),
        ]));
        // A nullable selected column and a variable-width selected column each
        // force the per-pair scan, even when values never vary.
        for (records, schema) in [(null_record, nullable_schema), (text_record, text_schema)] {
            let admitted = (0..2)
                .map(|row| AdmittedRow {
                    record: records.slice(row, 1).into(),
                    event_time: EventTime::from_micros(100),
                    row_id: row as u64,
                    retain: true,
                })
                .collect::<Vec<_>>();
            let opposite = [
                stored_row(&records.slice(0, 1)),
                stored_row(&records.slice(1, 1)),
            ]
            .into_iter()
            .chain([stored_row(&records.slice(0, 1))])
            .collect::<Vec<_>>();
            let matched = [(0, 0), (0, 1), (1, 2), (1, 0)]
                .map(|(pos, opposite_index)| MatchedPair {
                    pos,
                    opposite_index,
                })
                .to_vec();
            let output = JoinOutput {
                schema: &schema,
                projection: None,
                admitted: &admitted,
                opposite: &opposite,
                matched: &matched,
                incoming_is_left: true,
                operator_id: "fallback",
                admitted_charges: None,
            };
            FLAT_VISITS.with(|count| count.set(0));
            let ranges = output.ranges(EdgeBudget::new(2, 4_096).unwrap()).unwrap();
            assert!(
                FLAT_VISITS.with(Cell::get) > 0,
                "nullable or variable columns must keep per-pair costs"
            );
            let mut end = 0;
            for range in &ranges {
                assert_eq!(range.start, end);
                assert!(range.len() <= 2);
                assert_eq!(
                    output.materialize(range.clone()).unwrap().num_rows(),
                    range.len()
                );
                end = range.end;
            }
            assert_eq!(end, matched.len());
        }
    }

    #[test]
    fn test_flat_preflight_scratch_ignores_retained_coordinate_gaps() {
        let record = int64_row();
        let schema = int64_pair_schema();
        let admitted = vec![admitted_row(&record)];
        let mut opposite = super::super::retained::StateRows::default();
        let first = opposite.append(vec![stored_row(&record)])[0];
        for _ in 0..4_096 {
            let expired = opposite.append(vec![stored_row(&record)])[0];
            opposite.remove(expired);
        }
        let last = opposite.append(vec![stored_row(&record)])[0];
        assert!(
            last - first > 4_000_000,
            "expired chunks leave a coordinate gap"
        );
        let matched = [last, first, last].map(|opposite_index| MatchedPair {
            pos: 0,
            opposite_index,
        });
        let output = JoinOutput {
            schema: &schema,
            projection: None,
            admitted: &admitted,
            opposite: &opposite,
            matched: &matched,
            incoming_is_left: true,
            operator_id: "gaps",
            admitted_charges: None,
        };
        FLAT_VISITS.with(|count| count.set(0));
        let measured = allocation_counter::measure(|| {
            let ranges = output.ranges(EdgeBudget::new(2, 4096).unwrap()).unwrap();
            assert_eq!(ranges, vec![0..2, 2..3]);
        });
        assert_eq!(FLAT_VISITS.with(Cell::get), 3);
        assert!(
            measured.bytes_max < 64 * 1024,
            "preflight scratch must follow matched rows, not the retained coordinate gap: \
             {measured:?}"
        );
        assert_eq!(output.materialize(0..3).unwrap().num_rows(), 3);
    }

    #[test]
    fn test_single_chunk_output_takes_the_charge_fast_path() {
        let column: ArrayRef = Arc::new(Int64Array::from(vec![7_i64; 4]));
        let field = Field::new("value", DataType::Int64, false);
        let record =
            RecordBatch::try_new(Arc::new(Schema::new(vec![field.clone()])), vec![column]).unwrap();
        let schema = Arc::new(Schema::new(vec![
            field.clone().with_name("left_value"),
            field.with_name("right_value"),
        ]));
        let admitted: Vec<AdmittedRow> = (0..4)
            .map(|index| AdmittedRow {
                record: record.slice(index, 1).into(),
                event_time: EventTime::from_micros(100),
                row_id: index as u64,
                retain: true,
            })
            .collect();
        let opposite: Vec<StoredRow> = (0..4)
            .map(|index| StoredRow {
                record: record.slice(index, 1).into(),
                event_time: EventTime::from_micros(100),
                row_id: index as u64,
                charge: 81,
                encoded_key: Arc::new(vec![].into()),
            })
            .collect();
        let matched: Vec<MatchedPair> = (0..4)
            .map(|index| MatchedPair {
                pos: index,
                opposite_index: index,
            })
            .collect();
        let charges = vec![81_u64; 4];
        let output = JoinOutput {
            schema: &schema,
            projection: None,
            admitted: &admitted,
            opposite: &opposite,
            matched: &matched,
            incoming_is_left: true,
            operator_id: "fast",
            admitted_charges: Some(&charges),
        };
        FLAT_VISITS.with(|count| count.set(0));
        let ranges = output.ranges(EdgeBudget::new(4, 1 << 20).unwrap()).unwrap();
        assert_eq!(ranges, vec![0..4]);
        assert_eq!(
            FLAT_VISITS.with(Cell::get),
            0,
            "a provably single-chunk output must not visit matched rows"
        );
    }

    #[test]
    fn test_charge_fast_path_falls_back_when_the_bound_exceeds_the_budget() {
        super::super::output_projection::tests::serial_columns_and_budgets();
        let column: ArrayRef = Arc::new(Int64Array::from(vec![7_i64; 4]));
        let field = Field::new("value", DataType::Int64, false);
        let record =
            RecordBatch::try_new(Arc::new(Schema::new(vec![field.clone()])), vec![column]).unwrap();
        let schema = Arc::new(Schema::new(vec![
            field.clone().with_name("left_value"),
            field.with_name("right_value"),
        ]));
        let admitted: Vec<AdmittedRow> = (0..4)
            .map(|index| AdmittedRow {
                record: record.slice(index, 1).into(),
                event_time: EventTime::from_micros(100),
                row_id: index as u64,
                retain: true,
            })
            .collect();
        let opposite: Vec<StoredRow> = (0..4)
            .map(|index| StoredRow {
                record: record.slice(index, 1).into(),
                event_time: EventTime::from_micros(100),
                row_id: index as u64,
                charge: 81,
                encoded_key: Arc::new(vec![].into()),
            })
            .collect();
        let matched: Vec<MatchedPair> = (0..4)
            .map(|index| MatchedPair {
                pos: index,
                opposite_index: index,
            })
            .collect();
        let charges = vec![81_u64; 4];
        let output = JoinOutput {
            schema: &schema,
            projection: None,
            admitted: &admitted,
            opposite: &opposite,
            matched: &matched,
            incoming_is_left: true,
            operator_id: "fallback",
            admitted_charges: Some(&charges),
        };
        FLAT_VISITS.with(|count| count.set(0));
        // 4 pairs x (81 + 81) = 648 charge bytes exceeds 200, but exact flat
        // widths fit, so planning falls back past the charge bound to the
        // fixed-width arithmetic, still without per-row costs.
        let ranges = output.ranges(EdgeBudget::new(2, 200).unwrap()).unwrap();
        assert_eq!(ranges, vec![0..2, 2..4]);
        assert_eq!(FLAT_VISITS.with(Cell::get), 0);
    }
}

#[cfg(test)]
#[path = "materialization/source_width_tests.rs"]
mod source_width_tests;
