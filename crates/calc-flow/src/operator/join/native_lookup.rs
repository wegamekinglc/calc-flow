use super::columnar::FramedKey;
use super::{
    AdmittedRow, CompiledJoin, JoinTimeBounds, MatchedPair, SidePlan, StoredRow,
    StreamJoinOperator, enforce_match_limit,
};
use crate::{CalcFlowError, EventTime, Result};
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::execution::memory_pool::MemoryReservation;
use std::{collections::BTreeMap, sync::Arc};

const ENTRY_BYTES: usize = 128;
const BASE_BYTES: usize = 1_024;

#[cfg(test)]
#[path = "tests/native_index_allocation_tests.rs"]
mod allocation_tests;

pub(super) struct NativeIndex {
    entries: BTreeMap<(Arc<FramedKey>, EventTime, u64), usize>,
    credit: Arc<MemoryReservation>,
}

impl NativeIndex {
    #[cfg(test)]
    pub(super) fn funded_bytes(&self) -> usize {
        self.credit.size()
    }

    pub(super) fn new(rows: &[StoredRow], credit: MemoryReservation) -> Self {
        let mut index = Self {
            entries: BTreeMap::new(),
            credit: Arc::new(credit),
        };
        index.append(0, rows);
        index
    }

    pub(super) fn reserve(&self, rows: usize) -> Option<AppendCredit> {
        let bytes = rows.checked_mul(ENTRY_BYTES)?;
        self.credit.try_grow(bytes).ok()?;
        Some(AppendCredit {
            credit: Arc::clone(&self.credit),
            bytes,
        })
    }

    pub(super) fn append(&mut self, offset: usize, rows: &[StoredRow]) {
        for (index, row) in rows.iter().enumerate() {
            self.entries.insert(
                (Arc::clone(&row.encoded_key), row.event_time, row.row_id),
                offset + index,
            );
        }
    }

    pub(super) fn remove(&mut self, row: &StoredRow, moved: Option<(&StoredRow, usize)>) {
        self.entries
            .remove(&(Arc::clone(&row.encoded_key), row.event_time, row.row_id));
        self.credit.shrink(ENTRY_BYTES);
        if let Some((row, index)) = moved {
            *self
                .entries
                .get_mut(&(Arc::clone(&row.encoded_key), row.event_time, row.row_id))
                .expect("moved retained identity") = index;
        }
    }

    fn range<'a>(
        &'a self,
        key: &Arc<FramedKey>,
        range: (EventTime, EventTime),
    ) -> impl Iterator<Item = usize> + 'a {
        self.entries
            .range((Arc::clone(key), range.0, 0)..=(Arc::clone(key), range.1, u64::MAX))
            .map(|(_, &index)| index)
    }
}

pub(super) fn eligible(compiled: &CompiledJoin, left: &Schema) -> bool {
    compiled
        .left_key_indices
        .iter()
        .all(|&index| native_type(left.field(index).data_type()))
}

fn native_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean | DataType::Int16 | DataType::Int32 | DataType::Int64
    ) || matches!(
        data_type,
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64
    ) || matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Timestamp(..)
    )
}

pub(super) struct NativeMatches {
    pub(super) pairs: Vec<MatchedPair>,
    pub(super) keys: NativeKeys,
    pub(super) credit: MemoryReservation,
}

pub(super) struct NativeKeys {
    pub(super) keys: Vec<Arc<FramedKey>>,
    _credit: Arc<MemoryReservation>,
}

pub(super) struct AppendCredit {
    credit: Arc<MemoryReservation>,
    bytes: usize,
}

impl AppendCredit {
    pub(super) fn commit(mut self) {
        self.bytes = 0;
    }
}

impl Drop for AppendCredit {
    fn drop(&mut self) {
        self.credit.shrink(self.bytes);
    }
}

impl StreamJoinOperator {
    pub(super) fn optional_credit(&mut self, bytes: usize) -> Result<Option<MemoryReservation>> {
        let credit = self
            .runtime
            .runtime()?
            .incremental_reservation("stream-join-native");
        Ok(credit.try_grow(bytes).ok().map(|()| credit))
    }

    pub(super) fn native_matches(
        &mut self,
        plan: &SidePlan,
        admitted: &[AdmittedRow],
    ) -> Result<Option<NativeMatches>> {
        if !eligible(&self.compiled, self.input_schema(0)) {
            return Ok(None);
        }
        if !self.ensure_native_index(!plan.incoming_is_left)? {
            return Ok(None);
        }
        self.probe_native_index(plan, admitted)
    }

    fn probe_native_index(
        &mut self,
        plan: &SidePlan,
        admitted: &[AdmittedRow],
    ) -> Result<Option<NativeMatches>> {
        let key_bytes = probe_key_charge(admitted, &plan.key_indices)?;
        let Some(credit) = self.optional_credit(key_bytes)? else {
            return Ok(None);
        };
        let shared = Arc::new(credit);
        let keys = admitted
            .iter()
            .map(|row| {
                super::columnar::funded_key(
                    row.record.columns(),
                    row.record.offset(),
                    &plan.key_indices,
                    Arc::clone(&shared),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.collect_native_pairs(
            plan,
            admitted,
            NativeKeys {
                keys,
                _credit: shared,
            },
        )
    }

    fn collect_native_pairs(
        &mut self,
        plan: &SidePlan,
        admitted: &[AdmittedRow],
        keys: NativeKeys,
    ) -> Result<Option<NativeMatches>> {
        let opposite = opposite_rows(self, plan);
        let index = opposite
            .1
            .as_ref()
            .expect("native index built before probing");
        let count = count_pairs(
            index,
            admitted,
            &keys.keys,
            self.spec.bounds,
            plan,
            self.spec.limits.max_matches_per_input_batch,
        )?;
        enforce_match_limit(
            count,
            &mut self.state.metrics.match_limit_failures,
            self.spec.limits.max_matches_per_input_batch,
            &self.name,
        )?;
        let bytes = count
            .checked_mul(size_of::<MatchedPair>() + 256)
            .ok_or_else(|| scratch_error(&self.name))?;
        let Some(credit) = self.optional_credit(bytes)? else {
            return Ok(None);
        };
        let opposite = opposite_rows(self, plan);
        let index = opposite
            .1
            .as_ref()
            .expect("native index built before probing");
        let mut pairs = Vec::with_capacity(count);
        for (pos, row) in admitted.iter().enumerate() {
            let key = &keys.keys[pos];
            pairs.extend(
                index
                    .range(key, time_range(self.spec.bounds, plan, row.event_time))
                    .map(|opposite_index| MatchedPair {
                        pos,
                        opposite_index,
                    }),
            );
        }
        Ok(Some(NativeMatches {
            pairs,
            keys,
            credit,
        }))
    }

    pub(super) fn reserve_native_append(
        &mut self,
        incoming_is_left: bool,
        rows: usize,
    ) -> Result<Option<AppendCredit>> {
        if rows == 0 || !eligible(&self.compiled, self.input_schema(0)) {
            return Ok(None);
        }
        if !self.ensure_native_index(incoming_is_left)? {
            return Ok(None);
        }
        let retained = if incoming_is_left {
            &mut self.state.left
        } else {
            &mut self.state.right
        };
        let credit = retained
            .1
            .as_ref()
            .expect("native append index initialized")
            .reserve(rows);
        if credit.is_none() {
            retained.1 = None;
        }
        Ok(credit)
    }

    fn ensure_native_index(&mut self, left: bool) -> Result<bool> {
        let rows = if left {
            &self.state.left
        } else {
            &self.state.right
        };
        if rows.1.is_some() {
            return Ok(true);
        }
        let Some(bytes) = rows
            .len()
            .checked_mul(ENTRY_BYTES)
            .and_then(|bytes| bytes.checked_add(BASE_BYTES))
        else {
            return Ok(false);
        };
        let Some(credit) = self.optional_credit(bytes)? else {
            return Ok(false);
        };
        let rows = if left {
            &mut self.state.left
        } else {
            &mut self.state.right
        };
        rows.1 = Some(NativeIndex::new(rows, credit));
        Ok(true)
    }
}

fn time_range(bounds: JoinTimeBounds, plan: &SidePlan, time: EventTime) -> (EventTime, EventTime) {
    let (before, after) = if plan.incoming_is_left {
        (bounds.before_micros, bounds.after_micros)
    } else {
        (bounds.after_micros, bounds.before_micros)
    };
    let minimum = (i128::from(time.as_micros()) - i128::from(before)).max(i128::from(i64::MIN));
    let maximum = (i128::from(time.as_micros()) + i128::from(after)).min(i128::from(i64::MAX));
    (
        EventTime::from_micros(i64::try_from(minimum).expect("minimum clamped to EventTime")),
        EventTime::from_micros(i64::try_from(maximum).expect("maximum clamped to EventTime")),
    )
}

fn probe_key_charge(admitted: &[AdmittedRow], indices: &[usize]) -> Result<usize> {
    admitted.iter().try_fold(0_usize, |total, row| {
        indices.iter().try_fold(total, |bytes, &index| {
            let array = row.record.column(index);
            let value = usize::try_from(super::logical_cell_charge(
                array.as_ref(),
                row.record.offset(),
            )?)
            .map_err(|_| scratch_error("join"))?;
            let timezone = match array.data_type() {
                DataType::Timestamp(_, Some(timezone)) => timezone.len(),
                _ => 0,
            };
            value
                .checked_add(timezone)
                .and_then(|value| value.checked_add(64))
                .and_then(|value| value.checked_mul(4))
                .and_then(|value| bytes.checked_add(value))
                .ok_or_else(|| scratch_error("join"))
        })
    })
}

fn scratch_error(name: &str) -> CalcFlowError {
    CalcFlowError::DataFusion {
        node_id: Some(name.to_owned()),
        message: "native Join scratch size overflow".to_owned(),
    }
}

fn opposite_rows<'a>(operator: &'a StreamJoinOperator, plan: &SidePlan) -> &'a super::RetainedRows {
    if plan.incoming_is_left {
        &operator.state.right
    } else {
        &operator.state.left
    }
}

fn count_pairs(
    index: &NativeIndex,
    admitted: &[AdmittedRow],
    keys: &[Arc<FramedKey>],
    bounds: JoinTimeBounds,
    plan: &SidePlan,
    limit: u64,
) -> Result<usize> {
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);
    let mut count = 0_usize;
    for (row, key) in admitted.iter().zip(keys) {
        let remaining = limit.saturating_sub(count).saturating_add(1);
        count = count
            .checked_add(
                index
                    .range(key, time_range(bounds, plan, row.event_time))
                    .take(remaining)
                    .count(),
            )
            .ok_or_else(|| scratch_error("join"))?;
        if count > limit {
            break;
        }
    }
    Ok(count)
}
