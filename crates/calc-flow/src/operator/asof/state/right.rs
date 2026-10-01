use super::{Encoding, RightOrder, RowRef, SequenceColumn, SequenceKind, SequenceRef};
use std::{
    borrow::Cow,
    cmp::Ordering,
    collections::{BTreeSet, btree_set},
};

type OrderRef<'a> = (&'a i64, SequenceRef<'a>);
type RightRow<'a> = (OrderRef<'a>, Option<&'a RowRef>);
type ExpiredRow<'a> = (i64, Option<&'a Encoding>, Option<&'a RowRef>);

#[cfg(test)]
thread_local! {
    static COLUMN_MOVES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn take_column_moves() -> usize {
    COLUMN_MOVES.with(|moves| moves.replace(0))
}

/// Independent ordered columns for payload and identity-only history. Keeping
/// the two lifetimes separate lets each sweep visit only an expired prefix.
#[derive(Clone, Default)]
pub(in super::super) struct RightBucket {
    payloads: Option<Box<RightRun<Vec<Option<RowRef>>>>>,
    identities: RightRun<()>,
    // Older payloads can expire after later identity-only history was retained.
    // A tree absorbs those rows without shifting the whole identity column.
    general_identities: BTreeSet<RightOrder>,
}

static EMPTY_PAYLOADS: RightRun<Vec<Option<RowRef>>> = RightRun {
    times: Vec::new(),
    sequences: SequenceColumn::empty(SequenceKind::Canonical),
    values: Vec::new(),
    head: 0,
};

struct RightRun<V> {
    times: Vec<i64>,
    sequences: SequenceColumn,
    values: V,
    head: usize,
}

trait RunValues: Clone {
    type Value;

    fn clone_with_capacity(&self) -> Self;
    fn with_capacity(capacity: usize) -> Self;
    fn get(&self, index: usize) -> &Self::Value;
    fn push(&mut self, value: Self::Value);
    fn insert(&mut self, index: usize, value: Self::Value);
    fn replace(&mut self, index: usize, value: Self::Value);
    fn remove(&mut self, index: usize);
    fn take(&mut self, index: usize) -> Self::Value;
    fn capacity(&self) -> usize;
    fn compact(&mut self, head: usize);
}

impl RunValues for Vec<Option<RowRef>> {
    type Value = RowRef;

    fn clone_with_capacity(&self) -> Self {
        let mut values = Vec::with_capacity(self.capacity());
        values.extend_from_slice(self);
        values
    }

    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }

    fn get(&self, index: usize) -> &RowRef {
        self[index].as_ref().expect("live ASOF run value")
    }

    fn push(&mut self, value: RowRef) {
        Vec::push(self, Some(value));
    }

    fn insert(&mut self, index: usize, value: RowRef) {
        Vec::insert(self, index, Some(value));
    }

    fn replace(&mut self, index: usize, value: RowRef) {
        self[index] = Some(value);
    }

    fn remove(&mut self, index: usize) {
        Vec::remove(self, index);
    }

    fn take(&mut self, index: usize) -> RowRef {
        self[index].take().expect("live ASOF run value")
    }

    fn capacity(&self) -> usize {
        Vec::capacity(self)
    }

    fn compact(&mut self, head: usize) {
        *self = self.split_off(head).into_boxed_slice().into_vec();
    }
}

// Identity history has no payload column or per-row value allocation.
impl RunValues for () {
    type Value = ();

    fn clone_with_capacity(&self) -> Self {}

    fn with_capacity(_: usize) -> Self {}

    fn get(&self, _: usize) -> &() {
        self
    }

    fn push(&mut self, (): ()) {}

    fn insert(&mut self, _: usize, (): ()) {}

    fn replace(&mut self, _: usize, (): ()) {}

    fn remove(&mut self, _: usize) {}

    fn take(&mut self, _: usize) {}

    fn capacity(&self) -> usize {
        0
    }

    fn compact(&mut self, _: usize) {}
}

impl<V: RunValues> Clone for RightRun<V> {
    fn clone(&self) -> Self {
        let mut times = Vec::with_capacity(self.times.capacity());
        times.extend_from_slice(&self.times);
        Self {
            times,
            sequences: self.sequences.clone(),
            values: self.values.clone_with_capacity(),
            head: self.head,
        }
    }
}

impl<V: RunValues> Default for RightRun<V> {
    fn default() -> Self {
        Self::with_capacity(0, SequenceKind::Canonical)
    }
}

impl<V: RunValues> RightRun<V> {
    fn integer_columns(&self) -> Option<(&[i64], &[u8])> {
        Some((
            &self.times[self.head..],
            self.sequences.integer_slice(self.head..self.times.len())?,
        ))
    }

    fn projected_capacities(&self, removed: usize, appended: usize) -> [usize; 3] {
        let mut capacities = [
            self.times.capacity(),
            self.sequences.capacity(),
            self.values.capacity(),
        ];
        if capacities[0] == 0 && appended > 0 {
            capacities[0] = 1;
            capacities[1] = 1;
            if size_of::<V>() != 0 {
                capacities[2] = 1;
            }
        }
        let required = self.times.len() + appended;
        grow_column_capacities(&mut capacities[..2], required);
        if size_of::<V>() != 0 {
            while capacities[2] < required {
                capacities[2] = (capacities[2] * 2).max(4);
            }
        }
        let live = self.len() - removed + appended;
        if capacities.iter().any(|capacity| *capacity > live * 2) {
            capacities = [live, live, if size_of::<V>() == 0 { 0 } else { live }];
        }
        capacities
    }
    fn with_capacity(capacity: usize, kind: SequenceKind) -> Self {
        Self {
            times: Vec::with_capacity(capacity),
            sequences: SequenceColumn::with_capacity(capacity, kind),
            values: V::with_capacity(capacity),
            head: 0,
        }
    }

    fn len(&self) -> usize {
        self.times.len() - self.head
    }

    fn at(&self, index: usize) -> Option<(OrderRef<'_>, &V::Value)> {
        if index < self.head {
            return None;
        }
        Some((
            (self.times.get(index)?, self.sequences.get(index)?),
            self.values.get(index),
        ))
    }

    fn last(&self) -> Option<(OrderRef<'_>, &V::Value)> {
        self.at(self.times.len().checked_sub(1)?)
    }

    fn locate(&self, order: &RightOrder) -> Result<usize, usize> {
        let start = self.times[self.head..].partition_point(|time| *time < order.0) + self.head;
        let end = self.times[self.head..].partition_point(|time| *time <= order.0) + self.head;
        self.sequences.binary_search(start..end, &order.1)
    }

    fn insert(&mut self, order: RightOrder, value: V::Value) {
        if self.times.capacity() == 0 {
            // Avoid Vec's four-element minimum for sparse buckets: even one
            // identity must fit the existing committed state charge.
            self.times = Vec::with_capacity(1);
            self.sequences = SequenceColumn::with_capacity(1, self.sequences.kind());
            self.values = V::with_capacity(1);
        }
        if self.last().is_none_or(|(last, _)| last < order_ref(&order)) {
            self.times.push(order.0);
            self.sequences.push(order.1);
            self.values.push(value);
            return;
        }
        match self.locate(&order) {
            Ok(index) => self.values.replace(index, value),
            Err(index) => {
                #[cfg(test)]
                COLUMN_MOVES.with(|moves| moves.set(moves.get() + self.times.len() - index));
                self.times.insert(index, order.0);
                self.sequences.insert(index, order.1);
                self.values.insert(index, value);
            }
        }
    }

    fn remove(&mut self, order: &RightOrder) {
        if let Ok(index) = self.locate(order) {
            self.times.remove(index);
            self.sequences.remove(index);
            self.values.remove(index);
            self.compact();
        }
    }

    fn prefix_len(&self, predicate: impl Fn(i64) -> bool) -> usize {
        self.times[self.head..].partition_point(|time| predicate(*time))
    }

    fn drop_prefix(&mut self, count: usize) {
        let end = self.head + count;
        for index in self.head..end {
            self.sequences.take_owner(index);
            self.values.take(index);
        }
        self.head = end;
    }

    fn compact(&mut self) {
        let limit = self.len().saturating_mul(2);
        if self.times.capacity() <= limit
            && self.sequences.capacity() <= limit
            && self.values.capacity() <= limit
        {
            return;
        }
        self.times = self
            .times
            .split_off(self.head)
            .into_boxed_slice()
            .into_vec();
        self.sequences.compact(self.head);
        self.values.compact(self.head);
        self.head = 0;
    }

    fn advance_index(&self, time: i64, next: &mut usize) -> Option<usize> {
        *next = (*next).max(self.head);
        while self.times.get(*next).is_some_and(|value| *value <= time) {
            *next += 1;
        }
        next.checked_sub(1).filter(|index| *index >= self.head)
    }

    fn advance(&self, time: i64, next: &mut usize) -> Option<(OrderRef<'_>, &V::Value)> {
        self.at(self.advance_index(time, next)?)
    }
}

#[derive(Default)]
pub(in super::super) struct RightCursor {
    payload: usize,
    identity: usize,
}

impl RightBucket {
    pub fn eviction_pending(
        &self,
        status: &super::super::StreamAsofJoinStatus,
        tolerance: u64,
        threshold: i128,
    ) -> bool {
        self.payload_min()
            .is_some_and(|time| super::payload_expired(time, tolerance, threshold))
            || self
                .identity_min()
                .is_some_and(|time| super::identity_expired(time, status))
    }

    pub fn projected_eviction(
        &self,
        status: &super::super::StreamAsofJoinStatus,
        tolerance: u64,
        threshold: i128,
    ) -> (usize, u64, u64) {
        let payloads = self.payloads();
        let removed_payloads =
            payloads.prefix_len(|time| super::payload_expired(time, tolerance, threshold));
        let removed_ordered = self
            .identities
            .prefix_len(|time| super::identity_expired(time, status));
        let removed_general = self
            .general_identities
            .iter()
            .take_while(|row| super::identity_expired(row.0, status))
            .count();
        let mut general = self.general_identities.len() - removed_general;
        let mut append = 0;
        let mut last = self
            .identities
            .last()
            .map(|(order, ())| order)
            .filter(|order| !super::identity_expired(*order.0, status));
        for index in payloads.head..payloads.head + removed_payloads {
            if super::identity_expired(payloads.times[index], status) {
                continue;
            }
            let (order, _) = payloads.at(index).expect("expired payload");
            if last.as_ref().is_none_or(|last| last < &order) {
                append += 1;
                last = Some(order);
            } else {
                general += 1;
            }
        }
        let payload_rows = payloads.len() - removed_payloads;
        let ordered_rows = self.identities.len() - removed_ordered;
        if ordered_rows + append == 0 && general == 1 {
            append += 1;
            general = 0;
        }
        let rows = payload_rows + ordered_rows + append + general;
        let payload_capacities = payloads.projected_capacities(removed_payloads, 0);
        let identity_capacities = self
            .identities
            .projected_capacities(removed_ordered, append);
        let width = self.identities.sequences.element_bytes();
        let metadata = projected_eviction_metadata(
            payload_rows,
            payload_capacities,
            identity_capacities,
            width,
            general,
        );
        let changed = removed_payloads + removed_ordered + removed_general > 0;
        (rows, metadata, if changed { metadata + 256 } else { 0 })
    }
    pub fn projected_admission_bytes(&self, additional: usize) -> u64 {
        let current = self.metadata_bytes();
        let Some(run) = &self.payloads else {
            return current
                + size_of::<RightRun<Vec<Option<RowRef>>>>() as u64
                + additional as u64 * (16 + self.identities.sequences.element_bytes() as u64);
        };
        let growth = |capacity: usize, width: usize| {
            let required = run.times.len() + additional;
            let next = if capacity >= required {
                capacity
            } else {
                required.max(capacity * 2).max(4)
            };
            ((next - capacity) * width) as u64
        };
        current
            + growth(run.times.capacity(), 8)
            + growth(run.sequences.capacity(), run.sequences.element_bytes())
            + growth(run.values.capacity(), 8)
    }
    pub fn checkpoint_capacities(&self) -> [usize; 5] {
        [
            self.payloads().times.capacity(),
            self.payloads().sequences.capacity(),
            self.payloads().values.capacity(),
            self.identities.times.capacity(),
            self.identities.sequences.capacity(),
        ]
    }

    pub fn checkpoint_integer_columns(&self) -> Option<(&[i64], &[u8])> {
        if !self.general_identities.is_empty() {
            return None;
        }
        let run = if self.identities.len() == 0 {
            self.payloads().integer_columns()?
        } else if self.payloads().len() == 0 {
            self.identities.integer_columns()?
        } else {
            return None;
        };
        Some(run)
    }

    pub fn checkpoint_payload_refs(&self) -> Option<&[Option<RowRef>]> {
        if self.identities.len() != 0 || !self.general_identities.is_empty() {
            return None;
        }
        let run = self.payloads();
        Some(&run.values[run.head..])
    }

    pub fn with_index_capacities(capacities: [usize; 5], kind: SequenceKind) -> Self {
        let mut bucket = Self::with_sequence_kind(kind);
        if capacities[0] != 0 {
            bucket.payloads = Some(Box::new(RightRun {
                times: Vec::with_capacity(capacities[0]),
                sequences: SequenceColumn::with_capacity(capacities[1], kind),
                values: Vec::with_capacity(capacities[2]),
                head: 0,
            }));
        }
        bucket.identities.times = Vec::with_capacity(capacities[3]);
        bucket.identities.sequences = SequenceColumn::with_capacity(capacities[4], kind);
        bucket
    }

    pub fn push_index(&mut self, (time, sequence): RightOrder, tag: u8, row: Option<RowRef>) {
        match tag {
            1 => {
                let payloads = self.payloads.as_mut().expect("validated payload capacity");
                payloads.times.push(time);
                payloads.sequences.push(sequence);
                payloads.values.push(row);
            }
            0 => {
                self.identities.times.push(time);
                self.identities.sequences.push(sequence);
            }
            _ => {
                self.general_identities.insert((time, sequence));
            }
        }
    }

    pub fn payload_len(&self) -> usize {
        self.payloads().len()
    }

    pub fn metadata_bytes(&self) -> u64 {
        let payloads = self.payloads.as_ref().map_or(0, |run| {
            size_of::<RightRun<Vec<Option<RowRef>>>>()
                + run.times.capacity() * size_of::<i64>()
                + run.sequences.allocation_bytes()
                + run.values.capacity() * size_of::<Option<RowRef>>()
        });
        let identities = self.identities.times.capacity() * size_of::<i64>()
            + self.identities.sequences.allocation_bytes();
        let general = if self.general_identities.is_empty() {
            0
        } else {
            256 + self.general_identities.len() * 512
        };
        (payloads + identities + general) as u64
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_sequence_kind(kind: SequenceKind) -> Self {
        Self {
            identities: RightRun::with_capacity(0, kind),
            ..Self::default()
        }
    }

    pub fn reserve_payloads(&mut self, additional: usize) {
        if additional == 0 {
            return;
        }
        match self.payloads.as_mut() {
            Some(run) => {
                // Keep Vec's amortized growth for repeated small admissions.
                // A fresh bucket still allocates only the accepted batch size.
                run.times.reserve(additional);
                run.sequences.reserve(additional);
                run.values.reserve(additional);
            }
            None => {
                self.payloads = Some(Box::new(RightRun::with_capacity(
                    additional,
                    self.identities.sequences.kind(),
                )));
            }
        }
    }

    /// Admission already rejected every live identity collision. Do not
    /// search or compact the identity-only histories again for each new row.
    pub fn insert_admitted(&mut self, order: RightOrder, payload: RowRef) {
        self.payloads
            .as_mut()
            .expect("reserved ASOF payload columns")
            .insert(order, payload);
    }

    fn payloads(&self) -> &RightRun<Vec<Option<RowRef>>> {
        self.payloads.as_deref().unwrap_or(&EMPTY_PAYLOADS)
    }

    pub fn len(&self) -> usize {
        self.payloads().len() + self.identities.len() + self.general_identities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> RightRows<'_> {
        let mut general = self.general_identities.iter();
        let next_general = general.next().map(order_ref);
        RightRows {
            bucket: self,
            payload: self.payloads().head,
            identity: self.identities.head,
            general,
            next_general,
        }
    }

    pub fn checkpoint_rows(&self) -> impl Iterator<Item = (OrderRef<'_>, Option<&RowRef>, u8)> {
        let mut rows = self.iter();
        std::iter::from_fn(move || rows.next_tagged())
    }

    #[cfg(test)]
    pub fn values(&self) -> impl Iterator<Item = Option<&RowRef>> {
        self.iter().map(|(_, payload)| payload)
    }

    #[cfg(test)]
    pub fn capacity(&self) -> usize {
        self.payloads()
            .times
            .capacity()
            .max(self.payloads().sequences.capacity())
            .max(self.payloads().values.capacity())
            + self
                .identities
                .times
                .capacity()
                .max(self.identities.sequences.capacity())
    }

    pub fn keys(&self) -> impl Iterator<Item = OrderRef<'_>> {
        self.iter().map(|(order, _)| order)
    }

    pub fn last_key_value(&self) -> Option<RightRow<'_>> {
        let identity = later_identity(
            self.identities.last().map(|(order, ())| order),
            self.general_identities.last().map(order_ref),
        );
        merge_row(self.payloads().last(), identity.map(|order| (order, &())))
    }

    pub fn contains_key(&self, order: &RightOrder) -> bool {
        self.payloads().locate(order).is_ok()
            || self.identities.locate(order).is_ok()
            || self.general_identities.contains(order)
    }

    pub fn insert(&mut self, order: RightOrder, payload: Option<RowRef>) {
        if let Some(payload) = payload {
            self.identities.remove(&order);
            self.general_identities.remove(&order);
            self.compact_identity_storage();
            let kind = self.identities.sequences.kind();
            self.payloads
                .get_or_insert_with(|| Box::new(RightRun::with_capacity(0, kind)))
                .insert(order, payload);
        } else {
            if let Some(payloads) = self.payloads.as_mut() {
                payloads.remove(&order);
                if payloads.len() == 0 {
                    self.payloads = None;
                }
            }
            insert_identity(&mut self.identities, &mut self.general_identities, order);
        }
    }

    pub fn candidate(&self, time: i64, tolerance: u64) -> Option<&RowRef> {
        let payloads = self.payloads();
        let payload = payloads.head + payloads.prefix_len(|value| value <= time);
        if self.identities.len() == 0 && self.general_identities.is_empty() {
            return self.payload_candidate(payload.checked_sub(1)?, time, tolerance);
        }
        let identity = self.identities.head + self.identities.prefix_len(|value| value <= time);
        let identity = later_identity(
            identity
                .checked_sub(1)
                .and_then(|index| self.identities.at(index))
                .map(|(order, ())| order),
            self.general_candidate(time),
        );
        bounded_candidate(
            merge_row(
                payload.checked_sub(1).and_then(|index| payloads.at(index)),
                identity.map(|order| (order, &())),
            ),
            time,
            tolerance,
        )
    }

    /// A sorted payload column already places the greatest sequence last at
    /// each time. With no identity-only candidates, reconstructing sequence
    /// encodings cannot affect the selected row.
    fn payload_candidate(&self, index: usize, time: i64, tolerance: u64) -> Option<&RowRef> {
        let payloads = self.payloads();
        if index < payloads.head {
            return None;
        }
        let right_time = *payloads.times.get(index)?;
        (i128::from(right_time) >= i128::from(time) - i128::from(tolerance))
            .then(|| payloads.values.get(index))
    }

    pub fn cursor_at(&self, time: i64) -> RightCursor {
        RightCursor {
            payload: self.payloads().head + self.payloads().prefix_len(|value| value < time),
            identity: self.identities.head + self.identities.prefix_len(|value| value < time),
        }
    }

    pub fn candidate_monotonic(
        &self,
        time: i64,
        tolerance: u64,
        next: &mut RightCursor,
    ) -> Option<&RowRef> {
        if self.identities.len() == 0 && self.general_identities.is_empty() {
            let index = self.payloads().advance_index(time, &mut next.payload)?;
            return self.payload_candidate(index, time, tolerance);
        }
        let identity = later_identity(
            self.identities
                .advance(time, &mut next.identity)
                .map(|(order, ())| order),
            self.general_candidate(time),
        );
        bounded_candidate(
            merge_row(
                self.payloads().advance(time, &mut next.payload),
                identity.map(|order| (order, &())),
            ),
            time,
            tolerance,
        )
    }

    pub fn expired_rows<'a>(
        &'a self,
        status: &'a super::super::StreamAsofJoinStatus,
        tolerance: u64,
        threshold: i128,
    ) -> impl Iterator<Item = ExpiredRow<'a>> {
        let payloads = self.payloads();
        let payload_count =
            payloads.prefix_len(|time| super::payload_expired(time, tolerance, threshold));
        let identity_count = self
            .identities
            .prefix_len(|time| super::identity_expired(time, status));
        let payloads = (payloads.head..payloads.head + payload_count).map(|index| {
            (
                payloads.times[index],
                payloads.sequences.owner_encoding(index),
                Some(payloads.values.get(index)),
            )
        });
        let identities =
            (self.identities.head..self.identities.head + identity_count).map(|index| {
                (
                    self.identities.times[index],
                    self.identities.sequences.owner_encoding(index),
                    None,
                )
            });
        let general = self
            .general_identities
            .iter()
            .take_while(move |order| super::identity_expired(order.0, status))
            .map(|order| (order.0, Some(&order.1), None));
        payloads.chain(identities).chain(general)
    }

    pub fn evict(
        &mut self,
        status: &super::super::StreamAsofJoinStatus,
        tolerance: u64,
        threshold: i128,
    ) -> u64 {
        let identity_count = self
            .identities
            .prefix_len(|time| super::identity_expired(time, status));
        self.identities.drop_prefix(identity_count);
        while self
            .general_identities
            .first()
            .is_some_and(|order| super::identity_expired(order.0, status))
        {
            self.general_identities.pop_first();
        }
        let Some(payloads) = self.payloads.as_mut() else {
            self.compact_identity_storage();
            return 0;
        };
        let payload_count =
            payloads.prefix_len(|time| super::payload_expired(time, tolerance, threshold));
        let identities = &mut self.identities;
        let general = &mut self.general_identities;
        let end = payloads.head + payload_count;
        for index in payloads.head..end {
            let time = payloads.times[index];
            payloads.values.take(index);
            if super::identity_expired(time, status) {
                payloads.sequences.take_owner(index);
            } else {
                let sequence = payloads.sequences.take(index);
                insert_identity(identities, general, (time, sequence));
            }
        }
        payloads.head = end;
        payloads.compact();
        if payloads.len() == 0 {
            self.payloads = None;
        }
        self.compact_identity_storage();
        payload_count as u64
    }

    pub fn payload_min(&self) -> Option<i64> {
        self.payloads().times.get(self.payloads().head).copied()
    }

    pub fn identity_min(&self) -> Option<i64> {
        let ordered = self.identities.times.get(self.identities.head).copied();
        match (
            ordered,
            self.general_identities.first().map(|order| order.0),
        ) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (left, right) => left.or(right),
        }
    }

    fn general_candidate(&self, time: i64) -> Option<OrderRef<'_>> {
        let row = if time == i64::MAX {
            self.general_identities.last()
        } else {
            self.general_identities
                .range(..(time + 1, Encoding::from_slice(&[])))
                .next_back()
        };
        row.map(order_ref)
    }

    fn compact_identity_storage(&mut self) {
        // A lone tree entry's minimum node allocation exceeds the compact
        // column charge. Moving just that row back to an empty column is O(1).
        if self.identities.len() == 0 && self.general_identities.len() == 1 {
            let order = self
                .general_identities
                .pop_first()
                .expect("one ASOF identity");
            self.identities.insert(order, ());
        }
        if self.general_identities.is_empty() {
            self.general_identities = BTreeSet::new();
        }
        self.identities.compact();
    }
}

fn grow_column_capacities(capacities: &mut [usize], required: usize) {
    for capacity in capacities {
        while *capacity < required {
            *capacity = (*capacity * 2).max(4);
        }
    }
}

fn projected_eviction_metadata(
    payload_rows: usize,
    payload_capacities: [usize; 3],
    identity_capacities: [usize; 3],
    width: usize,
    general: usize,
) -> u64 {
    let payload_bytes = if payload_rows == 0 {
        0
    } else {
        size_of::<RightRun<Vec<Option<RowRef>>>>() as u64 + columns_bytes(payload_capacities, width)
    };
    let general_bytes = if general == 0 {
        0
    } else {
        256 + general as u64 * 512
    };
    payload_bytes + columns_bytes(identity_capacities, width) + general_bytes
}

fn columns_bytes(capacities: [usize; 3], sequence_bytes: usize) -> u64 {
    (capacities[0] * 8 + capacities[1] * sequence_bytes + capacities[2] * 8) as u64
}

fn order_ref(order: &RightOrder) -> OrderRef<'_> {
    (&order.0, Cow::Borrowed(&order.1))
}

fn later_identity<'a>(
    left: Option<OrderRef<'a>>,
    right: Option<OrderRef<'a>>,
) -> Option<OrderRef<'a>> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (left, right) => left.or(right),
    }
}

fn insert_identity(run: &mut RightRun<()>, general: &mut BTreeSet<RightOrder>, order: RightOrder) {
    if general.contains(&order) {
        return;
    }
    if run.last().is_none_or(|(last, ())| last < order_ref(&order)) || run.locate(&order).is_ok() {
        run.insert(order, ());
    } else {
        general.insert(order);
    }
}

fn merge_row<'a>(
    payload: Option<(OrderRef<'a>, &'a RowRef)>,
    identity: Option<(OrderRef<'a>, &'a ())>,
) -> Option<RightRow<'a>> {
    match (payload, identity) {
        (Some((order, payload)), Some((other, ()))) if order > other => {
            Some((order, Some(payload)))
        }
        (_, Some((order, ()))) => Some((order, None)),
        (Some((order, payload)), None) => Some((order, Some(payload))),
        (None, None) => None,
    }
}

fn bounded_candidate(row: Option<RightRow<'_>>, time: i64, tolerance: u64) -> Option<&RowRef> {
    let ((right_time, _), payload) = row?;
    (i128::from(*right_time) >= i128::from(time) - i128::from(tolerance))
        .then_some(payload)
        .flatten()
}

pub(in super::super) struct RightRows<'a> {
    bucket: &'a RightBucket,
    payload: usize,
    identity: usize,
    general: btree_set::Iter<'a, RightOrder>,
    next_general: Option<OrderRef<'a>>,
}

impl<'a> RightRows<'a> {
    fn next_tagged(&mut self) -> Option<(OrderRef<'a>, Option<&'a RowRef>, u8)> {
        let payload = self.bucket.payloads().at(self.payload);
        let ordered = self
            .bucket
            .identities
            .at(self.identity)
            .map(|(order, ())| order);
        let (identity, in_general) = earlier_identity(ordered, self.next_general.clone());
        let order = match (&payload, &identity) {
            (Some((left, _)), Some(right)) => left.cmp(right),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => return None,
        };
        if order == Ordering::Less {
            self.payload += 1;
            payload.map(|(order, row)| (order, Some(row), 1))
        } else {
            if in_general {
                self.next_general = self.general.next().map(order_ref);
            } else {
                self.identity += 1;
            }
            identity.map(|order| (order, None, if in_general { 2 } else { 0 }))
        }
    }
}

impl<'a> Iterator for RightRows<'a> {
    type Item = RightRow<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_tagged().map(|(order, row, _)| (order, row))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.bucket.payloads().times.len() - self.payload
            + self.bucket.identities.times.len()
            - self.identity
            + self.general.len()
            + usize::from(self.next_general.is_some());
        (remaining, Some(remaining))
    }
}

fn earlier_identity<'a>(
    left: Option<OrderRef<'a>>,
    right: Option<OrderRef<'a>>,
) -> (Option<OrderRef<'a>>, bool) {
    match (left, right) {
        (Some(left), Some(right)) if left < right => (Some(left), false),
        (_, Some(right)) => (Some(right), true),
        (left, None) => (left, false),
    }
}

impl ExactSizeIterator for RightRows<'_> {}

impl<'a> IntoIterator for &'a RightBucket {
    type Item = RightRow<'a>;
    type IntoIter = RightRows<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl FromIterator<(RightOrder, Option<RowRef>)> for RightBucket {
    fn from_iter<T: IntoIterator<Item = (RightOrder, Option<RowRef>)>>(iter: T) -> Self {
        let mut bucket = Self::new();
        for (order, payload) in iter {
            bucket.insert(order, payload);
        }
        bucket
    }
}
