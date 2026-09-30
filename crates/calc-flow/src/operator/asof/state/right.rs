use super::{BatchKey, Encoding, PayloadBatch, RightOrder, RowPayload};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, btree_set},
    sync::Arc,
};

type OrderRef<'a> = (&'a i64, &'a Encoding);
type RightRow<'a> = (OrderRef<'a>, Option<&'a RowPayload>);

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
    payloads: Option<Box<RightRun<Vec<Option<RowPayload>>>>>,
    identities: RightRun<()>,
    // Older payloads can expire after later identity-only history was retained.
    // A tree absorbs those rows without shifting the whole identity column.
    general_identities: BTreeSet<RightOrder>,
}

static EMPTY_PAYLOADS: RightRun<Vec<Option<RowPayload>>> = RightRun {
    times: Vec::new(),
    sequences: Vec::new(),
    values: Vec::new(),
    head: 0,
};

#[derive(Clone)]
struct RightRun<V> {
    times: Vec<i64>,
    sequences: Vec<Encoding>,
    values: V,
    head: usize,
}

trait RunValues: Clone {
    type Value;

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

impl RunValues for Vec<Option<RowPayload>> {
    type Value = RowPayload;

    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }

    fn get(&self, index: usize) -> &RowPayload {
        self[index].as_ref().expect("live ASOF run value")
    }

    fn push(&mut self, value: RowPayload) {
        Vec::push(self, Some(value));
    }

    fn insert(&mut self, index: usize, value: RowPayload) {
        Vec::insert(self, index, Some(value));
    }

    fn replace(&mut self, index: usize, value: RowPayload) {
        self[index] = Some(value);
    }

    fn remove(&mut self, index: usize) {
        Vec::remove(self, index);
    }

    fn take(&mut self, index: usize) -> RowPayload {
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

impl<V: RunValues> Default for RightRun<V> {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl<V: RunValues> RightRun<V> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            times: Vec::with_capacity(capacity),
            sequences: Vec::with_capacity(capacity),
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
            (self.times.get(index)?, &self.sequences[index]),
            self.values.get(index),
        ))
    }

    fn last(&self) -> Option<(OrderRef<'_>, &V::Value)> {
        self.at(self.times.len().checked_sub(1)?)
    }

    fn locate(&self, order: &RightOrder) -> Result<usize, usize> {
        let start = self.times[self.head..].partition_point(|time| *time < order.0) + self.head;
        let end = self.times[self.head..].partition_point(|time| *time <= order.0) + self.head;
        self.sequences[start..end]
            .binary_search(&order.1)
            .map(|index| start + index)
            .map_err(|index| start + index)
    }

    fn insert(&mut self, order: RightOrder, value: V::Value) {
        if self.times.capacity() == 0 {
            // Avoid Vec's four-element minimum for sparse buckets: even one
            // identity must fit the existing committed state charge.
            self.times = Vec::with_capacity(1);
            self.sequences = Vec::with_capacity(1);
            self.values = V::with_capacity(1);
        }
        if self
            .last()
            .is_none_or(|(last, _)| last < (&order.0, &order.1))
        {
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

    fn take_prefix(&mut self, count: usize, mut take: impl FnMut(i64, Encoding, V::Value)) {
        let end = self.head + count;
        for index in self.head..end {
            let sequence = std::mem::replace(&mut self.sequences[index], Encoding::from_slice(&[]));
            let value = self.values.take(index);
            take(self.times[index], sequence, value);
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
        self.sequences = self
            .sequences
            .split_off(self.head)
            .into_boxed_slice()
            .into_vec();
        self.values.compact(self.head);
        self.head = 0;
    }

    fn advance(&self, time: i64, next: &mut usize) -> Option<(OrderRef<'_>, &V::Value)> {
        *next = (*next).max(self.head);
        while self.times.get(*next).is_some_and(|value| *value <= time) {
            *next += 1;
        }
        self.at(next.checked_sub(1)?)
    }
}

#[derive(Default)]
pub(in super::super) struct RightCursor {
    payload: usize,
    identity: usize,
}

impl RightBucket {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reserve_payloads(&mut self, additional: usize) {
        if additional == 0 {
            return;
        }
        match self.payloads.as_mut() {
            Some(run) => {
                run.times.reserve_exact(additional);
                run.sequences.reserve_exact(additional);
                run.values.reserve_exact(additional);
            }
            None => self.payloads = Some(Box::new(RightRun::with_capacity(additional))),
        }
    }

    /// Admission already rejected every live identity collision. Do not
    /// search or compact the identity-only histories again for each new row.
    pub fn insert_admitted(&mut self, order: RightOrder, payload: RowPayload) {
        self.payloads
            .as_mut()
            .expect("reserved ASOF payload columns")
            .insert(order, payload);
    }

    fn payloads(&self) -> &RightRun<Vec<Option<RowPayload>>> {
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

    #[cfg(test)]
    pub fn values(&self) -> impl Iterator<Item = Option<&RowPayload>> {
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

    pub fn insert(&mut self, order: RightOrder, payload: Option<RowPayload>) {
        if let Some(payload) = payload {
            self.identities.remove(&order);
            self.general_identities.remove(&order);
            self.compact_identity_storage();
            self.payloads
                .get_or_insert_with(Box::default)
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

    pub fn candidate(&self, time: i64, tolerance: u64) -> Option<&RowPayload> {
        let payloads = self.payloads();
        let payload = payloads.head + payloads.prefix_len(|value| value <= time);
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
    ) -> Option<&RowPayload> {
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
    ) -> impl Iterator<Item = RightRow<'a>> {
        let payloads = self.payloads();
        let payload_count =
            payloads.prefix_len(|time| super::payload_expired(time, tolerance, threshold));
        let identity_count = self
            .identities
            .prefix_len(|time| super::identity_expired(time, status));
        let payloads = (payloads.head..payloads.head + payload_count)
            .map(|index| payloads.at(index).expect("expired ASOF payload"))
            .map(|(order, row)| (order, Some(row)));
        let identities = (self.identities.head..self.identities.head + identity_count)
            .map(|index| self.identities.at(index).expect("expired ASOF identity"))
            .map(|(order, ())| (order, None));
        let general = self
            .general_identities
            .iter()
            .take_while(move |order| super::identity_expired(order.0, status))
            .map(|order| (order_ref(order), None));
        payloads.chain(identities).chain(general)
    }

    pub fn evict(
        &mut self,
        status: &super::super::StreamAsofJoinStatus,
        tolerance: u64,
        threshold: i128,
        batches: &mut BTreeMap<BatchKey, (Arc<PayloadBatch>, usize)>,
    ) -> u64 {
        let identity_count = self
            .identities
            .prefix_len(|time| super::identity_expired(time, status));
        self.identities.take_prefix(identity_count, |_, _, ()| {});
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
        payloads.take_prefix(payload_count, |time, sequence, payload| {
            super::detach_batch(batches, &payload);
            if !super::identity_expired(time, status) {
                insert_identity(identities, general, (time, sequence));
            }
        });
        payloads.compact();
        if payloads.len() == 0 {
            self.payloads = None;
        }
        self.compact_identity_storage();
        payload_count as u64
    }

    pub fn payload_min(&self) -> Option<i64> {
        self.payloads()
            .at(self.payloads().head)
            .map(|(order, _)| *order.0)
    }

    pub fn identity_min(&self) -> Option<i64> {
        let ordered = self
            .identities
            .at(self.identities.head)
            .map(|(order, ())| *order.0);
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
        // A lone tree entry's minimum node allocation would exceed its v2
        // row charge. Moving just that row back to an empty column is O(1).
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

fn order_ref(order: &RightOrder) -> OrderRef<'_> {
    (&order.0, &order.1)
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
    if run
        .last()
        .is_none_or(|(last, ())| last < (&order.0, &order.1))
        || run.locate(&order).is_ok()
    {
        run.insert(order, ());
    } else {
        general.insert(order);
    }
}

fn merge_row<'a>(
    payload: Option<(OrderRef<'a>, &'a RowPayload)>,
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

fn bounded_candidate(row: Option<RightRow<'_>>, time: i64, tolerance: u64) -> Option<&RowPayload> {
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

impl<'a> Iterator for RightRows<'a> {
    type Item = RightRow<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let payload = self.bucket.payloads().at(self.payload);
        let ordered = self
            .bucket
            .identities
            .at(self.identity)
            .map(|(order, ())| order);
        let (identity, in_general) = earlier_identity(ordered, self.next_general);
        let order = match (payload, identity) {
            (Some((left, _)), Some(right)) => left.cmp(&right),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => return None,
        };
        if order == Ordering::Less {
            self.payload += 1;
            payload.map(|(order, row)| (order, Some(row)))
        } else {
            if in_general {
                self.next_general = self.general.next().map(order_ref);
            } else {
                self.identity += 1;
            }
            identity.map(|order| (order, None))
        }
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

impl FromIterator<(RightOrder, Option<RowPayload>)> for RightBucket {
    fn from_iter<T: IntoIterator<Item = (RightOrder, Option<RowPayload>)>>(iter: T) -> Self {
        let mut bucket = Self::new();
        for (order, payload) in iter {
            bucket.insert(order, payload);
        }
        bucket
    }
}
