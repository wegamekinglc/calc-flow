use std::{any::Any, collections::BTreeMap, sync::Arc};

use parking_lot::Mutex;

use super::super::{Result, StateBudget, operator_error};
use super::footprint::{Charges, Footprint};
use crate::StateSegment;

type Key = (usize, usize);
mod fixed;

#[derive(Default)]
struct Pool {
    total: u64,
    segments: BTreeMap<Key, SegmentEntry>,
    committed: Arc<Charges>,
    working: BTreeMap<usize, Work>,
    limit: StateBudget,
    fixed: Option<fixed::Empty>,
}
struct Work {
    _identity: Arc<()>,
    source: Arc<Charges>,
    footprint: Charges,
    reserved: u64,
}

pub(in super::super) struct WorkCredit {
    live: Arc<Mutex<Pool>>,
    identity: Option<usize>,
    committed: Arc<Mutex<Arc<Footprint>>>,
    source: Option<Arc<Footprint>>,
    pins: Footprint,
}

pub(in super::super) struct Commit {
    footprint: Arc<Footprint>,
    charges: Arc<Charges>,
}

impl Commit {
    pub(super) fn new(footprint: Footprint) -> Self {
        Self {
            charges: Arc::new(footprint.charges()),
            footprint: Arc::new(footprint),
        }
    }
}

struct SegmentEntry {
    _bytes: Arc<Vec<u8>>,
    _checksum: Arc<String>,
    footprint: Charges,
    groups: u64,
}

#[derive(Clone, Default)]
pub(in super::super) struct Resources(
    Arc<Mutex<Pool>>,
    Arc<crate::array::RollingWorkers>,
    Arc<Mutex<Arc<Footprint>>>,
    Arc<Mutex<crate::CancellationToken>>,
);

pub(in super::super) struct Credit {
    live: Arc<Mutex<Pool>>,
    bytes: u64,
}

struct CaptureOwner {
    _previous: Option<Arc<dyn Any + Send + Sync>>,
    live: Arc<Mutex<Pool>>,
    key: Key,
}

impl Resources {
    #[cfg(test)]
    pub(in super::super) fn hold_current_for_test(&self) -> WorkCredit {
        self.begin_work()
    }

    pub(in super::super) fn work_control(&self) -> crate::CancellationToken {
        self.3.lock().clone()
    }

    pub(in super::super) fn cancel_pending(&self) {
        let next = crate::CancellationToken::new();
        let previous = std::mem::replace(&mut *self.3.lock(), next);
        previous.cancel();
    }

    pub(in super::super) fn install_budget(
        &self,
        committed: u64,
        budget: StateBudget,
        node: &str,
    ) -> Result<()> {
        let mut pool = self.0.lock();
        admit_pool(&pool, committed, 0, budget, node)?;
        pool.limit = budget;
        Ok(())
    }

    pub(in super::super) fn workers(&self) -> Arc<crate::array::RollingWorkers> {
        Arc::clone(&self.1)
    }

    pub(super) fn reserve(
        &self,
        committed: u64,
        bytes: u64,
        budget: StateBudget,
        node: &str,
    ) -> Result<Credit> {
        let mut live = self.0.lock();
        admit_pool(&live, committed, bytes, budget, node)?;
        live.total = add(live.total, bytes, node)?;
        Ok(Credit {
            live: Arc::clone(&self.0),
            bytes,
        })
    }

    #[cfg(test)]
    pub(in super::super) fn check(
        &self,
        committed: u64,
        budget: StateBudget,
        node: &str,
    ) -> Result<()> {
        admit_pool(&self.0.lock(), committed, 0, budget, node)
    }

    pub(in super::super) fn retain(
        &self,
        segment: StateSegment,
        committed: u64,
        budget: StateBudget,
        node: &str,
    ) -> Result<StateSegment> {
        let (bytes, checksum) = segment.shared_allocations();
        let key = (
            Arc::as_ptr(&bytes) as usize,
            Arc::as_ptr(&checksum) as usize,
        );
        self.0
            .lock()
            .retain(key, bytes, checksum, (committed, budget), node)?;
        let pool = Arc::clone(&self.0);
        Ok(segment.map_owner(|previous| {
            Arc::new(CaptureOwner {
                _previous: previous,
                live: pool,
                key,
            })
        }))
    }
}

impl Pool {
    fn retain(
        &mut self,
        key: Key,
        bytes: Arc<Vec<u8>>,
        checksum: Arc<String>,
        admission: (u64, StateBudget),
        node: &str,
    ) -> Result<()> {
        if let Some(entry) = self.segments.get_mut(&key) {
            entry.groups = entry
                .groups
                .checked_add(1)
                .ok_or_else(|| operator_error(node, "rolling capture owner count overflowed"))?;
            return Ok(());
        }
        let footprint = segment_charges(&bytes, &checksum, node)?;
        self.segments.insert(
            key,
            SegmentEntry {
                _bytes: bytes,
                _checksum: checksum,
                footprint,
                groups: 1,
            },
        );
        if let Err(error) = admit_pool(self, admission.0, 0, admission.1, node) {
            self.segments.remove(&key);
            return Err(error);
        }
        Ok(())
    }
    fn release(&mut self, key: Key) {
        let entry = self
            .segments
            .get_mut(&key)
            .expect("live capture allocation");
        entry.groups = entry.groups.checked_sub(1).expect("live capture group");
        if entry.groups == 0 {
            self.segments
                .remove(&key)
                .expect("last live capture allocation");
        }
    }
}

impl Drop for CaptureOwner {
    fn drop(&mut self) {
        self.live.lock().release(self.key);
    }
}

impl Credit {
    pub(super) fn capture_segment(
        mut self,
        segment: StateSegment,
        committed: u64,
        budget: StateBudget,
        node: &str,
    ) -> Result<StateSegment> {
        let (bytes, checksum) = segment.shared_allocations();
        let key = (
            Arc::as_ptr(&bytes) as usize,
            Arc::as_ptr(&checksum) as usize,
        );
        self.install_capture(key, bytes, checksum, (committed, budget), node)?;
        let live = Arc::clone(&self.live);
        Ok(segment.map_owner(|previous| {
            Arc::new(CaptureOwner {
                _previous: previous,
                live,
                key,
            })
        }))
    }

    fn install_capture(
        &mut self,
        key: Key,
        bytes: Arc<Vec<u8>>,
        checksum: Arc<String>,
        admission: (u64, StateBudget),
        node: &str,
    ) -> Result<()> {
        let mut live = self.live.lock();
        let previous = live.total;
        live.total = live
            .total
            .checked_sub(self.bytes)
            .expect("registered read reservation");
        if let Err(error) = live.retain(key, bytes, checksum, admission, node) {
            live.total = previous;
            return Err(error);
        }
        self.bytes = 0;
        Ok(())
    }

    pub(super) fn adjust(
        &mut self,
        bytes: u64,
        committed: u64,
        budget: StateBudget,
        node: &str,
    ) -> Result<()> {
        let mut live = self.live.lock();
        let others = live
            .total
            .checked_sub(self.bytes)
            .expect("credit belongs to its ledger");
        let previous = live.total;
        live.total = others;
        let admission = admit_pool(&live, committed, bytes, budget, node);
        live.total = previous;
        admission?;
        live.total = add(others, bytes, node)?;
        self.bytes = bytes;
        Ok(())
    }
}
impl Drop for Credit {
    fn drop(&mut self) {
        let mut live = self.live.lock();
        live.total = live
            .total
            .checked_sub(self.bytes)
            .expect("live rolling credit remains reserved");
    }
}
fn admit_pool(
    pool: &Pool,
    committed: u64,
    bytes: u64,
    budget: StateBudget,
    node: &str,
) -> Result<()> {
    let mut footprint = pool_base(pool, committed, bytes, node)?;
    merge_segments(&mut footprint, pool, node)?;
    for work in pool.working.values() {
        merge_work(&mut footprint, work, node)?;
    }
    let total = footprint.bytes(node)?;
    let limit = budget.max_bytes().min(pool.limit.max_bytes());
    if total > limit {
        return Err(operator_error(
            node,
            &format!("rolling state budget peak requires {total} bytes; limit is {limit}"),
        ));
    }
    Ok(())
}

fn segment_charges(bytes: &Vec<u8>, checksum: &String, node: &str) -> Result<Charges> {
    let mut footprint = Charges::default();
    footprint.buffer(bytes.as_ptr() as usize, size(bytes.capacity(), node)?);
    footprint.buffer(checksum.as_ptr() as usize, size(checksum.capacity(), node)?);
    Ok(footprint)
}

fn merge_segments(footprint: &mut Charges, pool: &Pool, node: &str) -> Result<()> {
    for segment in pool.segments.values() {
        footprint.merge(&segment.footprint, node)?;
    }
    Ok(())
}

fn pool_base(pool: &Pool, committed: u64, bytes: u64, node: &str) -> Result<Charges> {
    let mut footprint = (*pool.committed).clone();
    let unmatched = committed.saturating_sub(footprint.bytes(node)?);
    footprint.add_owned(add(add(pool.total, bytes, node)?, unmatched, node)?, node)?;
    if let Some(fixed) = &pool.fixed {
        footprint.merge(fixed.charges(), node)?;
    }
    Ok(footprint)
}

fn merge_work(footprint: &mut Charges, work: &Work, node: &str) -> Result<()> {
    footprint.merge(&work.source, node)?;
    footprint.merge(&work.footprint, node)?;
    footprint.add_owned(work.reserved, node)
}

impl Resources {
    pub(super) fn begin_work(&self) -> WorkCredit {
        let source = Arc::clone(&self.2.lock());
        let identity = Arc::new(());
        let key = Arc::as_ptr(&identity) as usize;
        let mut live = self.0.lock();
        let charges = Arc::clone(&live.committed);
        live.working.insert(
            key,
            Work {
                _identity: identity,
                source: charges,
                footprint: Charges::default(),
                reserved: 0,
            },
        );
        WorkCredit {
            live: Arc::clone(&self.0),
            identity: Some(key),
            committed: Arc::clone(&self.2),
            source: Some(source),
            pins: Footprint::default(),
        }
    }

    #[cfg(test)]
    pub(super) fn reserve_work(
        &self,
        committed: Footprint,
        candidate: Footprint,
        reserved: u64,
        budget: StateBudget,
        node: &str,
    ) -> Result<WorkCredit> {
        let source = Arc::new(committed);
        let identity = Arc::new(());
        let key = Arc::as_ptr(&identity) as usize;
        let charges = Arc::new(source.charges());
        let mut live = self.0.lock();
        let previous = std::mem::replace(&mut live.committed, Arc::clone(&charges));
        live.working.insert(
            key,
            Work {
                _identity: identity,
                source: charges,
                footprint: candidate.charges(),
                reserved,
            },
        );
        if let Err(error) = admit_pool(&live, 0, 0, budget, node) {
            live.working.remove(&key);
            live.committed = previous;
            return Err(error);
        }
        drop(live);
        replace_pins(&self.2, Arc::clone(&source));
        Ok(WorkCredit {
            live: Arc::clone(&self.0),
            identity: Some(key),
            committed: Arc::clone(&self.2),
            source: Some(source),
            pins: candidate,
        })
    }

    #[cfg(test)]
    pub(in super::super) fn clear_committed(&self) {
        replace_pins(&self.2, Arc::default());
        self.0.lock().committed = Arc::default();
    }
}

impl WorkCredit {
    pub(super) fn source(&self) -> &Footprint {
        self.source.as_deref().expect("active source credit")
    }

    pub(super) fn adjust(
        &mut self,
        footprint: Footprint,
        reserved: u64,
        budget: StateBudget,
        node: &str,
    ) -> Result<()> {
        let charges = footprint.charges();
        self.admit_transition(charges.clone(), reserved, budget, node)?;
        let previous = std::mem::replace(&mut self.pins, footprint);
        drop(previous);
        let key = self.identity.expect("active work credit");
        self.live
            .lock()
            .working
            .get_mut(&key)
            .expect("registered work")
            .footprint = charges;
        Ok(())
    }

    fn admit_transition(
        &self,
        mut charges: Charges,
        reserved: u64,
        budget: StateBudget,
        node: &str,
    ) -> Result<()> {
        let key = self.identity.expect("active work credit");
        let mut live = self.live.lock();
        let work = live.working.get_mut(&key).expect("registered work");
        // Pinned owners overlap; the new quote includes all anonymous live scratch.
        charges.merge_owners(&work.footprint);
        let previous = std::mem::replace(&mut work.footprint, charges);
        let previous_reserved = std::mem::replace(&mut work.reserved, reserved);
        if let Err(error) = admit_pool(&live, 0, 0, budget, node) {
            let work = live.working.get_mut(&key).expect("registered work");
            work.footprint = previous;
            work.reserved = previous_reserved;
            return Err(error);
        }
        Ok(())
    }

    pub(in super::super) fn commit(mut self, prepared: Commit) {
        replace_pins(&self.committed, prepared.footprint);
        self.release_pins();
        let mut live = self.live.lock();
        live.committed = prepared.charges;
        let key = self.identity.take().expect("active work credit");
        live.working.remove(&key).expect("registered work");
    }

    fn release_pins(&mut self) {
        let source = self.source.take();
        let candidate = std::mem::take(&mut self.pins);
        drop(candidate);
        drop(source);
    }
}

fn replace_pins(pins: &Mutex<Arc<Footprint>>, footprint: Arc<Footprint>) {
    let previous = std::mem::replace(&mut *pins.lock(), footprint);
    drop(previous);
}

impl Drop for WorkCredit {
    fn drop(&mut self) {
        self.release_pins();
        if let Some(key) = self.identity.take() {
            self.live
                .lock()
                .working
                .remove(&key)
                .expect("registered work");
        }
    }
}

pub(super) fn add(left: u64, right: u64, node: &str) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| operator_error(node, "rolling state budget byte count overflowed"))
}
pub(super) fn size(bytes: usize, node: &str) -> Result<u64> {
    u64::try_from(bytes)
        .map_err(|_| operator_error(node, "rolling state budget byte count overflowed"))
}

#[cfg(test)]
mod footprint_tests {
    use super::*;
    use datafusion::arrow::buffer::Buffer;

    #[derive(Clone, Copy)]
    enum Refund {
        Drop,
        Commit,
        Clear,
        Adjust,
    }

    fn assert_work_refund_after_payload_drop(action: Refund) {
        type Bytes = <datafusion::object_store::PutPayload as IntoIterator>::Item;
        struct Payload {
            bytes: Vec<u8>,
            entered: std::sync::mpsc::SyncSender<()>,
            release: std::sync::Mutex<std::sync::mpsc::Receiver<()>>,
        }
        impl AsRef<[u8]> for Payload {
            fn as_ref(&self) -> &[u8] {
                &self.bytes
            }
        }
        impl Drop for Payload {
            fn drop(&mut self) {
                let _ = self.entered.send(());
                let _ = self.release.get_mut().unwrap().recv();
            }
        }
        let (entered, observed) = std::sync::mpsc::sync_channel(0);
        let (release, resumed) = std::sync::mpsc::channel();
        let buffer = Buffer::from(Bytes::from_owner(Payload {
            bytes: vec![7; 1024],
            entered,
            release: std::sync::Mutex::new(resumed),
        }));
        let mut footprint = Footprint::default();
        footprint.buffer(&buffer, "features").unwrap();
        let resources = Resources::default();
        let (committed, candidate) = if matches!(action, Refund::Commit | Refund::Clear) {
            (footprint, Footprint::default())
        } else {
            (Footprint::default(), footprint)
        };
        let work = resources
            .reserve_work(
                committed,
                candidate,
                0,
                StateBudget::new(10, 1024).unwrap(),
                "features",
            )
            .unwrap();
        drop(buffer);
        let owner_resources = resources.clone();
        let owner = std::thread::spawn(move || match action {
            Refund::Commit => work.commit(Commit::new(Footprint::default())),
            Refund::Drop => drop(work),
            Refund::Clear => {
                drop(work);
                owner_resources.clear_committed();
            }
            Refund::Adjust => {
                let mut work = work;
                work.adjust(
                    Footprint::default(),
                    0,
                    StateBudget::new(10, 1024).unwrap(),
                    "features",
                )
                .unwrap();
            }
        });
        observed
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        let admission = resources.check(0, StateBudget::new(10, 1023).unwrap(), "features");
        release.send(()).unwrap();
        owner.join().unwrap();
        assert!(
            admission.is_err(),
            "payload remains live while its destructor is blocked"
        );
        resources
            .check(0, StateBudget::new(10, 1).unwrap(), "features")
            .unwrap();
    }

    #[test]
    fn test_work_refund_waits_for_the_last_payload_owner_to_finish_drop() {
        assert_work_refund_after_payload_drop(Refund::Drop);
    }

    #[test]
    fn test_prepared_commit_refund_waits_for_old_payload_drop() {
        assert_work_refund_after_payload_drop(Refund::Commit);
    }

    #[test]
    fn test_reset_refund_waits_for_old_committed_payload_drop() {
        assert_work_refund_after_payload_drop(Refund::Clear);
    }

    #[test]
    fn test_candidate_adjust_refund_waits_for_old_payload_drop() {
        assert_work_refund_after_payload_drop(Refund::Adjust);
    }

    #[test]
    fn test_candidate_adjust_admits_the_old_and_new_owner_overlap() {
        let old_buffer = Buffer::from_vec(vec![7_u8; 1024]);
        let new_buffer = Buffer::from_vec(vec![8_u8; 1024]);
        let mut old = Footprint::default();
        old.buffer(&old_buffer, "features").unwrap();
        let mut new = Footprint::default();
        new.buffer(&new_buffer, "features").unwrap();
        let resources = Resources::default();
        let budget = StateBudget::new(10, 1536).unwrap();
        let mut work = resources
            .reserve_work(Footprint::default(), old, 0, budget, "features")
            .unwrap();
        assert!(
            work.adjust(new, 0, budget, "features").is_err(),
            "replacement cannot adopt a second allocation before the first is released"
        );
        resources
            .check(0, StateBudget::new(10, 1024).unwrap(), "features")
            .unwrap();
        assert!(
            resources
                .check(0, StateBudget::new(10, 1023).unwrap(), "features")
                .is_err()
        );
    }

    #[test]
    fn test_shared_state_packet_stays_charged_once_until_abandoned_work_settles() {
        use super::super::super::{BufferedRow, RollingStreamState, ScalarValue};
        let mut text = String::with_capacity(1024);
        text.push('x');
        let row = BufferedRow::new(
            Vec::new(),
            Vec::new(),
            0,
            vec![ScalarValue::Utf8(Some(text))],
        );
        let mut state = RollingStreamState::default();
        state.buffer.insert(row.identity.clone(), row);
        let state = Arc::new(state);
        let alive = Arc::downgrade(&state);
        let (_, footprint) =
            super::super::footprint_state::packet(Arc::clone(&state), "features").unwrap();
        let expected = 256
            + size_of::<ScalarValue>() as u64
            + 1024
            + super::super::footprint_state::packet_allocation_bytes("features").unwrap();
        assert_eq!(footprint.bytes("features").unwrap(), expected);
        let resources = Resources::default();
        let work = resources
            .reserve_work(
                footprint.clone(),
                Footprint::default(),
                0,
                StateBudget::new(10, expected).unwrap(),
                "features",
            )
            .unwrap();
        resources.clear_committed();
        drop(footprint);
        drop(state);
        assert!(alive.upgrade().is_some());
        assert!(
            resources
                .check(0, StateBudget::new(10, expected - 1).unwrap(), "features")
                .is_err()
        );
        drop(work);
        assert!(alive.upgrade().is_none());
        resources
            .check(0, StateBudget::new(10, 1).unwrap(), "features")
            .unwrap();
    }

    #[test]
    fn test_queued_work_borrows_committed_owners_without_copying_the_registry() {
        let mut footprint = Footprint::default();
        for _ in 0..4096 {
            footprint
                .buffer(&Buffer::from_vec(vec![7_u8]), "features")
                .unwrap();
        }
        let resources = Resources::default();
        let budget = StateBudget::new(10, 4096).unwrap();
        resources
            .reserve_work(
                Footprint::default(),
                footprint.clone(),
                0,
                budget,
                "features",
            )
            .unwrap()
            .commit(Commit::new(footprint));
        let mut work = None;
        let measured = allocation_counter::measure(|| {
            work = Some(resources.begin_work());
        });
        assert!(
            measured.bytes_total < 4096,
            "queued work copied the committed registry: {measured:?}"
        );
        resources.clear_committed();
        assert!(
            resources
                .check(0, StateBudget::new(10, 4095).unwrap(), "features")
                .is_err()
        );
        drop(work);
        resources
            .check(0, StateBudget::new(10, 1).unwrap(), "features")
            .unwrap();
    }

    #[test]
    fn test_reset_cancels_old_work_control_without_cancelling_next_generation() {
        let resources = Resources::default();
        let previous = resources.work_control();
        resources.cancel_pending();
        assert!(previous.is_cancelled());
        assert!(!resources.work_control().is_cancelled());
    }

    #[test]
    fn test_budget_change_restricts_abandoned_work_future_reservation_growth() {
        let resources = Resources::default();
        let original = StateBudget::default();
        let mut work = resources.reserve(0, 1024, original, "actual_node").unwrap();
        let lower = StateBudget::new(10, 2048).unwrap();
        resources.install_budget(0, lower, "actual_node").unwrap();
        assert!(work.adjust(4096, 0, original, "actual_node").is_err());
        let invalid = StateBudget::new(10, 1023).unwrap();
        assert!(resources.install_budget(0, invalid, "actual_node").is_err());
        work.adjust(1500, 0, original, "actual_node").unwrap();
        drop(work);
        resources
            .check(0, StateBudget::new(10, 1).unwrap(), "actual_node")
            .unwrap();
    }

    fn captured_buffer(resources: &Resources, budget: StateBudget) -> Buffer {
        type Bytes = <datafusion::object_store::PutPayload as IntoIterator>::Item;
        struct SegmentBytes(StateSegment);
        impl AsRef<[u8]> for SegmentBytes {
            fn as_ref(&self) -> &[u8] {
                self.0.bytes()
            }
        }
        let captured = resources
            .retain(StateSegment::new(vec![7; 1024]), 0, budget, "features")
            .unwrap();
        Buffer::from(Bytes::from_owner(SegmentBytes(captured)))
    }

    #[test]
    fn test_committed_foreign_capture_owner_does_not_cycle_the_credit_pool() {
        let resources = Resources::default();
        let pool = Arc::downgrade(&resources.0);
        let budget = StateBudget::new(10, 4096).unwrap();
        let buffer = captured_buffer(&resources, budget);
        let mut footprint = Footprint::default();
        footprint.buffer(&buffer, "features").unwrap();
        let work = resources
            .reserve_work(
                Footprint::default(),
                footprint.clone(),
                0,
                budget,
                "features",
            )
            .unwrap();
        work.commit(Commit::new(footprint));
        drop(buffer);
        drop(resources);
        assert!(
            pool.upgrade().is_none(),
            "capture credit and Arrow owner form a cycle"
        );
    }

    #[test]
    fn test_capture_and_candidate_alias_charge_one_underlying_allocation() {
        let resources = Resources::default();
        let budget = StateBudget::new(10, 1088).unwrap();
        let buffer = captured_buffer(&resources, budget);
        let mut footprint = Footprint::default();
        footprint.buffer(&buffer, "features").unwrap();
        let work = resources
            .reserve_work(
                Footprint::default(),
                footprint.clone(),
                0,
                budget,
                "features",
            )
            .unwrap();
        work.commit(Commit::new(footprint));
        resources.check(0, budget, "features").unwrap();
        resources.clear_committed();
        drop(buffer);
        resources
            .check(0, StateBudget::new(10, 1).unwrap(), "features")
            .unwrap();
    }

    #[test]
    fn test_worker_segment_transfers_one_allocation_to_capture_credit() {
        let resources = Resources::default();
        let segment = StateSegment::new(vec![7_u8; 1024]);
        let (bytes, checksum) = segment.shared_allocations();
        let charged = (bytes.capacity() + checksum.capacity()) as u64;
        let budget = StateBudget::new(10, charged).unwrap();
        let mut footprint = Footprint::default();
        footprint.segment(&segment, "actual_node").unwrap();
        assert_eq!(footprint.bytes("actual_node").unwrap(), charged);
        let work = resources
            .reserve_work(Footprint::default(), footprint, 0, budget, "actual_node")
            .unwrap();
        let captured = resources.retain(segment, 0, budget, "actual_node").unwrap();
        drop(work);
        assert!(
            resources
                .check(0, StateBudget::new(10, charged - 1).unwrap(), "actual_node")
                .is_err()
        );
        drop(bytes);
        drop(checksum);
        drop(captured);
        resources
            .check(0, StateBudget::new(10, 1).unwrap(), "actual_node")
            .unwrap();
    }

    #[test]
    fn test_read_reservation_becomes_capture_without_double_charging_encoded_bytes() {
        let resources = Resources::default();
        let budget = StateBudget::new(10, 5184).unwrap();
        let credit = resources.reserve(0, 5184, budget, "actual_node").unwrap();
        let segment = StateSegment::new(vec![7_u8; 1024]);
        let captured = credit
            .capture_segment(segment, 0, budget, "actual_node")
            .unwrap();
        resources
            .check(0, StateBudget::new(10, 1088).unwrap(), "actual_node")
            .unwrap();
        assert!(
            resources
                .check(0, StateBudget::new(10, 1087).unwrap(), "actual_node")
                .is_err()
        );
        drop(captured);
        resources
            .check(0, StateBudget::new(10, 1).unwrap(), "actual_node")
            .unwrap();
    }

    #[test]
    fn test_work_keeps_shared_allocation_charged_after_committed_reset() {
        let buffer = Buffer::from_vec(vec![7_u8; 1024]);
        let mut state = Footprint::default();
        state.buffer(&buffer, "features").unwrap();
        let resources = Resources::default();
        let credit = resources
            .reserve_work(
                state.clone(),
                state,
                0,
                StateBudget::new(10, 1024).unwrap(),
                "features",
            )
            .unwrap();
        resources.clear_committed();
        drop(buffer);
        assert!(
            resources
                .check(0, StateBudget::new(10, 1023).unwrap(), "features")
                .is_err()
        );
        drop(credit);
        resources
            .check(0, StateBudget::new(10, 1).unwrap(), "features")
            .unwrap();
    }
}
