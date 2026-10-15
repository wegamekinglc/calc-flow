use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use datafusion::execution::memory_pool::MemoryReservation;

static NEXT_NAMESPACE: AtomicU64 = AtomicU64::new(1);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct KeyAddress {
    namespace: u64,
    slot: u32,
}

#[derive(Clone)]
pub(super) struct NamespaceLease(Arc<Namespace>);

struct Namespace {
    number: u64,
    _credit: MemoryReservation,
}

impl NamespaceLease {
    #[cfg(test)]
    pub(super) fn funded_bytes(&self) -> usize {
        let Namespace {
            _credit: credit, ..
        } = self.0.as_ref();
        credit.size()
    }

    pub(super) fn new(credit: MemoryReservation) -> Option<Self> {
        let number = next_namespace(&NEXT_NAMESPACE)?;
        Some(Self(Arc::new(Namespace {
            number,
            _credit: credit,
        })))
    }

    pub(super) fn address(&self, slot: u32) -> Option<KeyAddress> {
        (slot != u32::MAX).then_some(KeyAddress {
            namespace: self.0.number,
            slot,
        })
    }

    pub(super) fn control_bytes() -> Option<usize> {
        arc_bytes::<Namespace>()
    }
}

impl KeyAddress {
    pub(super) fn slot(self) -> u32 {
        self.slot
    }
    pub(super) fn hash(self) -> u64 {
        self.namespace.rotate_left(32) ^ u64::from(self.slot)
    }
}

pub(super) struct IdentityLease<T>(Arc<Identity<T>>);

#[cfg(test)]
pub(super) struct WeakIdentity<T>(std::sync::Weak<Identity<T>>);

#[cfg(test)]
impl<T> WeakIdentity<T> {
    pub(super) fn strong_count(&self) -> usize {
        self.0.strong_count()
    }
}

struct Identity<T> {
    address: KeyAddress,
    _namespace: NamespaceLease,
    owner: T,
    _credit: MemoryReservation,
}

impl<T> Clone for IdentityLease<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T> IdentityLease<T> {
    #[cfg(test)]
    pub(super) fn funded_bytes(&self) -> usize {
        let Identity {
            _credit: credit, ..
        } = self.0.as_ref();
        credit.size()
    }

    pub(super) fn strong_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }
    #[cfg(test)]
    pub(super) fn downgrade(&self) -> WeakIdentity<T> {
        WeakIdentity(Arc::downgrade(&self.0))
    }
    pub(super) fn new(
        namespace: &NamespaceLease,
        slot: u32,
        owner: T,
        credit: MemoryReservation,
    ) -> Option<Self> {
        let address = namespace.address(slot)?;
        Some(Self(Arc::new(Identity {
            address,
            _namespace: namespace.clone(),
            owner,
            _credit: credit,
        })))
    }
    pub(super) fn address(&self) -> KeyAddress {
        self.0.address
    }
    pub(super) fn owner(&self) -> &T {
        &self.0.owner
    }
    pub(super) fn control_bytes() -> Option<usize> {
        arc_bytes::<Identity<T>>()
    }
}

fn arc_bytes<T>() -> Option<usize> {
    std::alloc::Layout::new::<[std::sync::atomic::AtomicUsize; 2]>()
        .extend(std::alloc::Layout::new::<T>())
        .ok()
        .map(|(layout, _)| layout.pad_to_align().size())
}

fn next_namespace(counter: &AtomicU64) -> Option<u64> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |number| {
            number.checked_add(1)
        })
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_persistent_dictionary_namespace_atomic_exhaustion_does_not_wrap() {
        let counter = AtomicU64::new(u64::MAX - 1);
        assert_eq!(next_namespace(&counter), Some(u64::MAX - 1));
        assert_eq!(next_namespace(&counter), None);
        assert_eq!(counter.load(Ordering::Relaxed), u64::MAX);
        assert_eq!(next_namespace(&counter), None);
    }
}
