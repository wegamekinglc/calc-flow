use super::*;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
use std::sync::Arc;

fn reservation(limit: usize) -> (Arc<dyn MemoryPool>, MemoryReservation) {
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(limit));
    let credit = MemoryConsumer::new("exact-buffer-test").register(&pool);
    (pool, credit)
}

#[test]
fn test_exact_buffer_prepaid_single_allocation_and_boundary() {
    let scratch = 47 + 257;
    let capacity = 513;
    let (legacy_pool, legacy_credit) = reservation(scratch + capacity);
    let mut legacy = PaidBuffer::new(&legacy_credit, scratch);
    legacy.append(&[0xAB; 256]).unwrap();
    assert!(legacy.append(&[0xCD; 257]).is_err());
    drop(legacy);
    drop(legacy_credit);
    assert_eq!(legacy_pool.reserved(), 0);

    let (pool, credit) = reservation(scratch + capacity);
    credit.try_grow(scratch).unwrap();
    let mut result = None;
    let allocations = allocation_counter::measure(|| {
        let mut bytes = PaidBuffer::exact(&credit, scratch, capacity).unwrap();
        assert_eq!(credit.size(), scratch + capacity);
        bytes.append(&[0xAB; 256]).unwrap();
        bytes.append(&[0xCD; 257]).unwrap();
        result = Some(bytes.into_bytes().unwrap());
    });
    let bytes = result.unwrap();
    assert_eq!((bytes.len(), bytes.capacity()), (capacity, capacity));
    assert_eq!(&bytes[..256], &[0xAB; 256]);
    assert_eq!(&bytes[256..], &[0xCD; 257]);
    assert_eq!(allocations.count_total, 1);
    assert_eq!(allocations.bytes_current, i64::try_from(capacity).unwrap());
    assert!(allocations.bytes_max <= u64::try_from(credit.size()).unwrap());
    drop(bytes);
    drop(credit);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_exact_buffer_rejects_oversize_writes_and_short_finish() {
    let (pool, credit) = reservation(3);
    let mut bytes = PaidBuffer::exact(&credit, 0, 3).unwrap();
    bytes.append(&[1, 2]).unwrap();
    assert!(bytes.append(&[3, 4]).is_err());
    bytes.append(&[3]).unwrap();
    assert_eq!(bytes.into_bytes().unwrap(), vec![1, 2, 3]);
    let mut short = PaidBuffer::exact(&credit, 0, 3).unwrap();
    short.append(&[1, 2]).unwrap();
    assert!(short.into_bytes().is_err());
    drop(credit);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_exact_buffer_rejects_capacity_overflow_before_admission() {
    let (pool, credit) = reservation(usize::MAX);
    assert!(PaidBuffer::exact(&credit, usize::MAX, 1).is_err());
    assert_eq!(pool.reserved(), 0);
    assert!(PaidBuffer::exact(&credit, 0, usize::MAX).is_err());
    assert_eq!(pool.reserved(), 0);
    let empty = PaidBuffer::exact(&credit, 0, 0)
        .unwrap()
        .into_bytes()
        .unwrap();
    assert_eq!((empty.len(), empty.capacity()), (0, 0));
    assert_eq!(pool.reserved(), 0);
}
