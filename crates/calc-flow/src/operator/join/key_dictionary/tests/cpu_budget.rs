use super::*;
use datafusion::arrow::{datatypes::Schema, record_batch::RecordBatch};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, UnboundedMemoryPool};

#[test]
fn cpu_budget_key_units_scale_with_cpu_and_row_work() {
    for (rows, cpus, expected) in [
        (256_000, 1, 1),
        (128_000, 8, 8),
        (256_000, 16, 16),
        (1_024_000, 64, 64),
        (64_000, 64, 4),
        (64_001, 64, 4),
        (96_000, 64, 6),
    ] {
        assert_eq!(work_units(rows, cpus), expected, "rows={rows}, cpus={cpus}");
    }
    for rows in 0..100 {
        assert!(work_units(rows, 64) <= rows);
    }
}

#[test]
fn cpu_budget_key_ranges_cover_rows_without_empty_or_overlapping_units() {
    for (rows, units) in [(64, 16), (65, 16), (1_001, 64), (128_003, 8)] {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let credit = MemoryConsumer::new("key-partition-test").register(&pool);
        let work = KeyWork {
            payload: RowPayload::from(RecordBatch::new_empty(Arc::new(Schema::empty()))),
            indices: Vec::new(),
            offsets: vec![(0, false); rows],
            units,
            name: "key-partition-test".into(),
            credit: Arc::new(credit),
            hook: None,
        };
        let mut next = 0;
        let mut minimum = usize::MAX;
        let mut maximum = 0;
        for unit in 0..work.unit_count() {
            let range = work.range(unit);
            assert_eq!(range.start, next);
            assert!(
                range.start < range.end,
                "rows={rows}, units={units}, unit={unit}"
            );
            assert!(range.end <= rows);
            minimum = minimum.min(range.len());
            maximum = maximum.max(range.len());
            next = range.end;
        }
        assert_eq!(next, rows);
        assert!(maximum - minimum <= 1);
    }
}

#[test]
fn cpu_budget_key_scratch_prepays_additional_worker_columns_and_tables() {
    let four = work_bytes(256_000, 64, 64, 4, "key-budget").unwrap();
    let sixteen = work_bytes(256_000, 64, 64, 16, "key-budget").unwrap();
    let per_unit = 64 * size_of::<LocalColumn<'_>>() + super::super::local_table_bytes(0).unwrap();
    assert!(sixteen >= four + 12 * per_unit);
    assert!(work_bytes(256_000, usize::MAX, 64, 16, "key-budget").is_err());
    assert!(work_bytes(256_000, 1, 1, usize::MAX, "key-budget").is_err());
}
