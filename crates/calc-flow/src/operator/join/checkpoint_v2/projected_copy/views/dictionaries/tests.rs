use std::cell::Cell;

use datafusion::{
    arrow::{
        array::StringArray,
        buffer::Buffer,
        datatypes::{Field, Schema},
    },
    execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool},
};

use super::super::super::seed;
use super::*;

pub(in crate::operator::join::checkpoint_v2::projected_copy::views) fn controls() {
    selected::<Int8Type>();
    selected::<Int16Type>();
    selected::<Int32Type>();
    selected::<Int64Type>();
    selected::<UInt8Type>();
    selected::<UInt16Type>();
    selected::<UInt32Type>();
    selected::<UInt64Type>();
    refusal_controls();
}

fn source<K: ArrowDictionaryKeyType>() -> DictionaryArray<K> {
    let keys = [Some(3), Some(2), None, Some(0), Some(2), Some(1), Some(3)]
        .into_iter()
        .map(|key| key.map(|key| K::Native::from_usize(key).unwrap()))
        .collect::<PrimitiveArray<K>>();
    let values = StringArray::from(vec!["old0", "old1", "old2", "unused"]);
    DictionaryArray::new(keys, Arc::new(values)).slice(1, 5)
}

fn funded(data_type: &DataType, pool: &Arc<dyn MemoryPool>) -> (MemoryReservation, Arc<Funding>) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        data_type.clone(),
        true,
    )]));
    let credit = MemoryConsumer::new("dictionary-copy-control").register(pool);
    let resident = credit.new_empty();
    reserve(&resident, seed(&schema).unwrap()).unwrap();
    let funding = Funding::new([Arc::clone(&schema), schema], resident, None);
    (credit, funding)
}

fn selected<K: ArrowDictionaryKeyType>() {
    let source = source::<K>();
    let original = source.to_data();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(32 << 20));
    let (workspace, funding) = funded(source.data_type(), &pool);
    let data = copy(&original, &K::DATA_TYPE, &workspace, &funding, &|| Ok(())).unwrap();
    let copied = DictionaryArray::<K>::from(data.clone());
    assert_eq!(
        copied
            .keys()
            .iter()
            .map(|key| key.map(ArrowNativeType::as_usize))
            .collect::<Vec<_>>(),
        [Some(0), None, Some(1), Some(0), Some(2)],
    );
    let values = copied
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        values.iter().collect::<Vec<_>>(),
        [Some("old2"), Some("old0"), Some("old1")]
    );
    assert_eq!(source.to_data(), original);
    assert_independent(&original, &data);
    assert!(validity_bytes(1).unwrap() >= copied.nulls().unwrap().buffer().capacity());
    drop((copied, data, funding, workspace));
    assert_eq!(pool.reserved(), 0);
}

fn assert_independent(source: &ArrayData, copied: &ArrayData) {
    let original = buffers(source);
    for buffer in buffers(copied) {
        for original in &original {
            let start = buffer.as_ptr() as usize;
            let other = original.as_ptr() as usize;
            assert!(
                buffer.is_empty()
                    || original.is_empty()
                    || start >= other + original.len()
                    || other >= start + buffer.len()
            );
        }
    }
}

fn buffers(data: &ArrayData) -> Vec<Buffer> {
    data.buffers()
        .iter()
        .cloned()
        .chain(data.nulls().map(|nulls| nulls.buffer().clone()))
        .chain(data.child_data().iter().flat_map(buffers))
        .collect()
}

fn refusal_controls() {
    let source = source::<Int32Type>().to_data();
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(32 << 20));
    let (workspace, funding) = funded(source.data_type(), &pool);
    let checks = Cell::new(0);
    let copied = copy(&source, &DataType::Int32, &workspace, &funding, &|| {
        checks.set(checks.get() + 1);
        Ok(())
    })
    .unwrap();
    let total = checks.get();
    drop((copied, funding, workspace));
    assert_eq!(pool.reserved(), 0);
    for fail_at in [1, total / 2, total] {
        cancelled(&source, &pool, fail_at);
    }
    let (workspace, funding) = funded(source.data_type(), &pool);
    let blocker = MemoryConsumer::new("dictionary-budget-blocker").register(&pool);
    blocker.try_grow(32 * (1 << 20) - pool.reserved()).unwrap();
    let result = copy(&source, &DataType::Int32, &workspace, &funding, &|| Ok(()));
    assert!(result.is_err());
    drop((result, blocker, workspace, funding));
    assert_eq!(pool.reserved(), 0);
}

fn cancelled(source: &ArrayData, pool: &Arc<dyn MemoryPool>, fail_at: usize) {
    let before = source.clone();
    let (workspace, funding) = funded(source.data_type(), pool);
    let checks = Cell::new(0);
    let result = copy(source, &DataType::Int32, &workspace, &funding, &|| {
        checks.set(checks.get() + 1);
        if checks.get() >= fail_at {
            return Err(crate::CalcFlowError::Cancelled {
                run_id: "dictionary-copy".into(),
            });
        }
        Ok(())
    });
    assert!(matches!(
        result,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(*source, before);
    drop((result, workspace, funding));
    assert_eq!(pool.reserved(), 0);
}
