use super::*;

#[tokio::test]
async fn test_off_mutation_requires_live_fresh_preparation_and_cold_parity() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let disabled = job().with_checkpointing(false);
    let off = StreamOperatorContext::new(&disabled, "prepared-reuse", None);
    let mut join = seeded(&context).await;
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    let bases = count_base(&mut join);
    join.prepare_checkpoint_async(&context).await.unwrap();
    admit_left(&mut join, &off).await;
    join.prepare_checkpoint_async(&context).await.unwrap();
    let current = join.checkpoint(Epoch::INITIAL).unwrap();
    let mut restored = operator();
    let restore = restored.restore(&current);
    if restore.is_ok() {
        assert_same_live(&join, &restored);
    }
    let base_count = bases.load(Ordering::SeqCst);
    drop((join, restored, current, context, off));
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    assert!(disabled.gather_owner().close_and_drain().await.is_empty());
    drop((enabled, disabled));
    assert_eq!(pool.reserved(), 0);
    restore.expect("ON preparation must replace a fresh base cached before OFF changes");
    assert_eq!(base_count, 2);
}

#[tokio::test]
async fn test_off_history_rejects_capture_until_live_base_preparation() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let disabled = job().with_checkpointing(false);
    let off = StreamOperatorContext::new(&disabled, "prepared-reuse", None);
    let mut join = seeded(&context).await;
    join.prepare_checkpoint_async(&context).await.unwrap();
    let initial = join.checkpoint(Epoch::INITIAL).unwrap();
    admit_left(&mut join, &off).await;
    let denied = join.checkpoint(Epoch::new(2).unwrap());
    let refused = denied.is_err();
    drop(denied);
    if refused {
        admit_left(&mut join, &context).await;
        let bases = count_base(&mut join);
        join.prepare_checkpoint_async(&context).await.unwrap();
        assert!(join.v2_writer.prepared_delta.is_none());
        assert!(join.v2_writer.history_trusted);
        assert_eq!(bases.load(Ordering::SeqCst), 1);
        let current = join.checkpoint(Epoch::new(2).unwrap()).unwrap();
        let mut restored = operator();
        restored.restore(&current).unwrap();
        assert_same_live(&join, &restored);
    }
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    drop((join, initial, context, off));
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    assert!(disabled.gather_owner().close_and_drain().await.is_empty());
    drop((enabled, disabled));
    assert_eq!(pool.reserved(), 0);
    assert!(
        refused,
        "untrusted capture must require live base preparation"
    );
}
