use super::*;

#[tokio::test]
async fn test_off_mutation_keeps_cached_fresh_and_manual_captures_untrusted() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let disabled = job().with_checkpointing(false);
    let off = StreamOperatorContext::new(&disabled, "prepared-reuse", None);
    let mut join = seeded(&context).await;
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    let bases = count_base(&mut join);
    join.prepare_checkpoint_async(&context).await.unwrap();
    assert!(join.v2_writer.history_trusted);
    let original = join.v2_writer.tracker.snapshot(None);
    admit_left(&mut join, &off).await;
    assert_eq!(join.v2_writer.tracker.snapshot(None), original);
    assert!(join.fresh_v2_preparation_matches());
    assert!(!join.v2_writer.history_trusted);
    join.prepare_checkpoint_async(&context).await.unwrap();
    assert_eq!(bases.load(Ordering::SeqCst), 1);
    assert!(!join.v2_writer.history_trusted);
    let cached = join.checkpoint(Epoch::INITIAL).unwrap();
    assert!(!join.v2_writer.history_trusted);
    let clean = join.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert!(!join.v2_writer.history_trusted);
    let mut invalid = cached.clone();
    invalid
        .inline_metadata
        .insert("layout_version".into(), 999.into());
    assert!(join.restore(&invalid).is_err());
    assert!(!join.v2_writer.history_trusted);
    join.prepare_v2_checkpoint(&context).await.unwrap();
    assert_eq!(bases.load(Ordering::SeqCst), 2);
    assert!(join.v2_writer.history_trusted);
    let current = join.checkpoint(Epoch::new(3).unwrap()).unwrap();
    let mut restored = operator();
    restored.restore(&current).unwrap();
    assert_same_live(&join, &restored);
    drop((join, restored, cached, clean, invalid, current));
    drop((context, off));
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    assert!(disabled.gather_owner().close_and_drain().await.is_empty());
    drop((enabled, disabled));
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_untrusted_history_survives_prepared_delta_capture() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let disabled = job().with_checkpointing(false);
    let off = StreamOperatorContext::new(&disabled, "prepared-reuse", None);
    let mut join = seeded(&context).await;
    join.prepare_checkpoint_async(&context).await.unwrap();
    let initial = join.checkpoint(Epoch::INITIAL).unwrap();
    admit_left(&mut join, &off).await;
    assert!(!join.v2_writer.history_trusted);
    admit_left(&mut join, &context).await;
    let bases = count_base(&mut join);
    join.prepare_checkpoint_async(&context).await.unwrap();
    assert!(join.v2_writer.prepared_delta.is_some());
    assert_eq!(bases.load(Ordering::SeqCst), 0);
    assert!(!join.v2_writer.history_trusted);
    let delta = join.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert!(!join.v2_writer.history_trusted);
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    drop((join, initial, delta));
    drop((context, off));
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    assert!(disabled.gather_owner().close_and_drain().await.is_empty());
    drop((enabled, disabled));
    assert_eq!(pool.reserved(), 0);
}
