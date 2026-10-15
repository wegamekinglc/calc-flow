use super::*;
use std::sync::{Mutex, Weak, mpsc};

struct Release(Option<mpsc::Sender<()>>);

impl Drop for Release {
    fn drop(&mut self) {
        if let Some(release) = self.0.take() {
            let _ = release.send(());
        }
    }
}

#[tokio::test]
async fn test_rebase_cancel_retains_only_serialized_inputs_until_actual_drain() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let (mut join, old) = checkpointed(&context).await;
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    let rows = Arc::downgrade(&join.state.left.0);
    let key = Arc::downgrade(join.state.left.key_at(0));
    let column = Arc::downgrade(join.state.left.at(0).record.column(0));
    let segments = old
        .segments
        .values()
        .map(|segment| Arc::downgrade(&segment.bytes_arc()))
        .collect::<Vec<_>>();
    let backing = old
        .segments
        .values()
        .map(|segment| segment.bytes_arc().capacity())
        .sum::<usize>();
    let (entered, started) = tokio::sync::oneshot::channel();
    let entered = Mutex::new(Some(entered));
    let (release, wait) = mpsc::channel();
    let release = Release(Some(release));
    let wait = Mutex::new(wait);
    join.set_checkpoint_writer_test_hook(Arc::new(move |credit| {
        assert_eq!(std::thread::current().name(), Some("calc-flow-gather"));
        entered
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .send(credit.size())
            .unwrap();
        wait.lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
    }));
    let before = edge_tests::image(&join);
    let mut preparing = Box::pin(join.prepare_checkpoint_async(&context));
    assert!(futures::poll!(preparing.as_mut()).is_pending());
    let paid = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::select! {
            started = started => started.unwrap(),
            result = preparing.as_mut() => panic!("writer completed before release: {result:?}"),
        }
    })
    .await
    .unwrap();
    enabled.cancellation().cancel();
    drop(preparing);
    assert_eq!(edge_tests::image(&join), before);
    drop((join, old));
    let mut drain = Box::pin(enabled.gather_owner().close_and_drain());
    let pending = futures::poll!(drain.as_mut()).is_pending();
    let retained = segments.iter().all(|segment| segment.upgrade().is_some());
    let rows_retired =
        rows.upgrade().is_none() && key.upgrade().is_none() && column.upgrade().is_none();
    let reserved = pool.reserved();
    drop(release);
    let failures = tokio::time::timeout(Duration::from_secs(5), drain)
        .await
        .unwrap();
    assert!(paid > 0);
    assert!(pending);
    assert!(
        retained,
        "blocked work must keep every serialized input owner"
    );
    assert!(
        rows_retired,
        "Rebase work must not retain live rows or decoded payloads"
    );
    assert!(reserved >= backing + paid);
    assert!(failures.is_empty());
    assert!(segments.iter().all(|segment| segment.upgrade().is_none()));
    let (home, generation, attempt) = enabled.gather_owner().funding();
    assert_eq!((generation, attempt), (0, 0));
    assert_eq!(pool.reserved(), home);
    drop(context);
    drop(enabled);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_rebase_old_indexes_retire_independently_of_payload_aliases() {
    let enabled = job();
    let context = StreamOperatorContext::new(&enabled, "prepared-reuse", None);
    let (mut join, old) = checkpointed(&context).await;
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    let indexes = old
        .segments
        .iter()
        .filter(|(name, _)| !name.contains("-payload-"))
        .map(|(_, segment)| {
            (
                Arc::downgrade(&segment.bytes_arc()),
                segment.bytes_arc().capacity(),
            )
        })
        .collect::<Vec<_>>();
    let old_controls = join.v2_writer.base.as_ref().unwrap().test_owners();
    let payload = old
        .segments
        .iter()
        .find(|(name, _)| name.starts_with("left-payload-"))
        .unwrap()
        .1
        .clone();
    let payload_backing = payload.bytes_arc().capacity();
    let shared = Arc::downgrade(&payload.bytes_arc());
    join.prepare_checkpoint_async(&context).await.unwrap();
    assert!(
        old_controls.iter().all(|owner| owner.upgrade().is_none()),
        "Rebase must not retain old Base control or owner-list funding"
    );
    let current = join.checkpoint(Epoch::new(5).unwrap()).unwrap();
    let current_indexes =
        ["left-base", "right-base"].map(|name| Arc::downgrade(&current.segments[name].bytes_arc()));
    assert!(indexes.iter().all(|(owner, _)| owner.upgrade().is_some()));
    let before_old_drop = pool.reserved();
    drop(old);
    let retired = indexes.iter().map(|(_, capacity)| capacity).sum::<usize>();
    assert!(indexes.iter().all(|(owner, _)| owner.upgrade().is_none()));
    assert!(before_old_drop - pool.reserved() >= retired);
    assert!(shared.upgrade().is_some());
    let before_current_drop = pool.reserved();
    drop((join, current));
    drop(context);
    assert!(enabled.gather_owner().close_and_drain().await.is_empty());
    drop(enabled);
    assert!(
        current_indexes
            .iter()
            .all(|owner| owner.upgrade().is_none())
    );
    assert!(shared.upgrade().is_some());
    assert!(pool.reserved() >= payload_backing);
    assert!(pool.reserved() < before_current_drop);
    drop(payload);
    assert!(Weak::upgrade(&shared).is_none());
    assert_eq!(pool.reserved(), 0);
}
