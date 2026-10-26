use std::sync::{Arc, atomic::Ordering};

use calc_flow::JobState;
use tokio::sync::Semaphore;

use super::account_support::*;

#[tokio::test]
async fn test_aggregate_topology_restores_uninitialized_and_ended_inputs() {
    tokio::time::timeout(DEADLINE, async {
        for partial in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let trades = vec![trade("alice", 10), trade("alice", -4)];
            let quotes = vec![quote(2.0, 1), quote(3.0, 3), quote(99.0, 2)];
            let (p_tx, p, p_probe) = source(trades.clone());
            let (_q_tx, q, q_probe) = source(quotes.clone());
            let (sink, mut output, sink_probe) = sink_fixture();
            let first = start(directory.path(), p, q, sink, 10).await;
            if partial {
                p_tx.send(()).unwrap();
                while first.status().operators["values"].input_batches == 0 {
                    tokio::task::yield_now().await;
                }
            }
            first.trigger_checkpoint().await.unwrap();
            assert!(
                output.try_recv().is_err(),
                "missing quote must remain uninitialized"
            );
            first.cancel().await;
            assert_released(&first, &p_probe, &q_probe, &sink_probe);

            let (p_tx, p, p_probe) = source(trades.clone());
            let (q_tx, q, q_probe) = source(quotes.clone());
            let (sink, mut output, sink_probe) = sink_fixture();
            let second = start(directory.path(), p, q, sink, 10).await;
            assert_eq!(*p_probe.opened_at.lock().unwrap(), [usize::from(partial)]);
            assert_eq!(*q_probe.opened_at.lock().unwrap(), [0]);
            if !partial {
                p_tx.send(()).unwrap();
            }
            q_tx.send(()).unwrap();
            assert_value(&output.recv().await.unwrap(), 20.0);
            q_tx.send(()).unwrap();
            assert_value(&output.recv().await.unwrap(), 30.0);
            p_tx.send(()).unwrap();
            assert_value(&output.recv().await.unwrap(), 18.0);
            drop(p_tx);
            p_probe.ended.notified().await;
            let epoch = second.trigger_checkpoint().await.unwrap();
            assert_eq!(
                second.trigger_checkpoint().await.unwrap(),
                epoch.next().unwrap()
            );
            second.cancel().await;
            assert_released(&second, &p_probe, &q_probe, &sink_probe);

            let (_p_tx, p, p_probe) = source(trades.clone());
            let (q_tx, q, q_probe) = source(quotes.clone());
            let (sink, mut output, sink_probe) = sink_fixture();
            let third = start(directory.path(), p, q, sink, 10).await;
            assert!(
                p_probe.opened_at.lock().unwrap().is_empty(),
                "ended trades must not reopen"
            );
            assert_eq!(*q_probe.opened_at.lock().unwrap(), [2]);
            q_tx.send(()).unwrap();
            assert_value(&output.recv().await.unwrap(), 18.0);
            drop(q_tx);
            assert_eq!(third.wait().await.state, JobState::Completed);
            assert_eq!(third.status().task_count, 0);
            assert_eq!(q_probe.closes.load(Ordering::SeqCst), 1);
            assert_eq!(sink_probe.closes.load(Ordering::SeqCst), 1);

            let (_p_tx, p, p_probe) = source(trades);
            let (_q_tx, q, q_probe) = source(quotes);
            let (sink, mut output, _) = sink_fixture();
            let terminal = start(directory.path(), p, q, sink, 10).await;
            assert_eq!(terminal.wait().await.state, JobState::Completed);
            assert!(p_probe.opened_at.lock().unwrap().is_empty());
            assert!(q_probe.opened_at.lock().unwrap().is_empty());
            assert!(
                output.try_recv().is_err(),
                "terminal restore must not replay output"
            );
        }
    })
    .await
    .expect("aggregate topology recovery timed out");
}

#[tokio::test]
async fn test_aggregate_topology_failed_or_cancelled_sink_replays_from_durable_cut() {
    tokio::time::timeout(DEADLINE, async {
        for fail in [false, true] {
            let directory = tempfile::tempdir().unwrap();
            let trades = vec![trade("alice", 10)];
            let quotes = vec![quote(2.0, 1), quote(3.0, 2)];
            let (p_tx, p, p_probe) = source(trades.clone());
            let (q_tx, q, q_probe) = source(quotes.clone());
            let (mut sink, mut output, sink_probe) = sink_fixture();
            let permits = Arc::new(Semaphore::new(1));
            sink.permits = Some(permits.clone());
            sink.fail_at = fail.then_some(2);
            let first = start(directory.path(), p, q, sink, 10).await;
            p_tx.send(()).unwrap();
            q_tx.send(()).unwrap();
            assert_value(&output.recv().await.unwrap(), 20.0);
            let epoch = first.trigger_checkpoint().await.unwrap();
            q_tx.send(()).unwrap();
            while sink_probe.writes.load(Ordering::SeqCst) < 2 {
                sink_probe.entered.notified().await;
            }
            assert!(output.try_recv().is_err());
            if fail {
                permits.add_permits(1);
                assert_eq!(first.wait().await.state, JobState::Failed);
            } else {
                assert_eq!(first.cancel().await.state, JobState::Cancelled);
            }
            assert_eq!(first.status().checkpoint.last_completed_epoch, Some(epoch));
            assert_released(&first, &p_probe, &q_probe, &sink_probe);

            let (_p_tx, p, p_probe) = source(trades);
            let (q_tx, q, q_probe) = source(quotes);
            let (sink, mut output, sink_probe) = sink_fixture();
            let recovered = start(directory.path(), p, q, sink, 10).await;
            assert_eq!(*p_probe.opened_at.lock().unwrap(), [1]);
            assert_eq!(*q_probe.opened_at.lock().unwrap(), [1]);
            q_tx.send(()).unwrap();
            assert_value(&output.recv().await.unwrap(), 30.0);
            recovered.cancel().await;
            assert_released(&recovered, &p_probe, &q_probe, &sink_probe);
        }
    })
    .await
    .expect("sink failure recovery timed out");
}

#[tokio::test]
async fn test_full_aggregate_snapshot_exceeding_edge_is_rejected_whole() {
    tokio::time::timeout(DEADLINE, async {
        let directory = tempfile::tempdir().unwrap();
        let (p_tx, p, p_probe) =
            source(vec![trade("alice", 1), trade("bob", 1), trade("carol", 1)]);
        let (q_tx, q, q_probe) = source(vec![quote(2.0, 1)]);
        let (sink, mut output, sink_probe) = sink_fixture();
        let job = start(directory.path(), p, q, sink, 2).await;
        p_tx.send(()).unwrap();
        q_tx.send(()).unwrap();
        assert_value(&output.recv().await.unwrap(), 2.0);
        p_tx.send(()).unwrap();
        assert_eq!(output.recv().await.unwrap().num_rows(), 2);
        p_tx.send(()).unwrap();
        assert_eq!(job.wait().await.state, JobState::Failed);
        assert!(
            output.try_recv().is_err(),
            "oversized aggregate must not be split into replacements"
        );
        assert_released(&job, &p_probe, &q_probe, &sink_probe);
    })
    .await
    .expect("aggregate edge rejection timed out");
}

#[tokio::test]
async fn test_slow_valuation_sink_backpressures_quote_source_and_cancels_cleanly() {
    tokio::time::timeout(DEADLINE, async {
        let directory = tempfile::tempdir().unwrap();
        let (p_tx, p, p_probe) = source(vec![trade("alice", 1)]);
        let (q_tx, q, q_probe) = source((1..=100).map(|n| quote(f64::from(n), n.into())).collect());
        let (mut sink, mut output, sink_probe) = sink_fixture();
        let permits = Arc::new(Semaphore::new(0));
        sink.permits = Some(permits.clone());
        let job = start(directory.path(), p, q, sink, 2).await;
        p_tx.send(()).unwrap();
        for _ in 0..100 {
            q_tx.send(()).unwrap();
        }
        sink_probe.entered.notified().await;
        // A blocked write must bound upstream admission, not accumulate every quote.
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(50), async {
                while q_probe.reads.load(Ordering::SeqCst) < 100 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .is_err()
        );
        assert!(output.try_recv().is_err());
        permits.add_permits(1);
        assert_value(&output.recv().await.unwrap(), 1.0);
        job.cancel().await;
        assert_released(&job, &p_probe, &q_probe, &sink_probe);
    })
    .await
    .expect("sink backpressure timed out");
}
