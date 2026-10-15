//! Differential checks for parallel capture's local and global route coordinates.

use super::*;

async fn admitted(
    mut operator: StreamAsofJoinOperator,
    left: &[Row],
    right: &[Row],
) -> StreamAsofJoinOperator {
    let job = StreamJobContext::new(907, "asof", JsonMap::new(), None, CancellationToken::new());
    let context = test_context(&job);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    for (port, rows) in [("right", right), ("left", left)]
        .into_iter()
        .filter(|(_, rows)| !rows.is_empty())
    {
        operator
            .process_data(port, input(&operator, rows), &context, &mut output)
            .await
            .unwrap();
    }
    assert!(output.drain("output").is_empty());
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    operator
}

async fn assert_capture(operator: &StreamAsofJoinOperator, count: usize) {
    let token = CancellationToken::new();
    let job = StreamJobContext::new(908, "asof", JsonMap::new(), None, token.clone());
    let context = test_context(&job);
    let status = operator.status.clone();
    let pool = operator.runtime.pool.clone();
    let reserved = pool.reserved();
    let (work, credit) = capture(operator, count, 3, &context).await.unwrap();
    let stop = GatherStop::from_job(&job);
    let matches = (0..work.unit_count())
        .map(|ordinal| work.run(ordinal, &stop).unwrap())
        .collect();
    let expected = operator
        .state
        .left
        .output_iter()
        .take(count)
        .map(|(key, _)| operator.state.candidate(key.1, *key.0, 1).copied())
        .collect::<Vec<_>>();
    assert_eq!(restore_order(matches, count), expected);
    assert_eq!(operator.status, status);
    drop((work, credit));
    assert_eq!(pool.reserved(), reserved);
    token.cancel();
    assert!(capture(operator, count, 3, &context).await.is_err());
    assert_eq!(pool.reserved(), reserved);
    assert_eq!(operator.status, status);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
}

async fn dictionary_boundaries_and_short_prefixes() {
    for keys in [1, 63, 64, 65] {
        for consumed in [0, 17] {
            let (operator, left, right) = fixture_with_keys(false, 520, keys);
            let mut operator = admitted(operator, &left, &right).await;
            operator.state.commit_left_prefix(consumed);
            let remaining = operator.state.left.len();
            let run = operator.state.left.output_runs().next().unwrap();
            assert_eq!(run.keyed_rows(run.len()).is_some(), keys <= 64);
            for count in [1, 17, remaining] {
                assert_capture(&operator, count).await;
            }
        }
    }
}

async fn independent_dictionaries_keep_global_positions_and_buckets() {
    let (operator, left, right) = fixture_with_keys(false, 4_096, 63);
    let mut operator = admitted(operator, &left, &right).await;
    operator.state.commit_left_prefix(17);
    assert!(
        operator
            .state
            .left
            .output_runs()
            .filter(|run| run.keyed_rows(run.len()).is_some())
            .count()
            >= 2
    );
    for count in [2_031, 2_050, operator.state.left.len()] {
        assert_capture(&operator, count).await;
    }
}

async fn retired_dictionary_slots_keep_their_original_ids() {
    let (operator, mut left, right) = fixture_with_keys(false, 520, 63);
    left[0].0 = "retired key".into();
    left[0].1 = -1;
    let mut operator = admitted(operator, &left, &right).await;
    operator.state.commit_left_prefix(1);
    let run = operator.state.left.output_runs().next().unwrap();
    let (keys, _) = run.keyed_rows(run.len()).unwrap();
    assert_eq!(keys.len(), 64);
    assert!(keys.iter().any(Option::is_none));
    assert_capture(&operator, operator.state.left.len()).await;
}

async fn legacy_rows_keep_generic_capture() {
    let (operator, left, right) = fixture(false, 520);
    let mut operator = admitted(operator, &left, &right).await;
    let rows = operator
        .state
        .left
        .iter()
        .map(|((time, key, sequence), row)| ((*time, key.clone(), sequence.into_owned()), row))
        .collect::<Vec<_>>();
    operator.state.left = state::LeftState::default();
    for (key, row) in rows {
        operator.state.left.insert(key, row);
    }
    assert!(
        operator
            .state
            .left
            .output_runs()
            .all(|run| run.keyed_rows(run.len()).is_none())
    );
    assert_capture(&operator, operator.state.left.len()).await;
}

pub(super) async fn check_capture_routes() {
    dictionary_boundaries_and_short_prefixes().await;
    independent_dictionaries_keep_global_positions_and_buckets().await;
    retired_dictionary_slots_keep_their_original_ids().await;
    legacy_rows_keep_generic_capture().await;
}
