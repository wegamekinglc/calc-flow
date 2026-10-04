use super::*;

fn arrivals() -> [Part; 3] {
    let initial = (0..64)
        .flat_map(|key| [(Some(key), Some(ONE)); 3])
        .collect::<Part>();
    let update = (0..257)
        .map(|row| {
            let key = if row % 7 == 0 { None } else { Some(11) };
            let value = [
                Some(ONE),
                Some(NEG_ZERO),
                Some(NAN),
                Some(POS_INF),
                Some(NEG_INF),
                None,
                Some(TWO),
            ][row % 7];
            (key, value)
        })
        .collect::<Part>();
    let continuation = vec![
        (Some(11), Some(SNAN)),
        (Some(99), Some(TWO)),
        (None, Some(ONE)),
    ];
    [initial, update, continuation]
}

async fn delta_case(dtype: &DataType, key_type: &DataType, composite: bool, filtered: bool) {
    let keys = if composite { "key, other" } else { "key" };
    let predicate = if filtered {
        "WHERE (accepted OR other) AND value >= 0"
    } else {
        ""
    };
    let text = format!("SELECT {keys}, {AGGREGATES} FROM events {predicate} GROUP BY {keys}");
    let job = job();
    let context = StreamOperatorContext::new(&job, "grouped_float", None);
    let mut state = filtered_operator(dtype, key_type, &text);
    let mut history = Vec::new();
    let mut pools = Vec::new();
    let mut base = None;
    for (sequence, part) in arrivals().into_iter().enumerate() {
        let parts = vec![part];
        let batch = filtered_input(dtype, key_type, &parts, sequence as u64);
        let weak = weak_arrays(&batch);
        let actual = process(&mut state, batch, &context).await;
        history.push(parts);
        oracle(
            &actual,
            &text,
            dtype,
            key_type,
            &history,
            if composite { 2 } else { 1 },
        )
        .await;
        state.prepare_compact_capture_async(&context).await.unwrap();
        let snapshot = capture(&mut state);
        assert_eq!(
            snapshot
                .segments
                .keys()
                .filter(|id| id.starts_with("group-delta-"))
                .count(),
            sequence
        );
        let current = snapshot.segments["group-state"].bytes_arc();
        if let Some(base) = &base {
            assert!(Arc::ptr_eq(base, &current));
        } else {
            base = Some(current);
        }
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        pools.push(
            state
                .stream_state
                .runtime()
                .unwrap()
                .incremental_memory_pool(),
        );
        drop((actual, state));
        state = filtered_operator(dtype, key_type, &text);
        StreamOperator::restore(&mut state, &snapshot).unwrap();
        same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
    }
    let parts = vec![vec![]];
    let actual = process(
        &mut state,
        filtered_input(dtype, key_type, &parts, 3),
        &context,
    )
    .await;
    history.push(parts);
    oracle(
        &actual,
        &text,
        dtype,
        key_type,
        &history,
        if composite { 2 } else { 1 },
    )
    .await;
    pools.push(
        state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool(),
    );
    drop((state, actual, base));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pools.iter().map(|pool| pool.reserved()).sum::<usize>(), 0);
}

#[tokio::test]
async fn test_compact_delta_float_filter_where_exact_bits_and_cold_continuation() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for key_type in [DataType::Int64, DataType::Utf8, DataType::LargeUtf8] {
            for composite in [false, true] {
                for filtered in [false, true] {
                    delta_case(&dtype, &key_type, composite, filtered).await;
                }
            }
        }
    }
}
