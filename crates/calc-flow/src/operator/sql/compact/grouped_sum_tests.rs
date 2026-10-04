use super::*;

const SUM_QUERY: &str = "SELECT key, SUM(value) AS total FROM events GROUP BY key";
const LARGE: Bits = (0, 0x4350_0000_0000_0000);
const NEG_LARGE: Bits = (0, 0xc350_0000_0000_0000);
const NAN_B: Bits = (0, 0xfff8_0000_0000_0002);

fn sum_operator() -> SqlOperator {
    operator_query(&DataType::Float64, SUM_QUERY)
}

async fn sum_oracle(actual: &Batch, prefix: &[Part], sequence: u64) {
    assert_query_oracle(actual, SUM_QUERY, &DataType::Float64, prefix, sequence).await;
}

fn sum_arrivals(reverse_zero: bool) -> Vec<Vec<Part>> {
    let (first, next) = if reverse_zero {
        (NEG_ZERO, ZERO)
    } else {
        (ZERO, NEG_ZERO)
    };
    let mut boundary = vec![(Some(12), Some(LARGE))];
    boundary.extend(vec![(Some(12), Some(ZERO)); 8191]);
    boundary.extend([(Some(12), Some(NEG_LARGE)), (Some(12), Some(ONE))]);
    vec![
        vec![vec![
            (Some(1), Some(LARGE)),
            (Some(2), Some(first)),
            (Some(3), None),
            (None, None),
            (Some(7), Some(NAN)),
            (Some(8), Some(POS_INF)),
            (Some(9), Some(NEG_INF)),
            (Some(10), Some(SNAN)),
            (Some(11), None),
        ]],
        vec![
            vec![
                (Some(1), Some(NEG_LARGE)),
                (Some(1), Some(ONE)),
                (Some(2), Some(next)),
                (Some(3), Some(TWO)),
                (Some(7), Some(NAN_B)),
            ],
            vec![
                (None, Some(NEG_ZERO)),
                (Some(8), Some(NEG_INF)),
                (Some(9), Some(POS_INF)),
                (Some(10), Some(ONE)),
            ],
        ],
        vec![
            vec![],
            boundary,
            vec![
                (Some(1), None),
                (Some(2), Some(NEG_ZERO)),
                (Some(3), None),
                (Some(7), Some(TWO)),
                (Some(11), None),
            ],
        ],
    ]
}

fn sum_capture(state: &mut SqlOperator, actual: &Batch) -> OperatorStateSnapshot {
    assert!(
        state.incremental.is_some() && state.compact.is_some() && state.retained.is_none(),
        "grouped Float64 SUM must own native state, not retained input"
    );
    let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["state_layout"], json!(3));
    assert_eq!(snapshot.inline_metadata["state_accounting"], json!(3));
    assert_eq!(
        snapshot
            .segments
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["batch-metadata", "control", "group-state", "logical-schema"]
    );
    let projection = state.compact.as_ref().unwrap().projection().unwrap();
    assert_eq!(
        projection.columns.logical_schema(),
        &schema(&DataType::Float64)
    );
    assert_eq!(projection.columns.ordinals(), &[0, 6]);
    let decoded = decode_sql_state(snapshot.segments["group-state"].bytes()).unwrap();
    let fields = decoded.table_payload().unwrap().schema().fields();
    assert_eq!(fields.len(), 2);
    assert_eq!(fields[0].name(), "key_0");
    assert_eq!(fields[0].data_type(), &DataType::Int64);
    assert!(fields[0].is_nullable());
    assert_eq!(fields[1].name(), "state_0_0");
    assert_eq!(fields[1].data_type(), &DataType::Float64);
    assert!(fields[1].is_nullable());
    assert_eq!(rows(&decoded), rows(actual));
    snapshot
}

#[tokio::test]
async fn test_grouped_float_sum_chronological_prefix_bits_own_native3() {
    for reverse_zero in [false, true] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "grouped_float", None);
        let mut state = sum_operator();
        let mut prefix = Vec::new();
        let mut weak = Vec::new();
        for (sequence, parts) in sum_arrivals(reverse_zero).into_iter().enumerate() {
            let batch = input(&DataType::Float64, &parts, sequence as u64);
            weak.extend(weak_arrays(&batch));
            prefix.extend(parts);
            let actual = process(&mut state, batch, &context).await;
            sum_oracle(&actual, &prefix, sequence as u64).await;
            if sequence > 0 {
                assert_eq!(rows(&actual)[&Some(1)][1], Cell::Float64(Some(ONE.1)));
            }
            if sequence == 2 {
                assert_eq!(rows(&actual)[&Some(12)][1], Cell::Float64(Some(ONE.1)));
                let snapshot = sum_capture(&mut state, &actual);
                assert_eq!(
                    snapshot.inline_metadata["rows"],
                    json!(prefix.iter().map(Vec::len).sum::<usize>())
                );
            }
        }
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop(state);
        assert_eq!(pool.reserved(), 0);
    }
}

async fn sum_roundtrip(reverse_zero: bool) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "grouped_float", None);
    let mut state = sum_operator();
    let mut prefix = Vec::new();
    let mut snapshot = None;
    for (sequence, parts) in sum_arrivals(reverse_zero).into_iter().enumerate() {
        prefix.extend(parts.clone());
        let actual = process(
            &mut state,
            input(&DataType::Float64, &parts, sequence as u64),
            &context,
        )
        .await;
        sum_oracle(&actual, &prefix, sequence as u64).await;
        if sequence == 2 {
            snapshot = Some(sum_capture(&mut state, &actual));
        }
    }
    let before = snapshot.unwrap();
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop(state);
    let mut restored = sum_operator();
    StreamOperator::restore(&mut restored, &before).unwrap();
    same_snapshot(&before, &restored.checkpoint(Epoch::INITIAL).unwrap());
    let empty = process(
        &mut restored,
        input(&DataType::Float64, &[vec![]], 3),
        &context,
    )
    .await;
    sum_oracle(&empty, &prefix, 3).await;
    drop(sum_capture(&mut restored, &empty));
    let next = vec![vec![
        (Some(1), Some(TWO)),
        (Some(3), Some(NEG_ZERO)),
        (Some(7), Some(SNAN)),
        (Some(10), Some(NAN_B)),
        (Some(11), Some(NEG_ZERO)),
        (None, Some(ONE)),
    ]];
    let batch = input(&DataType::Float64, &next, 4);
    let weak = weak_arrays(&batch);
    let actual = process(&mut restored, batch, &context).await;
    prefix.extend(next);
    sum_oracle(&actual, &prefix, 4).await;
    assert_eq!(
        rows(&actual)[&Some(1)][1],
        Cell::Float64(Some(3.0_f64.to_bits()))
    );
    let after = sum_capture(&mut restored, &actual);
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    let target = restored
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((restored, before, after, empty, actual));
    assert_eq!(pool.reserved(), 0);
    assert_eq!(target.reserved(), 0);
}

#[tokio::test]
async fn test_grouped_float_sum_current_capture_restore_empty_then_continue() {
    for reverse_zero in [false, true] {
        sum_roundtrip(reverse_zero).await;
    }
}

#[tokio::test]
async fn test_grouped_float_sum_emit_failure_preserves_capture_then_single_retry() {
    let job = job();
    let context = StreamOperatorContext::new(&job, "grouped_float", None);
    let mut state = sum_operator();
    let first = vec![vec![(Some(1), Some(LARGE)), (None, None)]];
    let initial = process(&mut state, input(&DataType::Float64, &first, 0), &context).await;
    sum_oracle(&initial, &first, 0).await;
    let before = state.checkpoint(Epoch::INITIAL).unwrap();
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let reserved = pool.reserved();
    let next = vec![vec![
        (Some(1), Some(NEG_LARGE)),
        (Some(1), Some(ONE)),
        (None, Some(SNAN)),
        (Some(4), Some(POS_INF)),
    ]];
    let rejected = input(&DataType::Float64, &next, 1);
    let weak = weak_arrays(&rejected);
    let failure = state
        .process_data("events", rejected, &context, &mut Reject)
        .await;
    assert!(
        matches!(failure, Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject-grouped-float")
    );
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    assert_eq!(pool.reserved(), reserved);
    same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
    let actual = process(&mut state, input(&DataType::Float64, &next, 1), &context).await;
    let all = first.into_iter().chain(next).collect::<Vec<_>>();
    sum_oracle(&actual, &all, 1).await;
    assert_eq!(rows(&actual)[&Some(1)][1], Cell::Float64(Some(ONE.1)));
    let after = sum_capture(&mut state, &actual);
    assert_eq!(after.inline_metadata["rows"], json!(6));
    drop((state, before, after, actual, initial));
    assert_eq!(pool.reserved(), 0);
}
