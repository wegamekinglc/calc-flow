use super::*;

#[path = "global_record_controls.rs"]
mod global_record_controls;

#[path = "global_integer_avg_tests.rs"]
mod global_integer_avg_tests;

#[path = "global_float_filter_tests.rs"]
mod global_float_filter_tests;

const SUM: &str = "SELECT SUM(value) AS total FROM events";
const AVG: &str = "SELECT AVG(value) AS mean FROM events";
const MIXED: &str = "SELECT SUM(value) AS total, AVG(value) AS mean FROM events";
const WITH_COUNTS: &str = "SELECT SUM(value) AS total, AVG(value) AS mean, COUNT(value) AS valid, COUNT(*) AS arrivals FROM events";
const WITH_EXTREMA: &str = "SELECT SUM(value) AS total, AVG(value) AS mean, MIN(value) AS lo, MAX(value) AS hi, COUNT(value) AS valid FROM events";
const LARGE: Bits = (0x5a80_0000, 0x4350_0000_0000_0000);
const NEG_LARGE: Bits = (0xda80_0000, 0xc350_0000_0000_0000);

fn cases() -> [(DataType, &'static str); 12] {
    [
        (DataType::Float32, SUM),
        (DataType::Float64, SUM),
        (DataType::Float32, AVG),
        (DataType::Float64, AVG),
        (DataType::Float32, MIXED),
        (DataType::Float64, MIXED),
        (DataType::Float32, WITH_COUNTS),
        (DataType::Float64, WITH_COUNTS),
        (DataType::Float32, WITH_EXTREMA),
        (DataType::Float64, WITH_EXTREMA),
        (
            DataType::Float32,
            "SELECT AVG(value) AS mean, SUM(value) AS total, AVG(value) AS again FROM events",
        ),
        (
            DataType::Float64,
            "SELECT AVG(value) AS mean, SUM(value) AS total, AVG(value) AS again FROM events",
        ),
    ]
}

fn finite_arrivals(reverse: bool) -> Vec<Vec<Part>> {
    let (first, second) = if reverse {
        (NEG_ZERO, ZERO)
    } else {
        (ZERO, NEG_ZERO)
    };
    let mut boundary = vec![Some(ZERO); 8194];
    boundary[0] = Some(LARGE);
    boundary[8192] = Some(NEG_LARGE);
    boundary[8193] = Some(ONE);
    vec![
        vec![vec![Some(first), None, Some(second)]],
        vec![boundary],
        vec![vec![Some(LARGE)], vec![Some(NEG_LARGE), Some(ONE)]],
        vec![vec![None; 7], vec![]],
    ]
}

fn special_arrivals() -> Vec<Vec<Part>> {
    vec![
        vec![vec![None; 3]],
        vec![vec![]],
        vec![vec![Some(NEG_ZERO), Some(ZERO), Some((1, 1)), None]],
        vec![vec![Some(POS_INF)], vec![Some(NEG_INF), Some(SNAN)]],
        vec![vec![Some(NAN_A), None], vec![Some(NEG_NAN), Some(NEG_SNAN)]],
    ]
}

async fn capture(
    state: &mut SqlOperator,
    dtype: &DataType,
    query: &str,
    parts: &[Part],
    sequence: u64,
) -> OperatorStateSnapshot {
    assert!(
        state.incremental.is_some() && state.compact.is_some() && state.retained.is_none(),
        "global floating SUM/AVG must own original-record native state"
    );
    let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["state_layout"], json!(3));
    assert_eq!(snapshot.inline_metadata["state_accounting"], json!(3));
    assert_eq!(
        snapshot.inline_metadata["rows"],
        json!(parts.iter().map(Vec::len).sum::<usize>())
    );
    assert_eq!(
        snapshot
            .segments
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["batch-metadata", "control", "group-state", "logical-schema"]
    );
    let projection = state.compact.as_ref().unwrap().projection().unwrap();
    assert_eq!(projection.columns.logical_schema(), &schema(dtype));
    assert_eq!(projection.columns.ordinals(), &[1]);
    assert_eq!(projection.columns.physical_schema().fields().len(), 1);
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .global_record_accumulator_state(query, &input(dtype, parts, sequence))
        .await;
    let values = expected
        .iter()
        .map(|value| cell(&value.to_array_of_size(1).unwrap(), 0))
        .collect::<Vec<_>>();
    let wire = decode_sql_state(snapshot.segments["group-state"].bytes()).unwrap();
    assert_eq!(rows(&wire), vec![values]);
    let fields = wire.table_payload().unwrap().schema().fields();
    let descriptor = state
        .incremental
        .as_ref()
        .unwrap()
        .native_descriptor("float_extrema")
        .unwrap();
    assert_eq!(fields, descriptor.wire_schema.fields());
    for (field, value) in fields.iter().zip(expected) {
        assert_eq!(field.data_type(), &value.data_type());
        assert_eq!(field.is_nullable(), value.data_type() != DataType::Int64);
    }
    snapshot
}

async fn prefixes(dtype: &DataType, query: &str, arrivals: Vec<Vec<Part>>) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = operator(dtype, query);
    let mut prefix = Vec::new();
    let mut weak = Vec::new();
    let final_sequence = u64::try_from(arrivals.len() - 1).unwrap();
    for (sequence, parts) in arrivals.into_iter().enumerate() {
        let batch = input(dtype, &parts, sequence as u64);
        weak.extend(weak_arrays(&batch));
        prefix.extend(parts);
        let actual = process(&mut state, batch, &context).await;
        assert_oracle(&actual, query, dtype, &prefix, sequence as u64).await;
    }
    let snapshot = capture(&mut state, dtype, query, &prefix, final_sequence).await;
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((state, snapshot));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn test_global_float_sum_average_original_record_prefix_bits_native3() {
    for (dtype, query) in cases() {
        for reverse in [false, true] {
            prefixes(&dtype, query, finite_arrivals(reverse)).await;
        }
        prefixes(&dtype, query, special_arrivals()).await;
    }
}

#[tokio::test]
async fn test_global_float_sum_average_count_native_original_record_state() {
    for dtype in [DataType::Float32, DataType::Float64] {
        prefixes(&dtype, WITH_COUNTS, finite_arrivals(false)).await;
        prefixes(&dtype, WITH_COUNTS, special_arrivals()).await;
    }
}

#[tokio::test]
async fn test_global_float_sum_average_extrema_native_original_record_state() {
    for dtype in [DataType::Float32, DataType::Float64] {
        prefixes(&dtype, WITH_EXTREMA, finite_arrivals(false)).await;
        prefixes(&dtype, WITH_EXTREMA, special_arrivals()).await;
        roundtrip(&dtype, WITH_EXTREMA).await;
    }
}

async fn roundtrip(dtype: &DataType, query: &str) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = operator(dtype, query);
    let mut prefix = Vec::new();
    for (sequence, parts) in finite_arrivals(false).into_iter().enumerate() {
        let batch = input(dtype, &parts, sequence as u64);
        prefix.extend(parts);
        let actual = process(&mut state, batch, &context).await;
        assert_oracle(&actual, query, dtype, &prefix, sequence as u64).await;
    }
    let before = capture(&mut state, dtype, query, &prefix, 3).await;
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop(state);
    let mut restored = operator(dtype, query);
    StreamOperator::restore(&mut restored, &before).unwrap();
    same_snapshot(&before, &restored.checkpoint(Epoch::INITIAL).unwrap());
    let empty = process(&mut restored, input(dtype, &[vec![]], 4), &context).await;
    assert_oracle(&empty, query, dtype, &prefix, 4).await;
    let empty_capture = capture(&mut restored, dtype, query, &prefix, 4).await;
    let next = vec![vec![Some(LARGE)], vec![Some(NEG_LARGE), Some(ONE), None]];
    let batch = input(dtype, &next, 5);
    let weak = weak_arrays(&batch);
    let actual = process(&mut restored, batch, &context).await;
    prefix.extend(next);
    assert_oracle(&actual, query, dtype, &prefix, 5).await;
    let after = capture(&mut restored, dtype, query, &prefix, 5).await;
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    let target = restored
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    drop((restored, before, empty, empty_capture, actual, after));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
    assert_eq!(target.reserved(), 0);
}

#[tokio::test]
async fn test_global_float_sum_average_native3_restore_empty_then_original_records() {
    for (dtype, query) in cases() {
        roundtrip(&dtype, query).await;
    }
}

#[tokio::test]
async fn test_global_float_sum_average_rejected_emit_refund_then_once_retry() {
    for (dtype, query) in cases() {
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_extrema", None);
        let mut state = operator(&dtype, query);
        let first = vec![vec![Some(LARGE), None]];
        let initial = process(&mut state, input(&dtype, &first, 0), &context).await;
        assert_oracle(&initial, query, &dtype, &first, 0).await;
        let before = state.checkpoint(Epoch::INITIAL).unwrap();
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        let reserved = pool.reserved();
        let next = vec![vec![Some(NEG_LARGE), Some(ONE), None]];
        let rejected = input(&dtype, &next, 1);
        let weak = weak_arrays(&rejected);
        let failure = state
            .process_data("events", rejected, &context, &mut Reject)
            .await;
        assert!(
            matches!(failure, Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject-extrema")
        );
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        assert_eq!(pool.reserved(), reserved);
        same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
        let actual = process(&mut state, input(&dtype, &next, 1), &context).await;
        let all = first.into_iter().chain(next).collect::<Vec<_>>();
        assert_oracle(&actual, query, &dtype, &all, 1).await;
        let after = capture(&mut state, &dtype, query, &all, 1).await;
        drop((state, before, after, initial, actual));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test]
async fn test_global_float_mixed_sum_average_releases_inputs_native3() {
    for dtype in [DataType::Float32, DataType::Float64] {
        let job = job();
        let context = StreamOperatorContext::new(&job, "float_extrema", None);
        let mut state = operator(&dtype, MIXED);
        let first = finite_arrivals(false).remove(1);
        let batch = input(&dtype, &first, 0);
        let weak = weak_arrays(&batch);
        let actual = process(&mut state, batch, &context).await;
        assert_oracle(&actual, MIXED, &dtype, &first, 0).await;
        let before = capture(&mut state, &dtype, MIXED, &first, 0).await;
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        drop(actual);
        let pool = state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop(state);
        let mut restored = operator(&dtype, MIXED);
        StreamOperator::restore(&mut restored, &before).unwrap();
        same_snapshot(&before, &restored.checkpoint(Epoch::INITIAL).unwrap());
        let next = vec![vec![Some(LARGE)], vec![Some(NEG_LARGE), Some(ONE)]];
        let actual = process(&mut restored, input(&dtype, &next, 1), &context).await;
        let all = first.into_iter().chain(next).collect::<Vec<_>>();
        assert_oracle(&actual, MIXED, &dtype, &all, 1).await;
        let after = capture(&mut restored, &dtype, MIXED, &all, 1).await;
        let target = restored
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool();
        drop((restored, before, after, actual));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(context);
        drop(job);
        assert_eq!(pool.reserved(), 0);
        assert_eq!(target.reserved(), 0);
    }
}

fn two_columns(parts: &[Part], sequence: u64) -> Batch {
    let first = input(&DataType::Float32, parts, sequence);
    let original = first.table_payload().unwrap();
    let mut fields = original.schema().fields().to_vec();
    fields.push(Arc::new(Field::new("other", DataType::Float64, true)));
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        original.schema().metadata().clone(),
    ));
    let records = original
        .batches()
        .iter()
        .zip(parts)
        .map(|(record, part)| {
            let mut arrays = record.columns().to_vec();
            arrays.push(Arc::new(Float64Array::from(
                part.iter()
                    .map(|value| {
                        if value.is_none() {
                            Some(2.0)
                        } else {
                            Some(-3.0)
                        }
                    })
                    .collect::<Vec<_>>(),
            )));
            RecordBatch::try_new(schema.clone(), arrays).unwrap()
        })
        .collect();
    Batch::table(records, metadata(sequence)).unwrap()
}

#[tokio::test]
async fn test_global_float_multiple_columns_native3_prefix_restore() {
    let query =
        "SELECT AVG(other) AS a, SUM(value) AS b, SUM(other) AS c, AVG(value) AS d FROM events";
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state =
        SqlOperator::new("float_extrema", query, vec!["events".into()], vec![]).unwrap();
    let mut prefix = Vec::new();
    let mut pools = Vec::new();
    for (sequence, parts) in finite_arrivals(false).into_iter().enumerate() {
        let batch = two_columns(&parts, sequence as u64);
        let weak = weak_arrays(&batch);
        let actual = process(&mut state, batch, &context).await;
        prefix.extend(parts);
        let oracle_input = two_columns(&prefix, sequence as u64);
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let expected = runtime
            .sql(
                query,
                &BTreeMap::from([("events".into(), oracle_input.clone())]),
                Some("oracle"),
            )
            .await
            .unwrap();
        assert_eq!(rows(&actual), rows(&expected));
        assert_eq!(
            actual.table_payload().unwrap().schema(),
            expected.table_payload().unwrap().schema()
        );
        assert_eq!(actual.metadata(), expected.metadata());
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(snapshot.inline_metadata["state_layout"], json!(3));
        assert_eq!(snapshot.inline_metadata["state_accounting"], json!(3));
        assert!(!snapshot.segments.contains_key("input-retained"));
        let wire = decode_sql_state(snapshot.segments["group-state"].bytes()).unwrap();
        let scalar_state = runtime
            .global_record_accumulator_state(query, &oracle_input)
            .await;
        assert_eq!(
            rows(&wire),
            vec![
                scalar_state
                    .iter()
                    .map(|value| cell(&value.to_array_of_size(1).unwrap(), 0))
                    .collect::<Vec<_>>()
            ]
        );
        assert_eq!(
            state
                .compact
                .as_ref()
                .unwrap()
                .projection()
                .unwrap()
                .columns
                .ordinals(),
            &[1, 2]
        );
        pools.push(
            state
                .stream_state
                .runtime()
                .unwrap()
                .incremental_memory_pool(),
        );
        drop(state);
        state = SqlOperator::new("float_extrema", query, vec!["events".into()], vec![]).unwrap();
        StreamOperator::restore(&mut state, &snapshot).unwrap();
        same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
    }
    pools.push(
        state
            .stream_state
            .runtime()
            .unwrap()
            .incremental_memory_pool(),
    );
    drop(state);
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(
        pools.iter().map(|pool| pool.reserved()).collect::<Vec<_>>(),
        vec![0; pools.len()]
    );
}
