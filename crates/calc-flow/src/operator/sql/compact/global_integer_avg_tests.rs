use super::*;
use datafusion::arrow::array::{
    Int8Array, Int16Array, Int32Array, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};

const MULTIPLE: &str =
    "SELECT AVG(value) AS mean, AVG(other) AS other_mean, AVG(value) AS repeated FROM events";
const MIXED_NUMERIC: &str =
    "SELECT SUM(price) AS total, AVG(value) AS mean, AVG(other) AS other_mean FROM events";
const RANGES: [(usize, usize); 4] = [(0, 8194), (0, 3), (8192, 2), (0, 0)];

fn integer_columns() -> [ArrayRef; 8] {
    macro_rules! column {
        ($array:ident, $native:ty) => {{
            let mut values = vec![Some(0 as $native); 8194];
            values[0] = Some(<$native>::MAX);
            values[7000] = None;
            values[8192] = Some(<$native>::MIN);
            values[8193] = Some(1);
            Arc::new($array::from(values)) as ArrayRef
        }};
    }
    [
        column!(Int8Array, i8),
        column!(Int16Array, i16),
        column!(Int32Array, i32),
        column!(Int64Array, i64),
        column!(UInt8Array, u8),
        column!(UInt16Array, u16),
        column!(UInt32Array, u32),
        column!(UInt64Array, u64),
    ]
}

fn integer_input(column: &ArrayRef, ranges: &[(usize, usize)], sequence: u64) -> Batch {
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("value", column.data_type().clone(), true)
                .with_metadata([("unit".into(), "integer".into())].into()),
            Field::new("other", DataType::UInt64, true),
            Field::new("price", DataType::Float32, true),
        ],
        [("origin".into(), "integer-avg".into())].into(),
    ));
    let records = ranges
        .iter()
        .map(|&(offset, length)| {
            let other = UInt64Array::from(
                (offset..offset + length)
                    .map(|row| (row % 3 != 0).then_some(if row % 2 == 0 { u64::MAX } else { 1 }))
                    .collect::<Vec<_>>(),
            );
            let prices = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875];
            let price = Float32Array::from(
                (offset..offset + length)
                    .map(|row| (row % 5 != 0).then_some(prices[row % 8]))
                    .collect::<Vec<_>>(),
            );
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    column.slice(offset, length),
                    Arc::new(other),
                    Arc::new(price),
                ],
            )
            .unwrap()
        })
        .collect();
    Batch::table(records, metadata(sequence)).unwrap()
}

fn integer_operator(query: &str) -> SqlOperator {
    SqlOperator::new("float_extrema", query, vec!["events".into()], vec![]).unwrap()
}

async fn integer_prefix(column: &ArrayRef, query: &str) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = integer_operator(query);
    let mut pools = Vec::new();
    for (sequence, range) in RANGES.iter().enumerate() {
        let input = integer_input(column, &[*range], sequence as u64);
        let weak = weak_arrays(&input);
        let actual = process(&mut state, input, &context).await;
        let prefix = integer_input(column, &RANGES[..=sequence], sequence as u64);
        let runtime = DataFusionRuntime::new(DataFusionConfig::default()).unwrap();
        let expected = runtime
            .sql(
                query,
                &BTreeMap::from([("events".into(), prefix.clone())]),
                Some("integer-oracle"),
            )
            .await
            .unwrap();
        assert_eq!(rows(&actual), rows(&expected));
        assert_eq!(
            actual.table_payload().unwrap().schema(),
            expected.table_payload().unwrap().schema()
        );
        assert_eq!(actual.metadata(), expected.metadata());
        assert!(
            state.incremental.is_some() && state.retained.is_none(),
            "global integer AVG must discard input history"
        );
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(snapshot.inline_metadata["state_layout"], json!(3));
        assert_eq!(snapshot.inline_metadata["state_accounting"], json!(3));
        assert!(!snapshot.segments.contains_key("input-retained"));
        let wire = decode_sql_state(snapshot.segments["group-state"].bytes()).unwrap();
        let values = runtime
            .global_record_accumulator_state(query, &prefix)
            .await;
        assert_eq!(
            rows(&wire),
            vec![
                values
                    .iter()
                    .map(|value| cell(&value.to_array_of_size(1).unwrap(), 0))
                    .collect::<Vec<_>>()
            ]
        );
        pools.push(
            state
                .stream_state
                .runtime()
                .unwrap()
                .incremental_memory_pool(),
        );
        drop(state);
        state = integer_operator(query);
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

#[tokio::test]
async fn test_global_integer_average_original_record_bits_native3() {
    for column in integer_columns() {
        for query in [AVG, MULTIPLE] {
            integer_prefix(&column, query).await;
        }
    }
}

#[tokio::test]
async fn test_global_integer_average_nondefault_empty_and_grouped_controls() {
    let column = integer_columns().into_iter().next().unwrap();
    let mut nondefault = integer_operator(AVG);
    nondefault.set_stream_resources(
        DataFusionConfig {
            target_partitions: 4,
            ..DataFusionConfig::default()
        },
        UdfRegistrySnapshot::default(),
        vec![],
    );
    for (state, range, query) in [
        (nondefault, (0, 3), AVG),
        (integer_operator(AVG), (0, 0), AVG),
        (
            integer_operator("SELECT other, AVG(value) AS mean FROM events GROUP BY other"),
            (0, 3),
            "SELECT other, AVG(value) AS mean FROM events GROUP BY other",
        ),
    ] {
        global_record_controls::raw_capture(state, integer_input(&column, &[range], 0), query)
            .await;
    }
}

#[tokio::test]
async fn test_global_integer_average_mixed_float_sum_native3() {
    let columns = integer_columns();
    for index in [3, 7] {
        integer_prefix(&columns[index], MIXED_NUMERIC).await;
    }
}

#[tokio::test]
async fn test_global_integer_average_emit_refund_and_once_retry() {
    let column = integer_columns()[3].clone();
    let job = job();
    let context = StreamOperatorContext::new(&job, "float_extrema", None);
    let mut state = integer_operator(MIXED_NUMERIC);
    drop(
        process(
            &mut state,
            integer_input(&column, &RANGES[..1], 0),
            &context,
        )
        .await,
    );
    let before = state.checkpoint(Epoch::INITIAL).unwrap();
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let reserved = pool.reserved();
    let batch = integer_input(&column, &RANGES[1..2], 1);
    let weak = weak_arrays(&batch);
    let failure = state
        .process_data("events", batch, &context, &mut Reject)
        .await;
    assert!(
        matches!(failure, Err(CalcFlowError::Operator { node_id, .. }) if node_id == "reject-extrema")
    );
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    assert_eq!(pool.reserved(), reserved);
    same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
    let actual = process(
        &mut state,
        integer_input(&column, &RANGES[1..2], 1),
        &context,
    )
    .await;
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            MIXED_NUMERIC,
            &BTreeMap::from([("events".into(), integer_input(&column, &RANGES[..2], 1))]),
            Some("integer-retry-oracle"),
        )
        .await
        .unwrap();
    assert_eq!(rows(&actual), rows(&expected));
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(actual.metadata(), expected.metadata());
    drop((state, before, actual, expected));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}
