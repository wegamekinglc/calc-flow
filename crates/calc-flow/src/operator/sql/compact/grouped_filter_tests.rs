use super::*;
use datafusion::{arrow::array::BooleanArray, execution::memory_pool::MemoryConsumer};

#[path = "grouped_where_tests.rs"]
mod predicates;

#[path = "delta_float_tests.rs"]
mod deltas;

const AGGREGATES: &str = "SUM(value) FILTER (WHERE accepted) AS total, AVG(value) FILTER (WHERE accepted) AS mean, SUM(value) FILTER (WHERE other) AS other_total, MIN(value) FILTER (WHERE other) AS lo, MAX(value) FILTER (WHERE other) AS hi, COUNT(value) FILTER (WHERE accepted) AS valid, COUNT(*) FILTER (WHERE other) AS selected, COUNT(*) AS rows";
const COUNTS: &str = "SELECT key, COUNT(value) FILTER (WHERE accepted) AS valid, COUNT(*) FILTER (WHERE other) AS selected, COUNT(*) AS rows FROM events GROUP BY key";

fn query(composite: bool) -> String {
    let keys = if composite { "key, other" } else { "key" };
    format!("SELECT {keys}, {AGGREGATES} FROM events GROUP BY {keys}")
}

fn filtered_schema(dtype: &DataType, key_type: &DataType) -> SchemaRef {
    let original = if key_type == &DataType::Int64 {
        schema(dtype)
    } else {
        string_schema(dtype, key_type)
    };
    let mut fields = original.fields().to_vec();
    fields[2] = Arc::new(Field::new("accepted", DataType::Boolean, true));
    fields[3] = Arc::new(Field::new("other", DataType::Boolean, true));
    Arc::new(Schema::new_with_metadata(
        fields,
        original.metadata().clone(),
    ))
}

fn filtered_input(dtype: &DataType, key_type: &DataType, parts: &[Part], sequence: u64) -> Batch {
    let original = if key_type == &DataType::Int64 {
        input(dtype, parts, sequence)
    } else {
        string_input(dtype, key_type, parts, sequence)
    };
    let phase = usize::try_from(sequence % 3).unwrap();
    let records = original
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .map(|record| {
            let accepted = (0..record.num_rows())
                .map(|row| [Some(true), Some(false), None][(row + phase) % 3])
                .collect::<Vec<_>>();
            let other = (0..record.num_rows())
                .map(|row| [None, Some(true), Some(false)][(row + phase) % 3])
                .collect::<Vec<_>>();
            let mut columns = record.columns().to_vec();
            columns[2] = Arc::new(BooleanArray::from(accepted));
            columns[3] = Arc::new(BooleanArray::from(other));
            RecordBatch::try_new(filtered_schema(dtype, key_type), columns).unwrap()
        })
        .collect();
    Batch::table(records, original.metadata().clone()).unwrap()
}

fn filtered_operator(dtype: &DataType, key_type: &DataType, text: &str) -> SqlOperator {
    SqlOperator::new("grouped_float", text, vec!["events".into()], vec![])
        .unwrap()
        .with_ports(
            vec![
                Port::with_schema_ref(
                    "events",
                    BatchKind::Table,
                    true,
                    Some(filtered_schema(dtype, key_type)),
                )
                .unwrap(),
            ],
            Port::new("output", BatchKind::Table, true, None).unwrap(),
        )
        .unwrap()
}

fn filtered_rows(batch: &Batch, keys: usize) -> BTreeMap<Vec<String>, Vec<Cell>> {
    let mut rows = BTreeMap::new();
    for record in batch.table_payload().unwrap().batches() {
        for row in 0..record.num_rows() {
            let key = record.columns()[..keys]
                .iter()
                .map(|array| format!("{:?}", ScalarValue::try_from_array(array, row).unwrap()))
                .collect();
            let values = record
                .columns()
                .iter()
                .map(|array| cell(array, row))
                .collect();
            assert!(rows.insert(key, values).is_none());
        }
    }
    rows
}

async fn oracle(
    actual: &Batch,
    text: &str,
    dtype: &DataType,
    key_type: &DataType,
    history: &[Vec<Part>],
    keys: usize,
) {
    let last = u64::try_from(history.len() - 1).unwrap();
    let records = history
        .iter()
        .enumerate()
        .flat_map(|(sequence, parts)| {
            filtered_input(dtype, key_type, parts, sequence as u64)
                .table_payload()
                .unwrap()
                .batches()
                .to_vec()
        })
        .collect();
    let template = filtered_input(dtype, key_type, &[vec![]], last);
    let prefix = Batch::table(records, template.metadata().clone()).unwrap();
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            text,
            &BTreeMap::from([("events".into(), prefix)]),
            Some("oracle"),
        )
        .await
        .unwrap();
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(actual.metadata(), expected.metadata());
    assert_eq!(actual.num_rows(), expected.num_rows());
    assert_eq!(filtered_rows(actual, keys), filtered_rows(&expected, keys));
}

fn capture(state: &mut SqlOperator) -> OperatorStateSnapshot {
    assert!(
        state.incremental.is_some() && state.compact.is_some() && state.retained.is_none(),
        "filtered grouped aggregates must retain native state instead of input history"
    );
    let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["state_layout"], json!(3));
    snapshot
}

#[tokio::test]
async fn test_grouped_filtered_float_exact_prefix_and_cold_continuation() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for key_type in [DataType::Int64, DataType::Utf8, DataType::LargeUtf8] {
            for composite in [false, true] {
                case(
                    &dtype,
                    &key_type,
                    &query(composite),
                    if composite { 2 } else { 1 },
                )
                .await;
            }
        }
    }
}

#[tokio::test]
async fn test_grouped_filtered_counts_exact_native_prefix_and_cold_continuation() {
    case(&DataType::Float64, &DataType::Int64, COUNTS, 1).await;
}

async fn case(dtype: &DataType, key_type: &DataType, text: &str, keys: usize) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "grouped_float", None);
    let mut state = filtered_operator(dtype, key_type, text);
    let mut history = Vec::new();
    let mut pools = Vec::new();
    for (sequence, parts) in string_arrivals().into_iter().enumerate() {
        let batch = filtered_input(dtype, key_type, &parts, sequence as u64);
        let weak = weak_arrays(&batch);
        let actual = process(&mut state, batch, &context).await;
        history.push(parts);
        oracle(&actual, text, dtype, key_type, &history, keys).await;
        let snapshot = capture(&mut state);
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        pools.push(
            state
                .stream_state
                .runtime()
                .unwrap()
                .incremental_memory_pool(),
        );
        drop(state);
        state = filtered_operator(dtype, key_type, text);
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
    oracle(&actual, text, dtype, key_type, &history, keys).await;
    capture(&mut state);
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
    assert_eq!(pools.iter().map(|pool| pool.reserved()).sum::<usize>(), 0);
}

#[tokio::test]
async fn test_grouped_filtered_refusal_refunds_then_retries_exactly() {
    refusal(DataType::Float64, DataType::Utf8, query(true)).await;
}

async fn refusal(dtype: DataType, key_type: DataType, text: String) {
    let job = job();
    let context = StreamOperatorContext::new(&job, "grouped_float", None);
    let mut state = filtered_operator(&dtype, &key_type, &text);
    let arrivals = string_arrivals();
    drop(
        process(
            &mut state,
            filtered_input(&dtype, &key_type, &arrivals[0], 0),
            &context,
        )
        .await,
    );
    let snapshot = capture(&mut state);
    let pool = state
        .stream_state
        .runtime()
        .unwrap()
        .incremental_memory_pool();
    let basis = pool.reserved();
    let pressure = MemoryConsumer::new("filtered-update-pressure").register(&pool);
    pressure.try_grow((1 << 30) - basis - 1).unwrap();
    let held = pool.reserved();
    let batch = filtered_input(&dtype, &key_type, &arrivals[1], 1);
    let weak = weak_arrays(&batch);
    let mut output = EdgeCollector::new(state.output_ports().to_vec());
    assert!(
        state
            .process_data("events", batch, &context, &mut output)
            .await
            .is_err()
    );
    assert!(output.drain("output").is_empty());
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    assert_eq!(pool.reserved(), held);
    drop(pressure);
    assert_eq!(pool.reserved(), basis);
    same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
    let batch = filtered_input(&dtype, &key_type, &arrivals[1], 1);
    let weak = weak_arrays(&batch);
    assert!(
        state
            .process_data("events", batch, &context, &mut Reject)
            .await
            .is_err()
    );
    assert!(weak.iter().all(|array| array.upgrade().is_none()));
    assert_eq!(pool.reserved(), basis);
    same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
    let actual = process(
        &mut state,
        filtered_input(&dtype, &key_type, &arrivals[1], 1),
        &context,
    )
    .await;
    oracle(&actual, &text, &dtype, &key_type, &arrivals[..2], 2).await;
    capture(&mut state);
    drop((state, snapshot));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}
