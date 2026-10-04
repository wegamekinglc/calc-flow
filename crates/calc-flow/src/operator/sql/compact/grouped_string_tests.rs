use super::*;
use datafusion::arrow::array::LargeStringArray;

const MIXED: &str = "SELECT key, SUM(value) AS total, AVG(value) AS mean FROM events GROUP BY key";
const ALL: &str = "SELECT key, MIN(value) AS lo, MAX(value) AS hi, COUNT(value) AS valid, COUNT(*) AS rows, SUM(value) AS total, AVG(value) AS mean FROM events GROUP BY key";

fn string_schema(dtype: &DataType, key_type: &DataType) -> SchemaRef {
    let original = schema(dtype);
    let mut fields = original.fields().to_vec();
    fields[0] = Arc::new(fields[0].as_ref().clone().with_data_type(key_type.clone()));
    Arc::new(Schema::new_with_metadata(
        fields,
        original.metadata().clone(),
    ))
}

fn key_name(key: i64) -> String {
    match key {
        1 => String::new(),
        2 => "A\0B".into(),
        3 => "é".into(),
        4 => "e\u{301}".into(),
        7 | 8 => format!("{}-{key}", "长键".repeat(1000)),
        _ => format!("标的😀-{key}"),
    }
}

fn string_arrivals() -> Vec<Vec<Part>> {
    let mut parts = arrivals(false);
    parts[0][0].extend([
        (Some(3), Some(ONE)),
        (Some(4), Some(TWO)),
        (Some(5), Some(NEG_ZERO)),
        (Some(11), Some((0x5a80_0000, 0x4350_0000_0000_0000))),
    ]);
    parts[1][0].extend([
        (Some(3), Some(NEG_ZERO)),
        (Some(4), Some(ONE)),
        (Some(11), Some((0xda80_0000, 0xc350_0000_0000_0000))),
        (Some(11), Some(ONE)),
    ]);
    parts
}

fn string_input(dtype: &DataType, key_type: &DataType, parts: &[Part], sequence: u64) -> Batch {
    rekey_batch(&input(dtype, parts, sequence), key_type)
}

fn rekey_batch(original: &Batch, key_type: &DataType) -> Batch {
    let records = original
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .map(|record| {
            let keys = record
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .iter()
                .map(|key| key.map(key_name))
                .collect::<Vec<_>>();
            let keys: ArrayRef = match key_type {
                DataType::Utf8 => Arc::new(StringArray::from(keys)),
                DataType::LargeUtf8 => Arc::new(LargeStringArray::from(keys)),
                _ => unreachable!(),
            };
            let mut columns = record.columns().to_vec();
            columns[0] = keys;
            let schema = record.schema();
            let mut fields = schema.fields().to_vec();
            fields[0] = Arc::new(fields[0].as_ref().clone().with_data_type(key_type.clone()));
            let schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
            RecordBatch::try_new(schema, columns).unwrap()
        })
        .collect();
    Batch::table(records, original.metadata().clone()).unwrap()
}

#[tokio::test]
async fn test_string_key_integer_sum_average_exact_prefix_and_restore() {
    use super::grouped_integer_avg_tests::{average_arrivals, average_input};
    for dtype in [DataType::Int64, DataType::UInt64] {
        for key_type in [DataType::Utf8, DataType::LargeUtf8] {
            let job = job();
            let context = StreamOperatorContext::new(&job, "grouped_float", None);
            let mut state = string_operator(&dtype, &key_type, ALL);
            let mut prefix = Vec::new();
            let mut pools = Vec::new();
            for (sequence, parts) in average_arrivals(&dtype).into_iter().enumerate() {
                let batch = rekey_batch(&average_input(&dtype, &parts, sequence as u64), &key_type);
                let weak = weak_arrays(&batch);
                let actual = process(&mut state, batch, &context).await;
                prefix.extend(parts);
                string_oracle(
                    &actual,
                    ALL,
                    rekey_batch(&average_input(&dtype, &prefix, sequence as u64), &key_type),
                )
                .await;
                let snapshot = string_capture(&mut state, &key_type);
                assert!(weak.iter().all(|array| array.upgrade().is_none()));
                pools.push(
                    state
                        .stream_state
                        .runtime()
                        .unwrap()
                        .incremental_memory_pool(),
                );
                drop(state);
                state = string_operator(&dtype, &key_type, ALL);
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
            assert_eq!(pools.iter().map(|pool| pool.reserved()).sum::<usize>(), 0);
        }
    }
}

fn string_operator(dtype: &DataType, key_type: &DataType, query: &str) -> SqlOperator {
    SqlOperator::new("grouped_float", query, vec!["events".into()], vec![])
        .unwrap()
        .with_ports(
            vec![
                Port::with_schema_ref(
                    "events",
                    BatchKind::Table,
                    true,
                    Some(string_schema(dtype, key_type)),
                )
                .unwrap(),
            ],
            Port::new("output", BatchKind::Table, true, None).unwrap(),
        )
        .unwrap()
}

fn string_rows(batch: &Batch) -> BTreeMap<Option<String>, Vec<Cell>> {
    let mut result = BTreeMap::new();
    for record in batch.table_payload().unwrap().batches() {
        for row in 0..record.num_rows() {
            let key = match ScalarValue::try_from_array(record.column(0), row).unwrap() {
                ScalarValue::Utf8(value) | ScalarValue::LargeUtf8(value) => value,
                value => panic!("unexpected group key: {value:?}"),
            };
            let values = record
                .columns()
                .iter()
                .map(|array| cell(array, row))
                .collect();
            assert!(result.insert(key, values).is_none());
        }
    }
    result
}

async fn string_oracle(actual: &Batch, query: &str, prefix: Batch) {
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
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
    assert_eq!(string_rows(actual), string_rows(&expected));
}

fn string_capture(state: &mut SqlOperator, key_type: &DataType) -> OperatorStateSnapshot {
    assert!(
        state.incremental.is_some() && state.compact.is_some() && state.retained.is_none(),
        "string SUM/AVG must own native state instead of retained input"
    );
    let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
    let control: Value = serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
    let factory = match key_type {
        DataType::Utf8 => "utf8_v1",
        DataType::LargeUtf8 => "large_utf8_v1",
        _ => unreachable!(),
    };
    assert_eq!(
        control["state_policy"]["sequential_grouped_float_v1"]["factory"],
        json!(factory)
    );
    snapshot
}

#[tokio::test]
async fn test_string_key_sum_average_exact_prefix_and_cold_continuation() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for key_type in [DataType::Utf8, DataType::LargeUtf8] {
            for query in [MIXED, ALL] {
                let job = job();
                let context = StreamOperatorContext::new(&job, "grouped_float", None);
                let mut state = string_operator(&dtype, &key_type, query);
                let mut prefix = Vec::new();
                let mut pools = Vec::new();
                for (sequence, parts) in string_arrivals().into_iter().enumerate() {
                    let batch = string_input(&dtype, &key_type, &parts, sequence as u64);
                    let weak = weak_arrays(&batch);
                    let actual = process(&mut state, batch, &context).await;
                    prefix.extend(parts);
                    string_oracle(
                        &actual,
                        query,
                        string_input(&dtype, &key_type, &prefix, sequence as u64),
                    )
                    .await;
                    let snapshot = string_capture(&mut state, &key_type);
                    assert!(weak.iter().all(|array| array.upgrade().is_none()));
                    pools.push(
                        state
                            .stream_state
                            .runtime()
                            .unwrap()
                            .incremental_memory_pool(),
                    );
                    drop(state);
                    state = string_operator(&dtype, &key_type, query);
                    StreamOperator::restore(&mut state, &snapshot).unwrap();
                    same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
                }
                let empty = process(
                    &mut state,
                    string_input(&dtype, &key_type, &[vec![]], 3),
                    &context,
                )
                .await;
                string_oracle(&empty, query, string_input(&dtype, &key_type, &prefix, 3)).await;
                pools.push(
                    state
                        .stream_state
                        .runtime()
                        .unwrap()
                        .incremental_memory_pool(),
                );
                drop(state);
                assert_eq!(pools.iter().map(|pool| pool.reserved()).sum::<usize>(), 0);
                assert!(job.gather_owner().close_and_drain().await.is_empty());
            }
        }
    }
}

#[path = "grouped_string_safety_tests.rs"]
mod safety;
