use super::*;
use datafusion::arrow::array::BooleanArray;

const QUERY: &str = "SELECT key, bucket, MIN(value) AS lo, MAX(value) AS hi, COUNT(value) AS valid, COUNT(*) AS rows, SUM(value) AS total, AVG(value) AS mean FROM events GROUP BY key, bucket";

fn composite_schema(dtype: &DataType, key_type: &DataType, bucket_type: &DataType) -> SchemaRef {
    let original = string_schema(dtype, key_type);
    let mut fields = original.fields().to_vec();
    fields[1] = Arc::new(Field::new("bucket", bucket_type.clone(), true));
    Arc::new(Schema::new_with_metadata(
        fields,
        original.metadata().clone(),
    ))
}

fn composite_input(
    dtype: &DataType,
    key_type: &DataType,
    bucket_type: &DataType,
    parts: &[Part],
    sequence: u64,
) -> Batch {
    rebucket(&string_input(dtype, key_type, parts, sequence), bucket_type)
}

fn rebucket(original: &Batch, bucket_type: &DataType) -> Batch {
    let records = original
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .map(|record| {
            let values = (0..record.num_rows())
                .map(|index| match index % 3 {
                    0 => Some(0),
                    1 => Some(1),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let bucket: ArrayRef = match bucket_type {
                DataType::Int64 => Arc::new(Int64Array::from(values)),
                DataType::Boolean => Arc::new(BooleanArray::from(
                    values
                        .into_iter()
                        .map(|value| value.map(|value| value != 0))
                        .collect::<Vec<_>>(),
                )),
                _ => unreachable!(),
            };
            let mut columns = record.columns().to_vec();
            columns[1] = bucket;
            let schema = record.schema();
            let mut fields = schema.fields().to_vec();
            fields[1] = Arc::new(Field::new("bucket", bucket_type.clone(), true));
            let schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
            RecordBatch::try_new(schema, columns).unwrap()
        })
        .collect();
    Batch::table(records, original.metadata().clone()).unwrap()
}

fn composite_operator(
    dtype: &DataType,
    key_type: &DataType,
    bucket_type: &DataType,
) -> SqlOperator {
    SqlOperator::new("grouped_float", QUERY, vec!["events".into()], vec![])
        .unwrap()
        .with_ports(
            vec![
                Port::with_schema_ref(
                    "events",
                    BatchKind::Table,
                    true,
                    Some(composite_schema(dtype, key_type, bucket_type)),
                )
                .unwrap(),
            ],
            Port::new("output", BatchKind::Table, true, None).unwrap(),
        )
        .unwrap()
}

fn composite_rows(batch: &Batch) -> BTreeMap<(Option<String>, Option<i64>), Vec<Cell>> {
    let mut result = BTreeMap::new();
    for record in batch.table_payload().unwrap().batches() {
        for row in 0..record.num_rows() {
            let key = match ScalarValue::try_from_array(record.column(0), row).unwrap() {
                ScalarValue::Utf8(value) | ScalarValue::LargeUtf8(value) => value,
                value => panic!("unexpected key: {value:?}"),
            };
            let bucket = match ScalarValue::try_from_array(record.column(1), row).unwrap() {
                ScalarValue::Int64(value) => value,
                ScalarValue::Boolean(value) => value.map(i64::from),
                value => panic!("unexpected bucket: {value:?}"),
            };
            let values = record
                .columns()
                .iter()
                .map(|array| cell(array, row))
                .collect();
            assert!(result.insert((key, bucket), values).is_none());
        }
    }
    result
}

async fn composite_oracle(actual: &Batch, prefix: Batch) {
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            QUERY,
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
    assert_eq!(composite_rows(actual), composite_rows(&expected));
}

fn composite_capture(state: &mut SqlOperator) -> OperatorStateSnapshot {
    assert!(
        state.incremental.is_some() && state.compact.is_some() && state.retained.is_none(),
        "composite SUM/AVG must own incremental state"
    );
    let snapshot = state.checkpoint(Epoch::INITIAL).unwrap();
    let control: Value = serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
    assert_eq!(
        control["state_policy"]["sequential_grouped_float_v1"]["factory"],
        json!("column_v1")
    );
    snapshot
}

#[tokio::test]
async fn test_composite_key_sum_average_exact_prefix_and_cold_continuation() {
    for dtype in [DataType::Float32, DataType::Float64] {
        for key_type in [DataType::Utf8, DataType::LargeUtf8] {
            for bucket_type in [DataType::Int64, DataType::Boolean] {
                let job = job();
                let context = StreamOperatorContext::new(&job, "grouped_float", None);
                let mut state = composite_operator(&dtype, &key_type, &bucket_type);
                let mut prefix = Vec::new();
                let mut pools = Vec::new();
                for (sequence, parts) in string_arrivals().into_iter().enumerate() {
                    let batch =
                        composite_input(&dtype, &key_type, &bucket_type, &parts, sequence as u64);
                    let weak = weak_arrays(&batch);
                    let actual = process(&mut state, batch, &context).await;
                    prefix.extend(parts);
                    composite_oracle(
                        &actual,
                        composite_input(&dtype, &key_type, &bucket_type, &prefix, sequence as u64),
                    )
                    .await;
                    let snapshot = composite_capture(&mut state);
                    assert!(weak.iter().all(|array| array.upgrade().is_none()));
                    pools.push(
                        state
                            .stream_state
                            .runtime()
                            .unwrap()
                            .incremental_memory_pool(),
                    );
                    drop(state);
                    state = composite_operator(&dtype, &key_type, &bucket_type);
                    StreamOperator::restore(&mut state, &snapshot).unwrap();
                    same_snapshot(&snapshot, &state.checkpoint(Epoch::INITIAL).unwrap());
                }
                let actual = process(
                    &mut state,
                    composite_input(&dtype, &key_type, &bucket_type, &[vec![]], 3),
                    &context,
                )
                .await;
                composite_oracle(
                    &actual,
                    composite_input(&dtype, &key_type, &bucket_type, &prefix, 3),
                )
                .await;
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
}

#[tokio::test]
async fn test_composite_key_integer_sum_average_exact_prefix_and_restore() {
    use super::super::grouped_integer_avg_tests::{average_arrivals, average_input};
    for dtype in [DataType::Int64, DataType::UInt64] {
        for key_type in [DataType::Utf8, DataType::LargeUtf8] {
            let bucket_type = DataType::Int64;
            let job = job();
            let context = StreamOperatorContext::new(&job, "grouped_float", None);
            let mut state = composite_operator(&dtype, &key_type, &bucket_type);
            let mut prefix = Vec::new();
            let mut pools = Vec::new();
            for (sequence, parts) in average_arrivals(&dtype).into_iter().enumerate() {
                let batch = rebucket(
                    &rekey_batch(&average_input(&dtype, &parts, sequence as u64), &key_type),
                    &bucket_type,
                );
                let weak = weak_arrays(&batch);
                let actual = process(&mut state, batch, &context).await;
                prefix.extend(parts);
                composite_oracle(
                    &actual,
                    rebucket(
                        &rekey_batch(&average_input(&dtype, &prefix, sequence as u64), &key_type),
                        &bucket_type,
                    ),
                )
                .await;
                let snapshot = composite_capture(&mut state);
                assert!(weak.iter().all(|array| array.upgrade().is_none()));
                pools.push(
                    state
                        .stream_state
                        .runtime()
                        .unwrap()
                        .incremental_memory_pool(),
                );
                drop(state);
                state = composite_operator(&dtype, &key_type, &bucket_type);
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

#[path = "grouped_composite_safety_tests.rs"]
mod safety;
