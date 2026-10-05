use std::{collections::BTreeMap, sync::Arc};

use datafusion::{
    arrow::{
        array::{ArrayRef, BooleanArray, Float64Array, Int64Array, LargeStringArray, StringArray},
        datatypes::{DataType, Field, Schema},
    },
    common::ScalarValue,
};

use super::super::*;
use crate::{CancellationToken, EdgeCollector, StreamJobContext};

type Row = (Option<i64>, Option<String>);

fn schema(dtype: &DataType) -> SchemaRef {
    Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("key", DataType::Int64, true),
            Field::new("value", dtype.clone(), true)
                .with_metadata([("unit".into(), "text".into())].into()),
            Field::new("amount", DataType::Float64, true),
            Field::new("selected", DataType::Boolean, false),
            Field::new("unused", DataType::Utf8, false),
        ],
        [("origin".into(), "string-extrema".into())].into(),
    ))
}

fn input(dtype: &DataType, parts: &[Vec<Row>], sequence: u64) -> Batch {
    let records = parts
        .iter()
        .map(|rows| {
            let values = rows.iter().map(|row| row.1.as_deref()).collect::<Vec<_>>();
            let values: ArrayRef = match dtype {
                DataType::Utf8 => Arc::new(StringArray::from(values)),
                DataType::LargeUtf8 => Arc::new(LargeStringArray::from(values)),
                _ => unreachable!(),
            };
            RecordBatch::try_new(
                schema(dtype),
                vec![
                    Arc::new(Int64Array::from(
                        rows.iter().map(|row| row.0).collect::<Vec<_>>(),
                    )),
                    values,
                    Arc::new(Float64Array::from(
                        rows.iter()
                            .map(|row| {
                                row.1.as_deref().map(|value| match value {
                                    "é" => 1e16,
                                    "e\u{301}" => -1e16,
                                    "" => -0.0,
                                    _ => 1.0,
                                })
                            })
                            .collect::<Vec<_>>(),
                    )),
                    Arc::new(BooleanArray::from(
                        rows.iter()
                            .map(|row| row.0.is_some_and(|key| key % 2 == 1))
                            .collect::<Vec<_>>(),
                    )),
                    Arc::new(StringArray::from(vec!["unneeded payload"; rows.len()])),
                ],
            )
            .unwrap()
        })
        .collect();
    Batch::table(
        records,
        BatchMetadata::new("texts", sequence, JsonMap::new()).unwrap(),
    )
    .unwrap()
}

fn operator(dtype: &DataType, query: &str) -> SqlOperator {
    SqlOperator::new("string_extrema", query, vec!["events".into()], vec![])
        .unwrap()
        .with_ports(
            vec![
                Port::with_schema_ref("events", BatchKind::Table, true, Some(schema(dtype)))
                    .unwrap(),
            ],
            Port::new("output", BatchKind::Table, true, None).unwrap(),
        )
        .unwrap()
}

fn rows(batch: &Batch) -> Vec<Vec<ScalarValue>> {
    let mut rows = batch
        .table_payload()
        .unwrap()
        .batches()
        .iter()
        .flat_map(|record| {
            (0..record.num_rows()).map(|row| {
                record
                    .columns()
                    .iter()
                    .map(|array| ScalarValue::try_from_array(array, row).unwrap())
                    .collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rows
}

#[tokio::test]
async fn test_string_extrema_own_native_state_and_preserve_exact_prefix_restore() {
    for dtype in [DataType::Utf8, DataType::LargeUtf8] {
        for grouped in [false, true] {
            let query = if grouped {
                "SELECT MAX(value) AS hi, key, MIN(value) AS lo, COUNT(value) AS valid, COUNT(*) AS rows, MIN(value) AS again FROM events GROUP BY key"
            } else {
                "SELECT MAX(value) AS hi, MIN(value) AS lo, COUNT(value) AS valid, COUNT(*) AS rows, MIN(value) AS again FROM events"
            };
            recovery_case(&dtype, query).await;
        }
    }
}

async fn recovery_case(dtype: &DataType, query: &str) {
    let job = StreamJobContext::new(941, "texts", JsonMap::new(), None, CancellationToken::new());
    let context = StreamOperatorContext::new(&job, "string_extrema", None);
    let mut state = operator(dtype, query);
    let mut prefix = Vec::new();
    let mut pools = Vec::new();
    for (sequence, parts) in arrivals().into_iter().enumerate() {
        let incoming = input(dtype, &parts, sequence as u64);
        let weak = incoming
            .table_payload()
            .unwrap()
            .batches()
            .iter()
            .flat_map(|record| record.columns().iter().map(Arc::downgrade))
            .collect::<Vec<_>>();
        prefix.extend(parts);
        let actual = process(&mut state, incoming, &context).await;
        assert_oracle(&actual, query, dtype, &prefix, sequence as u64).await;
        assert!(
            state.incremental.is_some() && state.compact.is_some() && state.retained.is_none(),
            "string MIN/MAX must release cumulative input and own native state"
        );
        assert!(weak.iter().all(|array| array.upgrade().is_none()));
        let saved = state.checkpoint(Epoch::INITIAL).unwrap();
        pools.push(
            state
                .stream_state
                .runtime()
                .unwrap()
                .incremental_memory_pool(),
        );
        drop(state);
        state = operator(dtype, query);
        StreamOperator::restore(&mut state, &saved).unwrap();
        let restored = state.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(saved.inline_metadata, restored.inline_metadata);
        for (id, segment) in &saved.segments {
            assert_eq!(segment.bytes(), restored.segments[id].bytes());
        }
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
    assert_eq!(pools.iter().map(|pool| pool.reserved()).sum::<usize>(), 0);
}

fn arrivals() -> Vec<Vec<Vec<Row>>> {
    vec![
        vec![
            vec![(None, None), (Some(1), Some("é".into())), (Some(2), None)],
            vec![(Some(1), Some("e\u{301}".into()))],
        ],
        vec![
            vec![],
            vec![
                (None, Some(String::new())),
                (Some(1), Some("界😀\0".repeat(8192))),
                (Some(3), None),
            ],
        ],
        vec![vec![
            (Some(2), Some("a\0z".into())),
            (Some(1), Some("z".into())),
            (Some(4), Some("😀".into())),
        ]],
        vec![vec![]],
    ]
}

async fn process(
    state: &mut SqlOperator,
    input: Batch,
    context: &StreamOperatorContext<'_>,
) -> Batch {
    let mut collector = EdgeCollector::new(state.output_ports().to_vec());
    state
        .process_data("events", input, context, &mut collector)
        .await
        .unwrap();
    let output = collector.drain("output");
    assert_eq!(output.len(), 1);
    output[0].as_data().unwrap().clone()
}

async fn assert_oracle(
    actual: &Batch,
    query: &str,
    dtype: &DataType,
    prefix: &[Vec<Row>],
    sequence: u64,
) {
    let expected = DataFusionRuntime::new(DataFusionConfig::default())
        .unwrap()
        .sql(
            query,
            &BTreeMap::from([("events".into(), input(dtype, prefix, sequence))]),
            Some("oracle"),
        )
        .await
        .unwrap();
    assert_eq!(
        actual.table_payload().unwrap().schema(),
        expected.table_payload().unwrap().schema()
    );
    assert_eq!(actual.metadata(), expected.metadata());
    let actual = rows(actual);
    let expected = rows(&expected);
    assert_eq!(actual, expected);
    for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
        if let (ScalarValue::Float64(actual), ScalarValue::Float64(expected)) = (actual, expected) {
            assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
        }
    }
}

#[tokio::test]
async fn test_string_extrema_where_filters_composite_keys_and_float_mixtures() {
    for dtype in [DataType::Utf8, DataType::LargeUtf8] {
        for query in [
            "SELECT key, MIN(value) FILTER (WHERE selected) AS lo, MAX(value) AS hi, SUM(amount) AS total, AVG(amount) AS mean, COUNT(*) AS rows FROM events GROUP BY key",
            "SELECT key, MIN(value) AS lo, MAX(value) AS hi, COUNT(*) AS rows FROM events WHERE selected GROUP BY key",
            "SELECT key, unused, MIN(value) AS lo, MAX(value) AS hi, COUNT(value) AS valid FROM events GROUP BY key, unused",
        ] {
            recovery_case(&dtype, query).await;
        }
    }
}

struct Reject;

#[async_trait::async_trait]
impl StreamCollector for Reject {
    async fn emit(&mut self, _: &str, _: Batch) -> Result<()> {
        Err(CalcFlowError::Operator {
            node_id: "reject-string-extrema".into(),
            message: "injected output failure".into(),
        })
    }
}

fn same_snapshot(before: &OperatorStateSnapshot, after: &OperatorStateSnapshot) {
    assert_eq!(before.inline_metadata, after.inline_metadata);
    assert_eq!(before.segments.len(), after.segments.len());
    for (id, segment) in &before.segments {
        assert_eq!(segment.bytes(), after.segments[id].bytes());
    }
}

#[tokio::test]
async fn test_string_extrema_pressure_and_refused_output_refund_then_retry() {
    use datafusion::execution::memory_pool::MemoryConsumer;
    for dtype in [DataType::Utf8, DataType::LargeUtf8] {
        for grouped in [false, true] {
            let query = if grouped {
                "SELECT key, MIN(value), MAX(value), COUNT(*) FROM events GROUP BY key"
            } else {
                "SELECT MIN(value), MAX(value), COUNT(*) FROM events"
            };
            let job =
                StreamJobContext::new(942, "texts", JsonMap::new(), None, CancellationToken::new());
            let context = StreamOperatorContext::new(&job, "string_extrema", None);
            let mut state = operator(&dtype, query);
            let mut prefix = arrivals().remove(0);
            drop(process(&mut state, input(&dtype, &prefix, 0), &context).await);
            let before = state.checkpoint(Epoch::INITIAL).unwrap();
            let pool = state
                .stream_state
                .runtime()
                .unwrap()
                .incremental_memory_pool();
            let basis = pool.reserved();
            let next = vec![vec![(Some(1), Some("高".repeat(65_536)))]];
            let pressure = MemoryConsumer::new("string-extrema-pressure").register(&pool);
            pressure.try_grow((1 << 30) - basis - 1).unwrap();
            let held = pool.reserved();
            let mut output = EdgeCollector::new(state.output_ports().to_vec());
            assert!(
                state
                    .process_data("events", input(&dtype, &next, 1), &context, &mut output)
                    .await
                    .is_err()
            );
            assert!(output.drain("output").is_empty());
            assert_eq!(pool.reserved(), held);
            drop(pressure);
            assert_eq!(pool.reserved(), basis);
            same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
            assert!(
                state
                    .process_data("events", input(&dtype, &next, 1), &context, &mut Reject)
                    .await
                    .is_err()
            );
            assert_eq!(pool.reserved(), basis);
            same_snapshot(&before, &state.checkpoint(Epoch::INITIAL).unwrap());
            let actual = process(&mut state, input(&dtype, &next, 1), &context).await;
            prefix.extend(next);
            assert_oracle(&actual, query, &dtype, &prefix, 1).await;
            drop((state, actual, before));
            assert!(job.gather_owner().close_and_drain().await.is_empty());
            assert_eq!(pool.reserved(), 0);
        }
    }
}

#[tokio::test]
async fn test_string_extrema_sparse_deltas_compact_and_cold_continue() {
    for dtype in [DataType::Utf8, DataType::LargeUtf8] {
        let query = "SELECT key, MIN(value), MAX(value), COUNT(*) FROM events GROUP BY key";
        let job =
            StreamJobContext::new(943, "texts", JsonMap::new(), None, CancellationToken::new());
        let context = StreamOperatorContext::new(&job, "string_extrema", None);
        let mut state = operator(&dtype, query);
        let mut history = vec![
            (0..32)
                .map(|key| (Some(key), Some(format!("000-{key}-长文本"))))
                .collect(),
        ];
        drop(process(&mut state, input(&dtype, &history, 0), &context).await);
        let mut previous = state.checkpoint(Epoch::INITIAL).unwrap();
        let mut pools = Vec::new();
        for sequence in 1..=34 {
            let next = vec![vec![(
                Some(7),
                Some(format!("{sequence:04}-{}", "😀\0".repeat(128))),
            )]];
            let actual = process(&mut state, input(&dtype, &next, sequence), &context).await;
            history.extend(next);
            assert_oracle(&actual, query, &dtype, &history, sequence).await;
            let saved = state.checkpoint(Epoch::INITIAL).unwrap();
            let frames = saved
                .segments
                .keys()
                .filter(|id| id.starts_with("group-delta-"))
                .count();
            assert_eq!(frames, usize::try_from(sequence % 33).unwrap());
            if sequence <= 32 {
                assert_eq!(
                    previous.segments["group-state"].sha256(),
                    saved.segments["group-state"].sha256()
                );
            }
            pools.push(
                state
                    .stream_state
                    .runtime()
                    .unwrap()
                    .incremental_memory_pool(),
            );
            drop(state);
            state = operator(&dtype, query);
            StreamOperator::restore(&mut state, &saved).unwrap();
            same_snapshot(&saved, &state.checkpoint(Epoch::INITIAL).unwrap());
            previous = saved;
        }
        pools.push(
            state
                .stream_state
                .runtime()
                .unwrap()
                .incremental_memory_pool(),
        );
        drop((state, previous));
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        assert_eq!(pools.iter().map(|pool| pool.reserved()).sum::<usize>(), 0);
    }
}
