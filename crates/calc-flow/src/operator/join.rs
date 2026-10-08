#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc, time::Duration};

    use datafusion::arrow::{
        array::{
            BinaryArray, DictionaryArray, FixedSizeBinaryArray, Int32Array, Int64Array,
            StringArray, TimestampMicrosecondArray,
        },
        datatypes::{DataType, Field, Int32Type, Schema, TimeUnit},
        record_batch::RecordBatch,
    };

    use super::*;
    use crate::{
        BatchMetadata, CancellationToken, EdgeBudget, EdgeCollector, IngressProgress,
        IngressProgressSnapshot, IngressState, JsonMap, OperatorMetadata, StreamJobContext,
        StreamMessageKind, StreamOperator,
    };

    fn left_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("amount", DataType::Int64, true),
        ]))
    }

    fn right_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "paid_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("status", DataType::Utf8, true),
        ]))
    }

    fn spec() -> StreamJoinSpec {
        StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
            JoinStateLimits::new(100, 1_000_000, 1_000).unwrap(),
        )
        .unwrap()
        .with_prefixes("authorization", "payment")
        .unwrap()
    }

    fn left_batch(times: Vec<i64>) -> Batch {
        let rows = times.len();
        Batch::table(
            vec![
                RecordBatch::try_new(
                    left_schema(),
                    vec![
                        Arc::new(Int64Array::from(vec![7; rows])),
                        Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
                        Arc::new(Int64Array::from(vec![42; rows])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    fn right_batch(times: Vec<i64>) -> Batch {
        let rows = times.len();
        Batch::table(
            vec![
                RecordBatch::try_new(
                    right_schema(),
                    vec![
                        Arc::new(Int64Array::from(vec![7; rows])),
                        Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
                        Arc::new(StringArray::from(vec!["paid"; rows])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    #[test]
    fn derives_exact_prefixed_ports() {
        let operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();

        assert_eq!(
            operator
                .input_ports()
                .iter()
                .map(Port::name)
                .collect::<Vec<_>>(),
            ["left", "right"]
        );
        let output = operator.output_ports()[0].schema().unwrap();
        assert_eq!(
            output
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect::<Vec<_>>(),
            [
                "authorization__account_id",
                "authorization__authorized_at",
                "authorization__amount",
                "payment__account_id",
                "payment__paid_at",
                "payment__status",
            ]
        );
        assert!(output.field(2).is_nullable());
        assert!(output.field(5).is_nullable());
    }

    #[test]
    fn rejects_values_beyond_the_json_safe_integer_domain() {
        let too_large = STREAM_JOIN_MAX_SAFE_JSON_INTEGER + 1;
        assert!(JoinStateLimits::new(too_large, 1, 1).is_err());
        assert!(JoinTimeBounds::new(Duration::from_micros(too_large), Duration::ZERO).is_err());
        assert!(JoinStateLimits::new(1, 1, 1).is_ok());
    }

    #[test]
    fn inclusive_bound_helper_accepts_both_edges() {
        let bounds =
            JoinTimeBounds::new(Duration::from_micros(10), Duration::from_micros(20)).unwrap();
        assert!(bounds.contains_pair(100, 90));
        assert!(bounds.contains_pair(100, 120));
        assert!(!bounds.contains_pair(100, 89));
        assert!(!bounds.contains_pair(100, 121));
    }

    #[test]
    fn output_frontier_uses_live_idle_and_ended_formulas() {
        let operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let snapshot = |left_state: IngressState,
                        left: Option<i64>,
                        right_state: IngressState,
                        right: Option<i64>| {
            IngressProgressSnapshot::new(BTreeMap::from([
                (
                    "left".into(),
                    IngressProgress::new(left_state, left.map(EventTime::from_micros)),
                ),
                (
                    "right".into(),
                    IngressProgress::new(right_state, right.map(EventTime::from_micros)),
                ),
            ]))
        };
        let micros = |value: Option<EventTime>| value.map(EventTime::as_micros);

        assert_eq!(
            micros(
                operator
                    .output_frontier_candidate(&snapshot(
                        IngressState::Idle,
                        Some(400_000_000),
                        IngressState::Active,
                        Some(100_000_000),
                    ))
                    .unwrap()
            ),
            Some(40_000_000)
        );
        assert_eq!(
            micros(
                operator
                    .output_frontier_candidate(&snapshot(
                        IngressState::Active,
                        Some(400_000_000),
                        IngressState::Ended,
                        Some(100_000_000),
                    ))
                    .unwrap()
            ),
            Some(100_000_000)
        );
        assert_eq!(
            micros(
                operator
                    .output_frontier_candidate(&snapshot(
                        IngressState::Ended,
                        Some(400_000_000),
                        IngressState::Active,
                        Some(100_000_000),
                    ))
                    .unwrap()
            ),
            Some(40_000_000)
        );
        assert_eq!(
            operator
                .output_frontier_candidate(&snapshot(
                    IngressState::Ended,
                    Some(400_000_000),
                    IngressState::Ended,
                    Some(100_000_000),
                ))
                .unwrap(),
            None
        );
        assert_eq!(
            operator
                .output_frontier_candidate(&snapshot(
                    IngressState::Active,
                    None,
                    IngressState::Active,
                    Some(100_000_000),
                ))
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn emits_duplicate_pairs_at_both_inclusive_boundaries() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job = StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data("left", left_batch(vec![100, 100]), &context, &mut collector)
            .await
            .unwrap();
        operator
            .process_data(
                "right",
                right_batch(vec![-299_999_900, 60_000_100, 60_000_101]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();

        let outputs = collector.drain("output");
        assert_eq!(outputs.len(), 1);
        assert!(
            outputs
                .iter()
                .all(|message| message.kind() == StreamMessageKind::Data)
        );
        let data = outputs[0].as_data().unwrap();
        assert_eq!(data.metadata().sequence(), 0);
        assert_eq!(data.num_rows(), 4);
        assert_eq!(
            paid_times(data),
            [-299_999_900, -299_999_900, 60_000_100, 60_000_100]
        );
    }

    fn paid_times(data: &Batch) -> Vec<i64> {
        data.table_payload()
            .unwrap()
            .batches()
            .iter()
            .flat_map(|record| {
                let times = record
                    .column_by_name("payment__paid_at")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap();
                (0..record.num_rows()).map(move |index| times.value(index))
            })
            .collect()
    }

    fn keyed_left_batch(keys: &[i64], times: &[i64]) -> Batch {
        Batch::table(
            vec![
                RecordBatch::try_new(
                    left_schema(),
                    vec![
                        Arc::new(Int64Array::from(keys.to_vec())),
                        Arc::new(
                            TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC"),
                        ),
                        Arc::new(Int64Array::from(vec![42; keys.len()])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    fn keyed_right_batch(keys: &[i64], times: &[i64]) -> Batch {
        Batch::table(
            vec![
                RecordBatch::try_new(
                    right_schema(),
                    vec![
                        Arc::new(Int64Array::from(keys.to_vec())),
                        Arc::new(
                            TimestampMicrosecondArray::from(times.to_vec()).with_timezone("UTC"),
                        ),
                        Arc::new(StringArray::from(vec!["paid"; keys.len()])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    #[tokio::test]
    async fn cross_batch_matches_preserve_dictionary_encoded_payloads() {
        // Retained rows from different input batches can carry the same
        // dictionary type over different dictionaries; one probe batch that
        // matches both must concatenate their dictionary columns without
        // losing or remapping values (Copilot review of PR #298).
        let dict_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new(
                "amount",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
        ]));
        let dictionary_row = |values: Vec<&str>, key: i32| {
            Batch::table(
                vec![
                    RecordBatch::try_new(
                        Arc::clone(&dict_schema),
                        vec![
                            Arc::new(Int64Array::from(vec![7])),
                            Arc::new(
                                TimestampMicrosecondArray::from(vec![100]).with_timezone("UTC"),
                            ),
                            Arc::new(DictionaryArray::<Int32Type>::new(
                                Int32Array::from(vec![key]),
                                Arc::new(StringArray::from(values)),
                            )),
                        ],
                    )
                    .unwrap(),
                ],
                BatchMetadata::default(),
            )
            .unwrap()
        };
        let mut operator =
            StreamJoinOperator::new("match", Arc::clone(&dict_schema), right_schema(), spec())
                .unwrap();
        let job = StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        // The same logical value "paid" under two distinct dictionaries.
        operator
            .process_data(
                "left",
                dictionary_row(vec!["paid"], 0),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        operator
            .process_data(
                "left",
                dictionary_row(vec!["other", "paid"], 1),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        operator
            .process_data("right", right_batch(vec![100]), &context, &mut collector)
            .await
            .unwrap();

        let outputs = collector.drain("output");
        assert_eq!(outputs.len(), 1);
        let data = outputs[0].as_data().unwrap();
        assert_eq!(data.num_rows(), 2);
        assert_eq!(
            dictionary_strings(data, "authorization__amount"),
            ["paid", "paid"]
        );
    }

    /// Decodes one dictionary-encoded output column back to its strings.
    fn dictionary_strings(data: &Batch, name: &str) -> Vec<String> {
        data.table_payload()
            .unwrap()
            .batches()
            .iter()
            .flat_map(|record| {
                let column = record
                    .column_by_name(name)
                    .expect("prefixed dictionary payload column");
                assert_eq!(
                    column.data_type(),
                    &DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
                );
                let typed = column
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()
                    .expect("declared dictionary column");
                let strings = typed
                    .values()
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("utf8 dictionary values");
                (0..record.num_rows()).map(move |index| {
                    strings
                        .value(usize::try_from(typed.keys().value(index)).unwrap())
                        .to_owned()
                })
            })
            .collect()
    }

    #[tokio::test]
    async fn matched_rows_accumulate_into_one_batched_message() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job = StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data(
                "left",
                keyed_left_batch(&[1, 2, 3], &[100, 100, 100]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        operator
            .process_data(
                "right",
                keyed_right_batch(&[3, 1, 2], &[100, 100, 100]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();

        let outputs = collector.drain("output");
        assert_eq!(
            outputs.len(),
            1,
            "one input batch's matched rows must leave as one message"
        );
        let data = outputs[0].as_data().unwrap();
        assert_eq!(data.metadata().sequence(), 0);
        assert_eq!(data.num_rows(), 3);
        // Rows keep the matched-pair order (probe position, then event time
        // and row id), so the right batch's key order drives the output.
        let keys = |name: &str| {
            data.table_payload()
                .unwrap()
                .batches()
                .iter()
                .flat_map(|record| {
                    let column = record
                        .column_by_name(name)
                        .unwrap()
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    (0..record.num_rows()).map(move |index| column.value(index))
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(keys("payment__account_id"), [3, 1, 2]);
        assert_eq!(keys("authorization__account_id"), [3, 1, 2]);
        assert_eq!(operator.status().emitted_match_rows, 3);
    }

    #[tokio::test]
    async fn batched_output_splits_into_edge_budget_chunks_in_order() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job = StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "match", None)
            .with_output_budget(EdgeBudget::new(2, 8 << 20).unwrap());
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data(
                "left",
                keyed_left_batch(&[1, 2, 3, 4], &[100, 100, 100, 100]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        operator
            .process_data(
                "right",
                keyed_right_batch(&[1, 2, 3, 4], &[100, 100, 100, 100]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();

        let outputs = collector.drain("output");
        assert_eq!(
            outputs.len(),
            2,
            "max_rows=2 splits four rows into two chunks"
        );
        assert_eq!(
            outputs
                .iter()
                .map(|message| message.as_data().unwrap().metadata().sequence())
                .collect::<Vec<_>>(),
            [0, 1]
        );
        let rows = outputs
            .iter()
            .flat_map(|message| {
                let data = message.as_data().unwrap();
                data.table_payload()
                    .unwrap()
                    .batches()
                    .iter()
                    .flat_map(|record| {
                        let column = record
                            .column_by_name("authorization__account_id")
                            .unwrap()
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap();
                        (0..record.num_rows()).map(move |index| column.value(index))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(rows, [1, 2, 3, 4], "chunking preserves matched order");
    }

    #[tokio::test]
    async fn output_chunks_own_only_their_materialized_payload() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None)
            .with_output_budget(EdgeBudget::new(2, 8 << 20).unwrap());
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", left_batch(vec![100; 4]), &context, &mut collector)
            .await
            .unwrap();
        let right = Batch::table(
            vec![
                RecordBatch::try_new(
                    right_schema(),
                    vec![
                        Arc::new(Int64Array::from(vec![7])),
                        Arc::new(TimestampMicrosecondArray::from(vec![100]).with_timezone("UTC")),
                        Arc::new(StringArray::from(vec!["r".repeat(1024)])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap();
        operator
            .process_data("right", right, &context, &mut collector)
            .await
            .unwrap();
        let outputs = collector.drain("output");
        assert_eq!(outputs.len(), 2);
        for (sequence, message) in outputs.iter().enumerate() {
            let batch = message.as_data().unwrap();
            assert_eq!(
                batch.metadata().sequence(),
                u64::try_from(sequence).unwrap()
            );
            let record = &batch.table_payload().unwrap().batches()[0];
            let payload = record
                .column(5)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            assert_eq!(payload.len(), 2);
            assert_eq!(payload.value(0), "r".repeat(1024));
            assert_eq!(
                payload.value_data().len(),
                2048,
                "a chunk must not retain the full fan-out output buffer"
            );
        }
    }

    #[tokio::test]
    async fn chunk_preflight_ignores_unreferenced_dictionary_payloads() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            left_schema().field(1).clone(),
            Field::new(
                "amount",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                false,
            ),
        ]));
        let mut operator =
            StreamJoinOperator::new("match", Arc::clone(&schema), right_schema(), spec()).unwrap();
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None)
            .with_output_budget(EdgeBudget::new(1, 64).unwrap());
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        for unused in ["a".repeat(512), "b".repeat(512)] {
            let record = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(vec![7])),
                    Arc::new(TimestampMicrosecondArray::from(vec![100]).with_timezone("UTC")),
                    Arc::new(DictionaryArray::<Int32Type>::new(
                        Int32Array::from(vec![1]),
                        Arc::new(StringArray::from(vec![unused.as_str(), "paid"])),
                    )),
                ],
            )
            .unwrap();
            operator
                .process_data(
                    "left",
                    Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut output,
                )
                .await
                .unwrap();
        }
        operator
            .process_data("right", right_batch(vec![100]), &context, &mut output)
            .await
            .unwrap();
        let chunks = output.drain("output");
        assert_eq!(chunks.len(), 2);
        for chunk in chunks {
            let batch = chunk.as_data().unwrap();
            assert!(batch.estimated_bytes().unwrap() <= 64);
            assert_eq!(dictionary_strings(batch, "authorization__amount"), ["paid"]);
        }
    }

    #[tokio::test]
    async fn oversized_row_fails_loudly_before_any_emission() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job = StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "match", None)
            .with_output_budget(EdgeBudget::new(10_000, 64).unwrap());
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data("left", left_batch(vec![100, 100]), &context, &mut collector)
            .await
            .unwrap();
        let wide = Batch::table(
            vec![
                RecordBatch::try_new(
                    right_schema(),
                    vec![
                        Arc::new(Int64Array::from(vec![7, 7])),
                        Arc::new(
                            TimestampMicrosecondArray::from(vec![100, 100]).with_timezone("UTC"),
                        ),
                        Arc::new(StringArray::from(vec!["paid".to_owned(), "p".repeat(512)])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap();
        let failure = operator
            .process_data("right", wide, &context, &mut collector)
            .await
            .unwrap_err();
        assert!(
            failure
                .to_string()
                .contains("one stream Join output row exceeds the effective edge byte budget"),
            "{failure}"
        );
        assert!(
            collector.drain("output").is_empty(),
            "an over-budget row must fail before any message leaves the operator"
        );
        assert_eq!(operator.status().emitted_match_rows, 0);
        assert_eq!(operator.status().right.retained_rows, 0);
    }

    struct BlockingChunkCollector {
        accepted: Vec<Batch>,
        cancel: CancellationToken,
    }

    #[async_trait]
    impl StreamCollector for BlockingChunkCollector {
        async fn emit(&mut self, _port: &str, batch: Batch) -> Result<()> {
            if self.accepted.is_empty() {
                self.accepted.push(batch);
                Ok(())
            } else {
                self.cancel.cancel();
                std::future::pending().await
            }
        }
    }

    #[tokio::test]
    async fn blocked_chunk_cancellation_restores_the_last_committed_checkpoint() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let cancel = CancellationToken::new();
        let run = StreamJobContext::new(1, "fingerprint", JsonMap::new(), None, cancel.clone());
        let budget = EdgeBudget::new(2, 8 << 20).unwrap();
        let context = StreamOperatorContext::new(&run, "match", None).with_output_budget(budget);
        let mut preload = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", left_batch(vec![100; 4]), &context, &mut preload)
            .await
            .unwrap();
        let checkpoint = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
        let mut blocked = BlockingChunkCollector {
            accepted: Vec::new(),
            cancel: cancel.clone(),
        };
        tokio::time::timeout(Duration::from_secs(1), async {
            tokio::select! {
                () = cancel.cancelled() => {},
                result = operator.process_data("right", right_batch(vec![100]), &context, &mut blocked) => {
                    panic!("second chunk must block until cancelled: {result:?}");
                }
            }
        }).await.unwrap();
        assert_eq!(blocked.accepted.len(), 1);
        assert_eq!(blocked.accepted[0].metadata().sequence(), 0);
        assert_eq!(operator.status().right.retained_rows, 0);
        assert_eq!(operator.status().emitted_match_rows, 0);
        assert_eq!(operator.state.next_output_sequence, 1);
        let mut restored =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        restored.restore(&checkpoint).unwrap();
        let resumed_job = job();
        let resumed =
            StreamOperatorContext::new(&resumed_job, "match", None).with_output_budget(budget);
        let mut output = EdgeCollector::new(restored.output_ports().to_vec());
        restored
            .process_data("right", right_batch(vec![100]), &resumed, &mut output)
            .await
            .unwrap();
        let chunks = output.drain("output");
        assert_eq!(
            chunks
                .iter()
                .map(|message| message.as_data().unwrap().num_rows())
                .sum::<usize>(),
            4
        );
        assert_eq!(
            chunks
                .iter()
                .map(|message| message.as_data().unwrap().metadata().sequence())
                .collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(restored.status().emitted_match_rows, 4);
        assert_eq!(restored.status().right.retained_rows, 1);
    }

    #[tokio::test]
    async fn wide_rows_split_along_the_byte_budget_axis() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job = job();
        // max_rows alone would keep all four wide rows in one message; the
        // byte bound is what must drive the split.
        let budget = EdgeBudget::new(1_000, 800).unwrap();
        let context = StreamOperatorContext::new(&job, "match", None).with_output_budget(budget);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data("left", left_batch(vec![100]), &context, &mut collector)
            .await
            .unwrap();
        let wide = Batch::table(
            vec![
                RecordBatch::try_new(
                    right_schema(),
                    vec![
                        Arc::new(Int64Array::from(vec![7; 4])),
                        Arc::new(
                            TimestampMicrosecondArray::from(vec![100, 100, 100, 100])
                                .with_timezone("UTC"),
                        ),
                        Arc::new(StringArray::from(vec!["s".repeat(256); 4])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap();
        operator
            .process_data("right", wide, &context, &mut collector)
            .await
            .unwrap();

        let outputs = collector.drain("output");
        assert!(
            outputs.len() > 1,
            "the byte bound must split rows that max_rows would keep together"
        );
        for message in &outputs {
            let data = message.as_data().unwrap();
            assert!(
                data.estimated_bytes().unwrap() <= budget.max_bytes,
                "every emitted message must respect the byte budget"
            );
        }
        let sequences = outputs
            .iter()
            .map(|message| message.as_data().unwrap().metadata().sequence())
            .collect::<Vec<_>>();
        assert_eq!(
            sequences,
            (0..u64::try_from(outputs.len()).unwrap()).collect::<Vec<_>>(),
            "sequences stay dense and increasing across byte-driven chunks"
        );
        let rows = outputs
            .iter()
            .flat_map(|message| {
                let data = message.as_data().unwrap();
                data.table_payload()
                    .unwrap()
                    .batches()
                    .iter()
                    .flat_map(|record| {
                        let column = record
                            .column_by_name("payment__status")
                            .unwrap()
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .unwrap();
                        (0..record.num_rows()).map(move |index| column.value(index).len())
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(rows, [256; 4], "all four wide rows arrive in order");
    }

    #[tokio::test]
    async fn sparse_null_payloads_fit_the_chunk_validity_bitmap_budget() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job = job();
        let budget = EdgeBudget::new(100, 433).unwrap();
        let context = StreamOperatorContext::new(&job, "match", None).with_output_budget(budget);
        let keys = (0..9).collect::<Vec<_>>();
        let record = RecordBatch::try_new(
            left_schema(),
            vec![
                Arc::new(Int64Array::from(keys.clone())),
                Arc::new(TimestampMicrosecondArray::from(vec![100; 9]).with_timezone("UTC")),
                Arc::new(Int64Array::from(
                    (0..9)
                        .map(|index| (index != 5).then_some(42))
                        .collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "left",
                Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                &context,
                &mut output,
            )
            .await
            .unwrap();
        operator
            .process_data(
                "right",
                keyed_right_batch(&keys, &[100; 9]),
                &context,
                &mut output,
            )
            .await
            .unwrap();
        let chunks = output.drain("output");
        assert_eq!(chunks.len(), 2);
        let mut values = Vec::new();
        for chunk in chunks {
            let batch = chunk.as_data().unwrap();
            assert!(batch.estimated_bytes().unwrap() <= budget.max_bytes);
            for record in batch.table_payload().unwrap().batches() {
                let column = record
                    .column(2)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                values.extend(column.iter());
            }
        }
        assert_eq!(
            values,
            (0..9)
                .map(|index| (index != 5).then_some(42))
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn exhausted_output_sequence_fails_before_building_a_message() {
        let mut seed =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut collector = EdgeCollector::new(seed.output_ports().to_vec());
        seed.process_data("left", left_batch(vec![100]), &context, &mut collector)
            .await
            .unwrap();
        let mut exhausted = seed.checkpoint(Epoch::new(1).unwrap()).unwrap();
        exhausted
            .inline_metadata
            .insert("next_output_sequence".into(), u64::MAX.into());

        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        operator.restore(&exhausted).unwrap();
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        let failure = operator
            .process_data("right", right_batch(vec![100]), &context, &mut collector)
            .await
            .unwrap_err();
        assert!(
            failure
                .to_string()
                .contains("output sequence overflowed before emission"),
            "{failure}"
        );
        assert!(collector.drain("output").is_empty());
        assert_eq!(operator.status().emitted_match_rows, 0);
    }

    fn job() -> StreamJobContext {
        StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
    }

    fn progress_context(
        job_context: &StreamJobContext,
        left: (IngressState, Option<i64>),
        right: (IngressState, Option<i64>),
    ) -> StreamOperatorContext<'_> {
        let snapshot = IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                IngressProgress::new(left.0, left.1.map(EventTime::from_micros)),
            ),
            (
                "right".into(),
                IngressProgress::new(right.0, right.1.map(EventTime::from_micros)),
            ),
        ]));
        StreamOperatorContext::for_task(
            job_context,
            "match",
            None,
            snapshot,
            EdgeBudget::default(),
            Arc::new(NoopLateMetrics),
        )
    }

    struct NoopLateMetrics;

    impl crate::operator::LateMetricSink for NoopLateMetrics {
        fn record(&self, _delta: crate::operator::LateMetricDelta) -> Result<()> {
            Ok(())
        }
    }

    fn reason_of(error: &CalcFlowError) -> Option<crate::StreamingFailureReason> {
        match error {
            CalcFlowError::OperatorReason { reason_code, .. } => Some(*reason_code),
            _ => None,
        }
    }

    fn checkpoint_metadata(operator: &mut StreamJoinOperator, epoch: u64) -> JsonMap {
        operator
            .checkpoint(Epoch::new(epoch).unwrap())
            .unwrap()
            .inline_metadata
    }

    #[tokio::test]
    async fn status_tracks_ingress_watermark_idle_reactivation_and_end() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let initial = operator.status();
        assert_eq!(initial.left.watermark_micros, None);
        assert!(!initial.left.idle && !initial.left.ended);
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("right", right_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        for (state, watermark) in [
            (IngressState::Active, i64::MIN),
            (IngressState::Idle, i64::MIN),
            (IngressState::Active, i64::MAX),
            (IngressState::Ended, i64::MAX),
        ] {
            let context = progress_context(
                &job_context,
                (IngressState::Active, None),
                (state, Some(watermark)),
            );
            operator
                .on_ingress_progress("right", &context)
                .await
                .unwrap();
            let status = operator.status();
            assert_eq!(
                status.right.watermark_micros,
                Some(EventTime::from_micros(watermark))
            );
            assert_eq!(status.right.idle, state == IngressState::Idle);
            assert_eq!(status.right.ended, state == IngressState::Ended);
            assert_eq!(status.left.watermark_micros, None);
        }
        let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
        assert_eq!(snapshot.inline_metadata["layout_version"], 1);
        assert!(
            snapshot.inline_metadata["metrics"]["right"]
                .get("watermark_micros")
                .is_none()
        );
        operator.restore(&snapshot).unwrap();
        assert_eq!(operator.status().right.watermark_micros, None);
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator.on_end(&context, &mut collector).await.unwrap();
        assert!(operator.status().left.ended && operator.status().right.ended);
        operator.reset().unwrap();
        assert!(!operator.status().right.ended);
        assert_eq!(operator.status().right.watermark_micros, None);
    }

    #[tokio::test]
    async fn late_rows_are_dropped_with_metrics_and_never_retained() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = progress_context(
            &job_context,
            (IngressState::Active, Some(500_000_000)),
            (IngressState::Active, None),
        );
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data(
                "left",
                left_batch(vec![100_000_000, 499_999_999]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();

        assert!(collector.drain("output").is_empty());
        let metadata = checkpoint_metadata(&mut operator, 1);
        let state = serde_json::to_string(&metadata["metrics"]["left"]).unwrap();
        assert!(state.contains("\"late_rows\":2"), "{state}");
        assert!(
            state.contains("\"late_affected_batches\":1")
                && state.contains("\"max_lateness_micros\":400000000"),
            "{state}"
        );
        assert!(
            state.contains("\"retained_rows\":0"),
            "late rows must never be retained: {state}"
        );
    }

    #[tokio::test]
    async fn null_event_time_and_null_key_rows_are_counted_not_stored() {
        let nullable_time_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                true,
            ),
            Field::new("amount", DataType::Int64, true),
        ]));
        let nullable_key_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, true),
            Field::new(
                "paid_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("status", DataType::Utf8, true),
        ]));
        let mut operator = StreamJoinOperator::new(
            "match",
            Arc::clone(&nullable_time_schema),
            Arc::clone(&nullable_key_schema),
            spec(),
        )
        .unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        let null_time = Batch::table(
            vec![
                RecordBatch::try_new(
                    Arc::clone(&nullable_time_schema),
                    vec![
                        Arc::new(Int64Array::from(vec![7])),
                        Arc::new(
                            TimestampMicrosecondArray::from(vec![None::<i64>]).with_timezone("UTC"),
                        ),
                        Arc::new(Int64Array::from(vec![42])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap();
        let nullable_key_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, true),
            Field::new(
                "paid_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("status", DataType::Utf8, true),
        ]));
        let null_key = Batch::table(
            vec![
                RecordBatch::try_new(
                    Arc::clone(&nullable_key_schema),
                    vec![
                        Arc::new(Int64Array::from(vec![None::<i64>])),
                        Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
                        Arc::new(StringArray::from(vec!["paid"])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap();

        operator
            .process_data("left", null_time, &context, &mut collector)
            .await
            .unwrap();
        operator
            .process_data("right", null_key, &context, &mut collector)
            .await
            .unwrap();

        assert!(collector.drain("output").is_empty());
        let metadata = checkpoint_metadata(&mut operator, 1);
        assert_eq!(metadata["metrics"]["left"]["null_event_time_rows"], 1);
        assert_eq!(metadata["metrics"]["right"]["null_key_rows"], 1);
        assert_eq!(metadata["metrics"]["left"]["retained_rows"], 0);
        assert_eq!(metadata["metrics"]["right"]["retained_rows"], 0);
    }

    #[tokio::test]
    async fn watermark_progress_evicts_expired_opposite_rows_and_end_clears_them() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "left",
                left_batch(vec![0, 100_000_000]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        let metadata = checkpoint_metadata(&mut operator, 1);
        assert_eq!(metadata["metrics"]["left"]["retained_rows"], 2);
        let right_plan = operator.side_plan("right").unwrap();
        operator.opposite_state_keys(&right_plan).unwrap();
        assert!(operator.retained_key_cache.left.is_some());

        let eviction = progress_context(
            &job_context,
            (IngressState::Active, Some(1_000_000)),
            (IngressState::Active, Some(150_000_000)),
        );
        operator
            .on_ingress_progress("right", &eviction)
            .await
            .unwrap();
        assert!(operator.retained_key_cache.left.is_none());
        let metadata = checkpoint_metadata(&mut operator, 2);
        let left_metrics = serde_json::to_string(&metadata["metrics"]["left"]).unwrap();
        assert!(
            left_metrics.contains("\"retained_rows\":1")
                && left_metrics.contains("\"evicted_rows\":1"),
            "{left_metrics}"
        );

        let ended = progress_context(
            &job_context,
            (IngressState::Active, Some(1_000_000)),
            (IngressState::Ended, Some(50_000_000)),
        );
        operator.opposite_state_keys(&right_plan).unwrap();
        assert!(operator.retained_key_cache.left.is_some());
        operator.on_ingress_progress("right", &ended).await.unwrap();
        assert!(operator.retained_key_cache.left.is_none());
        let metadata = checkpoint_metadata(&mut operator, 3);
        assert_eq!(metadata["metrics"]["left"]["retained_rows"], 0);
    }

    #[tokio::test]
    async fn unknown_ingress_and_data_after_end_fail_loudly() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        let unknown = operator
            .process_data("middle", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap_err();
        assert!(unknown.to_string().contains("unknown ingress"), "{unknown}");

        operator.on_end(&context, &mut collector).await.unwrap();
        let after_end = operator
            .process_data("left", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap_err();
        assert!(
            after_end.to_string().contains("data after end-of-input"),
            "{after_end}"
        );

        let progress = IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1))),
            ),
            (
                "right".into(),
                IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1))),
            ),
            (
                "middle".into(),
                IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1))),
            ),
        ]));
        let unknown_progress = operator
            .on_ingress_progress(
                "middle",
                &StreamOperatorContext::for_task(
                    &job_context,
                    "match",
                    None,
                    progress,
                    EdgeBudget::default(),
                    Arc::new(NoopLateMetrics),
                ),
            )
            .await
            .unwrap_err();
        assert!(
            unknown_progress.to_string().contains("unknown ingress"),
            "{unknown_progress}"
        );
    }

    #[tokio::test]
    async fn state_row_limit_failure_is_atomic_with_typed_reason() {
        let limited = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
            JoinStateLimits::new(1, 1_000_000, 1_000).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), limited).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data("left", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        let failure = operator
            .process_data("left", left_batch(vec![1]), &context, &mut collector)
            .await
            .unwrap_err();
        assert_eq!(
            reason_of(&failure),
            Some(crate::StreamingFailureReason::JoinStateLimitExceeded)
        );

        let metadata = checkpoint_metadata(&mut operator, 1);
        assert_eq!(metadata["metrics"]["state_limit_failures"], 1);
        assert_eq!(metadata["metrics"]["left"]["retained_rows"], 1);
        assert!(collector.drain("output").is_empty());
    }

    #[tokio::test]
    async fn match_limit_failure_is_atomic_with_typed_reason() {
        let limited = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
            JoinStateLimits::new(100, 1_000_000, 1).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), limited).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data("left", left_batch(vec![0, 0]), &context, &mut collector)
            .await
            .unwrap();
        let failure = operator
            .process_data("right", right_batch(vec![0, 0]), &context, &mut collector)
            .await
            .unwrap_err();
        assert_eq!(
            reason_of(&failure),
            Some(crate::StreamingFailureReason::JoinMatchLimitExceeded)
        );

        let metadata = checkpoint_metadata(&mut operator, 1);
        assert_eq!(metadata["metrics"]["match_limit_failures"], 1);
        assert_eq!(metadata["metrics"]["right"]["retained_rows"], 0);
        assert!(collector.drain("output").is_empty());
    }

    #[tokio::test]
    async fn checkpoint_and_restore_round_trip_preserves_state_and_counters() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        operator
            .process_data("right", right_batch(vec![1]), &context, &mut collector)
            .await
            .unwrap();
        collector.drain("output");
        let snapshot = operator.checkpoint(Epoch::new(7).unwrap()).unwrap();

        let same_epoch = operator.checkpoint(Epoch::new(7).unwrap()).unwrap_err();
        assert!(
            same_epoch.to_string().contains("did not advance"),
            "{same_epoch}"
        );

        let mut restored =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        restored.restore(&snapshot).unwrap();
        let round_trip = restored.checkpoint(Epoch::new(8).unwrap()).unwrap();
        let mut expected_metadata = snapshot.inline_metadata.clone();
        expected_metadata.insert("epoch".into(), 8.into());
        assert_eq!(round_trip.inline_metadata, expected_metadata);
        assert_eq!(round_trip.segments, snapshot.segments);

        let mut collector = EdgeCollector::new(restored.output_ports().to_vec());
        restored
            .process_data("right", right_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        let outputs = collector.drain("output");
        assert_eq!(outputs.len(), 1, "restored left state must still match");
    }

    #[tokio::test]
    async fn no_expiry_progress_visits_neither_retained_rows_nor_pending_ops() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "left",
                left_batch((0..80).collect()),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        reset_join_work();
        let progress = progress_context(
            &job_context,
            (IngressState::Active, None),
            (IngressState::Active, Some(60_000_000)),
        );
        operator
            .on_ingress_progress("right", &progress)
            .await
            .unwrap();
        let work = join_work();
        assert_eq!(work.retained_visits, 0);
        assert_eq!(work.pending_visits, 0);
        assert_eq!(operator.status().left.retained_rows, 80);
    }

    #[tokio::test]
    async fn sparse_out_of_order_eviction_visits_only_expired_identities() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        let mut times = (2..80).rev().collect::<Vec<_>>();
        times.splice(20..20, [0, 1]);
        operator
            .process_data("left", left_batch(times), &context, &mut collector)
            .await
            .unwrap();
        reset_join_work();
        let progress = progress_context(
            &job_context,
            (IngressState::Active, None),
            (IngressState::Active, Some(60_000_002)),
        );
        operator
            .on_ingress_progress("right", &progress)
            .await
            .unwrap();
        let work = join_work();
        assert_eq!(work.retained_visits, 2);
        assert_eq!(work.pending_visits, 2);
        assert_eq!(operator.status().left.retained_rows, 78);
        assert_eq!(operator.status().left.evicted_rows, 2);
        let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
        let mut restored =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        restored.restore(&snapshot).unwrap();
        restored
            .on_ingress_progress("right", &progress)
            .await
            .unwrap();
        assert_eq!(restored.status(), operator.status());
    }

    #[tokio::test]
    async fn retained_row_key_is_encoded_once_and_reused_for_its_charge() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        reset_join_work();
        operator
            .process_data("left", left_batch(vec![0, 1, 2]), &context, &mut collector)
            .await
            .unwrap();
        assert_eq!(join_work().key_encodings, 3);
    }

    #[tokio::test]
    async fn dirty_log_coalescing_visits_only_the_evicted_upsert() {
        for count in [3, 80] {
            let mut operator =
                StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
            let job_context = job();
            let context = StreamOperatorContext::new(&job_context, "match", None);
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            operator
                .process_data(
                    "left",
                    left_batch((0..count).collect()),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
            reset_join_work();
            let progress = progress_context(
                &job_context,
                (IngressState::Active, None),
                (IngressState::Active, Some(60_000_001)),
            );
            operator
                .on_ingress_progress("right", &progress)
                .await
                .unwrap();
            assert_eq!(join_work().pending_visits, 1);
            let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
            let mut restored =
                StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
            restored.restore(&snapshot).unwrap();
            assert_eq!(
                restored.status().left.retained_rows,
                u64::try_from(count - 1).unwrap()
            );
        }
    }

    #[tokio::test]
    async fn terminal_tombstones_preserve_stable_retention_order_after_sparse_eviction() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "left",
                left_batch(vec![3, 0, 4, 1, 5, 2]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
        for (epoch, watermark) in [(2, 60_000_001), (3, 60_000_003)] {
            let progress = progress_context(
                &job_context,
                (IngressState::Active, None),
                (IngressState::Active, Some(watermark)),
            );
            operator
                .on_ingress_progress("right", &progress)
                .await
                .unwrap();
            operator.checkpoint(Epoch::new(epoch).unwrap()).unwrap();
        }
        operator.on_end(&context, &mut collector).await.unwrap();
        assert_eq!(
            operator
                .state
                .deltas
                .pending
                .iter()
                .map(PendingOp::identity)
                .collect::<Vec<_>>(),
            vec![
                (JoinSide::Left, 0),
                (JoinSide::Left, 2),
                (JoinSide::Left, 4)
            ]
        );
    }

    #[tokio::test]
    async fn timestamp_type_is_decoded_once_per_input_record() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        reset_join_work();
        operator
            .process_data("left", left_batch(vec![0, 1, 2]), &context, &mut collector)
            .await
            .unwrap();
        assert_eq!(join_work().time_decoders, 1);
    }

    #[tokio::test]
    async fn batch_timestamp_decoding_preserves_units_nulls_and_negative_floor() {
        for timezone in [None, Some("UTC".into())] {
            for (unit, values, expected) in [
                (
                    TimeUnit::Second,
                    vec![Some(-1), None, Some(2)],
                    vec![-1_000_000, 2_000_000],
                ),
                (
                    TimeUnit::Millisecond,
                    vec![Some(-1), None, Some(2)],
                    vec![-1_000, 2_000],
                ),
                (
                    TimeUnit::Microsecond,
                    vec![Some(-1), None, Some(2)],
                    vec![-1, 2],
                ),
                (
                    TimeUnit::Nanosecond,
                    vec![Some(-1), None, Some(1_999)],
                    vec![-1, 1],
                ),
            ] {
                let data_type = DataType::Timestamp(unit, timezone.clone());
                let schema = Arc::new(Schema::new(vec![
                    Field::new("account_id", DataType::Int64, false),
                    Field::new("authorized_at", data_type.clone(), true),
                    Field::new("amount", DataType::Int64, true),
                ]));
                let times = datafusion::arrow::compute::cast(&Int64Array::from(values), &data_type)
                    .unwrap();
                let record = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![
                        Arc::new(Int64Array::from(vec![7; 3])),
                        times,
                        Arc::new(Int64Array::from(vec![42; 3])),
                    ],
                )
                .unwrap();
                let mut operator =
                    StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
                let job_context = job();
                let context = StreamOperatorContext::new(&job_context, "match", None);
                let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
                operator
                    .process_data(
                        "left",
                        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                        &context,
                        &mut collector,
                    )
                    .await
                    .unwrap();
                assert_eq!(
                    operator
                        .state
                        .left
                        .iter()
                        .map(|row| row.event_time.as_micros())
                        .collect::<Vec<_>>(),
                    expected
                );
                assert_eq!(operator.state.next_left_row_id, 3);
                assert_eq!(operator.status().left.null_event_time_rows, 1);
            }
        }
    }

    #[tokio::test]
    async fn batch_timestamp_overflow_keeps_the_failure_reason_and_drop_precedence() {
        for unit in [TimeUnit::Second, TimeUnit::Millisecond] {
            let data_type = DataType::Timestamp(unit, None);
            let schema = Arc::new(Schema::new(vec![
                Field::new("account_id", DataType::Int64, true),
                Field::new("authorized_at", data_type.clone(), true),
                Field::new("amount", DataType::Int64, true),
            ]));
            let times = datafusion::arrow::compute::cast(
                &Int64Array::from(vec![Some(0), Some(i64::MAX)]),
                &data_type,
            )
            .unwrap();
            let record = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from(vec![Some(7), None])),
                    times,
                    Arc::new(Int64Array::from(vec![42; 2])),
                ],
            )
            .unwrap();
            let mut operator =
                StreamJoinOperator::new("match", schema, right_schema(), spec()).unwrap();
            let job_context = job();
            let context = StreamOperatorContext::new(&job_context, "match", None);
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            let error = operator
                .process_data(
                    "left",
                    Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap_err();
            assert!(matches!(
                error,
                CalcFlowError::OperatorReason {
                    reason_code: crate::StreamingFailureReason::JoinTimeConversionFailed,
                    ..
                }
            ));
            assert!(collector.drain("output").is_empty());
            assert_eq!(operator.state.next_left_row_id, 0);
            assert_eq!(operator.status().left.retained_rows, 0);
        }
    }

    mod checkpoint_compaction_tests;
    mod columnar_state_tests;
    mod native_lookup_tests;
    mod sql_key_scratch_tests;

    async fn v1_fixture_captures() -> Vec<OperatorStateSnapshot> {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "left",
                left_batch(vec![30, 0, 40, 1, 50, 2]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        let mut captures = vec![operator.checkpoint(Epoch::new(1).unwrap()).unwrap()];
        let progress = progress_context(
            &job_context,
            (IngressState::Active, None),
            (IngressState::Active, Some(60_000_001)),
        );
        operator
            .on_ingress_progress("right", &progress)
            .await
            .unwrap();
        captures.push(operator.checkpoint(Epoch::new(2).unwrap()).unwrap());
        operator
            .process_data("left", left_batch(vec![20]), &context, &mut collector)
            .await
            .unwrap();
        captures.push(operator.checkpoint(Epoch::new(3).unwrap()).unwrap());
        let progress = progress_context(
            &job_context,
            (IngressState::Active, None),
            (IngressState::Active, Some(60_000_003)),
        );
        operator
            .on_ingress_progress("right", &progress)
            .await
            .unwrap();
        captures.push(operator.checkpoint(Epoch::new(4).unwrap()).unwrap());
        operator.prepare_checkpoint_async(&context).await.unwrap();
        operator
            .process_data("left", left_batch(vec![]), &context, &mut collector)
            .await
            .unwrap();
        captures.push(operator.checkpoint(Epoch::new(5).unwrap()).unwrap());
        captures
    }

    #[tokio::test]
    async fn frozen_v1_checkpoint_fixture_preserves_wire_bytes_and_continuation() {
        let captures = v1_fixture_captures().await;
        let wire = captures.iter().map(|snapshot| serde_json::json!({
            "inline_metadata": snapshot.inline_metadata,
            "segments": snapshot.segments.iter().map(|(name, segment)| (name.clone(), hex::encode(segment.bytes()))).collect::<BTreeMap<_, _>>(),
        })).collect::<Vec<_>>();
        let frozen: Vec<Value> =
            serde_json::from_str(include_str!("join/fixtures/checkpoint-v1.json")).unwrap();
        assert_eq!(wire, frozen);
        let snapshot = frozen.last().unwrap();
        let snapshot = OperatorStateSnapshot {
            inline_metadata: serde_json::from_value(snapshot["inline_metadata"].clone()).unwrap(),
            segments: snapshot["segments"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(name, bytes)| {
                    (
                        name.clone(),
                        StateSegment::new(hex::decode(bytes.as_str().unwrap()).unwrap()),
                    )
                })
                .collect(),
        };
        let mut restored =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        restored.restore(&snapshot).unwrap();
        assert_eq!(restored.status().left.retained_rows, 4);
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(restored.output_ports().to_vec());
        restored
            .process_data("right", right_batch(vec![20]), &context, &mut collector)
            .await
            .unwrap();
        assert_eq!(restored.status().emitted_match_rows, 4);
        assert_eq!(
            collector.drain("output")[0]
                .as_data()
                .unwrap()
                .metadata()
                .sequence(),
            0
        );
    }

    #[tokio::test]
    async fn checkpoint_encodes_dirty_upserts_from_carried_records_not_live_state() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", left_batch(vec![0, 1, 2]), &context, &mut collector)
            .await
            .unwrap();
        // The dirty log must encode from the records it carries at admission:
        // the checkpoint path may not scan live state per dirty row, or capture
        // cost grows with the total retained state instead of the dirty set.
        operator.state.left.clear();
        let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();

        let mut restored =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        restored.restore(&snapshot).unwrap();
        assert_eq!(restored.status().left.retained_rows, 3);
    }

    #[tokio::test]
    async fn checkpoint_shares_carried_segment_allocations_across_epochs() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        let first = operator.checkpoint(Epoch::INITIAL).unwrap();
        operator
            .process_data("left", left_batch(vec![1]), &context, &mut collector)
            .await
            .unwrap();
        let second = operator.checkpoint(Epoch::INITIAL.next().unwrap()).unwrap();

        // Capture cost must stay proportional to the dirty set (spec FR47): a
        // segment the operator already encoded is carried into the next
        // snapshot by sharing its allocation, never by copying its bytes.
        let mut shared = 0_usize;
        for (segment_id, carried) in &second.segments {
            if let Some(original) = first.segments.get(segment_id) {
                assert!(
                    Arc::ptr_eq(&original.bytes_arc(), &carried.bytes_arc()),
                    "carried segment {segment_id:?} must share its allocation"
                );
                shared += 1;
            }
        }
        assert_eq!(shared, 1, "epoch 1 delta carries into epoch 2");
        assert!(
            second.segments.contains_key("left-delta-2"),
            "epoch 2 dirty ops encode a fresh segment"
        );
    }

    #[tokio::test]
    async fn restore_rejects_tampered_checkpoints() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();

        let fresh = |snapshot: &OperatorStateSnapshot| {
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec())
                .unwrap()
                .restore(snapshot)
        };

        let mut bad_magic = snapshot.clone();
        bad_magic
            .segments
            .insert("left-delta-1".into(), StateSegment::new(vec![0_u8; 8]));
        assert!(fresh(&bad_magic).is_err(), "invalid magic must be rejected");

        let mut short_inventory = snapshot.clone();
        short_inventory.segments.remove("left-delta-1");
        assert!(
            fresh(&short_inventory).is_err(),
            "missing segment must be rejected"
        );

        let mut truncated = snapshot.clone();
        let segment = truncated.segments.get_mut("left-delta-1").unwrap();
        let mut bytes = segment.bytes().to_vec();
        bytes.truncate(bytes.len() - 1);
        *segment = StateSegment::new(bytes);
        assert!(
            fresh(&truncated).is_err(),
            "truncated segment must be rejected"
        );

        let mut wrong_layout = snapshot.clone();
        wrong_layout
            .inline_metadata
            .insert("layout_version".into(), 2.into());
        assert!(
            fresh(&wrong_layout).is_err(),
            "layout bump must be rejected"
        );

        let mut wrong_metrics = snapshot.clone();
        wrong_metrics
            .inline_metadata
            .entry("metrics".into())
            .or_default()["left"]["retained_rows"] = 99.into();
        assert!(
            fresh(&wrong_metrics).is_err(),
            "inconsistent retained metrics must be rejected"
        );

        let mut wrong_limits = snapshot.clone();
        wrong_limits
            .inline_metadata
            .entry("spec".into())
            .or_default()["limits"]["max_state_rows_per_side"] = 5.into();
        assert!(
            fresh(&wrong_limits).is_err(),
            "spec change must be rejected"
        );

        let mut bad_metadata = snapshot.clone();
        bad_metadata
            .inline_metadata
            .insert("layout_version".into(), "not-a-number".into());
        assert!(
            fresh(&bad_metadata).is_err(),
            "invalid metadata must be rejected"
        );

        assert!(fresh(&snapshot).is_ok(), "the untampered snapshot restores");
    }

    #[test]
    fn reset_clears_state_for_reuse() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let record = left_batch(vec![0]).table_payload().unwrap().batches()[0].slice(0, 1);
        operator.state.left.push(StoredRow {
            encoded_key: Arc::new(encode_join_key_v1(&record, 0, &[0]).unwrap().into()),
            record: record.into(),
            event_time: EventTime::from_micros(0),
            row_id: 0,
            charge: 64,
        });
        operator.state.metrics.left.retained_rows = 1;

        operator.reset().unwrap();

        assert!(operator.state.left.is_empty());
        assert_eq!(operator.state.metrics.left.retained_rows, 0);
    }

    #[test]
    fn unchanged_opposite_state_reuses_join_key_arrays() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let first_record = right_batch(vec![0]).table_payload().unwrap().batches()[0].clone();
        operator.state.right.push(StoredRow {
            encoded_key: Arc::new(encode_join_key_v1(&first_record, 0, &[0]).unwrap().into()),
            record: first_record.into(),
            event_time: EventTime::from_micros(0),
            row_id: 0,
            charge: 64,
        });
        let plan = operator.side_plan("left").unwrap();
        let first = operator.opposite_state_keys(&plan).unwrap();
        let reused = operator.opposite_state_keys(&plan).unwrap();
        assert!(Arc::ptr_eq(first.column(0), reused.column(0)));

        let second_record = right_batch(vec![1]).table_payload().unwrap().batches()[0].clone();
        operator.state.right.push(StoredRow {
            encoded_key: Arc::new(encode_join_key_v1(&second_record, 0, &[0]).unwrap().into()),
            record: second_record.into(),
            event_time: EventTime::from_micros(1),
            row_id: 1,
            charge: 64,
        });
        let rebuilt = operator.opposite_state_keys(&plan).unwrap();
        assert!(!Arc::ptr_eq(first.column(0), rebuilt.column(0)));
        assert_eq!(rebuilt.num_rows(), 2);
    }

    #[tokio::test]
    async fn admitted_rows_release_the_previous_key_cache() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        let plan = operator.side_plan("right").unwrap();
        operator.opposite_state_keys(&plan).unwrap();
        assert!(operator.retained_key_cache.left.is_some());
        operator
            .process_data("left", left_batch(vec![1]), &context, &mut collector)
            .await
            .unwrap();
        assert!(operator.retained_key_cache.left.is_none());
    }

    #[test]
    fn metadata_exposes_data_only_configuration_and_debug() {
        let operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        assert_eq!(operator.name(), "match");
        assert_eq!(operator.input_ports().len(), 2);
        assert_eq!(operator.output_ports().len(), 1);
        let configuration = operator.configuration();
        assert_eq!(configuration["join_type"], "inner");
        assert_eq!(configuration["left_event_time"], "authorized_at");
        assert!(!configuration.contains_key("callable"));
        let debug = format!("{operator:?}");
        assert!(debug.contains("match"), "{debug}");
    }

    #[test]
    fn spec_validation_rejects_invalid_declarations() {
        let bounds =
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap();
        let limits = JoinStateLimits::new(100, 1_000_000, 1_000).unwrap();

        let empty_keys = StreamJoinSpec::inner(
            Vec::<String>::new(),
            ["account_id"],
            "authorized_at",
            "paid_at",
            bounds,
            limits,
        );
        assert!(empty_keys.is_err());

        let unequal = StreamJoinSpec::inner(
            ["a", "b"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            bounds,
            limits,
        );
        assert!(unequal.is_err());

        let duplicate = StreamJoinSpec::inner(
            ["account_id", "account_id"],
            ["account_id", "account_id"],
            "authorized_at",
            "paid_at",
            bounds,
            limits,
        );
        assert!(duplicate.is_err());

        let empty_event_time = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "",
            "paid_at",
            bounds,
            limits,
        );
        assert!(empty_event_time.is_err());

        let valid = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            bounds,
            limits,
        )
        .unwrap();
        assert!(valid.clone().with_prefixes("same", "same").is_err());
        assert!(valid.clone().with_prefixes("not valid", "right").is_err());

        let prefixed = valid.with_prefixes("authorization", "payment").unwrap();
        assert_eq!(prefixed.left_keys(), ["account_id"]);
        assert_eq!(prefixed.right_keys(), ["account_id"]);
        assert_eq!(prefixed.left_event_time(), "authorized_at");
        assert_eq!(prefixed.right_event_time(), "paid_at");
        assert_eq!(prefixed.left_prefix(), "authorization");
        assert_eq!(prefixed.right_prefix(), "payment");
        assert_eq!(prefixed.join_type(), StreamJoinType::Inner);
        assert_eq!(prefixed.bounds().before(), Duration::from_secs(300));
        assert_eq!(prefixed.bounds().after(), Duration::from_secs(60));
        assert_eq!(prefixed.limits().max_state_rows_per_side(), 100);
        assert_eq!(prefixed.limits().max_state_bytes_per_side(), 1_000_000);
        assert_eq!(prefixed.limits().max_matches_per_input_batch(), 1_000);
        assert!(format!("{prefixed:?}").contains("StreamJoinSpec"));
    }

    #[test]
    fn serde_round_trips_and_rejects_unknown_or_wrong_kind_fields() {
        let source = r#"{
            "join_type": "inner",
            "left_keys": ["account_id"],
            "right_keys": ["account_id"],
            "left_event_time": "authorized_at",
            "right_event_time": "paid_at",
            "bounds": {"before_micros": 300000000, "after_micros": 60000000},
            "limits": {
                "max_state_rows_per_side": 100,
                "max_state_bytes_per_side": 1000000,
                "max_matches_per_input_batch": 1000
            }
        }"#;
        let parsed: StreamJoinSpec = serde_json::from_str(source).unwrap();
        assert_eq!(parsed.left_prefix(), "left");
        assert_eq!(parsed.right_prefix(), "right");
        let encoded = serde_json::to_value(&parsed).unwrap();
        assert_eq!(encoded["bounds"]["before_micros"], 300_000_000);

        let unknown = source.replace(
            "\"join_type\": \"inner\",",
            "\"join_type\": \"inner\", \"extra\": 1,",
        );
        assert!(serde_json::from_str::<StreamJoinSpec>(&unknown).is_err());

        let outer_join = source.replace("\"inner\"", "\"outer\"");
        assert!(serde_json::from_str::<StreamJoinSpec>(&outer_join).is_err());

        let unknown_bound = source.replace(
            "\"before_micros\": 300000000,",
            "\"before_micros\": 300000000, \"extra\": 1,",
        );
        assert!(serde_json::from_str::<StreamJoinSpec>(&unknown_bound).is_err());

        let unknown_limit = source.replace(
            "\"max_matches_per_input_batch\": 1000",
            "\"max_matches_per_input_batch\": 1000, \"extra\": 1",
        );
        assert!(serde_json::from_str::<StreamJoinSpec>(&unknown_limit).is_err());

        let zero_limit = source.replace(
            "\"max_state_rows_per_side\": 100",
            "\"max_state_rows_per_side\": 0",
        );
        assert!(serde_json::from_str::<StreamJoinSpec>(&zero_limit).is_err());
    }

    #[tokio::test]
    async fn event_time_columns_accept_every_timestamp_unit_and_reject_others() {
        for (unit, _value) in [
            (TimeUnit::Second, 1_i64),
            (TimeUnit::Millisecond, 1_000),
            (TimeUnit::Microsecond, 1_000_000),
            (TimeUnit::Nanosecond, 1_000_000_000),
        ] {
            let schema = Arc::new(Schema::new(vec![
                Field::new("account_id", DataType::Int64, false),
                Field::new(
                    "authorized_at",
                    DataType::Timestamp(unit, Some("UTC".into())),
                    false,
                ),
                Field::new("amount", DataType::Int64, true),
            ]));
            let operator = StreamJoinOperator::new("match", schema, right_schema(), spec());
            assert!(operator.is_ok(), "unit {unit:?} must be supported");
        }

        let not_a_timestamp = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new("authorized_at", DataType::Int64, false),
            Field::new("amount", DataType::Int64, true),
        ]));
        let rejected = StreamJoinOperator::new("match", not_a_timestamp, right_schema(), spec());
        assert!(
            rejected.is_err(),
            "non-timestamp event time must be rejected"
        );

        let missing_key = Arc::new(Schema::new(vec![
            Field::new("ledger_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
        ]));
        let rejected = StreamJoinOperator::new("match", missing_key, right_schema(), spec());
        assert!(rejected.is_err(), "missing key column must be rejected");

        let wrong_key_type = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Float64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("amount", DataType::Int64, true),
        ]));
        let rejected = StreamJoinOperator::new("match", wrong_key_type, right_schema(), spec());
        assert!(rejected.is_err(), "unsupported key type must be rejected");

        let zoned = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("America/New_York".into())),
                false,
            ),
            Field::new("amount", DataType::Int64, true),
        ]));
        let rejected = StreamJoinOperator::new("match", zoned, right_schema(), spec());
        assert!(rejected.is_err(), "non-UTC timezone must be rejected");
    }

    #[tokio::test]
    async fn variable_width_payloads_are_charged_and_limited_deterministically() {
        let payload_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("notes", DataType::Utf8, true),
            Field::new("blob", DataType::Binary, true),
            Field::new("tag", DataType::FixedSizeBinary(4), true),
        ]));
        let keys_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "paid_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("status", DataType::Utf8, true),
        ]));
        let tiny = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
            JoinStateLimits::new(100, 96, 1_000).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("match", Arc::clone(&payload_schema), keys_schema, tiny)
                .unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        let batch = Batch::table(
            vec![
                RecordBatch::try_new(
                    payload_schema,
                    vec![
                        Arc::new(Int64Array::from(vec![7])),
                        Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
                        Arc::new(StringArray::from(vec!["0123456789"])),
                        Arc::new(BinaryArray::from_opt_vec(vec![Some(&[0_u8; 8][..])])),
                        Arc::new(
                            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                                vec![Some([1_u8; 4])].into_iter(),
                                4,
                            )
                            .unwrap(),
                        ),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap();

        let failure = operator
            .process_data("left", batch, &context, &mut collector)
            .await
            .unwrap_err();
        assert_eq!(
            reason_of(&failure),
            Some(crate::StreamingFailureReason::JoinStateLimitExceeded)
        );
        let metadata = checkpoint_metadata(&mut operator, 1);
        assert_eq!(metadata["metrics"]["left"]["retained_rows"], 0);
    }
}
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    io::Cursor,
    mem::size_of,
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use datafusion::arrow::{
    array::{
        Array, ArrayAccessor, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray,
        DictionaryArray, FixedSizeListArray, LargeBinaryArray, LargeListArray, LargeListViewArray,
        LargeStringArray, ListArray, ListViewArray, MapArray, PrimitiveArray, RunArray,
        StringArray, StringViewArray, StructArray, TimestampMicrosecondArray,
        TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt64Array,
        UnionArray, new_empty_array,
    },
    compute::concat,
    datatypes::{
        ArrowPrimitiveType, DataType, Field, Int8Type, Int16Type, Int32Type, Int64Type,
        IntervalUnit, Schema, SchemaRef, TimeUnit, TimestampMicrosecondType,
        TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type,
        UInt16Type, UInt32Type, UInt64Type,
    },
    ipc::reader::StreamReader,
    record_batch::RecordBatch,
};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::Value;

use crate::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, DataFusionConfig, Epoch, EventTime,
    IngressProgress, IngressProgressSnapshot, JsonMap, OperatorStateSnapshot, Port, Result,
    StateSegment, StreamCollector, StreamOperator, StreamOperatorContext, UdfRegistrySnapshot,
    expression::{ValidatedQuery, parse_select_query},
};

use super::{OperatorMetadata, StreamRuntimeState, is_portable_identifier, validate_operator_name};

#[cfg(test)]
#[derive(Clone, Copy, Default)]
struct JoinWork {
    retained_visits: usize,
    pending_visits: usize,
    key_encodings: usize,
    time_decoders: usize,
    sql_probe_table_builds: usize,
}

#[cfg(test)]
thread_local! {
    static JOIN_WORK: std::cell::Cell<JoinWork> = const { std::cell::Cell::new(JoinWork {
        retained_visits: 0, pending_visits: 0, key_encodings: 0, time_decoders: 0,
        sql_probe_table_builds: 0,
    }) };
}

#[cfg(test)]
fn join_work() -> JoinWork {
    JOIN_WORK.get()
}

#[cfg(test)]
fn reset_join_work() {
    JOIN_WORK.set(JoinWork::default());
}

#[cfg(test)]
fn note_join_work(note: impl FnOnce(&mut JoinWork)) {
    let mut work = join_work();
    note(&mut work);
    JOIN_WORK.set(work);
}

/// The fixed logical bookkeeping charge for one retained Join row.
pub const STREAM_JOIN_STATE_ROW_OVERHEAD_BYTES_V1: u64 = 64;

/// Largest integer that round-trips exactly through ordinary JSON numbers.
pub const STREAM_JOIN_MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;

/// Supported Join semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StreamJoinType {
    /// Emit every pair with equal non-null keys and an event time inside the
    /// configured inclusive interval.
    Inner,
}

/// Inclusive event-time distance around one left row.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JoinTimeBounds {
    #[schemars(range(min = 0, max = 9_007_199_254_740_991_u64))]
    before_micros: u64,
    #[schemars(range(min = 0, max = 9_007_199_254_740_991_u64))]
    after_micros: u64,
}

impl JoinTimeBounds {
    /// Creates exact, non-negative microsecond bounds.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when either duration has
    /// sub-microsecond precision or exceeds the exact JSON integer domain.
    pub fn new(before: Duration, after: Duration) -> Result<Self> {
        Ok(Self {
            before_micros: exact_safe_duration_micros(before, "stream_join.bounds.before_micros")?,
            after_micros: exact_safe_duration_micros(after, "stream_join.bounds.after_micros")?,
        })
    }

    pub(crate) fn from_micros(before_micros: u64, after_micros: u64) -> Result<Self> {
        validate_safe_integer(before_micros, false, "stream_join.bounds.before_micros")?;
        validate_safe_integer(after_micros, false, "stream_join.bounds.after_micros")?;
        Ok(Self {
            before_micros,
            after_micros,
        })
    }

    /// Returns the exact preceding distance in microseconds.
    pub const fn before_micros(self) -> u64 {
        self.before_micros
    }

    /// Returns the exact following distance in microseconds.
    pub const fn after_micros(self) -> u64 {
        self.after_micros
    }

    /// Returns the preceding distance.
    pub const fn before(self) -> Duration {
        Duration::from_micros(self.before_micros)
    }

    /// Returns the following distance.
    pub const fn after(self) -> Duration {
        Duration::from_micros(self.after_micros)
    }

    pub(crate) fn contains_pair(self, left_micros: i64, right_micros: i64) -> bool {
        let left = i128::from(left_micros);
        let right = i128::from(right_micros);
        right >= left - i128::from(self.before_micros)
            && right <= left + i128::from(self.after_micros)
    }
}

impl<'de> Deserialize<'de> for JoinTimeBounds {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            before_micros: u64,
            after_micros: u64,
        }

        let fields = Fields::deserialize(deserializer)?;
        Self::from_micros(fields.before_micros, fields.after_micros).map_err(D::Error::custom)
    }
}

/// Hard logical state and per-input fan-out limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[allow(
    clippy::struct_field_names,
    reason = "the frozen public JSON field names all use the max_ limit prefix"
)]
#[serde(deny_unknown_fields)]
pub struct JoinStateLimits {
    #[schemars(range(min = 1, max = 9_007_199_254_740_991_u64))]
    max_state_rows_per_side: u64,
    #[schemars(range(min = 1, max = 9_007_199_254_740_991_u64))]
    max_state_bytes_per_side: u64,
    #[schemars(range(min = 1, max = 9_007_199_254_740_991_u64))]
    max_matches_per_input_batch: u64,
}

impl JoinStateLimits {
    /// Creates required positive Join limits.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when a value is zero or is
    /// larger than [`STREAM_JOIN_MAX_SAFE_JSON_INTEGER`].
    pub fn new(
        max_state_rows_per_side: u64,
        max_state_bytes_per_side: u64,
        max_matches_per_input_batch: u64,
    ) -> Result<Self> {
        validate_safe_integer(
            max_state_rows_per_side,
            true,
            "stream_join.limits.max_state_rows_per_side",
        )?;
        validate_safe_integer(
            max_state_bytes_per_side,
            true,
            "stream_join.limits.max_state_bytes_per_side",
        )?;
        validate_safe_integer(
            max_matches_per_input_batch,
            true,
            "stream_join.limits.max_matches_per_input_batch",
        )?;
        Ok(Self {
            max_state_rows_per_side,
            max_state_bytes_per_side,
            max_matches_per_input_batch,
        })
    }

    /// Maximum retained rows on either side.
    pub const fn max_state_rows_per_side(self) -> u64 {
        self.max_state_rows_per_side
    }

    /// Maximum logical retained bytes on either side.
    pub const fn max_state_bytes_per_side(self) -> u64 {
        self.max_state_bytes_per_side
    }

    /// Maximum pairs one accepted input batch may emit.
    pub const fn max_matches_per_input_batch(self) -> u64 {
        self.max_matches_per_input_batch
    }
}

impl<'de> Deserialize<'de> for JoinStateLimits {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[allow(
            clippy::struct_field_names,
            reason = "the wire DTO must preserve the frozen max_ field names"
        )]
        #[serde(deny_unknown_fields)]
        struct Fields {
            max_state_rows_per_side: u64,
            max_state_bytes_per_side: u64,
            max_matches_per_input_batch: u64,
        }

        let fields = Fields::deserialize(deserializer)?;
        Self::new(
            fields.max_state_rows_per_side,
            fields.max_state_bytes_per_side,
            fields.max_matches_per_input_batch,
        )
        .map_err(D::Error::custom)
    }
}

/// Immutable declaration for a two-input bounded inner stream Join.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StreamJoinSpec {
    join_type: StreamJoinType,
    left_keys: Vec<String>,
    right_keys: Vec<String>,
    left_event_time: String,
    right_event_time: String,
    bounds: JoinTimeBounds,
    limits: JoinStateLimits,
    left_prefix: String,
    right_prefix: String,
}

impl StreamJoinSpec {
    /// Canonical `left_prefix` default materialized after a clean raw pass.
    pub const DEFAULT_LEFT_PREFIX: &'static str = "left";

    /// Canonical `right_prefix` default materialized after a clean raw pass.
    pub const DEFAULT_RIGHT_PREFIX: &'static str = "right";

    /// Creates an inner Join with canonical `left` and `right` prefixes.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for empty, duplicate, or
    /// unequally sized key declarations and invalid event-time names.
    pub fn inner<L, R, LI, RI>(
        left_keys: L,
        right_keys: R,
        left_event_time: &str,
        right_event_time: &str,
        bounds: JoinTimeBounds,
        limits: JoinStateLimits,
    ) -> Result<Self>
    where
        L: IntoIterator<Item = LI>,
        R: IntoIterator<Item = RI>,
        LI: Into<String>,
        RI: Into<String>,
    {
        let left_keys = left_keys.into_iter().map(Into::into).collect::<Vec<_>>();
        let right_keys = right_keys.into_iter().map(Into::into).collect::<Vec<_>>();
        validate_key_names(&left_keys, &right_keys)?;
        validate_column_name(left_event_time, "stream_join.left_event_time")?;
        validate_column_name(right_event_time, "stream_join.right_event_time")?;
        Ok(Self {
            join_type: StreamJoinType::Inner,
            left_keys,
            right_keys,
            left_event_time: left_event_time.into(),
            right_event_time: right_event_time.into(),
            bounds,
            limits,
            left_prefix: "left".into(),
            right_prefix: "right".into(),
        })
    }

    /// Replaces both output prefixes without mutating the original value.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] unless both values are
    /// distinct portable identifiers.
    pub fn with_prefixes(mut self, left_prefix: &str, right_prefix: &str) -> Result<Self> {
        validate_prefixes(left_prefix, right_prefix)?;
        self.left_prefix = left_prefix.into();
        self.right_prefix = right_prefix.into();
        Ok(self)
    }

    pub const fn join_type(&self) -> StreamJoinType {
        self.join_type
    }

    pub fn left_keys(&self) -> &[String] {
        &self.left_keys
    }

    pub fn right_keys(&self) -> &[String] {
        &self.right_keys
    }

    pub fn left_event_time(&self) -> &str {
        &self.left_event_time
    }

    pub fn right_event_time(&self) -> &str {
        &self.right_event_time
    }

    pub const fn bounds(&self) -> JoinTimeBounds {
        self.bounds
    }

    pub const fn limits(&self) -> JoinStateLimits {
        self.limits
    }

    pub fn left_prefix(&self) -> &str {
        &self.left_prefix
    }

    pub fn right_prefix(&self) -> &str {
        &self.right_prefix
    }
}

impl<'de> Deserialize<'de> for StreamJoinSpec {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            join_type: StreamJoinType,
            left_keys: Vec<String>,
            right_keys: Vec<String>,
            left_event_time: String,
            right_event_time: String,
            bounds: JoinTimeBounds,
            limits: JoinStateLimits,
            #[serde(default = "default_left_prefix")]
            left_prefix: String,
            #[serde(default = "default_right_prefix")]
            right_prefix: String,
        }

        let fields = Fields::deserialize(deserializer)?;
        if fields.join_type != StreamJoinType::Inner {
            return Err(D::Error::custom("only inner stream joins are supported"));
        }
        Self::inner(
            fields.left_keys,
            fields.right_keys,
            &fields.left_event_time,
            &fields.right_event_time,
            fields.bounds,
            fields.limits,
        )
        .and_then(|spec| spec.with_prefixes(&fields.left_prefix, &fields.right_prefix))
        .map_err(D::Error::custom)
    }
}

/// Payload-free status snapshot for one side of a retained Join state
/// (api note "Payload-free Join status").
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StreamJoinSideStatus {
    /// Logically retained rows on this side.
    pub retained_rows: u64,
    /// Versioned logical byte charge of the retained rows.
    pub retained_bytes: u64,
    /// Rows removed by watermark eviction or a side End.
    pub evicted_rows: u64,
    /// Rows dropped as late under this side's watermark.
    pub late_rows: u64,
    /// Input batches that contained at least one late row.
    pub late_affected_batches: u64,
    /// Largest observed lateness, if any late row was seen.
    pub max_lateness: Option<Duration>,
    /// Rows dropped because their event time was null.
    pub null_event_time_rows: u64,
    /// Rows dropped because a key component was null.
    pub null_key_rows: u64,
    /// Most recent accepted ingress watermark, if one has been established.
    pub watermark_micros: Option<EventTime>,
    /// Whether this input is currently idle; idle preserves its watermark.
    pub idle: bool,
    /// Whether this input has permanently ended.
    pub ended: bool,
}

/// Payload-free Join status for one node (api note "Payload-free Join status").
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StreamJoinStatus {
    /// Retained left-side state.
    pub left: StreamJoinSideStatus,
    /// Retained right-side state.
    pub right: StreamJoinSideStatus,
    /// Match rows emitted so far.
    pub emitted_match_rows: u64,
    /// State-limit admission failures.
    pub state_limit_failures: u64,
    /// Match-limit admission failures.
    pub match_limit_failures: u64,
}

impl StreamJoinStatus {
    pub(crate) fn with_ingress_progress(mut self, progress: &IngressProgressSnapshot) -> Self {
        for (name, side) in [("left", &mut self.left), ("right", &mut self.right)] {
            if let Some(ingress) = progress.get(name) {
                side.watermark_micros = ingress.watermark();
                side.idle = ingress.state() == crate::IngressState::Idle;
                side.ended = ingress.state() == crate::IngressState::Ended;
            }
        }
        self
    }
}

fn side_status(metrics: &SideMetrics) -> StreamJoinSideStatus {
    StreamJoinSideStatus {
        retained_rows: metrics.retained_rows,
        retained_bytes: metrics.retained_bytes,
        evicted_rows: metrics.evicted_rows,
        late_rows: metrics.late_rows,
        late_affected_batches: metrics.late_affected_batches,
        max_lateness: metrics.max_lateness_micros.map(Duration::from_micros),
        null_event_time_rows: metrics.null_event_time_rows,
        null_key_rows: metrics.null_key_rows,
        watermark_micros: None,
        idle: false,
        ended: false,
    }
}

/// Stateful two-input bounded event-time Join.
pub struct StreamJoinOperator {
    name: String,
    spec: StreamJoinSpec,
    input_ports: [Port; 2],
    output_ports: [Port; 1],
    compiled: CompiledJoin,
    payload_schema_bytes: [Option<usize>; 2],
    payload_native_eligible: bool,
    runtime: StreamRuntimeState,
    state: StreamJoinState,
    retained_key_cache: RetainedKeyCache,
    ingress_progress: IngressProgressSnapshot,
    compaction_release: Option<tokio::sync::oneshot::Receiver<()>>,
    compaction_cleanup: Option<crate::runtime::streaming::gather_work::AttemptCleanup>,
    #[cfg(test)]
    checkpoint_gate: Option<std::sync::Mutex<checkpoint_compaction::TestGate>>,
    #[cfg(test)]
    checkpoint_retirement_gate: Option<std::sync::Mutex<checkpoint_compaction::TestRetirementGate>>,
    #[cfg(test)]
    metadata_test_hook: Option<MetadataTestHook>,
    #[cfg(test)]
    schema_test_hook: Option<SchemaTestHook>,
}

const MAX_RETAINED_KEY_CACHE_BYTES_PER_SIDE: usize = 32 * 1024 * 1024;

#[derive(Default)]
struct RetainedKeyCache {
    left: Option<CachedRetainedKeys>,
    right: Option<CachedRetainedKeys>,
}

struct CachedRetainedKeys {
    row_ids: Vec<u64>,
    batch: RecordBatch,
    scratch_funding: Option<Arc<sql_key_scratch::ScratchFunding>>,
}

struct SqlKeyOwners {
    _retained: Arc<Vec<StoredRow>>,
    admitted: Vec<AdmittedRow>,
    scratch: Option<Arc<sql_key_scratch::ScratchFunding>>,
}

impl SqlKeyOwners {
    fn finish(self) -> Vec<AdmittedRow> {
        let Self {
            _retained: retained,
            admitted,
            scratch,
        } = self;
        drop(retained);
        drop(scratch);
        admitted
    }
}

#[derive(Clone)]
struct CompiledJoin {
    left_key_indices: Vec<usize>,
    right_key_indices: Vec<usize>,
    left_event_time_index: usize,
    right_event_time_index: usize,
    equality_query: ValidatedQuery,
}

/// Scratch-table alias holding the admitted rows of the current input batch.
const PROBE_TABLE: &str = "probe_input";
/// Scratch-table alias holding the opposite side's retained state rows.
const STATE_TABLE: &str = "state_input";
/// Renamed key column prefix shared by both scratch tables.
const KEY_COLUMN_PREFIX: &str = "__cf_join_key_";
/// Position of one admitted row inside [`PROBE_TABLE`].
const PROBE_POS_COLUMN: &str = "__cf_join_pos";
/// Retained-state row id inside [`STATE_TABLE`].
const STATE_RID_COLUMN: &str = "__cf_join_row_id";

/// One incoming row that passed null-key, null-event-time, and lateness admission.
struct AdmittedRow {
    record: columnar::RowPayload,
    event_time: EventTime,
    row_id: u64,
    retain: bool,
}

/// Scratch accumulator for one input batch's admission pass.
struct AdmissionBundle {
    next_row_id: u64,
    metrics: SideMetrics,
    admitted: Vec<AdmittedRow>,
    /// Source row offsets of admitted rows, aligning charge vectors with
    /// `admitted` when admission drops rows.
    admitted_source_rows: Vec<usize>,
    had_late: bool,
}

/// Why an incoming physical row was dropped during admission.
#[derive(Clone, Copy)]
enum DropKind {
    NullEventTime,
    NullKey,
    Late(u64),
}

/// Classified disposition of one incoming physical row.
enum RowAdmission {
    Dropped(DropKind),
    Admitted(EventTime),
}

/// One time-qualified key-equal pair, ordered for emission.
struct MatchedPair {
    pos: usize,
    opposite_index: usize,
}

#[derive(Clone)]
struct StoredRow {
    record: columnar::RowPayload,
    event_time: EventTime,
    row_id: u64,
    charge: u64,
    encoded_key: Arc<columnar::FramedKey>,
}

/// One side of the Join state, in durable-identity order.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum JoinSide {
    Left,
    Right,
}

impl JoinSide {
    const fn as_str(self) -> &'static str {
        match self {
            JoinSide::Left => "left",
            JoinSide::Right => "right",
        }
    }
}

/// One dirty state change since the last captured checkpoint (spec FR45).
#[derive(Clone)]
enum PendingOp {
    Upsert {
        side: JoinSide,
        row_id: u64,
        event_time: EventTime,
        encoded_key: Arc<columnar::FramedKey>,
        record: columnar::RowPayload,
        charge: u64,
    },
    Tombstone {
        side: JoinSide,
        row_id: u64,
        event_time: EventTime,
        encoded_key: Arc<columnar::FramedKey>,
    },
}

impl PendingOp {
    fn identity(&self) -> (JoinSide, u64) {
        match self {
            PendingOp::Upsert { side, row_id, .. } | PendingOp::Tombstone { side, row_id, .. } => {
                (*side, *row_id)
            }
        }
    }
}

#[derive(Default)]
struct PendingLog {
    slots: Vec<Option<PendingEntry>>,
    free: Vec<usize>,
    upserts: HashMap<(JoinSide, u64), usize>,
    head: Option<usize>,
    tail: Option<usize>,
}

struct PendingEntry {
    op: PendingOp,
    previous: Option<usize>,
    next: Option<usize>,
}

impl PendingLog {
    fn is_empty(&self) -> bool {
        self.head.is_none()
    }

    fn iter(&self) -> impl Iterator<Item = &PendingOp> {
        let mut next = self.head;
        std::iter::from_fn(move || {
            let entry = self.slots[next?].as_ref().expect("live pending link");
            next = entry.next;
            Some(&entry.op)
        })
    }

    fn push(&mut self, op: PendingOp) {
        let slot = self.free.pop().unwrap_or(self.slots.len());
        if matches!(&op, PendingOp::Upsert { .. }) {
            let previous = self.upserts.insert(op.identity(), slot);
            debug_assert!(previous.is_none(), "pending upsert identities are unique");
        }
        let entry = Some(PendingEntry {
            op,
            previous: self.tail,
            next: None,
        });
        if slot == self.slots.len() {
            self.slots.push(entry);
        } else {
            self.slots[slot] = entry;
        }
        if let Some(tail) = self.tail {
            self.slots[tail].as_mut().expect("live pending tail").next = Some(slot);
        } else {
            self.head = Some(slot);
        }
        self.tail = Some(slot);
    }

    fn remove_upsert(&mut self, identity: (JoinSide, u64)) -> bool {
        let Some(slot) = self.upserts.remove(&identity) else {
            return false;
        };
        let entry = self.slots[slot].take().expect("indexed pending upsert");
        if let Some(previous) = entry.previous {
            self.slots[previous]
                .as_mut()
                .expect("live previous pending entry")
                .next = entry.next;
        } else {
            self.head = entry.next;
        }
        if let Some(next) = entry.next {
            self.slots[next]
                .as_mut()
                .expect("live next pending entry")
                .previous = entry.previous;
        } else {
            self.tail = entry.previous;
        }
        self.free.push(slot);
        true
    }

    fn clear(&mut self) {
        self.slots.clear();
        self.free.clear();
        self.upserts.clear();
        self.head = None;
        self.tail = None;
    }
}

/// Prepared checkpoint segments and the dirty log (spec FR45/FR47).
///
/// Bulk encoding and compaction are prepared asynchronously before capture;
/// `checkpoint` only shares the prepared segment allocations and encodes the
/// dirty ops.
#[derive(Default)]
struct DeltaTracking {
    base: BTreeMap<&'static str, StateSegment>,
    segments: BTreeMap<(u64, &'static str), StateSegment>,
    pending: PendingLog,
    segments_since_base: u32,
    needs_compaction: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
struct SideMetrics {
    retained_rows: u64,
    retained_bytes: u64,
    evicted_rows: u64,
    late_rows: u64,
    late_affected_batches: u64,
    max_lateness_micros: Option<u64>,
    null_event_time_rows: u64,
    null_key_rows: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
struct JoinMetrics {
    left: SideMetrics,
    right: SideMetrics,
    emitted_match_rows: u64,
    state_limit_failures: u64,
    match_limit_failures: u64,
}

#[derive(Default)]
struct StreamJoinState {
    left: RetainedRows,
    right: RetainedRows,
    left_expirations: ExpirationIndex,
    right_expirations: ExpirationIndex,
    next_left_row_id: u64,
    next_right_row_id: u64,
    next_output_sequence: u64,
    metrics: JoinMetrics,
    ended: bool,
    last_checkpoint_epoch: Option<Epoch>,
    deltas: DeltaTracking,
}

#[derive(Default)]
struct RetainedRows(
    Arc<Vec<StoredRow>>,
    Option<native_lookup::NativeIndex>,
    columnar::SparseQueue,
);

impl From<Vec<StoredRow>> for RetainedRows {
    fn from(rows: Vec<StoredRow>) -> Self {
        Self(Arc::new(rows), None, columnar::SparseQueue::default())
    }
}

impl RetainedRows {
    fn extend(&mut self, rows: Vec<StoredRow>) {
        for row in &rows {
            self.2.enqueue_if_due(&row.record);
        }
        if let Some(index) = &mut self.1 {
            index.append(self.0.len(), &rows);
        }
        std::ops::DerefMut::deref_mut(self).extend(rows);
    }

    fn swap_remove(&mut self, index: usize) -> StoredRow {
        let row = std::ops::DerefMut::deref_mut(self).swap_remove(index);
        if let Some(native) = &mut self.1 {
            native.remove(&row, self.0.get(index).map(|row| (row, index)));
        }
        row
    }

    fn clear(&mut self) {
        std::ops::DerefMut::deref_mut(self).clear();
        self.1 = None;
        self.2.clear();
    }
}

impl std::ops::Deref for RetainedRows {
    type Target = Vec<StoredRow>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for RetainedRows {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::get_mut(&mut self.0).expect("compaction input released before retained mutation")
    }
}

#[derive(Default)]
struct ExpirationIndex {
    entries: BTreeMap<(EventTime, u64), (usize, u128)>,
    next_ordinal: u128,
}

impl ExpirationIndex {
    fn restored(rows: &[StoredRow]) -> Self {
        Self {
            entries: rows
                .iter()
                .enumerate()
                .map(|(index, row)| ((row.event_time, row.row_id), (index, index as u128)))
                .collect(),
            next_ordinal: rows.len() as u128,
        }
    }

    fn append(&mut self, offset: usize, rows: &[StoredRow]) {
        for (index, row) in rows.iter().enumerate() {
            self.entries.insert(
                (row.event_time, row.row_id),
                (offset + index, self.next_ordinal + index as u128),
            );
        }
        self.next_ordinal += rows.len() as u128;
    }

    fn identities(&self, rows: &[StoredRow]) -> Vec<(u64, EventTime, Arc<columnar::FramedKey>)> {
        let mut ordered = self.entries.values().copied().collect::<Vec<_>>();
        ordered.sort_by_key(|(_, ordinal)| *ordinal);
        ordered
            .into_iter()
            .map(|(index, _)| {
                let row = &rows[index];
                (row.row_id, row.event_time, Arc::clone(&row.encoded_key))
            })
            .collect()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct JoinCheckpointMetadata {
    layout_version: u32,
    spec: StreamJoinSpec,
    next_left_row_id: u64,
    next_right_row_id: u64,
    next_output_sequence: u64,
    metrics: JoinMetrics,
    ended: bool,
    epoch: u64,
}

#[cfg(test)]
type MetadataTestHook =
    Arc<dyn Fn(Option<&datafusion::execution::memory_pool::MemoryReservation>, bool) + Send + Sync>;

#[cfg(test)]
type SchemaTestHook = MetadataTestHook;

#[derive(Clone, Copy)]
struct RestoreSchema<'a> {
    schema: &'a Schema,
    #[cfg(test)]
    hook: Option<&'a SchemaTestHook>,
    #[cfg(test)]
    credit: Option<&'a datafusion::execution::memory_pool::MemoryReservation>,
}

mod checkpoint_compaction;
mod columnar;
mod materialization;
mod metadata_validation;
mod native_lookup;
mod row_ipc;
mod sql_key_scratch;

struct PreparedMatches {
    pairs: Vec<MatchedPair>,
    keys: Option<native_lookup::NativeKeys>,
    credit: Option<datafusion::execution::memory_pool::MemoryReservation>,
}

impl PreparedMatches {
    fn legacy(pairs: Vec<MatchedPair>) -> Self {
        Self {
            pairs,
            keys: None,
            credit: None,
        }
    }
}

struct PreparedJoinBatch {
    output: Vec<MatchedPair>,
    admitted: Vec<AdmittedRow>,
    incoming_is_left: bool,
    retained: Vec<StoredRow>,
    next_row_id: u64,
    metrics: SideMetrics,
    /// Conservative per-admitted-row logical charges enabling the single
    /// chunk fast path; `None` keeps the generic measured planning.
    admitted_charges: Option<Vec<u64>>,
    native_append: Option<native_lookup::AppendCredit>,
    _native_scratch: Option<datafusion::execution::memory_pool::MemoryReservation>,
}

impl StreamJoinOperator {
    /// Compiles one Join declaration against two exact Arrow schemas.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for declaration errors and
    /// [`CalcFlowError::Compile`] for incompatible schemas.
    pub fn new(
        name: &str,
        left_schema: SchemaRef,
        right_schema: SchemaRef,
        spec: StreamJoinSpec,
    ) -> Result<Self> {
        validate_operator_name(name)?;
        validate_payload_charge_support(&left_schema, "left")?;
        validate_payload_charge_support(&right_schema, "right")?;
        let (output_schema, compiled) = compile_schemas(&left_schema, &right_schema, &spec)?;
        let payload_native_eligible = native_lookup::eligible(&compiled, &left_schema);
        let payload_schema_bytes = [
            columnar::schema_inventory(&left_schema),
            columnar::schema_inventory(&right_schema),
        ];
        Ok(Self {
            name: name.into(),
            spec,
            payload_schema_bytes,
            payload_native_eligible,
            input_ports: [
                Port::with_schema_ref("left", BatchKind::Table, true, Some(left_schema))?,
                Port::with_schema_ref("right", BatchKind::Table, true, Some(right_schema))?,
            ],
            output_ports: [Port::with_schema_ref(
                "output",
                BatchKind::Table,
                true,
                Some(output_schema),
            )?],
            compiled,
            runtime: StreamRuntimeState::new(),
            state: StreamJoinState::default(),
            retained_key_cache: RetainedKeyCache::default(),
            ingress_progress: IngressProgressSnapshot::default(),
            compaction_release: None,
            compaction_cleanup: None,
            #[cfg(test)]
            checkpoint_gate: None,
            #[cfg(test)]
            checkpoint_retirement_gate: None,
            #[cfg(test)]
            metadata_test_hook: None,
            #[cfg(test)]
            schema_test_hook: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn set_checkpoint_metadata_test_hook(&mut self, hook: MetadataTestHook) {
        self.metadata_test_hook = Some(hook);
    }

    #[cfg(test)]
    pub(crate) fn set_checkpoint_schema_test_hook(&mut self, hook: SchemaTestHook) {
        self.schema_test_hook = Some(hook);
    }

    /// Returns the immutable Join declaration.
    pub const fn spec(&self) -> &StreamJoinSpec {
        &self.spec
    }

    /// Returns a payload-free snapshot of retained state and observed ingress progress.
    ///
    /// Standalone restore has no ingress progress until a handler context supplies it.
    /// Non-terminal managed restart publishes restored progress before the running
    /// job's startup acknowledgement.
    pub fn status(&self) -> StreamJoinStatus {
        let mut status = StreamJoinStatus {
            left: side_status(&self.state.metrics.left),
            right: side_status(&self.state.metrics.right),
            emitted_match_rows: self.state.metrics.emitted_match_rows,
            state_limit_failures: self.state.metrics.state_limit_failures,
            match_limit_failures: self.state.metrics.match_limit_failures,
        }
        .with_ingress_progress(&self.ingress_progress);
        if self.state.ended {
            status.left.ended = true;
            status.right.ended = true;
            status.left.idle = false;
            status.right.idle = false;
        }
        status
    }

    pub(crate) fn set_stream_resources(
        &mut self,
        config: DataFusionConfig,
        udfs: UdfRegistrySnapshot,
    ) {
        self.runtime.set_resources(config, udfs, Vec::new());
    }

    pub(crate) const fn stream_runtime_initialized(&self) -> bool {
        self.runtime.is_initialized()
    }

    pub(crate) fn prepare_checkpoint_preload_runtime(&mut self) -> Result<()> {
        self.runtime.runtime()?;
        Ok(())
    }

    pub(crate) fn reserve_checkpoint_preload(
        &self,
        bytes: usize,
    ) -> Result<datafusion::execution::memory_pool::MemoryReservation> {
        let runtime = self
            .runtime
            .runtime
            .as_ref()
            .ok_or_else(|| CalcFlowError::Internal {
                message: "Join checkpoint preload runtime was not prepared".into(),
            })?;
        let credit = runtime.incremental_reservation("stream-join-preload");
        credit
            .try_grow(bytes)
            .map_err(|error| CalcFlowError::DataFusion {
                node_id: Some(self.name.clone()),
                message: error.to_string(),
            })?;
        Ok(credit)
    }

    #[cfg(test)]
    pub(crate) fn checkpoint_preload_test_pool(
        &mut self,
    ) -> Result<Arc<dyn datafusion::execution::memory_pool::MemoryPool>> {
        Ok(self.runtime.runtime()?.incremental_memory_pool())
    }

    pub(crate) fn output_frontier_candidate(
        &self,
        progress: &IngressProgressSnapshot,
    ) -> Result<Option<EventTime>> {
        let left = progress.get("left").ok_or_else(|| {
            operator_error(
                &self.name,
                "missing left ingress progress for output frontier",
            )
        })?;
        let right = progress.get("right").ok_or_else(|| {
            operator_error(
                &self.name,
                "missing right ingress progress for output frontier",
            )
        })?;
        let left_live = left.state() != crate::IngressState::Ended;
        let right_live = right.state() != crate::IngressState::Ended;
        let candidate = match (left_live, right_live) {
            (true, true) => match (left.watermark(), right.watermark()) {
                (Some(left), Some(right)) => Some(
                    (i128::from(left.as_micros()) - i128::from(self.spec.bounds.before_micros))
                        .min(
                            i128::from(right.as_micros())
                                - i128::from(self.spec.bounds.after_micros),
                        ),
                ),
                _ => None,
            },
            (true, false) => left.watermark().map(|left| {
                i128::from(left.as_micros()) - i128::from(self.spec.bounds.before_micros)
            }),
            (false, true) => right.watermark().map(|right| {
                i128::from(right.as_micros()) - i128::from(self.spec.bounds.after_micros)
            }),
            (false, false) => None,
        };
        candidate
            .filter(|candidate| *candidate >= i128::from(i64::MIN))
            .map(|candidate| {
                i64::try_from(candidate)
                    .map(EventTime::from_micros)
                    .map_err(|_| operator_error(&self.name, "output frontier exceeds EventTime"))
            })
            .transpose()
    }

    async fn prepare_batch(
        &mut self,
        ingress: &str,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedJoinBatch> {
        let plan = self.begin_batch(ingress, batch)?;
        let mut bundle = self.admission_bundle(&plan);
        let mut charges: Option<Vec<u64>> = Some(Vec::new());
        let mut source_row_base = 0_usize;
        for record in batch.table_payload()?.batches() {
            append_admission_charges(&mut charges, record)?;
            self.admit_record(
                record,
                &plan,
                ingress,
                context,
                &mut bundle,
                source_row_base,
            )
            .await?;
            source_row_base += record.num_rows();
        }
        bundle.finish(&self.name)?;
        let (matches, admitted) = self
            .evaluate_matches(&plan, std::mem::take(&mut bundle.admitted), context)
            .await;
        bundle.admitted = admitted;
        let matches = matches?;
        let admitted_charges = admitted_charge_cache(charges, &bundle.admitted_source_rows);
        self.finish_prepared(&plan, bundle, matches, admitted_charges)
    }

    fn finish_prepared(
        &mut self,
        plan: &SidePlan,
        bundle: AdmissionBundle,
        matches: PreparedMatches,
        admitted_charges: Option<Vec<u64>>,
    ) -> Result<PreparedJoinBatch> {
        let retained = retained_rows(
            &bundle.admitted,
            &plan.key_indices,
            &self.name,
            matches.keys.as_ref().map(|keys| keys.keys.as_slice()),
        )?;
        self.validate_state_admission(plan.incoming_is_left, &retained)?;
        let native_append = self.reserve_native_append(plan.incoming_is_left, retained.len())?;
        Ok(PreparedJoinBatch {
            output: matches.pairs,
            admitted: bundle.admitted,
            incoming_is_left: plan.incoming_is_left,
            retained,
            next_row_id: bundle.next_row_id,
            metrics: bundle.metrics,
            admitted_charges,
            native_append,
            _native_scratch: matches.credit,
        })
    }

    /// Validates the ingress and port contract before any admission work.
    fn begin_batch(&self, ingress: &str, batch: &Batch) -> Result<SidePlan> {
        if self.state.ended {
            return Err(operator_error(
                &self.name,
                "received data after end-of-input",
            ));
        }
        let plan = self.side_plan(ingress)?;
        self.input_ports[plan.port_index].validate(batch, &format!("{}.{}", self.name, ingress))?;
        Ok(plan)
    }

    fn side_plan(&self, ingress: &str) -> Result<SidePlan> {
        match ingress {
            "left" => Ok(SidePlan {
                incoming_is_left: true,
                port_index: 0,
                event_time_index: self.compiled.left_event_time_index,
                key_indices: self.compiled.left_key_indices.clone(),
            }),
            "right" => Ok(SidePlan {
                incoming_is_left: false,
                port_index: 1,
                event_time_index: self.compiled.right_event_time_index,
                key_indices: self.compiled.right_key_indices.clone(),
            }),
            _ => Err(operator_error(
                &self.name,
                &format!("unknown ingress {ingress:?}; expected left or right"),
            )),
        }
    }

    fn admission_bundle(&self, plan: &SidePlan) -> AdmissionBundle {
        let (next_row_id, metrics) = if plan.incoming_is_left {
            (self.state.next_left_row_id, self.state.metrics.left.clone())
        } else {
            (
                self.state.next_right_row_id,
                self.state.metrics.right.clone(),
            )
        };
        AdmissionBundle {
            next_row_id,
            metrics,
            admitted: Vec::new(),
            admitted_source_rows: Vec::new(),
            had_late: false,
        }
    }

    async fn admit_record(
        &mut self,
        record: &RecordBatch,
        plan: &SidePlan,
        ingress: &str,
        context: &StreamOperatorContext<'_>,
        bundle: &mut AdmissionBundle,
        source_row_base: usize,
    ) -> Result<()> {
        let mut quantum = columnar::Quantum::default();
        if self
            .can_copy_payload(record, plan, context, &mut quantum)
            .await?
            && let Some(mut selection) = columnar::CopySelection::reserve(self, record.num_rows())?
        {
            self.select_copy_rows(
                (record, plan, ingress),
                context,
                bundle,
                &mut selection,
                &mut quantum,
            )
            .await?;
            let shared = self
                .owned_payload(
                    record,
                    plan.port_index,
                    &selection.rows,
                    context,
                    &mut quantum,
                )
                .await?;
            bundle
                .append_copy_rows(
                    record,
                    shared.as_ref(),
                    &selection.rows,
                    context,
                    source_row_base,
                    &mut quantum,
                )
                .await?;
            return Ok(());
        }
        self.admit_legacy_record(record, plan, ingress, context, bundle, source_row_base)
    }

    async fn select_copy_rows(
        &self,
        source_record: (&RecordBatch, &SidePlan, &str),
        context: &StreamOperatorContext<'_>,
        bundle: &mut AdmissionBundle,
        selection: &mut columnar::CopySelection,
        quantum: &mut columnar::Quantum,
    ) -> Result<()> {
        let (record, plan, ingress) = source_record;
        let times = BatchEventTimes::new(
            record.column(plan.event_time_index).as_ref(),
            &self.name,
            ingress,
        )?;
        let opposite = context.ingress_progress().get(plan.opposite_ingress());
        for source in 0..record.num_rows() {
            quantum
                .step(context, plan.key_indices.len() + 4, 16)
                .await?;
            let row_id = bundle.reserve_row_id(&self.name)?;
            match self.classify_row(
                record,
                plan,
                &times,
                source,
                context.ingress_progress().get(ingress),
                ingress,
            )? {
                RowAdmission::Dropped(kind) => bundle.note_dropped(kind, &self.name)?,
                RowAdmission::Admitted(time) => selection.rows.push(columnar::SelectedRow {
                    source,
                    row_id,
                    time,
                    retain: should_retain(plan.incoming_is_left, time, opposite, self.spec.bounds),
                }),
            }
        }
        Ok(())
    }

    fn admit_legacy_record(
        &self,
        record: &RecordBatch,
        plan: &SidePlan,
        ingress: &str,
        context: &StreamOperatorContext<'_>,
        bundle: &mut AdmissionBundle,
        source_row_base: usize,
    ) -> Result<()> {
        let times = BatchEventTimes::new(
            record.column(plan.event_time_index).as_ref(),
            &self.name,
            ingress,
        )?;
        let opposite = context.ingress_progress().get(if plan.incoming_is_left {
            "right"
        } else {
            "left"
        });
        for row_index in 0..record.num_rows() {
            let row_id = bundle.reserve_row_id(&self.name)?;
            match self.classify_row(
                record,
                plan,
                &times,
                row_index,
                context.ingress_progress().get(ingress),
                ingress,
            )? {
                RowAdmission::Dropped(kind) => bundle.note_dropped(kind, &self.name)?,
                RowAdmission::Admitted(event_time) => {
                    bundle.push_admitted(AdmittedRow {
                        record: columnar::RowPayload::at(record, None, row_index),
                        event_time,
                        row_id,
                        retain: should_retain(
                            plan.incoming_is_left,
                            event_time,
                            opposite,
                            self.spec.bounds,
                        ),
                    });
                    bundle
                        .admitted_source_rows
                        .push(source_row_base + row_index);
                }
            }
        }
        Ok(())
    }

    fn classify_row(
        &self,
        record: &RecordBatch,
        plan: &SidePlan,
        times: &BatchEventTimes<'_>,
        row_index: usize,
        side_progress: Option<IngressProgress>,
        ingress: &str,
    ) -> Result<RowAdmission> {
        let Some(event_time) = times.at(row_index, &self.name, ingress)? else {
            return Ok(RowAdmission::Dropped(DropKind::NullEventTime));
        };
        if plan
            .key_indices
            .iter()
            .any(|&index| record.column(index).is_null(row_index))
        {
            return Ok(RowAdmission::Dropped(DropKind::NullKey));
        }
        match late_lateness(event_time, side_progress, &self.name)? {
            Some(lateness) => Ok(RowAdmission::Dropped(DropKind::Late(lateness))),
            None => Ok(RowAdmission::Admitted(event_time)),
        }
    }

    async fn evaluate_matches(
        &mut self,
        plan: &SidePlan,
        admitted: Vec<AdmittedRow>,
        context: &StreamOperatorContext<'_>,
    ) -> (Result<PreparedMatches>, Vec<AdmittedRow>) {
        let opposite = if plan.incoming_is_left {
            &self.state.right
        } else {
            &self.state.left
        };
        if admitted.is_empty() || opposite.is_empty() {
            return (Ok(PreparedMatches::legacy(Vec::new())), admitted);
        }
        match self.native_matches(plan, &admitted) {
            Ok(Some(native)) => {
                return (
                    Ok(PreparedMatches {
                        pairs: native.pairs,
                        keys: Some(native.keys),
                        credit: Some(native.credit),
                    }),
                    admitted,
                );
            }
            Err(error) => return (Err(error), admitted),
            Ok(None) => {}
        }
        let (matched, admitted) = self.legacy_matches(plan, admitted, context).await;
        (matched.map(PreparedMatches::legacy), admitted)
    }

    async fn legacy_matches(
        &mut self,
        plan: &SidePlan,
        admitted: Vec<AdmittedRow>,
        context: &StreamOperatorContext<'_>,
    ) -> (Result<Vec<MatchedPair>>, Vec<AdmittedRow>) {
        let state_keys = match self.owned_state_keys(plan, context).await {
            Ok(keys) => keys,
            Err(error) => return (Err(error), admitted),
        };
        let (equal_pairs, admitted) = self.sql_key_pairs_owned(plan, admitted, state_keys).await;
        let matched =
            equal_pairs.and_then(|pairs| self.ordered_sql_matches(plan, &admitted, pairs));
        (matched, admitted)
    }

    fn ordered_sql_matches(
        &mut self,
        plan: &SidePlan,
        admitted: &[AdmittedRow],
        equal_pairs: Vec<(u64, u64)>,
    ) -> Result<Vec<MatchedPair>> {
        let opposite = if plan.incoming_is_left {
            self.state.right.as_slice()
        } else {
            self.state.left.as_slice()
        };
        let matched =
            filter_and_order_pairs(&self.spec.bounds, plan, admitted, opposite, equal_pairs);
        enforce_match_limit(
            matched.len(),
            &mut self.state.metrics.match_limit_failures,
            self.spec.limits.max_matches_per_input_batch,
            &self.name,
        )?;
        Ok(matched)
    }

    async fn sql_key_pairs(
        &mut self,
        plan: &SidePlan,
        admitted: &[AdmittedRow],
        state_keys: RecordBatch,
    ) -> Result<Vec<(u64, u64)>> {
        let probe = probe_key_batch(
            admitted,
            &plan.key_indices,
            Some(self.input_schema(plan.port_index)),
        )?;
        let tables = equality_tables(probe, state_keys)?;
        let result = self
            .runtime
            .runtime()?
            .sql_validated(&self.compiled.equality_query, &tables, Some(&self.name))
            .await?;
        decode_key_pairs(&result)
    }

    async fn sql_key_pairs_owned(
        &mut self,
        plan: &SidePlan,
        admitted: Vec<AdmittedRow>,
        state_keys: sql_key_scratch::KeyBatch,
    ) -> (Result<Vec<(u64, u64)>>, Vec<AdmittedRow>) {
        let runtime = match self.runtime.runtime() {
            Ok(runtime) => runtime,
            Err(error) => return (Err(error), admitted),
        };
        if !runtime.serial_owned_sql() {
            drop(state_keys);
            return self.legacy_key_retry(plan, admitted).await;
        }
        let retained = if plan.incoming_is_left {
            &self.state.right.0
        } else {
            &self.state.left.0
        };
        let owner = SqlKeyOwners {
            _retained: Arc::clone(retained),
            admitted,
            scratch: state_keys.funding,
        };
        let tables = probe_key_batch(&owner.admitted, &plan.key_indices, None)
            .and_then(|probe| equality_tables(probe, state_keys.batch));
        let tables = match tables {
            Ok(tables) => tables,
            Err(error) => return (Err(error), owner.finish()),
        };
        self.run_owned_key_query(plan, crate::datafusion::owned::Input::new(tables, owner))
            .await
    }

    async fn run_owned_key_query(
        &mut self,
        plan: &SidePlan,
        input: crate::datafusion::owned::Input<SqlKeyOwners>,
    ) -> (Result<Vec<(u64, u64)>>, Vec<AdmittedRow>) {
        let result = self
            .runtime
            .runtime()
            .expect("initialized by scratch construction")
            .sql_equality_owned(&self.compiled.equality_query, input, Some(&self.name))
            .await;
        match result {
            Ok(result) => {
                let pairs = decode_key_pairs(result.batch());
                (pairs, result.finish().finish())
            }
            Err(failure) => self.finish_owned_key_failure(plan, failure).await,
        }
    }

    async fn finish_owned_key_failure(
        &mut self,
        plan: &SidePlan,
        failure: crate::datafusion::owned::Failure<SqlKeyOwners>,
    ) -> (Result<Vec<(u64, u64)>>, Vec<AdmittedRow>) {
        let retry = failure
            .error
            .retry_legacy(failure.input.owner().scratch.is_some());
        let (error, input) = failure.into_parts(Some(&self.name));
        let admitted = input.finish().finish();
        if retry {
            return self.legacy_key_retry(plan, admitted).await;
        }
        (
            Err(error.expect("non-retry failure keeps its source")),
            admitted,
        )
    }

    fn discard_paid_key_cache(&mut self, plan: &SidePlan) {
        let cached = if plan.incoming_is_left {
            &mut self.retained_key_cache.right
        } else {
            &mut self.retained_key_cache.left
        };
        if cached
            .as_ref()
            .is_some_and(|entry| entry.scratch_funding.is_some())
        {
            *cached = None;
        }
    }

    async fn legacy_key_retry(
        &mut self,
        plan: &SidePlan,
        admitted: Vec<AdmittedRow>,
    ) -> (Result<Vec<(u64, u64)>>, Vec<AdmittedRow>) {
        self.discard_paid_key_cache(plan);
        let result = match self.opposite_state_keys(plan) {
            Ok(keys) => self.sql_key_pairs(plan, &admitted, keys).await,
            Err(error) => Err(error),
        };
        (result, admitted)
    }

    fn opposite_state_keys(&mut self, plan: &SidePlan) -> Result<RecordBatch> {
        let declared = Arc::clone(self.input_schema(1 - plan.port_index));
        let (opposite, cached) = if plan.incoming_is_left {
            (&self.state.right, &mut self.retained_key_cache.right)
        } else {
            (&self.state.left, &mut self.retained_key_cache.left)
        };
        if let Some(existing) = cached.as_ref()
            && existing.row_ids.len() == opposite.len()
            && existing
                .row_ids
                .iter()
                .zip(opposite.iter())
                .all(|(row_id, row)| *row_id == row.row_id)
        {
            return Ok(existing.batch.clone());
        }
        let batch = state_key_batch(opposite, &self.compiled, plan, Some(&declared))?;
        let bytes = batch
            .columns()
            .iter()
            .try_fold(0_usize, |total, array| {
                total.checked_add(array.get_array_memory_size())
            })
            .and_then(|total| total.checked_add(opposite.len().checked_mul(size_of::<u64>())?));
        *cached = bytes
            .filter(|&bytes| bytes <= MAX_RETAINED_KEY_CACHE_BYTES_PER_SIDE)
            .map(|_| CachedRetainedKeys {
                row_ids: opposite.iter().map(|row| row.row_id).collect(),
                batch: batch.clone(),
                scratch_funding: None,
            });
        Ok(batch)
    }

    async fn owned_state_keys(
        &mut self,
        plan: &SidePlan,
        context: &StreamOperatorContext<'_>,
    ) -> Result<sql_key_scratch::KeyBatch> {
        let (rows, indices, cached) = if plan.incoming_is_left {
            (
                &self.state.right,
                &self.compiled.right_key_indices,
                &mut self.retained_key_cache.right,
            )
        } else {
            (
                &self.state.left,
                &self.compiled.left_key_indices,
                &mut self.retained_key_cache.left,
            )
        };
        if let Some(existing) = cached.as_ref()
            && existing.scratch_funding.is_some()
            && existing
                .row_ids
                .iter()
                .copied()
                .eq(rows.iter().map(|row| row.row_id))
        {
            return Ok(sql_key_scratch::KeyBatch {
                batch: existing.batch.clone(),
                funding: existing.scratch_funding.clone(),
            });
        }
        let keys =
            sql_key_scratch::state_keys(self.runtime.runtime()?, rows, indices, context).await?;
        if let Some(keys) = keys {
            let batch = keys.batch_owner();
            *cached = keys.into_cache();
            return Ok(batch);
        }
        self.discard_paid_key_cache(plan);
        self.opposite_state_keys(plan)
            .map(|batch| sql_key_scratch::KeyBatch {
                batch,
                funding: None,
            })
    }

    fn validate_state_admission(
        &mut self,
        incoming_is_left: bool,
        retained: &[StoredRow],
    ) -> Result<()> {
        let current = if incoming_is_left {
            &self.state.metrics.left
        } else {
            &self.state.metrics.right
        };
        let (rows, bytes) = prospective_state_charge(current, retained, &self.name)?;
        if !super::StateBudget::new(
            self.spec.limits.max_state_rows_per_side,
            self.spec.limits.max_state_bytes_per_side,
        )?
        .allows(rows, bytes)
        {
            self.state.metrics.state_limit_failures = checked_metric(
                self.state.metrics.state_limit_failures,
                1,
                &self.name,
                "state_limit_failures",
            )?;
            return Err(operator_reason(
                &self.name,
                crate::StreamingFailureReason::JoinStateLimitExceeded,
                "retained state limit exceeded",
            ));
        }
        Ok(())
    }

    fn validate_restored_limits(&self, left: &[StoredRow], right: &[StoredRow]) -> Result<()> {
        for (side, rows) in [("left", left), ("right", right)] {
            let row_count =
                u64::try_from(rows.len()).map_err(|_| CalcFlowError::CheckpointMismatch {
                    message: format!("stream Join {:?} {side} row count is too large", self.name),
                })?;
            let byte_count = rows
                .iter()
                .try_fold(0_u64, |total, row| total.checked_add(row.charge))
                .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                    message: format!("stream Join {:?} {side} byte charge overflowed", self.name),
                })?;
            if !super::StateBudget::new(
                self.spec.limits.max_state_rows_per_side,
                self.spec.limits.max_state_bytes_per_side,
            )?
            .allows(row_count, byte_count)
            {
                return Err(CalcFlowError::CheckpointMismatch {
                    message: format!(
                        "stream Join {:?} restored {side} state exceeds configured limits",
                        self.name
                    ),
                });
            }
        }
        Ok(())
    }

    /// Emits the prepared output as sequential chunk messages.
    ///
    /// If `output.emit` fails after k of n chunks, those k chunks have escaped
    /// while `commit_prepared` never runs. The failure aborts the operator task
    /// and job convergence discards the run's state, so no resumption observes
    /// the emitted-but-uncommitted gap.
    async fn emit_prepared(
        &mut self,
        prepared: &PreparedJoinBatch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if prepared.output.is_empty() {
            return Ok(());
        }
        let materializer = materialization::JoinOutput {
            schema: self.output_ports[0]
                .schema()
                .expect("compiled Join output schema"),
            admitted: &prepared.admitted,
            opposite: if prepared.incoming_is_left {
                &self.state.right
            } else {
                &self.state.left
            },
            matched: &prepared.output,
            incoming_is_left: prepared.incoming_is_left,
            operator_id: &self.name,
            admitted_charges: prepared.admitted_charges.as_deref(),
        };
        let ranges = materializer.ranges(context.output_budget())?;
        super::output_chunk::validate_output_sequence_range(
            &self.name,
            self.state.next_output_sequence,
            ranges.len(),
        )?;
        for range in ranges {
            context.check_cancelled()?;
            let record = materializer.materialize(range)?;
            let metadata =
                BatchMetadata::new(&self.name, self.state.next_output_sequence, BTreeMap::new())?;
            let message = Batch::table(vec![record], metadata)?;
            if message.estimated_bytes()? > context.output_budget().max_bytes {
                return Err(operator_error(
                    &self.name,
                    "validated Join chunk exceeded its byte budget",
                ));
            }
            output.emit("output", message).await?;
            self.state.next_output_sequence += 1;
        }
        Ok(())
    }

    fn record_prepared_emitted(&mut self, rows: usize) -> Result<()> {
        let emitted =
            u64::try_from(rows).map_err(|_| counter_overflow(&self.name, "emitted rows"))?;
        self.state.metrics.emitted_match_rows = checked_metric(
            self.state.metrics.emitted_match_rows,
            emitted,
            &self.name,
            "emitted_match_rows",
        )?;
        Ok(())
    }

    fn commit_prepared(&mut self, ingress: &str, prepared: PreparedJoinBatch) -> Result<()> {
        let mut metrics = prepared.metrics;
        (metrics.retained_rows, metrics.retained_bytes) =
            prospective_state_charge(&metrics, &prepared.retained, &self.name)?;
        let side = if ingress == "left" {
            JoinSide::Left
        } else {
            JoinSide::Right
        };
        for row in &prepared.retained {
            row.record.mark_live();
            self.state.deltas.pending.push(PendingOp::Upsert {
                side,
                row_id: row.row_id,
                event_time: row.event_time,
                encoded_key: Arc::clone(&row.encoded_key),
                record: row.record.clone(),
                charge: row.charge,
            });
        }
        if side == JoinSide::Left {
            self.state.next_left_row_id = prepared.next_row_id;
            if !prepared.retained.is_empty() {
                self.retained_key_cache.left = None;
            }
            self.state
                .left_expirations
                .append(self.state.left.len(), &prepared.retained);
            self.state.left.extend(prepared.retained);
            self.state.metrics.left = metrics;
        } else {
            self.state.next_right_row_id = prepared.next_row_id;
            if !prepared.retained.is_empty() {
                self.retained_key_cache.right = None;
            }
            self.state
                .right_expirations
                .append(self.state.right.len(), &prepared.retained);
            self.state.right.extend(prepared.retained);
            self.state.metrics.right = metrics;
        }
        if let Some(credit) = prepared.native_append {
            credit.commit();
        }
        Ok(())
    }

    fn evict_progress(&mut self, ingress: &str, progress: IngressProgress) -> Result<()> {
        match ingress {
            "left" => {
                let before = self.state.right.len();
                evict_opposite(
                    &mut self.state.right,
                    &mut self.state.right_expirations,
                    progress,
                    &mut self.state.metrics.right,
                    &mut self.state.deltas.pending,
                    EvictionPolicy {
                        extension_micros: self.spec.bounds.before_micros,
                        side: JoinSide::Right,
                        operator_id: &self.name,
                    },
                )?;
                if self.state.right.len() != before {
                    self.retained_key_cache.right = None;
                }
            }
            "right" => {
                let before = self.state.left.len();
                evict_opposite(
                    &mut self.state.left,
                    &mut self.state.left_expirations,
                    progress,
                    &mut self.state.metrics.left,
                    &mut self.state.deltas.pending,
                    EvictionPolicy {
                        extension_micros: self.spec.bounds.after_micros,
                        side: JoinSide::Left,
                        operator_id: &self.name,
                    },
                )?;
                if self.state.left.len() != before {
                    self.retained_key_cache.left = None;
                }
            }
            _ => {
                return Err(operator_error(
                    &self.name,
                    &format!("unknown ingress progress {ingress:?}"),
                ));
            }
        }
        Ok(())
    }

    fn required_progress(
        &self,
        ingress: &str,
        context: &StreamOperatorContext<'_>,
    ) -> Result<IngressProgress> {
        context.ingress_progress().get(ingress).ok_or_else(|| {
            operator_error(
                &self.name,
                &format!("missing progress for ingress {ingress:?}"),
            )
        })
    }

    fn decode_restored_sides(
        &self,
        snapshot: &OperatorStateSnapshot,
        schemas: Option<&metadata_validation::schema::OwnedExpectedSchemas>,
    ) -> Result<(Vec<StoredRow>, Vec<StoredRow>)> {
        restore_sides_from_segments(
            snapshot,
            self.restore_schema(schemas, 0),
            self.restore_schema(schemas, 1),
            &self.compiled.left_key_indices,
            &self.compiled.right_key_indices,
            &self.name,
        )
    }

    fn restore_schema<'a>(
        &'a self,
        schemas: Option<&'a metadata_validation::schema::OwnedExpectedSchemas>,
        side: usize,
    ) -> RestoreSchema<'a> {
        RestoreSchema {
            schema: schemas.map_or_else(
                || self.input_schema(side).as_ref(),
                |schemas| schemas.schema(side),
            ),
            #[cfg(test)]
            hook: self.schema_test_hook.as_ref(),
            #[cfg(test)]
            credit: schemas.map(metadata_validation::schema::OwnedExpectedSchemas::credit),
        }
    }

    fn input_schema(&self, port_index: usize) -> &SchemaRef {
        self.input_ports[port_index]
            .schema()
            .expect("stream Join inputs always have an exact schema")
    }

    fn validate_restored_join_rows(
        &self,
        metadata: &metadata_validation::ValidatedMetadata,
        left: &[StoredRow],
        right: &[StoredRow],
    ) -> Result<()> {
        validate_restored_rows(
            left,
            metadata.next_left_row_id,
            self.compiled.left_event_time_index,
            &self.compiled.left_key_indices,
            &self.name,
            "left",
        )?;
        validate_restored_rows(
            right,
            metadata.next_right_row_id,
            self.compiled.right_event_time_index,
            &self.compiled.right_key_indices,
            &self.name,
            "right",
        )
    }
}

impl fmt::Debug for StreamJoinOperator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamJoinOperator")
            .field("name", &self.name)
            .field("spec", &self.spec)
            .field("input_ports", &self.input_ports)
            .field("output_ports", &self.output_ports)
            .finish_non_exhaustive()
    }
}

impl OperatorMetadata for StreamJoinOperator {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_ports(&self) -> &[Port] {
        &self.input_ports
    }

    fn output_ports(&self) -> &[Port] {
        &self.output_ports
    }

    fn configuration(&self) -> JsonMap {
        let value = serde_json::to_value(&self.spec)
            .expect("validated stream Join configuration remains serializable");
        let Value::Object(values) = value else {
            unreachable!("stream Join configuration serializes as an object")
        };
        values.into_iter().collect()
    }
}

#[async_trait]
impl StreamOperator for StreamJoinOperator {
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()?;
        self.await_compaction_release(context).await?;
        let prepared = self.prepare_batch(ingress, &batch, context).await?;
        self.emit_prepared(&prepared, context, output).await?;
        self.record_prepared_emitted(prepared.output.len())?;
        self.commit_prepared(ingress, prepared)?;
        if self.has_sparse_candidates() {
            return self.repair_sparse_chunks(context).await;
        }
        self.ingress_progress = context.ingress_progress().clone();
        Ok(())
    }

    async fn on_ingress_progress(
        &mut self,
        ingress: &str,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let progress = self.required_progress(ingress, context)?;
        if self.compaction_release.is_none() && self.compaction_cleanup.is_none() {
            context.check_cancelled()?;
        } else {
            self.await_compaction_release(context).await?;
        }
        self.evict_progress(ingress, progress)?;
        if self.has_sparse_candidates() {
            return self.repair_sparse_chunks(context).await;
        }
        self.ingress_progress = context.ingress_progress().clone();
        Ok(())
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.await_compaction_release(context).await?;
        let left_identities = self.state.left_expirations.identities(&self.state.left);
        record_tombstones(
            &mut self.state.deltas.pending,
            JoinSide::Left,
            left_identities,
        );
        let right_identities = self.state.right_expirations.identities(&self.state.right);
        record_tombstones(
            &mut self.state.deltas.pending,
            JoinSide::Right,
            right_identities,
        );
        self.state.left.clear();
        self.state.right.clear();
        self.state.left_expirations = ExpirationIndex::default();
        self.state.right_expirations = ExpirationIndex::default();
        self.retained_key_cache = RetainedKeyCache::default();
        self.state.metrics.left.retained_rows = 0;
        self.state.metrics.left.retained_bytes = 0;
        self.state.metrics.right.retained_rows = 0;
        self.state.metrics.right.retained_bytes = 0;
        self.state.ended = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.state = StreamJoinState::default();
        self.retained_key_cache = RetainedKeyCache::default();
        self.ingress_progress = IngressProgressSnapshot::default();
        Ok(())
    }

    async fn prepare_checkpoint_async(
        &mut self,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        self.prepare_compaction(context).await
    }

    fn checkpoint(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot> {
        if self
            .state
            .last_checkpoint_epoch
            .is_some_and(|previous| epoch <= previous)
        {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "stream Join {:?} checkpoint epoch did not advance strictly",
                    self.name
                ),
            });
        }
        let metadata = JoinCheckpointMetadata {
            layout_version: 1,
            spec: self.spec.clone(),
            next_left_row_id: self.state.next_left_row_id,
            next_right_row_id: self.state.next_right_row_id,
            next_output_sequence: self.state.next_output_sequence,
            metrics: self.state.metrics.clone(),
            ended: self.state.ended,
            epoch: epoch.as_u64(),
        };
        let Value::Object(inline_metadata) =
            serde_json::to_value(metadata).map_err(|error| CalcFlowError::Internal {
                message: format!("stream Join checkpoint metadata encoding failed: {error}"),
            })?
        else {
            unreachable!("stream Join checkpoint metadata is an object")
        };
        // O(dirty/segment metadata) capture (spec FR47): prepared base and
        // carried delta segments share their allocations without copying, and
        // only the dirty ops since the last epoch encode here.
        let mut segments = BTreeMap::new();
        for (side, segment) in &self.state.deltas.base {
            segments.insert(format!("{side}-base"), segment.clone());
        }
        for ((segment_epoch, side), segment) in &self.state.deltas.segments {
            segments.insert(format!("{side}-delta-{segment_epoch}"), segment.clone());
        }
        if !self.state.deltas.pending.is_empty() {
            for (side, bytes) in encode_pending_delta(&self.state, epoch, &self.name)? {
                let segment = StateSegment::new(bytes);
                self.state
                    .deltas
                    .segments
                    .insert((epoch.as_u64(), side.as_str()), segment.clone());
                segments.insert(
                    format!("{}-delta-{}", side.as_str(), epoch.as_u64()),
                    segment,
                );
            }
            self.state.deltas.pending.clear();
            self.state.deltas.segments_since_base += 1;
            if self.state.deltas.segments_since_base >= JOIN_DELTA_COMPACTION_SEGMENTS {
                self.state.deltas.needs_compaction = true;
            }
        }
        self.state.last_checkpoint_epoch = Some(epoch);
        Ok(OperatorStateSnapshot {
            inline_metadata: inline_metadata.into_iter().collect(),
            segments,
        })
    }

    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        let metadata = self.parse_restore_metadata(snapshot)?;
        self.install_restored_metadata(snapshot, metadata, &|| Ok(()))
    }
}

impl StreamJoinOperator {
    fn parse_restore_metadata(
        &self,
        snapshot: &OperatorStateSnapshot,
    ) -> Result<metadata_validation::ValidatedMetadata> {
        #[cfg(test)]
        if let Some(hook) = &self.metadata_test_hook {
            hook(None, false);
        }
        let metadata = decode_join_metadata(snapshot, &self.name)?;
        #[cfg(test)]
        if let Some(hook) = &self.metadata_test_hook {
            hook(None, true);
        }
        if !checkpoint_metadata_compatible(&metadata, &self.spec) {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "stream Join {:?} checkpoint layout or specification is incompatible",
                    self.name
                ),
            });
        }
        Ok(metadata_validation::ValidatedMetadata::from(metadata))
    }

    fn install_restored_metadata(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        metadata: metadata_validation::ValidatedMetadata,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        self.install_restored_metadata_with_schemas(snapshot, metadata, None, check)
    }

    fn install_restored_metadata_with_schemas(
        &mut self,
        snapshot: &OperatorStateSnapshot,
        metadata: metadata_validation::ValidatedMetadata,
        schemas: Option<&metadata_validation::schema::OwnedExpectedSchemas>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let (left, right) = self.decode_restored_sides(snapshot, schemas)?;
        self.validate_restored_join_rows(&metadata, &left, &right)?;
        restored_retained_metrics_match(&metadata.metrics, &left, &right, &self.name)?;
        self.validate_restored_limits(&left, &right)?;
        let carried = carried_delta_segments(snapshot, &self.name)?;
        let base = carried_base_segments(snapshot);
        let segments_since_base =
            u32::try_from(carried.len()).map_err(|_| counter_overflow(&self.name, "segments"))?;
        let state = StreamJoinState {
            left_expirations: ExpirationIndex::restored(&left),
            right_expirations: ExpirationIndex::restored(&right),
            left: left.into(),
            right: right.into(),
            next_left_row_id: metadata.next_left_row_id,
            next_right_row_id: metadata.next_right_row_id,
            next_output_sequence: metadata.next_output_sequence,
            metrics: metadata.metrics,
            ended: metadata.ended,
            last_checkpoint_epoch: Epoch::new(metadata.epoch),
            deltas: DeltaTracking {
                base,
                segments: carried,
                segments_since_base,
                ..DeltaTracking::default()
            },
        };
        check()?;
        self.state = state;
        self.retained_key_cache = RetainedKeyCache::default();
        self.ingress_progress = IngressProgressSnapshot::default();
        Ok(())
    }
}

/// Compile-time per-ingress lookup for one input batch.
struct SidePlan {
    incoming_is_left: bool,
    port_index: usize,
    event_time_index: usize,
    key_indices: Vec<usize>,
}

impl SidePlan {
    fn opposite_ingress(&self) -> &'static str {
        if self.incoming_is_left {
            "right"
        } else {
            "left"
        }
    }
}

impl AdmissionBundle {
    async fn append_copy_rows(
        &mut self,
        record: &RecordBatch,
        shared: Option<&Arc<columnar::PayloadChunk>>,
        rows: &[columnar::SelectedRow],
        context: &StreamOperatorContext<'_>,
        source_row_base: usize,
        quantum: &mut columnar::Quantum,
    ) -> Result<()> {
        for (offset, row) in rows.iter().enumerate() {
            quantum.step(context, 1, 0).await?;
            self.push_admitted(AdmittedRow {
                record: columnar::RowPayload::at(
                    record,
                    shared,
                    if shared.is_some() { offset } else { row.source },
                ),
                event_time: row.time,
                row_id: row.row_id,
                retain: row.retain,
            });
            self.admitted_source_rows.push(source_row_base + row.source);
        }
        Ok(())
    }

    fn reserve_row_id(&mut self, operator_id: &str) -> Result<u64> {
        let row_id = self.next_row_id;
        self.next_row_id = self
            .next_row_id
            .checked_add(1)
            .ok_or_else(|| counter_overflow(operator_id, "row_id"))?;
        Ok(row_id)
    }

    fn note_dropped(&mut self, kind: DropKind, operator_id: &str) -> Result<()> {
        match kind {
            DropKind::NullEventTime => {
                self.metrics.null_event_time_rows = checked_metric(
                    self.metrics.null_event_time_rows,
                    1,
                    operator_id,
                    "null_event_time_rows",
                )?;
            }
            DropKind::NullKey => {
                self.metrics.null_key_rows =
                    checked_metric(self.metrics.null_key_rows, 1, operator_id, "null_key_rows")?;
            }
            DropKind::Late(lateness) => {
                self.metrics.late_rows =
                    checked_metric(self.metrics.late_rows, 1, operator_id, "late_rows")?;
                self.metrics.max_lateness_micros = Some(
                    self.metrics
                        .max_lateness_micros
                        .map_or(lateness, |current| current.max(lateness)),
                );
                self.had_late = true;
            }
        }
        Ok(())
    }

    fn push_admitted(&mut self, row: AdmittedRow) {
        self.admitted.push(row);
    }

    fn finish(&mut self, operator_id: &str) -> Result<()> {
        if self.had_late {
            self.metrics.late_affected_batches = checked_metric(
                self.metrics.late_affected_batches,
                1,
                operator_id,
                "late_affected_batches",
            )?;
        }
        Ok(())
    }
}

fn checkpoint_metadata_compatible(
    metadata: &JoinCheckpointMetadata,
    spec: &StreamJoinSpec,
) -> bool {
    metadata.layout_version == 1 && metadata.spec == *spec
}

fn decode_join_metadata(
    snapshot: &OperatorStateSnapshot,
    operator_id: &str,
) -> Result<JoinCheckpointMetadata> {
    serde_json::from_value::<JoinCheckpointMetadata>(Value::Object(
        snapshot.inline_metadata.clone().into_iter().collect(),
    ))
    .map_err(|error| CalcFlowError::CheckpointMismatch {
        message: format!("stream Join {operator_id:?} metadata is invalid: {error}"),
    })
}

/// Recomputes retained charges and rejects checkpoints that disagree with them.
fn restored_retained_metrics_match(
    metrics: &JoinMetrics,
    left: &[StoredRow],
    right: &[StoredRow],
    operator_id: &str,
) -> Result<()> {
    let mut left_metrics = metrics.left.clone();
    let mut right_metrics = metrics.right.clone();
    refresh_retained_metrics(&mut left_metrics, left, operator_id)?;
    refresh_retained_metrics(&mut right_metrics, right, operator_id)?;
    if side_retained_matches(&metrics.left, &left_metrics)
        && side_retained_matches(&metrics.right, &right_metrics)
    {
        return Ok(());
    }
    Err(CalcFlowError::CheckpointMismatch {
        message: format!("stream Join {operator_id:?} restored state charge is inconsistent"),
    })
}

fn side_retained_matches(recorded: &SideMetrics, recomputed: &SideMetrics) -> bool {
    recorded.retained_rows == recomputed.retained_rows
        && recorded.retained_bytes == recomputed.retained_bytes
}

/// Prospective (rows, bytes) charge if `retained` were installed next to `current`.
fn prospective_state_charge(
    current: &SideMetrics,
    retained: &[StoredRow],
    operator_id: &str,
) -> Result<(u64, u64)> {
    let rows = current
        .retained_rows
        .checked_add(state_row_count(retained, operator_id)?)
        .ok_or_else(|| counter_overflow(operator_id, "state rows"))?;
    let bytes = retained
        .iter()
        .try_fold(current.retained_bytes, |total, row| {
            total.checked_add(row.charge)
        })
        .ok_or_else(|| counter_overflow(operator_id, "state bytes"))?;
    Ok((rows, bytes))
}

fn state_row_count(rows: &[StoredRow], operator_id: &str) -> Result<u64> {
    u64::try_from(rows.len()).map_err(|_| counter_overflow(operator_id, "state rows"))
}

fn late_lateness(
    event_time: EventTime,
    progress: Option<IngressProgress>,
    operator_id: &str,
) -> Result<Option<u64>> {
    let Some(watermark) = progress.and_then(IngressProgress::watermark) else {
        return Ok(None);
    };
    if event_time >= watermark {
        return Ok(None);
    }
    let lateness =
        u64::try_from(i128::from(watermark.as_micros()) - i128::from(event_time.as_micros()))
            .map_err(|_| counter_overflow(operator_id, "lateness"))?;
    Ok(Some(lateness))
}

fn retained_rows(
    admitted: &[AdmittedRow],
    key_indices: &[usize],
    operator_id: &str,
    native_keys: Option<&[Arc<columnar::FramedKey>]>,
) -> Result<Vec<StoredRow>> {
    admitted
        .iter()
        .enumerate()
        .filter(|(_, row)| row.retain)
        .map(|(index, row)| {
            let encoded_key = match native_keys {
                Some(keys) => Arc::clone(&keys[index]),
                None => Arc::new(
                    encode_join_key_columns_v1(
                        row.record.columns(),
                        row.record.offset(),
                        key_indices,
                    )?
                    .into(),
                ),
            };
            let charge = state_columns_charge_with_key(
                row.record.columns(),
                row.record.offset(),
                encoded_key.len(),
                operator_id,
            )?;
            Ok(StoredRow {
                encoded_key,
                record: row.record.clone(),
                event_time: row.event_time,
                row_id: row.row_id,
                charge,
            })
        })
        .collect()
}

fn admitted_charge_cache(charges: Option<Vec<u64>>, rows: &[usize]) -> Option<Vec<u64>> {
    charges.map(|charges| rows.iter().map(|row| charges[*row]).collect())
}

fn append_admission_charges(charges: &mut Option<Vec<u64>>, record: &RecordBatch) -> Result<()> {
    let current =
        materialization::flat_row_charges(record, STREAM_JOIN_STATE_ROW_OVERHEAD_BYTES_V1)?;
    match (current, charges.as_mut()) {
        (Some(current), Some(charges)) => charges.extend(current),
        _ => *charges = None,
    }
    Ok(())
}

fn enforce_match_limit(
    count: usize,
    failures: &mut u64,
    limit: u64,
    operator_id: &str,
) -> Result<()> {
    let count = u64::try_from(count).map_err(|_| counter_overflow(operator_id, "match_count"))?;
    if count > limit {
        *failures = checked_metric(*failures, 1, operator_id, "match_limit_failures")?;
        return Err(operator_reason(
            operator_id,
            crate::StreamingFailureReason::JoinMatchLimitExceeded,
            "input batch match limit exceeded",
        ));
    }
    Ok(())
}

/// Builds the admitted-row scratch table with renamed key columns.
fn probe_key_batch(
    admitted: &[AdmittedRow],
    key_indices: &[usize],
    declared: Option<&Schema>,
) -> Result<RecordBatch> {
    let records = admitted
        .iter()
        .map(|row| row.record.view())
        .collect::<Vec<_>>();
    let positions = UInt64Array::from_iter_values(
        (0..admitted.len()).map(|index| u64::try_from(index).expect("row count fits u64")),
    );
    key_probe_batch(
        &records,
        key_indices,
        PROBE_POS_COLUMN,
        &positions,
        declared,
    )
}

/// Builds the retained-state scratch table with renamed key columns and row ids.
fn state_key_batch(
    opposite: &[StoredRow],
    compiled: &CompiledJoin,
    plan: &SidePlan,
    declared: Option<&Schema>,
) -> Result<RecordBatch> {
    let key_indices = if plan.incoming_is_left {
        &compiled.right_key_indices
    } else {
        &compiled.left_key_indices
    };
    let records = opposite
        .iter()
        .map(|row| row.record.view())
        .collect::<Vec<_>>();
    let row_ids = UInt64Array::from_iter_values(opposite.iter().map(|row| row.row_id));
    key_probe_batch(&records, key_indices, STATE_RID_COLUMN, &row_ids, declared)
}

fn key_probe_batch(
    records: &[columnar::RowView<'_>],
    key_indices: &[usize],
    extra_name: &str,
    extra: &UInt64Array,
    declared: Option<&Schema>,
) -> Result<RecordBatch> {
    let first = records
        .first()
        .expect("join probe batches always have at least one row");
    let source_schema = first.schema();
    let source_schema = declared.unwrap_or(&source_schema);
    let mut fields = Vec::with_capacity(key_indices.len() + 1);
    let mut columns = Vec::with_capacity(key_indices.len() + 1);
    for (position, &key_index) in key_indices.iter().enumerate() {
        let source = source_schema.field(key_index);
        fields.push(Field::new(
            format!("{KEY_COLUMN_PREFIX}{position}"),
            source.data_type().clone(),
            source.is_nullable(),
        ));
        let slices = records
            .iter()
            .map(|record| record.column(key_index).as_ref())
            .collect::<Vec<_>>();
        columns.push(canonical_column(
            concat_column(&slices)?,
            source.data_type(),
        ));
    }
    fields.push(Field::new(extra_name, DataType::UInt64, false));
    columns.push(Arc::new(extra.clone()));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(|error| {
        CalcFlowError::Internal {
            message: format!("stream Join equality probe assembly failed: {error}"),
        }
    })
}

fn concat_column(slices: &[&dyn Array]) -> Result<ArrayRef> {
    concat(slices).map_err(|error| CalcFlowError::Internal {
        message: format!("stream Join column concatenation failed: {error}"),
    })
}

fn equality_tables(probe: RecordBatch, state_keys: RecordBatch) -> Result<BTreeMap<String, Batch>> {
    #[cfg(test)]
    note_join_work(|work| work.sql_probe_table_builds += 1);
    Ok(BTreeMap::from([
        (
            PROBE_TABLE.into(),
            Batch::table(vec![probe], BatchMetadata::default())?,
        ),
        (
            STATE_TABLE.into(),
            Batch::table(vec![state_keys], BatchMetadata::default())?,
        ),
    ]))
}

fn decode_key_pairs(result: &Batch) -> Result<Vec<(u64, u64)>> {
    let mut pairs = Vec::new();
    for record in result.table_payload()?.batches() {
        let positions = u64_column(record, 0, "probe position")?;
        let row_ids = u64_column(record, 1, "state row id")?;
        for row_index in 0..record.num_rows() {
            pairs.push((positions.value(row_index), row_ids.value(row_index)));
        }
    }
    Ok(pairs)
}

fn u64_column<'a>(
    record: &'a RecordBatch,
    column_index: usize,
    field: &str,
) -> Result<&'a UInt64Array> {
    record
        .column(column_index)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| CalcFlowError::Internal {
            message: format!("stream Join equality result is missing the {field} column"),
        })
}

fn filter_and_order_pairs(
    bounds: &JoinTimeBounds,
    plan: &SidePlan,
    admitted: &[AdmittedRow],
    opposite: &[StoredRow],
    equal_pairs: Vec<(u64, u64)>,
) -> Vec<MatchedPair> {
    let row_id_index = index_by_row_id(opposite);
    let mut matched = equal_pairs
        .into_iter()
        .filter_map(|(pos, rid)| {
            let pos = usize::try_from(pos).expect("probe positions index admitted rows");
            let opposite_index = *row_id_index
                .get(&rid)
                .expect("state row ids index retained rows");
            let incoming = &admitted[pos];
            let candidate = &opposite[opposite_index];
            let in_bounds = if plan.incoming_is_left {
                bounds.contains_pair(
                    incoming.event_time.as_micros(),
                    candidate.event_time.as_micros(),
                )
            } else {
                bounds.contains_pair(
                    candidate.event_time.as_micros(),
                    incoming.event_time.as_micros(),
                )
            };
            in_bounds.then_some(MatchedPair {
                pos,
                opposite_index,
            })
        })
        .collect::<Vec<_>>();
    matched.sort_by_key(|pair| {
        let row = &opposite[pair.opposite_index];
        (pair.pos, row.event_time, row.row_id)
    });
    matched
}

fn index_by_row_id(opposite: &[StoredRow]) -> BTreeMap<u64, usize> {
    opposite
        .iter()
        .enumerate()
        .map(|(index, row)| (row.row_id, index))
        .collect()
}

/// Materializes one validated chunk of matched pairs into an independent record.
///
/// Each output column concatenates the per-pair single-row column slices in
/// matched-pair order, so row order is exactly the emission order the
/// per-row path produced.
fn materialize_output_record(
    output_schema: &SchemaRef,
    admitted: &[AdmittedRow],
    opposite: &[StoredRow],
    matched: &[MatchedPair],
    incoming_is_left: bool,
    operator_id: &str,
) -> Result<RecordBatch> {
    if matched.is_empty() {
        return Ok(RecordBatch::new_empty(Arc::clone(output_schema)));
    }
    let pair_records = |pair: &MatchedPair| {
        let incoming = &admitted[pair.pos];
        let candidate = &opposite[pair.opposite_index];
        if incoming_is_left {
            (&incoming.record, &candidate.record)
        } else {
            (&candidate.record, &incoming.record)
        }
    };
    let (first_left, first_right) = pair_records(&matched[0]);
    let left_width = first_left.num_columns();
    let right_width = first_right.num_columns();
    let mut columns = Vec::with_capacity(left_width + right_width);
    for column_index in 0..left_width + right_width {
        let slices = matched
            .iter()
            .map(|pair| {
                let (left, right) = pair_records(pair);
                if column_index < left_width {
                    left.column_view(column_index)
                } else {
                    right.column_view(column_index - left_width)
                }
            })
            .collect::<Vec<_>>();
        let references = slices.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        columns.push(canonical_column(
            concat_output_column(&references)?,
            output_schema.field(column_index).data_type(),
        ));
    }
    RecordBatch::try_new(Arc::clone(output_schema), columns)
        .map_err(|error| operator_error(operator_id, &format!("output projection failed: {error}")))
}

fn concat_output_column(slices: &[&dyn Array]) -> Result<ArrayRef> {
    if slices.len() > 1 && !matches!(slices[0].data_type(), DataType::Dictionary(..)) {
        return concat_column(slices);
    }
    // Avoid Arrow's singleton slice shortcut and discard unused dictionary values.
    let empty = new_empty_array(slices[0].data_type());
    let mut inputs = slices.to_vec();
    inputs.push(empty.as_ref());
    concat_column(&inputs)
}

fn canonical_column(column: ArrayRef, canonical: &DataType) -> ArrayRef {
    match canonical {
        DataType::Timestamp(TimeUnit::Second, _) => {
            canonical_timestamp::<TimestampSecondType>(&column, canonical)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            canonical_timestamp::<TimestampMillisecondType>(&column, canonical)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            canonical_timestamp::<TimestampMicrosecondType>(&column, canonical)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            canonical_timestamp::<TimestampNanosecondType>(&column, canonical)
        }
        _ => column,
    }
}

fn canonical_timestamp<T: ArrowPrimitiveType>(column: &ArrayRef, canonical: &DataType) -> ArrayRef {
    Arc::new(
        column
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .expect("validated timestamp type")
            .clone()
            .with_data_type(canonical.clone()),
    )
}

fn exact_safe_duration_micros(duration: Duration, field: &str) -> Result<u64> {
    if duration.subsec_nanos() % 1_000 != 0 {
        return Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: "must be an exact multiple of one microsecond".into(),
        });
    }
    let micros =
        u64::try_from(duration.as_micros()).map_err(|_| CalcFlowError::InvalidArgument {
            field: field.into(),
            message: format!("must be at most {STREAM_JOIN_MAX_SAFE_JSON_INTEGER}"),
        })?;
    validate_safe_integer(micros, false, field)?;
    Ok(micros)
}

fn validate_safe_integer(value: u64, positive: bool, field: &str) -> Result<()> {
    if (positive && value == 0) || value > STREAM_JOIN_MAX_SAFE_JSON_INTEGER {
        return Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: if positive {
                format!("must be in 1..={STREAM_JOIN_MAX_SAFE_JSON_INTEGER}")
            } else {
                format!("must be in 0..={STREAM_JOIN_MAX_SAFE_JSON_INTEGER}")
            },
        });
    }
    Ok(())
}

fn validate_key_names(left: &[String], right: &[String]) -> Result<()> {
    if left.is_empty() || left.len() != right.len() {
        return Err(CalcFlowError::InvalidArgument {
            field: "stream_join.keys".into(),
            message: "left_keys and right_keys must be non-empty and equally sized".into(),
        });
    }
    for (side, keys) in [("left", left), ("right", right)] {
        if keys.iter().any(String::is_empty)
            || keys.iter().collect::<BTreeSet<_>>().len() != keys.len()
        {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("stream_join.{side}_keys"),
                message: "must contain unique non-empty column names".into(),
            });
        }
    }
    Ok(())
}

fn validate_column_name(value: &str, field: &str) -> Result<()> {
    if value.is_empty() {
        Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: "must name one column".into(),
        })
    } else {
        Ok(())
    }
}

fn validate_prefixes(left: &str, right: &str) -> Result<()> {
    if !is_portable_identifier(left) || !is_portable_identifier(right) || left == right {
        return Err(CalcFlowError::InvalidArgument {
            field: "stream_join.prefixes".into(),
            message: "must be distinct non-empty portable identifiers".into(),
        });
    }
    Ok(())
}

fn compile_schemas(
    left: &Schema,
    right: &Schema,
    spec: &StreamJoinSpec,
) -> Result<(SchemaRef, CompiledJoin)> {
    validate_unique_fields(left, "left")?;
    validate_unique_fields(right, "right")?;
    let (left_key_indices, right_key_indices) = compile_key_pair_indices(left, right, spec)?;
    let left_event_time_index = event_time_index(left, &spec.left_event_time, "left_event_time")?;
    let right_event_time_index =
        event_time_index(right, &spec.right_event_time, "right_event_time")?;
    let fields = prefixed_output_fields(left, right, spec);
    Ok((
        Arc::new(Schema::new(fields)),
        CompiledJoin {
            left_key_indices,
            right_key_indices,
            left_event_time_index,
            right_event_time_index,
            equality_query: parse_select_query(&equality_query(spec.left_keys.len()))?,
        },
    ))
}

fn compile_key_pair_indices(
    left: &Schema,
    right: &Schema,
    spec: &StreamJoinSpec,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let mut left_key_indices = Vec::with_capacity(spec.left_keys.len());
    let mut right_key_indices = Vec::with_capacity(spec.right_keys.len());
    for (index, (left_key, right_key)) in spec.left_keys.iter().zip(&spec.right_keys).enumerate() {
        let left_field = field_by_name(left, left_key, "left_keys", index)?;
        let right_field = field_by_name(right, right_key, "right_keys", index)?;
        validate_key_pair_types(index, left_field, right_field)?;
        left_key_indices.push(
            left.index_of(left_key)
                .expect("field lookup succeeded above"),
        );
        right_key_indices.push(
            right
                .index_of(right_key)
                .expect("field lookup succeeded above"),
        );
    }
    Ok((left_key_indices, right_key_indices))
}

fn validate_key_pair_types(index: usize, left_field: &Field, right_field: &Field) -> Result<()> {
    if left_field.data_type() != right_field.data_type()
        || !supported_key_type(left_field.data_type())
    {
        return Err(CalcFlowError::Compile {
            message: format!(
                "stream Join key pair {index} requires identical supported Arrow types; left is {} and right is {}",
                left_field.data_type(),
                right_field.data_type()
            ),
        });
    }
    Ok(())
}

fn event_time_index(schema: &Schema, name: &str, field: &str) -> Result<usize> {
    validate_event_time(schema, name, field)?;
    Ok(schema
        .index_of(name)
        .expect("event-time lookup succeeded above"))
}

fn prefixed_output_fields(left: &Schema, right: &Schema, spec: &StreamJoinSpec) -> Vec<Arc<Field>> {
    left.fields()
        .iter()
        .map(|field| {
            Arc::new(field.as_ref().clone().with_name(format!(
                "{}__{}",
                spec.left_prefix,
                field.name()
            )))
        })
        .chain(right.fields().iter().map(|field| {
            Arc::new(field.as_ref().clone().with_name(format!(
                "{}__{}",
                spec.right_prefix,
                field.name()
            )))
        }))
        .collect()
}

/// One batched key-equality probe over the per-batch scratch tables.
///
/// The query returns the admitted-row position and the retained-state row id
/// for every key-equal pair; the closed time bound is applied afterwards in
/// checked `i128` Rust arithmetic.
fn equality_query(key_count: usize) -> String {
    let equality = (0..key_count)
        .map(|position| {
            let column = quote_identifier(&format!("{KEY_COLUMN_PREFIX}{position}"));
            format!("{PROBE_TABLE}.{column} = {STATE_TABLE}.{column}")
        })
        .collect::<Vec<_>>()
        .join(" AND ");
    format!(
        "SELECT {PROBE_TABLE}.{pos}, {STATE_TABLE}.{rid} FROM {PROBE_TABLE} INNER JOIN {STATE_TABLE} ON {equality}",
        pos = quote_identifier(PROBE_POS_COLUMN),
        rid = quote_identifier(STATE_RID_COLUMN),
    )
}

fn quote_identifier(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

/// Reports whether the frozen v1 state-charge table covers `data_type`
/// recursively; a genuinely new Arrow payload type fails construction instead
/// of being charged as zero (spec FR16/D16).
fn payload_charge_supported(data_type: &DataType) -> bool {
    match data_type {
        DataType::Null
        | DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Timestamp(_, _)
        | DataType::Duration(_)
        | DataType::Interval(_)
        | DataType::Decimal32(_, _)
        | DataType::Decimal64(_, _)
        | DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _)
        | DataType::FixedSizeBinary(_)
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => payload_charge_supported(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .all(|field| payload_charge_supported(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .all(|(_type_id, field)| payload_charge_supported(field.data_type())),
        DataType::Dictionary(_, value) => payload_charge_supported(value),
        DataType::RunEndEncoded(_, values) => payload_charge_supported(values.data_type()),
    }
}

/// Rejects schemas whose payload types have no versioned state charge.
fn validate_payload_charge_support(schema: &Schema, side: &str) -> Result<()> {
    for field in schema.fields() {
        if !payload_charge_supported(field.data_type()) {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("stream_join.{side}_schema.{}", field.name()),
                message: format!(
                    "unsupported_payload_type: {} has no versioned state charge",
                    field.data_type()
                ),
            });
        }
    }
    Ok(())
}

fn validate_unique_fields(schema: &Schema, side: &str) -> Result<()> {
    let mut names = BTreeSet::new();
    if schema.fields().is_empty()
        || schema
            .fields()
            .iter()
            .any(|field| !names.insert(field.name()))
    {
        return Err(CalcFlowError::Compile {
            message: format!("stream Join {side} schema must be non-empty with unique field names"),
        });
    }
    Ok(())
}

fn field_by_name<'a>(
    schema: &'a Schema,
    name: &str,
    field: &str,
    index: usize,
) -> Result<&'a Field> {
    schema
        .field_with_name(name)
        .map_err(|_| CalcFlowError::Compile {
            message: format!("stream_join.{field}[{index}] names missing column {name:?}"),
        })
}

fn validate_event_time(schema: &Schema, name: &str, field: &str) -> Result<()> {
    let column = schema
        .field_with_name(name)
        .map_err(|_| CalcFlowError::Compile {
            message: format!("stream_join.{field} names missing column {name:?}"),
        })?;
    let DataType::Timestamp(_, timezone) = column.data_type() else {
        return Err(CalcFlowError::Compile {
            message: format!("stream_join.{field} must be an Arrow timestamp"),
        });
    };
    if timezone
        .as_deref()
        .is_some_and(|timezone| timezone != "UTC")
    {
        return Err(CalcFlowError::Compile {
            message: format!("stream_join.{field} timestamp timezone must be UTC or absent"),
        });
    }
    Ok(())
}

pub(crate) fn supported_key_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, _)
    )
}

const JOIN_STATE_MAGIC: &[u8; 8] = b"CFJOIN1\0";

/// Number of carried delta segments that triggers compaction on the next
/// asynchronous checkpoint preparation (spec FR10).
const JOIN_DELTA_COMPACTION_SEGMENTS: u32 = 4;

const JOIN_DELTA_MAGIC: &[u8; 8] = b"CFJDLT1\0";
const JOIN_DELTA_UPSERT_TAG: u8 = 1;
const JOIN_DELTA_TOMBSTONE_TAG: u8 = 2;

/// Encodes the dirty ops of one epoch into per-side delta segments.
///
/// Upserts encode the records they carry from admission, so the encode cost is
/// proportional to the dirty set, never to the full state (spec FR47).
fn encode_pending_delta(
    state: &StreamJoinState,
    epoch: Epoch,
    operator_id: &str,
) -> Result<Vec<(JoinSide, Vec<u8>)>> {
    let mut encoded = Vec::new();
    let mut ipc_encoder = row_ipc::RowIpcEncoder::default();
    for side in [JoinSide::Left, JoinSide::Right] {
        let ops = state
            .deltas
            .pending
            .iter()
            .filter(|op| op.side() == side)
            .collect::<Vec<_>>();
        if ops.is_empty() {
            continue;
        }
        let mut segment = Vec::new();
        segment.extend_from_slice(JOIN_DELTA_MAGIC);
        segment.extend_from_slice(&ops.len().to_le_bytes());
        for op in ops {
            encode_delta_op(&mut segment, op, &mut ipc_encoder, operator_id)?;
        }
        let _ = epoch;
        encoded.push((side, segment));
    }
    Ok(encoded)
}

/// Appends one dirty op's tag, identity and (for upserts) carried row IPC.
fn encode_delta_op(
    segment: &mut Vec<u8>,
    op: &PendingOp,
    ipc_encoder: &mut row_ipc::RowIpcEncoder,
    operator_id: &str,
) -> Result<()> {
    segment.push(match op {
        PendingOp::Upsert { .. } => JOIN_DELTA_UPSERT_TAG,
        PendingOp::Tombstone { .. } => JOIN_DELTA_TOMBSTONE_TAG,
    });
    let (row_id, event_time, encoded_key) = match op {
        PendingOp::Upsert {
            row_id,
            event_time,
            encoded_key,
            ..
        }
        | PendingOp::Tombstone {
            row_id,
            event_time,
            encoded_key,
            ..
        } => (*row_id, *event_time, encoded_key.as_slice()),
    };
    segment.extend_from_slice(&row_id.to_le_bytes());
    segment.extend_from_slice(&event_time.as_micros().to_le_bytes());
    segment.extend_from_slice(
        &u64::try_from(encoded_key.len())
            .map_err(|_| counter_overflow(operator_id, "encoded key length"))?
            .to_le_bytes(),
    );
    segment.extend_from_slice(encoded_key);
    if let PendingOp::Upsert { record, charge, .. } = op {
        segment.extend_from_slice(&charge.to_le_bytes());
        let ipc = ipc_encoder.encode(&record.view(), operator_id, op.side().as_str())?;
        segment.extend_from_slice(
            &u64::try_from(ipc.len())
                .map_err(|_| counter_overflow(operator_id, "IPC length"))?
                .to_le_bytes(),
        );
        segment.extend_from_slice(&ipc);
    }
    Ok(())
}

impl PendingOp {
    const fn side(&self) -> JoinSide {
        match self {
            PendingOp::Upsert { side, .. } | PendingOp::Tombstone { side, .. } => *side,
        }
    }
}

/// Restores both sides by folding the base segment and the delta segments in
/// ascending `(epoch, segment_id)` order; later operations win (spec FR45).
fn restore_sides_from_segments(
    snapshot: &OperatorStateSnapshot,
    left_schema: RestoreSchema<'_>,
    right_schema: RestoreSchema<'_>,
    left_key_indices: &[usize],
    right_key_indices: &[usize],
    operator_id: &str,
) -> Result<(Vec<StoredRow>, Vec<StoredRow>)> {
    let mut inventory: Vec<(&str, SegmentKind)> = Vec::new();
    for segment_id in snapshot.segments.keys() {
        inventory.push((
            segment_id.as_str(),
            parse_segment_kind(segment_id, operator_id)?,
        ));
    }
    if inventory.is_empty() {
        return Err(CalcFlowError::CheckpointMismatch {
            message: format!("stream Join {operator_id:?} segment inventory is empty"),
        });
    }
    let mut left = fold_side(
        &snapshot.segments,
        &inventory,
        JoinSide::Left,
        left_schema,
        left_key_indices,
        operator_id,
    )?;
    let mut right = fold_side(
        &snapshot.segments,
        &inventory,
        JoinSide::Right,
        right_schema,
        right_key_indices,
        operator_id,
    )?;
    left.sort_by(identity_order);
    right.sort_by(identity_order);
    Ok((left, right))
}

fn identity_order(left: &StoredRow, right: &StoredRow) -> std::cmp::Ordering {
    (
        left.encoded_key.as_slice(),
        left.event_time.as_micros(),
        left.row_id,
    )
        .cmp(&(
            right.encoded_key.as_slice(),
            right.event_time.as_micros(),
            right.row_id,
        ))
}

/// Rebuilds the carried delta-segment map from a restored snapshot so the
/// next checkpoint carries them forward without re-encoding.
type CarriedDeltaSegments = BTreeMap<(u64, &'static str), StateSegment>;

fn carried_base_segments(snapshot: &OperatorStateSnapshot) -> BTreeMap<&'static str, StateSegment> {
    let mut base = BTreeMap::new();
    for (segment_id, segment) in &snapshot.segments {
        if let Some(side) = ["left", "right"]
            .into_iter()
            .find(|side| segment_id == &format!("{side}-base"))
        {
            base.insert(side, segment.clone());
        }
    }
    base
}

fn carried_delta_segments(
    snapshot: &OperatorStateSnapshot,
    operator_id: &str,
) -> Result<CarriedDeltaSegments> {
    let mut carried = BTreeMap::new();
    for (segment_id, segment) in &snapshot.segments {
        let side = ["left", "right"]
            .into_iter()
            .find(|side| segment_id.starts_with(side) && segment_id.contains("-delta-"));
        let Some(side) = side else {
            continue;
        };
        let epoch = segment_id
            .rsplit('-')
            .next()
            .and_then(|value| value.parse::<u64>().ok())
            .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                message: format!("stream Join {operator_id:?} segment id is invalid"),
            })?;
        carried.insert((epoch, side), segment.clone());
    }
    Ok(carried)
}

enum SegmentKind {
    Base,
    Delta(u64),
}

fn parse_segment_kind(segment_id: &str, operator_id: &str) -> Result<SegmentKind> {
    for side in ["left", "right"] {
        if segment_id == format!("{side}-base") {
            return Ok(SegmentKind::Base);
        }
        if let Some(rest) = segment_id.strip_prefix(&format!("{side}-delta-")) {
            let epoch = rest
                .parse::<u64>()
                .map_err(|_| CalcFlowError::CheckpointMismatch {
                    message: format!("stream Join {operator_id:?} segment id is invalid"),
                })?;
            return Ok(SegmentKind::Delta(epoch));
        }
    }
    Err(CalcFlowError::CheckpointMismatch {
        message: format!("stream Join {operator_id:?} segment id is invalid"),
    })
}

/// Folds one side's base and deltas into durable-identity order.
fn fold_side(
    segments: &BTreeMap<String, StateSegment>,
    inventory: &[(&str, SegmentKind)],
    side: JoinSide,
    schema: RestoreSchema<'_>,
    key_indices: &[usize],
    operator_id: &str,
) -> Result<Vec<StoredRow>> {
    let mut folded: BTreeMap<(Vec<u8>, i64, u64), StoredRow> = BTreeMap::new();
    let side_str = side.as_str();
    let mut ordered: Vec<(&str, &SegmentKind, &StateSegment)> = inventory
        .iter()
        .filter(|(segment_id, _)| segment_id.starts_with(side_str))
        .map(|(segment_id, kind)| (*segment_id, kind, &segments[*segment_id]))
        .collect();
    // BTreeMap iteration gives ascending segment ids; the base folds first.
    ordered.sort_by_key(|(segment_id, kind, _)| match kind {
        SegmentKind::Base => (u64::MIN, (*segment_id).to_owned()),
        SegmentKind::Delta(epoch) => (*epoch, (*segment_id).to_owned()),
    });
    for (_segment_id, kind, segment) in ordered {
        let bytes = segment.bytes();
        match kind {
            SegmentKind::Base => {
                for row in decode_side(bytes, schema, key_indices, operator_id, side_str)? {
                    folded.insert(
                        (
                            row.encoded_key.to_vec(),
                            row.event_time.as_micros(),
                            row.row_id,
                        ),
                        row,
                    );
                }
            }
            SegmentKind::Delta(_) => {
                decode_delta_segment(
                    bytes,
                    schema,
                    key_indices,
                    operator_id,
                    side_str,
                    &mut folded,
                )?;
            }
        }
    }
    Ok(folded.into_values().collect())
}

/// Applies one delta segment's upserts and tombstones to the fold.
fn decode_delta_segment(
    bytes: &[u8],
    schema: RestoreSchema<'_>,
    key_indices: &[usize],
    operator_id: &str,
    side: &str,
    folded: &mut BTreeMap<(Vec<u8>, i64, u64), StoredRow>,
) -> Result<()> {
    let mut offset = 0_usize;
    let op_count = delta_op_count(bytes, &mut offset, operator_id, side)?;
    let decoder = DeltaDecoder {
        schema,
        key_indices,
        name: operator_id,
        side,
    };
    let mut seen_identities = BTreeSet::new();
    for _ in 0..op_count {
        let header = decode_delta_header(bytes, &mut offset, operator_id, side)?;
        let identity = (
            header.key.clone(),
            header.event_time.as_micros(),
            header.row_id,
        );
        if !seen_identities.insert(identity.clone()) {
            return Err(checkpoint_error(
                operator_id,
                side,
                "delta segment repeats one row identity",
            ));
        }
        apply_delta_op(bytes, &mut offset, &decoder, header, identity, folded)?;
    }
    if offset != bytes.len() {
        return Err(checkpoint_error(
            operator_id,
            side,
            "delta segment has trailing bytes",
        ));
    }
    Ok(())
}

struct DeltaDecoder<'a> {
    schema: RestoreSchema<'a>,
    key_indices: &'a [usize],
    name: &'a str,
    side: &'a str,
}

struct DeltaHeader {
    tag: u8,
    row_id: u64,
    event_time: EventTime,
    key: Vec<u8>,
}

fn delta_op_count(bytes: &[u8], offset: &mut usize, name: &str, side: &str) -> Result<usize> {
    if take_segment_bytes(bytes, offset, JOIN_DELTA_MAGIC.len())? != JOIN_DELTA_MAGIC {
        return Err(checkpoint_error(name, side, "delta magic is invalid"));
    }
    usize::try_from(read_segment_u64(bytes, offset)?)
        .map_err(|_| checkpoint_error(name, side, "delta op count is invalid"))
}

fn decode_delta_header(
    bytes: &[u8],
    offset: &mut usize,
    name: &str,
    side: &str,
) -> Result<DeltaHeader> {
    let tag = take_segment_bytes(bytes, offset, 1)?[0];
    let row_id = read_segment_u64(bytes, offset)?;
    let event_time = EventTime::from_micros(read_segment_i64(bytes, offset)?);
    let length = usize::try_from(read_segment_u64(bytes, offset)?)
        .map_err(|_| checkpoint_error(name, side, "delta key length is invalid"))?;
    let key = take_segment_bytes(bytes, offset, length)?.to_vec();
    Ok(DeltaHeader {
        tag,
        row_id,
        event_time,
        key,
    })
}

fn apply_delta_op(
    bytes: &[u8],
    offset: &mut usize,
    decoder: &DeltaDecoder<'_>,
    header: DeltaHeader,
    identity: (Vec<u8>, i64, u64),
    folded: &mut BTreeMap<(Vec<u8>, i64, u64), StoredRow>,
) -> Result<()> {
    match header.tag {
        JOIN_DELTA_UPSERT_TAG => {
            let row = decode_delta_upsert(bytes, offset, decoder, header)?;
            folded.insert(identity, row);
        }
        JOIN_DELTA_TOMBSTONE_TAG => {
            folded.remove(&identity);
        }
        _ => {
            return Err(checkpoint_error(
                decoder.name,
                decoder.side,
                "delta op tag is invalid",
            ));
        }
    }
    Ok(())
}

fn decode_delta_upsert(
    bytes: &[u8],
    offset: &mut usize,
    decoder: &DeltaDecoder<'_>,
    header: DeltaHeader,
) -> Result<StoredRow> {
    let DeltaDecoder {
        schema,
        key_indices,
        name,
        side,
    } = *decoder;
    let charge = read_segment_u64(bytes, offset)?;
    let record = read_ipc_record(bytes, offset, schema, name, side)?;
    if encode_join_key_v1(&record, 0, key_indices)? != header.key {
        return Err(checkpoint_error(
            name,
            side,
            "delta upsert key does not match its record",
        ));
    }
    Ok(StoredRow {
        record: record.into(),
        event_time: header.event_time,
        row_id: header.row_id,
        charge,
        encoded_key: Arc::new(header.key.into()),
    })
}

fn read_ipc_record(
    bytes: &[u8],
    offset: &mut usize,
    schema: RestoreSchema<'_>,
    name: &str,
    side: &str,
) -> Result<RecordBatch> {
    let length = usize::try_from(read_segment_u64(bytes, offset)?)
        .map_err(|_| checkpoint_error(name, side, "IPC length is invalid"))?;
    decode_ipc_row(
        take_segment_bytes(bytes, offset, length)?,
        schema,
        name,
        side,
    )
}

fn encode_side(
    rows: &[StoredRow],
    operator_id: &str,
    side: &str,
    check: &impl Fn() -> Result<()>,
) -> Result<Vec<u8>> {
    let ordered = ordered_checkpoint_rows(rows, check)?;
    let mut output = Vec::new();
    output.extend_from_slice(JOIN_STATE_MAGIC);
    output.extend_from_slice(
        &u64::try_from(ordered.len())
            .map_err(|_| counter_overflow(operator_id, "checkpoint rows"))?
            .to_le_bytes(),
    );
    let mut ipc_encoder = row_ipc::RowIpcEncoder::default();
    for row in ordered {
        check()?;
        append_stored_row(&mut output, row, &mut ipc_encoder, operator_id, side)?;
    }
    check()?;
    Ok(output)
}

fn ordered_checkpoint_rows<'a>(
    rows: &'a [StoredRow],
    check: &impl Fn() -> Result<()>,
) -> Result<Vec<&'a StoredRow>> {
    check()?;
    let mut ordered = rows.iter().collect::<Vec<_>>();
    check()?;
    ordered.sort_by(|a, b| identity_order(a, b));
    check()?;
    Ok(ordered)
}

fn append_stored_row(
    output: &mut Vec<u8>,
    row: &StoredRow,
    ipc_encoder: &mut row_ipc::RowIpcEncoder,
    operator_id: &str,
    side: &str,
) -> Result<()> {
    let ipc = ipc_encoder.encode(&row.record.view(), operator_id, side)?;
    output.extend_from_slice(&row.row_id.to_le_bytes());
    output.extend_from_slice(&row.event_time.as_micros().to_le_bytes());
    output.extend_from_slice(&row.charge.to_le_bytes());
    output.extend_from_slice(
        &u64::try_from(ipc.len())
            .map_err(|_| counter_overflow(operator_id, "IPC length"))?
            .to_le_bytes(),
    );
    output.extend_from_slice(&ipc);
    Ok(())
}

fn decode_side(
    bytes: &[u8],
    expected_schema: RestoreSchema<'_>,
    key_indices: &[usize],
    operator_id: &str,
    side: &str,
) -> Result<Vec<StoredRow>> {
    let mut offset = 0_usize;
    let row_count = decode_side_header(bytes, &mut offset, operator_id, side)?;
    let rows = decode_side_rows(
        bytes,
        &mut offset,
        row_count,
        expected_schema,
        key_indices,
        operator_id,
        side,
    )?;
    if offset != bytes.len() {
        return Err(checkpoint_error(
            operator_id,
            side,
            "state segment has trailing bytes",
        ));
    }
    Ok(rows)
}

/// Validates the state magic and returns the declared row count.
fn decode_side_header(
    bytes: &[u8],
    offset: &mut usize,
    operator_id: &str,
    side: &str,
) -> Result<u64> {
    if take_segment_bytes(bytes, offset, JOIN_STATE_MAGIC.len())? != JOIN_STATE_MAGIC {
        return Err(checkpoint_error(
            operator_id,
            side,
            "state magic is invalid",
        ));
    }
    read_segment_u64(bytes, offset)
}

fn decode_side_rows(
    bytes: &[u8],
    offset: &mut usize,
    row_count: u64,
    expected_schema: RestoreSchema<'_>,
    key_indices: &[usize],
    operator_id: &str,
    side: &str,
) -> Result<Vec<StoredRow>> {
    let row_capacity = decode_row_capacity(row_count, bytes.len(), operator_id, side)?;
    let mut rows = Vec::with_capacity(row_capacity);
    for _ in 0..row_count {
        rows.push(decode_stored_row(
            bytes,
            offset,
            expected_schema,
            key_indices,
            operator_id,
            side,
        )?);
    }
    Ok(rows)
}

fn decode_row_capacity(
    row_count: u64,
    segment_len: usize,
    operator_id: &str,
    side: &str,
) -> Result<usize> {
    usize::try_from(row_count)
        .ok()
        .filter(|count| *count <= segment_len)
        .ok_or_else(|| checkpoint_error(operator_id, side, "row count is invalid"))
}

fn decode_stored_row(
    bytes: &[u8],
    offset: &mut usize,
    expected_schema: RestoreSchema<'_>,
    key_indices: &[usize],
    operator_id: &str,
    side: &str,
) -> Result<StoredRow> {
    let row_id = read_segment_u64(bytes, offset)?;
    let event_time = EventTime::from_micros(read_segment_i64(bytes, offset)?);
    let charge = read_segment_u64(bytes, offset)?;
    let record = read_ipc_record(bytes, offset, expected_schema, operator_id, side)?;
    let encoded_key = Arc::new(encode_join_key_v1(&record, 0, key_indices)?.into());
    Ok(StoredRow {
        record: record.into(),
        event_time,
        row_id,
        charge,
        encoded_key,
    })
}

fn decode_ipc_row(
    ipc: &[u8],
    expected_schema: RestoreSchema<'_>,
    operator_id: &str,
    side: &str,
) -> Result<RecordBatch> {
    let mut reader = StreamReader::try_new(Cursor::new(ipc), None).map_err(|error| {
        checkpoint_error(
            operator_id,
            side,
            &format!("IPC header is invalid: {error}"),
        )
    })?;
    #[cfg(test)]
    if let Some(hook) = expected_schema.hook {
        hook(expected_schema.credit, false);
    }
    if reader.schema().as_ref() != expected_schema.schema {
        return Err(checkpoint_error(
            operator_id,
            side,
            "IPC schema is incompatible",
        ));
    }
    let record = reader
        .next()
        .transpose()
        .map_err(|error| {
            checkpoint_error(operator_id, side, &format!("IPC row is invalid: {error}"))
        })?
        .filter(|record| record.num_rows() == 1)
        .ok_or_else(|| checkpoint_error(operator_id, side, "IPC must contain one row"))?;
    if reader.next().is_some() {
        return Err(checkpoint_error(
            operator_id,
            side,
            "IPC contains extra record batches",
        ));
    }
    Ok(record)
}

fn take_segment_bytes<'a>(bytes: &'a [u8], offset: &mut usize, length: usize) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| CalcFlowError::CheckpointMismatch {
            message: "stream Join state segment offset overflowed".into(),
        })?;
    let value = bytes
        .get(*offset..end)
        .ok_or_else(|| CalcFlowError::CheckpointMismatch {
            message: "stream Join state segment is truncated".into(),
        })?;
    *offset = end;
    Ok(value)
}

fn read_segment_u64(bytes: &[u8], offset: &mut usize) -> Result<u64> {
    let value = take_segment_bytes(bytes, offset, 8)?;
    Ok(u64::from_le_bytes(
        value.try_into().expect("exact eight-byte segment slice"),
    ))
}

fn read_segment_i64(bytes: &[u8], offset: &mut usize) -> Result<i64> {
    let value = take_segment_bytes(bytes, offset, 8)?;
    Ok(i64::from_le_bytes(
        value.try_into().expect("exact eight-byte segment slice"),
    ))
}

fn validate_restored_rows(
    rows: &[StoredRow],
    next_row_id: u64,
    event_index: usize,
    key_indices: &[usize],
    operator_id: &str,
    side: &str,
) -> Result<()> {
    let mut identities = BTreeSet::new();
    for row in rows {
        validate_restored_row_identity(row, &mut identities, next_row_id, operator_id, side)?;
        validate_restored_row_payload(row, event_index, key_indices, operator_id, side)?;
    }
    Ok(())
}

fn validate_restored_row_identity(
    row: &StoredRow,
    identities: &mut BTreeSet<(EventTime, u64)>,
    next_row_id: u64,
    operator_id: &str,
    side: &str,
) -> Result<()> {
    if row.row_id >= next_row_id || !identities.insert((row.event_time, row.row_id)) {
        return Err(checkpoint_error(
            operator_id,
            side,
            "row identity is invalid",
        ));
    }
    Ok(())
}

fn validate_restored_row_payload(
    row: &StoredRow,
    event_index: usize,
    key_indices: &[usize],
    operator_id: &str,
    side: &str,
) -> Result<()> {
    let record = row.record.view();
    let restored_event_time = event_time_at(&record, event_index, 0, operator_id, side)?
        .ok_or_else(|| checkpoint_error(operator_id, side, "stored event time is null"))?;
    let restored_charge = state_row_charge(&record, 0, key_indices, operator_id)?;
    if restored_event_time != row.event_time || restored_charge != row.charge {
        return Err(checkpoint_error(
            operator_id,
            side,
            "row event time or charge is inconsistent",
        ));
    }
    Ok(())
}

fn checkpoint_error(operator_id: &str, side: &str, message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: format!("stream Join {operator_id:?} {side} {message}"),
    }
}

fn should_retain(
    incoming_is_left: bool,
    event_time: EventTime,
    opposite: Option<IngressProgress>,
    bounds: JoinTimeBounds,
) -> bool {
    let Some(opposite) = opposite else {
        return true;
    };
    if opposite.state() == crate::IngressState::Ended {
        return false;
    }
    let Some(watermark) = opposite.watermark() else {
        return true;
    };
    let extension = if incoming_is_left {
        bounds.after_micros
    } else {
        bounds.before_micros
    };
    i128::from(event_time.as_micros()) + i128::from(extension) >= i128::from(watermark.as_micros())
}

#[derive(Clone, Copy)]
struct EvictionPolicy<'a> {
    extension_micros: u64,
    side: JoinSide,
    operator_id: &'a str,
}

fn evict_opposite(
    rows: &mut RetainedRows,
    expirations: &mut ExpirationIndex,
    progress: IngressProgress,
    metrics: &mut SideMetrics,
    pending: &mut PendingLog,
    policy: EvictionPolicy<'_>,
) -> Result<()> {
    let EvictionPolicy {
        extension_micros,
        side,
        operator_id,
    } = policy;
    let mut evicted = Vec::new();
    let mut bytes = 0_u64;
    while let Some((ordinal, row)) = take_expired(rows, expirations, progress, extension_micros) {
        rows.2.remove(&row.record);
        bytes = bytes
            .checked_add(row.charge)
            .ok_or_else(|| counter_overflow(operator_id, "evicted bytes"))?;
        evicted.push((ordinal, row));
    }
    if evicted.is_empty() {
        return Ok(());
    }
    evicted.sort_by_key(|(ordinal, _)| *ordinal);
    let count =
        u64::try_from(evicted.len()).map_err(|_| counter_overflow(operator_id, "evicted rows"))?;
    record_tombstones(
        pending,
        side,
        evicted
            .into_iter()
            .map(|(_, row)| (row.row_id, row.event_time, row.encoded_key))
            .collect(),
    );
    update_evicted_metrics(metrics, count, bytes, operator_id)
}

fn update_evicted_metrics(
    metrics: &mut SideMetrics,
    count: u64,
    bytes: u64,
    name: &str,
) -> Result<()> {
    metrics.evicted_rows = checked_metric(metrics.evicted_rows, count, name, "evicted_rows")?;
    metrics.retained_rows = metrics
        .retained_rows
        .checked_sub(count)
        .ok_or_else(|| counter_overflow(name, "retained rows"))?;
    metrics.retained_bytes = metrics
        .retained_bytes
        .checked_sub(bytes)
        .ok_or_else(|| counter_overflow(name, "retained bytes"))?;
    Ok(())
}

fn take_expired(
    rows: &mut RetainedRows,
    expirations: &mut ExpirationIndex,
    progress: IngressProgress,
    extension: u64,
) -> Option<(u128, StoredRow)> {
    let (&(time, _), _) = expirations.entries.first_key_value()?;
    let expired = progress.state() == crate::IngressState::Ended
        || progress.watermark().is_some_and(|watermark| {
            i128::from(time.as_micros()) + i128::from(extension) < i128::from(watermark.as_micros())
        });
    if !expired {
        return None;
    }
    let (_, (index, ordinal)) = expirations
        .entries
        .pop_first()
        .expect("expiration prefix exists");
    #[cfg(test)]
    note_join_work(|work| work.retained_visits += 1);
    let row = rows.swap_remove(index);
    if let Some(moved) = rows.get(index) {
        expirations
            .entries
            .get_mut(&(moved.event_time, moved.row_id))
            .expect("every live row has an expiration entry")
            .0 = index;
    }
    Some((ordinal, row))
}

/// Records evictions in the dirty log: an upsert still waiting for its first
/// checkpoint coalesces away; a captured row leaves a durable tombstone.
fn record_tombstones(
    pending: &mut PendingLog,
    side: JoinSide,
    evicted: Vec<(u64, EventTime, Arc<columnar::FramedKey>)>,
) {
    for (row_id, event_time, encoded_key) in evicted {
        #[cfg(test)]
        note_join_work(|work| work.pending_visits += 1);
        if pending.remove_upsert((side, row_id)) {
            continue;
        }
        pending.push(PendingOp::Tombstone {
            side,
            row_id,
            event_time,
            encoded_key,
        });
    }
}

fn refresh_retained_metrics(
    metrics: &mut SideMetrics,
    rows: &[StoredRow],
    operator_id: &str,
) -> Result<()> {
    metrics.retained_rows =
        u64::try_from(rows.len()).map_err(|_| counter_overflow(operator_id, "retained rows"))?;
    metrics.retained_bytes = rows
        .iter()
        .try_fold(0_u64, |total, row| {
            #[cfg(test)]
            note_join_work(|work| work.retained_visits += 1);
            total.checked_add(row.charge)
        })
        .ok_or_else(|| counter_overflow(operator_id, "retained bytes"))?;
    Ok(())
}

fn event_time_at(
    record: &RecordBatch,
    column_index: usize,
    row_index: usize,
    operator_id: &str,
    side: &str,
) -> Result<Option<EventTime>> {
    #[cfg(test)]
    note_join_work(|work| work.time_decoders += 1);
    let array = record.column(column_index).as_ref();
    if array.is_null(row_index) {
        return Ok(None);
    }
    let data_type = record.schema().field(column_index).data_type().clone();
    let value = match &data_type {
        DataType::Timestamp(TimeUnit::Second, _) => {
            downcast_timestamp::<TimestampSecondArray>(array, row_index, operator_id, side)?
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            downcast_timestamp::<TimestampMillisecondArray>(array, row_index, operator_id, side)?
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            downcast_timestamp::<TimestampMicrosecondArray>(array, row_index, operator_id, side)?
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            downcast_timestamp::<TimestampNanosecondArray>(array, row_index, operator_id, side)?
        }
        _ => {
            return Err(operator_reason(
                operator_id,
                crate::StreamingFailureReason::JoinTimeConversionFailed,
                &format!("{side} event time is not a timestamp"),
            ));
        }
    };
    EventTime::import_timestamp(value, &data_type, &format!("stream_join.{side}_event_time"))
        .map(Some)
        .map_err(|_| {
            operator_reason(
                operator_id,
                crate::StreamingFailureReason::JoinTimeConversionFailed,
                &format!("{side} event time cannot be represented"),
            )
        })
}

struct BatchEventTimes<'a> {
    array: &'a dyn Array,
    values: &'a [i64],
    unit: TimeUnit,
}

impl<'a> BatchEventTimes<'a> {
    fn new(array: &'a dyn Array, operator_id: &str, side: &str) -> Result<Self> {
        #[cfg(test)]
        note_join_work(|work| work.time_decoders += 1);
        let DataType::Timestamp(unit, timezone) = array.data_type() else {
            return Err(operator_reason(
                operator_id,
                crate::StreamingFailureReason::JoinTimeConversionFailed,
                &format!("{side} event time is not a timestamp"),
            ));
        };
        if timezone
            .as_deref()
            .is_some_and(|timezone| timezone != "UTC")
        {
            return Err(operator_reason(
                operator_id,
                crate::StreamingFailureReason::JoinTimeConversionFailed,
                &format!("{side} event time cannot be represented"),
            ));
        }
        macro_rules! values {
            ($array:ty) => {
                array
                    .as_any()
                    .downcast_ref::<$array>()
                    .map(|typed| typed.values().as_ref())
                    .ok_or_else(|| {
                        operator_reason(
                            operator_id,
                            crate::StreamingFailureReason::JoinTimeConversionFailed,
                            &format!("{side} timestamp array type mismatch"),
                        )
                    })?
            };
        }
        let values = match unit {
            TimeUnit::Second => values!(TimestampSecondArray),
            TimeUnit::Millisecond => values!(TimestampMillisecondArray),
            TimeUnit::Microsecond => values!(TimestampMicrosecondArray),
            TimeUnit::Nanosecond => values!(TimestampNanosecondArray),
        };
        Ok(Self {
            array,
            values,
            unit: *unit,
        })
    }

    fn at(&self, row: usize, operator_id: &str, side: &str) -> Result<Option<EventTime>> {
        if self.array.is_null(row) {
            return Ok(None);
        }
        let value = self.values[row];
        let micros = match self.unit {
            TimeUnit::Second => value.checked_mul(1_000_000),
            TimeUnit::Millisecond => value.checked_mul(1_000),
            TimeUnit::Microsecond => Some(value),
            TimeUnit::Nanosecond => Some(value.div_euclid(1_000)),
        }
        .ok_or_else(|| {
            operator_reason(
                operator_id,
                crate::StreamingFailureReason::JoinTimeConversionFailed,
                &format!("{side} event time cannot be represented"),
            )
        })?;
        Ok(Some(EventTime::from_micros(micros)))
    }
}

fn downcast_timestamp<T>(
    array: &dyn Array,
    row_index: usize,
    operator_id: &str,
    side: &str,
) -> Result<i64>
where
    T: Array + 'static,
    for<'a> &'a T: TimestampValue,
{
    let typed = array.as_any().downcast_ref::<T>().ok_or_else(|| {
        operator_reason(
            operator_id,
            crate::StreamingFailureReason::JoinTimeConversionFailed,
            &format!("{side} timestamp array type mismatch"),
        )
    })?;
    Ok(typed.timestamp_value(row_index))
}

trait TimestampValue {
    fn timestamp_value(self, row_index: usize) -> i64;
}

macro_rules! impl_timestamp_value {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl TimestampValue for &$ty {
                fn timestamp_value(self, row_index: usize) -> i64 {
                    self.value(row_index)
                }
            }
        )+
    };
}

impl_timestamp_value!(
    TimestampSecondArray,
    TimestampMillisecondArray,
    TimestampMicrosecondArray,
    TimestampNanosecondArray,
);

fn state_row_charge(
    record: &RecordBatch,
    row_index: usize,
    key_indices: &[usize],
    operator_id: &str,
) -> Result<u64> {
    let encoded_key = encode_join_key_v1(record, row_index, key_indices)?;
    state_row_charge_with_key(record, row_index, encoded_key.len(), operator_id)
}

fn state_row_charge_with_key(
    record: &RecordBatch,
    row_index: usize,
    encoded_key_len: usize,
    operator_id: &str,
) -> Result<u64> {
    state_columns_charge_with_key(record.columns(), row_index, encoded_key_len, operator_id)
}

fn state_columns_charge_with_key(
    columns: &[ArrayRef],
    row_index: usize,
    encoded_key_len: usize,
    operator_id: &str,
) -> Result<u64> {
    let key_bytes =
        u64::try_from(encoded_key_len).map_err(|_| counter_overflow(operator_id, "encoded key"))?;
    let payload = columns.iter().try_fold(0_u64, |total, column| {
        let charge = logical_cell_charge(column.as_ref(), row_index)?;
        total
            .checked_add(charge)
            .ok_or_else(|| counter_overflow(operator_id, "logical payload bytes"))
    })?;
    STREAM_JOIN_STATE_ROW_OVERHEAD_BYTES_V1
        .checked_add(key_bytes)
        .and_then(|value| value.checked_add(16))
        .and_then(|value| value.checked_add(payload))
        .ok_or_else(|| counter_overflow(operator_id, "state row charge"))
}

/// Logical charge of one non-null cell under the frozen V1 accounting table,
/// including its validity byte.
fn logical_cell_charge(array: &dyn Array, row_index: usize) -> Result<u64> {
    if array.is_null(row_index) {
        return Ok(1);
    }
    let data_type = array.data_type();
    if let Some(value) = fixed_cell_charge(data_type) {
        return validity_wrapped(value);
    }
    let Some(value) = variable_cell_charge(array, row_index)? else {
        return sized_cell_charge(array, row_index).and_then(validity_wrapped);
    };
    validity_wrapped(value)
}

fn fixed_cell_charge(data_type: &DataType) -> Option<u64> {
    if matches!(data_type, DataType::Null) {
        return Some(0);
    }
    if matches!(
        data_type,
        DataType::Boolean | DataType::Int8 | DataType::UInt8
    ) {
        return Some(1);
    }
    if matches!(
        data_type,
        DataType::Int16 | DataType::UInt16 | DataType::Float16
    ) {
        return Some(2);
    }
    if is_four_byte_cell(data_type) {
        return Some(4);
    }
    if is_eight_byte_cell(data_type) {
        return Some(8);
    }
    fixed_wide_cell_charge(data_type)
}

fn fixed_wide_cell_charge(data_type: &DataType) -> Option<u64> {
    if is_sixteen_byte_cell(data_type) {
        return Some(16);
    }
    if matches!(data_type, DataType::Decimal256(_, _)) {
        return Some(32);
    }
    None
}

fn is_four_byte_cell(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int32
            | DataType::UInt32
            | DataType::Float32
            | DataType::Date32
            | DataType::Time32(_)
            | DataType::Interval(IntervalUnit::YearMonth)
            | DataType::Decimal32(_, _)
    )
}

fn is_eight_byte_cell(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int64
            | DataType::UInt64
            | DataType::Float64
            | DataType::Date64
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Interval(IntervalUnit::DayTime)
            | DataType::Decimal64(_, _)
    )
}

fn is_sixteen_byte_cell(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Interval(IntervalUnit::MonthDayNano) | DataType::Decimal128(_, _)
    )
}

fn variable_cell_charge(array: &dyn Array, row_index: usize) -> Result<Option<u64>> {
    if let DataType::FixedSizeBinary(size) = array.data_type() {
        return fixed_size_binary_charge(*size).map(Some);
    }
    string_cell_charge(array, row_index)
        .or(binary_cell_charge(array, row_index))
        .transpose()
}

fn fixed_size_binary_charge(size: i32) -> Result<u64> {
    u64::try_from(size).map_err(|_| CalcFlowError::Internal {
        message: "negative FixedSizeBinary width".into(),
    })
}

fn string_cell_charge(array: &dyn Array, row_index: usize) -> Option<Result<u64>> {
    match array.data_type() {
        DataType::Utf8 => Some(downcast_cell_charge::<StringArray>(
            array, row_index, 4, "Utf8",
        )),
        DataType::LargeUtf8 => Some(downcast_cell_charge::<LargeStringArray>(
            array,
            row_index,
            8,
            "LargeUtf8",
        )),
        DataType::Utf8View => Some(downcast_cell_charge::<StringViewArray>(
            array, row_index, 16, "Utf8View",
        )),
        _ => None,
    }
}

fn binary_cell_charge(array: &dyn Array, row_index: usize) -> Option<Result<u64>> {
    match array.data_type() {
        DataType::Binary => Some(downcast_cell_charge::<BinaryArray>(
            array, row_index, 4, "Binary",
        )),
        DataType::LargeBinary => Some(downcast_cell_charge::<LargeBinaryArray>(
            array,
            row_index,
            8,
            "LargeBinary",
        )),
        DataType::BinaryView => Some(downcast_cell_charge::<BinaryViewArray>(
            array,
            row_index,
            16,
            "BinaryView",
        )),
        _ => None,
    }
}

/// Downcasts one variable-length array kind and charges prefix plus value.
fn downcast_cell_charge<T>(
    array: &dyn Array,
    row_index: usize,
    prefix: u64,
    label: &str,
) -> Result<u64>
where
    T: Array + 'static,
    for<'a> &'a T: CellBytes,
{
    let typed = array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| CalcFlowError::Internal {
            message: format!("{label} array type mismatch"),
        })?;
    prefix_cell_charge(prefix, typed.cell_len(row_index), label)
}

fn prefix_cell_charge(prefix: u64, len: usize, label: &str) -> Result<u64> {
    prefix
        .checked_add(u64::try_from(len).unwrap_or(u64::MAX))
        .ok_or_else(|| CalcFlowError::Internal {
            message: format!("{label} cell charge overflow"),
        })
}

trait CellBytes {
    fn cell_len(self, row_index: usize) -> usize;
}

macro_rules! impl_cell_bytes {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl CellBytes for &$ty {
                fn cell_len(self, row_index: usize) -> usize {
                    self.value(row_index).len()
                }
            }
        )+
    };
}

impl_cell_bytes!(
    StringArray,
    LargeStringArray,
    StringViewArray,
    BinaryArray,
    LargeBinaryArray,
    BinaryViewArray,
);

/// Charges nested and dictionary-encoded cells from the frozen logical table
/// (state-byte accounting v1); every traversal is by logical value, never by
/// buffer capacity.
fn sized_cell_charge(array: &dyn Array, row_index: usize) -> Result<u64> {
    match array.data_type() {
        DataType::List(_) => {
            let typed = list_array::<ListArray>(array)?;
            list_cell_charge(typed.value(row_index).as_ref(), 4)
        }
        DataType::LargeList(_) => {
            let typed = list_array::<LargeListArray>(array)?;
            list_cell_charge(typed.value(row_index).as_ref(), 8)
        }
        DataType::Map(_, _) => {
            let typed = list_array::<MapArray>(array)?;
            let entries = typed.value(row_index);
            let entries: &dyn Array = &entries;
            list_cell_charge(entries, 4)
        }
        DataType::ListView(_) => {
            let typed = list_array::<ListViewArray>(array)?;
            let child = typed.value(row_index);
            let child: &dyn Array = &child;
            list_cell_charge(child, 8)
        }
        DataType::LargeListView(_) => {
            let typed = list_array::<LargeListViewArray>(array)?;
            let child = typed.value(row_index);
            let child: &dyn Array = &child;
            list_cell_charge(child, 16)
        }
        DataType::FixedSizeList(_, _) => {
            let typed = array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| charge_type_mismatch("FixedSizeList"))?;
            let child = typed.value(row_index);
            let child: &dyn Array = &child;
            list_cell_charge(child, 0)
        }
        DataType::Struct(_) => {
            let typed = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| charge_type_mismatch("Struct"))?;
            typed.columns().iter().try_fold(0_u64, |total, column| {
                total
                    .checked_add(logical_cell_charge(column.as_ref(), row_index)?)
                    .ok_or_else(|| charge_overflow("struct cell"))
            })
        }
        DataType::Union(_, _) => {
            let typed = array
                .as_any()
                .downcast_ref::<UnionArray>()
                .ok_or_else(|| charge_type_mismatch("Union"))?;
            let child = typed.child(typed.type_id(row_index));
            let charge = logical_cell_charge(child.as_ref(), typed.value_offset(row_index))?;
            charge
                .checked_add(1)
                .ok_or_else(|| charge_overflow("union cell"))
        }
        DataType::Dictionary(_, _) => {
            let typed = array
                .as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()
                .ok_or_else(|| charge_type_mismatch("Dictionary"))?;
            let index =
                typed
                    .keys()
                    .value(row_index)
                    .try_into()
                    .map_err(|_| CalcFlowError::Internal {
                        message: "dictionary key does not fit usize".into(),
                    })?;
            logical_cell_charge(typed.values().as_ref(), index)
        }
        DataType::RunEndEncoded(..) => {
            let typed = array
                .as_any()
                .downcast_ref::<RunArray<Int32Type>>()
                .ok_or_else(|| charge_type_mismatch("RunEndEncoded"))?;
            logical_cell_charge(typed.values().as_ref(), typed.get_physical_index(row_index))
        }
        _ => Err(unsupported_payload_type(array.data_type())),
    }
}

/// Charges one list-like child slice: prefix plus each child cell.
fn list_cell_charge(child: &(dyn Array + '_), prefix: u64) -> Result<u64> {
    let sum = (0..child.len()).try_fold(0_u64, |total, index| {
        total
            .checked_add(logical_cell_charge(child, index)?)
            .ok_or_else(|| charge_overflow("list child cell"))
    })?;
    prefix
        .checked_add(sum)
        .ok_or_else(|| charge_overflow("list cell"))
}

fn list_array<'a, T>(array: &'a dyn Array) -> Result<&'a T>
where
    &'a T: ArrayAccessor,
    T: 'static + Array,
{
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| charge_type_mismatch("list"))
}

fn charge_type_mismatch(label: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("{label} array type mismatch in logical charge"),
    }
}

fn charge_overflow(label: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("logical {label} charge overflow"),
    }
}

/// Reports the construction-time rejection for payload types outside the
/// frozen state-byte accounting table (spec FR16/D16).
fn unsupported_payload_type(data_type: &DataType) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "stream_join.schema".into(),
        message: format!("unsupported_payload_type: {data_type} has no versioned state charge"),
    }
}

/// Version-1 type-tagged, length-delimited join key encoding (spec FR44).
///
/// Each key column contributes one block: tag byte, timezone length, timezone
/// bytes, value length, and raw value bytes. The encoding is stable and
/// unambiguous across units, timezone metadata, and column order.
fn encode_join_key_v1(
    record: &RecordBatch,
    row_index: usize,
    key_indices: &[usize],
) -> Result<Vec<u8>> {
    encode_join_key_columns_v1(record.columns(), row_index, key_indices)
}

fn encode_join_key_columns_v1(
    columns: &[ArrayRef],
    row_index: usize,
    key_indices: &[usize],
) -> Result<Vec<u8>> {
    #[cfg(test)]
    note_join_work(|work| work.key_encodings += 1);
    let mut encoded = Vec::new();
    for &index in key_indices {
        append_key_block(&mut encoded, columns[index].as_ref(), row_index)?;
    }
    Ok(encoded)
}

fn append_key_block(encoded: &mut Vec<u8>, array: &dyn Array, row_index: usize) -> Result<()> {
    let data_type = array.data_type();
    let tag = key_type_tag(data_type)?;
    let timezone = match data_type {
        DataType::Timestamp(_, Some(tz)) => tz.as_bytes(),
        _ => &[],
    };
    let value = key_value_bytes(array, row_index)?;
    encoded.push(tag);
    encoded.extend_from_slice(
        &u32::try_from(timezone.len())
            .map_err(|_| charge_overflow("key timezone length"))?
            .to_le_bytes(),
    );
    encoded.extend_from_slice(timezone);
    encoded.extend_from_slice(
        &u32::try_from(value.len())
            .map_err(|_| charge_overflow("key value length"))?
            .to_le_bytes(),
    );
    encoded.extend_from_slice(&value);
    Ok(())
}

fn key_type_tag(data_type: &DataType) -> Result<u8> {
    Ok(match data_type {
        DataType::Boolean => 1,
        DataType::Int8 => 2,
        DataType::Int16 => 3,
        DataType::Int32 => 4,
        DataType::Int64 => 5,
        DataType::UInt8 => 6,
        DataType::UInt16 => 7,
        DataType::UInt32 => 8,
        DataType::UInt64 => 9,
        DataType::Utf8 => 10,
        DataType::LargeUtf8 => 11,
        DataType::Date32 => 12,
        DataType::Date64 => 13,
        DataType::Timestamp(TimeUnit::Second, _) => 14,
        DataType::Timestamp(TimeUnit::Millisecond, _) => 15,
        DataType::Timestamp(TimeUnit::Microsecond, _) => 16,
        DataType::Timestamp(TimeUnit::Nanosecond, _) => 17,
        _ => {
            return Err(CalcFlowError::Internal {
                message: format!("join key type {data_type} has no version-1 tag"),
            });
        }
    })
}

fn key_value_bytes(array: &dyn Array, row_index: usize) -> Result<Vec<u8>> {
    let data_type = array.data_type().clone();
    if let DataType::Timestamp(unit, _) = &data_type {
        return timestamp_key_bytes(array, row_index, *unit);
    }
    Ok(match &data_type {
        DataType::Boolean => {
            let typed = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| charge_type_mismatch("Boolean"))?;
            vec![u8::from(typed.value(row_index))]
        }
        DataType::Int8 => {
            let typed = primitive::<Int8Type>(array)?;
            vec![
                u8::try_from(typed.value(row_index)).map_err(|_| CalcFlowError::Internal {
                    message: "int8 key does not fit u8".into(),
                })?,
            ]
        }
        DataType::UInt8 => vec![primitive::<UInt8Type>(array)?.value(row_index)],
        DataType::Int16 => little_endian_key_bytes::<Int16Type>(array, row_index)?,
        DataType::Int32 | DataType::Date32 => {
            little_endian_key_bytes::<Int32Type>(array, row_index)?
        }
        DataType::Int64 | DataType::Date64 => {
            little_endian_key_bytes::<Int64Type>(array, row_index)?
        }
        DataType::UInt16 => little_endian_key_bytes::<UInt16Type>(array, row_index)?,
        DataType::UInt32 => little_endian_key_bytes::<UInt32Type>(array, row_index)?,
        DataType::UInt64 => little_endian_key_bytes::<UInt64Type>(array, row_index)?,
        DataType::Utf8 => array
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| charge_type_mismatch("Utf8 key"))?
            .value(row_index)
            .as_bytes()
            .to_vec(),
        DataType::LargeUtf8 => array
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .ok_or_else(|| charge_type_mismatch("LargeUtf8 key"))?
            .value(row_index)
            .as_bytes()
            .to_vec(),
        _ => {
            return Err(CalcFlowError::Internal {
                message: format!("join key type {data_type} has no version-1 encoding"),
            });
        }
    })
}

/// Little-endian key encoding shared by the fixed-width join key natives.
trait KeyLeBytes {
    fn key_le_bytes(self) -> Vec<u8>;
}

macro_rules! impl_key_le_bytes {
    ($($native:ty),* $(,)?) => {
        $(
            impl KeyLeBytes for $native {
                fn key_le_bytes(self) -> Vec<u8> {
                    self.to_le_bytes().to_vec()
                }
            }
        )*
    };
}

impl_key_le_bytes!(i16, i32, i64, u16, u32, u64);

fn little_endian_key_bytes<T>(array: &dyn Array, row_index: usize) -> Result<Vec<u8>>
where
    T: ArrowPrimitiveType,
    T::Native: KeyLeBytes,
{
    Ok(primitive::<T>(array)?.value(row_index).key_le_bytes())
}

fn primitive<T>(array: &dyn Array) -> Result<&PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
{
    array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .ok_or_else(|| charge_type_mismatch("primitive key"))
}

fn validity_wrapped(value_bytes: u64) -> Result<u64> {
    value_bytes
        .checked_add(1)
        .ok_or_else(|| CalcFlowError::Internal {
            message: "logical cell charge overflow".into(),
        })
}

fn checked_metric(current: u64, delta: u64, operator_id: &str, field: &str) -> Result<u64> {
    current
        .checked_add(delta)
        .ok_or_else(|| counter_overflow(operator_id, field))
}

fn counter_overflow(operator_id: &str, field: &str) -> CalcFlowError {
    operator_reason(
        operator_id,
        crate::StreamingFailureReason::JoinCounterOverflow,
        &format!("{field} counter overflowed"),
    )
}

fn operator_reason(
    operator_id: &str,
    reason_code: crate::StreamingFailureReason,
    message: &str,
) -> CalcFlowError {
    CalcFlowError::OperatorReason {
        node_id: operator_id.into(),
        reason_code,
        message: message.into(),
    }
}

fn operator_error(operator_id: &str, message: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: operator_id.into(),
        message: message.into(),
    }
}

fn default_left_prefix() -> String {
    "left".into()
}

fn default_right_prefix() -> String {
    "right".into()
}

/// Encodes one timestamp key value as its native unit's little-endian bytes.
/// The key block already carries the unit-specific type tag and timezone, so
/// cross-unit unambiguity does not depend on converting to microseconds here.
fn timestamp_key_bytes(array: &dyn Array, row_index: usize, unit: TimeUnit) -> Result<Vec<u8>> {
    let raw = match unit {
        TimeUnit::Second => array
            .as_any()
            .downcast_ref::<TimestampSecondArray>()
            .ok_or_else(|| charge_type_mismatch("timestamp key"))?
            .value(row_index),
        TimeUnit::Millisecond => array
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .ok_or_else(|| charge_type_mismatch("timestamp key"))?
            .value(row_index),
        TimeUnit::Microsecond => array
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .ok_or_else(|| charge_type_mismatch("timestamp key"))?
            .value(row_index),
        TimeUnit::Nanosecond => array
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .ok_or_else(|| charge_type_mismatch("timestamp key"))?
            .value(row_index),
    };
    Ok(raw.to_le_bytes().to_vec())
}
