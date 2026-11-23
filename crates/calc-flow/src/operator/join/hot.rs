use super::{HotProjection, JoinTimeBounds, StreamJoinOperator};
use crate::Result;
use datafusion::arrow::{
    array::{Array, ArrayRef, AsArray, Float64Array},
    datatypes::{DataType, Float64Type, Schema, TimeUnit},
};
use std::sync::Arc;

#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum HotKind {
    Equi,
    Band,
}

pub(super) fn projection_kind(operator: &StreamJoinOperator) -> Option<HotKind> {
    let kind = HotKind::from_bounds(operator.spec.bounds)?;
    let schema = operator.output_ports[0]
        .schema()
        .expect("compiled Join output schema");
    if schema.fields().is_empty() || !kind.accepts(operator.hot_projection) {
        return None;
    }
    if operator.spec.left_keys() != ["symbol"]
        || operator.spec.right_keys() != ["symbol"]
        || operator.spec.left_event_time() != "event_time"
        || operator.spec.right_event_time() != "event_time"
    {
        return None;
    }
    [0, 1]
        .into_iter()
        .all(|side| kind.accepts_schema(operator.input_schema(side)))
        .then_some(kind)
}

impl HotKind {
    fn from_bounds(bounds: JoinTimeBounds) -> Option<Self> {
        if bounds.after_micros == 0 {
            Some(Self::Equi)
        } else if bounds.before_micros == bounds.after_micros
            && bounds.before_micros.is_multiple_of(1_000_000)
            && bounds.before_micros <= 60_000_000
        {
            Some(Self::Band)
        } else {
            None
        }
    }

    fn accepts(self, projection: Option<HotProjection>) -> bool {
        projection
            .is_none_or(|projection| projection.right_sequence.is_some() == (self == Self::Band))
    }

    fn accepts_schema(self, schema: &Schema) -> bool {
        let field = |name| {
            schema
                .column_with_name(name)
                .map(|(_, field)| field.data_type())
        };
        field("symbol") == Some(&DataType::Utf8)
            && field("sequence") == Some(&DataType::UInt64)
            && matches!(
                field("event_time"),
                Some(DataType::Timestamp(TimeUnit::Microsecond, _))
            )
            && self.accepts_prices(field("price"), field("factor"))
    }

    fn accepts_prices(self, price: Option<&DataType>, factor: Option<&DataType>) -> bool {
        let has_operand = price.is_some() || (self == Self::Equi && factor.is_some());
        has_operand
            && [price, factor]
                .into_iter()
                .flatten()
                .all(|ty| ty == &DataType::Float64)
    }
}

impl HotProjection {
    pub(super) fn gather(
        self,
        column: impl Fn(usize, &DataType) -> Result<ArrayRef>,
    ) -> Result<Vec<ArrayRef>> {
        let mut columns = Vec::with_capacity(2 + usize::from(self.right_sequence.is_some()));
        columns.push(column(self.sequence, &DataType::UInt64)?);
        if let Some(sequence) = self.right_sequence {
            columns.push(column(sequence, &DataType::UInt64)?);
        }
        let left = column(self.left_price, &DataType::Float64)?;
        let right = column(self.right_price, &DataType::Float64)?;
        let values = Self::products(
            left.as_primitive::<Float64Type>(),
            right.as_primitive::<Float64Type>(),
        );
        columns.push(Arc::new(values));
        Ok(columns)
    }

    fn products(left: &Float64Array, right: &Float64Array) -> Float64Array {
        let pairs = || left.values().iter().zip(right.values());
        // Preserve the nullable kernel's choice of payload when both operands are NaN.
        if left.null_count() == 0
            && right.null_count() == 0
            && !pairs().any(|(a, b)| a.is_nan() && b.is_nan())
        {
            return Float64Array::from_iter_values(pairs().map(|(a, b)| a * b));
        }
        left.iter()
            .zip(right)
            .map(|(left, right)| left.zip(right).map(|(a, b)| a * b))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    mod projected_gather;

    use super::*;
    use crate::operator::join::JoinSide;
    use crate::{
        Batch, BatchMetadata, CancellationToken, EdgeCollector, JoinStateLimits, JoinTimeBounds,
        JsonMap, OperatorMetadata, StreamJobContext, StreamJoinSpec, StreamOperator,
        StreamOperatorContext,
    };
    use datafusion::arrow::{
        array::{Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
        datatypes::{Field, Schema},
        record_batch::RecordBatch,
    };
    use std::{collections::BTreeMap, time::Duration};

    fn quote_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
        ]))
    }

    fn dimension_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("symbol", DataType::Utf8, false),
            Field::new("factor", DataType::Float64, false),
            Field::new("sequence", DataType::UInt64, false),
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
        ]))
    }

    fn spec() -> StreamJoinSpec {
        StreamJoinSpec::inner(
            ["symbol"],
            ["symbol"],
            "event_time",
            "event_time",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::ZERO).unwrap(),
            JoinStateLimits::new(100, 1_000_000, 100).unwrap(),
        )
        .unwrap()
        .with_prefixes("quote", "dimension")
        .unwrap()
    }

    fn static_inputs() -> (RecordBatch, RecordBatch) {
        let left = RecordBatch::try_new(
            quote_schema(),
            vec![
                Arc::new(TimestampMicrosecondArray::from(vec![0, 1_000_000]).with_timezone("UTC")),
                Arc::new(UInt64Array::from(vec![0, 1])),
                Arc::new(StringArray::from(vec!["S000", "S000"])),
                Arc::new(Float64Array::from(vec![2.0, 3.0])),
            ],
        )
        .unwrap();
        let right = RecordBatch::try_new(
            dimension_schema(),
            vec![
                Arc::new(StringArray::from(vec!["S000"])),
                Arc::new(Float64Array::from(vec![2.0])),
                Arc::new(UInt64Array::from(vec![0])),
                Arc::new(TimestampMicrosecondArray::from(vec![1_000_000]).with_timezone("UTC")),
            ],
        )
        .unwrap();
        (left, right)
    }

    fn static_left_history() -> RecordBatch {
        let (left, _) = static_inputs();
        RecordBatch::try_new(
            left.schema(),
            vec![
                Arc::new(
                    TimestampMicrosecondArray::from(vec![0, 301_000_000]).with_timezone("UTC"),
                ),
                left.column(1).clone(),
                left.column(2).clone(),
                left.column(3).clone(),
            ],
        )
        .unwrap()
    }

    fn static_right_history() -> RecordBatch {
        RecordBatch::try_new(
            dimension_schema(),
            vec![
                Arc::new(StringArray::from(vec!["S000", "S000"])),
                Arc::new(Float64Array::from(vec![2.0, 2.0])),
                Arc::new(UInt64Array::from(vec![0, 1])),
                Arc::new(TimestampMicrosecondArray::from(vec![0, 1_000_000]).with_timezone("UTC")),
            ],
        )
        .unwrap()
    }

    fn static_operator(fused: bool) -> StreamJoinOperator {
        let mut operator =
            StreamJoinOperator::new("hot", quote_schema(), dimension_schema(), spec()).unwrap();
        if fused {
            operator
                .set_hot_projection(
                    HotProjection {
                        right_sequence: None,
                        sequence: 1,
                        left_price: 3,
                        right_price: 5,
                    },
                    Arc::new(Schema::new(vec![
                        Field::new("sequence", DataType::UInt64, false),
                        Field::new("value", DataType::Float64, false),
                    ])),
                )
                .unwrap();
        }
        operator.enable_native_checkpoint().unwrap();
        operator
    }

    fn static_pairs(messages: &[crate::StreamMessage], fused: bool) -> Vec<(u64, f64)> {
        prefixed_pairs(messages, fused, "quote", "dimension", "factor")
    }

    fn band_pairs(messages: &[crate::StreamMessage], fused: bool) -> Vec<(u64, f64)> {
        prefixed_pairs(messages, fused, "left", "right", "price")
    }

    fn prefixed_pairs(
        messages: &[crate::StreamMessage],
        fused: bool,
        left_prefix: &str,
        right_prefix: &str,
        right_value: &str,
    ) -> Vec<(u64, f64)> {
        let sequence_name = if fused {
            "sequence".into()
        } else {
            format!("{left_prefix}__sequence")
        };
        let left_value_name = if fused {
            "value".into()
        } else {
            format!("{left_prefix}__price")
        };
        let right_value_name = format!("{right_prefix}__{right_value}");
        let mut pairs = Vec::new();
        for message in messages {
            let batch = message.as_data().unwrap();
            let record = &batch.table_payload().unwrap().batches()[0];
            let sequence = record
                .column(record.schema().index_of(&sequence_name).unwrap())
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            let values = if fused {
                record
                    .column(record.schema().index_of("value").unwrap())
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .clone()
            } else {
                let left = record
                    .column(record.schema().index_of(&left_value_name).unwrap())
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                let right = record
                    .column(record.schema().index_of(&right_value_name).unwrap())
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                left.iter()
                    .zip(right.iter())
                    .map(|(left, right)| Some(left.unwrap() * right.unwrap()))
                    .collect::<Float64Array>()
            };
            pairs.extend(
                sequence
                    .iter()
                    .zip(values.iter())
                    .map(|(sequence, value)| (sequence.unwrap(), value.unwrap())),
            );
        }
        pairs
    }

    fn hot_context(job: &StreamJobContext) -> StreamOperatorContext<'_> {
        StreamOperatorContext::new(job, "hot", None)
    }

    struct NoopLateMetrics;

    impl crate::operator::LateMetricSink for NoopLateMetrics {
        fn record(&self, _delta: crate::operator::LateMetricDelta) -> Result<()> {
            Ok(())
        }
    }

    fn hot_ended_context(job: &StreamJobContext) -> StreamOperatorContext<'_> {
        let progress = crate::IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                crate::IngressProgress::new(
                    crate::IngressState::Ended,
                    Some(crate::EventTime::from_micros(0)),
                ),
            ),
            (
                "right".into(),
                crate::IngressProgress::new(
                    crate::IngressState::Ended,
                    Some(crate::EventTime::from_micros(0)),
                ),
            ),
        ]));
        StreamOperatorContext::for_task(
            job,
            "hot",
            None,
            progress,
            crate::EdgeBudget::default(),
            Arc::new(NoopLateMetrics),
        )
    }

    async fn native_snapshot(operator: &mut StreamJoinOperator) -> crate::OperatorStateSnapshot {
        let name = operator.name.clone();
        let capture = operator
            .native_checkpoint(crate::Epoch::new(1).unwrap())
            .unwrap()
            .expect("hot Join native capture");
        let temporary = tempfile::tempdir().unwrap();
        let mut store = crate::state::v4::V4Store::new(temporary.path().join("lineage"));
        let entry = crate::state::v4::native::persist_native(
            &mut store,
            &name,
            capture,
            1,
            BTreeMap::new(),
            None,
        )
        .await
        .unwrap();
        let mut manifest = crate::state::v4::V4Manifest::from_bytes(
            include_str!("../../../tests/fixtures/checkpoint-v4/manifest-v4.json").as_bytes(),
        )
        .unwrap();
        manifest.operators = BTreeMap::from([(name.clone(), entry)]);
        manifest.epoch = 1;
        manifest.manifest_fingerprint = crate::state::v4::manifest_fingerprint(&manifest).unwrap();
        store.publish_manifest(&manifest).unwrap();
        crate::state::v4::native::load_native(&mut store, &manifest, &name).unwrap()
    }

    #[tokio::test]
    async fn hot_multi_accumulates_same_side_batches_before_joining() {
        for fused in [false, true] {
            let mut operator = static_operator(fused);
            let left = static_left_history();
            let job = StreamJobContext::new(
                1,
                "fingerprint",
                JsonMap::new(),
                None,
                CancellationToken::new(),
            )
            .with_checkpointing(false);
            let context = hot_context(&job);
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            for record in [left.slice(0, 1), left.slice(1, 1)] {
                operator
                    .process_data(
                        "left",
                        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                        &context,
                        &mut collector,
                    )
                    .await
                    .unwrap();
            }
            operator
                .process_data(
                    "right",
                    Batch::table(vec![static_right_history()], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
            assert_eq!(
                static_pairs(&collector.drain("output"), fused),
                [(0, 4.0), (1, 6.0)],
                "fused={fused}"
            );
            assert_eq!(operator.state.metrics.emitted_match_rows, 2);
        }
    }

    #[tokio::test]
    async fn hot_multi_continues_after_the_first_pair_without_replaying_matches() {
        for fused in [false, true] {
            let mut operator = static_operator(fused);
            let left = static_left_history();
            let job = StreamJobContext::new(
                1,
                "fingerprint",
                JsonMap::new(),
                None,
                CancellationToken::new(),
            )
            .with_checkpointing(false);
            let context = hot_context(&job);
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            for (side, record) in [
                ("left", left.slice(0, 1)),
                ("right", static_right_history()),
                ("left", left.slice(1, 1)),
            ] {
                operator
                    .process_data(
                        side,
                        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                        &context,
                        &mut collector,
                    )
                    .await
                    .unwrap();
            }
            let messages = collector.drain("output");
            assert_eq!(
                static_pairs(&messages, fused),
                [(0, 4.0), (1, 6.0)],
                "fused={fused}"
            );
            assert_eq!(messages[0].as_data().unwrap().metadata().sequence(), 0);
            assert_eq!(messages[1].as_data().unwrap().metadata().sequence(), 1);
            assert_eq!(operator.state.metrics.emitted_match_rows, 2);
            assert_eq!(operator.state.next_left_row_id, 2);
        }
    }

    #[tokio::test]
    async fn hot_multi_static_enforces_the_match_limit_before_emitting() {
        let mut operator = static_operator(false);
        operator.spec.limits.max_matches_per_input_batch = 1;
        let job = StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
        .with_checkpointing(false);
        let context = hot_context(&job);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        for (side, record) in [
            ("left", static_left_history()),
            ("right", static_right_history()),
        ] {
            let result = operator
                .process_data(
                    side,
                    Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await;
            if side == "left" {
                result.unwrap();
            } else {
                let error = result.expect_err("two static matches must exceed the cap of one");
                assert!(
                    error
                        .to_string()
                        .contains("input batch match limit exceeded")
                );
            }
        }
        assert!(collector.drain("output").is_empty());
        assert_eq!(operator.state.metrics.match_limit_failures, 1);
    }

    #[tokio::test]
    async fn hot_multi_native_checkpoint_preserves_a_pending_side() {
        for fused in [false, true] {
            let mut original = static_operator(fused);
            let left = static_left_history();
            let job = StreamJobContext::new(
                1,
                "fingerprint",
                JsonMap::new(),
                None,
                CancellationToken::new(),
            )
            .with_checkpointing(false);
            let context = hot_context(&job);
            let mut original_output = EdgeCollector::new(original.output_ports().to_vec());
            original
                .process_data(
                    "left",
                    Batch::table(vec![left], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut original_output,
                )
                .await
                .unwrap();
            assert!(original_output.drain("output").is_empty());
            assert_eq!(original.state.next_left_row_id, 2);
            let snapshot = native_snapshot(&mut original).await;
            let mut restored = static_operator(fused);
            restored.restore_native(&snapshot).unwrap();
            let mut restored_output = EdgeCollector::new(restored.output_ports().to_vec());
            restored
                .process_data(
                    "right",
                    Batch::table(vec![static_right_history()], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut restored_output,
                )
                .await
                .unwrap();
            let messages = restored_output.drain("output");
            assert_eq!(
                static_pairs(&messages, fused),
                [(0, 4.0), (1, 6.0)],
                "fused={fused}"
            );
            assert_eq!(messages[0].as_data().unwrap().metadata().sequence(), 0);
            assert_eq!(restored.state.metrics.emitted_match_rows, 2);
            assert_eq!(restored.state.next_right_row_id, 2);
        }
    }

    #[tokio::test]
    async fn hot_multi_native_checkpoint_restores_history_and_output_sequence() {
        for fused in [false, true] {
            let mut original = static_operator(fused);
            let left = static_left_history();
            let job = StreamJobContext::new(
                1,
                "fingerprint",
                JsonMap::new(),
                None,
                CancellationToken::new(),
            )
            .with_checkpointing(false);
            let context = hot_context(&job);
            let mut original_output = EdgeCollector::new(original.output_ports().to_vec());
            for (side, record) in [
                ("left", left.slice(0, 1)),
                ("right", static_right_history()),
            ] {
                original
                    .process_data(
                        side,
                        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                        &context,
                        &mut original_output,
                    )
                    .await
                    .unwrap();
            }
            assert_eq!(
                static_pairs(&original_output.drain("output"), fused),
                [(0, 4.0)]
            );
            let snapshot = native_snapshot(&mut original).await;
            let mut restored = static_operator(fused);
            restored.restore_native(&snapshot).unwrap();
            let mut restored_output = EdgeCollector::new(restored.output_ports().to_vec());
            restored
                .process_data(
                    "left",
                    Batch::table(vec![left.slice(1, 1)], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut restored_output,
                )
                .await
                .unwrap();
            let messages = restored_output.drain("output");
            assert_eq!(static_pairs(&messages, fused), [(1, 6.0)], "fused={fused}");
            assert_eq!(messages[0].as_data().unwrap().metadata().sequence(), 1);
            assert_eq!(restored.state.metrics.emitted_match_rows, 2);
            assert_eq!(restored.state.next_left_row_id, 2);
        }
    }

    fn large_static_inputs(rows: usize, dimensions: usize) -> (RecordBatch, RecordBatch) {
        let symbols = (0..rows)
            .map(|row| format!("S{:03}", row % dimensions))
            .collect::<Vec<_>>();
        let left = RecordBatch::try_new(
            quote_schema(),
            vec![
                Arc::new(
                    TimestampMicrosecondArray::from_iter_values(std::iter::repeat_n(0, rows))
                        .with_timezone("UTC"),
                ),
                Arc::new(UInt64Array::from_iter_values(
                    (0..rows).map(|row| u64::try_from(row).unwrap()),
                )),
                Arc::new(StringArray::from_iter_values(
                    symbols.iter().map(String::as_str),
                )),
                Arc::new(Float64Array::from_iter_values(std::iter::repeat_n(
                    2.0, rows,
                ))),
            ],
        )
        .unwrap();
        let right = RecordBatch::try_new(
            dimension_schema(),
            vec![
                Arc::new(StringArray::from_iter_values(
                    symbols[..dimensions].iter().map(String::as_str),
                )),
                Arc::new(Float64Array::from_iter_values(std::iter::repeat_n(
                    2.0, dimensions,
                ))),
                Arc::new(UInt64Array::from_iter_values(
                    (0..dimensions).map(|row| u64::try_from(row).unwrap()),
                )),
                Arc::new(
                    TimestampMicrosecondArray::from_iter_values(std::iter::repeat_n(0, dimensions))
                        .with_timezone("UTC"),
                ),
            ],
        )
        .unwrap();
        (left, right)
    }

    #[tokio::test]
    async fn hot_multi_native_checkpoint_accepts_large_static_history() {
        let rows = 1_000_000usize;
        let dimensions = 64usize;
        let capacity = u64::try_from(dimensions + rows).unwrap();
        let spec = StreamJoinSpec::inner(
            ["symbol"],
            ["symbol"],
            "event_time",
            "event_time",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::ZERO).unwrap(),
            JoinStateLimits::new(capacity, 1 << 30, capacity).unwrap(),
        )
        .unwrap()
        .with_prefixes("quote", "dimension")
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("hot", quote_schema(), dimension_schema(), spec).unwrap();
        operator
            .set_hot_projection(
                HotProjection {
                    right_sequence: None,
                    sequence: 1,
                    left_price: 3,
                    right_price: 5,
                },
                Arc::new(Schema::new(vec![
                    Field::new("sequence", DataType::UInt64, false),
                    Field::new("value", DataType::Float64, false),
                ])),
            )
            .unwrap();
        operator.enable_native_checkpoint().unwrap();
        let (left, right) = large_static_inputs(rows, dimensions);
        let job = StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = hot_context(&job);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        for (side, record) in [("left", left), ("right", right)] {
            operator
                .process_data(
                    side,
                    Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
        }
        assert_eq!(
            collector
                .drain("output")
                .iter()
                .map(|message| message.as_data().unwrap().num_rows())
                .sum::<usize>(),
            rows
        );
        let ended = hot_ended_context(&job);
        for side in ["left", "right"] {
            operator.on_ingress_progress(side, &ended).await.unwrap();
        }
        operator.on_end(&ended, &mut collector).await.unwrap();
        assert!(operator.state.ended);
        drop(native_snapshot(&mut operator).await);
        let core = operator.run_core.as_ref().unwrap();
        assert_eq!(core.rows(JoinSide::Left), 0);
        assert_eq!(core.rows(JoinSide::Right), 0);
        assert_eq!(operator.state.metrics.left.retained_rows, 0);
        assert_eq!(operator.state.metrics.right.retained_rows, 0);
    }

    #[tokio::test]
    async fn hot_join_emits_the_complete_downstream_schema() {
        for (projection, fused) in [(None, false), (Some(vec![1, 3, 5]), false), (None, true)] {
            let mut operator =
                StreamJoinOperator::new("hot", quote_schema(), dimension_schema(), spec()).unwrap();
            if fused {
                operator
                    .set_hot_projection(
                        HotProjection {
                            right_sequence: None,
                            sequence: 1,
                            left_price: 3,
                            right_price: 5,
                        },
                        Arc::new(Schema::new(vec![
                            Field::new("sequence", DataType::UInt64, false),
                            Field::new("value", DataType::Float64, false),
                        ])),
                    )
                    .unwrap();
            } else if let Some(columns) = &projection {
                operator.set_output_projection(columns.clone()).unwrap();
            }
            operator.enable_run_core();
            let (left, right) = static_inputs();
            let job = StreamJobContext::new(
                1,
                "fingerprint",
                JsonMap::new(),
                None,
                CancellationToken::new(),
            )
            .with_checkpointing(false);
            let context = StreamOperatorContext::new(&job, "hot", None);
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            operator
                .process_data(
                    "left",
                    Batch::table(vec![left], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
            operator
                .process_data(
                    "right",
                    Batch::table(vec![right], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();

            let messages = collector.drain("output");
            assert_eq!(messages.len(), 1);
            let output = messages[0].as_data().unwrap();
            assert_eq!(output.num_rows(), 1);
            let record = &output.table_payload().unwrap().batches()[0];
            assert_eq!(
                record.schema().fields().len(),
                if fused {
                    2
                } else {
                    projection.as_ref().map_or(8, Vec::len)
                }
            );
            if fused {
                let sequence = record
                    .column(record.schema().index_of("sequence").unwrap())
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .unwrap();
                assert_eq!(sequence.values(), &[1]);
                let value = record
                    .column(record.schema().index_of("value").unwrap())
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                assert_eq!(value.values(), &[6.0]);
            } else {
                let price = record
                    .column(record.schema().index_of("quote__price").unwrap())
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                assert_eq!(price.values(), &[3.0]);
                let factor = record
                    .column(record.schema().index_of("dimension__factor").unwrap())
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                assert_eq!(factor.values(), &[2.0]);
            }
            assert_eq!(operator.state.metrics.emitted_match_rows, 1);
            let core = operator.run_core.as_ref().unwrap();
            assert!(!core.hot_enabled());
            assert_eq!(core.rows(JoinSide::Left), 2);
            assert_eq!(core.rows(JoinSide::Right), 1);
        }
    }

    #[tokio::test]
    async fn hot_static_right_input_uses_ingress_order() {
        let mut operator = static_operator(false);
        let (left, _) = static_inputs();
        let left = RecordBatch::try_new(
            left.schema(),
            vec![
                Arc::new(TimestampMicrosecondArray::from(vec![1_000_000; 2]).with_timezone("UTC")),
                left.column(1).clone(),
                left.column(2).clone(),
                left.column(3).clone(),
            ],
        )
        .unwrap();
        let right = static_right_history();
        let job = StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
        .with_checkpointing(false);
        let context = hot_context(&job);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        for (side, record) in [("left", left), ("right", right)] {
            operator
                .process_data(
                    side,
                    Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
        }
        assert_eq!(
            static_pairs(&collector.drain("output"), false),
            [(0, 4.0), (1, 6.0), (0, 4.0), (1, 6.0)]
        );
        assert!(!operator.run_core.as_ref().unwrap().hot_enabled());
    }

    fn band_spec() -> StreamJoinSpec {
        StreamJoinSpec::inner(
            ["symbol"],
            ["symbol"],
            "event_time",
            "event_time",
            JoinTimeBounds::new(Duration::from_secs(5), Duration::from_secs(5)).unwrap(),
            JoinStateLimits::new(100, 1_000_000, 100).unwrap(),
        )
        .unwrap()
        .with_prefixes("left", "right")
        .unwrap()
    }

    fn band_history(prices: [f64; 2]) -> RecordBatch {
        RecordBatch::try_new(
            quote_schema(),
            vec![
                Arc::new(TimestampMicrosecondArray::from(vec![0, 10_000_000]).with_timezone("UTC")),
                Arc::new(UInt64Array::from(vec![0, 1])),
                Arc::new(StringArray::from(vec!["S000", "S000"])),
                Arc::new(Float64Array::from(prices.to_vec())),
            ],
        )
        .unwrap()
    }

    fn band_projection() -> HotProjection {
        HotProjection {
            sequence: 1,
            right_sequence: Some(5),
            left_price: 3,
            right_price: 7,
        }
    }

    fn band_operator(fused: bool) -> StreamJoinOperator {
        let mut operator =
            StreamJoinOperator::new("hot-band", quote_schema(), quote_schema(), band_spec())
                .unwrap();
        if fused {
            operator
                .set_hot_projection(
                    band_projection(),
                    Arc::new(Schema::new(vec![
                        Field::new("sequence", DataType::UInt64, false),
                        Field::new("right_sequence", DataType::UInt64, false),
                        Field::new("value", DataType::Float64, false),
                    ])),
                )
                .unwrap();
        }
        operator.enable_native_checkpoint().unwrap();
        operator
    }

    #[test]
    fn band_projection_admission_does_not_allocate_a_query() {
        let operator = band_operator(false);
        let cost = allocation_counter::measure(|| {
            assert!(operator.hot_projection_supported());
        });
        println!(
            "band projection eligibility allocated {} bytes",
            cost.bytes_total
        );
        assert!(cost.bytes_total <= 128, "{cost:?}");
    }

    #[tokio::test]
    async fn band_matches_subsecond_offsets_and_rejects_excess_pairs_before_delivery() {
        let job = StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
        .with_checkpointing(false);
        let context = hot_context(&job);
        let left = band_history([2.0, 3.0]);
        let mut columns = band_history([5.0, 7.0]).columns().to_vec();
        columns[0] = Arc::new(
            TimestampMicrosecondArray::from(vec![500_000, 10_500_000]).with_timezone("UTC"),
        );
        let right = RecordBatch::try_new(left.schema(), columns).unwrap();
        for limit in [1, 2] {
            let mut spec = band_spec();
            spec.limits.max_matches_per_input_batch = limit;
            let mut operator =
                StreamJoinOperator::new("subsecond", quote_schema(), quote_schema(), spec).unwrap();
            operator.enable_run_core();
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            operator
                .process_data(
                    "right",
                    Batch::table(vec![right.clone()], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
            let result = operator
                .process_data(
                    "left",
                    Batch::table(vec![left.clone()], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await;
            let emitted = collector.drain("output");
            if limit == 1 {
                assert!(
                    result
                        .unwrap_err()
                        .to_string()
                        .contains("input batch match limit exceeded")
                );
                assert!(emitted.is_empty());
                assert_eq!(operator.state.next_left_row_id, 0);
                assert_eq!(operator.state.metrics.match_limit_failures, 1);
            } else {
                result.unwrap();
                assert_eq!(band_pairs(&emitted, false), [(0, 10.0), (1, 21.0)]);
            }
        }
    }

    #[tokio::test]
    async fn hot_multi_band_native_checkpoint_restores_history_and_output_sequence() {
        for fused in [false, true] {
            let mut original = band_operator(fused);
            let left = band_history([2.0, 3.0]);
            let right = band_history([2.0, 2.0]);
            let job = StreamJobContext::new(
                1,
                "fingerprint",
                JsonMap::new(),
                None,
                CancellationToken::new(),
            )
            .with_checkpointing(false);
            let context = StreamOperatorContext::new(&job, "hot-band", None);
            let mut original_output = EdgeCollector::new(original.output_ports().to_vec());
            for (side, record) in [("left", left.slice(0, 1)), ("right", right)] {
                original
                    .process_data(
                        side,
                        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                        &context,
                        &mut original_output,
                    )
                    .await
                    .unwrap();
            }
            assert_eq!(
                band_pairs(&original_output.drain("output"), fused),
                [(0, 4.0)]
            );
            let snapshot = native_snapshot(&mut original).await;
            let mut restored = band_operator(fused);
            restored.restore_native(&snapshot).unwrap();
            let mut restored_output = EdgeCollector::new(restored.output_ports().to_vec());
            restored
                .process_data(
                    "left",
                    Batch::table(vec![left.slice(1, 1)], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut restored_output,
                )
                .await
                .unwrap();
            let messages = restored_output.drain("output");
            assert_eq!(band_pairs(&messages, fused), [(1, 6.0)], "fused={fused}");
            assert_eq!(messages[0].as_data().unwrap().metadata().sequence(), 1);
            assert_eq!(restored.state.metrics.emitted_match_rows, 2);
            assert_eq!(restored.state.next_left_row_id, 2);
        }
    }

    #[tokio::test]
    async fn hot_band_emits_requested_input_columns() {
        for (projection, fused) in [(None, false), (Some(vec![1, 3, 5, 7]), false), (None, true)] {
            let mut operator =
                StreamJoinOperator::new("hot-band", quote_schema(), quote_schema(), band_spec())
                    .unwrap();
            if fused {
                operator
                    .set_hot_projection(
                        band_projection(),
                        Arc::new(Schema::new(vec![
                            Field::new("sequence", DataType::UInt64, false),
                            Field::new("right_sequence", DataType::UInt64, false),
                            Field::new("value", DataType::Float64, false),
                        ])),
                    )
                    .unwrap();
            } else if let Some(columns) = &projection {
                operator.set_output_projection(columns.clone()).unwrap();
            }
            operator.enable_run_core();
            let record = RecordBatch::try_new(
                quote_schema(),
                vec![
                    Arc::new(
                        TimestampMicrosecondArray::from(vec![0, 1_000_000]).with_timezone("UTC"),
                    ),
                    Arc::new(UInt64Array::from(vec![0, 1])),
                    Arc::new(StringArray::from(vec!["S000", "S000"])),
                    Arc::new(Float64Array::from(vec![2.0, 3.0])),
                ],
            )
            .unwrap();
            let job = StreamJobContext::new(
                1,
                "fingerprint",
                JsonMap::new(),
                None,
                CancellationToken::new(),
            )
            .with_checkpointing(false);
            let context = StreamOperatorContext::new(&job, "hot-band", None);
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            for side in ["left", "right"] {
                operator
                    .process_data(
                        side,
                        Batch::table(vec![record.clone()], BatchMetadata::default()).unwrap(),
                        &context,
                        &mut collector,
                    )
                    .await
                    .unwrap();
            }

            let messages = collector.drain("output");
            assert_eq!(messages.len(), 1);
            let output = messages[0].as_data().unwrap();
            assert_eq!(output.num_rows(), 4);
            let record = &output.table_payload().unwrap().batches()[0];
            assert_eq!(
                record.schema().fields().len(),
                if fused {
                    3
                } else {
                    projection.as_ref().map_or(8, Vec::len)
                }
            );
            if fused {
                let value = record
                    .column(record.schema().index_of("value").unwrap())
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                assert_eq!(value.values(), &[4.0, 6.0, 6.0, 9.0]);
            } else {
                let left = record
                    .column(record.schema().index_of("left__price").unwrap())
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                assert_eq!(left.values(), &[2.0, 3.0, 2.0, 3.0]);
                let right = record
                    .column(record.schema().index_of("right__price").unwrap())
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                assert_eq!(right.values(), &[2.0, 2.0, 3.0, 3.0]);
            }
            let right_sequence_name = if fused {
                "right_sequence"
            } else {
                "right__sequence"
            };
            let right_sequence = record
                .column(record.schema().index_of(right_sequence_name).unwrap())
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            assert_eq!(right_sequence.values(), &[0, 0, 1, 1]);
            assert_eq!(operator.state.metrics.emitted_match_rows, 4);
            assert!(!operator.run_core.as_ref().unwrap().hot_enabled());
        }
    }

    #[tokio::test]
    async fn hot_band_uses_large_output_batches() {
        let rows = 70_000usize;
        let mut declaration = band_spec();
        declaration.limits.max_state_rows_per_side = 2 * rows as u64;
        declaration.limits.max_state_bytes_per_side = 64 << 20;
        let record = RecordBatch::try_new(
            quote_schema(),
            vec![
                Arc::new(
                    TimestampMicrosecondArray::from_iter_values(
                        (0..rows).map(|row| i64::try_from((row % 2) * 1_000_000).unwrap()),
                    )
                    .with_timezone("UTC"),
                ),
                Arc::new(UInt64Array::from_iter_values(
                    (0..rows).map(|row| u64::try_from(row).unwrap()),
                )),
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    "S000", rows,
                ))),
                Arc::new(Float64Array::from_iter_values(std::iter::repeat_n(
                    2.0, rows,
                ))),
            ],
        )
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("hot-band", quote_schema(), quote_schema(), declaration)
                .unwrap();
        operator.spec.limits.max_matches_per_input_batch = 140_000;
        operator
            .set_hot_projection(
                band_projection(),
                Arc::new(Schema::new(vec![
                    Field::new("sequence", DataType::UInt64, false),
                    Field::new("right_sequence", DataType::UInt64, false),
                    Field::new("value", DataType::Float64, false),
                ])),
            )
            .unwrap();
        operator.enable_run_core();
        let job = StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
        .with_checkpointing(false);
        let context = StreamOperatorContext::new(&job, "hot-band", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        for (side, input) in [("left", record.clone()), ("right", record.slice(0, 2))] {
            operator
                .process_data(
                    side,
                    Batch::table(vec![input], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
        }
        let messages = collector.drain("output");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].as_data().unwrap().num_rows(), 64_000);
        assert_eq!(
            messages
                .iter()
                .map(|message| message.as_data().unwrap().num_rows())
                .sum::<usize>(),
            2 * rows
        );
    }
    #[cfg(test)]
    fn hot_progress_context(
        job: &StreamJobContext,
        right_watermark: i64,
        budget: crate::EdgeBudget,
    ) -> StreamOperatorContext<'_> {
        let progress = crate::IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                crate::IngressProgress::new(
                    crate::IngressState::Active,
                    Some(crate::EventTime::from_micros(right_watermark)),
                ),
            ),
            (
                "right".into(),
                crate::IngressProgress::new(
                    crate::IngressState::Active,
                    Some(crate::EventTime::from_micros(right_watermark)),
                ),
            ),
        ]));
        StreamOperatorContext::for_task(
            job,
            "hot",
            None,
            progress,
            budget,
            Arc::new(NoopLateMetrics),
        )
    }

    #[cfg(test)]
    mod admission_tests {
        use super::*;

        fn operator() -> StreamJoinOperator {
            static_operator(false)
        }

        fn job() -> StreamJobContext {
            StreamJobContext::new(
                1,
                "fingerprint",
                JsonMap::new(),
                None,
                CancellationToken::new(),
            )
            .with_checkpointing(false)
        }

        #[tokio::test]
        async fn hot_admission_drops_late_and_null_rows_with_metrics() {
            let schema = Schema::new(
                quote_schema()
                    .fields()
                    .iter()
                    .map(|field| {
                        if field.name() == "symbol" {
                            field.as_ref().clone().with_nullable(true)
                        } else {
                            field.as_ref().clone()
                        }
                    })
                    .collect::<Vec<_>>(),
            );
            let mut operator = StreamJoinOperator::new(
                "hot",
                Arc::new(schema.clone()),
                dimension_schema(),
                spec(),
            )
            .unwrap();
            operator.enable_native_checkpoint().unwrap();
            let left = RecordBatch::try_new(
                Arc::new(schema),
                vec![
                    Arc::new(
                        TimestampMicrosecondArray::from(vec![-1, 1_000_000]).with_timezone("UTC"),
                    ),
                    Arc::new(UInt64Array::from(vec![0, 1])),
                    Arc::new(StringArray::from(vec![Some("S000"), None])),
                    Arc::new(Float64Array::from(vec![2.0, 3.0])),
                ],
            )
            .unwrap();
            let (_, right) = static_inputs();
            let job = job();
            let context = hot_progress_context(&job, 0, crate::EdgeBudget::default());
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            operator
                .process_data(
                    "left",
                    Batch::table(vec![left], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
            operator
                .process_data(
                    "right",
                    Batch::table(vec![right], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
            assert!(collector.drain("output").is_empty());
            assert_eq!(operator.state.metrics.left.late_rows, 1);
            assert_eq!(operator.state.metrics.left.null_key_rows, 1);
        }

        #[tokio::test]
        async fn hot_no_matches_emit_an_empty_output() {
            let mut operator = operator();
            let (left, right) = static_inputs();
            let job = job();
            let context = hot_context(&job);
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            for (side, record) in [("left", left.slice(0, 1)), ("right", right)] {
                operator
                    .process_data(
                        side,
                        Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                        &context,
                        &mut collector,
                    )
                    .await
                    .unwrap();
            }
            assert!(collector.drain("output").is_empty());
        }

        #[tokio::test]
        async fn hot_admission_enforces_state_row_limits() {
            let mut declaration = spec();
            declaration.limits.max_state_rows_per_side = 1;
            let mut operator =
                StreamJoinOperator::new("hot", quote_schema(), dimension_schema(), declaration)
                    .unwrap();
            operator.enable_native_checkpoint().unwrap();
            let (left, _) = static_inputs();
            let job = job();
            let context = hot_context(&job);
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            let error = operator
                .process_data(
                    "left",
                    Batch::table(vec![left], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap_err();
            assert!(matches!(
                error,
                crate::CalcFlowError::OperatorReason {
                    reason_code: crate::StreamingFailureReason::JoinStateLimitExceeded,
                    ..
                }
            ));
            assert_eq!(operator.state.metrics.state_limit_failures, 1);
            assert!(collector.drain("output").is_empty());
        }

        #[tokio::test]
        async fn hot_output_respects_the_edge_row_budget() {
            let mut operator =
                StreamJoinOperator::new("hot", quote_schema(), quote_schema(), band_spec())
                    .unwrap();
            operator.enable_run_core();
            let record = RecordBatch::try_new(
                quote_schema(),
                vec![
                    Arc::new(
                        TimestampMicrosecondArray::from(vec![0, 1_000_000]).with_timezone("UTC"),
                    ),
                    Arc::new(UInt64Array::from(vec![0, 1])),
                    Arc::new(StringArray::from(vec!["S000", "S000"])),
                    Arc::new(Float64Array::from(vec![2.0, 3.0])),
                ],
            )
            .unwrap();
            let job = job();
            let budget = crate::EdgeBudget::new(1, usize::MAX).unwrap();
            let context = hot_progress_context(&job, 0, budget);
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            for side in ["left", "right"] {
                operator
                    .process_data(
                        side,
                        Batch::table(vec![record.clone()], BatchMetadata::default()).unwrap(),
                        &context,
                        &mut collector,
                    )
                    .await
                    .unwrap();
            }
            let messages = collector.drain("output");
            assert_eq!(messages.len(), 4);
            assert!(
                messages
                    .iter()
                    .all(|message| message.as_data().unwrap().num_rows() == 1)
            );
        }

        #[tokio::test]
        async fn watermark_checkpoint_restores_compacted_locator_pages() {
            let mut original = operator();
            let job = job();
            let context = hot_context(&job);
            let mut collector = EdgeCollector::new(original.output_ports().to_vec());
            let record = RecordBatch::try_new(
                quote_schema(),
                vec![
                    Arc::new(
                        TimestampMicrosecondArray::from_iter_values(0..96).with_timezone("UTC"),
                    ),
                    Arc::new(UInt64Array::from_iter_values(0..96)),
                    Arc::new(StringArray::from(vec!["S000"; 96])),
                    Arc::new(Float64Array::from(vec![1.0; 96])),
                ],
            )
            .unwrap();
            original
                .process_data(
                    "left",
                    Batch::table(vec![record], BatchMetadata::default()).unwrap(),
                    &context,
                    &mut collector,
                )
                .await
                .unwrap();
            let progress = hot_progress_context(&job, 94, crate::EdgeBudget::default());
            original
                .on_ingress_progress("right", &progress)
                .await
                .unwrap();
            assert_eq!(original.status().left.retained_rows, 2);

            let snapshot = native_snapshot(&mut original).await;
            let mut restored = operator();
            restored.restore_native(&snapshot).unwrap();
            assert_eq!(
                restored.status().left.retained_bytes,
                original.status().left.retained_bytes
            );
            for name in ["left/group-0", "left/expiration"] {
                let segment = &snapshot.segments[name];
                let reader = datafusion::arrow::ipc::reader::FileReader::try_new(
                    std::io::Cursor::new(segment.bytes()),
                    None,
                )
                .unwrap();
                assert_eq!(
                    reader
                        .map(|record| record.unwrap().num_rows())
                        .sum::<usize>(),
                    2,
                    "{name}"
                );
            }
        }

        #[tokio::test]
        async fn hot_watermark_migration_evicts_expired_history() {
            let mut operator = operator();
            let (left, right) = static_inputs();
            let job = job();
            let context = hot_context(&job);
            let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
            for (side, batch) in [("left", left), ("right", right)] {
                operator
                    .process_data(
                        side,
                        Batch::table(vec![batch], BatchMetadata::default()).unwrap(),
                        &context,
                        &mut collector,
                    )
                    .await
                    .unwrap_or_else(|error| panic!("{side}: {error:?}"));
            }
            collector.drain("output");
            let progress = hot_progress_context(&job, 400_000_000, crate::EdgeBudget::default());
            operator
                .on_ingress_progress("right", &progress)
                .await
                .unwrap();
            assert!(!operator.run_core.as_ref().unwrap().hot_enabled());
            assert_eq!(operator.run_core.as_ref().unwrap().rows(JoinSide::Left), 0);
            assert_eq!(operator.state.metrics.left.evicted_rows, 2);
        }
    }
}
