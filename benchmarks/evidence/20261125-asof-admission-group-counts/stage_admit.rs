#![allow(dead_code, unused_imports)]
//! ASOF admission, settlement, and eviction measurements with Arrow output.

#[cfg(feature = "benchmark-internals")]
#[global_allocator]
static SYSTEM_ALLOCATOR: std::alloc::System = std::alloc::System;

#[cfg(not(feature = "benchmark-internals"))]
mod legacy {
    use async_trait::async_trait;
    use calc_flow::{
        AsofJoinSide, AsofStateLimits, Batch, BatchMetadata, CancellationToken, EventTime,
        IngressProgress, IngressProgressSnapshot, IngressState, JsonMap, StreamAsofJoinOperator,
        StreamAsofJoinSpec, StreamCollector, StreamJobContext, StreamOperator,
        StreamOperatorContext,
    };
    use datafusion::arrow::{
        array::{Array, Int64Array, TimestampMicrosecondArray},
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    };
    use serde_json::{Value, json};
    use std::{
        collections::BTreeMap,
        io::{self, Write},
        sync::Arc,
        time::{Duration, Instant},
    };

    const ROWS: usize = 100_000;
    const KEYS: usize = 64;
    const LARGE_BATCH: usize = 64_000;
    const TICK_BATCH: usize = 1_024;

    #[derive(Clone, Copy)]
    enum Case {
        AdmitSettle,
        EvictionTicks,
        OutOfOrder,
        CompositeKey,
    }

    impl Case {
        const ALL: [Self; 4] = [
            Self::AdmitSettle,
            Self::EvictionTicks,
            Self::OutOfOrder,
            Self::CompositeKey,
        ];

        const fn name(self) -> &'static str {
            match self {
                Self::AdmitSettle => "admit_settle_100k",
                Self::EvictionTicks => "eviction_ticks",
                Self::OutOfOrder => "out_of_order_within_watermark",
                Self::CompositeKey => "composite_key",
            }
        }

        const fn batch_rows(self) -> usize {
            if matches!(self, Self::EvictionTicks) {
                TICK_BATCH
            } else {
                LARGE_BATCH
            }
        }

        const fn reversed(self) -> bool {
            matches!(self, Self::OutOfOrder)
        }

        const fn composite(self) -> bool {
            matches!(self, Self::CompositeKey)
        }
    }

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, false),
            Field::new("subkey", DataType::Int64, false),
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("sequence", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]))
    }

    fn operator(case: Case) -> StreamAsofJoinOperator {
        let keys = if case.composite() {
            vec!["key".into(), "subkey".into()]
        } else {
            vec!["key".into()]
        };
        let side = |name: &str| {
            AsofJoinSide::new(
                keys.clone(),
                "time".into(),
                vec!["sequence".into()],
                name.into(),
            )
            .unwrap()
        };
        let spec = StreamAsofJoinSpec::new(
            side("left"),
            side("right"),
            Duration::ZERO,
            AsofStateLimits::new(250_000, 512 << 20).unwrap(),
        )
        .unwrap();
        StreamAsofJoinOperator::new("asof", schema(), schema(), spec).unwrap()
    }

    fn row_number(index: usize) -> i64 {
        i64::try_from(index).expect("bounded benchmark row fits i64")
    }

    fn input_batch(start: usize, len: usize, reversed: bool) -> Batch {
        let index = |offset: usize| start + if reversed { len - 1 - offset } else { offset };
        let record = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(
                    (0..len).map(|offset| row_number(index(offset) % KEYS)),
                )),
                Arc::new(Int64Array::from_iter_values(
                    (0..len).map(|offset| row_number(index(offset) % 7)),
                )),
                Arc::new(
                    TimestampMicrosecondArray::from_iter_values(
                        (0..len).map(|offset| row_number(index(offset) / KEYS)),
                    )
                    .with_timezone("UTC"),
                ),
                Arc::new(Int64Array::from_iter_values(
                    (0..len).map(|offset| row_number(index(offset))),
                )),
                Arc::new(Int64Array::from_iter_values(
                    (0..len).map(|offset| row_number(index(offset) * 3)),
                )),
            ],
        )
        .unwrap();
        Batch::table(vec![record], BatchMetadata::default()).unwrap()
    }

    fn prepared_batches(case: Case) -> Vec<(Batch, i64)> {
        (0..ROWS)
            .step_by(case.batch_rows())
            .map(|start| {
                let len = case.batch_rows().min(ROWS - start);
                let frontier = row_number((start + len - 1) / KEYS + 1);
                (input_batch(start, len, case.reversed()), frontier)
            })
            .collect()
    }

    fn progress(job: &StreamJobContext, frontier: i64) -> StreamOperatorContext<'_> {
        let snapshot = IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(frontier))),
            ),
            (
                "right".into(),
                IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(frontier))),
            ),
        ]));
        StreamOperatorContext::with_ingress_progress(
            job,
            "asof",
            Some(EventTime::from_micros(frontier)),
            snapshot,
        )
    }

    #[derive(Default)]
    struct Collector {
        output: Vec<Batch>,
    }

    #[async_trait]
    impl StreamCollector for Collector {
        async fn emit(&mut self, _port: &str, batch: Batch) -> calc_flow::Result<()> {
            self.output.push(batch);
            Ok(())
        }
    }

    fn validate_output(collector: &Collector, case: Case) -> usize {
        let mut rows = 0;
        let mut previous = None;
        for batch in &collector.output {
            for record in batch.table_payload().unwrap().batches() {
                let integer = |column: usize| {
                    record
                        .column(column)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                };
                let time = record
                    .column(2)
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .unwrap();
                for row in 0..record.num_rows() {
                    let sequence = integer(3).value(row);
                    let key = integer(0).value(row);
                    let subkey = if case.composite() {
                        integer(1).value(row)
                    } else {
                        0
                    };
                    let order = (time.value(row), key, subkey, sequence);
                    assert!(previous.is_none_or(|last| last <= order));
                    previous = Some(order);
                    assert!(!integer(8).is_null(row));
                    assert_eq!(integer(8).value(row), sequence);
                    assert_eq!(integer(9).value(row), sequence * 3);
                    rows += 1;
                }
            }
        }
        assert_eq!(rows, ROWS);
        rows
    }


    pub(super) fn main() {
        let case = Case::AdmitSettle;
        let batches = prepared_batches(case);
        let job = StreamJobContext::new(1, "asof-e2e", JsonMap::new(), None, CancellationToken::new());
        let mut operator = operator(case);
        let input = StreamOperatorContext::new(&job, "asof", None);
        let mut collector = Collector::default();
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        runtime.block_on(async {
            let started = Instant::now();
            for side in ["right", "left"] {
                for (batch, _) in &batches {
                    operator.process_data(side, batch.clone(), &input, &mut collector).await.unwrap();
                }
            }
            let seconds = started.elapsed().as_secs_f64();
            let admitted = operator.status();
            assert_eq!(admitted.pending_left_rows, ROWS as u64);
            assert_eq!(admitted.retained_right_rows, ROWS as u64);
            assert_eq!(admitted.emitted_left_rows, 0);
            assert!(collector.output.is_empty());
            let frontier = batches.last().unwrap().1;
            let context = progress(&job, frontier);
            let settled = Instant::now();
            operator.on_watermark(EventTime::from_micros(frontier), &context, &mut collector).await.unwrap();
            let watermark_seconds = settled.elapsed().as_secs_f64();
            let rows = validate_output(&collector, case);
            let status = operator.status();
            assert_eq!(status.emitted_left_rows, ROWS as u64);
            assert_eq!(status.matched_rows, ROWS as u64);
            assert_eq!(status.evicted_right_rows + status.retained_right_rows, ROWS as u64);
            println!("{}", json!({
                "seconds": seconds, "watermark_seconds": watermark_seconds,
                "output_rows": rows, "validated_all_rows": true,
                "evicted_right_rows": status.evicted_right_rows, "retained_right_rows": status.retained_right_rows,
                "pending_before_watermark": admitted.pending_left_rows,
                "retained_before_watermark": admitted.retained_right_rows
            }));
        });
    }
}
fn main() { legacy::main(); }
