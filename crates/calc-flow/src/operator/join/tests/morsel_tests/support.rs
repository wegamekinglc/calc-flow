use super::*;
use datafusion::arrow::array::{Array, ArrayRef};
use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryLimit, MemoryPool, MemoryReservation,
};
use std::sync::atomic::{AtomicUsize, Ordering};

pub(super) const TIMES: [i64; 8] = [2, 0, 1, 0, 2, 1, 0, 1];
pub(super) const ORDER: [usize; 8] = [1, 3, 6, 2, 5, 7, 0, 4];

pub(super) fn operator(projected: bool) -> StreamJoinOperator {
    let mut declaration = spec();
    declaration.limits = JoinStateLimits::new(100_000, 100_000_000, 1_000_000).unwrap();
    let mut operator =
        StreamJoinOperator::new("match", left_schema(), right_schema(), declaration).unwrap();
    if projected {
        operator.set_output_projection(vec![5]).unwrap();
    }
    operator
}

pub(super) fn input(left: bool, rows: usize, dimension: bool) -> Batch {
    let length = if dimension { 8 } else { rows + 2 };
    let position = |index: usize| source_position(index, dimension);
    let time = |index: usize| {
        if dimension {
            TIMES[index]
        } else {
            i64::try_from(position(index) % 2).unwrap()
        }
    };
    let payload: ArrayRef = if left {
        Arc::new(Int64Array::from_iter_values((0..length).map(|index| {
            i64::try_from(position(index) + if dimension { 100 } else { 0 }).unwrap()
        })))
    } else {
        Arc::new(StringArray::from_iter_values((0..length).map(|index| {
            format!("{}{}", if dimension { "D" } else { "R" }, position(index))
        })))
    };
    let record = RecordBatch::try_new(
        if left { left_schema() } else { right_schema() },
        vec![
            Arc::new(Int64Array::from(vec![7; length])),
            Arc::new(
                TimestampMicrosecondArray::from_iter_values((0..length).map(time))
                    .with_timezone("UTC"),
            ),
            payload,
        ],
    )
    .unwrap();
    let records = if dimension {
        vec![record]
    } else {
        let mid = rows / 2;
        vec![record.slice(1, mid), record.slice(1 + mid, rows - mid)]
    };
    Batch::table(records, BatchMetadata::default()).unwrap()
}

fn source_position(index: usize, dimension: bool) -> usize {
    if dimension {
        index
    } else {
        index.saturating_sub(1)
    }
}

pub(super) async fn preload(
    operator: &mut StreamJoinOperator,
    job: &StreamJobContext,
    incoming_left: bool,
) {
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data(
            if incoming_left { "right" } else { "left" },
            input(!incoming_left, 8, true),
            &StreamOperatorContext::new(job, "match", None),
            &mut output,
        )
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
}

pub(super) fn verify(
    collector: &mut EdgeCollector,
    rows: usize,
    incoming_left: bool,
    projected: bool,
) {
    let mut global = 0usize;
    for (sequence, message) in collector.drain("output").into_iter().enumerate() {
        let batch = message.as_data().unwrap();
        assert_eq!(batch.metadata().sequence(), sequence as u64);
        for record in batch.table_payload().unwrap().batches() {
            for row in 0..record.num_rows() {
                let position = global / 8;
                let opposite = ORDER[global % 8];
                let expected_status = if incoming_left {
                    format!("D{opposite}")
                } else {
                    format!("R{position}")
                };
                let status_column = if projected { 0 } else { 5 };
                assert_eq!(
                    record
                        .column(status_column)
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap()
                        .value(row),
                    expected_status
                );
                if !projected {
                    assert_full_row(record, row, position, opposite, incoming_left);
                }
                global += 1;
            }
        }
    }
    assert_eq!(global, rows * 8);
}

fn assert_full_row(
    record: &RecordBatch,
    row: usize,
    position: usize,
    opposite: usize,
    incoming_left: bool,
) {
    for ordinal in [0, 3] {
        assert_eq!(
            record
                .column(ordinal)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(row),
            7
        );
    }
    let expected = if incoming_left {
        position
    } else {
        opposite + 100
    };
    assert_eq!(
        record
            .column(2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(row),
        i64::try_from(expected).unwrap()
    );
    let times = if incoming_left {
        [i64::try_from(position % 2).unwrap(), TIMES[opposite]]
    } else {
        [TIMES[opposite], i64::try_from(position % 2).unwrap()]
    };
    for (ordinal, expected) in [1, 4].into_iter().zip(times) {
        assert_eq!(
            record
                .column(ordinal)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap()
                .value(row),
            expected
        );
    }
}

pub(super) fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

#[derive(Debug)]
pub(super) struct Pool {
    inner: GreedyMemoryPool,
    pub(super) live: AtomicUsize,
    pub(super) attempts: AtomicUsize,
    reject_at: usize,
}

impl Pool {
    pub(super) fn new(reject_at: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: GreedyMemoryPool::new(1 << 30),
            live: AtomicUsize::new(0),
            attempts: AtomicUsize::new(0),
            reject_at,
        })
    }

    fn morsel(credit: &MemoryReservation) -> bool {
        credit.consumer().name() == "sql-incremental:stream-join-morsel"
    }
}

impl fmt::Display for Pool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(formatter)
    }
}

impl MemoryPool for Pool {
    fn name(&self) -> &'static str {
        "morsel-refusal-test"
    }
    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
    fn grow(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.grow(credit, bytes);
        if Self::morsel(credit) {
            self.live.fetch_add(bytes, Ordering::SeqCst);
        }
    }
    fn shrink(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.shrink(credit, bytes);
        if Self::morsel(credit) {
            self.live.fetch_sub(bytes, Ordering::SeqCst);
        }
    }
    fn try_grow(&self, credit: &MemoryReservation, bytes: usize) -> datafusion::common::Result<()> {
        if Self::morsel(credit)
            && bytes > 0
            && self.attempts.fetch_add(1, Ordering::SeqCst) + 1 == self.reject_at
        {
            return Err(datafusion::common::DataFusionError::ResourcesExhausted(
                "morsel test refusal".into(),
            ));
        }
        self.inner.try_grow(credit, bytes)?;
        if Self::morsel(credit) {
            self.live.fetch_add(bytes, Ordering::SeqCst);
        }
        Ok(())
    }
}
