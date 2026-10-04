use std::sync::{Arc, Weak, mpsc};

use datafusion::{
    arrow::{
        array::{Array, ArrayRef, Int64Array},
        datatypes::DataType,
    },
    execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation},
};

use super::super::{ColumnRequest, MaterializationInput, OutputPlan, OutputSide, Span};
use crate::runtime::streaming::gather_work::{GatherPlan, GatherStop, RowGather};
use crate::{CalcFlowError, Result};

type RowRanges = Arc<parking_lot::Mutex<Vec<(usize, std::ops::Range<usize>)>>>;

pub(crate) struct Input {
    pub(crate) plan: Arc<dyn GatherPlan>,
    pub(crate) credit: MemoryReservation,
    pub(crate) probe: Probe,
}

pub(crate) struct Probe {
    pub(crate) entered: mpsc::Receiver<usize>,
    pub(crate) source: Weak<dyn Array>,
    pub(crate) shared_source: Option<Weak<dyn Array>>,
    gates: [Arc<Gate>; 2],
    ranges: RowRanges,
}

impl Probe {
    pub(crate) fn complete_row_ranges(&self) -> bool {
        let mut ranges = self.ranges.lock().clone();
        ranges.sort_unstable_by_key(|(_, range)| range.start);
        ranges == [(1, 0..50_000), (1, 50_000..100_000)]
    }

    pub(crate) fn release(&self, ordinal: usize) {
        self.gates[ordinal].release();
    }

    pub(crate) fn release_all(&self) {
        for gate in &self.gates {
            gate.release();
        }
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        self.release_all();
    }
}

struct Gate {
    open: parking_lot::Mutex<bool>,
    changed: parking_lot::Condvar,
}

impl Gate {
    fn release(&self) {
        *self.open.lock() = true;
        self.changed.notify_all();
    }

    fn wait(&self) {
        let mut open = self.open.lock();
        while !*open {
            self.changed.wait(&mut open);
        }
    }
}

struct Plan {
    input: MaterializationInput,
    entered: mpsc::Sender<usize>,
    gates: [Arc<Gate>; 2],
    first_error: bool,
    row_mode: bool,
    ranges: RowRanges,
}

impl GatherPlan for Plan {
    fn column_count(&self) -> usize {
        2
    }

    fn row_gather(&self) -> Result<Option<RowGather>> {
        if self.row_mode {
            self.input.row_gather()
        } else {
            Ok(None)
        }
    }

    fn shared_column(&self, ordinal: usize) -> Result<Option<ArrayRef>> {
        self.input.shared_column(ordinal)
    }

    fn gather_range(
        &self,
        ordinal: usize,
        range: std::ops::Range<usize>,
        stop: &GatherStop,
    ) -> Result<ArrayRef> {
        let unit = usize::from(range.start != 0);
        self.ranges.lock().push((ordinal, range.clone()));
        let _ = self.entered.send(unit);
        self.gates[unit].wait();
        let output = self.input.gather_range(ordinal, range, stop);
        if self.first_error && unit == 0 {
            drop(output);
            Err(CalcFlowError::Internal {
                message: "multi-row oracle error".into(),
            })
        } else {
            output
        }
    }

    fn gather(&self, ordinal: usize, stop: &GatherStop) -> Result<ArrayRef> {
        let _ = self.entered.send(ordinal);
        self.gates[ordinal].wait();
        let output = self.input.gather(ordinal, stop);
        if self.first_error && ordinal == 0 {
            drop(output);
            Err(CalcFlowError::Internal {
                message: "multi-column oracle error".into(),
            })
        } else {
            output
        }
    }
}

pub(crate) fn materialization(pool: &Arc<dyn MemoryPool>, first_error: bool) -> Input {
    build(pool, first_error, false)
}

pub(crate) fn row_materialization(pool: &Arc<dyn MemoryPool>, first_error: bool) -> Input {
    build(pool, first_error, true)
}

fn build(pool: &Arc<dyn MemoryPool>, first_error: bool, row_mode: bool) -> Input {
    let credit = MemoryConsumer::new("multi-column-native-oracle").register(pool);
    credit.try_grow(33_554_432).unwrap();
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from_iter_values(0..100_000)),
        Arc::new(Int64Array::from_iter_values(
            (0..100_000).map(|value| -value),
        )),
    ];
    let source = Arc::downgrade(&columns[usize::from(row_mode)]);
    let shared_source = row_mode.then(|| Arc::downgrade(&columns[0]));
    let gates = std::array::from_fn(|_| {
        Arc::new(Gate {
            open: parking_lot::Mutex::new(false),
            changed: parking_lot::Condvar::new(),
        })
    });
    let rows = OutputPlan {
        left: OutputSide {
            batches: vec![columns],
            positions: (0..100_000).rev().map(|row| (0, row)).collect(),
            spans: (0..100_000)
                .rev()
                .map(|row| Span {
                    source: 0,
                    start: row,
                    end: row + 1,
                })
                .collect(),
            has_nulls: false,
        },
        right: OutputSide {
            batches: Vec::new(),
            positions: Vec::new(),
            spans: Vec::new(),
            has_nulls: false,
        },
        len: 100_000,
        matched: 0,
        raw_bytes: 0,
    };
    let rows = if row_mode {
        OutputPlan {
            left: OutputSide {
                batches: vec![vec![rows.left.batches[0][0].clone()]],
                positions: (0..100_000).map(|row| (0, row)).collect(),
                spans: vec![Span {
                    source: 0,
                    start: 0,
                    end: 100_000,
                }],
                has_nulls: false,
            },
            right: OutputSide {
                batches: vec![vec![rows.left.batches[0][1].clone()]],
                positions: (0..100_000).rev().map(|row| (1, row)).collect(),
                spans: Vec::new(),
                has_nulls: false,
            },
            len: 100_000,
            matched: 100_000,
            raw_bytes: 0,
        }
    } else {
        rows
    };
    let input = MaterializationInput {
        rows,
        requests: (0..2)
            .map(|index| ColumnRequest {
                index,
                data_type: DataType::Int64,
            })
            .collect(),
        worker_gate: None,
        worker_probe: None,
    };
    let (entered_tx, entered) = mpsc::channel();
    let ranges = Arc::new(parking_lot::Mutex::new(Vec::with_capacity(2)));
    let plan = Arc::new(Plan {
        input,
        entered: entered_tx,
        gates: gates.clone(),
        first_error,
        row_mode,
        ranges: ranges.clone(),
    });
    Input {
        plan,
        credit,
        probe: Probe {
            entered,
            source,
            shared_source,
            gates,
            ranges,
        },
    }
}
