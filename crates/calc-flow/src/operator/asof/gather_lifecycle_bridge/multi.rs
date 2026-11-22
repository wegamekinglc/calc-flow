use std::sync::{Arc, Weak, mpsc};

use datafusion::{
    arrow::{
        array::{Array, ArrayRef, Int64Array, UInt32Array},
        compute::take,
    },
    execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation},
};

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
    columns: Vec<ArrayRef>,
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
        Ok(self.row_mode.then_some(RowGather {
            ordinal: 1,
            rows: 100_000,
            width: 8,
            sources: 2,
        }))
    }

    fn shared_column(&self, ordinal: usize) -> Result<Option<ArrayRef>> {
        Ok((self.row_mode && ordinal == 0).then(|| self.columns[0].clone()))
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
        stop.check()?;
        let indices = (range.start..range.end)
            .map(|row| {
                if ordinal == 0 {
                    u32::try_from(row).expect("fixed 100,000-row test")
                } else {
                    u32::try_from(99_999 - row).expect("fixed 100,000-row test")
                }
            })
            .collect::<Vec<_>>();
        let output = take(
            self.columns[ordinal].as_ref(),
            &UInt32Array::from(indices),
            None,
        )
        .map_err(|error| CalcFlowError::Internal {
            message: error.to_string(),
        })?;
        if self.first_error && unit == 0 {
            Err(CalcFlowError::Internal {
                message: "multi-row oracle error".into(),
            })
        } else {
            Ok(output)
        }
    }

    fn gather(&self, ordinal: usize, stop: &GatherStop) -> Result<ArrayRef> {
        let _ = self.entered.send(ordinal);
        self.gates[ordinal].wait();
        stop.check()?;
        if self.first_error && ordinal == 0 {
            return Err(CalcFlowError::Internal {
                message: "multi-column oracle error".into(),
            });
        }
        Ok(self.columns[ordinal].clone())
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
    let (entered_tx, entered) = mpsc::channel();
    let ranges = Arc::new(parking_lot::Mutex::new(Vec::with_capacity(2)));
    let plan = Arc::new(Plan {
        columns,
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
