use super::{
    Arc, AtomicBool, ErasedOutput, GatherHome, GatherOperatorId, GatherPlan, Ordering, Result,
    cancelled, internal,
};

#[derive(Clone, Copy, Eq, PartialEq)]
enum Phase {
    Pending,
    Running,
    Done,
}

struct Cell {
    phase: Phase,
    outcome: Option<Result<ErasedOutput>>,
}

pub(super) struct Columns {
    plan: Arc<dyn GatherPlan>,
    cells: Vec<Cell>,
    active: usize,
    remaining: usize,
    rows: Option<super::row_gather::Assembly>,
    stop: Arc<AtomicBool>,
}

impl Columns {
    pub(super) fn new(plan: Arc<dyn GatherPlan>) -> Result<Self> {
        let shape = if plan.parallelism() > 1 {
            plan.row_gather()?
        } else {
            None
        };
        let count = shape
            .as_ref()
            .map_or(plan.column_count(), |shape| shape.parts(plan.parallelism()));
        let rows = shape
            .map(|shape| super::row_gather::Assembly::new(plan.as_ref(), shape))
            .transpose()?;
        Ok(Self {
            plan,
            cells: (0..count)
                .map(|_| Cell {
                    phase: Phase::Pending,
                    outcome: None,
                })
                .collect(),
            active: 0,
            remaining: count,
            rows,
            stop: Arc::new(AtomicBool::new(false)),
        })
    }

    pub(super) fn control_bytes(plan: &dyn GatherPlan) -> Result<usize> {
        let shape = if plan.parallelism() > 1 {
            plan.row_gather()?
        } else {
            None
        };
        let count = shape
            .as_ref()
            .map_or(plan.column_count(), |shape| shape.parts(plan.parallelism()));
        let workspace = shape
            .as_ref()
            .map_or(Ok(0), |shape| shape.workspace_bytes(plan.parallelism()))?;
        count
            .checked_mul(size_of::<Cell>() + size_of::<super::ArrayRef>() + 128)
            .and_then(|bytes| bytes.checked_add(plan.column_count().checked_mul(64)?))
            .and_then(|bytes| bytes.checked_add(size_of::<Self>() + 128))
            .and_then(|bytes| bytes.checked_add(workspace))
            .ok_or_else(|| internal("parallel column control credit overflow"))
    }

    pub(super) fn count(&self) -> usize {
        self.cells.len()
    }

    pub(super) fn workers(&self) -> usize {
        self.plan.parallelism().clamp(1, 8).min(self.count())
    }

    pub(super) fn active(&self) -> usize {
        self.active
    }

    pub(super) fn claim(&mut self, ordinal: usize) -> Option<super::ClaimedWork> {
        let cell = self.cells.get_mut(ordinal)?;
        if cell.phase != Phase::Pending {
            return None;
        }
        cell.phase = Phase::Running;
        self.active += 1;
        let (target, range) = self.rows.as_ref().map_or((ordinal, None), |rows| {
            (
                rows.shape.ordinal,
                Some(rows.shape.range(ordinal, self.cells.len())),
            )
        });
        Some(super::ClaimedWork::Column {
            plan: self.plan.clone(),
            ordinal: target,
            range,
            siblings: self.stop.clone(),
        })
    }

    pub(super) fn complete(&mut self, ordinal: usize, outcome: Result<ErasedOutput>) -> bool {
        let Some(cell) = self.cells.get_mut(ordinal) else {
            return false;
        };
        if cell.phase != Phase::Running {
            return false;
        }
        if outcome.is_err() {
            self.stop.store(true, Ordering::Release);
        }
        cell.outcome = Some(outcome);
        cell.phase = Phase::Done;
        self.active -= 1;
        self.remaining -= 1;
        self.remaining == 0
    }

    pub(super) fn cancel_pending(&mut self, run_id: &str) {
        self.stop.store(true, Ordering::Release);
        for cell in &mut self.cells {
            if cell.phase == Phase::Pending {
                cell.phase = Phase::Done;
                cell.outcome = Some(Err(cancelled(run_id)));
                self.remaining -= 1;
            }
        }
    }

    pub(super) fn finish(
        self,
        home: &GatherHome,
        operator: &GatherOperatorId,
        stop: &super::GatherStop,
    ) -> Result<ErasedOutput> {
        let primary = self
            .cells
            .iter()
            .position(|cell| {
                matches!(&cell.outcome, Some(Err(error)) if !matches!(error, crate::CalcFlowError::Cancelled { .. }))
            })
            .or_else(|| self.cells.iter().position(|cell| matches!(&cell.outcome, Some(Err(_)))));
        let mut columns = Vec::with_capacity(self.cells.len());
        let mut error = None;
        for (ordinal, cell) in self.cells.into_iter().enumerate() {
            match cell
                .outcome
                .ok_or_else(|| internal("missing column outcome"))?
            {
                Ok(value) => {
                    columns.push(
                        *value
                            .downcast::<super::ArrayRef>()
                            .map_err(|_| internal("column output identity mismatch"))?,
                    );
                }
                Err(value) if Some(ordinal) == primary => error = Some(value),
                Err(value) => home.retain_error(operator, &value),
            }
        }
        let outcome = match error {
            Some(error) => Err(error),
            None => match self.rows {
                Some(rows) => rows.finish(&columns, stop),
                None => Ok(columns),
            },
        };
        drop(self.plan);
        Ok(Box::new(outcome?))
    }
}
