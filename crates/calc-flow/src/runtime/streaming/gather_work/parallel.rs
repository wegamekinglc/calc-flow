use super::{
    Arc, ClaimedWork, ErasedOutput, ErasedWork, GatherHome, GatherOperatorId, GatherStop,
    ReadyWork, Result, WorkPackage, cancelled, internal,
};

pub(crate) trait ParallelCpuWork: Send + Sync + 'static {
    type Output: Send + 'static;

    fn unit_count(&self) -> usize;
    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<Self::Output>;
}

pub(super) struct Package<W>(pub(super) Arc<W>);

impl<W: ParallelCpuWork> WorkPackage for Package<W> {
    type Output = Vec<W::Output>;

    fn control_bytes(&self) -> Result<usize> {
        if self.0.unit_count() == 0 {
            return Err(internal("parallel work requires at least one unit"));
        }
        self.0
            .unit_count()
            .checked_mul(size_of::<Cell>() + size_of::<Unit<W>>() + size_of::<W::Output>() + 128)
            .and_then(|bytes| bytes.checked_add(size_of::<Units>() + 128))
            .ok_or_else(|| internal("parallel work control credit overflow"))
    }

    fn into_work(self) -> Result<ReadyWork> {
        Ok(ReadyWork::Parallel(Units {
            cells: (0..self.0.unit_count())
                .map(|ordinal| Cell {
                    work: Some(Box::new(Unit {
                        work: self.0.clone(),
                        ordinal,
                    })),
                    outcome: None,
                    running: false,
                })
                .collect(),
            active: 0,
            remaining: self.0.unit_count(),
            collect: collect::<W::Output>,
        }))
    }
}

struct Unit<W> {
    work: Arc<W>,
    ordinal: usize,
}

impl<W: ParallelCpuWork> ErasedWork for Unit<W> {
    fn run(self: Box<Self>, stop: &GatherStop) -> Result<ErasedOutput> {
        stop.check()?;
        Ok(Box::new(self.work.run(self.ordinal, stop)?))
    }
}

struct Cell {
    work: Option<Box<dyn ErasedWork>>,
    outcome: Option<Result<ErasedOutput>>,
    running: bool,
}

pub(super) struct Units {
    cells: Vec<Cell>,
    active: usize,
    remaining: usize,
    collect: fn(Vec<ErasedOutput>) -> Result<ErasedOutput>,
}

impl Units {
    pub(super) fn count(&self) -> usize {
        self.cells.len()
    }
    pub(super) fn workers(&self) -> usize {
        self.count().clamp(1, 8)
    }
    pub(super) fn active(&self) -> usize {
        self.active
    }

    pub(super) fn claim(&mut self, ordinal: usize) -> Option<ClaimedWork> {
        let cell = self.cells.get_mut(ordinal)?;
        let work = cell.work.take()?;
        if cell.outcome.is_some() {
            return None;
        }
        cell.running = true;
        self.active += 1;
        Some(ClaimedWork::Single(work))
    }

    pub(super) fn complete(&mut self, ordinal: usize, outcome: Result<ErasedOutput>) -> bool {
        let Some(cell) = self.cells.get_mut(ordinal) else {
            return false;
        };
        if !cell.running {
            return false;
        }
        cell.outcome = Some(outcome);
        cell.running = false;
        self.active -= 1;
        self.remaining -= 1;
        self.remaining == 0
    }

    pub(super) fn cancel_pending(&mut self, run_id: &str) {
        for cell in &mut self.cells {
            if !cell.running && cell.outcome.is_none() {
                cell.outcome = Some(Err(cancelled(run_id)));
                self.remaining -= 1;
            }
        }
    }

    pub(super) fn finish(
        self,
        home: &GatherHome,
        operator: &GatherOperatorId,
    ) -> Result<ErasedOutput> {
        let primary = self.primary_failure();
        let mut values = Vec::with_capacity(self.cells.len());
        let mut error = None;
        for (ordinal, cell) in self.cells.into_iter().enumerate() {
            match cell
                .outcome
                .ok_or_else(|| internal("missing parallel work outcome"))?
            {
                Ok(value) => values.push(value),
                Err(value) if Some(ordinal) == primary => error = Some(value),
                Err(value) => home.retain_error(operator, &value),
            }
        }
        match error {
            Some(error) => Err(error),
            None => (self.collect)(values),
        }
    }

    fn primary_failure(&self) -> Option<usize> {
        self.cells.iter().position(|cell| {
            matches!(&cell.outcome, Some(Err(error)) if !matches!(error, crate::CalcFlowError::Cancelled { .. }))
        }).or_else(|| self.cells.iter().position(|cell| matches!(&cell.outcome, Some(Err(_)))))
    }
}

fn collect<T: Send + 'static>(values: Vec<ErasedOutput>) -> Result<ErasedOutput> {
    let values = values
        .into_iter()
        .map(|value| {
            value
                .downcast::<T>()
                .map(|value| *value)
                .map_err(|_| internal("parallel work output identity mismatch"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Box::new(values))
}
