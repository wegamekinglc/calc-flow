use std::ops::Range;

use datafusion::arrow::{array::ArrayRef, compute::concat};

use super::{GatherPlan, GatherStop, Result, internal};

pub(crate) struct RowGather {
    pub(crate) ordinal: usize,
    pub(crate) rows: usize,
    pub(crate) width: usize,
    pub(crate) sources: usize,
}

impl RowGather {
    pub(super) fn parts(&self, workers: usize) -> usize {
        workers.clamp(1, 8).min(self.rows)
    }

    pub(super) fn range(&self, ordinal: usize, parts: usize) -> Range<usize> {
        let width = self.rows / parts;
        let extra = self.rows % parts;
        let start = ordinal * width + ordinal.min(extra);
        start..start + width + usize::from(ordinal < extra)
    }

    pub(super) fn workspace_bytes(&self, workers: usize) -> Result<usize> {
        let parts = self.parts(workers);
        let buffers = self
            .rows
            .checked_mul(self.width)
            .and_then(|bytes| bytes.checked_add(self.rows.div_ceil(8)))
            .and_then(|bytes| bytes.checked_mul(2));
        let heads = self
            .sources
            .checked_add(1)
            .and_then(|sources| sources.checked_mul(256))
            .and_then(|bytes| bytes.checked_add(8_192))
            .and_then(|bytes| bytes.checked_mul(parts + 1));
        buffers
            .zip(heads)
            .and_then(|(buffers, heads)| buffers.checked_add(heads))
            .ok_or_else(|| internal("row gather peak credit overflow"))
    }
}

pub(super) struct Assembly {
    pub(super) shape: RowGather,
    shared: Vec<Option<ArrayRef>>,
}

impl Assembly {
    pub(super) fn new(plan: &dyn GatherPlan, shape: RowGather) -> Result<Self> {
        let shared = (0..plan.column_count())
            .map(|ordinal| {
                if ordinal == shape.ordinal {
                    Ok(None)
                } else {
                    plan.shared_column(ordinal)?
                        .map(Some)
                        .ok_or_else(|| internal("missing shared row-gather column"))
                }
            })
            .collect::<Result<_>>()?;
        Ok(Self { shape, shared })
    }

    pub(super) fn finish(mut self, parts: &[ArrayRef], stop: &GatherStop) -> Result<Vec<ArrayRef>> {
        stop.check()?;
        let inputs = parts.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        let array = concat(&inputs).map_err(|error| crate::CalcFlowError::Internal {
            message: format!("ASOF row concat: {error}"),
        })?;
        stop.check()?;
        if array.len() != self.shape.rows {
            return Err(internal("row concat length mismatch"));
        }
        self.shared[self.shape.ordinal] = Some(array);
        self.shared
            .into_iter()
            .map(|array| array.ok_or_else(|| internal("missing assembled column")))
            .collect()
    }
}
