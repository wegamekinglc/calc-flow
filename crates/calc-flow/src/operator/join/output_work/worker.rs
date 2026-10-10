use super::{
    funding::{self, OutputFunding},
    inputs::Inputs,
};
use crate::Result;
use crate::runtime::streaming::gather_work::{GatherStop, OwnedCpuWork, ParallelCpuWork};
use datafusion::arrow::{
    array::{Array, ArrayRef, UInt64Array},
    compute::{concat, take},
    datatypes::SchemaRef,
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::Arc;

pub(super) struct Fragment {
    pub(super) columns: Vec<ArrayRef>,
    pub(super) credit: Arc<MemoryReservation>,
}

pub(super) struct FragmentWork {
    pub(super) inputs: Inputs,
    pub(super) name: Arc<str>,
    #[cfg(test)]
    pub(super) hook: Option<Arc<dyn Fn(usize, bool) + Send + Sync>>,
}

impl ParallelCpuWork for FragmentWork {
    type Output = Fragment;

    fn unit_count(&self) -> usize {
        super::units(self.inputs.rows.len())
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<Fragment> {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(ordinal, false);
        }
        stop.check()?;
        let rows = &self.inputs.rows[super::range(self.inputs.rows.len(), ordinal)];
        let (left, right) = selections(rows, stop)?;
        let left_parent = self.inputs.left.as_ref().expect("owned left parent");
        let right_parent = self.inputs.right.as_ref().expect("owned right parent");
        let mut columns =
            Vec::with_capacity(left_parent.num_columns() + right_parent.num_columns());
        gather_columns(left_parent.columns(), &left, &mut columns, &self.name, stop)?;
        gather_columns(
            right_parent.columns(),
            &right,
            &mut columns,
            &self.name,
            stop,
        )?;
        stop.check()?;
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(ordinal, true);
        }
        Ok(Fragment {
            columns,
            credit: Arc::clone(self.inputs.credit.as_ref().expect("paid inputs")),
        })
    }
}

fn selections(
    rows: &[super::inputs::Selection],
    stop: &GatherStop,
) -> Result<(UInt64Array, UInt64Array)> {
    let mut left = Vec::with_capacity(rows.len());
    let mut right = Vec::with_capacity(rows.len());
    for (ordinal, row) in rows.iter().enumerate() {
        if ordinal.is_multiple_of(256) {
            stop.check()?;
        }
        left.push(row.left);
        right.push(row.right);
    }
    Ok((UInt64Array::from(left), UInt64Array::from(right)))
}

fn arrow_error(name: &str, error: &datafusion::arrow::error::ArrowError) -> crate::CalcFlowError {
    super::super::operator_error(name, &format!("output projection failed: {error}"))
}

fn gather_columns(
    inputs: &[ArrayRef],
    indices: &UInt64Array,
    columns: &mut Vec<ArrayRef>,
    name: &str,
    stop: &GatherStop,
) -> Result<()> {
    for input in inputs {
        stop.check()?;
        columns
            .push(take(input.as_ref(), indices, None).map_err(|error| arrow_error(name, &error))?);
    }
    Ok(())
}

pub(super) struct MergeInputs {
    pub(super) fragments: Vec<Fragment>,
    pub(super) released: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Drop for MergeInputs {
    fn drop(&mut self) {
        drop(std::mem::take(&mut self.fragments));
        if let Some(released) = self.released.take() {
            let _ = released.send(());
        }
    }
}

pub(super) struct MergeWork {
    pub(super) inputs: MergeInputs,
    pub(super) schema: SchemaRef,
    pub(super) funding: Arc<OutputFunding>,
    pub(super) name: Arc<str>,
}

impl OwnedCpuWork for MergeWork {
    type Output = RecordBatch;

    fn control_bytes(&self) -> Result<usize> {
        Ok(size_of::<Self>())
    }

    fn run(self, stop: &GatherStop) -> Result<RecordBatch> {
        let mut columns = Vec::with_capacity(self.schema.fields().len());
        for (index, field) in self.schema.fields().iter().enumerate() {
            stop.check()?;
            let fragments = self
                .inputs
                .fragments
                .iter()
                .map(|fragment| fragment.columns[index].as_ref())
                .collect::<Vec<&dyn Array>>();
            let array = concat(&fragments).map_err(|error| arrow_error(&self.name, &error))?;
            columns.push(
                funding::bind(array, field.data_type(), &self.funding)
                    .map_err(|error| arrow_error(&self.name, &error))?,
            );
        }
        stop.check()?;
        RecordBatch::try_new(Arc::clone(&self.schema), columns)
            .map_err(|error| arrow_error(&self.name, &error))
    }
}
