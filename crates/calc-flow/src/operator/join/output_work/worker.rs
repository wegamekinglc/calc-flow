use super::{
    funding::{self, OutputFunding},
    inputs::Inputs,
};
use crate::Result;
use crate::runtime::streaming::gather_work::{GatherStop, ParallelCpuWork};
use datafusion::arrow::{
    array::ArrayRef, compute::take, datatypes::SchemaRef, record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::MemoryReservation;
use std::sync::Arc;

pub(super) struct Fragment {
    pub(super) columns: Vec<ArrayRef>,
    pub(super) credit: Arc<MemoryReservation>,
}

pub(super) struct FragmentWork {
    pub(super) inputs: Inputs,
    pub(super) schema: SchemaRef,
    pub(super) funding: Arc<OutputFunding>,
    pub(super) name: Arc<str>,
    #[cfg(test)]
    pub(super) hook: Option<Arc<dyn Fn(usize, bool) + Send + Sync>>,
    #[cfg(test)]
    pub(super) take_hook: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
}

impl ParallelCpuWork for FragmentWork {
    type Output = Fragment;

    fn unit_count(&self) -> usize {
        super::units(
            self.inputs
                .left_indices
                .as_ref()
                .expect("paid indices")
                .len(),
            self.schema.fields().len(),
        )
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<Fragment> {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(ordinal, false);
        }
        stop.check()?;
        let range = super::range(self.schema.fields().len(), self.unit_count(), ordinal);
        let mut columns = Vec::with_capacity(range.len());
        for index in range {
            columns.push(self.column(index, stop)?);
        }
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

impl FragmentWork {
    fn column(&self, index: usize, stop: &GatherStop) -> Result<ArrayRef> {
        stop.check()?;
        let left = self.inputs.left.as_ref().expect("owned left parent");
        let right = self.inputs.right.as_ref().expect("owned right parent");
        let (parent, indices, local) = if index < left.num_columns() {
            (left, &self.inputs.left_indices, index)
        } else {
            (
                right,
                &self.inputs.right_indices,
                index - left.num_columns(),
            )
        };
        let indices = indices.as_ref().expect("paid indices");
        let array = take(parent.column(local).as_ref(), indices, None)
            .map_err(|error| arrow_error(&self.name, &error))?;
        #[cfg(test)]
        if let Some(hook) = &self.take_hook {
            hook(index, indices.len());
        }
        funding::bind(array, self.schema.field(index).data_type(), &self.funding)
            .map_err(|error| arrow_error(&self.name, &error))
    }
}

pub(super) fn into_record(
    fragments: Vec<Fragment>,
    schema: SchemaRef,
    name: &str,
) -> Result<RecordBatch> {
    let credit = Arc::clone(&fragments[0].credit);
    let mut columns = Vec::with_capacity(schema.fields().len());
    for fragment in fragments {
        columns.extend(fragment.columns);
        drop(fragment.credit);
    }
    drop(credit);
    RecordBatch::try_new(schema, columns).map_err(|error| arrow_error(name, &error))
}

fn arrow_error(name: &str, error: &datafusion::arrow::error::ArrowError) -> crate::CalcFlowError {
    super::super::operator_error(name, &format!("output projection failed: {error}"))
}
