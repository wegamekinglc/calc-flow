//! Owned sort work never captures raw ingress or caller schema owners.
use super::{BoundedOrderOperator, SortKey, SortTies, owned, workspace_bound};
use crate::Batch;
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherStop, OwnedCpuWork,
};
use crate::{Result, RunContext, StreamOperatorContext};
use datafusion::{
    arrow::{
        compute::{SortColumn, SortOptions, concat_batches, lexsort_to_indices, take},
        record_batch::RecordBatch,
    },
    execution::memory_pool::{MemoryPool, MemoryReservation},
};
use std::sync::Arc;

pub(super) struct FinalWork {
    history: owned::OwnedBatch,
    by: Vec<SortKey>,
    ties: SortTies,
    offset: u64,
    fetch: Option<u64>,
    columns: Vec<String>,
    pool: Arc<dyn MemoryPool>,
    _workspace: MemoryReservation,
    _declaration: Arc<MemoryReservation>,
}
impl FinalWork {
    fn execute(&self, check: &dyn Fn() -> Result<()>) -> Result<owned::OwnedBatch> {
        let history = &self.history;
        let pool: Arc<dyn MemoryPool> = self.pool.clone();
        check()?;
        let table = history.batch.table_payload()?;
        let input = concat_batches(table.schema(), table.batches()).map_err(owned::error)?;
        let indices = self.sorted_indices(&input)?;
        let batch = take_batch(&input, &indices, check)?;
        owned::normalize(&batch, &self.columns, &pool, None, check)
    }
    fn sorted_indices(&self, input: &RecordBatch) -> Result<datafusion::arrow::array::UInt32Array> {
        let mut keys = self
            .by
            .iter()
            .map(|key| SortColumn {
                values: input.column_by_name(key.column()).unwrap().clone(),
                options: Some(SortOptions {
                    descending: key.direction() == crate::SortDirection::Descending,
                    nulls_first: key.nulls() == crate::NullPlacement::First,
                }),
            })
            .collect::<Vec<_>>();
        if let SortTies::Sequence { columns } = &self.ties {
            keys.extend(columns.iter().map(|name| SortColumn {
                values: input.column_by_name(name).unwrap().clone(),
                options: Some(SortOptions {
                    descending: false,
                    nulls_first: false,
                }),
            }));
        }
        let indices = lexsort_to_indices(&keys, None).map_err(owned::error)?;
        let n = input.num_rows();
        let start = usize::try_from(self.offset).map_err(owned::error)?.min(n);
        let count = self
            .fetch
            .map_or(n - start, |k| usize::try_from(k).unwrap().min(n - start));
        Ok(indices.slice(start, count))
    }
}
impl OwnedCpuWork for FinalWork {
    type Output = owned::OwnedBatch;
    fn run(self, stop: &GatherStop) -> Result<Self::Output> {
        self.execute(&|| stop.check())
    }
}
impl BoundedOrderOperator {
    pub(super) fn final_work(
        &self,
        history: owned::OwnedBatch,
        pool: Arc<dyn MemoryPool>,
        declaration: Arc<MemoryReservation>,
    ) -> Result<FinalWork> {
        let workspace = owned::reserve(
            &pool,
            workspace_bound(&history, self.by.len() + self.columns.len())?,
            "sort workspace/output",
        )?;
        Ok(FinalWork {
            history,
            by: self.by.clone(),
            ties: self.spec.ties().clone(),
            offset: self.spec.offset(),
            fetch: self.fetch,
            columns: self.columns.clone(),
            pool,
            _workspace: workspace,
            _declaration: declaration,
        })
    }
    pub(super) async fn finalize_finite(
        &self,
        history: owned::OwnedBatch,
        pool: Arc<dyn MemoryPool>,
        declaration: Arc<MemoryReservation>,
        run: &RunContext,
    ) -> Result<owned::OwnedBatch> {
        run.check_cancelled()?;
        let work = self.final_work(history, pool, declaration)?;
        let stop = GatherStop::from_run(run);
        let output = tokio::task::spawn_blocking(move || work.run(&stop))
            .await
            .map_err(|e| crate::CalcFlowError::Internal {
                message: format!("bounded ordering worker: {e}"),
            })??;
        run.check_cancelled()?;
        Ok(output)
    }
    pub(super) async fn finalize_stream(
        &self,
        history: owned::OwnedBatch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<owned::OwnedBatch> {
        context.check_cancelled()?;
        let work = self.final_work(history, self.pool.clone(), self.declaration.clone())?;
        let credit = owned::reserve(&work.pool, 1024, "sort worker control")?;
        let scope = context
            .gather_client(
                GatherOperatorId::new(self.name.clone())
                    .with_reservation_owner(self.declaration.clone()),
            )
            .scope()?;
        let ticket = scope
            .submit_work(work, credit, GatherStop::from_job(context.job()))
            .await
            .map_err(admission_error)?;
        let output = ticket.finish().await?;
        context.check_cancelled()?;
        Ok(output.value)
    }
}

fn admission_error(failure: AdmissionFailure) -> crate::CalcFlowError {
    match failure {
        AdmissionFailure::Runtime(error) => error,
        AdmissionFailure::Budget { .. } => {
            owned::error("sort worker admission exceeds runtime budget")
        }
    }
}
fn take_batch(
    input: &RecordBatch,
    indices: &datafusion::arrow::array::UInt32Array,
    check: &dyn Fn() -> Result<()>,
) -> Result<Batch> {
    let arrays = input
        .columns()
        .iter()
        .map(|a| {
            check()?;
            take(a.as_ref(), indices, None).map_err(owned::error)
        })
        .collect::<Result<Vec<_>>>()?;
    let record = RecordBatch::try_new(input.schema(), arrays).map_err(owned::error)?;
    Batch::table(vec![record], crate::BatchMetadata::default())
}

pub(super) fn validate_string_capacity(
    dtype: &datafusion::arrow::datatypes::DataType,
    bytes: usize,
) -> Result<()> {
    use datafusion::arrow::datatypes::DataType;
    let limit = match dtype {
        DataType::Utf8 => 2_147_483_647_u64,
        DataType::LargeUtf8 => i64::MAX.unsigned_abs(),
        _ => return Ok(()),
    };
    if !u64::try_from(bytes).is_ok_and(|value| value <= limit) {
        return Err(crate::dataframe::invalid(
            "sort.workspace.string_capacity",
            "input_normalization_unsupported",
            "concatenated string offsets exceed their checked capacity",
        ));
    }
    Ok(())
}
pub(super) fn validate_concat(history: &owned::OwnedBatch) -> Result<()> {
    use datafusion::arrow::{
        array::{LargeStringArray, StringArray},
        datatypes::DataType,
    };
    let table = history.batch.table_payload()?;
    for (index, field) in table.schema().fields().iter().enumerate() {
        let dtype = field.data_type();
        if !matches!(dtype, DataType::Utf8 | DataType::LargeUtf8) {
            continue;
        }
        let bytes = table.batches().iter().try_fold(0, |total, record| {
            let array = record.column(index);
            let length = if dtype == &DataType::Utf8 {
                array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .value_data()
                    .len()
            } else {
                array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .unwrap()
                    .value_data()
                    .len()
            };
            owned::add(total, length)
        })?;
        validate_string_capacity(dtype, bytes)?;
    }
    Ok(())
}
