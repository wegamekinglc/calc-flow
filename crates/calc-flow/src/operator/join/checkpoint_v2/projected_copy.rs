//! Independently owned selected payloads, including nullable and recursive arrays.

use std::sync::Arc;

use datafusion::{
    arrow::{
        array::{Array, ArrayRef},
        datatypes::SchemaRef,
        record_batch::RecordBatch,
    },
    execution::memory_pool::MemoryReservation,
};

use super::super::{
    StreamJoinOperator, canonical_column,
    columnar::{OwnedSelection, SelectedRow, SelectedRows},
    concat_output_column, output_gather,
};
use super::{
    budget,
    geometry::invalid,
    ipc::{
        accounting::{self, add, product, reserve, sum},
        concat,
    },
    payload::{Funding, OwnedPayload},
};
use crate::runtime::streaming::gather_work::{
    GatherOperatorId, GatherStop, ObservedTicket, OwnedCpuWork, RetirementGuard,
    cleanup_control_bytes,
};
use crate::{Result, StreamOperatorContext};

pub(in crate::operator::join) mod views;

#[cfg(test)]
pub(in crate::operator::join) mod test_support;

#[cfg(test)]
pub(in crate::operator::join) fn assert_selected_copy_budget() {
    use datafusion::arrow::{
        array::{Float64Array, Int64Array, TimestampMicrosecondArray},
        datatypes::{Field, Schema},
    };
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
    let count = 4096;
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(
            (0..count)
                .map(|row| i64::try_from(row).unwrap())
                .collect::<Vec<_>>(),
        )),
        Arc::new(TimestampMicrosecondArray::from(vec![100; count]).with_timezone("UTC")),
        Arc::new(Float64Array::from(
            (0..count)
                .map(|row| (!row.is_multiple_of(3)).then_some(2.0))
                .collect::<Vec<_>>(),
        )),
    ];
    let fields = arrays
        .iter()
        .enumerate()
        .map(|(index, array)| Field::new(format!("c{index}"), array.data_type().clone(), true))
        .collect::<Vec<_>>();
    let record = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
    let pool = Arc::new(GreedyMemoryPool::new(1 << 20));
    let budget: Arc<dyn MemoryPool> = pool.clone();
    let credit = MemoryConsumer::new("selected-copy-budget").register(&budget);
    reserve(
        &credit,
        accounting::vector_peak::<SelectedRow>(count).unwrap(),
    )
    .unwrap();
    let rows = (0..count)
        .map(|source| SelectedRow {
            source,
            row_id: u64::try_from(source).unwrap(),
            time: crate::EventTime::from_micros(100),
        })
        .collect::<Vec<_>>();
    let copied = copy(
        &record,
        &rows,
        [record.schema(), record.schema()],
        &credit,
        None,
        &|| Ok(()),
    );
    let equivalent = copied
        .as_ref()
        .map(|copy| copy.record() == &record)
        .map_err(ToString::to_string);
    drop((copied, rows, credit));
    assert_eq!(pool.reserved(), 0);
    assert!(
        equivalent.unwrap(),
        "selected contiguous copy must fit its bounded budget"
    );
    assert_sparse_copy(&record, &budget);
}

#[cfg(test)]
fn assert_sparse_copy(
    record: &RecordBatch,
    budget: &Arc<dyn datafusion::execution::memory_pool::MemoryPool>,
) {
    use datafusion::arrow::{array::UInt64Array, compute::take};
    use datafusion::execution::memory_pool::MemoryConsumer;
    let sources = [2, 0, 1, 1, 7, 8];
    let credit = MemoryConsumer::new("sparse-copy-budget").register(budget);
    reserve(
        &credit,
        accounting::vector_peak::<SelectedRow>(sources.len()).unwrap(),
    )
    .unwrap();
    let rows = sources
        .iter()
        .map(|&source| SelectedRow {
            source,
            row_id: u64::try_from(source).unwrap(),
            time: crate::EventTime::from_micros(100),
        })
        .collect::<Vec<_>>();
    let indices = UInt64Array::from(
        sources
            .iter()
            .map(|&row| u64::try_from(row).unwrap())
            .collect::<Vec<_>>(),
    );
    let expected = RecordBatch::try_new(
        record.schema(),
        record
            .columns()
            .iter()
            .map(|array| take(array, &indices, None).unwrap())
            .collect(),
    )
    .unwrap();
    let copied = copy(
        record,
        &rows,
        [record.schema(), record.schema()],
        &credit,
        None,
        &|| Ok(()),
    );
    let equivalent = copied
        .as_ref()
        .map(|copy| copy.record() == &expected)
        .map_err(ToString::to_string);
    drop((copied, rows, credit));
    assert_eq!(budget.reserved(), 0);
    assert!(
        equivalent.unwrap(),
        "coalesced copy changed reordered, duplicate or sparse selections"
    );
}

struct CopyWork {
    record: RecordBatch,
    rows: OwnedSelection,
    schemas: [SchemaRef; 2],
    retirement: Option<RetirementGuard>,
    input_controls: MemoryReservation,
    #[cfg(test)]
    hook: Option<test_support::Hook>,
}

impl OwnedCpuWork for CopyWork {
    type Output = Arc<OwnedPayload>;

    fn control_bytes(&self) -> Result<usize> {
        sum(&[
            size_of::<Self>(),
            cleanup_control_bytes::<Arc<OwnedPayload>>(),
        ])
    }

    fn run(self, stop: &GatherStop) -> Result<Self::Output> {
        #[cfg(not(test))]
        {
            copy_selected(
                &self.record,
                self.rows.rows(),
                self.schemas.clone(),
                &self.input_controls,
                self.retirement,
                &|| stop.check(),
            )
        }
        #[cfg(test)]
        {
            let hook = self.hook.clone();
            test_support::note(hook.as_ref(), false);
            let result = copy_selected(
                &self.record,
                self.rows.rows(),
                self.schemas.clone(),
                &self.input_controls,
                self.retirement,
                &|| stop.check(),
            );
            if result.is_ok() {
                test_support::note(hook.as_ref(), true);
            }
            result
        }
    }
}

impl StreamJoinOperator {
    pub(in crate::operator::join) async fn copy_projected_payload(
        &mut self,
        record: RecordBatch,
        rows: OwnedSelection,
        input_controls: MemoryReservation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Arc<OwnedPayload>> {
        let scope = context
            .gather_client(GatherOperatorId::new(Arc::from(self.name.as_str())))
            .scope()?;
        let stop = GatherStop::from_job(context.job());
        let credit = input_controls.new_empty();
        let retirement = context.job().gather_owner().retain_retirement()?;
        let work = CopyWork {
            record,
            rows,
            schemas: [
                Arc::clone(self.retained_schema(0)),
                Arc::clone(self.retained_schema(1)),
            ],
            retirement: Some(context.job().gather_owner().retain_retirement()?),
            input_controls,
            #[cfg(test)]
            hook: test_support::hook(),
        };
        let ticket = scope
            .submit_observed_work(
                work,
                credit,
                stop.clone(),
                retirement,
                &mut self.compaction_cleanup,
            )
            .await
            .map_err(|failure| match failure {
                crate::runtime::streaming::gather_work::AdmissionFailure::Runtime(error) => error,
                crate::runtime::streaming::gather_work::AdmissionFailure::Budget {
                    source, ..
                } => crate::CalcFlowError::DataFusion {
                    node_id: Some(self.name.clone()),
                    message: source.to_string(),
                },
            });
        context.check_cancelled()?;
        let ticket = ticket?;
        self.finish_projected_copy(ticket, &stop, context).await
    }

    async fn finish_projected_copy(
        &mut self,
        ticket: ObservedTicket<Arc<OwnedPayload>>,
        stop: &GatherStop,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Arc<OwnedPayload>> {
        let result = tokio::select! {
            biased;
            () = output_gather::cancelled(context.job()) => {
                context.check_cancelled()?;
                unreachable!("retained copy stop requires cancellation or deadline");
            }
            result = ticket.finish() => result?,
        };
        context.check_cancelled()?;
        let mut copied = None;
        result.install(|value| {
            copied = Some(value);
            Ok(())
        })?;
        if let Some(cleanup) = self.compaction_cleanup.as_ref() {
            cleanup.wait(stop).await?;
        }
        self.compaction_cleanup = None;
        context.check_cancelled()?;
        Ok(copied.expect("one installed selected payload"))
    }
}

#[cfg(test)]
pub(super) fn copy(
    record: &RecordBatch,
    rows: &[SelectedRow],
    schemas: [SchemaRef; 2],
    credit: &MemoryReservation,
    retirement: Option<RetirementGuard>,
    check: &dyn Fn() -> Result<()>,
) -> Result<Arc<OwnedPayload>> {
    check()?;
    copy_selected(record, rows.into(), schemas, credit, retirement, check)
}

fn copy_selected(
    record: &RecordBatch,
    rows: SelectedRows<'_>,
    schemas: [SchemaRef; 2],
    credit: &MemoryReservation,
    retirement: Option<RetirementGuard>,
    check: &dyn Fn() -> Result<()>,
) -> Result<Arc<OwnedPayload>> {
    check()?;
    let workspace = credit.new_empty();
    let resident = credit.new_empty();
    reserve(&resident, seed(record.schema_ref())?)?;
    let funding = Funding::new(schemas, resident, retirement);
    copy_funded(record, rows, &workspace, &funding, check)
}

pub(super) fn copy_shared(
    record: &RecordBatch,
    rows: &[SelectedRow],
    schemas: [SchemaRef; 2],
    credit: &MemoryReservation,
    retirement: Option<Arc<RetirementGuard>>,
    check: &dyn Fn() -> Result<()>,
) -> Result<Arc<OwnedPayload>> {
    check()?;
    let workspace = credit.new_empty();
    let resident = credit.new_empty();
    reserve(
        &resident,
        sum(&[
            seed(record.schema_ref())?,
            accounting::arc::<RetirementGuard>()?,
        ])?,
    )?;
    let funding = Funding::shared(schemas, resident, retirement);
    copy_funded(record, rows.into(), &workspace, &funding, check)
}

fn copy_funded(
    record: &RecordBatch,
    rows: SelectedRows<'_>,
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<Arc<OwnedPayload>> {
    let mut columns = Vec::with_capacity(record.num_columns());
    for array in record.columns() {
        check()?;
        columns.push(copy_column(array, rows, workspace, funding, check)?);
    }
    let copied = RecordBatch::try_new(record.schema(), columns)
        .map_err(|_| invalid("retained selected record reconstruction failed"))?;
    OwnedPayload::rebind(&copied, record.schema(), funding, check)
}

fn seed(schema: &SchemaRef) -> Result<usize> {
    sum(&[
        budget::payload_seed()?,
        accounting::arc::<OwnedPayload>()?,
        accounting::schema_bytes(schema)?,
        product(schema.fields().len(), size_of::<ArrayRef>())?,
        size_of::<RecordBatch>(),
    ])
}

fn copy_column(
    array: &ArrayRef,
    rows: SelectedRows<'_>,
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    let slices_credit = workspace.new_empty();
    let slices = selected_slices(array, rows, &slices_credit, check)?;
    let references = slices.iter().map(AsRef::as_ref).collect::<Vec<_>>();
    let copied = copied_column(array, &references, workspace, funding, check)?;
    admit_rebind(&copied, workspace, funding, check)?;
    check()?;
    Ok(copied)
}

fn selected_slices(
    array: &ArrayRef,
    rows: SelectedRows<'_>,
    slices_credit: &MemoryReservation,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<ArrayRef>> {
    let runs = contiguous_runs(rows, check)?;
    reserve(slices_credit, slice_bytes(array.as_ref(), runs, check)?)?;
    slice_runs(array, rows, runs, check)
}

fn contiguous_runs(rows: SelectedRows<'_>, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    let mut runs = 0;
    let mut previous: Option<usize> = None;
    for row in rows.iter() {
        check()?;
        if previous.is_none_or(|previous| previous.checked_add(1) != Some(row.source)) {
            runs = add(runs, 1)?;
        }
        previous = Some(row.source);
    }
    Ok(runs)
}

fn slice_runs(
    array: &ArrayRef,
    rows: SelectedRows<'_>,
    runs: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<Vec<ArrayRef>> {
    let mut slices = Vec::with_capacity(runs);
    let mut start = 0;
    while start < rows.len() {
        check()?;
        let end = run_end(rows, start, check)?;
        slices.push(array.slice(rows.at(start).source, end - start));
        start = end;
    }
    debug_assert_eq!(slices.len(), runs);
    Ok(slices)
}

fn run_end(rows: SelectedRows<'_>, start: usize, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    for index in start + 1..rows.len() {
        check()?;
        if rows.at(index - 1).source.checked_add(1) != Some(rows.at(index).source) {
            return Ok(index);
        }
    }
    Ok(rows.len())
}

fn copied_column(
    array: &ArrayRef,
    references: &[&dyn Array],
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<ArrayRef> {
    concat::bulk::admit(references, workspace, funding, check)?;
    check()?;
    let copied = canonical_column(concat_output_column(references)?, array.data_type());
    views::independent(copied, workspace, funding, check)
}

fn admit_rebind(
    copied: &ArrayRef,
    workspace: &MemoryReservation,
    funding: &Arc<Funding>,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    reserve(workspace, concat::slice_controls(copied.as_ref(), check)?)?;
    let data = copied.to_data();
    funding.grow(rebind_bytes(&data)?)?;
    check()
}

fn slice_bytes(array: &dyn Array, rows: usize, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    sum(&[
        accounting::vector_peak::<ArrayRef>(rows)?,
        accounting::vector_peak::<&dyn Array>(rows)?,
        product(rows, slice_layout(array, check)?)?,
    ])
}

fn slice_layout(array: &dyn Array, check: &dyn Fn() -> Result<()>) -> Result<usize> {
    let nodes = accounting::shape_nodes(array.data_type())?;
    add(
        concat::slice_controls(array, check)?,
        accounting::array_controls(nodes, product(nodes, 3)?)?,
    )
}

fn rebind_bytes(data: &arrow_data::ArrayData) -> Result<usize> {
    let buffers = add(data.buffers().len(), usize::from(data.nulls().is_some()))?;
    let own = sum(&[
        accounting::array_controls(2, product(buffers, 2)?)?,
        accounting::data_type_bytes(data.data_type())?,
        product(buffers, super::payload::buffer_owner_bytes())?,
    ])?;
    data.child_data()
        .iter()
        .try_fold(own, |bytes, child| add(bytes, rebind_bytes(child)?))
}
