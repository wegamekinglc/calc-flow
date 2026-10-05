use super::{
    Arc, ExecutionProps, LogicalPlan, MemoryReservation, PhysicalExpr, RecordBatch, Result,
    SchemaRef, StreamOperatorContext, checked_bytes, create_physical_expr, df_error,
    ensure_reservation,
};
use datafusion::{
    arrow::compute::{SortColumn, SortOptions, concat_batches, lexsort_to_indices, take},
    logical_expr::{FetchType, Projection, SkipType, Sort},
};

use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherStop, OwnedCpuWork,
};

use super::native_expression::{NativeAggregateInput, describe_input, expression_work};

struct Shape<'a> {
    projection: &'a Projection,
    sort: Option<&'a Sort>,
    skip: usize,
    fetch: Option<usize>,
    limited: bool,
}

fn shape(mut plan: &LogicalPlan) -> Option<Shape<'_>> {
    let mut skip = 0;
    let mut fetch = None;
    let mut limited = false;
    if let LogicalPlan::Limit(limit) = plan {
        let SkipType::Literal(value) = limit.get_skip_type().ok()? else {
            return None;
        };
        let FetchType::Literal(count) = limit.get_fetch_type().ok()? else {
            return None;
        };
        skip = value;
        fetch = count;
        limited = true;
        plan = &limit.input;
    }
    let sort = if let LogicalPlan::Sort(sort) = plan {
        plan = &sort.input;
        Some(sort)
    } else {
        None
    };
    let LogicalPlan::Projection(projection) = plan else {
        return None;
    };
    Some(Shape {
        projection,
        sort,
        skip,
        fetch,
        limited,
    })
}

pub(super) fn projection(plan: &LogicalPlan) -> Option<&Projection> {
    shape(plan).map(|shape| shape.projection)
}

pub(super) fn work_nodes(plan: &LogicalPlan) -> Option<usize> {
    let shape = shape(plan)?;
    shape
        .sort
        .into_iter()
        .flat_map(|sort| &sort.expr)
        .try_fold(usize::from(shape.limited), |total, expression| {
            total.checked_add(expression_work(&expression.expr)?)
        })
}

#[derive(Clone)]
struct Key {
    expression: Arc<dyn PhysicalExpr>,
    options: SortOptions,
}

pub(super) struct OutputOrder {
    keys: Vec<Key>,
    skip: usize,
    fetch: Option<usize>,
    operator: GatherOperatorId,
}

pub(super) enum OrderPlan {
    Identity,
    Ordered(OutputOrder),
}

pub(in crate::operator::sql) struct OrderDescriptor {
    pub keys: Vec<(NativeAggregateInput, bool, bool)>,
    pub skip: usize,
    pub fetch: Option<usize>,
}

impl OutputOrder {
    pub(super) fn project_after_limit(&self) -> bool {
        self.keys.is_empty()
    }

    pub(super) fn bind(plan: &LogicalPlan, name: &str) -> Option<OrderPlan> {
        let shape = shape(plan)?;
        if !shape.limited && shape.sort.is_none() {
            return Some(OrderPlan::Identity);
        }
        let props = ExecutionProps::new();
        let keys = shape
            .sort
            .into_iter()
            .flat_map(|sort| &sort.expr)
            .map(|sort| {
                expression_work(&sort.expr)?;
                Some(Key {
                    expression: create_physical_expr(&sort.expr, &shape.projection.schema, &props)
                        .ok()?,
                    options: SortOptions {
                        descending: !sort.asc,
                        nulls_first: sort.nulls_first,
                    },
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let sort_fetch = shape
            .sort
            .and_then(|sort| sort.fetch)
            .map(|fetch| fetch.saturating_sub(shape.skip));
        let fetch = match (shape.fetch, sort_fetch) {
            (Some(outer), Some(inner)) => Some(outer.min(inner)),
            (Some(fetch), None) | (None, Some(fetch)) => Some(fetch),
            (None, None) => None,
        };
        Some(OrderPlan::Ordered(Self {
            keys,
            skip: shape.skip,
            fetch,
            operator: GatherOperatorId::new(name.to_owned().into()),
        }))
    }

    pub(super) fn descriptor(&self, schema: &SchemaRef, name: &str) -> Result<OrderDescriptor> {
        let keys = self
            .keys
            .iter()
            .map(|key| {
                Ok((
                    describe_input(key.expression.as_ref(), schema, 0, name)?,
                    key.options.descending,
                    key.options.nulls_first,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(OrderDescriptor {
            keys,
            skip: self.skip,
            fetch: self.fetch,
        })
    }

    pub(super) async fn apply(
        &self,
        records: Vec<RecordBatch>,
        reservation: &MemoryReservation,
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<Vec<RecordBatch>> {
        if self.keys.is_empty() {
            return Ok(clip(records, self.skip, self.fetch));
        }
        let width = checked_bytes(0, [(self.keys.len(), 256)], name)?;
        let bytes = records.iter().try_fold(8192, |bytes, record| {
            checked_bytes(
                bytes,
                [
                    (record.get_array_memory_size(), 4),
                    (record.num_rows(), width),
                    (1, 512),
                ],
                name,
            )
        })?;
        let credit = reservation.new_empty();
        ensure_reservation(&credit, bytes, name)?;
        let schema = records
            .first()
            .ok_or_else(|| df_error(name, "SQL ordered snapshot is absent"))?
            .schema();
        let work = SortWork {
            records,
            keys: self.keys.clone(),
            schema,
            skip: self.skip,
            fetch: self.fetch,
            name: name.to_owned(),
        };
        let scope = context.gather_client(self.operator.clone()).scope()?;
        let ticket = scope
            .submit_work(work, credit, GatherStop::from_job(context.job()))
            .await
            .map_err(|failure| match failure {
                AdmissionFailure::Budget { source, .. } => df_error(name, source),
                AdmissionFailure::Runtime(error) => error,
            })?;
        let output = ticket.finish().await?;
        context.check_cancelled()?;
        let records = output.value.clone();
        drop(output);
        Ok(records)
    }
}

fn clip(records: Vec<RecordBatch>, mut skip: usize, fetch: Option<usize>) -> Vec<RecordBatch> {
    let mut left = fetch.unwrap_or(usize::MAX);
    records
        .into_iter()
        .filter_map(|record| {
            let start = skip.min(record.num_rows());
            skip -= start;
            let count = left.min(record.num_rows() - start);
            left -= count;
            (count != 0).then(|| record.slice(start, count))
        })
        .collect()
}

struct SortWork {
    records: Vec<RecordBatch>,
    keys: Vec<Key>,
    schema: SchemaRef,
    skip: usize,
    fetch: Option<usize>,
    name: String,
}

impl OwnedCpuWork for SortWork {
    type Output = Vec<RecordBatch>;

    fn run(self, stop: &GatherStop) -> Result<Self::Output> {
        stop.check()?;
        let input = concat_batches(&self.schema, &self.records)
            .map_err(|error| df_error(&self.name, error))?;
        let keys = self
            .keys
            .iter()
            .map(|key| {
                stop.check()?;
                let values = key
                    .expression
                    .evaluate(&input)
                    .and_then(|value| value.into_array(input.num_rows()))
                    .map_err(|error| df_error(&self.name, error))?;
                Ok(SortColumn {
                    values,
                    options: Some(key.options),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let indices = lexsort_to_indices(
            &keys,
            self.fetch.map(|fetch| self.skip.saturating_add(fetch)),
        )
        .map_err(|error| df_error(&self.name, error))?;
        let start = self.skip.min(indices.len());
        let count = indices.len() - start;
        let indices = indices.slice(start, count);
        let columns = input
            .columns()
            .iter()
            .map(|column| {
                stop.check()?;
                take(column.as_ref(), &indices, None).map_err(|error| df_error(&self.name, error))
            })
            .collect::<Result<Vec<_>>>()?;
        stop.check()?;
        Ok(vec![
            RecordBatch::try_new(self.schema, columns)
                .map_err(|error| df_error(&self.name, error))?,
        ])
    }
}
