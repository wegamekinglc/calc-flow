use super::{
    ColumnRequest, GatherPlan, OutputRuntime, column_requests, materialize_batch,
    materialize_column,
};
use crate::{
    Batch, Result, StreamOperatorContext,
    operator::asof::output_plan::{FragmentCredits, OutputPlan},
    runtime::streaming::gather_work::{
        AdmissionFailure, GatherPlan as OwnedGatherPlan, GatherStop, RowGather,
    },
};
use datafusion::{
    arrow::{
        array::{Array, ArrayRef},
        compute::concat,
        datatypes::SchemaRef,
    },
    execution::memory_pool::MemoryReservation,
};
use std::sync::Arc;

struct SegmentedInput {
    plans: Vec<OutputPlan>,
    requests: Vec<ColumnRequest>,
    shared: Vec<Option<ArrayRef>>,
    left_fields: usize,
    rows: usize,
    _credits: Arc<FragmentCredits>,
}

impl OwnedGatherPlan for SegmentedInput {
    fn column_count(&self) -> usize {
        self.requests.len()
    }

    fn parallelism(&self) -> usize {
        (self.rows / 16_384).max(self.column_count()).max(1)
    }

    fn shared_column(&self, ordinal: usize) -> Result<Option<ArrayRef>> {
        Ok(self.shared[ordinal].clone())
    }

    fn row_gather(&self) -> Result<Option<RowGather>> {
        if self.rows / 16_384 < 2 {
            return Ok(None);
        }
        let mut copied = self
            .shared
            .iter()
            .enumerate()
            .filter_map(|(ordinal, column)| column.is_none().then_some(ordinal));
        let Some(ordinal) = copied.next() else {
            return Ok(None);
        };
        if copied.next().is_some() || self.requests[ordinal].index < self.left_fields {
            return Ok(None);
        }
        let Some(width) = self.requests[ordinal].data_type.primitive_width() else {
            return Ok(None);
        };
        Ok(Some(RowGather {
            ordinal,
            rows: self.rows,
            width,
            sources: self.plans.iter().map(|plan| plan.right.batches.len()).sum(),
        }))
    }

    fn gather_range(
        &self,
        ordinal: usize,
        range: std::ops::Range<usize>,
        stop: &GatherStop,
    ) -> Result<ArrayRef> {
        #[cfg(feature = "benchmark-internals")]
        let _phase = stop.join_cpu_trace().worker_stage(
            stop.join_cpu_operator(),
            "gather_worker",
            range.len(),
        );
        stop.check()?;
        let request = &self.requests[ordinal];
        if range.is_empty() || range.end > self.rows || request.index < self.left_fields {
            return Err(super::super::arrow_error(
                &datafusion::arrow::error::ArrowError::InvalidArgumentError(
                    "invalid segmented ASOF gather range".into(),
                ),
            ));
        }
        let mut offset = 0;
        let mut pieces = Vec::new();
        for plan in &self.plans {
            let start = range.start.saturating_sub(offset);
            let end = range.end.saturating_sub(offset).min(plan.len);
            if start < end {
                stop.check()?;
                let mut right = GatherPlan::new(&plan.right, true);
                right.positions = right.positions.map(|positions| &positions[start..end]);
                pieces.push(right.column(
                    request.index - self.left_fields,
                    &request.data_type,
                    end - start,
                )?);
            }
            offset += plan.len;
            if offset >= range.end {
                break;
            }
        }
        concatenate(pieces, stop)
    }

    fn gather(&self, ordinal: usize, stop: &GatherStop) -> Result<ArrayRef> {
        #[cfg(feature = "benchmark-internals")]
        let _phase = stop.join_cpu_trace().worker_stage(
            stop.join_cpu_operator(),
            "gather_worker",
            self.rows,
        );
        stop.check()?;
        if let Some(column) = &self.shared[ordinal] {
            return Ok(column.clone());
        }
        let mut pieces = Vec::with_capacity(self.plans.len());
        for plan in &self.plans {
            stop.check()?;
            pieces.push(materialize_column(
                &GatherPlan::new(&plan.left, false),
                &GatherPlan::new(&plan.right, true),
                plan.len,
                &self.requests[ordinal],
            )?);
        }
        concatenate(pieces, stop)
    }
}

fn concatenate(mut pieces: Vec<ArrayRef>, stop: &GatherStop) -> Result<ArrayRef> {
    stop.check()?;
    if pieces.len() == 1 {
        return Ok(pieces.pop().expect("one fragment array"));
    }
    let inputs = pieces.iter().map(AsRef::as_ref).collect::<Vec<_>>();
    let output = concat(&inputs).map_err(|error| super::super::arrow_error(&error))?;
    stop.check()?;
    Ok(output)
}

fn shared(
    plans: &[OutputPlan],
    request: &ColumnRequest,
    left_fields: usize,
) -> Result<Option<ArrayRef>> {
    let right = request.index >= left_fields;
    let index = if right {
        request.index - left_fields
    } else {
        request.index
    };
    let mut source: Option<ArrayRef> = None;
    let mut end = 0;
    for plan in plans {
        let side = if right { &plan.right } else { &plan.left };
        let [span] = side.spans.as_slice() else {
            return Ok(None);
        };
        if side.has_nulls || span.start != end {
            return Ok(None);
        }
        let column = &side.batches[span.source][index];
        if source
            .as_ref()
            .is_some_and(|source| !Arc::ptr_eq(source, column))
        {
            return Ok(None);
        }
        source = Some(column.clone());
        end = span.end;
    }
    let Some(source) = source else {
        return Ok(None);
    };
    if end != source.len() {
        return Ok(None);
    }
    let data = source.to_data();
    let visible = data
        .get_slice_memory_size()
        .map_err(|error| super::super::arrow_error(&error))?;
    Ok((data.get_buffer_memory_size() <= visible).then_some(source))
}

impl OutputRuntime {
    pub(in super::super) async fn materialize_fragments(
        &mut self,
        plans: Vec<OutputPlan>,
        credits: Arc<FragmentCredits>,
        schema: &SchemaRef,
        mut workspace: MemoryReservation,
        name: &str,
        context: &StreamOperatorContext<'_>,
    ) -> Result<(Batch, MemoryReservation)> {
        let rows = plans.iter().map(|plan| plan.len).sum();
        #[cfg(feature = "benchmark-internals")]
        let _phase = context
            .job()
            .join_cpu_trace()
            .stage(name, "materialize", rows);
        self.prepare_output(schema, &mut workspace, name, context)
            .await?;
        let requests = column_requests(schema, self.output_columns.as_deref());
        let left_fields = plans
            .first()
            .and_then(|plan| plan.left.batches.first())
            .map_or(0, Vec::len);
        let shared = requests
            .iter()
            .map(|request| shared(&plans, request, left_fields))
            .collect::<Result<Vec<_>>>()?;
        if shared.iter().all(Option::is_some) {
            return Ok((
                materialize_batch(
                    shared.into_iter().map(Option::unwrap).collect(),
                    schema,
                    rows,
                )?,
                workspace,
            ));
        }
        let input = Arc::new(SegmentedInput {
            plans,
            requests,
            shared,
            left_fields,
            rows,
            _credits: credits,
        });
        let scope = context.gather_client(self.operator.clone()).scope()?;
        let ticket = scope
            .submit(input, workspace, GatherStop::from_job(context.job()))
            .await
            .map_err(|failure| match failure {
                AdmissionFailure::Budget { stage, source } => super::super::reason(
                    name,
                    crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
                    &format!("native {stage} admission failed: {source}"),
                ),
                AdmissionFailure::Runtime(error) => error,
            })?;
        let output = ticket.finish().await?;
        Ok((
            materialize_batch(output.value, schema, rows)?,
            output.credit,
        ))
    }
}

#[cfg(test)]
mod tests;
