//! Materialize an already matched ASOF prefix directly from Arrow batches.

use super::output_plan::{OutputPlan, OutputSide, Span};
#[cfg(test)]
use super::state::{BatchKey, PayloadView};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherPlan as OwnedGatherPlan, GatherStop, RowGather,
};
use crate::{Batch, BatchMetadata, DataFusionConfig, Result, StreamOperatorContext};
#[cfg(test)]
use ahash::RandomState;
use arrow_data::transform::MutableArrayData;
use datafusion::execution::memory_pool::MemoryReservation;
use datafusion::{
    arrow::{
        array::{Array, ArrayRef, make_array, new_null_array},
        compute::interleave,
        datatypes::{DataType, SchemaRef},
        record_batch::{RecordBatch, RecordBatchOptions},
    },
    execution::memory_pool::{GreedyMemoryPool, MemoryPool},
};
#[cfg(test)]
use std::collections::HashMap;
use std::sync::Arc;

pub(super) struct OutputRuntime {
    pub pool: Arc<dyn MemoryPool>,
    config: DataFusionConfig,
    output_columns: Option<Vec<usize>>,
    operator: GatherOperatorId,
    #[cfg(test)]
    worker_gate: Option<(std::sync::mpsc::Sender<()>, Arc<std::sync::Barrier>)>,
    #[cfg(test)]
    worker_probe: Option<Arc<gather_lifecycle_bridge::WorkerProbe>>,
}

struct ColumnRequest {
    index: usize,
    data_type: DataType,
}

struct MaterializationInput {
    rows: OutputPlan,
    requests: Vec<ColumnRequest>,
    #[cfg(test)]
    worker_gate: Option<(std::sync::mpsc::Sender<()>, Arc<std::sync::Barrier>)>,
    #[cfg(test)]
    worker_probe: Option<Arc<gather_lifecycle_bridge::WorkerProbe>>,
}

impl OwnedGatherPlan for MaterializationInput {
    fn column_count(&self) -> usize {
        self.requests.len()
    }

    fn row_gather(&self) -> Result<Option<RowGather>> {
        if self.rows.len < 2 || self.parallelism() <= 1 {
            return Ok(None);
        }
        let mut copied = None;
        for ordinal in 0..self.requests.len() {
            if self.shared_column(ordinal)?.is_none() && copied.replace(ordinal).is_some() {
                return Ok(None);
            }
        }
        let Some(ordinal) = copied else {
            return Ok(None);
        };
        let request = &self.requests[ordinal];
        let left_fields = self.rows.left.batches.first().map_or(0, Vec::len);
        if request.index < left_fields {
            return Ok(None);
        }
        let Some(width) = request.data_type.primitive_width() else {
            return Ok(None);
        };
        Ok(Some(RowGather {
            ordinal,
            rows: self.rows.len,
            width,
            sources: self.rows.right.batches.len(),
        }))
    }

    fn shared_column(&self, ordinal: usize) -> Result<Option<ArrayRef>> {
        let request = &self.requests[ordinal];
        let left_fields = self.rows.left.batches.first().map_or(0, Vec::len);
        if request.index < left_fields {
            GatherPlan::new(&self.rows.left, false).shared_column(request.index)
        } else {
            Ok(None)
        }
    }

    fn gather_range(
        &self,
        ordinal: usize,
        range: std::ops::Range<usize>,
        stop: &GatherStop,
    ) -> Result<ArrayRef> {
        stop.check()?;
        if range.is_empty() || range.end > self.rows.len {
            return Err(super::arrow_error(
                &datafusion::arrow::error::ArrowError::InvalidArgumentError(
                    "invalid ASOF row range".into(),
                ),
            ));
        }
        let request = &self.requests[ordinal];
        let left_fields = self.rows.left.batches.first().map_or(0, Vec::len);
        if request.index < left_fields {
            return Err(super::arrow_error(
                &datafusion::arrow::error::ArrowError::InvalidArgumentError(
                    "left row gather unsupported".into(),
                ),
            ));
        }
        let mut right = GatherPlan::new(&self.rows.right, true);
        right.positions = right.positions.map(|positions| &positions[range.clone()]);
        right.column(request.index - left_fields, &request.data_type, range.len())
    }

    fn gather(&self, ordinal: usize, stop: &GatherStop) -> Result<ArrayRef> {
        #[cfg(test)]
        if ordinal == 0 {
            if let Some(probe) = &self.worker_probe {
                probe.wait();
            }
            if let Some((started, gate)) = &self.worker_gate {
                let _ = started.send(());
                gate.wait();
            }
        }
        stop.check()?;
        let left = GatherPlan::new(&self.rows.left, false);
        let right = GatherPlan::new(&self.rows.right, true);
        materialize_column(&left, &right, self.rows.len, &self.requests[ordinal])
    }
}

impl OutputRuntime {
    pub fn new(limit: usize, name: &str) -> Self {
        Self {
            pool: Arc::new(GreedyMemoryPool::new(limit)),
            config: DataFusionConfig::default(),
            output_columns: None,
            operator: GatherOperatorId::new(format!("operator:{name}").into()),
            #[cfg(test)]
            worker_gate: None,
            #[cfg(test)]
            worker_probe: None,
        }
    }

    pub fn configure(&mut self, config: DataFusionConfig) {
        self.config = config;
    }

    pub fn set_output_projection(&mut self, columns: Vec<usize>) {
        self.output_columns = Some(columns);
    }

    pub async fn materialize_plan(
        &mut self,
        owned: OutputPlan,
        schema: &SchemaRef,
        mut workspace: MemoryReservation,
        name: &str,
        context: &StreamOperatorContext<'_>,
    ) -> Result<(Batch, MemoryReservation)> {
        self.config.validate()?;
        tokio::task::yield_now().await;
        context.check_cancelled()?;
        reserve_column_types(schema, &mut workspace, name)?;
        let requests = column_requests(schema, self.output_columns.as_deref());
        let rows = owned.len;
        if requests.is_empty() {
            return Ok((materialize_batch(Vec::new(), schema, rows)?, workspace));
        }
        let left_fields = owned.left.batches.first().map_or(0, Vec::len);
        if requests.iter().all(|request| request.index < left_fields) {
            let left = GatherPlan::new(&owned.left, false);
            let shared = requests
                .iter()
                .map(|request| left.shared_column(request.index))
                .collect::<Result<Option<Vec<_>>>>()?;
            if let Some(columns) = shared {
                context.check_cancelled()?;
                return Ok((materialize_batch(columns, schema, rows)?, workspace));
            }
        }
        let input = Arc::new(MaterializationInput {
            rows: owned,
            requests,
            #[cfg(test)]
            worker_gate: self.worker_gate.clone(),
            #[cfg(test)]
            worker_probe: self.worker_probe.clone(),
        });
        let client = context.gather_client(self.operator.clone());
        let scope = client.scope()?;
        let ticket = scope
            .submit(input, workspace, GatherStop::from_job(context.job()))
            .await
            .map_err(|failure| match failure {
                AdmissionFailure::Budget { stage, source } => super::reason(
                    name,
                    crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
                    &format!("native {stage} admission failed: {source}"),
                ),
                AdmissionFailure::Runtime(error) => error,
            })?;
        let output = ticket.finish().await?;
        let batch = materialize_batch(output.value, schema, rows)?;
        Ok((batch, output.credit))
    }
}

fn reserve_column_types(
    schema: &SchemaRef,
    workspace: &mut MemoryReservation,
    name: &str,
) -> Result<()> {
    let types = schema.fields().iter().try_fold(256, |total, field| {
        super::checked(
            name,
            total,
            super::checked(name, field.data_type().size() as u64, 512)?,
        )
    })?;
    super::output_plan::grow_workspace(workspace, types, name)
}

#[cfg(test)]
impl OutputRuntime {
    async fn materialize(
        &mut self,
        rows: &[(PayloadView<'_>, Option<PayloadView<'_>>)],
        schema: &SchemaRef,
        workspace: MemoryReservation,
        check_cancelled: impl Fn() -> Result<()> + Send + Sync,
    ) -> Result<(Batch, MemoryReservation)> {
        let owned = OutputRows::capture(rows, &check_cancelled).await?;
        let plan = OutputPlan {
            left: owned.left,
            right: owned.right,
            len: owned.len,
            matched: rows.iter().filter(|(_, right)| right.is_some()).count() as u64,
            raw_bytes: 0,
        };
        let job = crate::StreamJobContext::new(
            0,
            "asof-output-test",
            crate::JsonMap::new(),
            None,
            crate::CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "asof", None);
        let result = self
            .materialize_plan(plan, schema, workspace, "asof", &context)
            .await;
        job.gather_owner().close_and_drain().await;
        result
    }
}

#[cfg(test)]
struct OutputRows {
    left: OutputSide,
    right: OutputSide,
    len: usize,
}

#[cfg(test)]
type SourceMap = HashMap<BatchKey, usize, RandomState>;

#[cfg(test)]
impl OutputRows {
    async fn capture(
        rows: &[(PayloadView<'_>, Option<PayloadView<'_>>)],
        check_cancelled: &(impl Fn() -> Result<()> + Sync),
    ) -> Result<Self> {
        let mut output = Self::empty(rows.len());
        let mut left_sources = SourceMap::with_hasher(RandomState::new());
        let mut right_sources = SourceMap::with_hasher(RandomState::new());
        for (ordinal, chunk) in rows.chunks(1_024).enumerate() {
            if ordinal > 0 {
                tokio::task::yield_now().await;
            }
            check_cancelled()?;
            for &(left, right) in chunk {
                output.left.push(Some(left), &mut left_sources, false);
                output.right.push(right, &mut right_sources, true);
            }
        }
        check_cancelled()?;
        Ok(output)
    }

    fn empty(len: usize) -> Self {
        Self {
            left: OutputSide::empty(len),
            right: OutputSide::empty(len),
            len,
        }
    }

    #[cfg(test)]
    fn new(rows: &[(PayloadView<'_>, Option<PayloadView<'_>>)]) -> Self {
        let mut output = Self::empty(rows.len());
        let mut left_sources = SourceMap::with_hasher(RandomState::new());
        let mut right_sources = SourceMap::with_hasher(RandomState::new());
        for &(left, right) in rows {
            output.left.push(Some(left), &mut left_sources, false);
            output.right.push(right, &mut right_sources, true);
        }
        output
    }
}

#[cfg(test)]
impl OutputSide {
    fn empty(len: usize) -> Self {
        Self {
            batches: Vec::new(),
            positions: Vec::with_capacity(len),
            spans: Vec::new(),
            has_nulls: false,
        }
    }

    fn push(&mut self, selected: Option<PayloadView<'_>>, by_key: &mut SourceMap, right: bool) {
        let Some(row) = selected else {
            self.positions.push((0, 0));
            self.has_nulls = true;
            return;
        };
        let source = *by_key.entry(row.batch.key).or_insert_with(|| {
            #[cfg(test)]
            super::workspace::record_output_source_registration();
            let source = self.batches.len();
            self.batches.push(row.batch.record.columns().to_vec());
            source
        });
        self.positions.push((source + usize::from(right), row.row));
        if !right {
            if let Some(span) = self.spans.last_mut()
                && span.source == source
                && span.end == row.row
            {
                span.end += 1;
            } else {
                self.spans.push(Span {
                    source,
                    start: row.row,
                    end: row.row + 1,
                });
            }
        }
    }
}

struct GatherPlan<'a> {
    batches: &'a [Vec<ArrayRef>],
    spans: &'a [Span],
    positions: Option<&'a [(usize, usize)]>,
    has_nulls: bool,
}

impl<'a> GatherPlan<'a> {
    fn new(rows: &'a OutputSide, right: bool) -> Self {
        Self {
            batches: &rows.batches,
            spans: &rows.spans,
            positions: right.then_some(rows.positions.as_slice()),
            has_nulls: rows.has_nulls,
        }
    }

    fn column(&self, index: usize, data_type: &DataType, len: usize) -> Result<ArrayRef> {
        if self.batches.is_empty() {
            return Ok(new_null_array(data_type, len));
        }
        if let Some(positions) = &self.positions {
            let null_column = self.has_nulls.then(|| new_null_array(data_type, 1));
            let first = null_column
                .as_ref()
                .map_or_else(|| self.batches[0][index].as_ref(), |column| column.as_ref());
            let columns = std::iter::once(first)
                .chain(self.batches.iter().map(|batch| batch[index].as_ref()))
                .collect::<Vec<&dyn Array>>();
            return interleave(&columns, positions).map_err(|error| super::arrow_error(&error));
        }
        if let Some(column) = self.shared_column(index)? {
            return Ok(column);
        }
        let data = self
            .batches
            .iter()
            .map(|batch| batch[index].to_data())
            .collect::<Vec<_>>();
        let nullable = data.iter().any(|data| data.nulls().is_some());
        let mut mutable = MutableArrayData::new(data.iter().collect(), nullable, len);
        for span in self.spans {
            mutable.extend(span.source, span.start, span.end);
        }
        Ok(make_array(mutable.freeze()))
    }

    fn shared_column(&self, index: usize) -> Result<Option<ArrayRef>> {
        let [span] = self.spans else {
            return Ok(None);
        };
        let column = &self.batches[span.source][index];
        if span.start != 0 || span.end != column.len() {
            return Ok(None);
        }
        let data = column.to_data();
        // Queue budgets charge visible slices. Reuse a complete array only
        // when it retains no additional, uncharged backing bytes.
        let visible = data
            .get_slice_memory_size()
            .map_err(|error| super::arrow_error(&error))?;
        Ok((data.get_buffer_memory_size() <= visible).then(|| Arc::clone(column)))
    }
}

#[cfg(test)]
fn materialize_rows(
    rows: &[(PayloadView<'_>, Option<PayloadView<'_>>)],
    schema: &SchemaRef,
) -> Result<Batch> {
    let owned = OutputRows::new(rows);
    let left = GatherPlan::new(&owned.left, false);
    let right = GatherPlan::new(&owned.right, true);
    materialize_plans(&left, &right, rows.len(), schema, None)
}

fn column_requests(schema: &SchemaRef, output_columns: Option<&[usize]>) -> Vec<ColumnRequest> {
    schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| ColumnRequest {
            index: output_columns.map_or(index, |columns| columns[index]),
            data_type: field.data_type().clone(),
        })
        .collect()
}

fn materialize_column(
    left: &GatherPlan<'_>,
    right: &GatherPlan<'_>,
    len: usize,
    request: &ColumnRequest,
) -> Result<ArrayRef> {
    let left_fields = left.batches.first().map_or(0, Vec::len);
    if request.index < left_fields {
        left.column(request.index, &request.data_type, len)
    } else {
        right.column(request.index - left_fields, &request.data_type, len)
    }
}

#[cfg(test)]
fn materialize_columns(
    left: &GatherPlan<'_>,
    right: &GatherPlan<'_>,
    len: usize,
    requests: &[ColumnRequest],
) -> Result<Vec<ArrayRef>> {
    requests
        .iter()
        .map(|request| materialize_column(left, right, len, request))
        .collect()
}

fn materialize_batch(columns: Vec<ArrayRef>, schema: &SchemaRef, len: usize) -> Result<Batch> {
    let options = RecordBatchOptions::new().with_row_count(Some(len));
    let record = RecordBatch::try_new_with_options(schema.clone(), columns, &options)
        .map_err(|error| super::arrow_error(&error))?;
    Batch::table(vec![record], BatchMetadata::default())
}

#[cfg(test)]
fn materialize_plans(
    left: &GatherPlan<'_>,
    right: &GatherPlan<'_>,
    len: usize,
    schema: &SchemaRef,
    output_columns: Option<&[usize]>,
) -> Result<Batch> {
    let requests = column_requests(schema, output_columns);
    let columns = materialize_columns(left, right, len, &requests)?;
    materialize_batch(columns, schema, len)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) mod gather_lifecycle_bridge;
