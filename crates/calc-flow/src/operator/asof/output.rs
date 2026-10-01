//! Materialize an already matched ASOF prefix directly from Arrow batches.

use super::state::{BatchKey, PayloadView};
use crate::{Batch, BatchMetadata, DataFusionConfig, Result};
use ahash::RandomState;
use arrow_data::transform::MutableArrayData;
use datafusion::execution::memory_pool::MemoryReservation;
use datafusion::{
    arrow::{
        array::{Array, ArrayRef, make_array, new_null_array},
        compute::interleave,
        datatypes::SchemaRef,
        record_batch::RecordBatch,
    },
    execution::memory_pool::{GreedyMemoryPool, MemoryPool},
};
use std::{borrow::Cow, collections::HashMap, sync::Arc};

pub(super) struct OutputRuntime {
    pub pool: Arc<dyn MemoryPool>,
    config: DataFusionConfig,
    #[cfg(test)]
    worker_gate: Option<(std::sync::mpsc::Sender<()>, Arc<std::sync::Barrier>)>,
}

impl OutputRuntime {
    pub fn new(limit: usize) -> Self {
        Self {
            pool: Arc::new(GreedyMemoryPool::new(limit)),
            config: DataFusionConfig::default(),
            #[cfg(test)]
            worker_gate: None,
        }
    }

    pub fn configure(&mut self, config: DataFusionConfig) {
        self.config = config;
    }

    pub async fn materialize(
        &mut self,
        rows: &[(PayloadView<'_>, Option<PayloadView<'_>>)],
        schema: &SchemaRef,
        workspace: MemoryReservation,
        check_cancelled: impl Fn() -> Result<()> + Send + Sync,
    ) -> Result<(Batch, MemoryReservation)> {
        self.config.validate()?;
        tokio::task::yield_now().await;
        let owned = OutputRows::capture(rows, &check_cancelled).await?;
        let schema = schema.clone();
        #[cfg(test)]
        let worker_gate = self.worker_gate.clone();
        // Plan construction and Arrow copies stay off the executor. The
        // reservation moves with the work, even if this future is dropped.
        let (result, workspace) = tokio::task::spawn_blocking(move || {
            #[cfg(test)]
            if let Some((started, gate)) = worker_gate {
                let _ = started.send(());
                gate.wait();
            }
            let left = GatherPlan::new(&owned.left, false);
            let right = GatherPlan::new(&owned.right, true);
            let result = materialize_plans(&left, &right, owned.len, &schema)?;
            Ok::<_, crate::CalcFlowError>((result, workspace))
        })
        .await
        .map_err(|error| crate::CalcFlowError::Internal {
            message: format!("ASOF output materialization task failed: {error}"),
        })??;
        Ok((result, workspace))
    }
}

#[derive(Clone, Copy)]
struct Span {
    source: usize,
    start: usize,
    end: usize,
}

/// One Arrow owner per distinct source, plus primitive row positions. A
/// blocking worker never owns the operator's per-row payload handles.
struct OutputSide {
    batches: Vec<Arc<RecordBatch>>,
    positions: Vec<(usize, usize)>,
}

struct OutputRows {
    left: OutputSide,
    right: OutputSide,
    len: usize,
}

type SourceMap = HashMap<BatchKey, usize, RandomState>;

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
                output.left.push(Some(left), &mut left_sources);
                output.right.push(right, &mut right_sources);
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
            output.left.push(Some(left), &mut left_sources);
            output.right.push(right, &mut right_sources);
        }
        output
    }
}

impl OutputSide {
    fn empty(len: usize) -> Self {
        Self {
            batches: Vec::new(),
            positions: Vec::with_capacity(len),
        }
    }

    fn push(&mut self, selected: Option<PayloadView<'_>>, by_key: &mut SourceMap) {
        let Some(row) = selected else {
            self.positions.push((usize::MAX, 0));
            return;
        };
        let source = *by_key.entry(row.batch.key).or_insert_with(|| {
            let source = self.batches.len();
            self.batches.push(Arc::clone(&row.batch.record));
            source
        });
        self.positions.push((source, row.row));
    }
}

struct GatherPlan<'a> {
    batches: &'a [Arc<RecordBatch>],
    spans: Vec<Span>,
    positions: Option<Cow<'a, [(usize, usize)]>>,
}

impl<'a> GatherPlan<'a> {
    fn new(rows: &'a OutputSide, right: bool) -> Self {
        let mut spans = Vec::new();
        if !right {
            for &(source, row) in &rows.positions {
                if let Some(Span {
                    source: previous,
                    end,
                    ..
                }) = spans.last_mut()
                    && *previous == source
                    && *end == row
                {
                    *end += 1;
                } else {
                    spans.push(Span {
                        source,
                        start: row,
                        end: row + 1,
                    });
                }
            }
        }
        Self {
            batches: &rows.batches,
            spans,
            positions: right.then(|| right_positions(rows)),
        }
    }

    fn column(
        &self,
        index: usize,
        data_type: &datafusion::arrow::datatypes::DataType,
        len: usize,
    ) -> Result<ArrayRef> {
        if self.batches.is_empty() {
            return Ok(new_null_array(data_type, len));
        }
        if let Some(positions) = &self.positions {
            let columns = self
                .batches
                .iter()
                .map(|batch| batch.column(index).as_ref())
                .collect::<Vec<&dyn Array>>();
            let null_column = positions
                .iter()
                .any(|(source, _)| *source == self.batches.len())
                .then(|| new_null_array(data_type, 1));
            let mut columns = columns;
            if let Some(null_column) = null_column.as_ref() {
                columns.push(null_column.as_ref());
            }
            return interleave(&columns, positions).map_err(|error| super::arrow_error(&error));
        }
        let data = self
            .batches
            .iter()
            .map(|batch| batch.column(index).to_data())
            .collect::<Vec<_>>();
        let nullable = data.iter().any(|data| data.nulls().is_some());
        let mut mutable = MutableArrayData::new(data.iter().collect(), nullable, len);
        for span in &self.spans {
            mutable.extend(span.source, span.start, span.end);
        }
        Ok(make_array(mutable.freeze()))
    }
}

fn right_positions(rows: &OutputSide) -> Cow<'_, [(usize, usize)]> {
    if rows
        .positions
        .iter()
        .any(|(source, _)| *source == usize::MAX)
    {
        Cow::Owned(
            rows.positions
                .iter()
                .map(|&(source, row)| {
                    (
                        if source == usize::MAX {
                            rows.batches.len()
                        } else {
                            source
                        },
                        row,
                    )
                })
                .collect(),
        )
    } else {
        Cow::Borrowed(rows.positions.as_slice())
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
    materialize_plans(&left, &right, rows.len(), schema)
}

fn materialize_plans(
    left: &GatherPlan<'_>,
    right: &GatherPlan<'_>,
    len: usize,
    schema: &SchemaRef,
) -> Result<Batch> {
    let left_fields = left.batches.first().map_or(0, |batch| batch.num_columns());
    let mut columns = Vec::with_capacity(schema.fields().len());
    for (index, field) in schema.fields().iter().enumerate() {
        let column = if index < left_fields {
            left.column(index, field.data_type(), len)?
        } else {
            right.column(index - left_fields, field.data_type(), len)?
        };
        columns.push(column);
    }
    let record = RecordBatch::try_new(schema.clone(), columns)
        .map_err(|error| super::arrow_error(&error))?;
    Batch::table(vec![record], BatchMetadata::default())
}

#[cfg(test)]
mod tests;
