//! Materialize an already matched ASOF prefix directly from Arrow batches.

use super::state::{BatchKey, PayloadBatch, RowPayload};
use crate::{Batch, BatchMetadata, DataFusionConfig, Result};
use arrow_data::transform::MutableArrayData;
use datafusion::execution::memory_pool::MemoryReservation;
use datafusion::{
    arrow::{
        array::{ArrayRef, make_array, new_null_array},
        datatypes::SchemaRef,
        record_batch::RecordBatch,
    },
    execution::memory_pool::{GreedyMemoryPool, MemoryPool},
};
use std::{collections::BTreeMap, sync::Arc};

pub(super) struct OutputRuntime {
    initialized: bool,
    pub pool: Arc<dyn MemoryPool>,
    config: DataFusionConfig,
    #[cfg(test)]
    worker_gate: Option<(std::sync::mpsc::Sender<()>, Arc<std::sync::Barrier>)>,
}

impl OutputRuntime {
    pub fn new(limit: usize) -> Self {
        Self {
            initialized: false,
            pool: Arc::new(GreedyMemoryPool::new(limit)),
            config: DataFusionConfig::default(),
            #[cfg(test)]
            worker_gate: None,
        }
    }

    pub const fn initialized(&self) -> bool {
        self.initialized
    }

    pub fn reset(&mut self) {
        self.initialized = false;
    }

    pub fn configure(&mut self, config: DataFusionConfig) {
        self.config = config;
    }

    pub async fn materialize(
        &mut self,
        rows: &[(&RowPayload, Option<&RowPayload>)],
        schema: &SchemaRef,
        workspace: MemoryReservation,
    ) -> Result<(Batch, MemoryReservation)> {
        self.config.validate()?;
        tokio::task::yield_now().await;
        let owned = rows
            .iter()
            .map(|(left, right)| ((*left).clone(), right.cloned()))
            .collect::<Vec<_>>();
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
            let refs = || owned.iter().map(|(left, right)| (left, right.as_ref()));
            let left = GatherPlan::new(refs(), false);
            let right = GatherPlan::new(refs(), true);
            let result = materialize_plans(&left, &right, owned.len(), &schema)?;
            Ok::<_, crate::CalcFlowError>((result, workspace))
        })
        .await
        .map_err(|error| crate::CalcFlowError::Internal {
            message: format!("ASOF output materialization task failed: {error}"),
        })??;
        self.initialized = true;
        Ok((result, workspace))
    }
}

#[derive(Clone, Copy)]
enum Span {
    Values {
        source: usize,
        start: usize,
        end: usize,
    },
    Nulls(usize),
}

struct GatherPlan {
    batches: Vec<Arc<PayloadBatch>>,
    spans: Vec<Span>,
}

impl GatherPlan {
    fn new<'a>(
        rows: impl IntoIterator<Item = (&'a RowPayload, Option<&'a RowPayload>)>,
        right: bool,
    ) -> Self {
        let mut batches = Vec::new();
        let mut by_key = BTreeMap::<BatchKey, usize>::new();
        let mut spans = Vec::new();
        for (left, candidate) in rows {
            let selected = if right { candidate } else { Some(left) };
            let Some(row) = selected else {
                if let Some(Span::Nulls(len)) = spans.last_mut() {
                    *len += 1;
                } else {
                    spans.push(Span::Nulls(1));
                }
                continue;
            };
            let source = match by_key.entry(row.batch.key) {
                std::collections::btree_map::Entry::Occupied(entry) => *entry.get(),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    let source = batches.len();
                    batches.push(row.batch.clone());
                    entry.insert(source);
                    source
                }
            };
            if let Some(Span::Values {
                source: previous,
                end,
                ..
            }) = spans.last_mut()
                && *previous == source
                && *end == row.row
            {
                *end += 1;
            } else {
                spans.push(Span::Values {
                    source,
                    start: row.row,
                    end: row.row + 1,
                });
            }
        }
        Self { batches, spans }
    }

    fn column(
        &self,
        index: usize,
        data_type: &datafusion::arrow::datatypes::DataType,
        len: usize,
    ) -> ArrayRef {
        if self.batches.is_empty() {
            return new_null_array(data_type, len);
        }
        let data = self
            .batches
            .iter()
            .map(|batch| batch.record.column(index).to_data())
            .collect::<Vec<_>>();
        let nullable = self.spans.iter().any(|span| matches!(span, Span::Nulls(_)))
            || data.iter().any(|data| data.nulls().is_some());
        let mut mutable = MutableArrayData::new(data.iter().collect(), nullable, len);
        for span in &self.spans {
            match *span {
                Span::Values { source, start, end } => mutable.extend(source, start, end),
                Span::Nulls(count) => mutable.extend_nulls(count),
            }
        }
        make_array(mutable.freeze())
    }
}

#[cfg(test)]
fn materialize_rows(
    rows: &[(&RowPayload, Option<&RowPayload>)],
    schema: &SchemaRef,
) -> Result<Batch> {
    let left = GatherPlan::new(rows.iter().copied(), false);
    let right = GatherPlan::new(rows.iter().copied(), true);
    materialize_plans(&left, &right, rows.len(), schema)
}

fn materialize_plans(
    left: &GatherPlan,
    right: &GatherPlan,
    len: usize,
    schema: &SchemaRef,
) -> Result<Batch> {
    let left_fields = left
        .batches
        .first()
        .map_or(0, |batch| batch.record.num_columns());
    let mut columns = Vec::with_capacity(schema.fields().len());
    for (index, field) in schema.fields().iter().enumerate() {
        let column = if index < left_fields {
            left.column(index, field.data_type(), len)
        } else {
            right.column(index - left_fields, field.data_type(), len)
        };
        columns.push(column);
    }
    let record = RecordBatch::try_new(schema.clone(), columns)
        .map_err(|error| super::arrow_error(&error))?;
    Batch::table(vec![record], BatchMetadata::default())
}

#[cfg(test)]
mod tests;
