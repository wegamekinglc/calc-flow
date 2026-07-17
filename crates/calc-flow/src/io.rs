use async_trait::async_trait;
use datafusion::arrow::{compute::concat_batches, record_batch::RecordBatch};
use serde_json::Value;

use crate::{Batch, BatchKind, CalcFlowError, Result, RunContext};

/// One replayable source item with the cursor committed after its delivery.
#[derive(Clone, Debug)]
pub struct SourceItem {
    pub batch: Batch,
    pub cursor: Option<Value>,
    pub sequence: u64,
}

/// Asynchronous replayable input contract.
#[async_trait]
pub trait Source: Send {
    /// Positions the source immediately after the supplied durable cursor.
    async fn open(&mut self, cursor: Option<Value>) -> Result<()>;
    /// Returns the next formed item, or `None` at end of input.
    async fn next(&mut self) -> Result<Option<SourceItem>>;
}

#[async_trait]
impl<T: Source + ?Sized> Source for Box<T> {
    async fn open(&mut self, cursor: Option<Value>) -> Result<()> {
        (**self).open(cursor).await
    }

    async fn next(&mut self) -> Result<Option<SourceItem>> {
        (**self).next().await
    }
}

/// Asynchronous destination for one named graph output.
#[async_trait]
pub trait Sink: Send {
    /// Delivers a batch under the exact context that produced it.
    async fn write(&mut self, batch: &Batch, context: &RunContext) -> Result<()>;
}

/// Coalesces adjacent table source items without mutating their Arrow buffers.
pub struct BatchingSource<S> {
    source: S,
    max_rows: usize,
    max_bytes: usize,
    pending: Option<SourceItem>,
    accumulated: Option<TableAccumulator>,
    faulted: bool,
}

struct TableAccumulator {
    schema: datafusion::arrow::datatypes::SchemaRef,
    batches: Vec<RecordBatch>,
    rows: usize,
    bytes: usize,
    latest: SourceItem,
}

impl<S> BatchingSource<S> {
    /// Wraps a source with positive row and Arrow-memory limits.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when either limit is zero.
    pub fn new(source: S, max_rows: usize, max_bytes: usize) -> Result<Self> {
        if max_rows == 0 {
            return Err(CalcFlowError::InvalidArgument {
                field: "max_rows".into(),
                message: "must be greater than zero".into(),
            });
        }
        if max_bytes == 0 {
            return Err(CalcFlowError::InvalidArgument {
                field: "max_bytes".into(),
                message: "must be greater than zero".into(),
            });
        }
        Ok(Self {
            source,
            max_rows,
            max_bytes,
            pending: None,
            accumulated: None,
            faulted: false,
        })
    }

    /// Returns the wrapped source.
    pub fn into_inner(self) -> S {
        self.source
    }
}

#[async_trait]
impl<S: Source> Source for BatchingSource<S> {
    async fn open(&mut self, cursor: Option<Value>) -> Result<()> {
        self.source.open(cursor).await?;
        self.pending = None;
        self.accumulated = None;
        self.faulted = false;
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceItem>> {
        if self.faulted {
            return Err(CalcFlowError::InvalidArgument {
                field: "source".into(),
                message: "batching source must be reopened after its previous invalid item".into(),
            });
        }
        if self.accumulated.is_none() {
            let Some(first) = take_next(&mut self.source, &mut self.pending).await? else {
                return Ok(None);
            };
            let (schema, batches, rows, bytes) = match table_parts(&first) {
                Ok(parts) => parts,
                Err(error) => {
                    self.faulted = true;
                    return Err(error);
                }
            };
            self.accumulated = Some(TableAccumulator {
                schema,
                batches,
                rows,
                bytes,
                latest: first,
            });
        }

        let first_exceeds_limit = self
            .accumulated
            .as_ref()
            .is_some_and(|group| group.rows > self.max_rows || group.bytes > self.max_bytes);
        if first_exceeds_limit {
            return self.take_accumulated();
        }

        loop {
            let candidate = match self.source.next().await {
                Ok(Some(candidate)) => candidate,
                Ok(None) => break,
                Err(error) => return Err(error),
            };
            let candidate_parts = match table_parts(&candidate) {
                Ok(parts) => parts,
                Err(error) => {
                    self.faulted = true;
                    return Err(error);
                }
            };
            let accumulated = self
                .accumulated
                .as_mut()
                .expect("validated accumulator remains present until emission");
            if candidate_parts.0 != accumulated.schema {
                self.faulted = true;
                return Err(CalcFlowError::InvalidArgument {
                    field: "source.batch.schema".into(),
                    message: "adjacent table batches must have identical schemas".into(),
                });
            }
            let exceeds = accumulated.rows.saturating_add(candidate_parts.2) > self.max_rows
                || accumulated.bytes.saturating_add(candidate_parts.3) > self.max_bytes;
            if exceeds {
                self.pending = Some(candidate);
                break;
            }
            accumulated.batches.extend(candidate_parts.1);
            accumulated.rows = accumulated.rows.saturating_add(candidate_parts.2);
            accumulated.bytes = accumulated.bytes.saturating_add(candidate_parts.3);
            accumulated.latest = candidate;
        }

        self.take_accumulated()
    }
}

impl<S> BatchingSource<S> {
    fn take_accumulated(&mut self) -> Result<Option<SourceItem>> {
        let accumulated = self
            .accumulated
            .take()
            .expect("emission requires a validated accumulator");
        match coalesced_item(
            &accumulated.schema,
            &accumulated.batches,
            &accumulated.latest,
        ) {
            Ok(item) => Ok(Some(item)),
            Err(error) => {
                self.faulted = true;
                Err(error)
            }
        }
    }
}

async fn take_next<S: Source>(
    source: &mut S,
    pending: &mut Option<SourceItem>,
) -> Result<Option<SourceItem>> {
    match pending.take() {
        Some(item) => Ok(Some(item)),
        None => source.next().await,
    }
}

fn table_parts(
    item: &SourceItem,
) -> Result<(
    datafusion::arrow::datatypes::SchemaRef,
    Vec<RecordBatch>,
    usize,
    usize,
)> {
    if item.batch.kind() != BatchKind::Table {
        return Err(CalcFlowError::InvalidArgument {
            field: "source.batch".into(),
            message: "batching sources accept only table batches".into(),
        });
    }
    let table = item.batch.table_payload()?;
    let batches = table.batches().to_vec();
    let bytes = batches.iter().try_fold(0_usize, |batch_total, batch| {
        batch
            .columns()
            .iter()
            .try_fold(batch_total, |column_total, column| {
                let bytes = column.to_data().get_slice_memory_size().map_err(|error| {
                    CalcFlowError::InvalidArgument {
                        field: "source.batch".into(),
                        message: format!("Arrow slice memory could not be measured: {error}"),
                    }
                })?;
                column_total
                    .checked_add(bytes)
                    .ok_or_else(|| CalcFlowError::InvalidArgument {
                        field: "source.batch".into(),
                        message: "Arrow slice memory size overflowed usize".into(),
                    })
            })
    })?;
    Ok((
        table.schema().clone(),
        batches,
        item.batch.num_rows(),
        bytes,
    ))
}

fn coalesced_item(
    schema: &datafusion::arrow::datatypes::SchemaRef,
    batches: &[RecordBatch],
    latest: &SourceItem,
) -> Result<SourceItem> {
    let record =
        concat_batches(schema, batches).map_err(|error| CalcFlowError::InvalidArgument {
            field: "source.batch".into(),
            message: format!("Arrow batches could not be concatenated: {error}"),
        })?;
    let batch = Batch::table(vec![record], latest.batch.metadata().clone())?;
    Ok(SourceItem {
        batch,
        cursor: latest.cursor.clone(),
        sequence: latest.sequence,
    })
}
