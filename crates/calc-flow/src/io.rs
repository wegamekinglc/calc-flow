use async_trait::async_trait;
use datafusion::arrow::{compute::concat_batches, record_batch::RecordBatch};
use serde_json::Value;

use crate::{Batch, BatchKind, CalcFlowError, Result, RunContext, batch::checked_accumulate};

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
    /// A single source item that exceeds either limit fails with
    /// [`CalcFlowError::InvalidArgument`] instead of being emitted alone:
    /// an oversize item can never fit through a bounded stream edge, so
    /// there is no "one oversize message" exception (spec S10.3). The
    /// failure is latched and the source must be reopened before further
    /// reads.
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

        if let Some(error) = self.single_item_oversize_error() {
            self.faulted = true;
            return Err(error);
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
            let (rows, bytes) = match (
                checked_accumulate(accumulated.rows, candidate_parts.2, "source.batch"),
                checked_accumulate(accumulated.bytes, candidate_parts.3, "source.batch"),
            ) {
                (Ok(rows), Ok(bytes)) => (rows, bytes),
                (Err(error), _) | (_, Err(error)) => {
                    self.faulted = true;
                    return Err(error);
                }
            };
            if rows > self.max_rows || bytes > self.max_bytes {
                self.pending = Some(candidate);
                break;
            }
            accumulated.batches.extend(candidate_parts.1);
            accumulated.rows = rows;
            accumulated.bytes = bytes;
            accumulated.latest = candidate;
        }

        self.take_accumulated()
    }
}

impl<S> BatchingSource<S> {
    /// A single source item larger than a configured limit can never fit
    /// through a bounded stream edge, so it is a typed error rather than a
    /// lone emission (spec S10.3: no "one oversize message" exception).
    fn single_item_oversize_error(&self) -> Option<CalcFlowError> {
        let group = self.accumulated.as_ref()?;
        let mut exceeded = Vec::new();
        if group.rows > self.max_rows {
            exceeded.push(format!(
                "{} rows exceed the {} row limit",
                group.rows, self.max_rows
            ));
        }
        if group.bytes > self.max_bytes {
            exceeded.push(format!(
                "{} bytes exceed the {} byte limit",
                group.bytes, self.max_bytes
            ));
        }
        if exceeded.is_empty() {
            return None;
        }
        Some(CalcFlowError::InvalidArgument {
            field: "source.batch".into(),
            message: format!(
                "single source item exceeds the batching limits: {}",
                exceeded.join(", ")
            ),
        })
    }

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
    let bytes = table.estimated_bytes().map_err(source_batch_error)?;
    Ok((
        table.schema().clone(),
        table.batches().to_vec(),
        item.batch.num_rows(),
        bytes,
    ))
}

/// Re-attributes a `TableBatch::estimated_bytes` failure, reported against
/// the bare `"batch"` field, to the source item's batch. This keeps the
/// batching boundary on the pre-M1.3 diagnostic contract (`"source.batch"`)
/// while the message carries the measurement detail.
fn source_batch_error(error: CalcFlowError) -> CalcFlowError {
    match error {
        CalcFlowError::InvalidArgument { message, .. } => CalcFlowError::InvalidArgument {
            field: "source.batch".into(),
            message,
        },
        error => error,
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    // A `TableBatch` whose estimation fails cannot be built in safe code:
    // Arrow validates every measurement failure mode at construction and a
    // `usize` overflow needs impossible allocations, so the failing result
    // `TableBatch::estimated_bytes` would return is constructed directly.
    #[test]
    fn measurement_failure_is_attributed_to_the_source_batch_field() {
        let failure = CalcFlowError::InvalidArgument {
            field: "batch".into(),
            message: "Arrow slice memory could not be measured: boom".into(),
        };
        assert!(matches!(
            source_batch_error(failure),
            CalcFlowError::InvalidArgument { ref field, ref message }
                if field == "source.batch"
                    && message == "Arrow slice memory could not be measured: boom"
        ));
    }

    #[test]
    fn measurement_overflow_is_attributed_to_the_source_batch_field() {
        let failure = CalcFlowError::InvalidArgument {
            field: "batch".into(),
            message: "size sum overflowed usize".into(),
        };
        assert!(matches!(
            source_batch_error(failure),
            CalcFlowError::InvalidArgument { ref field, ref message }
                if field == "source.batch" && message == "size sum overflowed usize"
        ));
    }

    #[test]
    fn source_batch_error_passes_non_argument_errors_through() {
        let error = CalcFlowError::Cancelled {
            run_id: "run-1".into(),
        };
        assert!(matches!(
            source_batch_error(error),
            CalcFlowError::Cancelled { ref run_id } if run_id == "run-1"
        ));
    }
}
