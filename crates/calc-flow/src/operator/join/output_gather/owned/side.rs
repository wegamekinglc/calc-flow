use super::super::super::{columnar::Quantum, concat_output_column, native_lookup::scratch_error};
use super::{ArrayRef, MemoryReservation, Result, RowPayload, StreamOperatorContext};
use datafusion::arrow::{
    array::UInt64Array,
    compute::{interleave, take},
};
use std::collections::BTreeMap;

pub(super) enum OwnedSide {
    Single {
        source: RowPayload,
        rows: UInt64Array,
    },
    Multiple {
        sources: Vec<RowPayload>,
        rows: Vec<(usize, usize)>,
    },
}

impl OwnedSide {
    pub(super) async fn capture<'a>(
        mut payloads: impl ExactSizeIterator<Item = &'a RowPayload>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Self> {
        let capacity = payloads.len();
        let first = payloads.next().expect("nonempty matched range");
        let source = first.clone();
        let identity = first.columns().as_ptr() as usize;
        let mut payloads = std::iter::once(first).chain(payloads);
        let mut rows = Vec::with_capacity(capacity);
        let mut quantum = Quantum::default();
        while rows.len() < capacity {
            let count = (capacity - rows.len()).min(64);
            let bytes = 32 * count;
            quantum.step(context, count, bytes).await?;
            #[cfg(test)]
            tests::note_charge(tests::Phase::Capture, count, bytes);
            for _ in 0..count {
                let payload = payloads.next().expect("exact matched range");
                if payload.columns().as_ptr() as usize != identity {
                    return Self::capture_multiple(
                        source,
                        rows,
                        payload,
                        payloads,
                        capacity,
                        &mut quantum,
                        context,
                    )
                    .await;
                }
                rows.push(payload.offset() as u64);
            }
        }
        Ok(Self::Single {
            source,
            rows: UInt64Array::from(rows),
        })
    }

    async fn capture_multiple<'a>(
        source: RowPayload,
        prefix: Vec<u64>,
        transition: &'a RowPayload,
        payloads: impl Iterator<Item = &'a RowPayload>,
        capacity: usize,
        quantum: &mut Quantum,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Self> {
        let mut rows = Vec::with_capacity(capacity);
        for chunk in prefix.chunks(64) {
            let bytes = 32 * chunk.len();
            quantum.step(context, chunk.len(), bytes).await?;
            #[cfg(test)]
            tests::note_charge(tests::Phase::Promotion, chunk.len(), bytes);
            rows.extend(
                chunk
                    .iter()
                    .map(|&row| (0, usize::try_from(row).expect("source offset fits usize"))),
            );
            #[cfg(test)]
            tests::note_promoted(chunk.len());
        }
        drop(prefix);
        context.check_cancelled()?;
        let identity = source.columns().as_ptr() as usize;
        let mut sources = vec![source];
        let mut previous = (identity, 0);
        let mut parents = BTreeMap::from([previous]);
        let mut payloads = std::iter::once(transition).chain(payloads);
        while rows.len() < capacity {
            let count = (capacity - rows.len()).min(64);
            let bytes = 32 * count;
            quantum.step(context, count, bytes).await?;
            #[cfg(test)]
            tests::note_charge(tests::Phase::Capture, count, bytes);
            for _ in 0..count {
                let payload = payloads.next().expect("exact matched range");
                let identity = payload.columns().as_ptr() as usize;
                if identity != previous.0 {
                    let index = *parents.entry(identity).or_insert_with(|| {
                        let index = sources.len();
                        sources.push(payload.clone());
                        index
                    });
                    previous = (identity, index);
                }
                rows.push((previous.1, payload.offset()));
            }
        }
        Ok(Self::Multiple { sources, rows })
    }

    pub(super) fn width(&self) -> usize {
        match self {
            Self::Single { source, .. } => source.num_columns(),
            Self::Multiple { sources, .. } => sources[0].num_columns(),
        }
    }

    pub(super) fn column(
        &self,
        index: usize,
        origin: &MemoryReservation,
        name: &str,
    ) -> Result<ArrayRef> {
        let result = match self {
            Self::Single { source, rows } => take(source.column(index).as_ref(), rows, None),
            Self::Multiple { sources, rows } => {
                let columns = sources
                    .iter()
                    .map(|source| source.column(index).as_ref())
                    .collect::<Vec<_>>();
                interleave(&columns, rows)
            }
        };
        result
            .or_else(|_| self.concat(index, origin, name))
            .map_err(|error| crate::CalcFlowError::DataFusion {
                node_id: Some(name.to_owned()),
                message: format!("Join column gather failed: {error}"),
            })
    }

    fn concat(
        &self,
        index: usize,
        origin: &MemoryReservation,
        name: &str,
    ) -> std::result::Result<ArrayRef, datafusion::arrow::error::ArrowError> {
        let rows = match self {
            Self::Single { rows, .. } => rows.len(),
            Self::Multiple { rows, .. } => rows.len(),
        };
        let bytes = rows
            .checked_mul(512)
            .and_then(|bytes| bytes.checked_add(4096))
            .ok_or_else(|| {
                datafusion::arrow::error::ArrowError::ComputeError(scratch_error(name).to_string())
            })?;
        let scratch = origin.new_empty();
        scratch.try_grow(bytes).map_err(|error| {
            datafusion::arrow::error::ArrowError::ComputeError(error.to_string())
        })?;
        let slices = match self {
            Self::Single { source, rows } => rows
                .values()
                .iter()
                .map(|&row| {
                    source
                        .column(index)
                        .slice(usize::try_from(row).expect("source offset fits usize"), 1)
                })
                .collect::<Vec<_>>(),
            Self::Multiple { sources, rows } => rows
                .iter()
                .map(|&(source, row)| sources[source].column(index).slice(row, 1))
                .collect::<Vec<_>>(),
        };
        let references = slices.iter().map(AsRef::as_ref).collect::<Vec<_>>();
        concat_output_column(&references)
            .map_err(|error| datafusion::arrow::error::ArrowError::ComputeError(error.to_string()))
    }
}

#[cfg(test)]
#[path = "side_tests.rs"]
mod tests;
