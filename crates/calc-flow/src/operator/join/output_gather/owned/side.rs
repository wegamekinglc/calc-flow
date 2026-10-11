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
        mut payloads: impl Iterator<Item = &'a RowPayload>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Self> {
        let first = payloads.next().expect("nonempty matched range");
        let mut sources = vec![first.clone()];
        let identity = first.columns().as_ptr() as usize;
        let mut previous = (identity, 0);
        let mut parents = BTreeMap::from([previous]);
        let mut rows = Vec::with_capacity(payloads.size_hint().0 + 1);
        rows.push((0, first.offset()));
        let mut quantum = Quantum::default();
        for payload in payloads {
            quantum.step(context, 1, 32).await?;
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
        if sources.len() == 1 {
            let rows = UInt64Array::from(
                rows.into_iter()
                    .map(|(_, row)| row as u64)
                    .collect::<Vec<_>>(),
            );
            Ok(Self::Single {
                source: sources.pop().unwrap(),
                rows,
            })
        } else {
            Ok(Self::Multiple { sources, rows })
        }
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
