use super::super::super::columnar::Quantum;
use super::{ArrayRef, MemoryReservation, PayloadRef, Result, RowPayload, StreamOperatorContext};
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
        mut payloads: impl ExactSizeIterator<Item = PayloadRef<'a>>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Self> {
        let capacity = payloads.len();
        let first = payloads.next().expect("nonempty matched range");
        let source = first.owner().expect("detached owned gather input").clone();
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
        transition: PayloadRef<'a>,
        payloads: impl Iterator<Item = PayloadRef<'a>>,
        capacity: usize,
        quantum: &mut Quantum,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Self> {
        let mut rows = Vec::with_capacity(capacity);
        Self::promote_prefix(&prefix, &mut rows, quantum, context).await?;
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
                        sources.push(
                            payload
                                .owner()
                                .expect("detached owned gather input")
                                .clone(),
                        );
                        index
                    });
                    previous = (identity, index);
                }
                rows.push((previous.1, payload.offset()));
            }
        }
        Ok(Self::Multiple { sources, rows })
    }

    async fn promote_prefix(
        prefix: &[u64],
        rows: &mut Vec<(usize, usize)>,
        quantum: &mut Quantum,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
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
        Ok(())
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
        _origin: &MemoryReservation,
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
        result.map_err(|error| crate::CalcFlowError::DataFusion {
            node_id: Some(name.to_owned()),
            message: format!("Join column gather failed: {error}"),
        })
    }
}

#[cfg(test)]
#[path = "side_tests.rs"]
mod tests;
