use super::{canonical_column, columnar::RowPayload, concat_output_column};
use crate::Result;
use datafusion::arrow::{
    array::{ArrayRef, UInt64Array},
    compute::{interleave, take},
    datatypes::DataType,
};
use std::collections::BTreeMap;

pub(super) enum SideGather<'a> {
    Single {
        columns: &'a [ArrayRef],
        rows: UInt64Array,
    },
    Multiple {
        columns: Vec<&'a [ArrayRef]>,
        rows: Vec<(usize, usize)>,
    },
}

impl<'a> SideGather<'a> {
    pub(super) fn new(mut payloads: impl Iterator<Item = &'a RowPayload>) -> Option<Self> {
        let first = payloads.next()?;
        let columns = first.shared_columns()?;
        let mut rows = Vec::with_capacity(payloads.size_hint().0 + 1);
        rows.push(u64::try_from(first.offset()).ok()?);
        for payload in payloads.by_ref() {
            if !std::ptr::eq(columns, payload.shared_columns()?) {
                return Self::multiple(columns, rows, std::iter::once(payload).chain(payloads));
            }
            rows.push(u64::try_from(payload.offset()).ok()?);
        }
        Some(Self::Single {
            columns,
            rows: UInt64Array::from(rows),
        })
    }

    fn multiple(
        first_columns: &'a [ArrayRef],
        first_rows: Vec<u64>,
        payloads: impl Iterator<Item = &'a RowPayload>,
    ) -> Option<Self> {
        let mut columns = vec![first_columns];
        let mut previous = ((first_columns.as_ptr(), first_columns.len()), 0);
        let mut sources = BTreeMap::from([previous]);
        let mut rows = Vec::with_capacity(first_rows.len() + payloads.size_hint().0);
        for row in first_rows {
            rows.push((0, usize::try_from(row).ok()?));
        }
        for payload in payloads {
            let source = payload.shared_columns()?;
            let identity = (source.as_ptr(), source.len());
            if identity != previous.0 {
                #[cfg(test)]
                super::note_join_work(|work| work.output_source_lookups += 1);
                let index = *sources.entry(identity).or_insert_with(|| {
                    let index = columns.len();
                    columns.push(source);
                    index
                });
                previous = (identity, index);
            }
            rows.push((previous.1, payload.offset()));
        }
        Some(Self::Multiple { columns, rows })
    }

    fn column(&self, index: usize) -> Option<ArrayRef> {
        let source = match self {
            Self::Single { columns, .. } => &columns[index],
            Self::Multiple { columns, .. } => &columns[0][index],
        };
        if !independently_compact(source.data_type()) {
            return None;
        }
        match self {
            Self::Single { rows, .. } => {
                let taken = take(source.as_ref(), rows, None).ok()?;
                #[cfg(test)]
                super::note_join_work(|work| work.output_column_takes += 1);
                Some(taken)
            }
            Self::Multiple { columns, rows } => {
                let sources = columns
                    .iter()
                    .map(|source| source[index].as_ref())
                    .collect::<Vec<_>>();
                let interleaved = interleave(&sources, rows).ok()?;
                #[cfg(test)]
                super::note_join_work(|work| work.output_column_interleaves += 1);
                Some(interleaved)
            }
        }
    }
}

fn independently_compact(data_type: &DataType) -> bool {
    data_type.is_primitive()
        || matches!(
            data_type,
            DataType::Null
                | DataType::Boolean
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::FixedSizeBinary(_)
        )
}

pub(super) fn column<'a>(
    gather: Option<&SideGather<'_>>,
    index: usize,
    canonical: &DataType,
    payloads: impl Iterator<Item = &'a RowPayload>,
) -> Result<ArrayRef> {
    if let Some(taken) = gather.and_then(|gather| gather.column(index)) {
        return Ok(canonical_column(taken, canonical));
    }
    let slices = payloads
        .map(|payload| payload.column_view(index))
        .collect::<Vec<_>>();
    let references = slices.iter().map(AsRef::as_ref).collect::<Vec<_>>();
    Ok(canonical_column(
        concat_output_column(&references)?,
        canonical,
    ))
}
