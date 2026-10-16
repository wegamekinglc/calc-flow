use super::{PayloadRef, canonical_column, concat_output_column, operator_error};
use crate::Result;
use datafusion::arrow::{
    array::{Array, ArrayRef, BinaryViewArray, StringViewArray, UInt64Array, make_array},
    compute::{interleave, take},
    datatypes::DataType,
};
use std::collections::BTreeMap;

mod owned;
pub(super) use owned::{ColumnResult, cancelled, materialize};

#[cfg(test)]
pub(super) mod test_support;

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
    pub(super) fn new(mut payloads: impl Iterator<Item = PayloadRef<'a>>) -> Option<Self> {
        let first = payloads.next()?;
        let columns = first.columns();
        let mut rows = Vec::with_capacity(payloads.size_hint().0 + 1);
        rows.push(u64::try_from(first.offset()).ok()?);
        for payload in payloads.by_ref() {
            if !std::ptr::eq(columns, payload.columns()) {
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
        payloads: impl Iterator<Item = PayloadRef<'a>>,
    ) -> Option<Self> {
        let mut columns = vec![first_columns];
        let mut previous = ((first_columns.as_ptr(), first_columns.len()), 0);
        let mut sources = BTreeMap::from([previous]);
        let mut rows = Vec::with_capacity(first_rows.len() + payloads.size_hint().0);
        for row in first_rows {
            rows.push((0, usize::try_from(row).ok()?));
        }
        for payload in payloads {
            let source = payload.columns();
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

    fn column(&self, index: usize) -> Result<ArrayRef> {
        #[cfg(test)]
        match self {
            Self::Single { .. } => super::note_join_work(|work| work.output_column_takes += 1),
            Self::Multiple { .. } => {
                super::note_join_work(|work| work.output_column_interleaves += 1);
            }
        }
        self.selected_column(index)
    }

    pub(super) fn selected_column(&self, index: usize) -> Result<ArrayRef> {
        let result = match self {
            Self::Single { columns, rows } => take(columns[index].as_ref(), rows, None),
            Self::Multiple { columns, rows } => {
                let sources = columns
                    .iter()
                    .map(|source| source[index].as_ref())
                    .collect::<Vec<_>>();
                interleave(&sources, rows)
            }
        };
        result.map_err(|error| {
            operator_error("Join output", &format!("column gather failed: {error}"))
        })
    }
}

pub(super) fn column(
    gather: Option<&SideGather<'_>>,
    index: usize,
    canonical: &DataType,
) -> Result<ArrayRef> {
    #[cfg(test)]
    let _observation = test_support::observe(test_support::hook(), index);
    let gather =
        gather.ok_or_else(|| operator_error("Join output", "missing nonempty gather selection"))?;
    Ok(canonical_column(
        compact_gather(gather.column(index)?)?,
        canonical,
    ))
}

pub(super) fn flat_owned_type(data_type: &DataType) -> bool {
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

fn compact_gather(column: ArrayRef) -> Result<ArrayRef> {
    if flat_owned_type(column.data_type()) {
        return Ok(column);
    }
    // Bulk compaction detaches encoded children and discards unused dictionary values.
    detach_views(concat_output_column(&[column.as_ref()])?)
}

fn detach_views(column: ArrayRef) -> Result<ArrayRef> {
    if let Some(values) = column.as_any().downcast_ref::<StringViewArray>() {
        return Ok(std::sync::Arc::new(values.gc()));
    }
    if let Some(values) = column.as_any().downcast_ref::<BinaryViewArray>() {
        return Ok(std::sync::Arc::new(values.gc()));
    }
    let data = column.to_data();
    if data.child_data().is_empty() {
        return Ok(column);
    }
    let children = data
        .child_data()
        .iter()
        .map(|child| detach_views(make_array(child.clone())).map(|child| child.to_data()))
        .collect::<Result<Vec<_>>>()?;
    data.into_builder()
        .child_data(children)
        .build()
        .map(make_array)
        .map_err(|error| {
            operator_error(
                "Join output",
                &format!("gather child compaction failed: {error}"),
            )
        })
}
