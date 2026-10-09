use super::{canonical_column, columnar::RowPayload, concat_output_column};
use crate::Result;
use datafusion::arrow::{
    array::{ArrayRef, UInt64Array},
    compute::take,
    datatypes::DataType,
};

pub(super) struct SideGather<'a> {
    columns: &'a [ArrayRef],
    rows: UInt64Array,
}

impl<'a> SideGather<'a> {
    pub(super) fn new(mut payloads: impl Iterator<Item = &'a RowPayload>) -> Option<Self> {
        let first = payloads.next()?;
        let columns = first.shared_columns()?;
        let mut rows = Vec::with_capacity(payloads.size_hint().0 + 1);
        rows.push(u64::try_from(first.offset()).ok()?);
        for payload in payloads {
            if !std::ptr::eq(columns, payload.shared_columns()?) {
                return None;
            }
            rows.push(u64::try_from(payload.offset()).ok()?);
        }
        Some(Self {
            columns,
            rows: UInt64Array::from(rows),
        })
    }

    fn column(&self, index: usize) -> Option<ArrayRef> {
        let source = &self.columns[index];
        if !independently_compact(source.data_type()) {
            return None;
        }
        let taken = take(source.as_ref(), &self.rows, None).ok()?;
        #[cfg(test)]
        super::note_join_work(|work| work.output_column_takes += 1);
        Some(taken)
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
