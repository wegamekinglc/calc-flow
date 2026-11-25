use super::{ColumnarState, RowLocator, Selection, invalid};
use crate::Result;
use crate::operator::asof::columnar::run::{
    logical_array_indices_charge, logical_full_output_charge, logical_null_charge,
};
use datafusion::arrow::buffer::BooleanBuffer;
use std::collections::BTreeMap;

impl ColumnarState {
    pub(in crate::operator::asof) fn output_charge(
        &self,
        selection: &Selection,
        projection: Option<&[(usize, usize)]>,
    ) -> Result<usize> {
        if projection.is_some_and(|fields| fields.iter().any(|&(side, _)| side > 1)) {
            return Err(invalid("ASOF projection side is invalid"));
        }
        let charge = |side| match projection {
            Some(fields) => self.side_output_charge(
                selection,
                side,
                &fields
                    .iter()
                    .filter_map(move |&(source, column)| (source == side).then_some(column)),
            ),
            None => self.side_output_charge(selection, side, &(0..self.column_count(side))),
        };
        add_charge(charge(0)?, charge(1)?)
    }

    fn side_output_charge(
        &self,
        selection: &Selection,
        side: usize,
        columns: &(impl Iterator<Item = usize> + Clone),
    ) -> Result<usize> {
        if columns.clone().next().is_none() {
            return Ok(0);
        }
        let (locators, validity) = if side == 0 {
            (&selection.left, None)
        } else {
            (&selection.right, Some(&selection.right_validity))
        };
        let unmatched = validity.map_or(0, |bits| locators.len() - bits.count_set_bits());
        let mut total = self.column_overhead(side, columns.clone(), locators.len(), unmatched)?;
        if unmatched == 0
            && let Some(records) = self.complete_runs(side, locators)
        {
            return records.iter().try_fold(total, |charge, record| {
                columns
                    .clone()
                    .map(|column| logical_full_output_charge(record.column(column)))
                    .try_fold(charge, add_charge)
            });
        }
        for (run, rows) in selected_rows(locators, validity) {
            let record = self
                .run(side, run)
                .payload_record()
                .expect("selected live ASOF payload");
            total = columns
                .clone()
                .map(|column| logical_array_indices_charge(record.column(column), &rows))
                .try_fold(total, add_charge)?;
        }
        Ok(total)
    }

    fn column_overhead(
        &self,
        side: usize,
        mut columns: impl Iterator<Item = usize>,
        rows: usize,
        unmatched: usize,
    ) -> Result<usize> {
        columns.try_fold(0, |total, column| {
            let field = self.schemas[side]
                .fields()
                .get(column)
                .ok_or_else(|| invalid("ASOF projection column is out of bounds"))?;
            let total = add_charge(total, 256)?;
            if unmatched == 0 {
                return Ok(total);
            }
            let total = total
                .checked_add(rows.div_ceil(8))
                .ok_or_else(|| invalid("ASOF output validity charge overflowed"))?;
            total
                .checked_add(logical_null_charge(field.data_type(), unmatched))
                .ok_or_else(|| invalid("ASOF output null-data charge overflowed"))
        })
    }
}

fn selected_rows(
    locators: &[RowLocator],
    validity: Option<&BooleanBuffer>,
) -> BTreeMap<u64, Vec<usize>> {
    let mut rows = BTreeMap::<_, Vec<_>>::new();
    for (index, locator) in locators.iter().enumerate() {
        if validity.is_none_or(|bits| bits.value(index)) {
            rows.entry(locator.run)
                .or_default()
                .push(locator.row as usize);
        }
    }
    rows
}

fn add_charge(total: usize, bytes: usize) -> Result<usize> {
    total
        .checked_add(bytes)
        .ok_or_else(|| invalid("ASOF output workspace charge overflowed"))
}
