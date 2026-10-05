use super::{
    ArrayRef, BooleanArray, CandidateMap, DataType, Group, MemoryReservation, Result, ScalarValue,
    checked_bytes, df_error, ensure_reservation,
};
use datafusion::arrow::array::{Array, LargeStringArray, StringArray};

pub(super) const STATE_STRING_FACTOR: usize = 8;

pub(super) struct Bounds {
    entries: Vec<Entry>,
    base: usize,
}

struct Entry {
    aggregate: usize,
    state: usize,
    partial: usize,
}

impl Bounds {
    pub(super) fn new(
        indices: &[usize],
        previous: Option<&Group>,
        base: usize,
        credit: &MemoryReservation,
        name: &str,
    ) -> Result<Option<Box<Self>>> {
        if indices.is_empty() {
            return Ok(None);
        }
        let base = checked_bytes(
            base,
            [(1, size_of::<Self>()), (indices.len(), size_of::<Entry>())],
            name,
        )?;
        let total = indices.iter().try_fold(0, |total, &aggregate| {
            let width = previous.map_or(0, |group| scalar_bytes(&group.states[aggregate][0]));
            checked_bytes(total, [(width, 1)], name)
        })?;
        ensure_reservation(
            credit,
            checked_bytes(base, [(total, STATE_STRING_FACTOR)], name)?,
            name,
        )?;
        let entries = indices
            .iter()
            .map(|&aggregate| Entry {
                aggregate,
                state: previous.map_or(0, |group| scalar_bytes(&group.states[aggregate][0])),
                partial: 0,
            })
            .collect();
        Ok(Some(Box::new(Self { entries, base })))
    }

    pub(super) fn observe(
        &mut self,
        arguments: &[Vec<ArrayRef>],
        filters: &[Option<&BooleanArray>],
        row: usize,
        credit: &MemoryReservation,
        name: &str,
    ) -> Result<usize> {
        let mut growth = 0;
        let mut state = 0;
        for entry in &mut self.entries {
            if filters
                .get(entry.aggregate)
                .copied()
                .flatten()
                .is_none_or(|filter| !filter.is_null(row) && filter.value(row))
            {
                let width = array_bytes(&arguments[entry.aggregate][0], row);
                entry.state = entry.state.max(width);
                if width > entry.partial {
                    growth = checked_bytes(growth, [(width - entry.partial, 1)], name)?;
                    entry.partial = width;
                }
            }
            state = checked_bytes(state, [(entry.state, 1)], name)?;
        }
        if !self.entries.is_empty() {
            ensure_reservation(
                credit,
                checked_bytes(self.base, [(state, STATE_STRING_FACTOR)], name)?,
                name,
            )?;
        }
        Ok(growth)
    }
}

pub(super) fn prepare_grouped(
    arguments: &[Vec<ArrayRef>],
    filters: &[Option<&BooleanArray>],
    selection: Option<&BooleanArray>,
    indices: &[usize],
    slots: &[usize],
    candidates: &mut CandidateMap,
    name: &str,
) -> Result<usize> {
    let mut growth = 0;
    for (row, &rank) in indices.iter().enumerate() {
        if selection.is_some_and(|filter| !super::predicate::selected(filter, row)) {
            continue;
        }
        let candidate = candidates
            .get_mut(&slots[rank])
            .expect("string extrema candidate");
        let variable = candidate.variable.as_mut().expect("string extrema bounds");
        growth = checked_bytes(
            growth,
            [(
                variable.observe(arguments, filters, row, &candidate.group.reservation, name)?,
                1,
            )],
            name,
        )?;
    }
    Ok(growth)
}

fn array_bytes(array: &ArrayRef, row: usize) -> usize {
    if array.is_null(row) {
        return 0;
    }
    match array.data_type() {
        DataType::Utf8 => array
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(row)
            .len()
            .max(8),
        DataType::LargeUtf8 => array
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .unwrap()
            .value(row)
            .len()
            .max(8),
        _ => unreachable!("proven string extrema input"),
    }
}

fn scalar_bytes(value: &ScalarValue) -> usize {
    match value {
        ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) => value.len().max(8),
        _ => 0,
    }
}

pub(super) fn result_bytes(group: &Group, name: &str) -> Result<usize> {
    group.results.iter().try_fold(0, |total, value| {
        checked_bytes(total, [(scalar_bytes(value), 1)], name)
    })
}

pub(super) fn state_width(dtype: &DataType, name: &str) -> Result<usize> {
    match dtype {
        DataType::Utf8 | DataType::LargeUtf8 => Ok(size_of::<Option<Vec<u8>>>()),
        _ => dtype
            .primitive_width()
            .ok_or_else(|| df_error(name, "unsupported native state width")),
    }
}
