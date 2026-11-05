use super::{
    Arc, ArrayRef, CHUNK_ROWS, DataFusionRuntime, DataType, GroupKey, GroupValues,
    MemoryReservation, PhysicalExpr, RecordBatch, Result, Schema, StreamOperatorContext,
    checked_bytes, df_error, grouped_sum, size_of,
};
use crate::StateBudget;
use datafusion::{
    arrow::{
        array::{StringArray, UInt32Array},
        compute::take,
        datatypes::Field,
    },
    execution::memory_pool::UnboundedMemoryPool,
    physical_expr::{aggregate::AggregateFunctionExpr, expressions::Column},
    physical_plan::aggregates::group_values::multi_group_by::GroupValuesColumn,
};

#[cfg(test)]
#[path = "columnar_key_capacity_tests.rs"]
mod tests;

#[derive(Clone, Copy)]
pub(super) struct Plan {
    columns: [usize; 2],
}

impl Plan {
    pub(super) fn bind(
        bound: &super::PhysicalSqlPlan,
        checks: &[Arc<dyn PhysicalExpr>],
        runtime: &DataFusionRuntime,
    ) -> Option<Self> {
        let (aggregates, _, _, predicate, _, keys) = bound;
        if predicate.is_some() || !checks.is_empty() || !single_sum_column(aggregates) {
            return None;
        }
        let [left, right] = keys.as_slice() else {
            return None;
        };
        if std::any::Any::type_id(runtime.incremental_memory_pool().as_ref())
            != std::any::TypeId::of::<UnboundedMemoryPool>()
        {
            return None;
        }
        Some(Self {
            columns: [utf8_column(left)?, utf8_column(right)?],
        })
    }

    pub(super) async fn prepare(
        self,
        records: &[RecordBatch],
        rows: usize,
        budget: Option<StateBudget>,
        reservation: MemoryReservation,
        context: &StreamOperatorContext<'_>,
        name: &str,
    ) -> Result<Option<Box<Dictionary>>> {
        let Some(groups) = selected_groups(rows, budget) else {
            return Ok(None);
        };
        let Some(bytes) = self.census(records, context).await? else {
            return Ok(None);
        };
        let Some(envelope) = Envelope::new(rows, groups, bytes, name) else {
            return Ok(None);
        };
        if reservation.try_grow(envelope.charge).is_err() {
            return Ok(None);
        }
        Ok(Dictionary::new(envelope, reservation, name)
            .ok()
            .map(Box::new))
    }

    async fn census(
        self,
        records: &[RecordBatch],
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<[usize; 2]>> {
        let mut bytes = [0_usize; 2];
        for (index, record) in records.iter().enumerate() {
            context.check_cancelled()?;
            if index != 0 && index % 128 == 0 {
                tokio::task::yield_now().await;
            }
            for (total, column) in bytes.iter_mut().zip(self.columns) {
                let Some(span) = utf8_span(record.column(column)) else {
                    return Ok(None);
                };
                let Some(next) = total.checked_add(span) else {
                    return Ok(None);
                };
                *total = next;
            }
        }
        Ok(Some(bytes))
    }
}

fn single_sum_column(expressions: &[Arc<AggregateFunctionExpr>]) -> bool {
    let [expression] = expressions else {
        return false;
    };
    grouped_sum::integer_sum(expression)
        && matches!(expression.expressions().as_slice(), [argument] if argument.is::<Column>())
}

fn utf8_column(key: &GroupKey) -> Option<usize> {
    (key.field.data_type() == &DataType::Utf8)
        .then(|| key.expression.downcast_ref::<Column>().map(Column::index))
        .flatten()
}

fn utf8_span(array: &ArrayRef) -> Option<usize> {
    let array = array.as_any().downcast_ref::<StringArray>()?;
    let offsets = array.value_offsets();
    usize::try_from(offsets.last()?.checked_sub(offsets[0])?).ok()
}

pub(super) fn selected_groups(rows: usize, budget: Option<StateBudget>) -> Option<usize> {
    let budget =
        budget.filter(|budget| budget.max_bytes() == StateBudget::default().max_bytes())?;
    let groups = usize::try_from(budget.max_rows()).ok()?;
    (groups <= rows / 2).then_some(groups)
}

#[derive(Clone, Copy)]
struct Envelope {
    groups: usize,
    rows: usize,
    charge: usize,
}

impl Envelope {
    fn new(rows: usize, groups: usize, bytes: [usize; 2], name: &str) -> Option<Self> {
        let rows_in_chunk = rows.min(CHUNK_ROWS);
        let groups = groups.checked_add(rows_in_chunk)?.min(rows);
        if groups > usize::try_from(i64::MAX).ok()?
            || bytes
                .iter()
                .any(|bytes| *bytes > usize::try_from(i32::MAX).unwrap())
        {
            return None;
        }
        let charge = workspace_bytes(groups, rows_in_chunk, bytes, name)?;
        (charge <= usize::try_from(isize::MAX).ok()?).then_some(Self {
            groups,
            rows: rows_in_chunk,
            charge,
        })
    }
}

fn round64(bytes: usize) -> Option<usize> {
    bytes.checked_add(63)?.checked_div(64)?.checked_mul(64)
}

fn key_builder_bytes(groups: usize, bytes: usize, name: &str) -> Option<usize> {
    checked_bytes(
        256,
        [
            (round64(bytes)?.max(8192), 4),
            (groups.checked_add(5)?, 4 * size_of::<i32>()),
            (round64(groups.div_ceil(8))?, 4),
        ],
        name,
    )
    .ok()
}

fn workspace_bytes(groups: usize, rows: usize, bytes: [usize; 2], name: &str) -> Option<usize> {
    let buckets = groups
        .checked_mul(4)?
        .max(256)
        .checked_next_power_of_two()?;
    let hash = checked_bytes(256, [(buckets, 2 * (size_of::<(u64, u64)>() + 1))], name).ok()?;
    let canonical = checked_bytes(4096, [(bytes[0], 4), (bytes[1], 4), (rows, 128)], name).ok()?;
    checked_bytes(
        4096,
        [
            (key_builder_bytes(groups, bytes[0], name)?, 1),
            (key_builder_bytes(groups, bytes[1], name)?, 1),
            (hash, 1),
            (rows, 4 * size_of::<u64>()),
            (groups, size_of::<usize>()),
            (rows, size_of::<[usize; 8]>()),
            (canonical, 1),
        ],
        name,
    )
    .ok()
}

pub(super) struct Dictionary {
    groups: GroupValuesColumn<true>,
    indices: Vec<usize>,
    ranks: Vec<usize>,
    first_rows: Vec<u32>,
    capacity: usize,
    reservation: MemoryReservation,
}

fn reserved_vec<T>(capacity: usize, name: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|error| df_error(name, error))?;
    Ok(values)
}

impl Dictionary {
    fn new(envelope: Envelope, reservation: MemoryReservation, name: &str) -> Result<Self> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key0", DataType::Utf8, true),
            Field::new("key1", DataType::Utf8, true),
        ]));
        Ok(Self {
            groups: GroupValuesColumn::try_new(schema).map_err(|error| df_error(name, error))?,
            indices: reserved_vec(envelope.rows, name)?,
            ranks: reserved_vec(envelope.groups, name)?,
            first_rows: reserved_vec(envelope.rows, name)?,
            capacity: envelope.groups,
            reservation,
        })
    }

    pub(super) fn intern(&mut self, keys: &[ArrayRef], name: &str) -> Result<()> {
        let previous = self.groups.len();
        self.groups
            .intern(keys, &mut self.indices)
            .map_err(|error| df_error(name, error))?;
        self.capture_first_rows(previous, name)?;
        self.validate_capacity(name)
    }

    fn capture_first_rows(&mut self, mut next: usize, name: &str) -> Result<()> {
        if self.groups.len() > self.capacity {
            return Err(df_error(name, "columnar key capacity exceeded"));
        }
        self.first_rows.clear();
        for (row, index) in self.indices.iter().enumerate() {
            if *index == next {
                self.first_rows
                    .push(u32::try_from(row).map_err(|error| df_error(name, error))?);
                next = next
                    .checked_add(1)
                    .ok_or_else(|| df_error(name, "columnar group count overflowed"))?;
            }
        }
        if next != self.groups.len() {
            return Err(df_error(
                name,
                "columnar key first-appearance order differs",
            ));
        }
        Ok(())
    }

    pub(super) fn new_keys(&self, keys: &[ArrayRef], name: &str) -> Result<Option<Vec<ArrayRef>>> {
        if self.first_rows.is_empty() {
            return Ok(None);
        }
        let indices = UInt32Array::from(self.first_rows.clone());
        keys.iter()
            .map(|array| {
                take(array.as_ref(), &indices, None).map_err(|error| df_error(name, error))
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    pub(super) fn add_rank(&mut self, rank: usize) {
        self.ranks.push(rank);
    }

    pub(super) fn remap(&mut self, name: &str) -> Result<&[usize]> {
        for index in &mut self.indices {
            *index = *self
                .ranks
                .get(*index)
                .ok_or_else(|| df_error(name, "columnar group ID has no partial rank"))?;
        }
        Ok(&self.indices)
    }

    fn validate_capacity(&self, name: &str) -> Result<()> {
        let actual = checked_bytes(
            self.groups.size(),
            [
                (self.indices.capacity(), size_of::<usize>()),
                (self.ranks.capacity(), size_of::<usize>()),
                (self.first_rows.capacity(), size_of::<u32>()),
            ],
            name,
        )?;
        if actual > self.reservation.size() {
            return Err(df_error(name, "columnar keys exceeded reserved capacity"));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn reserved_bytes(&self) -> usize {
        self.reservation.size()
    }
}
