use super::{BoundedOrderOperator, SortTies, owned, workspace_bound};
use crate::{Result, dataframe::invalid};
use datafusion::{
    arrow::{
        compute::{
            LexicographicalComparator, SortColumn, SortOptions, concat_batches, lexsort_to_indices,
        },
        datatypes::{DataType, SchemaRef},
    },
    execution::memory_pool::MemoryPool,
};
use std::sync::Arc;

pub(super) fn validate_schema(
    schema: &SchemaRef,
    by: &[super::SortKey],
    spec: &super::SortSpec,
) -> Result<()> {
    let SortTies::Sequence { columns } = spec.ties() else {
        return Ok(());
    };
    for (i, name) in columns.iter().enumerate() {
        let path = format!("sort.spec.ties.columns[{i}]");
        let field = schema.field_with_name(name).map_err(|_| {
            invalid(
                &path,
                "unresolved_type",
                "sequence requires an existing input field",
            )
        })?;
        if by.iter().any(|key| key.column() == name) {
            return Err(invalid(
                &path,
                "duplicate_name",
                "sequence must not overlap primary sort keys",
            ));
        }
        if field.is_nullable() {
            return Err(invalid(
                &path,
                "ordering_required",
                "sequence requires non-NULL fields, unique over the whole H",
            ));
        }
        if !matches!(
            field.data_type(),
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Utf8
                | DataType::LargeUtf8
        ) {
            return Err(invalid(
                &path,
                "unsupported_type",
                "sequence requires integer or Utf8/LargeUtf8 fields",
            ));
        }
    }
    Ok(())
}
impl BoundedOrderOperator {
    pub(super) fn validate_sequence(
        &self,
        history: &owned::OwnedBatch,
        pool: &Arc<dyn MemoryPool>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let SortTies::Sequence { columns } = self.spec.ties() else {
            return Ok(());
        };
        let _credit = owned::reserve(
            pool,
            workspace_bound(history, columns.len())?,
            "sequence validation workspace",
        )?;
        check()?;
        let table = history.batch.table_payload()?;
        let input = concat_batches(table.schema(), table.batches()).map_err(owned::error)?;
        let mut keys = Vec::with_capacity(columns.len());
        for name in columns {
            let values = input.column_by_name(name).unwrap();
            if values.null_count() != 0 {
                return Err(invalid(
                    "sort.spec.ties",
                    "ordering_required",
                    "sequence must be non-NULL and unique over the whole H",
                ));
            }
            keys.push(SortColumn {
                values: values.clone(),
                options: Some(SortOptions {
                    descending: false,
                    nulls_first: false,
                }),
            });
        }
        let indices = lexsort_to_indices(&keys, None).map_err(owned::error)?;
        let compare = LexicographicalComparator::try_new(&keys).map_err(owned::error)?;
        for pair in indices.values().windows(2) {
            owned::check_row(pair[0] as usize, check)?;
            if compare.compare(pair[0] as usize, pair[1] as usize).is_eq() {
                return Err(invalid(
                    "sort.spec.ties",
                    "ordering_required",
                    "sequence tuple must be unique over the whole H; candidate not committed",
                ));
            }
        }
        check()
    }
    pub(super) fn validate_candidate(
        &self,
        history: &[Arc<owned::OwnedBatch>],
        pool: &Arc<dyn MemoryPool>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        if matches!(self.spec.ties(), SortTies::Unspecified) {
            return Ok(());
        }
        let combined = owned::combine(
            history,
            self.output[0].schema().unwrap(),
            &self.columns,
            pool,
            check,
        )?;
        self.validate_sequence(&combined, pool, check)
    }
}
