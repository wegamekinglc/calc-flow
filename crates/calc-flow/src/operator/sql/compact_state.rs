#[path = "compact_state/export.rs"]
mod export;

pub(in crate::operator::sql) use super::native_expression::NativeAggregateInput;
use super::native_expression::describe_input;
use super::{
    AggregateFunctionExpr, Arc, ArrayRef, CHUNK_ROWS, DataType, Field, Group, IncrementalSql,
    MemoryReservation, RecordBatch, Result, ScalarValue, Schema, SchemaRef, checked_bytes,
    df_error, ensure_reservation, native_grouped_result,
};
use datafusion::arrow::{
    array::{Array, Int64Array, LargeStringArray, StringArray, UInt64Array},
    datatypes::FieldRef,
};

const STATE_CHUNK_ROWS: usize = 128;

#[derive(Clone, Copy)]
enum StateColumn {
    Key(usize),
    Aggregate(usize, usize),
}

pub(in crate::operator::sql) struct NativeStateDescriptor {
    pub(in crate::operator::sql) key_inputs: Vec<NativeAggregateInput>,
    pub(in crate::operator::sql) key_fields: Vec<FieldRef>,
    pub(in crate::operator::sql) aggregate_names: Vec<String>,
    pub(in crate::operator::sql) aggregate_inputs: Vec<Vec<NativeAggregateInput>>,
    pub(in crate::operator::sql) aggregate_filters: Vec<Option<NativeAggregateInput>>,
    pub(in crate::operator::sql) count_all_rows: Vec<bool>,
    filtered_input: bool,
    pub(in crate::operator::sql) state_fields: Vec<Vec<FieldRef>>,
    pub(in crate::operator::sql) result_fields: Vec<FieldRef>,
    pub(in crate::operator::sql) wire_schema: SchemaRef,
    pub(in crate::operator::sql) output_schema: SchemaRef,
    pub(in crate::operator::sql) projection: Vec<NativeAggregateInput>,
    pub(in crate::operator::sql) post_filter: Option<NativeAggregateInput>,
    pub(in crate::operator::sql) post_order: Option<super::output_order::OrderDescriptor>,
    pub(in crate::operator::sql) expression_identity_bytes: usize,
    pub(in crate::operator::sql) group_count: usize,
    pub(in crate::operator::sql) policy: &'static str,
    _reservation: MemoryReservation,
}

pub(in crate::operator::sql) struct PaidNativeStateRecords {
    records: Vec<RecordBatch>,
    pub(in crate::operator::sql) descriptor: NativeStateDescriptor,
    _reservation: MemoryReservation,
}

impl PaidNativeStateRecords {
    pub(in crate::operator::sql) fn into_descriptor(self) -> NativeStateDescriptor {
        let Self {
            records,
            descriptor,
            _reservation: reservation,
        } = self;
        drop(records);
        drop(reservation);
        descriptor
    }

    pub(in crate::operator::sql) fn records(&self) -> &[RecordBatch] {
        &self.records
    }

    #[cfg(test)]
    pub(in crate::operator::sql) fn reserved_bytes(&self) -> usize {
        let Self {
            _reservation: reservation,
            descriptor,
            ..
        } = self;
        let NativeStateDescriptor {
            _reservation: descriptor_reservation,
            ..
        } = descriptor;
        reservation.size() + descriptor_reservation.size()
    }
}

impl IncrementalSql {
    pub(in crate::operator::sql) fn native_descriptor(
        &self,
        name: &str,
    ) -> Result<NativeStateDescriptor> {
        let reservation = self.descriptor_reservation(name)?;
        let (key_fields, key_inputs) = self.native_keys_descriptor(name)?;
        let aggregate_names = self
            .aggregates
            .iter()
            .map(|expression| expression.fun().name().to_owned())
            .collect::<Vec<_>>();
        let aggregate_inputs = self
            .aggregates
            .iter()
            .map(|expression| {
                expression
                    .expressions()
                    .iter()
                    .map(|input| describe_input(input.as_ref(), &self.schema, 0, name))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let count_all_rows = self.count_all_rows(&aggregate_names, &aggregate_inputs);
        let aggregate_filters = self.native_aggregate_filters(name)?;
        let projection = self
            .projection
            .iter()
            .map(|expression| describe_input(expression.as_ref(), &self.aggregate_schema, 0, name))
            .collect::<Result<Vec<_>>>()?;
        let post_filter = self
            .post_filter
            .as_ref()
            .map(|expression| describe_input(expression.as_ref(), &self.aggregate_schema, 0, name))
            .transpose()?;
        let post_order = self.native_output_order(name)?;
        let expression_identity_bytes = aggregate_inputs
            .iter()
            .flatten()
            .chain(&key_inputs)
            .chain(aggregate_filters.iter().flatten())
            .chain(&projection)
            .chain(&post_filter)
            .chain(
                post_order
                    .iter()
                    .flat_map(|order| order.keys.iter().map(|key| &key.0)),
            )
            .try_fold(0, |bytes, input| {
                checked_bytes(bytes, [(input.identity_bytes(name)?, 1)], name)
            })?;
        let state_fields = self
            .aggregates
            .iter()
            .map(|expression| {
                expression
                    .state_fields()
                    .map_err(|error| df_error(name, error))
            })
            .collect::<Result<Vec<_>>>()?;
        let result_fields = self
            .aggregates
            .iter()
            .map(|expression| expression.field())
            .collect::<Vec<_>>();
        let fields = key_fields
            .iter()
            .enumerate()
            .map(|(key, field)| field.as_ref().clone().with_name(format!("key_{key}")))
            .chain(
                state_fields
                    .iter()
                    .enumerate()
                    .flat_map(|(aggregate, fields)| {
                        fields.iter().enumerate().map(move |(state, field)| {
                            field
                                .as_ref()
                                .clone()
                                .with_name(format!("state_{aggregate}_{state}"))
                        })
                    }),
            )
            .collect::<Vec<_>>();
        Ok(NativeStateDescriptor {
            key_inputs,
            key_fields,
            aggregate_names,
            aggregate_inputs,
            aggregate_filters,
            count_all_rows,
            filtered_input: self.predicate.is_some(),
            state_fields,
            result_fields,
            wire_schema: Arc::new(Schema::new(fields)),
            output_schema: self.output_schema.clone(),
            projection,
            post_filter,
            post_order,
            expression_identity_bytes,
            group_count: self.groups.len(),
            policy: self.native_policy(),
            _reservation: reservation,
        })
    }

    fn native_keys_descriptor(
        &self,
        name: &str,
    ) -> Result<(Vec<FieldRef>, Vec<NativeAggregateInput>)> {
        let key_fields = self
            .keys
            .iter()
            .map(|key| key.field.clone())
            .collect::<Vec<_>>();
        let key_inputs = self
            .keys
            .iter()
            .map(|key| describe_input(key.expression.as_ref(), &self.schema, 0, name))
            .collect::<Result<Vec<_>>>()?;
        Ok((key_fields, key_inputs))
    }

    fn native_aggregate_filters(&self, name: &str) -> Result<Vec<Option<NativeAggregateInput>>> {
        (0..self.aggregates.len())
            .map(|index| {
                self.aggregate_filters
                    .get(index)
                    .and_then(Option::as_ref)
                    .map(|filter| describe_input(filter.as_ref(), &self.schema, 0, name))
                    .transpose()
            })
            .collect()
    }

    fn native_output_order(
        &self,
        name: &str,
    ) -> Result<Option<super::output_order::OrderDescriptor>> {
        self.post_order
            .as_ref()
            .map(|order| order.descriptor(&self.output_schema, name))
            .transpose()
    }

    fn descriptor_reservation(&self, name: &str) -> Result<MemoryReservation> {
        let schema_bytes = [&self.schema, &self.aggregate_schema, &self.output_schema]
            .into_iter()
            .try_fold(0, |bytes, schema| {
                let encoded = super::super::ipc::schema_bytes(schema)
                    .map_err(|error| df_error(name, error))?;
                checked_bytes(bytes, [(encoded, 1)], name)
            })?;
        let bytes = checked_bytes(
            4096,
            [
                (schema_bytes, 4),
                (self.keys.len(), 256),
                (self.aggregates.len(), 1024),
                (self.projection_nodes, 512),
                (self.input_nodes, 512),
            ],
            name,
        )?;
        let reservation = self.reservation.new_empty();
        ensure_reservation(&reservation, bytes, name)?;
        Ok(reservation)
    }

    fn count_all_rows(
        &self,
        functions: &[String],
        inputs: &[Vec<NativeAggregateInput>],
    ) -> Vec<bool> {
        functions
            .iter()
            .zip(inputs)
            .enumerate()
            .map(|(index, (function, inputs))| {
                function == "count"
                    && self.aggregate_filters.get(index).is_none_or(Option::is_none)
                    && matches!(inputs.as_slice(), [NativeAggregateInput::Literal(value)] if !value.is_null())
            })
            .collect()
    }

    pub(in crate::operator::sql) fn export_native_state(
        &self,
        name: &str,
        mut check_cancelled: impl FnMut() -> Result<()>,
    ) -> Result<PaidNativeStateRecords> {
        check_cancelled()?;
        let mut cursor = export::ExportCursor::new(self, name)?;
        while !cursor.step(&mut check_cancelled)? {}
        check_cancelled()?;
        Ok(cursor.finish())
    }

    pub(in crate::operator::sql) fn checkpoint_changes(&self) -> usize {
        self.dirty.slots().len()
    }

    pub(in crate::operator::sql) fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub(in crate::operator::sql) fn export_dirty_state(
        &self,
        name: &str,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<PaidNativeStateRecords> {
        let mut cursor = export::ExportCursor::dirty(self, name)?;
        while !cursor.step(&mut || check())? {}
        check()?;
        Ok(cursor.finish())
    }

    pub(in crate::operator::sql) async fn export_dirty_state_async(
        &self,
        name: &str,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<PaidNativeStateRecords> {
        let mut cursor = export::ExportCursor::dirty(self, name)?;
        loop {
            if cursor.step(&mut check)? {
                break;
            }
            check()?;
            tokio::task::yield_now().await;
        }
        check()?;
        Ok(cursor.finish())
    }

    pub(in crate::operator::sql) async fn export_native_state_async(
        &self,
        name: &str,
        mut check_cancelled: impl FnMut() -> Result<()>,
    ) -> Result<PaidNativeStateRecords> {
        check_cancelled()?;
        let mut cursor = export::ExportCursor::new(self, name)?;
        loop {
            let complete = cursor.step(&mut check_cancelled)?;
            check_cancelled()?;
            if complete {
                break;
            }
            tokio::task::yield_now().await;
        }
        Ok(cursor.finish())
    }

    pub(in crate::operator::sql) fn import_native_state(
        mut self,
        records: &[RecordBatch],
        historical_rows: u64,
        seen_input: bool,
        mut check_cancelled: impl FnMut() -> Result<()>,
        name: &str,
    ) -> Result<Self> {
        check_cancelled()?;
        if !self.groups.is_empty() || !self.index.is_empty() {
            return Err(df_error(name, "native state import requires an empty plan"));
        }
        let descriptor = self.native_descriptor(name)?;
        let validation = self.reservation.new_empty();
        ensure_reservation(
            &validation,
            checked_bytes(0, [(self.aggregates.len(), size_of::<u64>())], name)?,
            name,
        )?;
        let groups = validate_records(
            records,
            &descriptor,
            historical_rows,
            true,
            &mut check_cancelled,
            name,
        )?;
        validate_ledger(
            groups,
            historical_rows,
            seen_input,
            self.keys.is_empty(),
            self.predicate.is_some(),
            name,
        )?;
        if let Some(global) = &self.global_records {
            global.validate_state(records, historical_rows, name)?;
        }
        self.reserve_groups(groups, groups, name)?;
        self.load_records(records, &descriptor, false, &mut check_cancelled, name)?;
        check_cancelled()?;
        Ok(self)
    }

    pub(in crate::operator::sql) fn apply_delta_state(
        &mut self,
        records: &[RecordBatch],
        historical_rows: u64,
        check: &dyn Fn() -> Result<()>,
        name: &str,
    ) -> Result<()> {
        let descriptor = self.native_descriptor(name)?;
        let validation = self.reservation.new_empty();
        ensure_reservation(
            &validation,
            checked_bytes(4096, [(self.aggregates.len(), size_of::<u64>())], name)?,
            name,
        )?;
        let groups = validate_records(
            records,
            &descriptor,
            historical_rows,
            false,
            &mut || check(),
            name,
        )?;
        let capacity = self
            .groups
            .len()
            .checked_add(groups)
            .ok_or_else(|| df_error(name, "native delta capacity overflowed"))?;
        self.reserve_groups(groups, capacity, name)?;
        self.load_records(records, &descriptor, true, &mut || check(), name)
    }

    pub(in crate::operator::sql) fn validate_checkpoint_history(
        &self,
        rows: u64,
        seen: bool,
        check: &dyn Fn() -> Result<()>,
        name: &str,
    ) -> Result<()> {
        validate_ledger(
            self.groups.len(),
            rows,
            seen,
            self.keys.is_empty(),
            self.predicate.is_some(),
            name,
        )?;
        let descriptor = self.native_descriptor(name)?;
        let reservation = self.reservation.new_empty();
        ensure_reservation(
            &reservation,
            checked_bytes(4096, [(self.aggregates.len(), size_of::<u64>())], name)?,
            name,
        )?;
        let mut counts = vec![0; self.aggregates.len()];
        for (row, group) in self.groups.iter().enumerate() {
            if row % STATE_CHUNK_ROWS == 0 {
                check()?;
            }
            for (index, function) in descriptor.aggregate_names.iter().enumerate() {
                let count = match (function.as_str(), group.states[index].first()) {
                    ("count", Some(ScalarValue::Int64(Some(count)))) => u64::try_from(*count)
                        .map_err(|_| df_error(name, "native COUNT state is negative"))?,
                    ("avg", Some(ScalarValue::UInt64(count))) => count.unwrap_or(0),
                    _ => continue,
                };
                add_count(&mut counts[index], count, rows, name)?;
            }
        }
        for (count, all) in counts.iter().zip(&descriptor.count_all_rows) {
            if *all && !descriptor.filtered_input && *count != rows {
                return Err(df_error(
                    name,
                    "native all-row COUNT differs from historical rows",
                ));
            }
        }
        check()
    }

    fn load_records(
        &mut self,
        records: &[RecordBatch],
        descriptor: &NativeStateDescriptor,
        replace: bool,
        check_cancelled: &mut impl FnMut() -> Result<()>,
        name: &str,
    ) -> Result<()> {
        let workspace = self.reservation.new_empty();
        let mut seen = std::collections::BTreeSet::new();
        let rows = records.iter().try_fold(0usize, |sum, record| {
            sum.checked_add(record.num_rows())
                .ok_or_else(|| df_error(name, "native delta row count overflowed"))
        })?;
        let seen_credit = self.reservation.new_empty();
        ensure_reservation(
            &seen_credit,
            checked_bytes(4096, [(usize::from(replace) * rows, 128)], name)?,
            name,
        )?;
        for record in records {
            if record.num_rows() == 0 {
                check_cancelled()?;
            }
            for start in (0..record.num_rows()).step_by(STATE_CHUNK_ROWS) {
                check_cancelled()?;
                let rows = STATE_CHUNK_ROWS.min(record.num_rows() - start);
                let variable = record_variable_bytes(record, start, rows, name)?;
                let charge = checked_bytes(
                    4096,
                    [
                        (self.finalizer_bytes, 1),
                        (self.aggregate_bytes, 1),
                        (variable, 4),
                        (
                            rows,
                            checked_bytes(
                                256,
                                [
                                    (self.keys.len(), 128),
                                    (
                                        descriptor.wire_schema.fields().len(),
                                        size_of::<ScalarValue>() * 2,
                                    ),
                                ],
                                name,
                            )?,
                        ),
                    ],
                    name,
                )?;
                ensure_reservation(&workspace, charge, name)?;
                let chunk = record.slice(start, rows);
                let encoded = if let Some(converter) = &self.converter {
                    let columns = chunk.columns()[..self.keys.len()].to_vec();
                    Some(
                        converter
                            .convert_columns(&columns)
                            .map_err(|error| df_error(name, error))?,
                    )
                } else {
                    None
                };
                for row in 0..rows {
                    let key = encoded.as_ref().map(|encoded| encoded.row(row));
                    let key = key.as_ref().map_or(&[][..], |key| key.as_ref());
                    if !replace && self.index.contains_key(key) {
                        return Err(df_error(name, "native state contains duplicate keys"));
                    }
                    let group = self.import_group(&chunk, row, key, descriptor, name)?;
                    if replace && !seen.insert(group.key.clone()) {
                        return Err(df_error(name, "native state contains duplicate keys"));
                    }
                    if let Some(&slot) = self.index.get(key) {
                        self.groups[slot] = group;
                    } else {
                        let slot = self.groups.len();
                        self.index.insert(group.key.clone(), slot);
                        self.groups.push(group);
                    }
                }
            }
        }
        Ok(())
    }

    fn import_group(
        &self,
        record: &RecordBatch,
        row: usize,
        key: &[u8],
        descriptor: &NativeStateDescriptor,
        name: &str,
    ) -> Result<Group> {
        let variable = record_variable_bytes(record, row, 1, name)?;
        let charge = checked_bytes(
            self.aggregate_bytes,
            [
                (key.len(), 4),
                (variable, 4),
                (1, size_of::<Group>()),
                (
                    descriptor.wire_schema.fields().len(),
                    size_of::<ScalarValue>() * 2,
                ),
                (
                    self.aggregates.len(),
                    size_of::<Vec<ScalarValue>>() + size_of::<ScalarValue>(),
                ),
                (self.keys.len(), size_of::<ScalarValue>()),
            ],
            name,
        )?;
        let reservation = self.reservation.new_empty();
        ensure_reservation(&reservation, charge, name)?;
        let values = record.columns()[..self.keys.len()]
            .iter()
            .map(|array| {
                ScalarValue::try_from_array(array, row).map_err(|error| df_error(name, error))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut states = Vec::with_capacity(self.aggregates.len());
        let mut results = Vec::with_capacity(self.aggregates.len());
        let mut column = self.keys.len();
        for (aggregate, expression) in self.aggregates.iter().enumerate() {
            let width = descriptor.state_fields[aggregate].len();
            let state = record.columns()[column..column + width]
                .iter()
                .map(|array| {
                    ScalarValue::try_from_array(array, row).map_err(|error| df_error(name, error))
                })
                .collect::<Result<Vec<_>>>()?;
            let result = if let Some(global) = &self.global_records {
                global.result(aggregate, &state, name)?
            } else if self.requires_grouped_float_proof()
                && super::grouped_float::selected(expression)
            {
                super::grouped_sum::result(&state, name)?
            } else {
                restored_result(expression, &state, !self.keys.is_empty(), name)?
            };
            validate_scalar(&result, &descriptor.result_fields[aggregate], name)?;
            states.push(state);
            results.push(result);
            column += width;
        }
        Ok(Group {
            key: Arc::from(key),
            values: Arc::from(values),
            states,
            results,
            reservation,
        })
    }
}

fn validate_scalar(value: &ScalarValue, field: &Field, name: &str) -> Result<()> {
    if value.data_type() != *field.data_type() || (!field.is_nullable() && value.is_null()) {
        return Err(df_error(name, "native scalar differs from trusted field"));
    }
    Ok(())
}

fn validate_records(
    records: &[RecordBatch],
    descriptor: &NativeStateDescriptor,
    historical_rows: u64,
    complete: bool,
    check_cancelled: &mut impl FnMut() -> Result<()>,
    name: &str,
) -> Result<usize> {
    if records.is_empty() {
        return Err(df_error(name, "native state is missing its schema record"));
    }
    let mut groups = 0usize;
    let mut counts = vec![0u64; descriptor.aggregate_names.len()];
    for record in records {
        if record.num_rows() == 0 {
            check_cancelled()?;
        }
        if record.schema() != descriptor.wire_schema {
            return Err(df_error(
                name,
                "native state schema differs from trusted plan",
            ));
        }
        for (array, field) in record.columns().iter().zip(descriptor.wire_schema.fields()) {
            if !field.is_nullable() && array.null_count() != 0 {
                return Err(df_error(name, "native state has nulls in a required field"));
            }
        }
        groups = groups
            .checked_add(record.num_rows())
            .ok_or_else(|| df_error(name, "native group count overflowed"))?;
        for start in (0..record.num_rows()).step_by(STATE_CHUNK_ROWS) {
            check_cancelled()?;
            for row in start..(start + STATE_CHUNK_ROWS).min(record.num_rows()) {
                validate_counts(record, row, descriptor, historical_rows, &mut counts, name)?;
            }
        }
    }
    for (count, all_rows) in counts.iter().zip(&descriptor.count_all_rows) {
        if complete && *all_rows && !descriptor.filtered_input && *count != historical_rows {
            return Err(df_error(
                name,
                "native all-row COUNT differs from historical rows",
            ));
        }
    }
    Ok(groups)
}

fn validate_counts(
    record: &RecordBatch,
    row: usize,
    descriptor: &NativeStateDescriptor,
    historical_rows: u64,
    totals: &mut [u64],
    name: &str,
) -> Result<()> {
    let mut column = descriptor.key_fields.len();
    for (aggregate, function) in descriptor.aggregate_names.iter().enumerate() {
        match function.as_str() {
            "count" => {
                let array = record
                    .column(column)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| df_error(name, "native COUNT state is not Int64"))?;
                let count = u64::try_from(array.value(row))
                    .map_err(|_| df_error(name, "native COUNT state is negative"))?;
                add_count(&mut totals[aggregate], count, historical_rows, name)?;
                if descriptor.count_all_rows[aggregate]
                    && !descriptor.key_fields.is_empty()
                    && count == 0
                {
                    return Err(df_error(name, "native grouped all-row COUNT is zero"));
                }
            }
            "avg" => {
                let counts = record
                    .column(column)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| df_error(name, "native AVG count is not UInt64"))?;
                let count = if counts.is_null(row) {
                    0
                } else {
                    counts.value(row)
                };
                add_count(&mut totals[aggregate], count, historical_rows, name)?;
                let null_sum = record.column(column + 1).is_null(row);
                if count > historical_rows || (count == 0) != null_sum {
                    return Err(df_error(name, "native AVG count and sum are inconsistent"));
                }
            }
            _ => {
                if historical_rows == 0 && !record.column(column).is_null(row) {
                    return Err(df_error(
                        name,
                        "native aggregate has a value before input rows",
                    ));
                }
            }
        }
        column += descriptor.state_fields[aggregate].len();
    }
    Ok(())
}

fn add_count(total: &mut u64, count: u64, historical_rows: u64, name: &str) -> Result<()> {
    *total = total
        .checked_add(count)
        .filter(|&total| total <= historical_rows)
        .ok_or_else(|| df_error(name, "native aggregate count exceeds historical rows"))?;
    Ok(())
}

fn validate_ledger(
    groups: usize,
    rows: u64,
    seen: bool,
    scalar: bool,
    filtered: bool,
    name: &str,
) -> Result<()> {
    let valid = if !seen {
        rows == 0 && groups == 0
    } else if scalar {
        groups == 1
    } else {
        u64::try_from(groups)
            .is_ok_and(|groups| groups <= rows && (rows == 0 || groups != 0 || filtered))
    };
    if !valid {
        return Err(df_error(
            name,
            "native group census differs from historical ledger",
        ));
    }
    Ok(())
}

fn record_variable_bytes(
    record: &RecordBatch,
    start: usize,
    rows: usize,
    name: &str,
) -> Result<usize> {
    let mut bytes = 0;
    for array in record.columns() {
        for row in start..start + rows {
            let width = match array.data_type() {
                DataType::Utf8 => array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| df_error(name, "native Utf8 key differs"))?
                    .value(row)
                    .len(),
                DataType::LargeUtf8 => array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .ok_or_else(|| df_error(name, "native LargeUtf8 key differs"))?
                    .value(row)
                    .len(),
                _ => 0,
            };
            bytes = checked_bytes(bytes, [(width, 1)], name)?;
        }
    }
    Ok(bytes)
}

fn restored_result(
    expression: &AggregateFunctionExpr,
    state: &[ScalarValue],
    grouped: bool,
    name: &str,
) -> Result<ScalarValue> {
    if grouped && expression.fun().name() == "avg" {
        return native_grouped_result(expression, state, name);
    }
    let mut accumulator = expression
        .create_accumulator()
        .map_err(|error| df_error(name, error))?;
    let arrays = state
        .iter()
        .map(|value| value.to_array().map_err(|error| df_error(name, error)))
        .collect::<Result<Vec<_>>>()?;
    accumulator
        .merge_batch(&arrays)
        .map_err(|error| df_error(name, error))?;
    accumulator
        .evaluate()
        .map_err(|error| df_error(name, error))
}

#[path = "compact_state_tests.rs"]
#[cfg(test)]
mod tests;
