#[path = "compact_state/export.rs"]
mod export;

pub(in crate::operator::sql) use super::native_expression::NativeAggregateInput;
use super::native_expression::describe_input;
use super::{
    AggregateFunctionExpr, Arc, ArrayRef, CHUNK_ROWS, DataType, Field, Group, GroupAggregates,
    IncrementalSql, MemoryReservation, RecordBatch, Result, ScalarValue, Schema, SchemaRef,
    checked_bytes, df_error, ensure_reservation, native_grouped_result,
};
use crate::operator::keyed::ExportValue;
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
    pub(in crate::operator::sql) input_checks: Vec<NativeAggregateInput>,
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
    pub(in crate::operator::sql) ordered_last: Option<OrderedLastDescriptor>,
    pub(in crate::operator::sql) policy: &'static str,
    pub(in crate::operator::sql) key_layout: Option<String>,
    _reservation: MemoryReservation,
}

pub(in crate::operator::sql) struct OrderedLastDescriptor {
    pub keys: Vec<(NativeAggregateInput, bool, bool)>,
    pub ignore_nulls: bool,
}

pub(in crate::operator::sql) struct PaidNativeStateRecords {
    records: Vec<RecordBatch>,
    pub(in crate::operator::sql) descriptor: NativeStateDescriptor,
    _reservation: MemoryReservation,
}

impl PaidNativeStateRecords {
    fn new(
        records: Vec<RecordBatch>,
        descriptor: NativeStateDescriptor,
        reservation: MemoryReservation,
    ) -> Self {
        Self {
            records,
            descriptor,
            _reservation: reservation,
        }
    }

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

struct ValidatedNativeImport {
    descriptor: NativeStateDescriptor,
    groups: usize,
    _validation: MemoryReservation,
}

struct NativeAggregateDescriptor {
    aggregate_names: Vec<String>,
    aggregate_inputs: Vec<Vec<NativeAggregateInput>>,
    count_all_rows: Vec<bool>,
    aggregate_filters: Vec<Option<NativeAggregateInput>>,
    input_checks: Vec<NativeAggregateInput>,
}

struct NativeOutputDescriptor {
    projection: Vec<NativeAggregateInput>,
    post_filter: Option<NativeAggregateInput>,
    post_order: Option<super::output_order::OrderDescriptor>,
}

impl IncrementalSql {
    pub(in crate::operator::sql) fn native_descriptor(
        &self,
        name: &str,
    ) -> Result<NativeStateDescriptor> {
        let reservation = self.descriptor_reservation(name)?;
        let (key_fields, key_inputs) = self.native_keys_descriptor(name)?;
        let aggregates = self.native_aggregate_descriptor(name)?;
        let output = self.native_output_descriptor(name)?;
        let (state_fields, ordered_last) = self.native_state_layout(name)?;
        let expression_identity_bytes = Self::native_expression_identity_bytes(
            name,
            &key_inputs,
            &aggregates,
            &output,
            ordered_last.as_ref(),
        )?;
        let NativeAggregateDescriptor {
            aggregate_names,
            aggregate_inputs,
            count_all_rows,
            aggregate_filters,
            input_checks,
        } = aggregates;
        let NativeOutputDescriptor {
            projection,
            post_filter,
            post_order,
        } = output;
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
        let key_layout = self
            .keyed_sum
            .as_ref()
            .map(super::keyed_sum::KeyedSum::layout_fingerprint)
            .or_else(|| {
                self.keyed_state
                    .as_ref()
                    .map(super::keyed_state::KeyedState::layout_fingerprint)
            })
            .map(hex::encode);
        let wire_metadata = key_layout
            .as_ref()
            .map(|layout| std::collections::HashMap::from([("key_layout".into(), layout.clone())]))
            .unwrap_or_default();
        Ok(NativeStateDescriptor {
            key_inputs,
            key_fields,
            aggregate_names,
            aggregate_inputs,
            aggregate_filters,
            input_checks,
            count_all_rows,
            filtered_input: self.predicate.is_some(),
            state_fields,
            result_fields,
            wire_schema: Arc::new(Schema::new_with_metadata(fields, wire_metadata)),
            output_schema: self.output_schema.clone(),
            projection,
            post_filter,
            post_order,
            expression_identity_bytes,
            group_count: self.group_count(),
            ordered_last,
            policy: self.native_policy(),
            key_layout,
            _reservation: reservation,
        })
    }

    fn native_expression_identity_bytes(
        name: &str,
        key_inputs: &[NativeAggregateInput],
        aggregates: &NativeAggregateDescriptor,
        output: &NativeOutputDescriptor,
        ordered_last: Option<&OrderedLastDescriptor>,
    ) -> Result<usize> {
        aggregates
            .aggregate_inputs
            .iter()
            .flatten()
            .chain(key_inputs)
            .chain(aggregates.aggregate_filters.iter().flatten())
            .chain(&aggregates.input_checks)
            .chain(&output.projection)
            .chain(&output.post_filter)
            .chain(
                output
                    .post_order
                    .iter()
                    .flat_map(|order| order.keys.iter().map(|key| &key.0)),
            )
            .chain(
                ordered_last
                    .into_iter()
                    .flat_map(|ordered| ordered.keys.iter().map(|key| &key.0)),
            )
            .try_fold(0, |bytes, input| {
                checked_bytes(bytes, [(input.identity_bytes(name)?, 1)], name)
            })
    }

    fn native_aggregate_descriptor(&self, name: &str) -> Result<NativeAggregateDescriptor> {
        let aggregate_names = self
            .aggregates
            .iter()
            .map(|expression| expression.fun().name().to_owned())
            .collect::<Vec<_>>();
        let aggregate_inputs = self.native_aggregate_inputs(name)?;
        let count_all_rows = self.count_all_rows(&aggregate_names, &aggregate_inputs);
        let aggregate_filters = self.native_aggregate_filters(name)?;
        let input_checks = self.native_input_checks(name)?;
        Ok(NativeAggregateDescriptor {
            aggregate_names,
            aggregate_inputs,
            count_all_rows,
            aggregate_filters,
            input_checks,
        })
    }

    fn native_output_descriptor(&self, name: &str) -> Result<NativeOutputDescriptor> {
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
        Ok(NativeOutputDescriptor {
            projection,
            post_filter,
            post_order,
        })
    }

    fn native_aggregate_inputs(&self, name: &str) -> Result<Vec<Vec<NativeAggregateInput>>> {
        self.aggregates
            .iter()
            .map(|expression| {
                expression
                    .expressions()
                    .iter()
                    .map(|input| describe_input(input.as_ref(), &self.schema, 0, name))
                    .collect()
            })
            .collect()
    }

    fn native_input_checks(&self, name: &str) -> Result<Vec<NativeAggregateInput>> {
        self.input_checks
            .iter()
            .map(|expression| describe_input(expression.as_ref(), &self.schema, 0, name))
            .collect()
    }

    fn native_state_layout(
        &self,
        name: &str,
    ) -> Result<(Vec<Vec<FieldRef>>, Option<OrderedLastDescriptor>)> {
        let fields = self
            .aggregates
            .iter()
            .map(|expression| {
                let fields = expression
                    .state_fields()
                    .map_err(|error| df_error(name, error))?;
                let last = fields.len().saturating_sub(1);
                Ok(fields
                    .into_iter()
                    .enumerate()
                    .map(|(index, field)| {
                        if self.ordered_last {
                            Arc::new(field.as_ref().clone().with_nullable(index != last))
                        } else {
                            field
                        }
                    })
                    .collect())
            })
            .collect::<Result<_>>()?;
        Ok((fields, self.ordered_last_descriptor(name)?))
    }

    fn ordered_last_descriptor(&self, name: &str) -> Result<Option<OrderedLastDescriptor>> {
        if !self.ordered_last {
            return Ok(None);
        }
        let expression = &self.aggregates[0];
        Ok(Some(OrderedLastDescriptor {
            keys: expression
                .order_bys()
                .iter()
                .map(|sort| {
                    Ok((
                        describe_input(sort.expr.as_ref(), &self.schema, 0, name)?,
                        sort.options.descending,
                        sort.options.nulls_first,
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
            ignore_nulls: expression.ignore_nulls(),
        }))
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
        let reservation = self.scratch.new_empty();
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
        if self.keyed_sum.is_some() || self.keyed_state.is_some() {
            return self.export_keyed(None, name, &mut check_cancelled);
        }
        let mut cursor = export::ExportCursor::new(self, name)?;
        while !cursor.step(&mut check_cancelled)? {}
        check_cancelled()?;
        Ok(cursor.finish())
    }

    pub(in crate::operator::sql) fn checkpoint_changes(&self) -> usize {
        self.dirty.slots().len()
    }

    pub(in crate::operator::sql) fn group_count(&self) -> usize {
        if let Some(keyed) = &self.keyed_sum {
            keyed.len()
        } else if let Some(keyed) = &self.keyed_state {
            keyed.len()
        } else {
            self.groups.len()
        }
    }

    fn export_keyed(
        &self,
        slots: Option<&[usize]>,
        name: &str,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<PaidNativeStateRecords> {
        let (descriptor, reservation) = self.prepare_keyed_export(slots, name)?;
        let records = self.keyed_export_records(slots, descriptor.wire_schema.clone(), check)?;
        check()?;
        Ok(PaidNativeStateRecords::new(
            records,
            descriptor,
            reservation,
        ))
    }

    fn prepare_keyed_export(
        &self,
        slots: Option<&[usize]>,
        name: &str,
    ) -> Result<(NativeStateDescriptor, MemoryReservation)> {
        let mut descriptor = self.native_descriptor(name)?;
        let rows = slots.map_or(self.group_count(), <[usize]>::len);
        descriptor.group_count = rows;
        let charge = self.keyed_export_charge(&descriptor, rows, slots, name)?;
        let reservation = self.scratch.new_empty();
        ensure_reservation(&reservation, charge, name)?;
        Ok((descriptor, reservation))
    }

    fn keyed_export_charge(
        &self,
        descriptor: &NativeStateDescriptor,
        rows: usize,
        slots: Option<&[usize]>,
        name: &str,
    ) -> Result<usize> {
        let record_count = rows.div_ceil(CHUNK_ROWS).max(1);
        let columns = checked_bytes(0, [(descriptor.wire_schema.fields().len(), 512)], name)?;
        let charge = checked_bytes(
            4096,
            [
                (record_count, size_of::<RecordBatch>()),
                (record_count, columns),
            ],
            name,
        )?;
        let output_charge = if let Some(keyed) = &self.keyed_sum {
            keyed.output_charge(slots)?
        } else if let Some(keyed) = &self.keyed_state {
            keyed.output_charge(slots)?
        } else {
            return Err(df_error(name, "keyed aggregate plan is missing"));
        };
        checked_bytes(charge, [(output_charge, 1)], name)
    }

    fn keyed_export_records(
        &self,
        slots: Option<&[usize]>,
        schema: SchemaRef,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<RecordBatch>> {
        if let Some(keyed) = &self.keyed_sum {
            keyed.export(slots, schema, check)
        } else if let Some(keyed) = &self.keyed_state {
            keyed.export(slots, schema, check)
        } else {
            Err(df_error("keyed export", "keyed aggregate plan is missing"))
        }
    }

    fn keyed_export_rows(
        &self,
        slots: Option<&[usize]>,
        name: &str,
    ) -> Result<Vec<Vec<ExportValue>>> {
        if let Some(keyed) = &self.keyed_sum {
            keyed.export_rows(slots)
        } else if let Some(keyed) = &self.keyed_state {
            keyed.export_rows(slots)
        } else {
            Err(df_error(name, "keyed aggregate plan is missing"))
        }
    }

    async fn export_keyed_async(
        &self,
        slots: Option<Vec<usize>>,
        name: &str,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<PaidNativeStateRecords> {
        let (descriptor, reservation) = self.prepare_keyed_export(slots.as_deref(), name)?;
        let rows = self.keyed_export_rows(slots.as_deref(), name)?;
        let mut cursor = export::ScalarExportCursor::new(rows, descriptor, reservation, name);
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

    pub(in crate::operator::sql) fn export_dirty_state(
        &self,
        name: &str,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<PaidNativeStateRecords> {
        if self.keyed_sum.is_some() || self.keyed_state.is_some() {
            return self.export_dirty_keyed(name, check);
        }
        let mut cursor = export::ExportCursor::dirty(self, name)?;
        while !cursor.step(&mut || check())? {}
        check()?;
        Ok(cursor.finish())
    }

    fn export_dirty_keyed(
        &self,
        name: &str,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<PaidNativeStateRecords> {
        let mut slots = self.dirty.slots().to_vec();
        slots.sort_unstable();
        self.export_keyed(Some(&slots), name, &mut || check())
    }

    pub(in crate::operator::sql) async fn export_dirty_state_async(
        &self,
        name: &str,
        mut check: impl FnMut() -> Result<()>,
    ) -> Result<PaidNativeStateRecords> {
        if self.keyed_sum.is_some() || self.keyed_state.is_some() {
            let mut slots = self.dirty.slots().to_vec();
            slots.sort_unstable();
            return self.export_keyed_async(Some(slots), name, check).await;
        }
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
        if self.keyed_sum.is_some() || self.keyed_state.is_some() {
            return self.export_keyed_async(None, name, check_cancelled).await;
        }
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
        self.ensure_native_import_empty(name)?;
        let ValidatedNativeImport {
            descriptor,
            groups,
            _validation,
        } = self.prepare_native_import(
            records,
            historical_rows,
            seen_input,
            &mut check_cancelled,
            name,
        )?;
        self.install_native_import(records, &descriptor, groups, &mut check_cancelled, name)?;
        check_cancelled()?;
        Ok(self)
    }

    fn ensure_native_import_empty(&self, name: &str) -> Result<()> {
        if !self.groups.is_empty()
            || !self.index.is_empty()
            || self
                .keyed_sum
                .as_ref()
                .is_some_and(|keyed| !keyed.is_empty())
            || self
                .keyed_state
                .as_ref()
                .is_some_and(|keyed| !keyed.is_empty())
        {
            return Err(df_error(name, "native state import requires an empty plan"));
        }
        Ok(())
    }

    fn install_native_import(
        &mut self,
        records: &[RecordBatch],
        descriptor: &NativeStateDescriptor,
        groups: usize,
        check_cancelled: &mut impl FnMut() -> Result<()>,
        name: &str,
    ) -> Result<()> {
        self.reserve_groups(groups, groups, name)?;
        self.load_records(records, descriptor, false, check_cancelled, name)?;
        Ok(())
    }

    fn prepare_native_import(
        &self,
        records: &[RecordBatch],
        historical_rows: u64,
        seen_input: bool,
        check_cancelled: &mut impl FnMut() -> Result<()>,
        name: &str,
    ) -> Result<ValidatedNativeImport> {
        let descriptor = self.native_descriptor(name)?;
        let validation = self.validation_credit(0, name)?;
        let groups = validate_records(
            records,
            &descriptor,
            historical_rows,
            true,
            check_cancelled,
            name,
        )?;
        self.check_import_group_budget(groups, name)?;
        self.validate_import_history(records, groups, (historical_rows, seen_input), name)?;
        Ok(ValidatedNativeImport {
            descriptor,
            groups,
            _validation: validation,
        })
    }

    fn validate_import_history(
        &self,
        records: &[RecordBatch],
        groups: usize,
        history: (u64, bool),
        name: &str,
    ) -> Result<()> {
        validate_ledger(
            groups,
            history.0,
            history.1,
            self.keys.is_empty(),
            self.predicate.is_some(),
            name,
        )?;
        if history.1
            && let Some(global) = &self.global_records
        {
            global.validate_state(records, history.0, name)?;
        }
        Ok(())
    }

    fn validation_credit(&self, base: usize, name: &str) -> Result<MemoryReservation> {
        let reservation = self.scratch.new_empty();
        ensure_reservation(
            &reservation,
            checked_bytes(base, [(self.aggregates.len(), size_of::<u64>())], name)?,
            name,
        )?;
        Ok(reservation)
    }

    pub(in crate::operator::sql) fn apply_delta_state(
        &mut self,
        records: &[RecordBatch],
        historical_rows: u64,
        check: &dyn Fn() -> Result<()>,
        name: &str,
    ) -> Result<()> {
        let descriptor = self.native_descriptor(name)?;
        let _validation = self.validation_credit(4096, name)?;
        let mut check = check;
        let groups = validate_records(
            records,
            &descriptor,
            historical_rows,
            false,
            &mut check,
            name,
        )?;
        let capacity = self
            .group_count()
            .checked_add(groups)
            .ok_or_else(|| df_error(name, "native delta capacity overflowed"))?;
        self.reserve_groups(groups, capacity, name)?;
        self.load_records(records, &descriptor, true, &mut check, name)
    }

    pub(in crate::operator::sql) fn validate_checkpoint_history(
        &self,
        rows: u64,
        seen: bool,
        check: &dyn Fn() -> Result<()>,
        name: &str,
    ) -> Result<()> {
        validate_ledger(
            self.group_count(),
            rows,
            seen,
            self.keys.is_empty(),
            self.predicate.is_some(),
            name,
        )?;
        let descriptor = self.native_descriptor(name)?;
        let _validation = self.validation_credit(4096, name)?;
        let counts = self.history_counts(&descriptor, rows, check, name)?;
        validate_all_row_totals(&counts, &descriptor, rows, true, name)?;
        check()
    }

    fn history_counts(
        &self,
        descriptor: &NativeStateDescriptor,
        rows: u64,
        check: &dyn Fn() -> Result<()>,
        name: &str,
    ) -> Result<Vec<u64>> {
        if let Some(keyed) = &self.keyed_state {
            return keyed.history_counts(&descriptor.aggregate_names, rows, check, name);
        }
        let mut counts = vec![0; self.aggregates.len()];
        for (row, group) in self.groups.iter().enumerate() {
            if row % STATE_CHUNK_ROWS == 0 {
                check()?;
            }
            add_group_counts(group, descriptor, rows, &mut counts, name)?;
        }
        Ok(counts)
    }

    fn load_records(
        &mut self,
        records: &[RecordBatch],
        descriptor: &NativeStateDescriptor,
        replace: bool,
        check_cancelled: &mut impl FnMut() -> Result<()>,
        name: &str,
    ) -> Result<()> {
        if let Some(keyed) = &self.keyed_sum {
            return keyed.restore_records(
                records,
                self.keys.len(),
                replace,
                self.compact_budget,
                check_cancelled,
            );
        }
        if let Some(keyed) = &self.keyed_state {
            return keyed.restore_records(
                records,
                super::keyed_state::RestoreSpec::new(
                    self.keys.len(),
                    replace,
                    &self.aggregates,
                    self.global_records.as_ref(),
                    self.sequential.is_some(),
                ),
                self.compact_budget,
                check_cancelled,
            );
        }
        let workspace = self.scratch.new_empty();
        let mut seen = std::collections::BTreeSet::new();
        let _seen_credit = self.load_seen_credit(records, replace, name)?;
        for record in records {
            self.load_record(
                record,
                descriptor,
                replace,
                &mut (&workspace, &mut seen),
                check_cancelled,
                name,
            )?;
        }
        Ok(())
    }

    fn load_record(
        &mut self,
        record: &RecordBatch,
        descriptor: &NativeStateDescriptor,
        replace: bool,
        buffers: &mut (
            &MemoryReservation,
            &mut std::collections::BTreeSet<Arc<[u8]>>,
        ),
        check_cancelled: &mut impl FnMut() -> Result<()>,
        name: &str,
    ) -> Result<()> {
        if record.num_rows() == 0 {
            check_cancelled()?;
        }
        for start in (0..record.num_rows()).step_by(STATE_CHUNK_ROWS) {
            check_cancelled()?;
            let rows = STATE_CHUNK_ROWS.min(record.num_rows() - start);
            let charge = self.load_chunk_charge(record, start, rows, descriptor, name)?;
            ensure_reservation(buffers.0, charge, name)?;
            let chunk = record.slice(start, rows);
            self.load_state_chunk(&chunk, descriptor, replace, buffers.1, name)?;
        }
        Ok(())
    }

    fn load_seen_credit(
        &self,
        records: &[RecordBatch],
        replace: bool,
        name: &str,
    ) -> Result<MemoryReservation> {
        let rows = records.iter().try_fold(0usize, |sum, record| {
            sum.checked_add(record.num_rows())
                .ok_or_else(|| df_error(name, "native delta row count overflowed"))
        })?;
        let seen_credit = self.scratch.new_empty();
        ensure_reservation(
            &seen_credit,
            checked_bytes(4096, [(usize::from(replace) * rows, 128)], name)?,
            name,
        )?;
        Ok(seen_credit)
    }

    fn load_chunk_charge(
        &self,
        record: &RecordBatch,
        start: usize,
        rows: usize,
        descriptor: &NativeStateDescriptor,
        name: &str,
    ) -> Result<usize> {
        let variable = record_variable_bytes(record, start, rows, name)?;
        checked_bytes(
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
        )
    }

    fn load_state_chunk(
        &mut self,
        chunk: &RecordBatch,
        descriptor: &NativeStateDescriptor,
        replace: bool,
        seen: &mut std::collections::BTreeSet<Arc<[u8]>>,
        name: &str,
    ) -> Result<()> {
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
        for row in 0..chunk.num_rows() {
            let key = encoded.as_ref().map(|encoded| encoded.row(row));
            let key = key.as_ref().map_or(&[][..], |key| key.as_ref());
            self.validate_import_key(key, replace, name)?;
            let group = self.import_group(chunk, row, key, descriptor, name)?;
            self.install_imported_group(group, replace, seen, name)?;
        }
        Ok(())
    }

    fn validate_import_key(&self, key: &[u8], replace: bool, name: &str) -> Result<()> {
        if self.index.contains_key(key) {
            if !replace {
                return Err(df_error(name, "native state contains duplicate keys"));
            }
        } else {
            self.check_import_group_budget(self.groups.len() + 1, name)?;
        }
        Ok(())
    }

    fn install_imported_group(
        &mut self,
        group: Group,
        replace: bool,
        seen: &mut std::collections::BTreeSet<Arc<[u8]>>,
        name: &str,
    ) -> Result<()> {
        if replace && !seen.insert(group.key.clone()) {
            return Err(df_error(name, "native state contains duplicate keys"));
        }
        if let Some(&slot) = self.index.get(group.key.as_ref()) {
            self.groups[slot] = group;
        } else {
            let slot = self.groups.len();
            self.index.insert(group.key.clone(), slot);
            self.groups.push(group);
        }
        Ok(())
    }

    fn check_import_group_budget(&self, count: usize, name: &str) -> Result<()> {
        if self
            .compact_budget
            .is_some_and(|b| count as u64 > b.max_rows())
        {
            return Err(df_error(
                name,
                "SQL compact state group budget exceeded on restore",
            ));
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
        let (states, results) = self.import_aggregate_states(record, row, descriptor, name)?;
        Ok(Group {
            key: Arc::from(key),
            values: Arc::from(values),
            aggregates: GroupAggregates::restored(states, results),
            reservation,
        })
    }
    fn import_aggregate_states(
        &self,
        record: &RecordBatch,
        row: usize,
        descriptor: &NativeStateDescriptor,
        name: &str,
    ) -> Result<(Vec<Vec<ScalarValue>>, Vec<ScalarValue>)> {
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
            let result = self.imported_result(aggregate, expression, &state, name)?;
            validate_scalar(&result, &descriptor.result_fields[aggregate], name)?;
            states.push(state);
            results.push(result);
            column += width;
        }
        Ok((states, results))
    }

    fn imported_result(
        &self,
        aggregate: usize,
        expression: &AggregateFunctionExpr,
        state: &[ScalarValue],
        name: &str,
    ) -> Result<ScalarValue> {
        if self.ordered_last {
            super::ordered_last::result(state, expression.ignore_nulls(), name)
        } else if let Some(global) = &self.global_records {
            global.result(aggregate, state, name)
        } else if self.requires_grouped_float_proof() && super::grouped_float::selected(expression)
        {
            super::grouped_sum::result(state, name)
        } else {
            restored_result(expression, state, !self.keys.is_empty(), name)
        }
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
        let rows = validate_record_envelope(record, descriptor, check_cancelled, name)?;
        groups = groups
            .checked_add(rows)
            .ok_or_else(|| df_error(name, "native group count overflowed"))?;
        validate_record_counts(
            record,
            descriptor,
            historical_rows,
            &mut counts,
            check_cancelled,
            name,
        )?;
    }
    validate_all_row_totals(&counts, descriptor, historical_rows, complete, name)?;
    Ok(groups)
}

fn validate_record_envelope(
    record: &RecordBatch,
    descriptor: &NativeStateDescriptor,
    check_cancelled: &mut impl FnMut() -> Result<()>,
    name: &str,
) -> Result<usize> {
    if record.num_rows() == 0 {
        check_cancelled()?;
    }
    validate_record_schema(record, descriptor, name)?;
    Ok(record.num_rows())
}

fn validate_record_schema(
    record: &RecordBatch,
    descriptor: &NativeStateDescriptor,
    name: &str,
) -> Result<()> {
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
    Ok(())
}

fn validate_record_counts(
    record: &RecordBatch,
    descriptor: &NativeStateDescriptor,
    historical_rows: u64,
    totals: &mut [u64],
    check_cancelled: &mut impl FnMut() -> Result<()>,
    name: &str,
) -> Result<()> {
    for start in (0..record.num_rows()).step_by(STATE_CHUNK_ROWS) {
        check_cancelled()?;
        for row in start..(start + STATE_CHUNK_ROWS).min(record.num_rows()) {
            validate_counts(record, row, descriptor, historical_rows, totals, name)?;
        }
    }
    Ok(())
}

fn validate_all_row_totals(
    totals: &[u64],
    descriptor: &NativeStateDescriptor,
    rows: u64,
    complete: bool,
    name: &str,
) -> Result<()> {
    for (count, all_rows) in totals.iter().zip(&descriptor.count_all_rows) {
        if complete && *all_rows && !descriptor.filtered_input && *count != rows {
            return Err(df_error(
                name,
                "native all-row COUNT differs from historical rows",
            ));
        }
    }
    Ok(())
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
                validate_count_state(
                    record.column(column),
                    row,
                    historical_rows,
                    &mut totals[aggregate],
                    descriptor.count_all_rows[aggregate] && !descriptor.key_fields.is_empty(),
                    name,
                )?;
            }
            "avg" => {
                validate_avg_state(
                    record.column(column),
                    record.column(column + 1),
                    row,
                    historical_rows,
                    &mut totals[aggregate],
                    name,
                )?;
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

fn validate_count_state(
    array: &ArrayRef,
    row: usize,
    historical_rows: u64,
    total: &mut u64,
    grouped_all: bool,
    name: &str,
) -> Result<()> {
    let array = array
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| df_error(name, "native COUNT state is not Int64"))?;
    let count = u64::try_from(array.value(row))
        .map_err(|_| df_error(name, "native COUNT state is negative"))?;
    add_count(total, count, historical_rows, name)?;
    if grouped_all && count == 0 {
        return Err(df_error(name, "native grouped all-row COUNT is zero"));
    }
    Ok(())
}

fn validate_avg_state(
    counts: &ArrayRef,
    sum: &ArrayRef,
    row: usize,
    historical_rows: u64,
    total: &mut u64,
    name: &str,
) -> Result<()> {
    let counts = counts
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| df_error(name, "native AVG count is not UInt64"))?;
    let count = if counts.is_null(row) {
        0
    } else {
        counts.value(row)
    };
    add_count(total, count, historical_rows, name)?;
    let null_sum = sum.is_null(row);
    if count > historical_rows || (count == 0) != null_sum {
        return Err(df_error(name, "native AVG count and sum are inconsistent"));
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
            let width = variable_value_bytes(array, row, name)?;
            bytes = checked_bytes(bytes, [(width, 1)], name)?;
        }
    }
    Ok(bytes)
}

fn variable_value_bytes(array: &ArrayRef, row: usize, name: &str) -> Result<usize> {
    Ok(match array.data_type() {
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
    })
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

fn add_group_counts(
    group: &Group,
    descriptor: &NativeStateDescriptor,
    rows: u64,
    totals: &mut [u64],
    name: &str,
) -> Result<()> {
    for (index, function) in descriptor.aggregate_names.iter().enumerate() {
        let state = group.aggregates.state(index);
        let count = match (function.as_str(), state.first()) {
            ("count", Some(ScalarValue::Int64(Some(count)))) => u64::try_from(*count)
                .map_err(|_| df_error(name, "native COUNT state is negative"))?,
            ("avg", Some(ScalarValue::UInt64(count))) => count.unwrap_or(0),
            _ => continue,
        };
        add_count(&mut totals[index], count, rows, name)?;
    }
    Ok(())
}
