use std::sync::Arc;

use datafusion::{
    arrow::{
        datatypes::{Field, FieldRef, Schema, SchemaRef},
        record_batch::RecordBatch,
    },
    common::ScalarValue,
    execution::memory_pool::MemoryReservation,
};
use serde_json::{Value, json};

use super::super::{
    SqlOperator, encode_sql_state, incremental,
    incremental::compact_state::{NativeAggregateInput, NativeStateDescriptor},
    ipc, retention, sql_state_error,
};
use super::control::CompactIdentity;
use crate::{Batch, BatchMetadata, DataFusionRuntime, Result};

pub(super) struct PaidIdentity {
    pub value: CompactIdentity,
    pub _reservation: MemoryReservation,
}

pub(super) fn build(
    operator: &SqlOperator,
    logical: &SchemaRef,
    physical: &SchemaRef,
    ordinals: Vec<usize>,
    descriptor: &NativeStateDescriptor,
) -> Result<PaidIdentity> {
    let runtime = operator.retention_runtime()?;
    validate_ports(operator, logical, descriptor)?;
    let reservation = reserve(runtime, logical, physical, descriptor, &operator.name)?;
    let value = identity_value(operator, runtime, logical, physical, ordinals, descriptor)?;
    Ok(PaidIdentity {
        value,
        _reservation: reservation,
    })
}

fn validate_ports(
    operator: &SqlOperator,
    logical: &SchemaRef,
    descriptor: &NativeStateDescriptor,
) -> Result<()> {
    if operator.input_ports[0]
        .schema()
        .is_some_and(|schema| schema != logical)
        || operator.output_ports[0]
            .schema()
            .is_some_and(|schema| schema != &descriptor.output_schema)
    {
        return Err(sql_state_error(
            "SQL compact identity differs from declared ports",
        ));
    }
    Ok(())
}

fn identity_value(
    operator: &SqlOperator,
    runtime: &DataFusionRuntime,
    logical: &SchemaRef,
    physical: &SchemaRef,
    ordinals: Vec<usize>,
    descriptor: &NativeStateDescriptor,
) -> Result<CompactIdentity> {
    Ok(CompactIdentity {
        query_sha256: operator.query_digest(),
        input_alias: operator.aliases[0].clone(),
        runtime_config: runtime.compact_runtime_config(),
        logical_schema_sha256: retention::schema_digest(logical)?,
        physical_schema_sha256: retention::schema_digest(physical)?,
        retained_ordinals: ordinals,
        state_schema_sha256: retention::schema_digest(&descriptor.wire_schema)?,
        output_schema_sha256: retention::schema_digest(&descriptor.output_schema)?,
        native_descriptor: native_json(descriptor)?,
    })
}

fn reserve(
    runtime: &DataFusionRuntime,
    logical: &SchemaRef,
    physical: &SchemaRef,
    descriptor: &NativeStateDescriptor,
    name: &str,
) -> Result<MemoryReservation> {
    let bound = [
        logical,
        physical,
        &descriptor.wire_schema,
        &descriptor.output_schema,
    ]
    .into_iter()
    .try_fold(131_072, |bytes, schema| {
        incremental::checked_bytes(bytes, [(ipc::schema_bytes(schema)?, 64)], name)
    })?;
    let bound =
        incremental::checked_bytes(bound, [(descriptor.expression_identity_bytes, 1)], name)?;
    let reservation = runtime.incremental_reservation(name);
    incremental::ensure_reservation(&reservation, bound, name)?;
    Ok(reservation)
}

fn fields_json(fields: &[FieldRef]) -> Result<Value> {
    fields
        .iter()
        .map(|field| {
            Ok(
                json!({"name":field.name(), "schema_sha256":retention::schema_digest(
            &Arc::new(Schema::new(vec![field.clone()]))
        )?}),
            )
        })
        .collect::<Result<Vec<_>>>()
        .map(Value::Array)
}

fn input_json(input: &NativeAggregateInput) -> Result<Value> {
    match input {
        NativeAggregateInput::Column { index, field } => Ok(json!({
            "kind":"column", "index":index, "field":fields_json(std::slice::from_ref(field))?
        })),
        NativeAggregateInput::Literal(value) => literal_json(value),
        NativeAggregateInput::Cast { input, field, safe } => Ok(json!({
            "kind":"cast", "input":input_json(input)?,
            "field":fields_json(std::slice::from_ref(field))?, "safe":safe,
            "format_policy":"datafusion_default"
        })),
        NativeAggregateInput::TryCast { input, dtype } => Ok(json!({
            "kind":"try_cast", "input":input_json(input)?,
            "field":fields_json(&[Arc::new(Field::new("try_cast", dtype.clone(), true))])?,
            "format_policy":"datafusion_default"
        })),
        NativeAggregateInput::Binary {
            left,
            op,
            right,
            fail_on_overflow,
        } => binary_json(left, *op, right, *fail_on_overflow),
        NativeAggregateInput::Unary { input, op } => Ok(json!({
            "kind":"unary", "input":input_json(input)?, "operator":op
        })),
        NativeAggregateInput::Case {
            operand,
            branches,
            fallback,
        } => case_json(operand.as_deref(), branches, fallback.as_deref()),
    }
}

fn literal_json(value: &ScalarValue) -> Result<Value> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "literal",
        value.data_type(),
        value.is_null(),
    )]));
    let array = value
        .to_array()
        .map_err(|error| sql_state_error(&error.to_string()))?;
    let record = RecordBatch::try_new(schema.clone(), vec![array])
        .map_err(|error| sql_state_error(&error.to_string()))?;
    let batch = Batch::table(vec![record], BatchMetadata::default())?;
    let segment = super::super::StateSegment::new(encode_sql_state(&batch)?);
    Ok(
        json!({"kind":"literal", "schema_sha256":retention::schema_digest(&schema)?,
        "ipc_sha256":segment.sha256()}),
    )
}

fn binary_json(
    left: &NativeAggregateInput,
    op: datafusion::logical_expr::Operator,
    right: &NativeAggregateInput,
    fail_on_overflow: bool,
) -> Result<Value> {
    Ok(
        json!({"kind":"binary", "left":input_json(left)?, "operator":op.to_string(),
        "right":input_json(right)?, "fail_on_overflow":fail_on_overflow}),
    )
}

fn case_json(
    operand: Option<&NativeAggregateInput>,
    branches: &[(NativeAggregateInput, NativeAggregateInput)],
    fallback: Option<&NativeAggregateInput>,
) -> Result<Value> {
    let branches = branches
        .iter()
        .map(|(when, then)| Ok(json!({"when":input_json(when)?,"then":input_json(then)?})))
        .collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"kind":"case","operand":operand.map(input_json).transpose()?,
        "branches":branches,"fallback":fallback.map(input_json).transpose()?}),
    )
}

fn native_json(descriptor: &NativeStateDescriptor) -> Result<Value> {
    let aggregates = descriptor
        .aggregate_names
        .iter()
        .enumerate()
        .map(|(slot, function)| aggregate_json(descriptor, slot, function))
        .collect::<Result<Vec<_>>>()?;
    let keys = fields_json(&descriptor.key_fields)?;
    let key_inputs = descriptor
        .key_inputs
        .iter()
        .map(input_json)
        .collect::<Result<Vec<_>>>()?;
    let input_checks = descriptor
        .input_checks
        .iter()
        .map(input_json)
        .collect::<Result<Vec<_>>>()?;
    native_output_json(descriptor, &aggregates, &keys, &key_inputs, &input_checks)
}

fn aggregate_json(
    descriptor: &NativeStateDescriptor,
    slot: usize,
    function: &str,
) -> Result<Value> {
    let inputs = descriptor.aggregate_inputs[slot]
        .iter()
        .map(input_json)
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({"function":function, "inputs":inputs,
        "filter":descriptor.aggregate_filters[slot].as_ref().map(input_json).transpose()?,
        "all_rows":descriptor.count_all_rows[slot],
        "state_fields":fields_json(&descriptor.state_fields[slot])?,
        "result_field":fields_json(std::slice::from_ref(&descriptor.result_fields[slot]))?}))
}

fn native_output_json(
    descriptor: &NativeStateDescriptor,
    aggregates: &[Value],
    keys: &Value,
    key_inputs: &[Value],
    input_checks: &[Value],
) -> Result<Value> {
    Ok(json!({"policy":descriptor.policy, "keys":keys,
        "key_inputs":key_inputs, "input_checks":input_checks, "aggregates":aggregates,
        "projection":descriptor.projection.iter().map(input_json).collect::<Result<Vec<_>>>()?,
        "post_filter":descriptor.post_filter.as_ref().map(input_json).transpose()?,
        "post_order":descriptor.post_order.as_ref().map(|order| order_json(&order.keys, order.skip, order.fetch)).transpose()?,
        "wire_schema_sha256":retention::schema_digest(&descriptor.wire_schema)?,
        "output_schema_sha256":retention::schema_digest(&descriptor.output_schema)?}))
}

fn order_json(
    keys: &[(NativeAggregateInput, bool, bool)],
    skip: usize,
    fetch: Option<usize>,
) -> Result<Value> {
    let keys = keys.iter().map(|(input, descending, nulls_first)| {
        Ok(json!({"input":input_json(input)?,"descending":descending,"nulls_first":nulls_first}))
    }).collect::<Result<Vec<_>>>()?;
    Ok(json!({"keys":keys,"skip":skip,"fetch":fetch}))
}
