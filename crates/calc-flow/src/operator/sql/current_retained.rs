use std::{collections::BTreeMap, sync::Arc};

use datafusion::{arrow::datatypes::SchemaRef, execution::memory_pool::MemoryReservation};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{
    RetainedSqlInput, SqlOperator, StateSegment, decode_sql_state, incremental, ipc, metadata,
    record_copy_reservation, retention, sql_state_error,
};
use crate::{Batch, DataFusionConfig, DataFusionRuntime, JsonMap, OperatorStateSnapshot, Result};

#[derive(PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RetainedIdentity {
    query_sha256: String,
    input_alias: String,
    udfs: Vec<Value>,
    runtime_config: DataFusionConfig,
    logical_schema_sha256: String,
    physical_schema_sha256: String,
    retained_ordinals: Vec<usize>,
    output_schema_sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RetainedControl {
    state_layout: u32,
    state_accounting: u32,
    datafusion_version: String,
    state_policy: String,
    identity: RetainedIdentity,
    rows: u64,
    bytes: u64,
    logical_schema_sha256: String,
    input_sha256: String,
    batch_metadata_sha256: String,
}

pub(super) struct RetainedCapture {
    pub snapshot: OperatorStateSnapshot,
    _reservation: Arc<MemoryReservation>,
    _metadata: Arc<metadata::SqlMetadata>,
}

struct RetainedBinding {
    logical_segment: StateSegment,
    physical: SchemaRef,
    projection: Option<Arc<retention::SqlProjection>>,
    identity: RetainedIdentity,
    _reservation: Arc<MemoryReservation>,
}

impl RetainedControl {
    fn inline(&self, segment: &StateSegment) -> JsonMap {
        JsonMap::from([
            ("state_layout".into(), json!(4)),
            ("state_accounting".into(), json!(4)),
            ("query_sha256".into(), json!(self.identity.query_sha256)),
            ("rows".into(), json!(self.rows)),
            ("bytes".into(), json!(self.bytes)),
            ("control_sha256".into(), json!(segment.sha256())),
        ])
    }

    fn validate(&self, snapshot: &OperatorStateSnapshot, binding: &RetainedBinding) -> Result<()> {
        self.validate_identity(&binding.identity)?;
        self.validate_digests(snapshot)?;
        if snapshot.inline_metadata != self.inline(&snapshot.segments["control"]) {
            return Err(sql_state_error("SQL retained inline mirrors are invalid"));
        }
        Ok(())
    }

    fn validate_identity(&self, identity: &RetainedIdentity) -> Result<()> {
        if self.state_layout != 4
            || self.state_accounting != 4
            || self.datafusion_version != "54.0.0"
            || self.state_policy != "retained_v1"
            || self.identity != *identity
        {
            return Err(sql_state_error("SQL retained control identity is invalid"));
        }
        Ok(())
    }

    fn validate_digests(&self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        let segments = &snapshot.segments;
        if self.logical_schema_sha256 != segments["logical-schema"].sha256()
            || self.identity.logical_schema_sha256 != self.logical_schema_sha256
            || self.input_sha256 != segments["input-retained"].sha256()
            || self.batch_metadata_sha256 != segments["batch-metadata"].sha256()
        {
            return Err(sql_state_error("SQL retained segment census is invalid"));
        }
        Ok(())
    }
}

pub(super) fn capture(
    operator: &SqlOperator,
    state: &RetainedSqlInput,
    check: &dyn Fn() -> Result<()>,
) -> Result<Arc<RetainedCapture>> {
    check()?;
    let runtime = operator.retention_runtime()?;
    let binding = capture_binding(operator, state, check)?;
    let input = capture_input(operator, runtime, state, check)?;
    let encoded_metadata = capture_metadata(operator, runtime, state, check)?;
    let reservation = capture_reservation(operator, &binding.logical_segment, &input.segment)?;
    let control = RetainedControl {
        state_layout: 4,
        state_accounting: 4,
        datafusion_version: "54.0.0".into(),
        state_policy: "retained_v1".into(),
        rows: state.rows,
        bytes: state.bytes,
        logical_schema_sha256: binding.logical_segment.sha256().into(),
        input_sha256: input.segment.sha256().into(),
        batch_metadata_sha256: encoded_metadata.segment.sha256().into(),
        identity: binding.identity,
    };
    let encoded_control = encode_control(operator, &control, check)?;
    let snapshot = OperatorStateSnapshot {
        inline_metadata: control.inline(&encoded_control),
        segments: BTreeMap::from([
            ("control".into(), encoded_control),
            ("input-retained".into(), input.segment.clone()),
            ("logical-schema".into(), binding.logical_segment),
            ("batch-metadata".into(), encoded_metadata.segment.clone()),
        ]),
    };
    owned_capture(snapshot, reservation, encoded_metadata, check)
}

fn capture_binding(
    operator: &SqlOperator,
    state: &RetainedSqlInput,
    check: &dyn Fn() -> Result<()>,
) -> Result<RetainedBinding> {
    let logical = state.projection.as_ref().map_or_else(
        || state.records[0].schema(),
        |projection| projection.columns.logical_schema().clone(),
    );
    let binding = binding(operator, &logical, check)?;
    if binding.physical != state.records[0].schema()
        || binding.projection.is_some() != state.projection.is_some()
    {
        return Err(sql_state_error(
            "SQL retained input differs from current dependencies",
        ));
    }
    Ok(binding)
}

fn capture_input(
    operator: &SqlOperator,
    runtime: &DataFusionRuntime,
    state: &RetainedSqlInput,
    check: &dyn Fn() -> Result<()>,
) -> Result<Arc<ipc::SqlInputSegment>> {
    if let Some(input) = &state.segment {
        Ok(input.clone())
    } else {
        let materialized = state.materialize(runtime, &operator.name)?;
        ipc::encode(
            &materialized.batch,
            runtime.incremental_reservation(&operator.name),
            check,
        )
    }
}

fn capture_metadata(
    operator: &SqlOperator,
    runtime: &DataFusionRuntime,
    state: &RetainedSqlInput,
    check: &dyn Fn() -> Result<()>,
) -> Result<Arc<metadata::SqlMetadata>> {
    if let Some(metadata) = &state.metadata_segment {
        Ok(metadata.clone())
    } else {
        metadata::encode(
            &state.metadata,
            metadata::reserve(runtime, &state.metadata, &operator.name)?,
            check,
        )
    }
}

fn owned_capture(
    mut snapshot: OperatorStateSnapshot,
    reservation: Arc<MemoryReservation>,
    metadata: Arc<metadata::SqlMetadata>,
    check: &dyn Fn() -> Result<()>,
) -> Result<Arc<RetainedCapture>> {
    for segment in snapshot.segments.values_mut() {
        *segment = segment.clone().with_owner(reservation.clone());
    }
    check()?;
    Ok(Arc::new(RetainedCapture {
        snapshot,
        _reservation: reservation,
        _metadata: metadata,
    }))
}

fn binding(
    operator: &SqlOperator,
    logical: &SchemaRef,
    check: &dyn Fn() -> Result<()>,
) -> Result<RetainedBinding> {
    check()?;
    validate_logical_binding(operator, logical)?;
    let reservation = descriptor_reservation(operator, logical)?;
    let runtime = operator.retention_runtime()?;
    let plan = runtime.retained_sql_plan_sync(
        &operator.validated,
        &operator.aliases[0],
        logical.clone(),
        &operator.name,
        runtime.incremental_reservation(&operator.name),
    )?;
    check()?;
    build_binding(operator, runtime, logical, &plan, reservation, check)
}

fn validate_logical_binding(operator: &SqlOperator, logical: &SchemaRef) -> Result<()> {
    if !operator.stream_aggregate
        || operator.aliases.len() != 1
        || operator.input_ports[0]
            .schema()
            .is_some_and(|declared| declared != logical)
    {
        return Err(sql_state_error("SQL retained logical binding is invalid"));
    }
    Ok(())
}

fn build_binding(
    operator: &SqlOperator,
    runtime: &DataFusionRuntime,
    logical: &SchemaRef,
    plan: &crate::datafusion::compact::PaidSqlPlan,
    reservation: Arc<MemoryReservation>,
    check: &dyn Fn() -> Result<()>,
) -> Result<RetainedBinding> {
    let projection = trusted_projection(operator, logical, plan)?;
    let (physical, output) = binding_schemas(operator, logical, projection.as_deref(), plan)?;
    let logical_segment = retention::encode_schema(logical)?.with_owner(reservation.clone());
    let identity = retained_identity(
        operator,
        runtime,
        logical,
        &physical,
        &output,
        projection.as_deref(),
        &logical_segment,
    )?;
    check()?;
    Ok(RetainedBinding {
        logical_segment,
        physical,
        projection,
        identity,
        _reservation: reservation,
    })
}

fn binding_schemas(
    operator: &SqlOperator,
    logical: &SchemaRef,
    projection: Option<&retention::SqlProjection>,
    plan: &crate::datafusion::compact::PaidSqlPlan,
) -> Result<(SchemaRef, SchemaRef)> {
    let physical = projection.map_or_else(
        || logical.clone(),
        |projection| projection.columns.physical_schema().clone(),
    );
    let output = Arc::new(plan.analyzed.schema().as_arrow().clone());
    if operator.output_ports[0]
        .schema()
        .is_some_and(|declared| declared != &output)
    {
        return Err(sql_state_error(
            "SQL retained output differs from its declared port",
        ));
    }
    Ok((physical, output))
}

fn retained_identity(
    operator: &SqlOperator,
    runtime: &DataFusionRuntime,
    logical: &SchemaRef,
    physical: &SchemaRef,
    output: &SchemaRef,
    projection: Option<&retention::SqlProjection>,
    logical_segment: &StateSegment,
) -> Result<RetainedIdentity> {
    Ok(RetainedIdentity {
        query_sha256: operator.query_digest(),
        input_alias: operator.aliases[0].clone(),
        udfs: operator.udfs.iter().map(super::udf_configuration).collect(),
        runtime_config: runtime.compact_runtime_config(),
        logical_schema_sha256: logical_segment.sha256().into(),
        physical_schema_sha256: retention::schema_digest(physical)?,
        retained_ordinals: projection.map_or_else(
            || (0..logical.fields().len()).collect(),
            |projection| projection.columns.ordinals().to_vec(),
        ),
        output_schema_sha256: retention::schema_digest(output)?,
    })
}

fn trusted_projection(
    operator: &SqlOperator,
    logical: &SchemaRef,
    plan: &crate::datafusion::compact::PaidSqlPlan,
) -> Result<Option<Arc<retention::SqlProjection>>> {
    if !operator.udfs.is_empty() {
        return Ok(None);
    }
    let projection = retention::SqlProjection::resolve(
        operator.retention_runtime()?,
        &operator.validated,
        &operator.aliases[0],
        logical.clone(),
        &operator.name,
    )?;
    projection
        .map(|projection| {
            Ok(
                (retention::plan_dependencies_fit(&plan.raw, &projection.columns)?
                    && retention::plan_dependencies_fit(&plan.analyzed, &projection.columns)?)
                .then_some(projection),
            )
        })
        .transpose()
        .map(Option::flatten)
}

fn descriptor_reservation(
    operator: &SqlOperator,
    logical: &SchemaRef,
) -> Result<Arc<MemoryReservation>> {
    let functions = operator.udfs.iter().try_fold(0usize, |bytes, udf| {
        incremental::checked_bytes(
            bytes,
            [
                (udf.provider().len(), 64),
                (udf.name().len(), 64),
                (udf.version().len(), 64),
            ],
            &operator.name,
        )
    })?;
    let bytes = incremental::checked_bytes(
        16_384,
        [
            (ipc::schema_bytes(logical)?, 64),
            (operator.query.len(), 64),
            (operator.aliases[0].len(), 64),
            (functions, 2),
        ],
        &operator.name,
    )?;
    let reservation = operator
        .retention_runtime()?
        .incremental_reservation(&operator.name);
    incremental::ensure_reservation(&reservation, bytes, &operator.name)?;
    Ok(Arc::new(reservation))
}

fn capture_reservation(
    operator: &SqlOperator,
    logical: &StateSegment,
    input: &StateSegment,
) -> Result<Arc<MemoryReservation>> {
    let reservation = operator
        .retention_runtime()?
        .incremental_reservation(&operator.name);
    let bytes = incremental::checked_bytes(
        16_384,
        [(logical.bytes().len(), 2), (input.bytes().len(), 2)],
        &operator.name,
    )?;
    incremental::ensure_reservation(&reservation, bytes, &operator.name)?;
    Ok(Arc::new(reservation))
}

fn encode_control(
    operator: &SqlOperator,
    control: &RetainedControl,
    check: &dyn Fn() -> Result<()>,
) -> Result<StateSegment> {
    check()?;
    let reservation = control_reservation(operator, control)?;
    let bytes = serde_json::to_vec(control).map_err(|error| sql_state_error(&error.to_string()))?;
    if bytes
        .capacity()
        .checked_add(256)
        .is_none_or(|size| size > reservation.size())
    {
        return Err(sql_state_error(
            "SQL retained control exceeded its prepaid bound",
        ));
    }
    check()?;
    Ok(StateSegment::new(bytes).with_owner(reservation))
}

fn control_reservation(
    operator: &SqlOperator,
    control: &RetainedControl,
) -> Result<Arc<MemoryReservation>> {
    let reservation = operator
        .retention_runtime()?
        .incremental_reservation(&operator.name);
    let bytes = incremental::checked_bytes(
        16_384,
        [
            (control.identity.input_alias.len(), 64),
            (control.identity.retained_ordinals.len(), 64),
            (operator.query.len(), 64),
        ],
        &operator.name,
    )?;
    let bytes = control
        .identity
        .udfs
        .iter()
        .try_fold(bytes, |bytes, value| {
            incremental::checked_bytes(bytes, [(value_bytes(value, 1)?, 4)], &operator.name)
        })?;
    incremental::ensure_reservation(&reservation, bytes, &operator.name)?;
    Ok(Arc::new(reservation))
}

fn value_bytes(value: &Value, depth: usize) -> Result<usize> {
    if depth > crate::json::MAX_JSON_DEPTH {
        return Err(sql_state_error("SQL retained control exceeds JSON depth"));
    }
    match value {
        Value::String(value) => value
            .len()
            .checked_add(64)
            .ok_or_else(|| sql_state_error("SQL retained control charge overflowed")),
        Value::Array(values) => values.iter().try_fold(64usize, |bytes, value| {
            bytes
                .checked_add(value_bytes(value, depth + 1)?)
                .ok_or_else(|| sql_state_error("SQL retained control charge overflowed"))
        }),
        Value::Object(values) => values.iter().try_fold(64usize, |bytes, (key, value)| {
            bytes
                .checked_add(key.len())
                .and_then(|bytes| bytes.checked_add(128))
                .and_then(|bytes| bytes.checked_add(value_bytes(value, depth + 1).ok()?))
                .ok_or_else(|| sql_state_error("SQL retained control charge overflowed"))
        }),
        _ => Ok(64),
    }
}

pub(super) fn prepare_restore(
    operator: &SqlOperator,
    snapshot: &OperatorStateSnapshot,
    check: &dyn Fn() -> Result<()>,
) -> Result<(RetainedSqlInput, Arc<RetainedCapture>)> {
    check()?;
    validate_retained_inventory(snapshot)?;
    let runtime = operator.retention_runtime()?;
    let RestoreDescriptor {
        control,
        binding,
        control_reservation,
        schema_reservation,
    } = restore_descriptor(operator, snapshot, check)?;
    let (decoded, backing) = restored_input(operator, snapshot, &control, &binding, check)?;
    let records = restored_records(
        operator,
        runtime,
        snapshot,
        binding.projection,
        &control,
        &decoded,
        backing,
    )?;
    finish_retained_restore(
        operator,
        snapshot,
        records,
        (control_reservation, schema_reservation),
        check,
    )
}

struct RestoreDescriptor {
    control: RetainedControl,
    binding: RetainedBinding,
    control_reservation: MemoryReservation,
    schema_reservation: MemoryReservation,
}

struct RestoredRecords {
    retained: RetainedSqlInput,
    metadata: Arc<metadata::SqlMetadata>,
    copies: MemoryReservation,
}

fn validate_retained_inventory(snapshot: &OperatorStateSnapshot) -> Result<()> {
    if snapshot.segments.len() != 4
        || [
            "control",
            "input-retained",
            "logical-schema",
            "batch-metadata",
        ]
        .into_iter()
        .any(|name| !snapshot.segments.contains_key(name))
    {
        return Err(sql_state_error("SQL retained segment inventory is invalid"));
    }
    Ok(())
}

fn restore_descriptor(
    operator: &SqlOperator,
    snapshot: &OperatorStateSnapshot,
    check: &dyn Fn() -> Result<()>,
) -> Result<RestoreDescriptor> {
    let control_reservation = decode_reservation(operator, &snapshot.segments["control"], 64)?;
    let value = crate::json::parse_json_value(
        snapshot.segments["control"].bytes(),
        "SQL retained control",
    )?;
    let control: RetainedControl =
        serde_json::from_value(value).map_err(|error| sql_state_error(&error.to_string()))?;
    let schema_reservation =
        decode_reservation(operator, &snapshot.segments["logical-schema"], 32)?;
    let logical = retention::decode_schema(&snapshot.segments["logical-schema"])?;
    let binding = binding(operator, &logical, check)?;
    control.validate(snapshot, &binding)?;
    Ok(RestoreDescriptor {
        control,
        binding,
        control_reservation,
        schema_reservation,
    })
}

fn restored_input(
    operator: &SqlOperator,
    snapshot: &OperatorStateSnapshot,
    control: &RetainedControl,
    binding: &RetainedBinding,
    check: &dyn Fn() -> Result<()>,
) -> Result<(Batch, Arc<MemoryReservation>)> {
    check()?;
    let backing = Arc::new(decode_reservation(
        operator,
        &snapshot.segments["input-retained"],
        4,
    )?);
    let decoded = decode_sql_state(snapshot.segments["input-retained"].bytes())?;
    check()?;
    if decoded.table_payload()?.schema() != &binding.physical {
        return Err(sql_state_error("SQL retained physical schema is invalid"));
    }
    operator.validate_checkpoint_charge(&decoded, control.rows, control.bytes)?;
    Ok((decoded, backing))
}

fn restored_records(
    operator: &SqlOperator,
    runtime: &DataFusionRuntime,
    snapshot: &OperatorStateSnapshot,
    projection: Option<Arc<retention::SqlProjection>>,
    control: &RetainedControl,
    decoded: &Batch,
    backing: Arc<MemoryReservation>,
) -> Result<RestoredRecords> {
    let (latest, metadata) = metadata::decode(
        runtime,
        &snapshot.segments["batch-metadata"],
        &operator.name,
    )?;
    let table = decoded.table_payload()?;
    let copies = record_copy_reservation(
        runtime,
        &operator.name,
        table.batches().len(),
        table.schema().fields().len(),
        1,
    )?;
    let mut retained = RetainedSqlInput {
        records: Vec::new(),
        metadata: latest,
        projection,
        projection_checked: true,
        backing_reservations: vec![backing.clone()],
        reservation: None,
        segment: Some(ipc::SqlInputSegment::restored(
            snapshot.segments["input-retained"].clone(),
            backing,
        )),
        metadata_segment: Some(metadata.clone()),
        rows: control.rows,
        bytes: control.bytes,
    };
    retained.reserve_append(
        table.batches().len(),
        runtime,
        &operator.name,
        table.schema().fields().len(),
    )?;
    retained.records.extend(table.batches().iter().cloned());
    Ok(RestoredRecords {
        retained,
        metadata,
        copies,
    })
}

fn finish_retained_restore(
    operator: &SqlOperator,
    snapshot: &OperatorStateSnapshot,
    records: RestoredRecords,
    credits: (MemoryReservation, MemoryReservation),
    check: &dyn Fn() -> Result<()>,
) -> Result<(RetainedSqlInput, Arc<RetainedCapture>)> {
    let RestoredRecords {
        retained,
        metadata,
        copies,
    } = records;
    let (control_reservation, schema_reservation) = credits;
    let reservation = capture_reservation(
        operator,
        &snapshot.segments["logical-schema"],
        &snapshot.segments["input-retained"],
    )?;
    let capture = owned_capture(snapshot.clone(), reservation, metadata, check)?;
    drop((copies, control_reservation, schema_reservation));
    check()?;
    Ok((retained, capture))
}

fn decode_reservation(
    operator: &SqlOperator,
    segment: &StateSegment,
    factor: usize,
) -> Result<MemoryReservation> {
    let reservation = operator
        .retention_runtime()?
        .incremental_reservation(&operator.name);
    let bytes =
        incremental::checked_bytes(16_384, [(segment.bytes().len(), factor)], &operator.name)?;
    incremental::ensure_reservation(&reservation, bytes, &operator.name)?;
    Ok(reservation)
}
