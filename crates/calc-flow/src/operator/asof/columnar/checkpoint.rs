use super::{
    ColumnarState, run::Run, run::SequenceColumn, run::identity_schema, state::IndexCapacities,
};
use crate::operator::asof::{AsofJoinSide, StreamAsofJoinSpec, admission, codec};
use crate::operator::keyed::{GroupId, MORSEL_ROWS};
use crate::{Result, StateSegment};
use datafusion::arrow::{
    array::{Array, BooleanArray, Int64Array, UInt32Array},
    datatypes::{Schema, SchemaRef},
    record_batch::RecordBatch,
};
use datafusion::execution::memory_pool::MemoryPool;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

const CONTROL_ID: &str = "asof-columnar-v1-control";
const KEYS_ID: &str = "asof-columnar-v1-keys";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Control {
    version: u32,
    next_run: u64,
    keys: bool,
    runs: Vec<RunDescriptor>,
    capacities: IndexCapacities,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RunDescriptor {
    side: u8,
    run: u64,
    rows: u64,
    large_sequence: bool,
    payload: bool,
    payload_charge: u64,
    identity_charge: u64,
}

impl ColumnarState {
    pub(in crate::operator::asof) fn capture(
        &self,
        limit: usize,
    ) -> Result<BTreeMap<String, StateSegment>> {
        if self.next_run == 0
            && self.keys.is_empty()
            && self.left_runs.is_empty()
            && self.right_runs.is_empty()
        {
            return Ok(BTreeMap::new());
        }
        let mut segments = BTreeMap::new();
        let mut scratch = Vec::new();
        if !self.keys.is_empty() {
            let record = self.key_record()?;
            let bytes = codec::encode_batch(&record, limit, &mut scratch)?;
            segments.insert(KEYS_ID.into(), StateSegment::new(bytes));
        }
        let mut descriptors = Vec::new();
        for (side, runs) in [(0, &self.left_runs), (1, &self.right_runs)] {
            for (&run_id, run) in runs {
                let large_sequence = matches!(run.sequences(), SequenceColumn::LargeBinary(_));
                let identity = identity_record(run, large_sequence)?;
                let identity_id = identity_id(side, run_id);
                let bytes = codec::encode_batch(&identity, limit, &mut scratch)?;
                segments.insert(identity_id, StateSegment::new(bytes));
                let (payload_charge, identity_charge) = run.charges();
                if let Some(payload) = run.payload_record() {
                    let bytes = codec::encode_batch(payload, limit, &mut scratch)?;
                    segments.insert(payload_id(side, run_id), StateSegment::new(bytes));
                }
                descriptors.push(RunDescriptor {
                    side: u8::from(side == 1),
                    run: run_id,
                    rows: run.rows() as u64,
                    large_sequence,
                    payload: run.payload_record().is_some(),
                    payload_charge: if run.payload_record().is_some() {
                        payload_charge as u64
                    } else {
                        0
                    },
                    identity_charge: identity_charge as u64,
                });
            }
        }
        let control = Control {
            version: 1,
            next_run: self.next_run,
            keys: !self.keys.is_empty(),
            runs: descriptors,
            capacities: self.index_capacities(),
        };
        let mut bytes = Vec::new();
        serde_json::to_writer(codec::BoundedWriter::growable(&mut bytes, limit), &control)
            .map_err(|error| {
                super::super::reason(
                    &self.name,
                    crate::StreamingFailureReason::AsofWorkspaceLimitExceeded,
                    &error.to_string(),
                )
            })?;
        segments.insert(CONTROL_ID.into(), StateSegment::new(bytes));
        Ok(segments)
    }

    pub(in crate::operator::asof) fn restore(
        name: &str,
        spec: &StreamAsofJoinSpec,
        schemas: &[SchemaRef; 2],
        pool: &Arc<dyn MemoryPool>,
        segments: &BTreeMap<String, StateSegment>,
    ) -> Result<Self> {
        let mut state = Self::new(name, spec, schemas.clone(), pool)
            .map_err(|error| mismatch(&error.to_string()))?;
        if segments.is_empty() {
            return Ok(state);
        }
        let control = segments
            .get(CONTROL_ID)
            .ok_or_else(|| mismatch("missing columnar control segment"))?;
        verify_checksum(control)?;
        let control: Control = serde_json::from_slice(control.bytes())
            .map_err(|error| mismatch(&error.to_string()))?;
        if control.version != 1 || control.keys != segments.contains_key(KEYS_ID) {
            return Err(mismatch(
                "invalid columnar control version or key descriptor",
            ));
        }
        let mut expected = BTreeSet::from([CONTROL_ID.to_owned()]);
        if control.keys {
            expected.insert(KEYS_ID.into());
            let schema = key_schema(spec, schemas)?;
            let record = decode(segments.get(KEYS_ID), &schema, u64::from(u32::MAX))?;
            if record.num_rows() == 0 {
                return Err(mismatch("columnar key segment is empty"));
            }
            state
                .restore_keys(&record)
                .map_err(|error| mismatch(&error.to_string()))?;
        }
        let mut maximum_run = None;
        for descriptor in &control.runs {
            if descriptor.side > 1 || descriptor.rows == 0 || descriptor.rows > u64::from(u32::MAX)
            {
                return Err(mismatch("invalid columnar run descriptor"));
            }
            let side = usize::from(descriptor.side);
            let run_id = descriptor.run;
            maximum_run = Some(maximum_run.map_or(run_id, |current: u64| current.max(run_id)));
            let identity_name = identity_id(side, run_id);
            let payload_name = payload_id(side, run_id);
            expected.insert(identity_name.clone());
            let identity_schema = identity_schema(descriptor.large_sequence);
            let identity = decode(
                segments.get(&identity_name),
                &identity_schema,
                spec.limits().max_state_rows(),
            )?;
            let payload = if descriptor.payload {
                expected.insert(payload_name.clone());
                Some(decode(
                    segments.get(&payload_name),
                    &schemas[side],
                    spec.limits().max_state_rows(),
                )?)
            } else {
                if segments.contains_key(&payload_name) {
                    return Err(mismatch("undescribed columnar payload segment"));
                }
                None
            };
            let run = decode_run(&identity, payload, descriptor)?;
            if run
                .groups()
                .iter()
                .any(|group| group.index() >= state.keys.len())
            {
                return Err(mismatch("run group exceeds the restored shared key domain"));
            }
            validate_payload(&mut state, side, &run, run_id, spec)?;
            state
                .restore_run(side, run_id, run)
                .map_err(|error| mismatch(&error.to_string()))?;
        }
        if control.next_run == u64::MAX {
            return Err(mismatch("columnar run identifier space is exhausted"));
        }
        if !control.keys && control.capacities.pending != 0 {
            return Err(mismatch("keyless columnar control owns pending capacity"));
        }
        if maximum_run.is_some_and(|run| control.next_run <= run) {
            return Err(mismatch(
                "columnar next run identifier is not beyond retained runs",
            ));
        }
        if segments.keys().any(|name| !expected.contains(name)) {
            return Err(mismatch("snapshot contains undescribed columnar segments"));
        }
        state
            .restore_index_capacities(&control.capacities)
            .map_err(|error| mismatch(&error.to_string()))?;
        state.finish_restore(control.next_run);
        Ok(state)
    }

    fn key_record(&self) -> Result<RecordBatch> {
        let groups = (0..self.keys.len())
            .map(GroupId::new)
            .collect::<Result<Vec<_>>>()?;
        let columns = self.keys.columns(&groups)?;
        RecordBatch::try_new(key_schema(&self.spec, &self.schemas)?, columns)
            .map_err(|error| super::super::arrow_error(&error))
    }
}

fn decode_run(
    identity: &RecordBatch,
    payload: Option<RecordBatch>,
    descriptor: &RunDescriptor,
) -> Result<Run> {
    let rows = usize::try_from(descriptor.rows).map_err(|_| mismatch("run rows exceed usize"))?;
    if identity.num_rows() != rows
        || payload
            .as_ref()
            .is_some_and(|payload| payload.num_rows() != rows)
        || identity
            .columns()
            .iter()
            .any(|column| column.null_count() != 0)
    {
        return Err(mismatch(
            "columnar identity and payload lengths or nulls differ",
        ));
    }
    let groups = identity
        .column(0)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("validated snapshot identity schema")
        .values()
        .iter()
        .map(|group| GroupId::new(*group as usize))
        .collect::<Result<Vec<_>>>()?;
    let times = identity
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("validated snapshot identity schema")
        .values()
        .to_vec();
    let sequences = SequenceColumn::from_array(identity.column(2))?;
    let payload_live = identity
        .column(3)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("validated snapshot identity schema")
        .values()
        .iter()
        .collect::<Vec<_>>();
    let identity_live = identity
        .column(4)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .expect("validated snapshot identity schema")
        .values()
        .iter()
        .collect::<Vec<_>>();
    if descriptor.side == 0 && payload_live != identity_live {
        return Err(mismatch("left payload and identity liveness differ"));
    }
    if payload_live.iter().any(|live| *live) != descriptor.payload {
        return Err(mismatch("payload descriptor contradicts live rows"));
    }
    let payload_charge = usize::try_from(descriptor.payload_charge)
        .map_err(|_| mismatch("payload charge exceeds usize"))?;
    let identity_charge = usize::try_from(descriptor.identity_charge)
        .map_err(|_| mismatch("identity charge exceeds usize"))?;
    if identity_charge == 0 || (descriptor.payload && payload_charge == 0) {
        return Err(mismatch("columnar run has a zero charge"));
    }
    Run::restored(
        payload.map(Arc::new),
        groups,
        times,
        sequences,
        payload_live,
        identity_live,
        (payload_charge, identity_charge),
    )
}

fn validate_payload(
    state: &mut ColumnarState,
    side: usize,
    run: &Run,
    run_id: u64,
    spec: &StreamAsofJoinSpec,
) -> Result<()> {
    let Some(payload) = run.payload_record() else {
        return Ok(());
    };
    let declaration = declaration(spec, side);
    for name in admission::identity_column_names(declaration) {
        if admission::identity_column_nulls(payload, name) {
            return Err(mismatch("restored payload identity contains nulls"));
        }
    }
    let times = admission::times(payload, declaration);
    let sequences = SequenceColumn::from_names(payload, declaration.sequence_by())?;
    for row in 0..run.rows() {
        let locator = Run::locator(run_id, row);
        if times.value(row) != run.time(locator) || sequences.value(row) != run.sequence(locator) {
            return Err(mismatch(
                "payload time or sequence differs from its run identity",
            ));
        }
    }
    let cut = state.keys.cut();
    let result = (|| {
        for start in (0..payload.num_rows()).step_by(MORSEL_ROWS) {
            let len = MORSEL_ROWS.min(payload.num_rows() - start);
            let arrays = declaration
                .keys()
                .iter()
                .map(|name| {
                    let index = payload.schema().index_of(name).expect("validated key");
                    payload.column(index).slice(start, len)
                })
                .collect::<Vec<_>>();
            let groups = state
                .keys
                .route_with_limit(&arrays, usize::MAX, &mut || Ok(()))
                .map_err(|error| mismatch(&error.to_string()))?;
            for (offset, group) in groups.into_iter().enumerate() {
                if group != run.groups()[start + offset] {
                    return Err(mismatch("payload key differs from its run group"));
                }
            }
        }
        Ok(())
    })();
    state.keys.rollback(cut);
    result
}

fn identity_record(run: &Run, large_sequence: bool) -> Result<RecordBatch> {
    let groups = run
        .groups()
        .iter()
        .map(|group| u32::try_from(group.index()).expect("GroupId is backed by u32"))
        .collect::<Vec<_>>();
    let columns = vec![
        Arc::new(UInt32Array::from(groups)) as _,
        Arc::new(Int64Array::from(run.times().to_vec())) as _,
        run.sequences().array(),
        Arc::new(BooleanArray::from(run.payload_liveness().to_vec())) as _,
        Arc::new(BooleanArray::from(run.identity_liveness().to_vec())) as _,
    ];
    RecordBatch::try_new(identity_schema(large_sequence), columns)
        .map_err(|error| super::super::arrow_error(&error))
}

fn key_schema(spec: &StreamAsofJoinSpec, schemas: &[SchemaRef; 2]) -> Result<SchemaRef> {
    let fields = spec
        .left()
        .keys()
        .iter()
        .map(|name| schemas[0].field_with_name(name).cloned().map(Arc::new))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| super::super::arrow_error(&error))?;
    Ok(Arc::new(Schema::new(fields)))
}

fn declaration(spec: &StreamAsofJoinSpec, side: usize) -> &AsofJoinSide {
    if side == 0 { spec.left() } else { spec.right() }
}

fn identity_id(side: usize, run: u64) -> String {
    format!("asof-columnar-v1-{side}-{run:020}-identity")
}

fn payload_id(side: usize, run: u64) -> String {
    format!("asof-columnar-v1-{side}-{run:020}-payload")
}

fn decode(
    segment: Option<&StateSegment>,
    schema: &SchemaRef,
    max_rows: u64,
) -> Result<RecordBatch> {
    let segment = segment.ok_or_else(|| mismatch("missing columnar data segment"))?;
    verify_checksum(segment)?;
    let digest = codec::schema_digest(schema)?;
    codec::decode_table_batch(segment.bytes(), &digest, schema, max_rows)
        .map_err(|error| mismatch(&error.to_string()))
}

fn verify_checksum(segment: &StateSegment) -> Result<()> {
    if hex::encode(Sha256::digest(segment.bytes())) != segment.sha256() {
        return Err(mismatch("columnar segment checksum mismatch"));
    }
    Ok(())
}

fn mismatch(message: &str) -> crate::CalcFlowError {
    crate::CalcFlowError::CheckpointMismatch {
        message: format!("ASOF columnar snapshot: {message}"),
    }
}
