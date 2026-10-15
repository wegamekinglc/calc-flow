use std::sync::Arc;

use datafusion::execution::memory_pool::MemoryReservation;
use serde_json::Value;

use super::super::{StoredRow, StreamJoinSpec};
use super::geometry::invalid;
use super::history;
use super::ipc::accounting::{add, arc, product, reserve, sum};
use super::payload::Funding;
use crate::runtime::streaming::gather_work::RetirementGuard;
use crate::{JsonMap, Result};

pub(in crate::operator::join) struct ContainerFunding {
    pub(super) credit: Arc<MemoryReservation>,
    _retirement: Option<RetirementGuard>,
}

impl ContainerFunding {
    pub(super) fn new(
        credit: MemoryReservation,
        retirement: Option<RetirementGuard>,
    ) -> Result<Arc<Self>> {
        reserve(
            &credit,
            sum(&[
                arc::<Self>()?,
                arc::<MemoryReservation>()?,
                registration_bytes()?,
            ])?,
        )?;
        Ok(Arc::new(Self {
            credit: Arc::new(credit),
            _retirement: retirement,
        }))
    }

    pub(super) fn grow(&self, bytes: usize) -> Result<()> {
        reserve(&self.credit, bytes)
    }
}

pub(super) fn metadata_bytes(metadata: &JsonMap, spec: &StreamJoinSpec) -> Result<usize> {
    sum(&[
        product(map_bytes(metadata)?, 3)?,
        product(spec_bytes(spec)?, 3)?,
        size_of::<super::super::metadata_validation::ValidatedMetadata>(),
    ])
}

fn map_bytes(metadata: &JsonMap) -> Result<usize> {
    let initial = tree::<String, Value>(metadata.len())?;
    metadata.iter().try_fold(initial, |bytes, (key, value)| {
        sum(&[bytes, key.len(), value_bytes(value)?])
    })
}

fn value_bytes(value: &Value) -> Result<usize> {
    match value {
        Value::String(text) => Ok(text.len()),
        Value::Array(values) => values.iter().try_fold(
            product(values.len(), size_of::<Value>())?,
            |bytes, value| add(bytes, value_bytes(value)?),
        ),
        Value::Object(values) => values.iter().try_fold(
            tree::<String, Value>(values.len())?,
            |bytes, (key, value)| sum(&[bytes, key.len(), value_bytes(value)?]),
        ),
        _ => Ok(0),
    }
}

pub(super) fn spec_bytes(spec: &StreamJoinSpec) -> Result<usize> {
    let text = sum(&[
        spec.left_event_time.len(),
        spec.right_event_time.len(),
        spec.left_prefix.len(),
        spec.right_prefix.len(),
    ])?;
    spec.left_keys.iter().chain(&spec.right_keys).try_fold(
        add(
            text,
            product(
                add(spec.left_keys.len(), spec.right_keys.len())?,
                size_of::<String>(),
            )?,
        )?,
        |bytes, key| add(bytes, key.len()),
    )
}

pub(super) fn payload_seed() -> Result<usize> {
    sum(&[
        arc::<Funding>()?,
        arc::<MemoryReservation>()?,
        registration_bytes()?,
    ])
}

pub(super) fn registration_bytes() -> Result<usize> {
    let legacy = super::super::metadata_validation::inventory::registration_controls()
        .ok_or_else(|| invalid("V2 registration charge overflow"))?;
    let extra = "stream-join-v2-restore-input".len() - "stream-join-metadata".len();
    add(legacy, product(extra, 3)?)
}

pub(super) fn containers(rows: usize) -> Result<usize> {
    sum(&[
        product(rows, size_of::<StoredRow>() + 128)?,
        product(rows.div_ceil(1024), 16_384)?,
        product(2, arc::<super::super::StateRows>()?)?,
        sql_loan_controls()?,
    ])
}

pub(in crate::operator::join) fn expiration_credit(
    workspace: &MemoryReservation,
    capacity: usize,
    controls: usize,
) -> Result<MemoryReservation> {
    let credit = workspace.new_empty();
    reserve(&credit, expiration_bytes(capacity, controls)?)?;
    Ok(credit)
}

pub(in crate::operator::join) fn optional_expiration_credit(
    workspace: &MemoryReservation,
    capacity: usize,
    controls: usize,
) -> Result<Option<MemoryReservation>> {
    let credit = workspace.new_empty();
    match credit.try_grow(expiration_bytes(capacity, controls)?) {
        Ok(()) => Ok(Some(credit)),
        Err(datafusion::common::DataFusionError::ResourcesExhausted(_)) => Ok(None),
        Err(error) => Err(crate::CalcFlowError::DataFusion {
            node_id: None,
            message: error.to_string(),
        }),
    }
}

fn expiration_bytes(capacity: usize, controls: usize) -> Result<usize> {
    sum(&[
        product(capacity, 2 * size_of::<usize>() + size_of::<u128>())?,
        controls,
        registration_bytes()?,
    ])
}

pub(super) fn tree<K, V>(count: usize) -> Result<usize> {
    history::tree::<K, V>(count).ok_or_else(|| invalid("V2 candidate tree charge overflow"))
}

fn sql_loan_controls() -> Result<usize> {
    use super::loans::V2SqlOwners;
    use crate::datafusion::owned::{Failure, Input, Output};
    sum(&[
        size_of::<V2SqlOwners>(),
        size_of::<Input<V2SqlOwners>>(),
        size_of::<Output<V2SqlOwners>>(),
        size_of::<Failure<V2SqlOwners>>(),
    ])
}
