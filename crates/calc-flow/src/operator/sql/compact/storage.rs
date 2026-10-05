use std::sync::Arc;

use datafusion::{arrow::datatypes::SchemaRef, execution::memory_pool::MemoryReservation};

use super::super::{
    PreparedRetention, SqlOperator, incremental, ipc, metadata, retention, sql_state_error,
};
use super::{capture::CompactCapture, control::QuotaLedger};
use crate::{Batch, BatchMetadata, CalcFlowError, Result};

pub(super) struct CompactColumns {
    pub logical: SchemaRef,
    pub physical: SchemaRef,
    pub ordinals: Vec<usize>,
    pub logical_segment: super::super::StateSegment,
    pub projection: Option<Arc<retention::SqlProjection>>,
    _reservation: Arc<MemoryReservation>,
}

pub(in crate::operator::sql) struct CompactSqlState {
    pub(in crate::operator::sql) ledger: QuotaLedger,
    pub(in crate::operator::sql) metadata: BatchMetadata,
    pub(super) columns: Arc<CompactColumns>,
    pub(in crate::operator::sql) capture: Option<Arc<CompactCapture>>,
    _reservation: MemoryReservation,
}

impl CompactSqlState {
    #[cfg(test)]
    pub(in crate::operator::sql) fn recovery_credit(
        &self,
    ) -> (usize, Vec<std::sync::Weak<MemoryReservation>>) {
        let Self {
            _reservation: state_credit,
            columns,
            ..
        } = self;
        let CompactColumns {
            _reservation: column_credit,
            ..
        } = columns.as_ref();
        (
            state_credit.size() + column_credit.size(),
            vec![Arc::downgrade(column_credit)],
        )
    }

    pub(in crate::operator::sql) fn projection(&self) -> Option<Arc<retention::SqlProjection>> {
        self.columns.projection.clone()
    }

    fn ignores_empty_schema(&self, operator: &SqlOperator, batch: &Batch) -> bool {
        batch.num_rows() == 0
            && operator.input_ports[0].schema().is_none()
            && self.columns.projection.is_none()
    }

    pub(in crate::operator::sql) fn check_input(
        &self,
        operator: &SqlOperator,
        batch: &Batch,
    ) -> Result<()> {
        if self.ignores_empty_schema(operator, batch) {
            return Ok(());
        }
        let schema = batch.table_payload()?.schema();
        if self.columns.projection.is_some() && schema != &self.columns.logical {
            return Err(CalcFlowError::InvalidArgument {
                field: "retained_columns".into(),
                message: "input does not match the logical retained schema".into(),
            });
        }
        check_schema(&self.columns.logical, schema)
    }

    pub(super) fn restored(
        operator: &SqlOperator,
        columns: Arc<CompactColumns>,
        ledger: QuotaLedger,
        metadata: BatchMetadata,
        capture: Option<Arc<CompactCapture>>,
    ) -> Result<Self> {
        let reservation = state_credit(operator, &metadata)?;
        Ok(Self {
            ledger,
            metadata,
            columns,
            capture,
            _reservation: reservation,
        })
    }
}

pub(in crate::operator::sql) fn prepare_update(
    operator: &SqlOperator,
    prepared: &PreparedRetention,
) -> Result<CompactSqlState> {
    let ledger = update_ledger(operator, &prepared.batch)?;
    validate_budget(operator, ledger)?;
    let reservation = state_credit(operator, prepared.batch.metadata())?;
    let columns = update_columns(operator, prepared)?;
    let capture = operator
        .compact
        .as_ref()
        .and_then(|state| state.capture.clone());
    Ok(CompactSqlState {
        ledger,
        metadata: prepared.batch.metadata().clone(),
        columns,
        capture,
        _reservation: reservation,
    })
}

fn update_ledger(operator: &SqlOperator, batch: &Batch) -> Result<QuotaLedger> {
    if let Some(state) = &operator.compact {
        extend_ledger(state.ledger, batch)
    } else {
        let (rows, bytes) = operator.accumulated_charge(batch, operator.retained.as_ref())?;
        Ok(QuotaLedger {
            rows,
            bytes,
            seen_input: true,
        })
    }
}

fn batch_charge(batch: &Batch) -> Result<(u64, u64)> {
    let rows = u64::try_from(batch.num_rows())
        .map_err(|_| sql_state_error("input row count exceeds u64"))?;
    let bytes = if rows == 0 {
        0
    } else {
        u64::try_from(batch.estimated_bytes()?)
            .map_err(|_| sql_state_error("input byte count exceeds u64"))?
    };
    Ok((rows, bytes))
}

fn extend_ledger(previous: QuotaLedger, batch: &Batch) -> Result<QuotaLedger> {
    let (rows, bytes) = batch_charge(batch)?;
    Ok(QuotaLedger {
        rows: previous
            .rows
            .checked_add(rows)
            .ok_or_else(|| sql_state_error("retained row count overflowed"))?,
        bytes: previous
            .bytes
            .checked_add(bytes)
            .ok_or_else(|| sql_state_error("retained byte count overflowed"))?,
        seen_input: true,
    })
}

fn update_columns(
    operator: &SqlOperator,
    prepared: &PreparedRetention,
) -> Result<Arc<CompactColumns>> {
    if let Some(state) = &operator.compact {
        if !state.ignores_empty_schema(operator, &prepared.batch) {
            check_schema(
                &state.columns.physical,
                prepared.batch.table_payload()?.schema(),
            )?;
        }
        Ok(state.columns.clone())
    } else {
        if let Some(state) = operator.retained.as_ref() {
            check_schema(
                &state.records[0].schema(),
                prepared.batch.table_payload()?.schema(),
            )?;
        }
        prepare_columns(operator, prepared)
    }
}

pub(super) fn columns(
    operator: &SqlOperator,
    logical: SchemaRef,
    projection: Option<Arc<retention::SqlProjection>>,
) -> Result<Arc<CompactColumns>> {
    let runtime = operator.retention_runtime()?;
    let reservation = runtime.incremental_reservation(&operator.name);
    let bytes =
        incremental::checked_bytes(8192, [(ipc::schema_bytes(&logical)?, 16)], &operator.name)?;
    incremental::ensure_reservation(&reservation, bytes, &operator.name)?;
    let reservation = Arc::new(reservation);
    let (physical, ordinals, segment) = if let Some(projection) = &projection {
        (
            projection.columns.physical_schema().clone(),
            projection.columns.ordinals().to_vec(),
            projection.logical_segment.clone(),
        )
    } else {
        (
            logical.clone(),
            (0..logical.fields().len()).collect(),
            retention::encode_schema(&logical)?,
        )
    };
    if segment
        .bytes()
        .len()
        .checked_mul(2)
        .is_none_or(|bytes| bytes > reservation.size())
    {
        return Err(sql_state_error(
            "SQL compact logical schema exceeded its prepaid bound",
        ));
    }
    Ok(Arc::new(CompactColumns {
        logical,
        physical,
        ordinals,
        logical_segment: segment.with_owner(reservation.clone()),
        projection,
        _reservation: reservation,
    }))
}

fn prepare_columns(
    operator: &SqlOperator,
    prepared: &PreparedRetention,
) -> Result<Arc<CompactColumns>> {
    let logical = prepared.projection.as_ref().map_or_else(
        || {
            prepared
                .batch
                .table_payload()
                .map(|table| table.schema().clone())
        },
        |projection| Ok(projection.columns.logical_schema().clone()),
    )?;
    if let Some(declared) = operator.input_ports[0].schema() {
        check_schema(declared, &logical)?;
    }
    columns(operator, logical, prepared.projection.clone())
}

fn check_schema(expected: &SchemaRef, actual: &SchemaRef) -> Result<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(CalcFlowError::InvalidArgument {
            field: "batches".into(),
            message: "schemas must match".into(),
        })
    }
}

pub(super) fn validate_budget(operator: &SqlOperator, ledger: QuotaLedger) -> Result<()> {
    if operator
        .state_budget
        .is_some_and(|budget| !budget.allows(ledger.rows, ledger.bytes))
    {
        return Err(CalcFlowError::Operator {
            node_id: operator.name.clone(),
            message: "SQL aggregate retained input exceeds the configured state budget".into(),
        });
    }
    Ok(())
}

fn state_credit(operator: &SqlOperator, latest: &BatchMetadata) -> Result<MemoryReservation> {
    let reservation = metadata::reserve(operator.retention_runtime()?, latest, &operator.name)?;
    let bytes = incremental::checked_bytes(reservation.size(), [(1, 8192)], &operator.name)?;
    incremental::ensure_reservation(&reservation, bytes, &operator.name)?;
    Ok(reservation)
}
