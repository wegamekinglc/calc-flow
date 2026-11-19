#![allow(
    dead_code,
    reason = "P5 run adapter is tested before native production activation"
)]

use super::{
    AdmittedBatch, JoinCheckpointMetadata, JoinSide, RetainedRows, SideMetrics, StoredRow,
    StreamJoinOperator,
    admitted::AdmittedRef,
    checked_metric, columnar, counter_overflow, encode_join_key_v1, operator_error, output_gather,
    retained::Rows,
    runs::{Incoming, PreparedRuns, RunOutputContext, RunState},
    state_row_charge_with_key,
};
use crate::operator::keyed::{KeySchema, KeyStore, MORSEL_ROWS};
use crate::state::v4::native::{NativeCapture, NativePage, NativePartition};
use crate::{Batch, BatchMetadata, Result, StreamCollector, StreamOperatorContext};
use datafusion::arrow::{
    array::BooleanArray,
    record_batch::{RecordBatch, RecordBatchOptions},
};
use std::{collections::BTreeMap, sync::Arc};

pub(super) struct RunCore {
    state: RunState,
    keys: KeyStore,
}

impl RunCore {
    pub(super) fn new(operator: &mut StreamJoinOperator) -> Result<Self> {
        let compiled = operator.retained_compiled();
        let schema = operator.retained_schema(0);
        let fields = compiled
            .left_key_indices
            .iter()
            .map(|&index| schema.fields[index].clone())
            .collect();
        let keys = KeyStore::new(
            KeySchema::groupable(fields)?,
            operator
                .runtime
                .runtime()?
                .incremental_reservation("stream-join-run-keys"),
        );
        Ok(Self {
            state: RunState::new(&operator.name, operator.spec.limits, operator.spec.bounds)?,
            keys,
        })
    }

    pub(super) fn native_capture(
        &self,
        operator: &StreamJoinOperator,
        epoch: crate::Epoch,
    ) -> Result<Option<NativeCapture>> {
        let mut metrics = operator.state.metrics.clone();
        for side in [JoinSide::Left, JoinSide::Right] {
            let rows = u64::try_from(self.state.rows(side)).unwrap_or(u64::MAX);
            let bytes = u64::try_from(self.state.bytes(side)).unwrap_or(u64::MAX);
            if side == JoinSide::Left {
                metrics.left.retained_rows = rows;
                metrics.left.retained_bytes = bytes;
            } else {
                metrics.right.retained_rows = rows;
                metrics.right.retained_bytes = bytes;
            }
        }
        let metadata = JoinCheckpointMetadata {
            layout_version: 12,
            spec: operator.spec.clone(),
            next_left_row_id: self.state.next_row_id(JoinSide::Left),
            next_right_row_id: self.state.next_row_id(JoinSide::Right),
            next_output_sequence: self.state.next_output_sequence(),
            metrics,
            ended: operator.state.ended,
            epoch: epoch.as_u64(),
        };
        let metadata = serde_json::to_value(metadata).map_err(|error| {
            operator_error(&operator.name, &format!("native metadata: {error}"))
        })?;
        let serde_json::Value::Object(metadata) = metadata else {
            unreachable!("Join native metadata is an object")
        };
        let mut pages = self.state.native_pages(epoch)?;
        let keys = self.keys.record()?;
        let key_rows = u64::try_from(keys.num_rows())
            .map_err(|_| counter_overflow(&operator.name, "native key rows"))?;
        pages.push(NativePage::arrow("keys", &keys, key_rows, 0..key_rows));
        Ok(Some(NativeCapture::new(
            metadata.into_iter().collect(),
            vec![NativePartition {
                key_group: 0,
                pages,
            }],
        )?))
    }

    pub(super) fn restore_native(
        operator: &mut StreamJoinOperator,
        metadata: &JoinCheckpointMetadata,
        segments: &BTreeMap<String, crate::StateSegment>,
    ) -> Result<Self> {
        let mut core = Self::new(operator)?;
        core.state = RunState::restore_native(
            &operator.name,
            metadata.spec.limits,
            metadata.spec.bounds,
            segments,
            &mut core.keys,
        )?;
        Ok(core)
    }

    pub(super) fn validate_native_metadata(
        &self,
        operator: &StreamJoinOperator,
        metadata: &JoinCheckpointMetadata,
    ) -> Result<()> {
        let mismatch = || crate::CalcFlowError::CheckpointMismatch {
            message: format!(
                "stream Join {:?} native state counters differ",
                operator.name
            ),
        };
        let counts_match =
            |actual: usize, expected: u64| u64::try_from(actual).ok() == Some(expected);
        for side in [JoinSide::Left, JoinSide::Right] {
            let metrics = if side == JoinSide::Left {
                &metadata.metrics.left
            } else {
                &metadata.metrics.right
            };
            if !counts_match(self.state.rows(side), metrics.retained_rows)
                || !counts_match(self.state.bytes(side), metrics.retained_bytes)
            {
                return Err(mismatch());
            }
        }
        if self.state.next_row_id(JoinSide::Left) != metadata.next_left_row_id
            || self.state.next_row_id(JoinSide::Right) != metadata.next_right_row_id
            || self.state.next_output_sequence() != metadata.next_output_sequence
            || self.state.match_limit_failures() != metadata.metrics.match_limit_failures
            || metadata.ended
                && (self.state.rows(JoinSide::Left) != 0 || self.state.rows(JoinSide::Right) != 0)
        {
            return Err(mismatch());
        }
        Ok(())
    }

    pub(super) fn from_legacy(operator: &mut StreamJoinOperator) -> Result<Self> {
        let mut core = Self::new(operator)?;
        for side in [JoinSide::Left, JoinSide::Right] {
            let rows = if side == JoinSide::Left {
                &operator.state.left
            } else {
                &operator.state.right
            };
            let compiled = operator.retained_compiled();
            let keys = if side == JoinSide::Left {
                &compiled.left_key_indices
            } else {
                &compiled.right_key_indices
            };
            for (_, row) in rows.indexed() {
                let view = row.record.view();
                let incoming = Incoming::route(
                    &view,
                    keys,
                    vec![0],
                    vec![row.event_time],
                    vec![row.row_id],
                    BooleanArray::from(vec![true]),
                    row.row_id
                        .checked_add(1)
                        .ok_or_else(|| counter_overflow(&operator.name, "restored row_id"))?,
                    &mut core.keys,
                    &mut || Ok(()),
                )?;
                drop(
                    core.state.apply(
                        side,
                        incoming,
                        operator.output_ports[0]
                            .schema()
                            .expect("compiled Join output schema"),
                        operator.output_columns.as_deref(),
                        operator.retained_schema(0).fields().len(),
                        crate::EdgeBudget::default(),
                    )?,
                );
            }
        }
        core.state.restore_counters(
            [
                operator.state.next_left_row_id,
                operator.state.next_right_row_id,
            ],
            operator.state.next_output_sequence,
            operator.state.metrics.match_limit_failures,
        );
        Ok(core)
    }

    pub(super) fn sync_legacy(&self, operator: &mut StreamJoinOperator) -> Result<()> {
        let left = self.legacy_side(operator, JoinSide::Left)?;
        let right = self.legacy_side(operator, JoinSide::Right)?;
        operator.state.left = RetainedRows::from(left.0);
        operator.state.right = RetainedRows::from(right.0);
        operator.state.metrics.left.retained_rows =
            u64::try_from(operator.state.left.len()).unwrap_or(u64::MAX);
        operator.state.metrics.right.retained_rows =
            u64::try_from(operator.state.right.len()).unwrap_or(u64::MAX);
        operator.state.metrics.left.retained_bytes = left.1;
        operator.state.metrics.right.retained_bytes = right.1;
        Ok(())
    }

    fn legacy_side(
        &self,
        operator: &StreamJoinOperator,
        side: JoinSide,
    ) -> Result<(Vec<StoredRow>, u64)> {
        let mut rows = self.state.retained(side);
        rows.sort_by_key(|(time, row_id, _)| (time.as_micros(), *row_id));
        let compiled = operator.retained_compiled();
        let keys = if side == JoinSide::Left {
            &compiled.left_key_indices
        } else {
            &compiled.right_key_indices
        };
        let mut charge = 0u64;
        let rows = rows
            .into_iter()
            .map(|(event_time, row_id, payload)| {
                let parent = RecordBatch::try_new(
                    Arc::clone(payload.schema_ref()),
                    payload.columns().to_vec(),
                )
                .map_err(|error| {
                    operator_error(&operator.name, &format!("legacy capture failed: {error}"))
                })?;
                let offset = payload.offset();
                let encoded_key = encode_join_key_v1(&parent, offset, keys)?;
                let row_charge =
                    state_row_charge_with_key(&parent, offset, encoded_key.len(), &operator.name)?;
                charge = charge
                    .checked_add(row_charge)
                    .ok_or_else(|| counter_overflow(&operator.name, "state bytes"))?;
                Ok(StoredRow {
                    record: columnar::RowPayload::from(parent).with_offset(offset),
                    event_time,
                    row_id,
                    charge: row_charge,
                    encoded_key: Arc::new(encoded_key.into()),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((rows, charge))
    }

    pub(super) fn rows(&self, side: JoinSide) -> usize {
        self.state.rows(side)
    }

    #[cfg(test)]
    pub(super) fn fingerprint(&self) -> [u8; 32] {
        self.state.snapshot().fingerprint()
    }

    pub(super) async fn process(
        &mut self,
        operator: &mut StreamJoinOperator,
        ingress: &str,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let plan = operator.begin_batch(ingress, batch)?;
        let mut bundle = operator.admission_bundle(&plan, batch)?;
        operator
            .admit_batch_records(ingress, batch, &plan, context, &mut bundle)
            .await?;
        bundle.finish(&operator.name)?;
        operator
            .project_admitted_payloads(&mut bundle.admitted, plan.port_index, context)
            .await?;
        let metrics = std::mem::take(&mut bundle.metrics);
        let cut = self.keys.cut();
        let prepared = self.prepare(operator, &plan, &bundle.admitted, context);
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                self.keys.rollback(cut);
                operator.state.metrics.match_limit_failures = self.state.match_limit_failures();
                if matches!(
                    &error,
                    crate::CalcFlowError::OperatorReason {
                        reason_code: crate::StreamingFailureReason::JoinStateLimitExceeded,
                        ..
                    }
                ) {
                    operator.state.metrics.state_limit_failures = checked_metric(
                        operator.state.metrics.state_limit_failures,
                        1,
                        &operator.name,
                        "state_limit_failures",
                    )?;
                }
                return Err(error);
            }
        };
        let emitted = match checked_metric(
            operator.state.metrics.emitted_match_rows,
            u64::try_from(prepared.matched_rows())
                .map_err(|_| counter_overflow(&operator.name, "matched rows"))?,
            &operator.name,
            "emitted_match_rows",
        ) {
            Ok(emitted) => emitted,
            Err(error) => {
                self.keys.rollback(cut);
                return Err(error);
            }
        };
        for item in prepared.outputs() {
            for (record, sequence) in item.records.iter().zip(&item.sequences) {
                let result = async {
                    context.check_cancelled()?;
                    let metadata = BatchMetadata::new(&operator.name, *sequence, BTreeMap::new())?;
                    output
                        .emit("output", Batch::table(vec![record.clone()], metadata)?)
                        .await
                }
                .await;
                if let Err(error) = result {
                    self.keys.rollback(cut);
                    return Err(error);
                }
            }
        }
        let side = if plan.incoming_is_left {
            JoinSide::Left
        } else {
            JoinSide::Right
        };
        drop(prepared.commit(&mut self.state)?);
        self.sync_side(operator, side, metrics);
        operator.state.next_left_row_id = self.state.next_row_id(JoinSide::Left);
        operator.state.next_right_row_id = self.state.next_row_id(JoinSide::Right);
        operator.state.next_output_sequence = self.state.next_output_sequence();
        operator.state.metrics.emitted_match_rows = emitted;
        operator.state.metrics.match_limit_failures = self.state.match_limit_failures();
        operator.ingress_progress = context.ingress_progress().clone();
        Ok(())
    }

    fn prepare(
        &mut self,
        operator: &StreamJoinOperator,
        plan: &super::SidePlan,
        admitted: &AdmittedBatch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedRuns> {
        let side = if plan.incoming_is_left {
            JoinSide::Left
        } else {
            JoinSide::Right
        };
        let compiled = operator.retained_compiled();
        let key_indices = if plan.incoming_is_left {
            &compiled.left_key_indices
        } else {
            &compiled.right_key_indices
        };
        let incoming = route(admitted, key_indices, &mut self.keys, &mut || {
            context.check_cancelled()
        })?;
        self.state.prepare_batch(
            side,
            incoming,
            RunOutputContext::new(
                operator.output_ports[0]
                    .schema()
                    .expect("compiled Join output schema"),
                operator.output_columns.as_deref(),
                operator.retained_schema(0).fields().len(),
                context.output_budget(),
            ),
            &mut || context.check_cancelled(),
        )
    }

    pub(super) fn evict(
        &mut self,
        operator: &mut StreamJoinOperator,
        ingress: &str,
        progress: crate::IngressProgress,
    ) -> Result<()> {
        let (side, extension) = match ingress {
            "left" => (JoinSide::Right, operator.spec.bounds.before_micros),
            "right" => (JoinSide::Left, operator.spec.bounds.after_micros),
            _ => {
                return Err(operator_error(
                    &operator.name,
                    &format!("unknown ingress progress {ingress:?}"),
                ));
            }
        };
        let expiring = if progress.state() == crate::IngressState::Ended {
            self.state.rows(side)
        } else {
            progress.watermark().map_or(0, |watermark| {
                self.state.expiring_rows(side, extension, watermark)
            })
        };
        let mut metrics = if side == JoinSide::Left {
            operator.state.metrics.left.clone()
        } else {
            operator.state.metrics.right.clone()
        };
        metrics.evicted_rows = checked_metric(
            metrics.evicted_rows,
            u64::try_from(expiring)
                .map_err(|_| counter_overflow(&operator.name, "evicted rows"))?,
            &operator.name,
            "evicted_rows",
        )?;
        let eviction = if progress.state() == crate::IngressState::Ended {
            Some(self.state.evict_end(side)?)
        } else {
            progress
                .watermark()
                .map(|watermark| self.state.evict(side, extension, watermark))
                .transpose()?
        };
        let Some(eviction) = eviction else {
            return Ok(());
        };
        debug_assert_eq!(eviction.expired_rows, expiring);
        self.sync_side(operator, side, metrics);
        Ok(())
    }

    fn sync_side(
        &self,
        operator: &mut StreamJoinOperator,
        side: JoinSide,
        mut metrics: SideMetrics,
    ) {
        let rows = u64::try_from(self.state.rows(side)).unwrap_or(u64::MAX);
        let bytes = u64::try_from(self.state.bytes(side)).unwrap_or(u64::MAX);
        metrics.retained_rows = rows;
        metrics.retained_bytes = bytes;
        if side == JoinSide::Left {
            operator.state.metrics.left = metrics;
        } else {
            operator.state.metrics.right = metrics;
        }
    }
}

fn route(
    admitted: &AdmittedBatch,
    key_indices: &[usize],
    keys: &mut KeyStore,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<Incoming>> {
    (0..admitted.len())
        .step_by(MORSEL_ROWS)
        .map(|start| {
            let end = (start + MORSEL_ROWS).min(admitted.len());
            let rows = admitted
                .iter()
                .skip(start)
                .take(end - start)
                .collect::<Vec<_>>();
            let payload = morsel(&rows)?;
            let times = rows.iter().map(|row| row.event_time).collect();
            let row_ids = rows.iter().map(|row| row.row_id).collect::<Vec<_>>();
            let next_row_id = row_ids
                .last()
                .expect("nonempty morsel")
                .checked_add(1)
                .ok_or_else(|| counter_overflow("Join runs", "row_id"))?;
            let retain = BooleanArray::from(rows.iter().map(|row| row.retain).collect::<Vec<_>>());
            Incoming::route(
                &payload,
                key_indices,
                (0..rows.len()).collect(),
                times,
                row_ids,
                retain,
                next_row_id,
                keys,
                check,
            )
        })
        .collect()
}

fn morsel(rows: &[AdmittedRef<'_>]) -> Result<RecordBatch> {
    let first = rows
        .first()
        .ok_or_else(|| operator_error("Join runs", "cannot route an empty admitted morsel"))?;
    let gather = output_gather::SideGather::new(rows.iter().map(|row| row.record));
    let schema = first.record.schema_ref();
    let columns = schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| output_gather::column(gather.as_ref(), index, field.data_type()))
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new_with_options(
        Arc::clone(schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(rows.len())),
    )
    .map_err(|error| operator_error("Join runs", &format!("admitted morsel failed: {error}")))
}

pub(super) fn supported(operator: &StreamJoinOperator) -> bool {
    let compiled = operator.retained_compiled();
    let left = operator.retained_schema(0);
    compiled
        .left_key_indices
        .iter()
        .zip(&compiled.right_key_indices)
        .all(|(&left, &right)| {
            operator.retained_schema(0).fields[left].data_type()
                == operator.retained_schema(1).fields[right].data_type()
        })
        && KeySchema::groupable(
            compiled
                .left_key_indices
                .iter()
                .map(|&index| left.fields[index].clone())
                .collect(),
        )
        .is_ok()
}
