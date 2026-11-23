use super::{
    AdmittedBatch, JoinCheckpointMetadata, JoinSide, SideMetrics, StreamJoinOperator,
    admitted::AdmittedRef,
    checked_metric, counter_overflow, operator_error, output_gather,
    runs::{Incoming, NativePageCache, PreparedRuns, RouteInput, RunOutputContext, RunState},
};
#[cfg(test)]
use super::{
    RetainedRows, StoredRow, columnar, encode_join_key_v1, retained::Rows,
    state_row_charge_with_key,
};
use crate::operator::keyed::{KeySchema, KeyStore, MORSEL_ROWS};
use crate::state::v4::native::{NativeCapture, NativePage, NativePageChange, NativePartition};
use crate::{Batch, BatchMetadata, EventTime, Result, StreamCollector, StreamOperatorContext};
use datafusion::arrow::{
    array::{BooleanArray, TimestampMicrosecondArray},
    datatypes::SchemaRef,
    record_batch::{RecordBatch, RecordBatchOptions},
};
use std::{collections::BTreeMap, sync::Arc};

pub(super) struct RunCore {
    state: RunState,
    hot: Option<super::hot::HotJoinCore>,
    keys: KeyStore,
    pages: NativePageCache,
    key_page: Option<(usize, NativePage)>,
    #[cfg(test)]
    key_reuses: usize,
    #[cfg(test)]
    last_change_count: usize,
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
            hot: super::hot::HotJoinCore::new(operator),
            keys,
            pages: NativePageCache::default(),
            key_page: None,
            #[cfg(test)]
            key_reuses: 0,
            #[cfg(test)]
            last_change_count: 0,
        })
    }

    fn migrate_hot(&mut self, operator: &mut StreamJoinOperator) -> Result<()> {
        let Some([left, right]) = self
            .hot
            .as_ref()
            .filter(|hot| hot.has_history())
            .map(super::hot::HotJoinCore::history)
        else {
            return Ok(());
        };
        let mut migrated = Self::new(operator)?;
        migrated.hot = None;
        for (side, batch) in [(JoinSide::Left, left), (JoinSide::Right, right)] {
            if let Some(batch) = batch {
                migrated.seed_hot(operator, side, &batch)?;
            }
        }
        migrated.state.restore_counters(
            [
                operator.state.next_left_row_id,
                operator.state.next_right_row_id,
            ],
            operator.state.next_output_sequence,
            operator.state.metrics.match_limit_failures,
        );
        for side in [JoinSide::Left, JoinSide::Right] {
            let metrics = if side == JoinSide::Left {
                operator.state.metrics.left.clone()
            } else {
                operator.state.metrics.right.clone()
            };
            migrated.sync_side(operator, side, metrics);
        }
        operator.state.next_left_row_id = migrated.state.next_row_id(JoinSide::Left);
        operator.state.next_right_row_id = migrated.state.next_row_id(JoinSide::Right);
        operator.state.next_output_sequence = migrated.state.next_output_sequence();
        *self = migrated;
        let progress = operator.ingress_progress.clone();
        for (ingress, progress) in progress.by_ingress() {
            self.evict(operator, ingress, *progress)?;
        }
        Ok(())
    }

    fn seed_hot(
        &mut self,
        operator: &StreamJoinOperator,
        side: JoinSide,
        batch: &Batch,
    ) -> Result<()> {
        let compiled = operator.retained_compiled();
        let (event_time, keys) = if side == JoinSide::Left {
            (
                compiled.left_event_time_index,
                compiled.left_key_indices.as_slice(),
            )
        } else {
            (
                compiled.right_event_time_index,
                compiled.right_key_indices.as_slice(),
            )
        };
        let mut next_row_id = self.state.next_row_id(side);
        let mut incoming = Vec::new();
        for record in batch.table_payload()?.batches() {
            let record = retained_hot_record(operator, side, record)?;
            let times = record
                .column(event_time)
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .expect("hot Join time is microseconds");
            for start in (0..record.num_rows()).step_by(MORSEL_ROWS) {
                let rows = (record.num_rows() - start).min(MORSEL_ROWS);
                let row_ids = (0..rows)
                    .map(|row| {
                        let row = u64::try_from(row)
                            .map_err(|_| counter_overflow(&operator.name, "hot row_id"))?;
                        next_row_id
                            .checked_add(row)
                            .ok_or_else(|| counter_overflow(&operator.name, "hot row_id"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let following_row_id = next_row_id
                    .checked_add(
                        u64::try_from(rows)
                            .map_err(|_| counter_overflow(&operator.name, "hot row_id"))?,
                    )
                    .ok_or_else(|| counter_overflow(&operator.name, "hot row_id"))?;
                let times = times
                    .iter()
                    .skip(start)
                    .take(rows)
                    .map(|time| {
                        time.map(EventTime::from_micros).ok_or_else(|| {
                            operator_error(&operator.name, "hot history has null event time")
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let cut = self.keys.cut();
                let chunk = Incoming::route(
                    RouteInput::new(
                        &record.slice(start, rows),
                        keys,
                        (0..rows).collect(),
                        times,
                        row_ids,
                        BooleanArray::from(vec![true; rows]),
                        following_row_id,
                    ),
                    &mut self.keys,
                    &mut || Ok(()),
                );
                match chunk {
                    Ok(chunk) => incoming.push(chunk),
                    Err(error) => {
                        self.keys.rollback(cut);
                        return Err(error);
                    }
                }
                next_row_id = following_row_id;
            }
        }
        if incoming.is_empty() {
            return Ok(());
        }
        let output = RunOutputContext::new(
            run_output_schema(operator),
            operator.output_columns.as_deref(),
            operator.retained_schema(0).fields().len(),
            crate::EdgeBudget::default(),
        );
        let prepared = self
            .state
            .prepare_batch(side, incoming, output, &mut || Ok(()))?;
        drop(prepared.commit(&mut self.state)?);
        Ok(())
    }

    pub(super) fn native_capture(
        &mut self,
        operator: &mut StreamJoinOperator,
        epoch: crate::Epoch,
    ) -> Result<Option<NativeCapture>> {
        let terminal = operator.state.ended
            || ["left", "right"].into_iter().all(|ingress| {
                operator
                    .ingress_progress
                    .get(ingress)
                    .is_some_and(|progress| progress.state() == crate::IngressState::Ended)
            });
        if terminal {
            self.hot = None;
        } else {
            self.migrate_hot(operator)?;
        }
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
        let pages = self.state.native_pages(epoch, &mut self.pages)?;
        let mut changes = Vec::new();
        let key_rows = self.keys.len();
        if self
            .key_page
            .as_ref()
            .is_some_and(|(cached_rows, _)| *cached_rows == key_rows)
        {
            #[cfg(test)]
            {
                self.key_reuses += 1;
            }
        } else {
            let keys = self.keys.record()?;
            let generation = u64::try_from(keys.num_rows())
                .map_err(|_| counter_overflow(&operator.name, "native key rows"))?;
            let page = NativePage::arrow("keys", &keys, generation, 0..generation);
            changes.push(NativePageChange::upsert(
                page.clone(),
                self.key_page.as_ref().map(|(_, previous)| previous),
            )?);
            self.key_page = Some((key_rows, page));
        }
        let pages = pages.extend(changes)?;
        #[cfg(test)]
        {
            self.last_change_count = pages.change_count();
        }
        let lineage = Arc::new(pages.clone());
        let capture = NativeCapture::new(
            metadata.into_iter().collect(),
            vec![NativePartition {
                key_group: 0,
                pages,
            }],
        )?;
        self.pages.lineage = Some(lineage);
        self.state.finish_native_capture();
        Ok(Some(capture))
    }

    #[cfg(test)]
    pub(super) fn native_reuse_counts(&self) -> [usize; 3] {
        let [identities, indexes] = self.pages.reuse_counts();
        [identities, indexes, self.key_reuses]
    }

    #[cfg(test)]
    pub(super) fn native_change_count(&self) -> usize {
        self.last_change_count
    }

    #[cfg(test)]
    pub(super) fn native_reuse_and_change_counts(&self) -> ([usize; 3], usize) {
        (self.native_reuse_counts(), self.native_change_count())
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
        core.hot = None;
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

    #[cfg(test)]
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
                    RouteInput::new(
                        &view,
                        keys,
                        vec![0],
                        vec![row.event_time],
                        vec![row.row_id],
                        BooleanArray::from(vec![true]),
                        row.row_id
                            .checked_add(1)
                            .ok_or_else(|| counter_overflow(&operator.name, "restored row_id"))?,
                    ),
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
        if core.state.rows(JoinSide::Left) + core.state.rows(JoinSide::Right) > 0 {
            core.hot = None;
        }
        Ok(core)
    }

    #[cfg(test)]
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

    #[cfg(test)]
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
    pub(super) fn hot_enabled(&self) -> bool {
        self.hot.is_some()
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
        let requires_generic = self
            .hot
            .as_ref()
            .is_some_and(|hot| hot.requires_generic(ingress));
        if requires_generic
            || self
                .hot
                .as_ref()
                .is_some_and(|hot| hot.needs_migration(ingress))
        {
            self.migrate_hot(operator)?;
            if requires_generic {
                self.hot = None;
            }
        }
        #[cfg(feature = "benchmark-internals")]
        let admit_span =
            context
                .job()
                .join_cpu_trace()
                .stage(&operator.name, "run_admit", batch.num_rows());
        let plan = operator.begin_batch(ingress, batch)?;
        let mut bundle = operator.admission_bundle(&plan, batch)?;
        bundle.quantum = super::columnar_quantum::Quantum::bulk();
        operator
            .admit_batch_records(ingress, batch, &plan, context, &mut bundle)
            .await?;
        bundle.finish(&operator.name)?;
        #[cfg(test)]
        operator
            .project_admitted_payloads(&mut bundle.admitted, plan.port_index, context)
            .await?;
        #[cfg(feature = "benchmark-internals")]
        drop(admit_span);
        if let Some(input) = prepare_hot_input(
            operator,
            &plan,
            batch,
            &bundle,
            self.hot.is_some(),
            &mut self.keys,
        )? {
            let hot = self
                .hot
                .as_mut()
                .expect("hot admission requires a hot core");
            return hot.process(operator, input, context, output).await;
        }
        if self.hot.is_some() {
            self.migrate_hot(operator)?;
            self.hot = None;
        }
        self.process_admitted(operator, plan, bundle, context, output)
            .await
    }

    async fn process_admitted(
        &mut self,
        operator: &mut StreamJoinOperator,
        plan: super::SidePlan,
        bundle: super::AdmissionBundle,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let super::AdmissionBundle {
            metrics, admitted, ..
        } = bundle;
        let cut = self.keys.cut();
        let prepared = self.prepare(operator, &plan, &admitted, context);
        drop(admitted);
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
        if let Err(error) = emit_prepared(operator, &prepared, context, output).await {
            self.keys.rollback(cut);
            return Err(error);
        }
        let side = if plan.incoming_is_left {
            JoinSide::Left
        } else {
            JoinSide::Right
        };
        #[cfg(feature = "benchmark-internals")]
        let _commit = context.job().join_cpu_trace().stage(
            &operator.name,
            "run_commit",
            prepared.matched_rows(),
        );
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
        #[cfg(feature = "benchmark-internals")]
        let mut route_span =
            context
                .job()
                .join_cpu_trace()
                .stage(&operator.name, "run_route", admitted.len());
        let incoming = route(
            admitted,
            key_indices,
            &mut self.keys,
            &mut || context.check_cancelled(),
            #[cfg(feature = "benchmark-internals")]
            route_span.units(),
        )?;
        #[cfg(feature = "benchmark-internals")]
        drop(route_span);
        let output = RunOutputContext::new(
            run_output_schema(operator),
            operator.output_columns.as_deref(),
            operator.retained_schema(0).fields().len(),
            context.output_budget(),
        );
        #[cfg(feature = "benchmark-internals")]
        let output = output.with_trace(context.job().join_cpu_trace());
        self.state
            .prepare_batch(side, incoming, output, &mut || context.check_cancelled())
    }

    pub(super) fn evict(
        &mut self,
        operator: &mut StreamJoinOperator,
        ingress: &str,
        progress: crate::IngressProgress,
    ) -> Result<()> {
        if self
            .hot
            .as_ref()
            .is_some_and(super::hot::HotJoinCore::has_history)
        {
            self.migrate_hot(operator)?;
        }
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

fn prepare_hot_input<'a>(
    operator: &StreamJoinOperator,
    plan: &super::SidePlan,
    batch: &'a Batch,
    bundle: &'a super::AdmissionBundle,
    enabled: bool,
    key_store: &mut KeyStore,
) -> Result<Option<super::hot::HotInput<'a>>> {
    if !enabled
        || bundle.admitted.len() != batch.num_rows()
        || !bundle.admitted.iter().all(|row| row.retain)
    {
        return Ok(None);
    }
    let compiled = operator.retained_compiled();
    let keys = if plan.incoming_is_left {
        &compiled.left_key_indices
    } else {
        &compiled.right_key_indices
    };
    let charge = super::runs::hot_batch_charge(batch, keys, key_store)?;
    let within_rows = u64::try_from(charge.0).unwrap_or(u64::MAX)
        <= operator.spec.limits.max_state_rows_per_side();
    let within_bytes = u64::try_from(charge.1).unwrap_or(u64::MAX)
        <= operator.spec.limits.max_state_bytes_per_side();
    if !(within_rows && within_bytes) {
        return Ok(None);
    }
    let side = if plan.incoming_is_left {
        JoinSide::Left
    } else {
        JoinSide::Right
    };
    Ok(Some(super::hot::HotInput {
        side,
        batch,
        next_row_id: bundle.next_row_id,
        metrics: bundle.metrics.clone(),
        charge,
    }))
}

fn run_output_schema(operator: &StreamJoinOperator) -> &SchemaRef {
    if operator.hot_projection.is_some() {
        &operator.logical_output_schema
    } else {
        operator.output_ports[0]
            .schema()
            .expect("compiled Join output schema")
    }
}

fn retained_hot_record(
    operator: &StreamJoinOperator,
    side: JoinSide,
    record: &RecordBatch,
) -> Result<RecordBatch> {
    let schema = operator.retained_schema(side.index());
    let columns = schema
        .fields()
        .iter()
        .map(|field| {
            record
                .schema()
                .index_of(field.name())
                .map(|index| record.column(index).clone())
                .map_err(|error| {
                    operator_error(
                        &operator.name,
                        &format!("hot history column {:?}: {error}", field.name()),
                    )
                })
        })
        .collect::<Result<Vec<_>>>()?;
    RecordBatch::try_new_with_options(
        Arc::clone(schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(record.num_rows())),
    )
    .map_err(|error| operator_error(&operator.name, &format!("hot history: {error}")))
}

async fn emit_prepared(
    operator: &StreamJoinOperator,
    prepared: &PreparedRuns,
    context: &StreamOperatorContext<'_>,
    output: &mut dyn StreamCollector,
) -> Result<()> {
    #[cfg(feature = "benchmark-internals")]
    let emit_span =
        context
            .job()
            .join_cpu_trace()
            .stage(&operator.name, "run_emit", prepared.matched_rows());
    for item in prepared.outputs() {
        for (record, sequence) in item.records.iter().zip(&item.sequences) {
            context.check_cancelled()?;
            let record = super::hot::project_run_record(operator, record)?;
            let metadata = BatchMetadata::new(&operator.name, *sequence, BTreeMap::new())?;
            output
                .emit("output", Batch::table(vec![record], metadata)?)
                .await?;
        }
    }
    #[cfg(feature = "benchmark-internals")]
    drop(emit_span);
    Ok(())
}

fn route(
    admitted: &AdmittedBatch,
    key_indices: &[usize],
    keys: &mut KeyStore,
    check: &mut dyn FnMut() -> Result<()>,
    #[cfg(feature = "benchmark-internals")] visits: Option<&mut usize>,
) -> Result<Vec<Incoming>> {
    let mut remaining = admitted.iter();
    #[cfg(feature = "benchmark-internals")]
    {
        remaining = remaining.count_visits(visits);
    }
    (0..admitted.len().div_ceil(MORSEL_ROWS))
        .map(|_| {
            let rows = remaining.by_ref().take(MORSEL_ROWS).collect::<Vec<_>>();
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
                RouteInput::new(
                    &payload,
                    key_indices,
                    (0..rows.len()).collect(),
                    times,
                    row_ids,
                    retain,
                    next_row_id,
                ),
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
    let schema = first.record.schema_ref();
    let columns = first.record.columns();
    let first_offset = first.record.offset();
    let dense = rows.iter().enumerate().all(|(index, row)| {
        std::ptr::eq(columns, row.record.columns()) && row.record.offset() == first_offset + index
    });
    if dense
        && columns
            .iter()
            .zip(schema.fields())
            .all(|(column, field)| column.data_type() == field.data_type())
    {
        return RecordBatch::try_new_with_options(
            Arc::clone(schema),
            columns
                .iter()
                .map(|column| column.slice(first_offset, rows.len()))
                .collect(),
            &RecordBatchOptions::new().with_row_count(Some(rows.len())),
        )
        .map_err(|error| operator_error("Join runs", &format!("admitted morsel failed: {error}")));
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::join::PayloadRef;
    use datafusion::arrow::array::{Array, ArrayRef, Int64Array};

    #[cfg(feature = "benchmark-internals")]
    #[tokio::test]
    async fn run_route_trace_counts_admitted_row_visits_across_morsels() {
        use crate::{
            CancellationToken, EdgeCollector, JoinStateLimits, JoinTimeBounds, OperatorMetadata,
            StreamJobContext, StreamJoinSpec, StreamOperator,
        };
        use std::time::Duration;

        let rows = 2 * MORSEL_ROWS + 1;
        let record = |ordinals: Vec<i64>| {
            let len = ordinals.len();
            RecordBatch::try_from_iter([
                ("key", Arc::new(Int64Array::from(vec![1; len])) as ArrayRef),
                (
                    "ts",
                    Arc::new(TimestampMicrosecondArray::from(vec![0; len])) as ArrayRef,
                ),
                ("ordinal", Arc::new(Int64Array::from(ordinals)) as ArrayRef),
            ])
            .unwrap()
        };
        let ordinals = (0..i64::try_from(rows).unwrap()).rev().collect::<Vec<_>>();
        let left = record(ordinals.clone());
        let right = record(vec![42]);
        let spec = StreamJoinSpec::inner(
            ["key"],
            ["key"],
            "ts",
            "ts",
            JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
            JoinStateLimits::new(20_000, 64 << 20, 20_000).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("probe", left.schema(), right.schema(), spec).unwrap();
        operator.set_output_projection(vec![2, 5]).unwrap();
        operator.enable_run_core();
        let job =
            StreamJobContext::new(1, "trace", BTreeMap::new(), None, CancellationToken::new())
                .with_checkpointing(false);
        let trace = job.join_cpu_trace();
        trace.start(1, "trace").unwrap();
        trace.mark("ready_timer_start").unwrap();
        let context = StreamOperatorContext::new(&job, "probe", None);
        let mut output = EdgeCollector::new(operator.output_ports().to_vec());
        let split = MORSEL_ROWS - 1;
        for (side, records) in [
            ("right", vec![right]),
            ("left", vec![left.slice(0, 0)]),
            (
                "left",
                vec![left.slice(0, split), left.slice(split, rows - split)],
            ),
        ] {
            let batch = Batch::table(records, BatchMetadata::default()).unwrap();
            operator
                .process_data(side, batch, &context, &mut output)
                .await
                .unwrap();
        }
        trace.mark("ready_timer_end").unwrap();

        let messages = output.drain("output");
        let records = messages.iter().flat_map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
        });
        let schema = operator.output_ports()[0].schema().unwrap();
        let result = datafusion::arrow::compute::concat_batches(schema, records).unwrap();
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.column(0).as_ref(), &Int64Array::from(ordinals));
        assert_eq!(result.column(1).as_ref(), &Int64Array::from(vec![42; rows]));

        let capture = trace.take().unwrap();
        let visits = capture["events"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|event| event["stage"] == "run_route")
            .map(|event| {
                (
                    event["rows"].as_u64().unwrap(),
                    event["units"].as_u64().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let routed = u64::try_from(rows).unwrap();
        assert_eq!(visits, [(1, 1), (0, 0), (routed, routed)]);
    }

    #[test]
    fn morsel_shares_dense_source_buffers() {
        let record = RecordBatch::try_from_iter(vec![
            (
                "key",
                Arc::new(Int64Array::from(vec![10, 11, 12, 13])) as ArrayRef,
            ),
            (
                "value",
                Arc::new(Int64Array::from(vec![20, 21, 22, 23])) as ArrayRef,
            ),
        ])
        .unwrap();
        let rows = (1usize..3)
            .map(|row| AdmittedRef {
                record: PayloadRef::Record(&record, row),
                event_time: EventTime::from_micros(i64::try_from(row).unwrap()),
                row_id: u64::try_from(row).unwrap(),
                retain: true,
            })
            .collect::<Vec<_>>();
        let actual = morsel(&rows).unwrap();
        let expected = record
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        let actual = actual
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(actual.values(), &[11, 12]);
        assert_eq!(
            actual.to_data().buffers()[0].data_ptr(),
            expected.to_data().buffers()[0].data_ptr()
        );
    }
}
