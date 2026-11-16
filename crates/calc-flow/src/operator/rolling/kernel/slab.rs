//! State-table ownership for typed rolling stream transitions.

use std::{sync::Arc, time::Instant};

use datafusion::arrow::array::{ArrayRef, Float64Builder, UInt64Builder};
use datafusion::arrow::record_batch::RecordBatch;

use super::{
    OutputStorage, Result, RollingKernelExecution, RollingKernelMetrics, RollingKernelPlan,
    RollingKernelState, RollingMetricsRecorder, RollingStage, RollingWork, StateCharge, Statistic,
    StreamKernelUpdate, TypedFrame, TypedGroupInput, TypedGroupPlan, TypedOutputKind,
    TypedOutputPlan, internal_error, nanos, operator_error, valid_float64,
};
use crate::state::{EntityId, EntityKey, StateTable, ValueKind};

pub(super) struct SlabUpdate {
    previous_charge: StateCharge,
    entities: usize,
    created: bool,
}

impl SlabUpdate {
    pub(in crate::operator::rolling) fn recurrence_seeds(
        &self,
    ) -> Vec<Vec<Option<super::RecurrenceSeed>>> {
        vec![vec![None]; self.entities]
    }

    pub(in crate::operator::rolling) fn charge_change(
        &self,
        base: &RollingKernelState,
        node_id: &str,
    ) -> Result<(StateCharge, StateCharge)> {
        let new = charge(
            base.slab
                .as_ref()
                .ok_or_else(|| internal_error("rolling slab is absent"))?,
            node_id,
        )?;
        Ok((self.previous_charge, new))
    }

    pub(in crate::operator::rolling) fn rollback(self, base: &mut RollingKernelState) {
        base.slab
            .as_mut()
            .expect("rolling slab is present")
            .rollback()
            .expect("an active rolling slab transaction rolls back");
        if self.created {
            base.slab = None;
        }
    }

    #[allow(
        clippy::unused_self,
        reason = "consuming the update enforces that each slab transaction commits once"
    )]
    pub(in crate::operator::rolling) fn commit(self, base: &mut RollingKernelState) {
        base.slab
            .as_mut()
            .expect("rolling slab is present")
            .commit()
            .expect("an active rolling slab transaction commits");
    }
}

fn charge(table: &StateTable, node_id: &str) -> Result<StateCharge> {
    Ok(StateCharge {
        rows: u64::try_from(table.logical_rows())
            .map_err(|_| operator_error(node_id, "rolling slab row charge overflowed"))?,
        bytes: u64::try_from(table.logical_bytes())
            .map_err(|_| operator_error(node_id, "rolling slab byte charge overflowed"))?,
    })
}

enum SlabBuilder {
    Float(Float64Builder),
    Count(UInt64Builder),
}

impl SlabBuilder {
    fn new(output: &TypedOutputPlan, rows: usize) -> Result<Self> {
        match output.storage {
            OutputStorage::Float64 => Ok(Self::Float(Float64Builder::with_capacity(rows))),
            OutputStorage::Count => Ok(Self::Count(UInt64Builder::with_capacity(rows))),
            _ => Err(internal_error("rolling slab output storage mismatch")),
        }
    }

    fn append(
        &mut self,
        table: &StateTable,
        entity: EntityId,
        output: &TypedOutputPlan,
    ) -> Result<()> {
        let valid = table.numeric_valid_count(0, entity)?;
        if valid < output.min_periods {
            match self {
                Self::Float(builder) => builder.append_null(),
                Self::Count(builder) => builder.append_null(),
            }
            return Ok(());
        }
        match (self, output.kind) {
            (Self::Count(builder), TypedOutputKind::Statistic(Statistic::Count)) => {
                builder.append_value(valid);
            }
            (Self::Float(builder), TypedOutputKind::Statistic(Statistic::Sum)) => {
                builder.append_value(table.numeric_sum(0, entity)?);
            }
            (Self::Float(builder), TypedOutputKind::Statistic(Statistic::Mean)) => {
                builder.append_option(table.numeric_mean(0, entity)?);
            }
            (Self::Float(builder), TypedOutputKind::Statistic(Statistic::Variance)) => {
                builder.append_value(table.numeric_variance(0, entity, output.ddof)?);
            }
            (Self::Float(builder), TypedOutputKind::Statistic(Statistic::Stddev)) => {
                builder.append_value(table.numeric_stddev(0, entity, output.ddof)?);
            }
            _ => return Err(internal_error("rolling slab output kind mismatch")),
        }
        Ok(())
    }

    fn finish(self) -> ArrayRef {
        match self {
            Self::Float(mut builder) => Arc::new(builder.finish()),
            Self::Count(mut builder) => Arc::new(builder.finish()),
        }
    }
}

impl RollingKernelPlan {
    pub(in crate::operator::rolling) fn supports_slab(&self) -> bool {
        matches!(
            self.groups.as_slice(),
            [TypedGroupPlan::Numeric {
                frame: TypedFrame::Rows(_),
                ..
            }]
        ) && !self.outputs.is_empty()
            && self.outputs.iter().all(|output| {
                output.group == 0
                    && match output.kind {
                        TypedOutputKind::Statistic(Statistic::Count) => {
                            output.storage == OutputStorage::Count
                        }
                        TypedOutputKind::Statistic(
                            Statistic::Sum
                            | Statistic::Mean
                            | Statistic::Variance
                            | Statistic::Stddev,
                        ) => output.storage == OutputStorage::Float64,
                        _ => false,
                    }
            })
    }

    pub(in crate::operator::rolling) fn can_use_slab(&self, state: &RollingKernelState) -> bool {
        self.supports_slab() && (state.slab.is_some() || state.states.is_empty())
    }

    fn ensure_slab(&self, state: &mut RollingKernelState) -> Result<()> {
        if state.slab.is_some() {
            return Ok(());
        }
        let [
            TypedGroupPlan::Numeric {
                frame: TypedFrame::Rows(window),
                ..
            },
        ] = self.groups.as_slice()
        else {
            return Err(internal_error(
                "rolling slab plan is not one Float64 Rows group",
            ));
        };
        let mut table = StateTable::new();
        table.add_keyed_column("transition_count", ValueKind::UInt64);
        table.add_numeric_rows_column("numeric", *window);
        state.slab = Some(table);
        Ok(())
    }

    pub(in crate::operator::rolling) fn encode_slab_state(
        &self,
        state: &RollingKernelState,
    ) -> Result<Vec<u8>> {
        let table = state
            .slab
            .as_ref()
            .ok_or_else(|| internal_error("rolling layout-7 state is absent"))?;
        crate::state::encode_layout7(table)
    }

    pub(in crate::operator::rolling) fn decode_slab_state(
        &self,
        bytes: &[u8],
        last_identity: Option<Vec<u8>>,
    ) -> Result<RollingKernelState> {
        let mut state = RollingKernelState::default();
        self.ensure_slab(&mut state)?;
        let table = state.slab.as_mut().expect("slab initialized");
        let snapshot = crate::state::decode_layout7(table, bytes)?;
        table.restore(snapshot)?;
        state.kernel_fingerprint = Some(self.fingerprint.clone());
        state.last_identity = last_identity;
        Ok(state)
    }

    pub(in crate::operator::rolling) fn slab_charge(
        &self,
        state: &RollingKernelState,
        node_id: &str,
    ) -> Result<Option<StateCharge>> {
        state
            .slab
            .as_ref()
            .map(|table| charge(table, node_id))
            .transpose()
    }

    pub(in crate::operator::rolling) fn slab_last_identity(
        state: &RollingKernelState,
    ) -> Option<Vec<u8>> {
        state.slab.as_ref()?;
        state.last_identity.clone()
    }

    pub(in crate::operator::rolling) fn prepare_slab_ordered_stream(
        &self,
        state: &mut RollingKernelState,
        input: &RecordBatch,
        node_id: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<StreamKernelUpdate> {
        self.validate_state(state, node_id)?;
        let created = state.slab.is_none();
        self.ensure_slab(state)?;
        let columns = self.typed_inputs(input, node_id)?;
        let [TypedGroupInput::Single(values)] = columns.as_slice() else {
            return Err(internal_error(
                "rolling slab input is not one Float64 column",
            ));
        };
        let last_identity = self.last_ordered_identity(input, node_id)?;
        let started = Instant::now();
        let routing = self.encoded_entity_routing(
            input,
            state.slab.as_ref().map_or(0, StateTable::entity_count),
            node_id,
        )?;
        let entity_encode_ns = nanos(started.elapsed());
        let previous_charge = charge(state.slab.as_ref().expect("slab initialized"), node_id)?;
        state.slab.as_mut().expect("slab initialized").begin()?;
        let entities = routing.keys.len();
        let entity_rows = routing.rows.clone();
        let local_ids = routing.entity_ids.clone();
        let result = (|| {
            let table = state.slab.as_mut().expect("slab initialized");
            let resolved = routing
                .keys
                .iter()
                .map(|key| table.resolve_or_append(EntityKey::new(key.clone())))
                .collect::<Result<Vec<EntityId>>>()?;
            let ids = routing
                .entity_ids
                .iter()
                .map(|&local| resolved[local])
                .collect::<Vec<_>>();
            let mut builders = self
                .outputs
                .iter()
                .map(|output| SlabBuilder::new(output, input.num_rows()))
                .collect::<Result<Vec<_>>>()?;
            let _stage = observer.map(|recorder| recorder.stage(RollingStage::NumericUpdate));
            for (row, &entity) in ids.iter().enumerate() {
                let transitions = table.u64(0, entity)?.checked_add(1).ok_or_else(|| {
                    operator_error(node_id, "rolling entity transition count overflowed")
                })?;
                table.set_u64(0, entity, transitions)?;
                table.push_numeric_rows(
                    0,
                    entity,
                    valid_float64(values, row, self.nan_as_value),
                )?;
                for (builder, output) in builders.iter_mut().zip(&self.outputs) {
                    builder.append(table, entity, output)?;
                }
            }
            if let Some(recorder) = observer {
                recorder.add(RollingWork::NumericRows, input.num_rows());
                recorder.add(RollingWork::NumericSerialCalls, 1);
            }
            Ok(builders
                .into_iter()
                .map(SlabBuilder::finish)
                .collect::<Vec<_>>())
        })();
        let columns = match result {
            Ok(columns) => columns,
            Err(error) => {
                state.slab.as_mut().expect("slab initialized").rollback()?;
                if created {
                    state.slab = None;
                }
                return Err(error);
            }
        };
        let local = RollingKernelState {
            kernel_fingerprint: Some(self.fingerprint.clone()),
            last_identity,
            ..RollingKernelState::default()
        };
        Ok(StreamKernelUpdate {
            execution: RollingKernelExecution {
                columns,
                entity_ids: local_ids,
                entity_rows,
                metrics: RollingKernelMetrics {
                    entity_encode_ns,
                    entities,
                    input_rows: input.num_rows(),
                    output_rows: input.num_rows(),
                    state_bytes: state.slab.as_ref().map_or(0, StateTable::logical_bytes),
                    ..RollingKernelMetrics::default()
                },
                state: local,
            },
            in_place: None,
            slab: Some(SlabUpdate {
                previous_charge,
                entities,
                created,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::{
        array::{Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    };

    use super::super::{
        KernelComplexity, KernelSelection, OutputStorage, ROLLING_KERNEL_PLAN_VERSION,
        RollingKernelPlan, RollingKernelState, Statistic, TypedFrame, TypedGroupPlan,
        TypedOutputKind, TypedOutputPlan,
    };

    fn plan() -> RollingKernelPlan {
        RollingKernelPlan {
            version: ROLLING_KERNEL_PLAN_VERSION,
            state_layout_version: 7,
            selection: KernelSelection::OrderedPrimitive,
            complexity: KernelComplexity::AmortizedConstant,
            event_time_index: 0,
            order_columns: vec![0, 1],
            partition_columns: vec![1],
            sequence_columns: Vec::new(),
            nan_as_value: false,
            shared_rows: None,
            groups: vec![TypedGroupPlan::Numeric {
                input_index: 2,
                frame: TypedFrame::Rows(2),
                dispersion: true,
            }],
            outputs: [
                (Statistic::Mean, OutputStorage::Float64),
                (Statistic::Sum, OutputStorage::Float64),
                (Statistic::Count, OutputStorage::Count),
                (Statistic::Variance, OutputStorage::Float64),
            ]
            .into_iter()
            .map(|(statistic, storage)| TypedOutputPlan {
                group: 0,
                kind: TypedOutputKind::Statistic(statistic),
                storage,
                min_periods: 1,
                ddof: 0,
            })
            .collect(),
            fallback_reason: None,
            estimated_state_bytes_per_entity: 0,
            fingerprint: "slab-test".to_owned(),
            checkpoint_fingerprint: None,
            decay_pool: None,
        }
    }

    fn input(times: &[i64], values: &[Option<f64>]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Float64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(TimestampMicrosecondArray::from_iter_values(
                    times.iter().copied(),
                )),
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    "a",
                    times.len(),
                ))),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        )
        .unwrap()
    }

    #[test]
    fn kernel_slab_commit_and_rollback_preserve_aggregates_and_history() {
        let plan = plan();
        let mut state = RollingKernelState::default();
        let empty = state.canonical_state_for_test();
        let first = input(&[1, 2], &[Some(1.0), Some(3.0)]);
        let mut update = plan
            .prepare_slab_ordered_stream(&mut state, &first, "r", None)
            .unwrap();
        let columns = update.take_columns();
        let mean = columns[0].as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(mean.value(0).to_bits(), 1.0_f64.to_bits());
        assert_eq!(mean.value(1).to_bits(), 2.0_f64.to_bits());
        let sum = columns[1].as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(sum.value(1).to_bits(), 4.0_f64.to_bits());
        let count = columns[2].as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(count.value(1), 2);
        let variance = columns[3].as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(variance.value(1).to_bits(), 1.0_f64.to_bits());
        update.rollback(&mut state);
        assert_eq!(state.canonical_state_for_test(), empty);

        let update = plan
            .prepare_slab_ordered_stream(&mut state, &first, "r", None)
            .unwrap();
        update.commit(&mut state);
        let committed = state.canonical_state_for_test();
        let second = input(&[3], &[Some(5.0)]);
        let mut update = plan
            .prepare_slab_ordered_stream(&mut state, &second, "r", None)
            .unwrap();
        let columns = update.take_columns();
        let mean = columns[0].as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(mean.value(0).to_bits(), 4.0_f64.to_bits());
        update.rollback(&mut state);
        assert_eq!(state.canonical_state_for_test(), committed);
    }
}
