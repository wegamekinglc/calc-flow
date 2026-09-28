//! Direct transitions for exclusive row-window mean plans such as SMA20.
//!
//! The loop applies the same `Float64NumericState` transition and mean
//! readout as the generic typed kernel, without per-row group/output dispatch.

use std::sync::Arc;

use super::{
    ArrayRef, Float64Array, Float64Builder, OutputStorage, RollingKernelPlan,
    RollingMetricsRecorder, RollingWork, Statistic, TypedEntityState, TypedFloatReadout,
    TypedFloatReadoutKind, TypedFrame, TypedGroupInput, TypedGroupPlan, TypedOutputKind,
    TypedOutputPlan, TypedRowInputs, TypedWindowState, float_mean, internal_error, operator_error,
    valid_float64,
};
use crate::Result;

/// One mean read with its output-specific minimum period.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MeanRead {
    group: usize,
    min_periods: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MeanReadout {
    Mean(MeanRead),
    Difference(MeanRead, MeanRead),
}

/// One or two row-window numeric groups read by one or two Float64 mean or
/// mean-difference outputs; every other plan uses the generic kernel.
#[derive(Debug, Eq, PartialEq)]
pub(super) struct MeanKernel {
    readouts: Vec<MeanReadout>,
}

impl MeanKernel {
    pub(super) fn compile(plan: &RollingKernelPlan) -> Option<Self> {
        let row_windows = plan.groups.iter().all(|group| {
            matches!(
                group,
                TypedGroupPlan::Numeric {
                    frame: TypedFrame::Rows(_),
                    ..
                }
            )
        });
        let arity = (1..=2).contains(&plan.groups.len()) && (1..=2).contains(&plan.outputs.len());
        if !row_windows || !arity {
            return None;
        }
        let groups = plan.groups.len();
        let readouts = plan
            .outputs
            .iter()
            .map(|output| mean_readout(output, groups))
            .collect::<Option<Vec<_>>>()?;
        Some(Self { readouts })
    }
}

fn mean_readout(output: &TypedOutputPlan, groups: usize) -> Option<MeanReadout> {
    if output.storage != OutputStorage::Float64 {
        return None;
    }
    match output.kind {
        TypedOutputKind::Statistic(Statistic::Mean) => {
            mean_read(output.group, output.min_periods, groups).map(MeanReadout::Mean)
        }
        TypedOutputKind::Difference { left, right } => Some(MeanReadout::Difference(
            float_mean_read(left, groups)?,
            float_mean_read(right, groups)?,
        )),
        _ => None,
    }
}

fn float_mean_read(readout: TypedFloatReadout, groups: usize) -> Option<MeanRead> {
    (readout.kind == TypedFloatReadoutKind::Mean)
        .then(|| mean_read(readout.group, readout.min_periods, groups))
        .flatten()
}

fn mean_read(group: usize, min_periods: u64, groups: usize) -> Option<MeanRead> {
    (group < groups).then_some(MeanRead { group, min_periods })
}

/// Applies one batch in row order and returns the Float64 output columns.
pub(super) fn fill_mean_rows(
    plan: &RollingKernelPlan,
    kernel: &MeanKernel,
    inputs: TypedRowInputs<'_>,
    states: &mut [Arc<TypedEntityState>],
    node_id: &str,
    observer: Option<&RollingMetricsRecorder>,
) -> Result<Vec<ArrayRef>> {
    let mut rows = MeanRows {
        plan,
        inputs,
        entities: touched_entities(states, inputs.entity_ids, observer),
        node_id,
        processed: 0,
    };
    let columns = rows.fill_arity(&kernel.readouts);
    if let Some(recorder) = observer {
        recorder.add(RollingWork::NumericRows, rows.processed);
    }
    columns
}

fn finish<const OUTPUTS: usize>(builders: [Float64Builder; OUTPUTS]) -> Vec<ArrayRef> {
    builders
        .into_iter()
        .map(|mut builder| Arc::new(builder.finish()) as ArrayRef)
        .collect()
}

fn single(column: &TypedGroupInput) -> Result<&Float64Array> {
    match column {
        TypedGroupInput::Single(values) => Ok(values),
        _ => Err(group_mismatch()),
    }
}

#[cold]
fn group_mismatch() -> crate::CalcFlowError {
    internal_error("typed rolling group state does not match its input plan")
}

/// Takes unique ownership of each touched entity once per batch; untouched
/// shared states stay shared.
fn touched_entities<'a>(
    states: &'a mut [Arc<TypedEntityState>],
    entity_ids: &[usize],
    observer: Option<&RollingMetricsRecorder>,
) -> Vec<Option<&'a mut TypedEntityState>> {
    let mut touched = vec![false; states.len()];
    for &entity_id in entity_ids {
        if let Some(slot) = touched.get_mut(entity_id) {
            *slot = true;
        }
    }
    let mut copied = 0;
    let entities = states
        .iter_mut()
        .zip(touched)
        .map(|(state, touched)| {
            touched.then(|| {
                copied += usize::from(Arc::strong_count(state) > 1);
                Arc::make_mut(state)
            })
        })
        .collect();
    if let Some(recorder) = observer {
        recorder.add(RollingWork::CopiedEntities, copied);
    }
    entities
}

struct MeanRows<'a> {
    plan: &'a RollingKernelPlan,
    inputs: TypedRowInputs<'a>,
    entities: Vec<Option<&'a mut TypedEntityState>>,
    node_id: &'a str,
    processed: usize,
}

impl MeanRows<'_> {
    fn fill_arity(&mut self, readouts: &[MeanReadout]) -> Result<Vec<ArrayRef>> {
        match (self.inputs.columns, readouts) {
            ([first], &[readout]) => self.fill([single(first)?], [readout]).map(finish),
            ([first], &[left, right]) => self.fill([single(first)?], [left, right]).map(finish),
            ([first, second], &[readout]) => self
                .fill([single(first)?, single(second)?], [readout])
                .map(finish),
            ([first, second], &[left, right]) => self
                .fill([single(first)?, single(second)?], [left, right])
                .map(finish),
            _ => Err(group_mismatch()),
        }
    }

    /// The generic kernel's per-row order: transition count, every group
    /// update, then every output readout.
    fn fill<const GROUPS: usize, const OUTPUTS: usize>(
        &mut self,
        values: [&Float64Array; GROUPS],
        readouts: [MeanReadout; OUTPUTS],
    ) -> Result<[Float64Builder; OUTPUTS]> {
        let (plan, inputs, node_id) = (self.plan, self.inputs, self.node_id);
        let (nan_as_value, profile) = (plan.nan_as_value, plan.numerical_profile);
        let entities = &mut self.entities;
        let processed = &mut self.processed;
        let rows = inputs.entity_ids.len();
        let mut builders = std::array::from_fn(|_| Float64Builder::with_capacity(rows));
        for (row, &entity_id) in inputs.entity_ids.iter().enumerate() {
            let Some(Some(entity)) = entities.get_mut(entity_id) else {
                return Err(internal_error("prepared rolling entity IDs are not dense"));
            };
            let Some(transitions) = entity.transition_count.checked_add(1) else {
                return Err(operator_error(
                    node_id,
                    "rolling entity transition count overflowed",
                ));
            };
            entity.transition_count = transitions;
            *processed += 1;
            let event_time = inputs.event_times.value(row);
            for (group, values) in entity.groups.iter_mut().zip(values) {
                let TypedWindowState::Numeric(state) = group else {
                    return Err(group_mismatch());
                };
                let sample = valid_float64(values, row, nan_as_value);
                state.update(event_time, sample, profile, transitions, node_id)?;
            }
            for (builder, readout) in builders.iter_mut().zip(readouts) {
                builder.append_option(read_readout(&entity.groups, readout)?);
            }
        }
        Ok(builders)
    }
}

fn read_readout(groups: &[TypedWindowState], readout: MeanReadout) -> Result<Option<f64>> {
    match readout {
        MeanReadout::Mean(read) => read_mean(groups, read),
        MeanReadout::Difference(left, right) => {
            let left = read_mean(groups, left)?;
            let right = read_mean(groups, right)?;
            Ok(left.zip(right).map(|(left, right)| left - right))
        }
    }
}

fn read_mean(groups: &[TypedWindowState], read: MeanRead) -> Result<Option<f64>> {
    let Some(TypedWindowState::Numeric(state)) = groups.get(read.group) else {
        return Err(internal_error("typed mean readout state mismatch"));
    };
    Ok((state.accumulator.valid_count >= read.min_periods).then(|| float_mean(&state.accumulator)))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::{Array, ArrayRef, Float64Array};

    use super::super::{
        DerivedBuilder, KernelComplexity, KernelSelection, OutputStorage,
        ROLLING_KERNEL_PLAN_VERSION, RollingKernelPlan, RollingNumericalProfile, Statistic,
        TimestampMicrosecondArray, TypedEntityState, TypedFloatReadout, TypedFloatReadoutKind,
        TypedFrame, TypedGroupInput, TypedGroupPlan, TypedOutputKind, TypedOutputPlan,
        TypedRowInputs, fill_typed_rows,
    };
    use super::{MeanKernel, fill_mean_rows};

    fn mean_output(group: usize, min_periods: u64) -> TypedOutputPlan {
        TypedOutputPlan {
            group,
            kind: TypedOutputKind::Statistic(Statistic::Mean),
            storage: OutputStorage::Float64,
            min_periods,
            ddof: 0,
        }
    }

    fn mean_readout(group: usize, min_periods: u64) -> TypedFloatReadout {
        TypedFloatReadout {
            group,
            kind: TypedFloatReadoutKind::Mean,
            min_periods,
            ddof: 0,
        }
    }

    fn difference(left: TypedFloatReadout, right: TypedFloatReadout) -> TypedOutputPlan {
        TypedOutputPlan {
            group: 0,
            kind: TypedOutputKind::Difference { left, right },
            storage: OutputStorage::Float64,
            min_periods: 0,
            ddof: 0,
        }
    }

    fn rows(input_index: usize, window: usize) -> TypedGroupPlan {
        TypedGroupPlan::Numeric {
            input_index,
            frame: TypedFrame::Rows(window),
        }
    }

    fn plan(groups: Vec<TypedGroupPlan>, outputs: Vec<TypedOutputPlan>) -> RollingKernelPlan {
        RollingKernelPlan {
            version: ROLLING_KERNEL_PLAN_VERSION,
            state_layout_version: 3,
            numerical_profile: RollingNumericalProfile::StableV1,
            selection: KernelSelection::OrderedPrimitive,
            complexity: KernelComplexity::AmortizedConstant,
            event_time_index: 0,
            order_columns: vec![0],
            partition_columns: Vec::new(),
            sequence_columns: Vec::new(),
            nan_as_value: false,
            groups,
            outputs,
            fallback_reason: None,
            estimated_state_bytes_per_entity: 0,
            fingerprint: "test-kernel".to_owned(),
        }
    }

    #[test]
    fn mean_kernel_compiles_only_exclusive_row_window_means() {
        let sma = plan(vec![rows(1, 20)], vec![mean_output(0, 20)]);
        assert!(MeanKernel::compile(&sma).is_some());
        let spread = plan(
            vec![rows(1, 5), rows(1, 20)],
            vec![difference(mean_readout(0, 5), mean_readout(1, 20))],
        );
        assert!(MeanKernel::compile(&spread).is_some());

        let count = TypedOutputPlan {
            kind: TypedOutputKind::Statistic(Statistic::Count),
            storage: OutputStorage::Count,
            ..mean_output(0, 1)
        };
        let variance = TypedOutputPlan {
            kind: TypedOutputKind::Statistic(Statistic::Variance),
            ..mean_output(0, 1)
        };
        let ewma = TypedFloatReadout {
            kind: TypedFloatReadoutKind::Ewma,
            ..mean_readout(0, 1)
        };
        let duration = TypedGroupPlan::Numeric {
            input_index: 1,
            frame: TypedFrame::Duration(10),
        };
        let rejected = [
            plan(vec![rows(1, 20)], vec![mean_output(0, 1), count]),
            plan(vec![rows(1, 20)], vec![variance]),
            plan(
                vec![rows(1, 20)],
                vec![difference(mean_readout(0, 1), ewma)],
            ),
            plan(vec![duration], vec![mean_output(0, 1)]),
            plan(vec![rows(1, 20)], vec![mean_output(1, 1)]),
            plan(vec![rows(1, 20)], vec![mean_output(0, 1); 3]),
            plan(
                vec![rows(1, 2), rows(1, 3), rows(1, 4)],
                vec![mean_output(0, 1)],
            ),
            plan(
                vec![TypedGroupPlan::Signed {
                    input_index: 1,
                    frame: TypedFrame::Rows(3),
                }],
                vec![mean_output(0, 1)],
            ),
        ];
        for plan in rejected {
            assert!(MeanKernel::compile(&plan).is_none(), "{:?}", plan.outputs);
        }
    }

    fn price(seed: u8) -> Option<f64> {
        match seed % 12 {
            0 => None,
            1 => Some(f64::NAN),
            2 => Some(f64::INFINITY),
            3 => Some(f64::NEG_INFINITY),
            4 => Some(1e155),
            5 => Some(-1e155),
            other => Some(f64::from(other) * 0.3 - 1.7),
        }
    }

    /// Output bits, entity state, and transition counts after one batch.
    type Observed = (Vec<Vec<Option<u64>>>, String);

    fn observe(columns: &[ArrayRef], states: &[Arc<TypedEntityState>]) -> Observed {
        let bits = columns
            .iter()
            .map(|column| {
                let values = column.as_any().downcast_ref::<Float64Array>().unwrap();
                (0..values.len())
                    .map(|row| values.is_valid(row).then(|| values.value(row).to_bits()))
                    .collect()
            })
            .collect();
        (bits, format!("{states:?}"))
    }

    fn generic(
        plan: &RollingKernelPlan,
        inputs: TypedRowInputs<'_>,
        mut states: Vec<Arc<TypedEntityState>>,
    ) -> Result<Observed, String> {
        let mut builders = plan
            .outputs
            .iter()
            .map(|output| DerivedBuilder::new(*output, inputs.entity_ids.len()))
            .collect::<Vec<_>>();
        fill_typed_rows(plan, inputs, &mut states, &mut builders, "r", None)
            .map_err(|error| error.to_string())?;
        let columns = builders
            .into_iter()
            .map(DerivedBuilder::finish)
            .collect::<crate::Result<Vec<_>>>()
            .unwrap();
        Ok(observe(&columns, &states))
    }

    fn compiled(
        plan: &RollingKernelPlan,
        inputs: TypedRowInputs<'_>,
        mut states: Vec<Arc<TypedEntityState>>,
    ) -> Result<Observed, String> {
        let kernel = MeanKernel::compile(plan).unwrap();
        let columns = fill_mean_rows(plan, &kernel, inputs, &mut states, "r", None)
            .map_err(|error| error.to_string())?;
        Ok(observe(&columns, &states))
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig {
            cases: 512,
            failure_persistence: None,
            ..proptest::prelude::ProptestConfig::default()
        })]

        #[test]
        fn compiled_mean_rows_match_the_generic_typed_kernel(
            windows in proptest::collection::vec(1_usize..6, 1..3),
            min_periods in proptest::collection::vec(0_u64..6, 2),
            spread in proptest::bool::ANY,
            nan_as_value in proptest::bool::weighted(0.2),
            stable_v2 in proptest::bool::weighted(0.2),
            warm in proptest::collection::vec((0_usize..3, 0_u8..=255), 0..12),
            batch in proptest::collection::vec((0_usize..4, 0_u8..=255), 0..24),
        ) {
            let groups = windows.iter().map(|&window| rows(1, window)).collect::<Vec<_>>();
            let last = groups.len() - 1;
            let outputs = if spread {
                vec![
                    difference(mean_readout(0, min_periods[0]), mean_readout(last, min_periods[1])),
                    mean_output(last, min_periods[1]),
                ]
            } else {
                vec![mean_output(0, min_periods[0])]
            };
            let mut plan = plan(groups, outputs);
            plan.nan_as_value = nan_as_value;
            if stable_v2 {
                plan.numerical_profile = RollingNumericalProfile::StableV2Preview;
            }
            let batch_inputs = |rows: &[(usize, u8)]| {
                let event_times = TimestampMicrosecondArray::from_iter_values(
                    (0..rows.len()).map(|row| i64::try_from(row).unwrap()),
                );
                let prices = rows.iter().map(|row| price(row.1)).collect::<Float64Array>();
                let columns = (0..plan.groups.len())
                    .map(|_| TypedGroupInput::Single(prices.clone()))
                    .collect::<Vec<_>>();
                let entity_ids = rows.iter().map(|row| row.0).collect::<Vec<_>>();
                (columns, event_times, entity_ids)
            };
            let fresh = (0..4)
                .map(|_| Arc::new(TypedEntityState::new(&plan.groups, 4)))
                .collect::<Vec<_>>();
            let (columns, event_times, entity_ids) = batch_inputs(&warm);
            let mut resident = fresh.clone();
            let mut builders = plan
                .outputs
                .iter()
                .map(|output| DerivedBuilder::new(*output, warm.len()))
                .collect::<Vec<_>>();
            let warm_inputs = TypedRowInputs {
                columns: &columns,
                event_times: &event_times,
                entity_ids: &entity_ids,
            };
            if fill_typed_rows(&plan, warm_inputs, &mut resident, &mut builders, "r", None).is_err() {
                resident = fresh;
            }
            let (columns, event_times, entity_ids) = batch_inputs(&batch);
            let inputs = TypedRowInputs {
                columns: &columns,
                event_times: &event_times,
                entity_ids: &entity_ids,
            };
            let expected = generic(&plan, inputs, resident.clone());
            let actual = compiled(&plan, inputs, resident);
            proptest::prop_assert_eq!(actual, expected);
        }
    }
}
