use super::controlled;
use crate::operator::rolling::{
    BTreeMap, BufferedRow, CompiledEvaluation, CompiledRollingOutput, CompiledRollingSpec,
    EntityRollingState, EntityRowView, HistoryUpdates, IncrementalScanState, KeyValue, Result,
    RollingHistories, RollingMetricsRecorder, RollingStage, RollingWork, RollingWorkBatch,
    ScalarValue, WindowState, compute_output_value, evict_retained_history, history,
    history_event_time, operator_error, slide_windows,
};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;
type Columns = Vec<Vec<Option<ScalarValue>>>;

#[derive(Debug)]
pub(in super::super) struct ComputedCells {
    pub columns: Columns,
    pub touched: HistoryUpdates,
}

pub(in super::super) fn compute(
    rows: &[BufferedRow],
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    node: &str,
    observer: Option<&RollingMetricsRecorder>,
    preparations: Option<&mut RollingWorkBatch<'_>>,
    check: Check<'_>,
) -> Result<ComputedCells> {
    Evaluation {
        rows,
        histories,
        compiled,
        node,
        observer,
    }
    .run(check, preparations)
}
struct Evaluation<'a> {
    rows: &'a [BufferedRow],
    histories: &'a RollingHistories,
    compiled: &'a CompiledRollingSpec,
    node: &'a str,
    observer: Option<&'a RollingMetricsRecorder>,
}
struct NumericState<'a> {
    count: &'a mut u64,
    windows: &'a mut [WindowState],
    indicators: &'a mut [super::State],
    scans: &'a mut [Option<IncrementalScanState>],
}
impl Evaluation<'_> {
    fn run(
        &self,
        check: Check<'_>,
        mut preparations: Option<&mut RollingWorkBatch<'_>>,
    ) -> Result<ComputedCells> {
        check()?;
        let mut columns = self.columns();
        let entities = self.groups(check)?;
        let mut touched = Vec::with_capacity(entities.len());
        for (index, (entity, indices)) in entities.into_iter().enumerate() {
            controlled::tick(index, check)?;
            let state = self.entity(entity, &indices, &mut columns, check)?;
            if let Some(work) = preparations.as_deref_mut() {
                history::observe_replacement(
                    indices.len(),
                    self.histories.by_entity.get(entity),
                    &state,
                    work,
                );
            }
            touched.push((entity.clone(), state));
        }
        check()?;
        Ok(ComputedCells { columns, touched })
    }
    fn columns(&self) -> Columns {
        let _stage = self
            .observer
            .map(|recorder| recorder.stage(RollingStage::ArrowOutput));
        self.compiled
            .outputs
            .iter()
            .map(|_| vec![None; self.rows.len()])
            .collect()
    }
    fn groups(&self, check: Check<'_>) -> Result<BTreeMap<&Vec<Option<KeyValue>>, Vec<usize>>> {
        let _stage = self
            .observer
            .map(|recorder| recorder.stage(RollingStage::EntityResolution));
        let mut entities: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for (index, row) in self.rows.iter().enumerate() {
            controlled::tick(index, check)?;
            entities
                .entry(&row.identity.entity)
                .or_default()
                .push(index);
        }
        if let Some(recorder) = self.observer {
            recorder.add(RollingWork::ResolvedRows, self.rows.len());
            recorder.add(RollingWork::TouchedEntities, entities.len());
        }
        Ok(entities)
    }
    fn entity(
        &self,
        entity: &Vec<Option<KeyValue>>,
        indices: &[usize],
        columns: &mut Columns,
        check: Check<'_>,
    ) -> Result<EntityRollingState> {
        let mut state = self.prior(entity, check)?;
        self.numeric(&mut state, indices, columns, check)?;
        self.history(&mut state, indices, check)?;
        Ok(state)
    }
    fn prior(
        &self,
        entity: &Vec<Option<KeyValue>>,
        check: Check<'_>,
    ) -> Result<EntityRollingState> {
        let _stage = self
            .observer
            .map(|recorder| recorder.stage(RollingStage::StatePreparation));
        let prior = self.histories.by_entity.get(entity);
        if let Some(recorder) = self.observer {
            recorder.add(RollingWork::CopiedEntities, usize::from(prior.is_some()));
        }
        match prior {
            Some(state) if !self.compiled.indicator_groups.is_empty() => {
                controlled::copy_entity(state, check)
            }
            Some(state) => Ok(state.clone()),
            None => Ok(EntityRollingState::fresh(self.compiled)),
        }
    }
    fn numeric(
        &self,
        state: &mut EntityRollingState,
        indices: &[usize],
        columns: &mut Columns,
        check: Check<'_>,
    ) -> Result<()> {
        let _stage = self
            .observer
            .map(|recorder| recorder.stage(RollingStage::NumericUpdate));
        let view = EntityRowView {
            rows: self.rows,
            indices,
            history: &state.rows,
            event_time_index: self.compiled.event_time_index,
        };
        let mut scans = self
            .compiled
            .outputs
            .iter()
            .map(|output| IncrementalScanState::from_history(output, &view))
            .collect::<Vec<_>>();
        let mut numeric = NumericState {
            count: &mut state.transition_count,
            windows: &mut state.windows,
            indicators: &mut state.indicators,
            scans: &mut scans,
        };
        for (position, &row) in indices.iter().enumerate() {
            controlled::tick(position, check)?;
            self.row(&view, (position, row), &mut numeric, columns, check)?;
        }
        Ok(())
    }
    fn row(
        &self,
        view: &EntityRowView<'_>,
        position: (usize, usize),
        state: &mut NumericState<'_>,
        columns: &mut Columns,
        check: Check<'_>,
    ) -> Result<()> {
        let (position, row) = position;
        let indicator_values = self.advance_numeric(view, (position, row), state)?;
        for (ordinal, output) in self.compiled.outputs.iter().enumerate() {
            controlled::tick(ordinal, check)?;
            columns[ordinal][row] = Some(self.readout(
                view,
                (position, row),
                (ordinal, output),
                state,
                &indicator_values,
                check,
            )?);
        }
        Ok(())
    }
    fn advance_numeric(
        &self,
        view: &EntityRowView<'_>,
        position: (usize, usize),
        state: &mut NumericState<'_>,
    ) -> Result<Vec<Option<f64>>> {
        let (position, row) = position;
        *state.count = state.count.checked_add(1).ok_or_else(|| {
            operator_error(self.node, "rolling entity transition count overflowed")
        })?;
        if let Some(recorder) = self.observer {
            recorder.add(RollingWork::NumericRows, 1);
        }
        slide_windows(
            view,
            position,
            row,
            *state.count,
            self.compiled,
            state.windows,
            self.node,
        )?;
        super::advance(
            &self.compiled.indicator_groups,
            state.indicators,
            &self.rows[row].values,
            self.node,
            *state.count - 1,
        )
    }
    fn readout(
        &self,
        view: &EntityRowView<'_>,
        (position, row): (usize, usize),
        (ordinal, output): (usize, &CompiledRollingOutput),
        state: &mut NumericState<'_>,
        indicators: &[Option<f64>],
        check: Check<'_>,
    ) -> Result<ScalarValue> {
        match output.evaluation {
            CompiledEvaluation::Indicator(group) => Ok(ScalarValue::Float64(indicators[group])),
            CompiledEvaluation::Lag { periods } if !self.compiled.indicator_groups.is_empty() => {
                lag(view, position, output, periods, self.node, check)
            }
            _ => self.ordinary_readout(view, (position, row), (ordinal, output), state),
        }
    }
    fn ordinary_readout(
        &self,
        view: &EntityRowView<'_>,
        (position, row): (usize, usize),
        (ordinal, output): (usize, &CompiledRollingOutput),
        state: &mut NumericState<'_>,
    ) -> Result<ScalarValue> {
        if let Some(scan) = &mut state.scans[ordinal] {
            Ok(scan.advance(
                view.history.len() + position,
                view.value(view.history.len() + position, output.input_index),
            ))
        } else {
            compute_output_value(view, position, row, output, state.windows, self.node)
        }
    }
    fn history(
        &self,
        state: &mut EntityRollingState,
        indices: &[usize],
        check: Check<'_>,
    ) -> Result<()> {
        let _stage = self
            .observer
            .map(|recorder| recorder.stage(RollingStage::HistoryMaintenance));
        for (position, &row) in indices.iter().enumerate() {
            controlled::tick(position, check)?;
            state.rows.push_back(self.rows[row].values.clone());
        }
        if self.compiled.indicator_groups.is_empty() {
            evict_retained_history(state, self.compiled);
        } else {
            evict_checked(state, self.compiled, check)?;
        }
        Ok(())
    }
}
fn lag(
    view: &EntityRowView<'_>,
    position: usize,
    output: &CompiledRollingOutput,
    periods: u64,
    node: &str,
    check: Check<'_>,
) -> Result<ScalarValue> {
    let periods = usize::try_from(periods)
        .map_err(|_| operator_error(node, "rolling periods does not fit usize"))?;
    let combined = view
        .history
        .len()
        .checked_add(position)
        .ok_or_else(|| operator_error(node, "rolling lag position overflowed"))?;
    if combined < periods {
        return super::materialize::canonical::null_scalar(&output.input_type, node, check);
    }
    Ok(view.value(combined - periods, output.input_index).clone())
}

fn evict_checked(
    state: &mut EntityRollingState,
    compiled: &CompiledRollingSpec,
    check: Check<'_>,
) -> Result<()> {
    let rows = usize::try_from(compiled.max_row_retention).unwrap_or(usize::MAX);
    let bound = compiled
        .max_duration_micros
        .zip(
            state
                .rows
                .back()
                .map(|values| history_event_time(values, compiled)),
        )
        .map(|(micros, last)| i128::from(last) - i128::from(micros));
    let initial = state.rows.len();
    while state.rows.len() > rows {
        controlled::tick(initial - state.rows.len(), check)?;
        if bound.is_some_and(|bound| {
            state
                .rows
                .front()
                .is_some_and(|values| i128::from(history_event_time(values, compiled)) > bound)
        }) {
            break;
        }
        state.rows.pop_front();
    }
    Ok(())
}
