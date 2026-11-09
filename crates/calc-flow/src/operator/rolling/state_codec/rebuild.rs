type RestoreCheck<'a> = &'a mut dyn FnMut() -> Result<()>;

pub(super) fn rebuild_windows(
    histories: &mut RollingHistories,
    compiled: &CompiledRollingSpec,
    node: &str,
) -> Result<()> {
    rebuild_windows_checked(histories, compiled, node, &mut || Ok(()))
}
pub(super) fn rebuild_windows_checked(
    histories: &mut RollingHistories,
    compiled: &CompiledRollingSpec,
    node: &str,
    check: RestoreCheck<'_>,
) -> Result<()> {
    for state in histories.by_entity.values_mut() {
        check()?;
        rebuild_entity(state, compiled, node, check)?;
    }
    check()
}
struct Rebuild<'a> {
    state: &'a EntityRollingState,
    compiled: &'a CompiledRollingSpec,
    node: &'a str,
    last_time: Option<i64>,
}
fn rebuild_entity(
    state: &mut EntityRollingState,
    compiled: &CompiledRollingSpec,
    node: &str,
    check: RestoreCheck<'_>,
) -> Result<()> {
    let persisted = std::mem::take(&mut state.windows);
    let mut windows = fresh_windows(compiled);
    let context = Rebuild {
        state,
        compiled,
        node,
        last_time: state
            .rows
            .back()
            .map(|values| history_event_time(values, compiled)),
    };
    for (index, group) in compiled.window_groups.iter().enumerate() {
        check()?;
        rebuild_group(
            &context,
            group,
            index,
            &mut windows[index],
            &persisted,
            check,
        )?;
    }
    state.windows = windows;
    Ok(())
}
fn rebuild_group(
    context: &Rebuild<'_>,
    group: &CompiledWindowGroup,
    index: usize,
    window: &mut WindowState,
    persisted: &[WindowState],
    check: RestoreCheck<'_>,
) -> Result<()> {
    match group {
        CompiledWindowGroup::Numeric {
            input_index,
            frame,
            sum_class,
        } => rebuild_numeric(context, window, *input_index, *frame, *sum_class, check),
        CompiledWindowGroup::Pair {
            left_index,
            right_index,
            frame,
        } => rebuild_pair(context, window, (*left_index, *right_index), *frame, check),
        CompiledWindowGroup::Extrema {
            input_index, frame, ..
        } => rebuild_extrema(context, window, *input_index, *frame, check),
        _ => {
            restore_saved(group, window, persisted.get(index));
            Ok(())
        }
    }
}
fn restore_saved(
    group: &CompiledWindowGroup,
    window: &mut WindowState,
    saved: Option<&WindowState>,
) {
    match (group, saved) {
        (CompiledWindowGroup::Cumulative { .. }, Some(state @ WindowState::Cumulative(_)))
        | (CompiledWindowGroup::Ewma { .. }, Some(state @ WindowState::Ewma(_))) => {
            *window = state.clone();
        }
        _ => {}
    }
}
fn rebuild_numeric(
    context: &Rebuild<'_>,
    window: &mut WindowState,
    index: usize,
    frame: CompiledFrame,
    class: SumClass,
    check: RestoreCheck<'_>,
) -> Result<()> {
    let WindowState::Numeric(accumulator) = window else {
        return Err(internal_error("rolling numeric group state mismatch"));
    };
    let start = retained_window_start(frame, context.state, context.compiled, check)?;
    if spec_uses_stable_v2(context.compiled) && class == SumClass::Float {
        *accumulator = stable_numeric(context, index, start, check)?;
    } else {
        numeric_rows(context, index, start, accumulator, check)?;
    }
    numeric_expiry(accumulator, frame, context.last_time);
    Ok(())
}
fn numeric_expiry(
    accumulator: &mut super::WindowAccumulator,
    frame: CompiledFrame,
    last_time: Option<i64>,
) {
    if let CompiledFrame::Duration(micros) = frame {
        accumulator.expired_through = expired_through_bound(last_time, micros);
    }
}

fn numeric_rows(
    context: &Rebuild<'_>,
    index: usize,
    start: usize,
    accumulator: &mut super::WindowAccumulator,
    check: RestoreCheck<'_>,
) -> Result<()> {
    for values in context.state.rows.iter().skip(start) {
        check()?;
        let value = &values[index];
        if is_valid_sample(value) {
            accumulator.add(value, context.node)?;
        }
    }
    Ok(())
}
fn stable_numeric(
    context: &Rebuild<'_>,
    index: usize,
    start: usize,
    check: RestoreCheck<'_>,
) -> Result<super::WindowAccumulator> {
    rebuild_control::fold(context.state.rows.iter().skip(start), check, |rows| {
        kernel::stable_v2_float64_accumulator(
            rows.filter_map(|values| {
                let value = &values[index];
                is_valid_sample(value).then(|| float_sample(value))
            }),
            context.node,
        )
    })
}
fn rebuild_pair(
    context: &Rebuild<'_>,
    window: &mut WindowState,
    indices: (usize, usize),
    frame: CompiledFrame,
    check: RestoreCheck<'_>,
) -> Result<()> {
    let WindowState::Pair(accumulator) = window else {
        return Err(internal_error("rolling pair group state mismatch"));
    };
    let start = retained_window_start(frame, context.state, context.compiled, check)?;
    if spec_uses_stable_v2(context.compiled) {
        *accumulator = stable_pair(context, indices, start, check)?;
    } else {
        pair_rows(context, indices, start, accumulator, check)?;
    }
    if let CompiledFrame::Duration(micros) = frame {
        accumulator.expired_through = expired_through_bound(context.last_time, micros);
    }
    Ok(())
}
fn pair_rows(
    context: &Rebuild<'_>,
    indices: (usize, usize),
    start: usize,
    accumulator: &mut super::PairAccumulator,
    check: RestoreCheck<'_>,
) -> Result<()> {
    for values in context.state.rows.iter().skip(start) {
        check()?;
        let x = &values[indices.0];
        let y = &values[indices.1];
        if is_valid_sample(x) && is_valid_sample(y) {
            accumulator.add(x, y, context.node)?;
        }
    }
    Ok(())
}
fn stable_pair(
    context: &Rebuild<'_>,
    indices: (usize, usize),
    start: usize,
    check: RestoreCheck<'_>,
) -> Result<super::PairAccumulator> {
    rebuild_control::fold(context.state.rows.iter().skip(start), check, |rows| {
        kernel::stable_v2_pair_accumulator(
            rows.filter_map(|values| pair_sample(values, indices)),
            context.node,
        )
    })
}
fn pair_sample(values: &[ScalarValue], indices: (usize, usize)) -> Option<(f64, f64)> {
    let x = &values[indices.0];
    let y = &values[indices.1];
    (is_valid_sample(x) && is_valid_sample(y)).then(|| (float_sample(x), float_sample(y)))
}
fn rebuild_extrema(
    context: &Rebuild<'_>,
    window: &mut WindowState,
    index: usize,
    frame: CompiledFrame,
    check: RestoreCheck<'_>,
) -> Result<()> {
    let WindowState::Extrema(accumulator) = window else {
        return Err(internal_error("rolling extrema group state mismatch"));
    };
    let rows = usize::try_from(frame.rows())
        .map_err(|_| internal_error("rolling frame rows do not fit usize"))?;
    let mut cursor = 0;
    for position in 0..context.state.rows.len() {
        check()?;
        rebuild_extrema_row(context, accumulator, index, position, check)?;
        expire_extrema(
            context,
            accumulator,
            (index, position, rows),
            frame,
            &mut cursor,
            check,
        )?;
    }
    Ok(())
}
fn rebuild_extrema_row(
    context: &Rebuild<'_>,
    accumulator: &mut ExtremaAccumulator,
    index: usize,
    position: usize,
    check: RestoreCheck<'_>,
) -> Result<()> {
    let values = &context.state.rows[position];
    let key = history_extrema_key(values, context.compiled, context.node, check)?;
    let value = &values[index];
    if is_valid_sample(value) {
        accumulator.add_checked(key.clone(), value.clone(), check)?;
    }
    Ok(())
}
fn expire_extrema(
    context: &Rebuild<'_>,
    accumulator: &mut ExtremaAccumulator,
    indices: (usize, usize, usize),
    frame: CompiledFrame,
    cursor: &mut usize,
    check: RestoreCheck<'_>,
) -> Result<()> {
    match frame {
        CompiledFrame::Rows(_) => expire_rows(context, accumulator, indices, check),
        CompiledFrame::Duration(micros) => expire_duration(
            context,
            accumulator,
            indices.0,
            indices.1,
            micros,
            cursor,
            check,
        ),
    }
}
fn expire_rows(
    context: &Rebuild<'_>,
    accumulator: &mut ExtremaAccumulator,
    (index, position, rows): (usize, usize, usize),
    check: RestoreCheck<'_>,
) -> Result<()> {
    if position >= rows {
        let leaving = &context.state.rows[position - rows];
        if is_valid_sample(&leaving[index]) {
            accumulator.remove();
        }
        let key = history_extrema_key(leaving, context.compiled, context.node, check)?;
        accumulator.expire_through_key_checked(&key, check)?;
    }
    Ok(())
}
fn expire_duration(
    context: &Rebuild<'_>,
    accumulator: &mut ExtremaAccumulator,
    index: usize,
    position: usize,
    micros: u64,
    cursor: &mut usize,
    check: RestoreCheck<'_>,
) -> Result<()> {
    let time = history_event_time(&context.state.rows[position], context.compiled);
    let bound = i128::from(time) - i128::from(micros);
    while *cursor < position
        && i128::from(history_event_time(
            &context.state.rows[*cursor],
            context.compiled,
        )) <= bound
    {
        check()?;
        if is_valid_sample(&context.state.rows[*cursor][index]) {
            accumulator.remove();
        }
        *cursor += 1;
    }
    accumulator.expire_through_time_checked(bound, check)?;
    accumulator.expired_through = bound;
    Ok(())
}
fn expired_through_bound(last_time: Option<i64>, micros: u64) -> i128 {
    last_time.map_or(i128::MIN, |last| i128::from(last) - i128::from(micros))
}
fn history_extrema_key(
    values: &[ScalarValue],
    compiled: &CompiledRollingSpec,
    node: &str,
    check: RestoreCheck<'_>,
) -> Result<ExtremaKey> {
    let sequence = compiled
        .sequence_columns
        .iter()
        .map(|column| {
            check()?;
            KeyValue::from_required_scalar(&values[column.index], node)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ExtremaKey {
        event_time: history_event_time(values, compiled),
        sequence,
    })
}
fn retained_window_start(
    frame: CompiledFrame,
    state: &EntityRollingState,
    compiled: &CompiledRollingSpec,
    check: RestoreCheck<'_>,
) -> Result<usize> {
    let length = state.rows.len();
    match frame {
        CompiledFrame::Rows(rows) => {
            Ok(length.saturating_sub(usize::try_from(rows).unwrap_or(usize::MAX)))
        }
        CompiledFrame::Duration(micros) => duration_start(state, compiled, micros, check),
    }
}
fn duration_start(
    state: &EntityRollingState,
    compiled: &CompiledRollingSpec,
    micros: u64,
    check: RestoreCheck<'_>,
) -> Result<usize> {
    let Some(last) = state.rows.back() else {
        return Ok(state.rows.len());
    };
    let bound = i128::from(history_event_time(last, compiled)) - i128::from(micros);
    for (index, values) in state.rows.iter().enumerate() {
        check()?;
        if i128::from(history_event_time(values, compiled)) > bound {
            return Ok(index);
        }
    }
    Ok(state.rows.len())
}
