//! One timed sample history for same-column deferred floating-point windows.

use std::{collections::VecDeque, mem::size_of};

use super::{
    Float64NumericState, TimedSample, TypedFrame, TypedGroupInput, TypedGroupPlan,
    TypedWindowState, WindowAccumulator, add_float64, float_arithmetic_error,
    in_place::EvictedSamples, internal_error, remove_float64, valid_float64,
};
use crate::Result;

#[derive(Clone, Debug)]
pub(super) struct DurationSamples {
    values: VecDeque<TimedSample<f64>>,
    heads: Box<[usize]>,
}

pub(super) fn eligible(groups: &[TypedGroupPlan]) -> bool {
    let [
        TypedGroupPlan::Numeric {
            input_index: first,
            frame: TypedFrame::Duration(first_width),
            ..
        },
        rest @ ..,
    ] = groups
    else {
        return false;
    };
    rest.iter().all(|plan| {
        matches!(plan, TypedGroupPlan::Numeric {
        input_index, frame: TypedFrame::Duration(_), ..
    } if input_index == first)
    }) && rest.iter().any(|plan| {
        matches!(plan,
        TypedGroupPlan::Numeric { frame: TypedFrame::Duration(width), .. } if width != first_width)
    })
}

impl DurationSamples {
    pub(super) fn new(groups: usize) -> Box<Self> {
        Box::new(Self {
            values: VecDeque::new(),
            heads: vec![0; groups].into_boxed_slice(),
        })
    }

    pub(super) fn len(&self) -> usize {
        self.values.len()
    }

    pub(super) fn bytes(&self) -> usize {
        size_of::<Self>()
            + size_of_val(&*self.heads)
            + self.values.capacity() * size_of::<TimedSample<f64>>()
    }

    pub(super) fn logical_work(&self) -> Option<usize> {
        self.heads.iter().try_fold(0_usize, |work, &head| {
            work.checked_add(self.len().checked_sub(head)?)
        })
    }

    pub(super) fn advance(
        &mut self,
        group: &mut TypedWindowState,
        ordinal: usize,
        sample: TimedSample<f64>,
        node: &str,
    ) -> Result<()> {
        let TypedWindowState::Numeric(state) = group else {
            return Err(internal_error(
                "shared Duration state requires a numeric accumulator",
            ));
        };
        let TypedFrame::Duration(micros) = state.samples.frame else {
            return Err(internal_error(
                "shared Duration state requires a time frame",
            ));
        };
        let bound = i128::from(sample.event_time) - i128::from(micros);
        let head = &mut self.heads[ordinal];
        while self
            .values
            .get(*head)
            .is_some_and(|row| i128::from(row.event_time) <= bound)
        {
            if let Some(value) = self.values[*head].value {
                remove_float64(&mut state.accumulator, value)?;
            }
            *head += 1;
        }
        finish(
            state,
            sample.value,
            self.values
                .range(*head..)
                .map(|row| row.value)
                .chain(std::iter::once(sample.value)),
            node,
        )
    }

    pub(super) fn commit(
        &mut self,
        sample: TimedSample<f64>,
        mut removed: impl FnMut(TimedSample<f64>),
    ) {
        let count = self.heads.iter().copied().min().unwrap_or(0);
        if count > 0 {
            self.values.drain(..count).for_each(&mut removed);
            for head in &mut self.heads {
                *head -= count;
            }
        }
        self.values.push_back(sample);
    }

    pub(super) fn update(
        &mut self,
        groups: &mut [TypedWindowState],
        input: &TypedGroupInput,
        row: usize,
        event_time: i64,
        nan_as_value: bool,
        node: &str,
    ) -> Result<TimedSample<f64>> {
        let TypedGroupInput::Single(column) = input else {
            return Err(internal_error(
                "shared Duration state requires a floating-point column",
            ));
        };
        let sample = TimedSample {
            event_time,
            value: valid_float64(column, row, nan_as_value),
        };
        for (ordinal, group) in groups.iter_mut().enumerate() {
            self.advance(group, ordinal, sample, node)?;
        }
        Ok(sample)
    }
}

fn finish(
    state: &mut Float64NumericState,
    value: Option<f64>,
    history: impl Iterator<Item = Option<f64>> + Clone,
    node: &str,
) -> Result<()> {
    let accumulator = &mut state.accumulator;
    accumulator.numeric.reset_if_empty();
    if let Some(value) = value {
        add_float64(accumulator, value, node)?;
    }
    accumulator
        .numeric
        .finish_duration(history)
        .map_err(|error| float_arithmetic_error(error, node))?;
    accumulator.sync_float_sum();
    Ok(())
}

pub(super) struct DurationUndo {
    windows: Vec<(WindowAccumulator, usize)>,
    remaining: usize,
    capacity: usize,
    evicted: EvictedSamples,
}

impl DurationUndo {
    pub(super) fn new(samples: &DurationSamples, groups: &[TypedWindowState]) -> Result<Self> {
        let mut windows = Vec::with_capacity(groups.len());
        for (group, &head) in groups.iter().zip(&samples.heads) {
            let TypedWindowState::Numeric(state) = group else {
                return Err(internal_error(
                    "shared Duration undo requires a numeric accumulator",
                ));
            };
            windows.push((state.accumulator, head));
        }
        Ok(Self {
            windows,
            remaining: samples.len(),
            capacity: samples.values.capacity(),
            evicted: EvictedSamples::None,
        })
    }

    pub(super) fn len(&self) -> usize {
        self.windows.len()
    }

    pub(super) fn removed(&mut self, sample: TimedSample<f64>) {
        if self.remaining > 0 {
            self.remaining -= 1;
            self.evicted.push(sample);
        }
    }

    pub(super) fn rollback(self, samples: &mut DurationSamples, groups: &mut [TypedWindowState]) {
        for ((group, head), (accumulator, old_head)) in
            groups.iter_mut().zip(&mut samples.heads).zip(self.windows)
        {
            let TypedWindowState::Numeric(state) = group else {
                unreachable!("Duration undo validated the numeric state")
            };
            state.accumulator = accumulator;
            *head = old_head;
        }
        samples.values.truncate(self.remaining);
        self.evicted.restore(&mut samples.values);
        samples.values.shrink_to(self.capacity);
    }
}
