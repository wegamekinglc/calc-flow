use super::{
    AdmittedGroup, AdmittedRows, ColumnarPlan, MemoryReservation, NativeKeys, PayloadCharges,
    Result, Selection, StreamOperatorContext,
};
use crate::operator::join::{AdmittedBatch, operator_error};
use crate::runtime::streaming::gather_work::{
    AdmissionFailure, GatherOperatorId, GatherStop, ParallelCpuWork, cleanup_control_bytes,
};
use std::{ops::Range, sync::Arc};

mod chunks;

const ROWS_PER_UNIT: usize = 8_192;

struct Inputs {
    rows: AdmittedBatch,
    keys: NativeKeys,
    control: Arc<MemoryReservation>,
}

struct SelectionWork {
    inputs: Arc<Inputs>,
    ranges: Vec<Range<usize>>,
    counts: Vec<usize>,
    credit: Option<Arc<MemoryReservation>>,
    name: Arc<str>,
    #[cfg(test)]
    hook: Option<PreparationHook>,
}

impl ParallelCpuWork for SelectionWork {
    type Output = Selection;

    fn unit_count(&self) -> usize {
        self.ranges.len()
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<Self::Output> {
        #[cfg(feature = "benchmark-internals")]
        let _phase = stop.join_cpu_trace().worker_stage(
            &self.name,
            "retained_selection",
            self.ranges[ordinal].len(),
        );
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook("selection", ordinal, self.ranges[ordinal].len());
        }
        let mut result = Selection::new(self.counts[ordinal], self.credit.clone());
        result.controls = Some(Arc::clone(&self.inputs.control));
        let rows = &self.inputs.rows;
        let mut start = self.ranges[ordinal].start;
        let end = self.ranges[ordinal].end;
        while start < end {
            stop.check_abandoned()?;
            let group_end = rows.group_end(start).min(end);
            let group = AdmittedGroup::new(rows, start..group_end);
            for offset in (0..group.len()).step_by(64) {
                stop.check_abandoned()?;
                let block = group.block(offset, (offset + 64).min(group.len()));
                let mut selected = block.retained_bits(0, block.len());
                if selected == 0 {
                    continue;
                }
                let charges = PayloadCharges::on_worker(block, selected, stop)?;
                while selected != 0 {
                    let index = selected.trailing_zeros() as usize;
                    selected &= selected - 1;
                    let position = start + offset + index;
                    let charge = charges.state_charge(
                        index,
                        block.row(index).record,
                        self.inputs.keys.key(position).len(),
                        &self.name,
                    )?;
                    result.positions.push(position);
                    result.charges.push(charge);
                    result.bytes = result.bytes.checked_add(charge).ok_or_else(|| {
                        crate::operator::join::counter_overflow(&self.name, "state bytes")
                    })?;
                }
            }
            start = group_end;
        }
        stop.check()?;
        Ok(result)
    }
}

fn units(rows: usize, capacity: usize) -> usize {
    capacity.min(rows / ROWS_PER_UNIT).max(1)
}

fn ranges(rows: usize, units: usize) -> Vec<Range<usize>> {
    let width = rows / units;
    let extra = rows % units;
    (0..units)
        .map(|ordinal| {
            let start = ordinal * width + ordinal.min(extra);
            start..start + width + usize::from(ordinal < extra)
        })
        .collect()
}

fn retained_count(rows: &AdmittedBatch, range: &Range<usize>) -> usize {
    (range.start..range.end)
        .step_by(64)
        .map(|start| {
            rows.retained_bits(start, (range.end - start).min(64))
                .count_ones() as usize
        })
        .sum()
}

fn controls(
    master: &MemoryReservation,
    units: usize,
    extra: usize,
    name: &str,
) -> Option<Arc<MemoryReservation>> {
    let bytes = units
        .checked_mul(2 * size_of::<Selection>() + size_of::<Range<usize>>() + size_of::<usize>())?
        .checked_add(size_of::<Inputs>() + size_of::<SelectionWork>() + 4096)?
        .checked_add(cleanup_control_bytes::<Vec<Selection>>())?
        .checked_add(extra)?
        .checked_add(name.len())?;
    let credit = master.new_empty();
    credit.try_grow(bytes).ok()?;
    Some(Arc::new(credit))
}

impl ColumnarPlan {
    pub(in crate::operator::join) async fn prepare_owned(
        rows: &mut AdmittedBatch,
        keys: &mut Option<NativeKeys>,
        master: &MemoryReservation,
        reuse_only: bool,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Self> {
        #[cfg(feature = "benchmark-internals")]
        let _phase = context.job().join_cpu_trace().stage(
            context.operator_id(),
            "retained_selection",
            rows.len(),
        );
        let capacity = context.job().gather_owner().parallelism();
        let retained = if units(rows.len(), capacity) > 1 {
            retained_count(rows, &(0..rows.len()))
        } else {
            rows.len()
        };
        if retained == 0 {
            context.check_cancelled()?;
            #[cfg(feature = "benchmark-internals")]
            context.job().join_cpu_trace().note(
                context.operator_id(),
                "retained_selection",
                "no_retained_rows",
                rows.len(),
                0,
                0,
            );
            return Self::new(0, master, context.operator_id());
        }
        let units = units(retained, capacity);
        let control = (units > 1)
            .then(|| controls(master, units, 0, context.operator_id()))
            .flatten();
        let Some(control) = control else {
            #[cfg(feature = "benchmark-internals")]
            context.job().join_cpu_trace().note(
                context.operator_id(),
                "retained_selection",
                if units <= 1 {
                    "cpu_or_grain"
                } else {
                    "control_budget"
                },
                rows.len(),
                units,
                0,
            );
            return Self::prepare(rows, keys.as_ref().expect("native keys"), master, context).await;
        };
        let ranges = ranges(rows.len(), units);
        let counts = ranges
            .iter()
            .map(|range| retained_count(rows, range))
            .collect::<Vec<_>>();
        let count = counts.iter().sum();
        let Ok(credit) = Self::reserve_credit(count, master, context.operator_id()) else {
            drop(ranges);
            drop(counts);
            drop(control);
            return Self::prepare(rows, keys.as_ref().expect("native keys"), master, context).await;
        };
        #[cfg(feature = "benchmark-internals")]
        context.job().join_cpu_trace().note(
            context.operator_id(),
            "retained_selection",
            "prepaid",
            rows.len(),
            units,
            control.size() + credit.as_ref().map_or(0, |credit| credit.size()),
        );
        let inputs = Arc::new(Inputs {
            rows: std::mem::take(rows),
            keys: keys.take().expect("native keys"),
            control,
        });
        let work = Arc::new(SelectionWork {
            inputs: Arc::clone(&inputs),
            ranges,
            counts,
            credit,
            name: context.operator_id().into(),
            #[cfg(test)]
            hook: preparation_hook(),
        });
        let result = dispatch(work, context, &inputs.control, reuse_only).await?;
        let inputs = Arc::try_unwrap(inputs).map_err(|_| {
            operator_error(context.operator_id(), "retained workers still own input")
        })?;
        *rows = inputs.rows;
        *keys = Some(inputs.keys);
        drop(inputs.control);
        let Some(parts) = result else {
            return Self::prepare(rows, keys.as_ref().expect("native keys"), master, context).await;
        };
        Self::from_parts(parts, context.operator_id())
    }

    fn from_parts(mut parts: Vec<Selection>, name: &str) -> Result<Self> {
        let mut end = 0usize;
        let mut bytes = 0u64;
        for part in &mut parts {
            end += part.positions.len();
            part.end = end;
            bytes = bytes
                .checked_add(part.bytes)
                .ok_or_else(|| crate::operator::join::counter_overflow(name, "state bytes"))?;
        }
        let control = parts[0].controls.as_ref().expect("selection controls");
        let retained_controls = parts
            .capacity()
            .checked_mul(size_of::<Selection>())
            .and_then(|bytes| {
                bytes.checked_add(size_of::<MemoryReservation>() + 2 * size_of::<usize>())
            })
            .ok_or_else(|| operator_error(name, "retained selection control overflow"))?;
        let released_controls = control
            .size()
            .checked_sub(retained_controls)
            .ok_or_else(|| operator_error(name, "retained selection control exhausted"))?;
        // Settled workers leave only the selection fragment metadata alive.
        control.shrink(released_controls);
        Ok(Self {
            single: Selection::new(0, None),
            parts,
            bytes,
        })
    }
}

async fn dispatch<W: ParallelCpuWork>(
    work: Arc<W>,
    context: &StreamOperatorContext<'_>,
    credit: &Arc<MemoryReservation>,
    reuse_only: bool,
) -> Result<Option<Vec<W::Output>>> {
    let scope = context
        .gather_client(GatherOperatorId::new(context.operator_id().into()))
        .scope()?;
    let scope = if reuse_only {
        scope.reuse_only()
    } else {
        scope
    };
    let stop = GatherStop::from_job(context.job());
    let retirement = context.job().gather_owner().retain_retirement()?;
    let mut cleanup = None;
    let ticket = scope
        .submit_observed_parallel_work(
            work,
            credit.new_empty(),
            stop.clone(),
            retirement,
            &mut cleanup,
        )
        .await;
    let result = match ticket {
        Ok(ticket) => {
            let result = tokio::select! {
                biased;
                () = crate::operator::join::output_gather::cancelled(context.job()) => {
                    context.check_cancelled()?;
                    unreachable!("retained stop requires cancellation or deadline");
                }
                result = ticket.finish() => result,
            };
            let mut fragments = None;
            result?.install(|values| {
                fragments = Some(values);
                Ok(())
            })?;
            Ok(fragments)
        }
        Err(AdmissionFailure::Budget { .. }) => Ok(None),
        Err(AdmissionFailure::Runtime(error)) => Err(error),
    };
    context.check_cancelled()?;
    if let Some(cleanup) = cleanup {
        cleanup.wait(&stop).await?;
    }
    context.check_cancelled()?;
    result
}

#[cfg(test)]
type PreparationHook = Arc<dyn Fn(&'static str, usize, usize) + Send + Sync>;

#[cfg(test)]
thread_local! {
    static PREPARATION_HOOK: std::cell::RefCell<Option<PreparationHook>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(in crate::operator::join) struct PreparationObservation(Option<PreparationHook>);

#[cfg(test)]
impl Drop for PreparationObservation {
    fn drop(&mut self) {
        PREPARATION_HOOK.with(|current| {
            current.replace(self.0.take());
        });
    }
}

#[cfg(test)]
pub(in crate::operator::join) fn observe_preparation_for_test(
    hook: PreparationHook,
) -> PreparationObservation {
    PreparationObservation(PREPARATION_HOOK.with(|current| current.replace(Some(hook))))
}

#[cfg(test)]
fn preparation_hook() -> Option<PreparationHook> {
    PREPARATION_HOOK.with(|current| current.borrow().clone())
}
