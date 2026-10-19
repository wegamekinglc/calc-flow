use super::super::{
    AdmittedRows, Append, CHUNK_ROWS, Chunk, ColumnarPlan, MemoryReservation, NativeKeys, Result,
    StateRows, StreamOperatorContext, chunk_base_bytes, grant, reserve_references,
    selected_sources,
};
use super::{Inputs, controls, dispatch, ranges, units};
#[cfg(test)]
use super::{PreparationHook, preparation_hook};
use crate::operator::join::{AdmittedBatch, operator_error};
use crate::runtime::streaming::gather_work::{GatherStop, ParallelCpuWork, cleanup_control_bytes};
use parking_lot::Mutex;
use std::{ops::Range, sync::Arc};

struct ChunkSpec {
    range: Range<usize>,
    sources: usize,
    keys: usize,
    credit: Option<MemoryReservation>,
}

struct ChunkWork {
    inputs: Arc<Inputs>,
    plan: ColumnarPlan,
    units: Vec<Mutex<Option<Vec<ChunkSpec>>>>,
    #[cfg(feature = "benchmark-internals")]
    name: Arc<str>,
    #[cfg(test)]
    hook: Option<PreparationHook>,
}

struct Fragment {
    chunks: Vec<Chunk>,
    _control: Arc<MemoryReservation>,
}

struct Funding {
    control: Arc<MemoryReservation>,
    references: Option<MemoryReservation>,
    specs: Vec<Mutex<Option<Vec<ChunkSpec>>>>,
}

impl ParallelCpuWork for ChunkWork {
    type Output = Fragment;

    fn unit_count(&self) -> usize {
        self.units.len()
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<Self::Output> {
        stop.check()?;
        let specs = self.units[ordinal]
            .lock()
            .take()
            .expect("unique retained unit");
        #[cfg(any(test, feature = "benchmark-internals"))]
        let rows = specs.iter().map(|spec| spec.range.len()).sum();
        #[cfg(feature = "benchmark-internals")]
        let _phase = stop
            .join_cpu_trace()
            .worker_stage(&self.name, "retained_chunks", rows);
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook("chunks", ordinal, rows);
        }
        let mut chunks = Vec::with_capacity(specs.len());
        let mut cursor = (&self.inputs.rows as &dyn AdmittedRows).cursor();
        for spec in specs {
            stop.check_abandoned()?;
            let mut chunk =
                Chunk::reserved_keys(spec.range.len(), spec.sources, spec.keys, spec.credit);
            for (position, charge) in self.plan.range(spec.range) {
                let row = cursor.row(position);
                chunk.push_columns(
                    row.record,
                    row.event_time,
                    row.row_id,
                    charge,
                    self.inputs.keys.key(position),
                );
            }
            chunks.push(chunk);
        }
        stop.check()?;
        Ok(Fragment {
            chunks,
            _control: Arc::clone(&self.inputs.control),
        })
    }
}

fn specs(
    rows: &AdmittedBatch,
    keys: &NativeKeys,
    plan: &ColumnarPlan,
    units: usize,
    master: &MemoryReservation,
    name: &str,
) -> Result<Vec<Mutex<Option<Vec<ChunkSpec>>>>> {
    let chunks = plan.len().div_ceil(CHUNK_ROWS);
    let mut result = Vec::with_capacity(units);
    for range in ranges(chunks, units) {
        let mut work = Vec::with_capacity(range.len());
        for ordinal in range {
            let start = ordinal * CHUNK_ROWS;
            let range = start..(start + CHUNK_ROWS).min(plan.len());
            let sources = selected_sources(rows, plan.range(range.clone()));
            let key_count = range.len().min(keys.keys.len());
            let bytes = chunk_base_bytes(range.len(), sources, key_count).ok_or_else(|| {
                crate::operator::join::native_lookup::scratch_error("Join state chunks")
            })?;
            let credit = grant(master, bytes, name, false)?;
            work.push(ChunkSpec {
                range,
                sources,
                keys: key_count,
                credit,
            });
        }
        result.push(Mutex::new(Some(work)));
    }
    Ok(result)
}

fn extra_controls(rows: usize, units: usize) -> Option<usize> {
    rows.div_ceil(CHUNK_ROWS)
        .checked_mul(size_of::<ChunkSpec>())?
        .checked_add(
            units
                .checked_mul(size_of::<Mutex<Option<Vec<ChunkSpec>>>>() + size_of::<Fragment>())?,
        )?
        .checked_add(size_of::<ChunkWork>() + cleanup_control_bytes::<Vec<Fragment>>())
}

impl StateRows {
    fn fund_chunks(
        &self,
        rows: &AdmittedBatch,
        keys: &NativeKeys,
        plan: &ColumnarPlan,
        master: &MemoryReservation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Option<Funding>> {
        let units = units(plan.len(), context.job().gather_owner().parallelism());
        let control = if units > 1 {
            extra_controls(plan.len(), units)
                .and_then(|extra| controls(master, units, extra, context.operator_id()))
        } else {
            None
        };
        let Some(control) = control else {
            #[cfg(feature = "benchmark-internals")]
            context.job().join_cpu_trace().note(
                context.operator_id(),
                "retained_chunks",
                if units <= 1 {
                    "cpu_or_grain"
                } else {
                    "control_budget"
                },
                plan.len(),
                units,
                0,
            );
            return Ok(None);
        };
        self.validate_append(plan.len())?;
        let Ok(references) = reserve_references(master, plan.len(), context.operator_id(), false)
        else {
            drop(control);
            return Ok(None);
        };
        let Ok(specs) = specs(rows, keys, plan, units, master, context.operator_id()) else {
            drop(references);
            drop(control);
            return Ok(None);
        };
        #[cfg(feature = "benchmark-internals")]
        context.job().join_cpu_trace().note(
            context.operator_id(),
            "retained_chunks",
            "prepaid",
            plan.len(),
            units,
            control.size()
                + specs
                    .iter()
                    .map(|unit| {
                        unit.lock()
                            .as_ref()
                            .unwrap()
                            .iter()
                            .map(|spec| spec.credit.as_ref().map_or(0, MemoryReservation::size))
                            .sum::<usize>()
                    })
                    .sum::<usize>(),
        );
        Ok(Some(Funding {
            control,
            references,
            specs,
        }))
    }

    async fn prepare_columns_fallback(
        &self,
        rows: &AdmittedBatch,
        keys: &NativeKeys,
        plan: ColumnarPlan,
        master: &MemoryReservation,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Append> {
        let plan = plan.into_serial(master, context).await?;
        self.prepare_columns(rows, keys, &plan, master, context)
            .await
    }

    pub(in crate::operator::join) async fn prepare_columns_owned(
        &self,
        rows: &mut AdmittedBatch,
        keys: &mut Option<NativeKeys>,
        plan: ColumnarPlan,
        master: &MemoryReservation,
        reuse_only: bool,
        context: &StreamOperatorContext<'_>,
    ) -> Result<Append> {
        #[cfg(feature = "benchmark-internals")]
        let _phase = context.job().join_cpu_trace().stage(
            context.operator_id(),
            "retained_chunks",
            plan.len(),
        );
        let Some(Funding {
            control,
            references,
            specs,
        }) = self.fund_chunks(
            rows,
            keys.as_ref().expect("native keys"),
            &plan,
            master,
            context,
        )?
        else {
            return self
                .prepare_columns_fallback(
                    rows,
                    keys.as_ref().expect("native keys"),
                    plan,
                    master,
                    context,
                )
                .await;
        };
        let inputs = Arc::new(Inputs {
            rows: std::mem::take(rows),
            keys: keys.take().expect("native keys"),
            control,
        });
        let work = Arc::new(ChunkWork {
            inputs: Arc::clone(&inputs),
            plan,
            units: specs,
            #[cfg(feature = "benchmark-internals")]
            name: context.operator_id().into(),
            #[cfg(test)]
            hook: preparation_hook(),
        });
        let result = dispatch(Arc::clone(&work), context, &inputs.control, reuse_only).await?;
        let work = Arc::try_unwrap(work).map_err(|_| {
            operator_error(context.operator_id(), "retained chunk work still running")
        })?;
        let plan = work.plan;
        drop(work.units);
        drop(work.inputs);
        #[cfg(feature = "benchmark-internals")]
        drop(work.name);
        let inputs = Arc::try_unwrap(inputs).map_err(|_| {
            operator_error(context.operator_id(), "retained chunk inputs still owned")
        })?;
        *rows = inputs.rows;
        *keys = Some(inputs.keys);
        drop(inputs.control);
        let Some(fragments) = result else {
            drop(references);
            return self
                .prepare_columns_fallback(
                    rows,
                    keys.as_ref().expect("native keys"),
                    plan,
                    master,
                    context,
                )
                .await;
        };
        let mut chunks = Vec::with_capacity(plan.len().div_ceil(CHUNK_ROWS));
        for mut fragment in fragments {
            chunks.append(&mut fragment.chunks);
        }
        Ok(Append {
            chunks,
            indices: self.append_indices(plan.len()),
            count: plan.len(),
            control: references,
        })
    }
}
