use std::{ops::Range, sync::Arc};

use datafusion::execution::memory_pool::MemoryReservation;
use parking_lot::Mutex;

#[cfg(test)]
use super::note_owner_remap;
use super::{COPY_BYTES, COPY_VISITS, Fragment, KeyFlow, LocalIntern, RowPayload, dispatch};
use crate::runtime::streaming::gather_work::{GatherStop, ParallelCpuWork};
use crate::{Result, StreamOperatorContext};

pub(in super::super) struct Group {
    fragments: Vec<Fragment>,
    payload: RowPayload,
    credit: Arc<MemoryReservation>,
}

impl Group {
    pub(super) fn new(
        fragments: Vec<Fragment>,
        payload: RowPayload,
        credit: Arc<MemoryReservation>,
    ) -> Self {
        Self {
            fragments,
            payload,
            credit,
        }
    }

    pub(in super::super) async fn finish(
        mut self,
        pending: &mut super::super::Pending,
        local: &LocalIntern,
        flow: &mut KeyFlow<'_, '_>,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        for fragment in &mut self.fragments {
            for (post, frequency) in fragment
                .local
                .posts
                .iter_mut()
                .zip(&mut fragment.local.retains)
            {
                let global = *post as usize;
                *post = local.posts[global];
                fragment.local.splits.push(local.splits[global]);
                // Retained references were merged; reuse their prepaid lane for resolved frequencies.
                *frequency = 0;
                flow.step(1, 0).await?;
            }
        }
        let work = Arc::new(RemapWork {
            fragments: self
                .fragments
                .into_iter()
                .map(|fragment| Mutex::new(Some(fragment)))
                .collect(),
            _payload: self.payload,
            #[cfg(test)]
            hook: super::super::tests::remap_worker_hook(),
            _credit: Arc::clone(&self.credit),
        });
        let fragments = if let Some(fragments) = dispatch(
            Arc::clone(&work),
            context,
            &self.credit,
            Arc::from(context.operator_id()),
        )
        .await?
        {
            fragments
        } else {
            let mut fragments = Vec::with_capacity(work.fragments.len());
            for cell in &work.fragments {
                let mut fragment = cell.lock().take().expect("unsubmitted remap fragment");
                for start in (0..fragment.local.locals.len()).step_by(COPY_VISITS) {
                    let end = (start + COPY_VISITS).min(fragment.local.locals.len());
                    remap_rows(&mut fragment, start..end);
                    #[cfg(test)]
                    note_owner_remap(end - start);
                    flow.step(end - start, 0).await?;
                }
                fragments.push(fragment);
            }
            fragments
        };
        for fragment in fragments {
            for (&id, &frequency) in fragment.local.posts.iter().zip(&fragment.local.retains) {
                pending.note_frequency(id, frequency as usize);
                flow.step(1, 0).await?;
            }
            for ids in fragment.local.locals.chunks(COPY_BYTES / size_of::<u32>()) {
                pending.ids.extend_from_slice(ids);
                flow.step(0, size_of_val(ids)).await?;
            }
        }
        Ok(())
    }
}

struct RemapWork {
    fragments: Vec<Mutex<Option<Fragment>>>,
    _payload: RowPayload,
    #[cfg(test)]
    hook: Option<super::super::tests::WorkerHook>,
    _credit: Arc<MemoryReservation>,
}

impl ParallelCpuWork for RemapWork {
    type Output = Fragment;

    fn unit_count(&self) -> usize {
        self.fragments.len()
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> Result<Fragment> {
        stop.check()?;
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(ordinal);
        }
        let mut fragment = self.fragments[ordinal]
            .lock()
            .take()
            .expect("one remap per fragment");
        for start in (0..fragment.local.locals.len()).step_by(COPY_VISITS) {
            stop.check()?;
            let end = (start + COPY_VISITS).min(fragment.local.locals.len());
            remap_rows(&mut fragment, start..end);
        }
        stop.check()?;
        Ok(fragment)
    }
}

fn remap_rows(fragment: &mut Fragment, range: Range<usize>) {
    for ordinal in range {
        let distinct = fragment.local.locals[ordinal] as usize;
        let split = fragment.local.splits[distinct];
        let position = fragment.start + ordinal;
        let id = if split != u32::MAX && position < split as usize {
            u32::MAX
        } else {
            fragment.local.posts[distinct]
        };
        fragment.local.locals[ordinal] = id;
        if id != u32::MAX {
            fragment.local.retains[distinct] = fragment.local.retains[distinct].saturating_add(1);
        }
    }
}
