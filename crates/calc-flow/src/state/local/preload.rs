use std::{mem::size_of, sync::Arc};

use datafusion::execution::memory_pool::MemoryReservation;

use super::{
    LocalStateLineageBackend, ManagedSegmentPaths, StateHandle, read_validated_file, worker,
};
use crate::{Result, StateSegment};

#[cfg(test)]
pub(super) type ReadHook =
    Arc<dyn Fn(&[u8], usize, &Arc<MemoryReservation>, usize) -> Result<()> + Send + Sync>;

pub(crate) struct PaidBytes {
    bytes: Vec<u8>,
    credit: Arc<MemoryReservation>,
}

impl PaidBytes {
    pub(crate) fn into_segment(self, checksum: String) -> StateSegment {
        StateSegment::from_validated(self.bytes, checksum).with_owner(self.credit)
    }
}

struct PaidRead {
    paths: ManagedSegmentPaths,
    handle: StateHandle,
    #[cfg(test)]
    hook: Option<ReadHook>,
    #[cfg(test)]
    path_peak: usize,
    credit: Arc<MemoryReservation>,
}

impl PaidRead {
    fn read(self) -> Result<PaidBytes> {
        let bytes = read_validated_file(&self.paths.committed, &self.handle)?;
        let output = PaidBytes {
            bytes,
            credit: self.credit.clone(),
        };
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook(
                &output.bytes,
                output.bytes.capacity(),
                &output.credit,
                self.path_peak,
            )?;
        }
        Ok(output)
    }
}

impl LocalStateLineageBackend {
    pub(crate) fn supports_prepaid_load(&self) -> bool {
        certified_root(&self.root.path)
    }

    pub(crate) fn prepaid_load_controls(&self, handle: &StateHandle) -> Option<usize> {
        let paths = path_controls(self.root.path.as_os_str().as_encoded_bytes().len(), handle)?;
        let strings = handle_strings(handle)?;
        let worker = blocking_controls()?;
        paths.checked_add(strings)?.checked_add(worker)
    }

    pub(crate) async fn load_segment_prepaid(
        &self,
        handle: &StateHandle,
        credit: Arc<MemoryReservation>,
    ) -> Result<PaidBytes> {
        #[cfg(not(test))]
        let paths = self.managed_paths(handle)?;
        #[cfg(test)]
        let (paths, path_peak) = measured_paths(self, handle)?;
        let input = PaidRead {
            paths,
            handle: handle.clone(),
            #[cfg(test)]
            hook: self.prepaid_read_hook.lock().clone(),
            #[cfg(test)]
            path_peak,
            credit,
        };
        let _guard = self.publication.lock().await;
        worker(move || input.read()).await
    }

    #[cfg(test)]
    pub(crate) fn set_prepaid_read_hook(&self, hook: ReadHook) {
        *self.prepaid_read_hook.lock() = Some(hook);
    }
}

#[cfg(not(windows))]
fn certified_root(_: &std::path::Path) -> bool {
    true
}

#[cfg(windows)]
fn certified_root(root: &std::path::Path) -> bool {
    use std::path::{Component, Prefix};
    matches!(root.components().next(), Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::Verbatim(_) | Prefix::VerbatimUNC(..) | Prefix::VerbatimDisk(_)))
}

fn handle_strings(handle: &StateHandle) -> Option<usize> {
    [
        handle.operator_id(),
        handle.segment_id(),
        handle.relative_path(),
        handle.sha256(),
    ]
    .iter()
    .try_fold(0usize, |bytes, value| bytes.checked_add(value.len()))
}

fn path_controls(root: usize, handle: &StateHandle) -> Option<usize> {
    let longest = longest_path(root, handle)?;
    let paths = path_slots(longest)?;
    let bytes = paths.checked_add(formatted_strings(20, 64)?)?;
    #[cfg(windows)]
    let bytes = bytes.checked_add(wide_controls(longest)?)?;
    Some(bytes)
}

fn path_slots(length: usize) -> Option<usize> {
    // Four returned paths plus two simultaneous clone/reallocation intermediates.
    6usize.checked_mul(length.checked_mul(2)?.checked_add(8)?)
}

fn longest_path(root: usize, handle: &StateHandle) -> Option<usize> {
    let epoch = 20; // u64 decimal width.
    let hash = 64; // SHA-256 hexadecimal width.
    let staging_suffix = ["staging".len(), hash, epoch, hash, hash + ".tmp".len()];
    let staged = staging_suffix
        .iter()
        .try_fold(root, |bytes, component| bytes.checked_add(component + 1))?;
    let committed = root
        .checked_add(handle.relative_path().len())?
        .checked_add(1)?;
    Some(staged.max(committed))
}

fn formatted_strings(epoch: usize, hash: usize) -> Option<usize> {
    let stem = epoch + 1 + hash;
    let relative = "committed".len() + 3 + 2 * hash + stem;
    let rendered = [
        stem,
        relative + ".arrow".len(),
        relative + ".segment".len(),
        epoch,
        hash + ".tmp".len(),
    ];
    rendered.iter().try_fold(2 * hash, |bytes, length| {
        bytes.checked_add(length.checked_mul(3)?.checked_add(16)?)
    })
}

#[cfg(windows)]
fn wide_controls(length: usize) -> Option<usize> {
    // File metadata and File::open each construct a len+1 UTF-16 Vec for certified verbatim paths.
    length.checked_add(1)?.checked_mul(2 * size_of::<u16>())
}

fn blocking_controls() -> Option<usize> {
    let input = size_of::<PaidRead>();
    let future = input.checked_add(size_of::<usize>())?; // BlockingTask's Option<closure>.
    let output = size_of::<std::result::Result<Result<PaidBytes>, tokio::task::JoinError>>();
    let scheduler =
        size_of::<tokio::runtime::Handle>() + size_of::<Option<Arc<dyn Fn() + Send + Sync>>>();
    let header = 3 * size_of::<usize>() + size_of::<u64>();
    let trailer = 2 * size_of::<usize>()
        + size_of::<Option<std::task::Waker>>()
        + size_of::<Option<Arc<dyn Fn() + Send + Sync>>>();
    let fixed = header + scheduler + size_of::<u64>() + trailer;
    let cell = fixed
        .checked_add(future)?
        .checked_add(output)?
        .checked_add(size_of::<usize>())?;
    task_allocation(cell)?.checked_add(input)
}

fn task_allocation(bytes: usize) -> Option<usize> {
    bytes.checked_add(255)?.checked_div(256)?.checked_mul(256)
}

#[cfg(test)]
fn measured_paths(
    backend: &LocalStateLineageBackend,
    handle: &StateHandle,
) -> Result<(ManagedSegmentPaths, usize)> {
    let mut paths = None;
    let measured = allocation_counter::measure(|| {
        paths = Some(backend.managed_paths(handle));
    });
    Ok((
        paths.expect("measured path constructor")?,
        usize::try_from(measured.bytes_max).expect("allocator peak fits usize"),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};

    #[tokio::test(flavor = "current_thread")]
    async fn test_actual_blocking_result_keeps_wire_credit_until_last_segment_drop() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
        let credit = MemoryConsumer::new("blocking-result-test").register(&pool);
        credit.try_grow(4_096).unwrap();
        let credit = Arc::new(credit);
        let weak_credit = Arc::downgrade(&credit);
        let bytes = worker(move || {
            Ok(PaidBytes {
                bytes: vec![7; 4_096],
                credit,
            })
        })
        .await
        .unwrap();
        assert_eq!(pool.reserved(), 4_096);
        assert!(weak_credit.upgrade().is_some());
        let segment = bytes.into_segment("0".repeat(64));
        assert_eq!(segment.bytes(), &[7; 4_096]);
        let last = segment.clone();
        let weak_bytes = Arc::downgrade(&segment.bytes_arc());
        drop(segment);
        assert!(weak_bytes.upgrade().is_some());
        assert_eq!(pool.reserved(), 4_096);
        drop(last);
        assert!(weak_bytes.upgrade().is_none());
        assert!(weak_credit.upgrade().is_none());
        assert_eq!(pool.reserved(), 0);
    }
}
