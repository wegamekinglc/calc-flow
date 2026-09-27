//! Committed ASOF checkpoint bytes that drop finalized left prefixes lazily.

use crate::StateSegment;

/// Canonical encoding of the committed state. Finalizing a left prefix only
/// records the removed byte span over the shared `base`; the canonical bytes
/// are copied and hashed once, when a capture compacts the view, instead of
/// once per finalized chunk.
pub(in super::super) struct PreparedSegment {
    base: StateSegment,
    drained: Option<Drained>,
}

#[derive(Clone, Copy)]
struct Drained {
    left_rows: u64,
    skip: usize,
}

impl PreparedSegment {
    pub(in super::super) const fn new(segment: StateSegment) -> Self {
        Self {
            base: segment,
            drained: None,
        }
    }

    /// Whether finalized left rows are still skipped over the shared base.
    pub(in super::super) const fn is_drained(&self) -> bool {
        self.drained.is_some()
    }

    /// Encoded length of the canonical bytes.
    pub(in super::super) fn len(&self) -> usize {
        self.base.bytes().len() - self.skip()
    }

    /// Owned buffer capacity charged for the canonical bytes; a drained view
    /// charges the exact-capacity buffer its materialization allocates.
    pub(in super::super) fn capacity(&self) -> usize {
        match self.drained {
            None => self.base.bytes_arc().capacity(),
            Some(_) => self.len(),
        }
    }

    /// Drops the next `removed` encoded bytes of left rows after the header,
    /// leaving `left_rows` pending rows and every right bucket untouched.
    pub(in super::super) fn drain_left(&self, removed: usize, left_rows: u64) -> Self {
        Self {
            base: self.base.clone(),
            drained: Some(Drained {
                left_rows,
                skip: self.skip() + removed,
            }),
        }
    }

    /// Returns the canonical segment, copying a drained view's live bytes.
    pub(in super::super) fn canonical(&self) -> StateSegment {
        self.drained.map_or_else(
            || self.base.clone(),
            |drained| materialize(&self.base, drained, self.len()),
        )
    }

    fn skip(&self) -> usize {
        self.drained.map_or(0, |drained| drained.skip)
    }
}

fn materialize(base: &StateSegment, drained: Drained, length: usize) -> StateSegment {
    let bytes = base.bytes();
    let mut canonical = Vec::with_capacity(length);
    canonical.extend_from_slice(&bytes[..8]);
    canonical.extend_from_slice(&drained.left_rows.to_le_bytes());
    canonical.extend_from_slice(&bytes[16..24]);
    canonical.extend_from_slice(&bytes[24 + drained.skip..]);
    StateSegment::new(canonical)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded(left_rows: u64, body: &[u8]) -> Vec<u8> {
        let mut bytes = b"CFASOF01".to_vec();
        bytes.extend_from_slice(&left_rows.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(body);
        bytes
    }

    #[test]
    fn drained_views_materialize_canonical_bytes_over_the_shared_base() {
        let whole = PreparedSegment::new(StateSegment::new(encoded(3, b"aaabbbcccRIGHT")));
        let first = whole.drain_left(3, 2);
        assert_eq!((first.len(), first.capacity()), (35, 35));
        assert_eq!(
            first.canonical(),
            StateSegment::new(encoded(2, b"bbbcccRIGHT"))
        );
        let second = first.drain_left(3, 1);
        assert_eq!(
            second.canonical(),
            StateSegment::new(encoded(1, b"cccRIGHT"))
        );
        assert_eq!(
            second.drain_left(3, 0).canonical().bytes(),
            encoded(0, b"RIGHT")
        );
        assert_eq!(
            whole.canonical().bytes(),
            encoded(3, b"aaabbbcccRIGHT"),
            "the shared base stays unchanged"
        );
    }

    /// The per-prefix copy that `PreparedSegment` replaced, kept verbatim so
    /// the lazy view is pinned to the bytes and capacity it used to produce.
    fn eager_drain(current: &StateSegment, removed: usize, left_rows: u64) -> StateSegment {
        let mut bytes = Vec::with_capacity(current.bytes().len() - removed);
        bytes.extend_from_slice(&current.bytes()[..8]);
        bytes.extend_from_slice(&left_rows.to_le_bytes());
        bytes.extend_from_slice(&current.bytes()[16..24]);
        current.bytes()[24 + removed..]
            .chunks(64 * 1024)
            .for_each(|chunk| bytes.extend_from_slice(chunk));
        StateSegment::new(bytes)
    }

    #[test]
    fn drained_views_equal_the_eager_prefix_copy() {
        let body = (0..300_000_u32)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let mut base = encoded(5, &body);
        base[16..24].copy_from_slice(&7_u64.to_le_bytes());
        let mut eager = StateSegment::new(base);
        let mut view = PreparedSegment::new(eager.clone());
        for (left_rows, removed) in (0..5_u64).rev().zip([1, 4095, 70_000, 65_536, 17]) {
            eager = eager_drain(&eager, removed, left_rows);
            view = view.drain_left(removed, left_rows);
            assert_eq!(view.len(), eager.bytes().len());
            assert_eq!(view.capacity(), eager.bytes_arc().capacity());
            assert_eq!(view.canonical(), eager, "bytes and digest match");
        }
    }
}
