use super::{
    Arc, BatchKey, BucketCut, Change, Cursor, DecodedDelta, Encoding, Identity, MAGIC,
    MemoryReservation, OwnerReader, PayloadBatch, PreparedLeftChunk, Result, SequenceKind, Version,
    mismatch, read_version,
};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

struct Header {
    capacities: [usize; 16],
    counts: [usize; 3],
    owner_capacity: usize,
    changes: usize,
    left: usize,
    buckets: usize,
}

fn header(
    cursor: &mut Cursor<'_>,
    previous: usize,
    kinds: [SequenceKind; 2],
    max_rows: u64,
) -> Result<Header> {
    if cursor.take(8)? != MAGIC {
        return Err(mismatch("ASOF current delta magic differs"));
    }
    for kind in kinds {
        super::super::sequence_kind(cursor, kind)?;
    }
    if cursor.take(6)? != [0; 6] {
        return Err(mismatch("ASOF log delta padding differs"));
    }
    let capacities = cursor.capacities(super::super::CAPACITY_WIDTHS)?;
    let counts = cursor.addresses::<3>()?;
    super::super::validate_header_counts(counts[0], counts[1], counts[2], max_rows)?;
    super::super::validate_header_capacities(
        capacities[..5].try_into().expect("native capacities"),
        counts[0],
        counts[1],
    )?;
    for capacity in &capacities[5..8] {
        super::super::require_capacity(*capacity, counts[1])?;
        u32::try_from(*capacity)
            .map_err(|_| mismatch("ASOF log heap capacity exceeds handle domain"))?;
    }
    super::super::validate_shard_capacities(
        capacities[8..].try_into().expect("eight shards"),
        counts[1],
    )?;
    if cursor.address()? != previous {
        return Err(mismatch("ASOF log owner predecessor count differs"));
    }
    let owner_capacity = cursor.capacity(size_of::<Encoding>())?;
    let [changes, left, buckets] = cursor.addresses()?;
    if changes > cursor.bytes.len() / 11 || left > changes || buckets > cursor.bytes.len() / 17 {
        return Err(mismatch("ASOF log change counts exceed encoded size"));
    }
    Ok(Header {
        capacities,
        counts,
        owner_capacity,
        changes,
        left,
        buckets,
    })
}

fn owner_domain(cursor: &Cursor<'_>, previous: usize, capacity: usize) -> Result<usize> {
    let mut prefix = cursor.clone();
    let additional = prefix.address()?;
    let count = previous
        .checked_add(additional)
        .filter(|count| u32::try_from(*count).is_ok())
        .ok_or_else(|| mismatch("ASOF log owner count exceeds handle domain"))?;
    if capacity < count || capacity > count.saturating_mul(2).max(4) {
        return Err(mismatch("ASOF log owner capacity differs"));
    }
    Ok(count)
}

fn scan_version(
    cursor: &mut Cursor<'_>,
    left: bool,
    kind: SequenceKind,
    max_rows: u64,
) -> Result<()> {
    match cursor.byte()? {
        0 => Ok(()),
        1 if left => {
            let rows = cursor.address()?;
            let capacities = cursor.capacities([
                8,
                4,
                size_of::<Option<Encoding>>(),
                8,
                4,
                kind.storage_bytes(),
            ])?;
            validate_left_version(rows as u64, capacities, max_rows)
        }
        1 => match cursor.byte()? {
            0 | 2 => Ok(()),
            1 => {
                cursor.take(12)?;
                Ok(())
            }
            _ => Err(mismatch("ASOF log storage tag differs")),
        },
        _ => Err(mismatch("ASOF log version tag differs")),
    }
}

fn validate_left_version(rows: u64, capacities: [usize; 6], max_rows: u64) -> Result<()> {
    if rows == 0 || rows > max_rows {
        return Err(mismatch("ASOF log left row count exceeds limits"));
    }
    let count = usize::try_from(rows)
        .map_err(|_| mismatch("ASOF log left count exceeds address domain"))?;
    for index in [0, 4, 5] {
        super::super::require_capacity(capacities[index], count)?;
    }
    for index in [2, 3] {
        super::super::require_capacity(capacities[index], 1)?;
    }
    if capacities[1] != 0 {
        super::super::require_capacity(capacities[1], count)?;
    }
    Ok(())
}

fn scan_change(cursor: &mut Cursor<'_>, kinds: [SequenceKind; 2], max_rows: u64) -> Result<()> {
    let left = match cursor.byte()? {
        0 => {
            cursor.take(8)?;
            true
        }
        1 => {
            cursor.take(
                24 + usize::try_from(kinds[1].reference_bytes())
                    .expect("sequence width fits address domain"),
            )?;
            false
        }
        _ => return Err(mismatch("ASOF log identity tag differs")),
    };
    scan_version(cursor, left, kinds[0], max_rows)?;
    scan_version(cursor, left, kinds[0], max_rows)
}

fn bucket_state(
    cursor: &mut Cursor<'_>,
    kind: SequenceKind,
    max_rows: u64,
) -> Result<Option<(u64, [usize; 5])>> {
    match cursor.byte()? {
        0 => Ok(None),
        1 => {
            let rows = cursor.integer()?;
            if rows == 0 || rows > max_rows {
                return Err(mismatch("ASOF log bucket row count exceeds limits"));
            }
            let capacities =
                cursor.capacities([8, kind.storage_bytes(), 8, 8, kind.storage_bytes()])?;
            Ok(Some((rows, capacities)))
        }
        _ => Err(mismatch("ASOF log bucket state tag differs")),
    }
}

pub(super) fn restore_charge(
    bytes: &[u8],
    previous: usize,
    kinds: [SequenceKind; 2],
    max_rows: u64,
    max_bytes: u64,
) -> Result<u64> {
    restore_charge_checked(bytes, previous, kinds, max_rows, max_bytes, || Ok(()))
}

fn restore_charge_checked(
    bytes: &[u8],
    previous: usize,
    kinds: [SequenceKind; 2],
    max_rows: u64,
    max_bytes: u64,
    mut check_cancelled: impl FnMut() -> Result<()>,
) -> Result<u64> {
    let mut cursor = Cursor::new(bytes, max_bytes);
    let header = header(&mut cursor, previous, kinds, max_rows)?;
    let owners = owner_domain(&cursor, previous, header.owner_capacity)?;
    let mut charge =
        super::super::owners::restore_charge_checked(&mut cursor, &mut check_cancelled)?;
    charge = super::super::restore_add(charge, owners as u64 * 256 + 512)?;
    charge = super::super::restore_add(
        charge,
        header.changes as u64 * (size_of::<Change>() + 2048) as u64,
    )?;
    charge = super::super::restore_add(
        charge,
        header.left as u64 * size_of::<(BatchKey, PreparedLeftChunk)>() as u64,
    )?;
    charge = super::super::restore_add(
        charge,
        header.buckets as u64 * size_of::<BucketCut>() as u64,
    )?;
    for ordinal in 0..header.changes {
        check_every(ordinal, &mut check_cancelled)?;
        scan_change(&mut cursor, kinds, max_rows)?;
    }
    let mut rows = 0;
    for ordinal in 0..header.left {
        check_every(ordinal, &mut check_cancelled)?;
        let (count, bytes) = super::super::scan_left(&mut cursor, max_rows - rows)?;
        rows = super::super::restore_add(rows, count)?;
        charge = super::super::restore_add(charge, bytes)?;
    }
    for ordinal in 0..header.buckets {
        check_every(ordinal, &mut check_cancelled)?;
        cursor.take(16)?;
        bucket_state(&mut cursor, kinds[1], max_rows)?;
    }
    cursor.finish()?;
    Ok(charge)
}

fn change(
    cursor: &mut Cursor<'_>,
    owners: &mut OwnerReader,
    kinds: [SequenceKind; 2],
    max_rows: u64,
) -> Result<Change> {
    let identity = match cursor.byte()? {
        0 => Identity::Left((0, cursor.integer()?)),
        1 => {
            let time = i64::from_le_bytes(cursor.take(8)?.try_into().expect("time width"));
            let key = owners.reference(cursor)?;
            let sequence = super::super::sequence(cursor, kinds[1], owners)?;
            Identity::Right((time, key, sequence))
        }
        _ => return Err(mismatch("ASOF log identity tag differs")),
    };
    let change = Change {
        before: read_version(cursor, &identity, kinds[0])?,
        after: read_version(cursor, &identity, kinds[0])?,
        identity,
    };
    change.validate()?;
    for version in [change.before, change.after].into_iter().flatten() {
        if let Version::Left { rows, capacities } = version {
            validate_left_version(rows, capacities, max_rows)?;
        }
    }
    if let (Some(Version::Left { rows: before, .. }), Some(Version::Left { rows: after, .. })) =
        (change.before, change.after)
        && before < after
    {
        return Err(mismatch("ASOF log left prefix grows"));
    }
    Ok(change)
}

fn validate_payloads(
    change: &Change,
    batches: &BTreeMap<BatchKey, Arc<PayloadBatch>>,
) -> Result<()> {
    for version in [change.before, change.after].into_iter().flatten() {
        if let Version::Right {
            payload: Some((batch, row)),
            ..
        } = version
        {
            let payload = batches
                .get(&batch)
                .ok_or_else(|| mismatch("ASOF log right payload batch is missing"))?;
            if row as usize >= payload.record.num_rows() {
                return Err(mismatch("ASOF log right payload row differs"));
            }
        }
    }
    Ok(())
}

fn check_every(ordinal: usize, check_cancelled: &mut impl FnMut() -> Result<()>) -> Result<()> {
    if ordinal.is_multiple_of(128) {
        check_cancelled()?;
    }
    Ok(())
}

fn decode_buckets(
    cursor: &mut Cursor<'_>,
    owners: &mut OwnerReader,
    changes: &[Change],
    header: &Header,
    kind: SequenceKind,
    limits: &crate::AsofStateLimits,
    check_cancelled: &mut impl FnMut() -> Result<()>,
) -> Result<Vec<BucketCut>> {
    let expected = changes
        .iter()
        .filter_map(|change| match &change.identity {
            Identity::Left(_) => None,
            Identity::Right((_, key, _)) => Some(key),
        })
        .collect::<BTreeSet<_>>();
    let mut buckets = Vec::with_capacity(header.buckets);
    for ordinal in 0..header.buckets {
        check_every(ordinal, check_cancelled)?;
        let key = owners.reference(cursor)?;
        if buckets
            .last()
            .is_some_and(|last: &BucketCut| last.key >= key)
        {
            return Err(mismatch("ASOF log bucket capacity order differs"));
        }
        let state = bucket_state(cursor, kind, limits.max_state_rows())?;
        buckets.push(BucketCut { key, state });
    }
    if expected
        .iter()
        .any(|key| buckets.binary_search_by(|cut| cut.key.cmp(key)).is_err())
    {
        return Err(mismatch("ASOF log bucket capacity inventory differs"));
    }
    Ok(buckets)
}

pub(super) fn decode(
    bytes: &[u8],
    previous: &OwnerReader,
    batches: &BTreeMap<BatchKey, Arc<PayloadBatch>>,
    kinds: [SequenceKind; 2],
    limits: &crate::AsofStateLimits,
    workspace: MemoryReservation,
    mut check_cancelled: impl FnMut() -> Result<()>,
) -> Result<DecodedDelta> {
    let max_rows = limits.max_state_rows();
    let max_bytes = limits.max_state_bytes();
    check_cancelled()?;
    let workspace_bytes = restore_charge_checked(
        bytes,
        previous.count(),
        kinds,
        max_rows,
        max_bytes,
        &mut check_cancelled,
    )?;
    if u64::try_from(workspace.size()).map_or(true, |bytes| bytes < workspace_bytes) {
        return Err(mismatch("ASOF log decoding exceeds prepaid workspace"));
    }
    let mut cursor = Cursor::new(bytes, max_bytes);
    let header = header(&mut cursor, previous.count(), kinds, max_rows)?;
    let mut owners = previous.clone();
    owners.append_checked(&mut cursor, &mut check_cancelled)?;
    let mut changes = Vec::with_capacity(header.changes);
    for ordinal in 0..header.changes {
        check_every(ordinal, &mut check_cancelled)?;
        let change = change(&mut cursor, &mut owners, kinds, max_rows)?;
        if changes
            .last()
            .is_some_and(|last: &Change| last.identity >= change.identity)
        {
            return Err(mismatch("ASOF log change order is not strict"));
        }
        validate_payloads(&change, batches)?;
        changes.push(change);
    }
    let mut left = Vec::with_capacity(header.left);
    let mut previous_batch = None;
    let mut rows = 0;
    for ordinal in 0..header.left {
        check_every(ordinal, &mut check_cancelled)?;
        let mut preview = cursor.clone();
        let left_header =
            super::super::read_left_header(&mut preview, kinds[0], max_rows - rows, &mut None)?;
        let expected = changes
            .binary_search_by(|change| change.identity.cmp(&Identity::Left(left_header.batch)))
            .ok()
            .map(|index| &changes[index])
            .ok_or_else(|| mismatch("ASOF log left installation has no change"))?;
        if expected.before.is_some()
            || expected.after
                != Some(Version::Left {
                    rows: left_header.rows as u64,
                    capacities: left_header.capacities,
                })
        {
            return Err(mismatch("ASOF log left installation version differs"));
        }
        let chunk = super::super::read_left(
            &mut cursor,
            &mut owners,
            batches,
            kinds[0],
            max_rows - rows,
            &mut previous_batch,
        )?;
        rows += left_header.rows as u64;
        left.push((left_header.batch, chunk));
    }
    if changes
        .iter()
        .filter(|change| matches!(change.identity, Identity::Left(_)) && change.before.is_none())
        .count()
        != left.len()
    {
        return Err(mismatch("ASOF log left installation inventory differs"));
    }
    let buckets = decode_buckets(
        &mut cursor,
        &mut owners,
        &changes,
        &header,
        kinds[1],
        limits,
        &mut check_cancelled,
    )?;
    cursor.finish()?;
    owners.finish()?;
    check_cancelled()?;
    Ok(DecodedDelta {
        capacities: header.capacities,
        counts: header.counts,
        changes,
        left,
        buckets,
        owners,
        workspace,
    })
}
