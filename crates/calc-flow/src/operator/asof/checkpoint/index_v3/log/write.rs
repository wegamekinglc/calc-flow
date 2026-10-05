use super::{
    Arc, Change, EncodedDelta, Encoding, HEADER_BYTES, Identity, Input, MAGIC, MemoryReservation,
    OwnerWriter, Result, SequenceKind, StateSegment, Version, mismatch, version_bytes,
};
use crate::operator::asof::checked;
use std::collections::{BTreeMap, BTreeSet};

fn references<'a>(input: &'a Input<'_>) -> impl Iterator<Item = &'a Encoding> + 'a {
    let changes = input
        .changes
        .iter()
        .flat_map(|change| match &change.identity {
            Identity::Left(_) => [None, None],
            Identity::Right((_, key, sequence)) => [
                Some(key),
                (input.kinds[1] == SequenceKind::Canonical).then_some(sequence),
            ],
        })
        .flatten();
    let left = input.left.iter().flat_map(|(_, data, head)| {
        data.keys
            .iter()
            .flatten()
            .chain(
                data.sequences
                    .iter()
                    .skip(*head)
                    .filter_map(|sequence| match sequence {
                        std::borrow::Cow::Borrowed(encoding)
                            if input.kinds[0] == SequenceKind::Canonical =>
                        {
                            Some(encoding)
                        }
                        _ => None,
                    }),
            )
    });
    changes
        .chain(left)
        .chain(input.buckets.iter().map(|bucket| &bucket.key))
}

fn validate(input: &Input<'_>, cancel: &dyn Fn() -> Result<()>) -> Result<()> {
    validate_changes(input, cancel)?;
    validate_buckets(input)?;
    validate_left_installations(input, cancel)
}

fn validate_changes(input: &Input<'_>, cancel: &dyn Fn() -> Result<()>) -> Result<()> {
    let mut previous = None;
    for (ordinal, change) in input.changes.iter().enumerate() {
        super::super::check_step(ordinal, cancel)?;
        change.validate()?;
        if previous.is_some_and(|last| last >= &change.identity) {
            return Err(mismatch("ASOF log change order is not strict"));
        }
        previous = Some(&change.identity);
    }
    Ok(())
}

fn validate_buckets(input: &Input<'_>) -> Result<()> {
    let expected = input
        .changes
        .iter()
        .filter_map(|change| match &change.identity {
            Identity::Left(_) => None,
            Identity::Right((_, key, _)) => Some(key),
        })
        .collect::<BTreeSet<_>>();
    if input
        .buckets
        .windows(2)
        .any(|cuts| cuts[0].key >= cuts[1].key)
        || expected.iter().any(|key| {
            input
                .buckets
                .binary_search_by(|cut| cut.key.cmp(key))
                .is_err()
        })
    {
        return Err(mismatch("ASOF log bucket capacity inventory differs"));
    }
    Ok(())
}

fn validate_left_installations(input: &Input<'_>, cancel: &dyn Fn() -> Result<()>) -> Result<()> {
    let mut previous = None;
    for (ordinal, (batch, data, head)) in input.left.iter().enumerate() {
        super::super::check_step(ordinal, cancel)?;
        if previous.is_some_and(|last| last >= *batch) {
            return Err(mismatch("ASOF log left installation order is not strict"));
        }
        previous = Some(*batch);
        validate_left_installation(input, *batch, data, *head)?;
    }
    if input
        .changes
        .iter()
        .filter(|change| matches!(change.identity, Identity::Left(_)) && change.before.is_none())
        .count()
        != input.left.len()
    {
        return Err(mismatch("ASOF log left installation inventory differs"));
    }
    Ok(())
}

fn validate_left_installation(
    input: &Input<'_>,
    batch: super::BatchKey,
    data: &super::ChunkData,
    head: usize,
) -> Result<()> {
    let change = input
        .changes
        .binary_search_by(|change| change.identity.cmp(&Identity::Left(batch)))
        .ok()
        .map(|index| &input.changes[index])
        .ok_or_else(|| mismatch("ASOF log left installation has no change"))?;
    let rows = data
        .sequences
        .len()
        .checked_sub(head)
        .ok_or_else(|| mismatch("ASOF log left head differs"))?;
    let capacities = [
        data.times.inner().capacity() / 8,
        data.positions.as_ref().map_or(0, Vec::capacity),
        data.keys.capacity(),
        data.key_counts.capacity(),
        data.key_ids.capacity(),
        data.sequences.capacity(),
    ];
    if change.before.is_some()
        || change.after
            != Some(Version::Left {
                rows: rows as u64,
                capacities,
            })
    {
        return Err(mismatch("ASOF log left installation version differs"));
    }
    Ok(())
}

fn row_bytes(input: &Input<'_>, name: &str) -> Result<u64> {
    let mut bytes = HEADER_BYTES;
    for change in input.changes {
        let identity = match change.identity {
            Identity::Left(_) => 9,
            Identity::Right(_) => 25 + input.kinds[1].reference_bytes(),
        };
        bytes = checked(
            name,
            bytes,
            identity + version_bytes(change.before) + version_bytes(change.after),
        )?;
    }
    bytes = left_row_bytes(bytes, input, name)?;
    for bucket in input.buckets {
        bytes = checked(
            name,
            bytes,
            17 + if bucket.state.is_some() { 48 } else { 0 },
        )?;
    }
    Ok(bytes)
}

fn left_row_bytes(mut bytes: u64, input: &Input<'_>, name: &str) -> Result<u64> {
    for (batch, data, head) in input.left {
        if batch.0 != 0 {
            return Err(mismatch("ASOF log left batch side differs"));
        }
        bytes = checked(
            name,
            bytes,
            super::super::left_length(data, *head, input.kinds[0], name)?,
        )?;
    }
    Ok(bytes)
}

fn owner_bound(
    input: &Input<'_>,
    previous: &OwnerWriter,
    name: &str,
    cancel: &dyn Fn() -> Result<()>,
) -> Result<(usize, u64)> {
    let mut new = BTreeMap::new();
    for (ordinal, encoding) in references(input).enumerate() {
        super::super::check_step(ordinal, cancel)?;
        if let Some((address, _)) = encoding.allocation()
            && !previous.contains(encoding)
        {
            new.entry(address).or_insert(encoding.owner_wire_length());
        }
    }
    let count = previous
        .count()
        .checked_add(new.len())
        .filter(|count| u32::try_from(*count).is_ok())
        .ok_or_else(|| mismatch("ASOF log owner domain overflowed"))?;
    let bytes = new
        .values()
        .try_fold(8, |total, bytes| checked(name, total, *bytes))?;
    Ok((count, bytes))
}

pub(super) fn encode(
    input: &Input<'_>,
    previous: &OwnerWriter,
    live_allocation: impl Fn(usize) -> bool,
    mut reserve: impl FnMut(u64) -> Result<MemoryReservation>,
    limit: u64,
    name: &str,
    cancel: &dyn Fn() -> Result<()>,
) -> Result<EncodedDelta> {
    cancel()?;
    let PreparedOwners {
        owners,
        owner_capacity,
        owner_bytes,
        scratch: _scratch,
        copy_credit: _copy_credit,
    } = prepare_owners(input, previous, &mut reserve, name, cancel)?;
    let owner_credit = retained_owner_credit(&owners, &live_allocation, &mut reserve, name)?;
    let (capacity, lease) = wire_reservation(input, owner_bytes, &mut reserve, limit, name)?;
    let segment = encode_segment(
        input,
        previous.count(),
        owner_capacity,
        &owners,
        capacity,
        lease,
        cancel,
    )?;
    Ok(EncodedDelta {
        segment,
        owners,
        owner_credit,
    })
}

struct PreparedOwners {
    owners: OwnerWriter,
    owner_capacity: usize,
    owner_bytes: u64,
    scratch: MemoryReservation,
    copy_credit: MemoryReservation,
}

fn encoding_scratch_bytes(
    input: &Input<'_>,
    name: &str,
    cancel: &dyn Fn() -> Result<()>,
) -> Result<u64> {
    let references = references(input)
        .enumerate()
        .try_fold(0_u64, |count, (ordinal, _)| {
            super::super::check_step(ordinal, cancel)?;
            Ok::<u64, crate::CalcFlowError>(count + 1)
        })?;
    let sorting = input
        .left
        .iter()
        .map(|(_, data, _)| data.keys.len() as u64 * 32)
        .max()
        .unwrap_or(0);
    checked(name, references * 192 + 512, sorting)
}

fn prepare_owners(
    input: &Input<'_>,
    previous: &OwnerWriter,
    reserve: &mut impl FnMut(u64) -> Result<MemoryReservation>,
    name: &str,
    cancel: &dyn Fn() -> Result<()>,
) -> Result<PreparedOwners> {
    let scratch = reserve(encoding_scratch_bytes(input, name, cancel)?)?;
    validate(input, cancel)?;
    let (owner_count, owner_bytes) = owner_bound(input, previous, name, cancel)?;
    let owner_capacity = previous.capacity().max(owner_count);
    let copy_credit = reserve(
        384 + owner_count as u64 * 128 + owner_capacity as u64 * size_of::<Encoding>() as u64,
    )?;
    let mut owners = previous.copy_with_capacity(owner_capacity)?;
    register_owners(&mut owners, input, cancel)?;
    Ok(PreparedOwners {
        owners,
        owner_capacity,
        owner_bytes,
        scratch,
        copy_credit,
    })
}

fn register_owners(
    owners: &mut OwnerWriter,
    input: &Input<'_>,
    cancel: &dyn Fn() -> Result<()>,
) -> Result<()> {
    register_change_owners(owners, input, cancel)?;
    register_left_owners(owners, input, cancel)?;
    register_bucket_owners(owners, input, cancel)
}

fn register_change_owners(
    owners: &mut OwnerWriter,
    input: &Input<'_>,
    cancel: &dyn Fn() -> Result<()>,
) -> Result<()> {
    for (ordinal, change) in input.changes.iter().enumerate() {
        super::super::check_step(ordinal, cancel)?;
        if let Identity::Right((_, key, sequence)) = &change.identity {
            owners.register(key);
            if input.kinds[1] == SequenceKind::Canonical {
                owners.register(sequence);
            }
        }
    }
    Ok(())
}

fn register_left_owners(
    owners: &mut OwnerWriter,
    input: &Input<'_>,
    cancel: &dyn Fn() -> Result<()>,
) -> Result<()> {
    for (_, data, head) in input.left {
        let mut keys = data.keys.iter().flatten().collect::<Vec<_>>();
        keys.sort_unstable();
        for (ordinal, key) in keys.into_iter().enumerate() {
            super::super::check_step(ordinal, cancel)?;
            owners.register(key);
        }
        if input.kinds[0] == SequenceKind::Canonical {
            for (ordinal, sequence) in data.sequences.iter().skip(*head).enumerate() {
                super::super::check_step(ordinal, cancel)?;
                owners.register(sequence.as_ref());
            }
        }
    }
    Ok(())
}

fn register_bucket_owners(
    owners: &mut OwnerWriter,
    input: &Input<'_>,
    cancel: &dyn Fn() -> Result<()>,
) -> Result<()> {
    for (ordinal, bucket) in input.buckets.iter().enumerate() {
        super::super::check_step(ordinal, cancel)?;
        owners.register(&bucket.key);
    }
    Ok(())
}

fn retained_owner_credit(
    owners: &OwnerWriter,
    live_allocation: &impl Fn(usize) -> bool,
    reserve: &mut impl FnMut(u64) -> Result<MemoryReservation>,
    name: &str,
) -> Result<MemoryReservation> {
    let retired = owners
        .allocations()
        .filter(|(address, _)| !live_allocation(*address))
        .try_fold(0, |bytes, (_, allocation)| checked(name, bytes, allocation))?;
    let owner_credit = reserve(checked(name, owners.metadata_bytes(), retired)?)?;
    Ok(owner_credit)
}

fn wire_reservation(
    input: &Input<'_>,
    owner_bytes: u64,
    reserve: &mut impl FnMut(u64) -> Result<MemoryReservation>,
    limit: u64,
    name: &str,
) -> Result<(usize, MemoryReservation)> {
    let length = checked(name, row_bytes(input, name)?, owner_bytes)?;
    if length > limit {
        return Err(mismatch("ASOF log delta exceeds byte limits"));
    }
    let lease = reserve(checked(name, length, 256)?)?;
    let capacity =
        usize::try_from(length).map_err(|_| mismatch("ASOF log length exceeds address domain"))?;
    Ok((capacity, lease))
}

fn encode_segment(
    input: &Input<'_>,
    previous: usize,
    owner_capacity: usize,
    owners: &OwnerWriter,
    capacity: usize,
    lease: MemoryReservation,
    cancel: &dyn Fn() -> Result<()>,
) -> Result<StateSegment> {
    let mut bytes = Vec::with_capacity(capacity);
    write_header(&mut bytes, input, previous, owner_capacity);
    owners.write_since_checked(&mut bytes, previous, cancel)?;
    write_rows(&mut bytes, input, owners, cancel)?;
    if bytes.len() != capacity {
        return Err(mismatch("ASOF log delta encoded length differs"));
    }
    #[cfg(test)]
    super::super::super::cost::index_bytes(bytes.len());
    let segment = StateSegment::new(bytes).with_owner(Arc::new(lease));
    Ok(segment)
}

fn write_rows(
    bytes: &mut Vec<u8>,
    input: &Input<'_>,
    owners: &OwnerWriter,
    cancel: &dyn Fn() -> Result<()>,
) -> Result<()> {
    for (ordinal, change) in input.changes.iter().enumerate() {
        super::super::check_step(ordinal, cancel)?;
        write_change(bytes, change, input.kinds[1], owners);
    }
    for (ordinal, (batch, data, head)) in input.left.iter().enumerate() {
        super::super::check_step(ordinal, cancel)?;
        super::super::write_left_checked(
            bytes,
            *batch,
            data,
            *head,
            input.kinds[0],
            owners,
            cancel,
        )?;
    }
    write_buckets(bytes, input, owners, cancel)
}

fn write_buckets(
    bytes: &mut Vec<u8>,
    input: &Input<'_>,
    owners: &OwnerWriter,
    cancel: &dyn Fn() -> Result<()>,
) -> Result<()> {
    for (ordinal, bucket) in input.buckets.iter().enumerate() {
        super::super::check_step(ordinal, cancel)?;
        owners.reference(bytes, &bucket.key);
        bytes.push(u8::from(bucket.state.is_some()));
        if let Some((rows, capacities)) = bucket.state {
            super::super::put(bytes, rows);
            for capacity in capacities {
                super::super::put(bytes, capacity as u64);
            }
        }
    }
    Ok(())
}

fn write_header(bytes: &mut Vec<u8>, input: &Input<'_>, previous: usize, owner_capacity: usize) {
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&[
        input.kinds[0].flag(),
        input.kinds[1].flag(),
        0,
        0,
        0,
        0,
        0,
        0,
    ]);
    for value in input.capacities.into_iter().chain(input.counts).chain([
        previous,
        owner_capacity,
        input.changes.len(),
        input.left.len(),
        input.buckets.len(),
    ]) {
        super::super::put(bytes, value as u64);
    }
}

fn write_change(bytes: &mut Vec<u8>, change: &Change, kind: SequenceKind, owners: &OwnerWriter) {
    match &change.identity {
        Identity::Left((_, id)) => {
            bytes.push(0);
            super::super::put(bytes, *id);
        }
        Identity::Right((time, key, sequence)) => {
            #[cfg(test)]
            super::super::super::cost::index_rows(1);
            bytes.push(1);
            bytes.extend_from_slice(&time.to_le_bytes());
            owners.reference(bytes, key);
            super::super::write_sequence(bytes, kind, sequence, owners);
        }
    }
    for version in [change.before, change.after] {
        bytes.push(u8::from(version.is_some()));
        match version {
            None => {}
            Some(Version::Left { rows, capacities }) => {
                super::super::put(bytes, rows);
                for capacity in capacities {
                    super::super::put(bytes, capacity as u64);
                }
            }
            Some(Version::Right { tag, payload }) => {
                bytes.push(tag);
                if let Some(((_, batch), row)) = payload {
                    super::super::put(bytes, batch);
                    bytes.extend_from_slice(&row.to_le_bytes());
                }
            }
        }
    }
}
