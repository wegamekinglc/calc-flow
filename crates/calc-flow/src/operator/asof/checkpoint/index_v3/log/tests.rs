use super::*;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};

fn pool() -> Arc<dyn MemoryPool> {
    Arc::new(GreedyMemoryPool::new(1 << 20))
}

fn reserve(pool: &Arc<dyn MemoryPool>, bytes: u64) -> Result<MemoryReservation> {
    let lease = MemoryConsumer::new("asof-delta-test").register(pool);
    lease
        .try_grow(usize::try_from(bytes).unwrap())
        .map_err(|_| mismatch("test pool refuses credit"))?;
    Ok(lease)
}

fn left(rows: u64) -> Version {
    let count = usize::try_from(rows).unwrap();
    Version::Left {
        rows,
        capacities: [count, 0, 1, 1, count, count],
    }
}

#[test]
fn test_a12_journal_insert_delete_coalesces_and_releases_actual_credit() {
    let pool = pool();
    let mut journal = journal::Journal::default();
    let insert = Change {
        identity: Identity::Left((0, 1)),
        before: None,
        after: Some(left(3)),
    };
    let prepared = journal
        .prepare(
            &[insert.clone()],
            |_| true,
            |bytes| reserve(&pool, bytes),
            "asof",
        )
        .unwrap();
    assert!(pool.reserved() > 0);
    journal.install(prepared);
    assert_eq!(journal.changes(), &[insert]);
    let delete = Change {
        identity: Identity::Left((0, 1)),
        before: Some(left(3)),
        after: None,
    };
    let empty = journal
        .prepare(&[delete], |_| true, |bytes| reserve(&pool, bytes), "asof")
        .unwrap();
    assert!(empty.is_empty());
    journal.install(empty);
    assert_eq!(journal.bytes(), 0);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_a12_journal_payload_transition_refusal_preserves_installed_state_and_credit() {
    let pool = pool();
    let key = Encoding::from_slice(b"retired-key-allocation");
    let identity = Identity::Right((4, key.clone(), Encoding::from_slice(b"s")));
    let payload = Version::Right {
        tag: 1,
        payload: Some(((1, 7), 0)),
    };
    let identity_only = Version::Right {
        tag: 2,
        payload: None,
    };
    let mut journal = journal::Journal::default();
    let prepared = journal
        .prepare(
            &[Change {
                identity: identity.clone(),
                before: Some(payload),
                after: Some(identity_only),
            }],
            |_| false,
            |bytes| reserve(&pool, bytes),
            "asof",
        )
        .unwrap();
    journal.install(prepared);
    let retained = pool.reserved();
    assert!(journal.bytes() >= key.allocation().unwrap().1);
    let before = journal.changes().to_vec();
    let mut calls = 0;
    let refused = journal.prepare(
        &[Change {
            identity,
            before: Some(identity_only),
            after: None,
        }],
        |_| false,
        |bytes| {
            calls += 1;
            if calls == 2 {
                return Err(mismatch("test candidate refusal"));
            }
            reserve(&pool, bytes)
        },
        "asof",
    );
    assert!(refused.is_err());
    assert_eq!(journal.changes(), before);
    assert_eq!(pool.reserved(), retained);
    drop(journal);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_a12_row_delta_encodes_one_identity_and_restores_canonical_owner_under_credit() {
    let pool = pool();
    let key = Encoding::from_slice(b"canonical-hot-key");
    let changes = [Change {
        identity: Identity::Right((
            4,
            key.clone(),
            SequenceKind::Unsigned(8).decode_integer(&u64::MAX.to_le_bytes()),
        )),
        before: None,
        after: Some(Version::Right {
            tag: 2,
            payload: None,
        }),
    }];
    let buckets = [BucketCut {
        key: key.clone(),
        state: Some((1, [0; 5])),
    }];
    let input = Input {
        capacities: [0, 0, 1, 4, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        counts: [0, 1, 0],
        kinds: [SequenceKind::Canonical, SequenceKind::Unsigned(8)],
        changes: &changes,
        left: &[],
        buckets: &buckets,
    };
    super::super::super::cost::take();
    let encoded = encode(
        &input,
        &OwnerWriter::default(),
        |_| false,
        |bytes| reserve(&pool, bytes),
        4096,
        "asof",
    )
    .unwrap();
    let cost = super::super::super::cost::take();
    assert_eq!(cost.index_rows, 1);
    assert!(cost.index_bytes <= 4096);
    assert!(pool.reserved() >= encoded.segment.bytes().len() + 256);
    assert_eq!(encoded.owners.count(), 1);
    assert!(encoded.owner_credit.size() > 0);
    let mut empty = Cursor::new(&[0; 8], 4096);
    let owners = OwnerReader::read(&mut empty).unwrap();
    let charge = restore_charge(encoded.segment.bytes(), 0, input.kinds, 100, 4096).unwrap();
    let decoded = decode(
        encoded.segment.bytes(),
        &owners,
        &BTreeMap::new(),
        input.kinds,
        &crate::AsofStateLimits::new(100, 4096).unwrap(),
        reserve(&pool, charge).unwrap(),
        || Ok(()),
    )
    .unwrap();
    assert_eq!(decoded.changes, changes);
    assert_eq!(decoded.capacities, input.capacities);
    assert_eq!(decoded.counts, input.counts);
    assert_eq!(decoded.buckets[0].key, key);
    assert_eq!(decoded.buckets[0].state, Some((1, [0; 5])));
    assert!(decoded.left.is_empty());
    assert_eq!(decoded.owners.count(), 1);
    assert_eq!(decoded.workspace.size() as u64, charge);
    let retained = pool.reserved();
    let mut corrupt = encoded.segment.bytes().to_vec();
    corrupt[10] = 1;
    assert!(restore_charge(&corrupt, 0, input.kinds, 100, 4096).is_err());
    assert_eq!(pool.reserved(), retained);
    drop(decoded);
    drop(encoded);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_a12_materialized_ancestry_skips_progress_cut_and_rejects_wrong_predecessor() {
    let pool = pool();
    let fingerprint = "0".repeat(64);
    let (base, base_segment) = chain::encode(
        1,
        1,
        None,
        &fingerprint,
        1,
        b"base",
        reserve(&pool, 388).unwrap(),
    )
    .unwrap();
    let (delta, delta_segment) = chain::encode(
        1,
        3,
        Some(&base),
        &fingerprint,
        1,
        b"delta",
        reserve(&pool, 389).unwrap(),
    )
    .unwrap();
    let mut segments = BTreeMap::new();
    segments.insert(base.name(), base_segment);
    segments.insert(delta.name(), delta_segment);
    let inventory = [base.clone(), delta];
    let workspace = reserve(&pool, 2048).unwrap();
    let frames =
        chain::validate_chain(
            &inventory,
            &segments,
            4,
            &fingerprint,
            &workspace,
            || Ok(()),
        )
        .unwrap();
    assert_eq!(frames[1].preceding_epoch, 1);
    assert_eq!(frames[1].records, 1);
    assert_eq!(frames[1].body, b"delta");
    let mut incorrect = base.clone();
    incorrect.epoch = 2;
    let (bad, bad_segment) = chain::encode(
        1,
        3,
        Some(&incorrect),
        &fingerprint,
        1,
        b"delta",
        reserve(&pool, 389).unwrap(),
    )
    .unwrap();
    let mut bad_segments = segments.clone();
    bad_segments.insert(bad.name(), bad_segment);
    assert!(
        chain::validate_chain(
            &[base, bad],
            &bad_segments,
            4,
            &fingerprint,
            &workspace,
            || Ok(())
        )
        .is_err()
    );
    let mut cancellation_calls = 0;
    assert!(
        chain::validate_chain(&inventory, &segments, 4, &fingerprint, &workspace, || {
            cancellation_calls += 1;
            Err(mismatch("test cancellation"))
        })
        .is_err()
    );
    assert_eq!(cancellation_calls, 1);
}

#[test]
fn test_a12_compaction_bounds_delta_count_and_retired_owner_bytes() {
    let mut inventory = vec![chain::Descriptor {
        generation: 1,
        ordinal: 0,
        epoch: 1,
        sha256: "0".repeat(64),
        bytes: 4096,
    }];
    assert!(!chain::requires_compaction(
        &inventory, 64, 64, 0, 4096, 8192
    ));
    assert!(chain::requires_compaction(
        &inventory, 64, 64, 4096, 4096, 16384
    ));
    assert!(chain::requires_compaction(
        &inventory, 4096, 64, 0, 4096, 16384
    ));
    for ordinal in 1..=32 {
        inventory.push(chain::Descriptor {
            generation: 1,
            ordinal,
            epoch: u64::from(ordinal) + 1,
            sha256: "0".repeat(64),
            bytes: 64,
        });
    }
    assert!(chain::requires_compaction(
        &inventory, 64, 64, 0, 4096, 16384
    ));
}
