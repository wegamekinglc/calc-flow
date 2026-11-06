use super::*;

type LegacyStorage = (Vec<Vec<ScalarValue>>, Vec<ScalarValue>);
type LegacyGroup = (
    Arc<[u8]>,
    Arc<[ScalarValue]>,
    LegacyStorage,
    MemoryReservation,
);
type LegacyCandidate = (
    Vec<Option<Box<dyn Accumulator>>>,
    LegacyGroup,
    Option<Box<variable_extrema::Bounds>>,
);

#[test]
fn test_grouped_inline_compact_layout_preserves_generic_container_sizes() {
    eprintln!(
        "Inline state layout: ScalarValue={} payload={}/{} Group={}/{} Candidate={}/{} map-row={}/{} plan={}",
        size_of::<ScalarValue>(),
        size_of::<GroupAggregates>(),
        size_of::<LegacyStorage>(),
        size_of::<Group>(),
        size_of::<LegacyGroup>(),
        size_of::<Candidate>(),
        size_of::<LegacyCandidate>(),
        size_of::<(usize, Candidate)>(),
        size_of::<(usize, LegacyCandidate)>(),
        size_of::<IncrementalSql>()
    );
    assert!(size_of::<GroupAggregates>() <= size_of::<LegacyStorage>());
    assert!(size_of::<Group>() <= size_of::<LegacyGroup>());
    assert!(size_of::<Candidate>() <= size_of::<LegacyCandidate>());
    assert!(size_of::<(usize, Candidate)>() <= size_of::<(usize, LegacyCandidate)>());
    assert!(
        size_of::<IncrementalSql>() <= 4096,
        "cached inline kind must fit the existing plan structural floor"
    );
}

async fn full_container_case(single: bool) -> allocation_counter::AllocationInfo {
    let fixture = allocation_fixture(single).await;
    let mut retained = None;
    let allocations = allocation_counter::measure(|| {
        let mut candidates = CandidateGroups::new(false);
        for (rank, previous) in fixture.plan.groups.iter().enumerate() {
            let candidate = updated_candidate(&fixture, previous, rank);
            candidates
                .insert(rank, candidate, "inline-container")
                .unwrap();
        }
        let container = fixture
            .plan
            .prepare_container(&candidates, GROUPS, "inline-container")
            .unwrap();
        retained = Some((candidates, container));
    });
    let (mut candidates, container) = retained.unwrap();
    let paid = candidates
        .values()
        .map(|candidate| candidate.group.reservation.size())
        .sum::<usize>()
        + container.reservation.size();
    eprintln!(
        "Full staged inline={single}: heap={allocations:?}, paid={paid}, map_capacity={}, group_capacity={}, index_capacity={}",
        candidates.generic_mut().capacity(),
        container.groups.capacity(),
        container.index.capacity()
    );
    assert!(allocations.bytes_max <= u64::try_from(paid).unwrap());
    assert!(container.groups.capacity() >= GROUPS);
    assert!(container.index.capacity() >= GROUPS);
    drop((candidates, container, fixture.partial, fixture.plan));
    assert_eq!(fixture.runtime.incremental_memory_pool().reserved(), 0);
    allocations
}

#[tokio::test]
async fn test_grouped_inline_full_candidate_map_and_group_container_are_funded() {
    let single = full_container_case(true).await;
    let mixed = full_container_case(false).await;
    assert!(single.bytes_current > 0);
    assert!(
        mixed.bytes_current > single.bytes_current,
        "generic aggregate state must remain represented in the full owned footprint"
    );
}

#[tokio::test]
async fn test_grouped_inline_candidate_exact_credit_and_one_byte_shortfall_are_atomic() {
    for single in [true, false] {
        let fixture = allocation_fixture(single).await;
        let previous = &fixture.plan.groups[0];
        let candidate = fixture
            .plan
            .candidate(Some(previous), &previous.key, None, "inline-budget")
            .unwrap();
        let credit = candidate.group.reservation.size();
        drop(candidate);
        let baseline = fixture.runtime.compact_state_bytes();
        let before = fixture
            .plan
            .export_native_state("inline-budget", || Ok(()))
            .unwrap();
        fixture
            .runtime
            .set_compact_state_limit(
                Some(u64::try_from(baseline + credit - 1).unwrap()),
                "inline-budget",
            )
            .unwrap();
        assert!(
            fixture
                .plan
                .candidate(Some(previous), &previous.key, None, "inline-budget")
                .is_err()
        );
        assert_eq!(fixture.runtime.compact_state_bytes(), baseline);
        let after = fixture
            .plan
            .export_native_state("inline-budget", || Ok(()))
            .unwrap();
        assert_eq!(before.records(), after.records());
        fixture
            .runtime
            .set_compact_state_limit(
                Some(u64::try_from(baseline + credit).unwrap()),
                "inline-budget",
            )
            .unwrap();
        let admitted = fixture
            .plan
            .candidate(Some(previous), &previous.key, None, "inline-budget")
            .unwrap();
        assert_eq!(fixture.runtime.compact_state_bytes(), baseline + credit);
        drop(admitted);
        assert_eq!(fixture.runtime.compact_state_bytes(), baseline);
    }
}
