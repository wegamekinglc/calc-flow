mod checkpoint_v4_support;

use std::{collections::BTreeSet, sync::Arc};

use checkpoint_v4_support::{
    golden::{GoldenBundle, GoldenCategory, golden_cases, validate},
    protocol::{Event, FailureKind, Fault, ProtocolError, ProtocolHarness, Recovery, SinkDecision},
    table::{FakePersistence, FakeTable},
};

fn write(table: &mut FakeTable, group: u32, slot: u32, bytes: &[u8]) {
    table.begin();
    table.append(group, slot, bytes);
    table.commit();
}

fn seeded_protocol(fault: Option<Fault>) -> ProtocolHarness {
    let mut harness = ProtocolHarness::new();
    write(harness.table_mut(), 0, 0, b"seed");
    match fault {
        Some(fault) => harness.request_with_fault(1, fault),
        None => harness.request(1),
    }
    .unwrap();
    harness
}

fn assert_order(events: &[Event], expected: &[Event]) {
    let mut offset = 0;
    for event in expected {
        let relative = events[offset..]
            .iter()
            .position(|candidate| candidate == event)
            .unwrap_or_else(|| panic!("missing {event:?} after index {offset}: {events:?}"));
        offset += relative + 1;
    }
}

fn assert_failed(harness: &ProtocolHarness, recovery: Recovery, decision: SinkDecision) {
    assert_eq!(harness.recovery(), Some(recovery));
    assert_eq!(harness.sink_decision(), decision);
    assert!(harness.stopped());
    assert!(harness.workers_joined());
    assert!(harness.lease_released());
    assert_eq!(harness.reservation_bytes(), 0);
    assert!(harness.source_acks().is_empty());
    assert_eq!(harness.last_completed(), None);
}

#[test]
fn test_fake_table_preserves_frozen_bytes_and_cow_generations() {
    let mut table = FakeTable::seed(true, &[]);
    write(&mut table, 0, 0, b"a");
    let original_storage = Arc::as_ptr(table.page(0, 0).unwrap().storage());
    write(&mut table, 0, 0, b"b");
    assert_eq!(
        Arc::as_ptr(table.page(0, 0).unwrap().storage()),
        original_storage
    );

    let frozen = table.freeze(1).unwrap();
    let frozen_id = table.page(0, 0).unwrap().id;
    assert_eq!(frozen.bytes_for(frozen_id), Some(&b"ab"[..]));
    write(&mut table, 0, 0, b"c");
    let current = table.page(0, 0).unwrap();
    assert_eq!(current.bytes(), b"abc");
    assert_eq!(current.id.generation, frozen_id.generation + 1);
    assert_ne!(Arc::as_ptr(current.storage()), original_storage);
    assert_eq!(frozen.bytes_for(frozen_id), Some(&b"ab"[..]));

    let cow_storage = Arc::as_ptr(current.storage());
    write(&mut table, 0, 0, b"d");
    assert_eq!(
        Arc::as_ptr(table.page(0, 0).unwrap().storage()),
        cow_storage
    );
    assert_eq!(table.page(0, 0).unwrap().bytes(), b"abcd");
    assert_eq!(frozen.bytes_for(frozen_id), Some(&b"ab"[..]));
}

#[test]
fn test_fake_table_undo_restores_refs_dirty_lists_and_counters() {
    let mut table = FakeTable::seed(true, &[(0, 0, b"a")]);
    write(&mut table, 0, 0, b"b");
    let before = table.page(0, 0).unwrap();
    let before_id = before.id;
    let before_storage = Arc::as_ptr(before.storage());
    let before_bytes = table.logical_bytes();

    table.begin();
    table.append(0, 0, b"c");
    table.append(0, 1, b"new");
    table.rollback();
    let restored = table.page(0, 0).unwrap();
    assert_eq!(restored.id, before_id);
    assert_eq!(Arc::as_ptr(restored.storage()), before_storage);
    assert_eq!(restored.bytes(), b"ab");
    assert!(table.page(0, 1).is_none());
    assert_eq!(table.dirty_slots(0), [0]);
    assert_eq!(table.first_dirty_slots(0), [0]);
    assert_eq!(table.logical_bytes(), before_bytes);

    table.begin();
    table.delete(0, 0);
    table.rollback();
    assert_eq!(
        Arc::as_ptr(table.page(0, 0).unwrap().storage()),
        before_storage
    );
    assert_eq!(table.page(0, 0).unwrap().bytes(), b"ab");
    assert_eq!(table.dirty_slots(0), [0]);

    table.begin();
    table.append(0, 0, b"ignored");
    table.delete(0, 0);
    table.rollback();
    assert_eq!(
        Arc::as_ptr(table.page(0, 0).unwrap().storage()),
        before_storage
    );
    assert_eq!(table.page(0, 0).unwrap().bytes(), b"ab");
    assert_eq!(table.dirty_slots(0), [0]);

    table.begin();
    assert!(table.freeze(1).unwrap_err().contains("open undo"));
    table.rollback();

    let mut off = FakeTable::seed(false, &[]);
    write(&mut off, 0, 0, b"off");
    assert!(off.dirty_slots(0).is_empty());
    assert!(off.first_dirty_slots(0).is_empty());
}

#[test]
fn test_first_dirty_marks_append_only_once() {
    let mut table = FakeTable::seed(true, &[]);
    table.begin();
    table.append(0, 1, b"a");
    table.append(0, 1, b"b");
    table.append(0, 0, b"c");
    table.append(0, 1, b"d");
    table.commit();
    assert_eq!(table.first_dirty_slots(0), [1, 0]);
    assert_eq!(table.dirty_slots(0), [0, 1]);
}

#[test]
fn test_protocol_forwards_before_gated_persistence_and_commits_in_order() {
    let mut harness = seeded_protocol(None);
    harness.append_data(1, 0, 0, b"post").unwrap();
    assert_order(
        harness.events(),
        &[
            Event::SourceCut(1),
            Event::CaptureAck(1),
            Event::Barrier(1),
            Event::Data(1),
        ],
    );
    assert!(!harness.events().contains(&Event::Persisted(1)));
    assert!(harness.manifest_epoch().is_none());
    assert!(harness.source_acks().is_empty());
    assert!(harness.reservation_bytes() > 0);

    assert_eq!(
        harness.publish_manifest(1),
        Err(ProtocolError::Prerequisite("persistence has not resolved"))
    );
    assert_eq!(
        harness.commit_sink(1),
        Err(ProtocolError::Prerequisite("manifest is not durable"))
    );
    assert_eq!(
        harness.acknowledge_source(1),
        Err(ProtocolError::Prerequisite("sink has not committed"))
    );

    harness.resolve_persistence(1).unwrap();
    assert_eq!(
        harness.publish_manifest(1),
        Err(ProtocolError::Prerequisite("sink has not pre-committed"))
    );
    assert!(harness.manifest_epoch().is_none());
    harness.precommit_sink(1).unwrap();
    harness.publish_manifest(1).unwrap();
    harness.commit_sink(1).unwrap();
    harness.acknowledge_source(1).unwrap();
    assert_order(
        harness.events(),
        &[
            Event::Persisted(1),
            Event::SinkPrecommitted(1),
            Event::ManifestDurable(1),
            Event::SinkCommitted(1),
            Event::SourceAcked(1),
            Event::Completed(1),
        ],
    );
    assert_eq!(harness.last_completed(), Some(1));
    assert_eq!(harness.source_acks(), [1]);
    assert_eq!(harness.manifest_epoch(), Some(1));
    assert_eq!(harness.reservation_bytes(), 0);
    assert_eq!(
        harness.durable_image().page((0, 0)).unwrap().bytes(),
        b"seed"
    );
    assert_eq!(harness.table().page(0, 0).unwrap().bytes(), b"seedpost");
}

#[test]
fn test_protocol_backpressures_next_epoch_and_rejects_early_barrier() {
    let mut harness = seeded_protocol(None);
    let events = harness.events().len();
    assert_eq!(harness.request(2), Err(ProtocolError::Backpressure(1)));
    assert_eq!(harness.events().len(), events);
    assert_eq!(harness.receive_barrier(2), Err(ProtocolError::Conflict));
    assert_failed(&harness, Recovery::PreviousManifest, SinkDecision::Abort);

    let mut terminal = ProtocolHarness::new();
    terminal.request_terminal(1).unwrap();
    assert!(
        !terminal
            .events()
            .iter()
            .any(|event| matches!(event, Event::Barrier(_)))
    );
    assert!(terminal.events().contains(&Event::TerminalCapture(1)));
    assert_eq!(
        terminal.append_data(1, 0, 0, b"late"),
        Err(ProtocolError::Prerequisite("terminal ingress is ended"))
    );
    terminal.complete_epoch(1).unwrap();
    assert_eq!(terminal.last_completed(), Some(1));
}

#[test]
fn test_protocol_deduplicates_evidence_and_rejects_conflicts() {
    let mut harness = seeded_protocol(None);
    harness.resolve_persistence(1).unwrap();
    let digest = harness.persistence_digest(1).unwrap();
    let root_hashes = harness.persistence().root_hash_calls();
    harness.submit_persistence(1, &digest).unwrap();
    harness.submit_persistence(1, &digest).unwrap();
    harness.resolve_persistence(1).unwrap();
    assert_eq!(harness.persistence().root_hash_calls(), root_hashes);
    assert_eq!(
        harness.submit_persistence(1, "conflicting-content"),
        Err(ProtocolError::Conflict)
    );
    assert_failed(&harness, Recovery::PreviousManifest, SinkDecision::Abort);
    assert!(harness.manifest_epoch().is_none());

    let mut foreign = seeded_protocol(None);
    assert_eq!(
        foreign.submit_persistence(2, "future"),
        Err(ProtocolError::Conflict)
    );
    assert_failed(&foreign, Recovery::PreviousManifest, SinkDecision::Abort);
}

#[test]
fn test_persistence_counts_single_hashes_and_epoch_fsync_batches() {
    let mut table = FakeTable::seed(true, &[]);
    table.begin();
    table.append(0, 0, b"a");
    table.append(0, 1, b"bb");
    table.append(1, 0, b"c");
    table.commit();
    let ids = [(0, 0), (0, 1), (1, 0)].map(|(group, slot)| table.page(group, slot).unwrap().id);
    let first = table.freeze(1).unwrap();
    let mut persistence = FakePersistence::default();
    let first = persistence.persist(&first, None).unwrap();
    assert_eq!(first.counters.content_hashes, 5);
    assert_eq!(first.counters.file_syncs, 5);
    assert_eq!(first.counters.staging_directory_syncs, 2);
    assert_eq!(first.counters.committed_directory_syncs, 2);
    assert_eq!(first.counters.carried_pages, 0);
    for id in ids {
        assert_eq!(persistence.page_hash_calls(id), 1);
    }

    table.publish(first.image.clone());
    table.settle();
    write(&mut table, 0, 0, b"x");
    let second = table.freeze(2).unwrap();
    let second = persistence.persist(&second, None).unwrap();
    assert_eq!(second.counters.content_hashes, 2);
    assert_eq!(second.counters.file_syncs, 2);
    assert_eq!(second.counters.staging_directory_syncs, 1);
    assert_eq!(second.counters.committed_directory_syncs, 1);
    assert_eq!(second.counters.carried_pages, 2);
    assert_eq!(second.counters.carried_roots, 1);
    for id in ids {
        assert_eq!(persistence.page_hash_calls(id), 1);
    }

    table.publish(second.image.clone());
    table.settle();
    table.begin();
    table.delete(0, 1);
    table.commit();
    let third = table.freeze(3).unwrap();
    let third = persistence.persist(&third, None).unwrap();
    assert_eq!(third.counters.content_hashes, 1);
    assert_eq!(third.counters.file_syncs, 1);
    assert!(third.image.page((0, 1)).is_none());
    assert_eq!(persistence.root_hash_calls(), 4);
    assert_eq!(persistence.read_back_hashes(), 0);
}

#[test]
fn test_capture_write_checksum_panic_and_disk_full_fail_closed() {
    let mut freeze = ProtocolHarness::new();
    assert_eq!(
        freeze.request_with_fault(1, Fault::Freeze),
        Err(ProtocolError::Failed(FailureKind::Freeze))
    );
    assert!(
        !freeze
            .events()
            .iter()
            .any(|event| matches!(event, Event::CaptureAck(_)))
    );
    assert_failed(&freeze, Recovery::PreviousManifest, SinkDecision::Abort);

    for (fault, failure) in [
        (Fault::Write, FailureKind::Write),
        (Fault::Checksum, FailureKind::Checksum),
        (Fault::DiskFullState, FailureKind::DiskFull),
        (Fault::WorkerPanic, FailureKind::WorkerPanic),
    ] {
        let mut harness = seeded_protocol(Some(fault));
        harness.precommit_sink(1).unwrap();
        assert_eq!(
            harness.resolve_persistence(1),
            Err(ProtocolError::Failed(failure))
        );
        assert_failed(&harness, Recovery::PreviousManifest, SinkDecision::Abort);
        assert!(harness.manifest_epoch().is_none());
    }
}

#[test]
fn test_manifest_publication_failures_choose_abort_or_preserve() {
    for (fault, recovery, decision, installed) in [
        (
            Fault::DiskFullManifestAbsent,
            Recovery::PreviousManifest,
            SinkDecision::Abort,
            false,
        ),
        (
            Fault::ManifestUncertain,
            Recovery::RecoveryRequired,
            SinkDecision::Preserve,
            false,
        ),
        (
            Fault::ManifestInstalled,
            Recovery::RecoveryRequired,
            SinkDecision::Preserve,
            true,
        ),
    ] {
        let mut harness = seeded_protocol(Some(fault));
        harness.precommit_sink(1).unwrap();
        harness.resolve_persistence(1).unwrap();
        assert!(harness.publish_manifest(1).is_err());
        assert_failed(&harness, recovery, decision);
        assert_eq!(harness.manifest_epoch(), installed.then_some(1));
    }
}

#[test]
fn test_post_durable_sink_and_source_failures_require_recovery_without_replay() {
    let mut sink = seeded_protocol(Some(Fault::SinkCommit));
    sink.precommit_sink(1).unwrap();
    sink.resolve_persistence(1).unwrap();
    sink.publish_manifest(1).unwrap();
    assert_eq!(
        sink.commit_sink(1),
        Err(ProtocolError::Failed(FailureKind::SinkCommit))
    );
    assert_failed(&sink, Recovery::RecoveryRequired, SinkDecision::Preserve);
    assert_eq!(sink.manifest_epoch(), Some(1));

    let mut source = seeded_protocol(Some(Fault::SourceAck));
    source.precommit_sink(1).unwrap();
    source.resolve_persistence(1).unwrap();
    source.publish_manifest(1).unwrap();
    source.commit_sink(1).unwrap();
    assert_eq!(
        source.acknowledge_source(1),
        Err(ProtocolError::Failed(FailureKind::SourceAck))
    );
    assert_failed(&source, Recovery::RecoveryRequired, SinkDecision::Committed);
    assert_eq!(source.manifest_epoch(), Some(1));
    assert!(!source.events().contains(&Event::SourceAcked(1)));
}

#[test]
fn test_cancel_process_exit_and_stop_converge_on_both_publication_sides() {
    for after_durable in [false, true] {
        let mut cancel = seeded_protocol(None);
        cancel.precommit_sink(1).unwrap();
        cancel.resolve_persistence(1).unwrap();
        if after_durable {
            cancel.publish_manifest(1).unwrap();
        }
        assert!(cancel.cancel(1).is_err());
        let expected = if after_durable {
            (Recovery::RecoveryRequired, SinkDecision::Preserve)
        } else {
            (Recovery::PreviousManifest, SinkDecision::Abort)
        };
        assert_failed(&cancel, expected.0, expected.1);

        let mut exit = seeded_protocol(None);
        exit.precommit_sink(1).unwrap();
        exit.resolve_persistence(1).unwrap();
        if after_durable {
            exit.publish_manifest(1).unwrap();
        }
        assert!(exit.process_exit(1).is_err());
        assert_failed(&exit, expected.0, expected.1);

        let mut stop = seeded_protocol(None);
        stop.precommit_sink(1).unwrap();
        stop.resolve_persistence(1).unwrap();
        if after_durable {
            stop.publish_manifest(1).unwrap();
        }
        stop.stop();
        assert_failed(&stop, expected.0, expected.1);
        assert_eq!(stop.events().last(), Some(&Event::Stopped));
    }

    let mut completed = seeded_protocol(None);
    completed.complete_epoch(1).unwrap();
    completed.stop();
    assert!(completed.stopped());
    assert!(completed.workers_joined());
    assert!(completed.lease_released());
    assert_eq!(completed.reservation_bytes(), 0);
    assert_eq!(completed.last_completed(), Some(1));
    assert_eq!(completed.source_acks(), [1]);
    assert_eq!(completed.sink_decision(), SinkDecision::Committed);
}

#[test]
fn test_restore_and_replay_suffix_are_bit_identical() {
    let mut control = seeded_protocol(None);
    control.complete_epoch(1).unwrap();
    let checkpoint = control.durable_image();
    let mut restored = FakeTable::restore(&checkpoint, true);
    assert_eq!(control.table().state_bytes(), restored.state_bytes());

    for (group, slot, bytes) in [(0, 0, b"one".as_slice()), (0, 1, b"two".as_slice())] {
        write(control.table_mut(), group, slot, bytes);
        write(&mut restored, group, slot, bytes);
    }
    assert_eq!(control.table().state_bytes(), restored.state_bytes());

    let control_freeze = control.table_mut().freeze(2).unwrap();
    let restored_freeze = restored.freeze(2).unwrap();
    let mut control_objects = FakePersistence::default();
    let mut restored_objects = FakePersistence::default();
    let control_state = control_objects.persist(&control_freeze, None).unwrap();
    let restored_state = restored_objects.persist(&restored_freeze, None).unwrap();
    assert_eq!(
        control_state.evidence_digest(),
        restored_state.evidence_digest()
    );
    assert_eq!(
        control_state.image.state_bytes(),
        restored_state.image.state_bytes()
    );
}

#[test]
fn test_v4_golden_is_canonical_and_restores_exact_page_bits() {
    let bundle = GoldenBundle::valid();
    assert_eq!(
        bundle.canonical_manifest(),
        include_str!("fixtures/checkpoint-v4/manifest-v4.json").trim_end()
    );
    assert_eq!(
        bundle.canonical_recovery(),
        include_str!("fixtures/checkpoint-v4/recovery-v4.json").trim_end()
    );
    assert_eq!(
        bundle.canonical_partition0(),
        include_str!("fixtures/checkpoint-v4/partition-0.json").trim_end()
    );
    let restored = validate(&bundle).unwrap();
    assert_eq!(restored.source_sequence, 128);
    let mut expected = b"0:0:3:".to_vec();
    expected.extend_from_slice(include_bytes!("fixtures/checkpoint-v4/page-0.bin"));
    assert_eq!(restored.state_bytes, expected);
}

#[test]
fn test_v4_golden_rejects_identity_participant_partition_and_page_mutations() {
    let valid = GoldenBundle::valid();
    let cases = golden_cases();
    assert_eq!(cases.len(), 22);
    assert_eq!(
        cases
            .iter()
            .map(|case| &case.name)
            .collect::<BTreeSet<_>>()
            .len(),
        cases.len()
    );
    for (category, count) in [
        (GoldenCategory::Identity, 7),
        (GoldenCategory::Participant, 3),
        (GoldenCategory::Partition, 5),
        (GoldenCategory::Page, 6),
        (GoldenCategory::Compatibility, 1),
    ] {
        assert_eq!(
            cases
                .iter()
                .filter(|case| case.category == category)
                .count(),
            count
        );
    }
    for case in cases {
        let mutated = valid.mutated(&case);
        assert_eq!(
            validate(&mutated).unwrap_err(),
            case.expected_error,
            "{}",
            case.name
        );
    }
}
