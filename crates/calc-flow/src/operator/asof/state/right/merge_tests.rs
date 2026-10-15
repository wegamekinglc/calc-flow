use super::*;
use crate::{CalcFlowError, EventTime, StreamAsofJoinStatus};

fn sequence(kind: SequenceKind, value: u8) -> Encoding {
    match kind.width() {
        Some(width) => kind.decode_integer(&u64::from(value).to_le_bytes()[..width]),
        None => Encoding::from_slice(&[value; 32]),
    }
}

fn wire(bucket: &RightBucket) -> Vec<(i64, Encoding, Option<RowRef>, u8)> {
    bucket
        .checkpoint_rows()
        .map(|((time, sequence), row, tag)| (*time, sequence.into_owned(), row.copied(), tag))
        .collect()
}

fn retained(kind: SequenceKind) -> RightBucket {
    let mut bucket = RightBucket::with_sequence_kind(kind);
    for time in 0..512 {
        bucket.insert(
            (time, sequence(kind, 0)),
            Some(RowRef::fixture(u32::try_from(time).unwrap())),
        );
    }
    for time in [512, 1_000, 250] {
        bucket.insert((time, sequence(kind, 5)), None);
    }
    let mut status = StreamAsofJoinStatus::default();
    status.left.watermark_micros = Some(EventTime::from_micros(201));
    status.right.watermark_micros = Some(EventTime::from_micros(0));
    bucket.evict(&status, 0, 201);
    assert!(bucket.payloads().head > 0);
    assert!(!bucket.general_identities.is_empty());
    bucket
}

#[test]
fn bulk_admission_preserves_retired_prefix_and_identity_history() {
    let kinds = [SequenceKind::Canonical].into_iter().chain(
        [1, 2, 4, 8]
            .into_iter()
            .flat_map(|width| [SequenceKind::Signed(width), SequenceKind::Unsigned(width)]),
    );
    for kind in kinds {
        let bucket = retained(kind);
        let before = wire(&bucket);
        let rows = [-10, 300, 1_100]
            .into_iter()
            .map(|time| ((time, sequence(kind, 1)), RowRef::fixture(900)))
            .collect::<Vec<_>>();
        let mut expected = bucket.clone();
        expected.reserve_payloads(rows.len());
        for (order, row) in &rows {
            expected.insert_admitted(order.clone(), *row);
        }
        let merged = bucket.with_admitted(&rows, &|| Ok(())).unwrap();
        let mut in_place = bucket.clone();
        in_place.reserve_payloads(rows.len());
        let allocation = allocation_counter::measure(|| {
            in_place
                .merge_admitted(
                    rows.iter()
                        .map(|((time, sequence), row)| (*time, sequence.clone(), *row)),
                    &|| Ok(()),
                )
                .unwrap();
        });
        assert_eq!(
            allocation.count_total, 0,
            "reserved merge allocated row intermediates: {allocation:?}"
        );
        assert_eq!(wire(&in_place), wire(&expected));
        eprintln!("reserved per-key merge {kind:?}: {allocation:?}");
        let columnar = bucket
            .with_admitted_columns(
                rows.iter()
                    .map(|((time, sequence), row)| (*time, sequence.clone(), *row)),
                &|| Ok(()),
            )
            .unwrap();
        assert_eq!(wire(&columnar), wire(&expected));
        assert_eq!(
            columnar.checkpoint_capacities(),
            expected.checkpoint_capacities()
        );
        assert_eq!(wire(&merged), wire(&expected));
        assert_eq!(
            merged.checkpoint_capacities(),
            expected.checkpoint_capacities()
        );
        assert_eq!(
            merged.metadata_bytes(),
            bucket.projected_admission_bytes(rows.len())
        );
        for tolerance in [0, 5, 1_000] {
            let expected_matches = (-20..1_200)
                .map(|time| merged.candidate(time, tolerance).copied())
                .collect::<Vec<_>>();
            GENERAL_SEARCHES.replace(0);
            let mut cursor = merged.cursor_at(-20);
            let actual = (-20..1_200)
                .map(|time| {
                    merged
                        .candidate_monotonic(time, tolerance, &mut cursor)
                        .copied()
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected_matches);
            assert_eq!(
                GENERAL_SEARCHES.replace(0),
                0,
                "merge repeatedly searched identity history"
            );
            for time in -20..1_200 {
                assert_eq!(
                    merged.candidate(time, tolerance),
                    expected.candidate(time, tolerance)
                );
            }
        }
        assert_eq!(wire(&bucket), before);
    }
}

#[test]
fn cancelled_bulk_admission_preserves_original_bucket() {
    let bucket = retained(SequenceKind::Canonical);
    let before = wire(&bucket);
    let rows = (
        (2_000, sequence(SequenceKind::Canonical, 1)),
        RowRef::fixture(900),
    );
    let checks = std::cell::Cell::new(0);
    let error = bucket.with_admitted(&[rows], &|| {
        checks.set(checks.get() + 1);
        if checks.get() == 4 {
            Err(CalcFlowError::Cancelled {
                run_id: "merge".into(),
            })
        } else {
            Ok(())
        }
    });
    assert!(matches!(error, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(wire(&bucket), before);
    checks.set(0);
    let error = bucket.with_admitted_columns(
        std::iter::once((
            2_000,
            sequence(SequenceKind::Canonical, 1),
            RowRef::fixture(900),
        )),
        &|| {
            checks.set(checks.get() + 1);
            if checks.get() == 2 {
                Err(CalcFlowError::Cancelled {
                    run_id: "columnar-merge".into(),
                })
            } else {
                Ok(())
            }
        },
    );
    assert!(matches!(error, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(wire(&bucket), before);
}
