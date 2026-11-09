use super::*;
use std::collections::{BTreeMap, VecDeque};

#[test]
fn test_prefix_metadata_quote_covers_real_borrowed_entity_tails() {
    for (entities, rows, retain) in [
        (0, 0, false),
        (1, 1, true),
        (2, 129, false),
        (2, 129, true),
        (17, 4_097, true),
    ] {
        let keys = (0..entities)
            .map(|index| index.to_string())
            .collect::<Vec<_>>();
        let mut tails: BTreeMap<&String, VecDeque<i64>> = BTreeMap::new();
        let observation = allocation_counter::measure(|| {
            for index in 0..rows {
                let tail = tails.entry(&keys[index % entities]).or_default();
                if retain {
                    tail.push_back(i64::try_from(index).unwrap());
                }
            }
        });
        let quote = plan::<&String>(rows, entities, retain, "prefix").unwrap();
        assert!(
            quote >= observation.bytes_max,
            "entities={entities}, rows={rows}, observed={observation:?}, quote={quote}"
        );
    }
}

#[test]
fn test_prefix_metadata_quote_counts_identity_even_without_retained_rows() {
    assert_eq!(plan::<&String>(0, 0, false, "prefix").unwrap(), 0);
    assert!(plan::<&String>(4, 2, false, "prefix").unwrap() > 0);
    assert!(plan::<&String>(usize::MAX, usize::MAX, true, "prefix").is_err());
}

#[test]
fn test_prefix_global_work_cursor_bounds_nested_cancellation() {
    let mut position = 0;
    let mut completed = 0;
    let error = (0..4_097)
        .try_for_each(|_| {
            tick(&mut position, &mut || {
                Err(CalcFlowError::Cancelled {
                    run_id: "prefix".into(),
                })
            })?;
            completed += 1;
            Ok::<(), CalcFlowError>(())
        })
        .unwrap_err();
    assert_eq!(completed, 4_096);
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
}
