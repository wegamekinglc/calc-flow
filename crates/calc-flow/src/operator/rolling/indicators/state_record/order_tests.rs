use super::*;
use datafusion::scalar::ScalarValue;

#[test]
fn test_prepaid_borrowed_sort_preserves_scalar_order_and_owners() {
    let values = [
        ScalarValue::Utf8(Some("zeta".into())),
        ScalarValue::Utf8(Some("alpha".into())),
        ScalarValue::Utf8(None),
        ScalarValue::Utf8(Some("nested".into())),
    ];
    let mut keys = vec![&values[1], &values[0], &values[2], &values[0], &values[3]];
    let mut spare = Vec::with_capacity(keys.len());
    let expected = {
        let mut result = keys.clone();
        result.sort_by(|left, right| left.partial_cmp(right).unwrap());
        result.dedup();
        result
    };
    let mut sorted = None;
    let allocations = allocation_counter::measure(|| {
        sorted = Some(
            sort_unique(
                &mut keys,
                &mut spare,
                |left: &ScalarValue, right| left.partial_cmp(right).unwrap(),
                &mut || Ok(()),
            )
            .unwrap(),
        );
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(sorted.unwrap(), expected.len());
    assert_eq!(keys, expected);
    assert!(
        keys.iter()
            .all(|key| values.iter().any(|value| std::ptr::eq(*key, value)))
    );
}

#[test]
fn test_prepaid_borrowed_sort_checks_capacity_and_long_cancellation() {
    let values = (0..16_385).rev().collect::<Vec<_>>();
    let mut keys = values.iter().collect::<Vec<_>>();
    let mut spare = Vec::new();
    assert!(sort_unique(&mut keys, &mut spare, Ord::cmp, &mut || Ok(())).is_err());
    assert_eq!(keys.len(), values.len());
    spare.reserve_exact(values.len());
    let mut checks = 0;
    let error = sort_unique(&mut keys, &mut spare, Ord::cmp, &mut || {
        checks += 1;
        if checks == 4 {
            Err(CalcFlowError::Cancelled {
                run_id: "ordered-entities".into(),
            })
        } else {
            Ok(())
        }
    })
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    assert_eq!(checks, 4);
}

#[test]
fn test_prepaid_borrowed_sort_empty_and_duplicate_lengths() {
    for length in [0, 1, 2, 3, 31, 4_097] {
        let values = (0..length).map(|value| value % 7).collect::<Vec<_>>();
        let mut keys = values.iter().collect::<Vec<_>>();
        let mut spare = Vec::with_capacity(length);
        sort_unique(&mut keys, &mut spare, Ord::cmp, &mut || Ok(())).unwrap();
        assert_eq!(
            keys.into_iter().copied().collect::<Vec<_>>(),
            (0..length.min(7)).collect::<Vec<_>>()
        );
    }
}
