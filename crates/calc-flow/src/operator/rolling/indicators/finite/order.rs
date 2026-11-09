use super::Check;
use crate::operator::rolling::{BufferedRow, Result, operator_error};

/// In-place heap sort keeps every comparison/swap cooperative without allocating
/// an additional row buffer. Duplicate identities fail before evaluation.
pub(super) fn sort(rows: &mut [BufferedRow], node: &str, check: Check<'_>) -> Result<()> {
    check()?;
    for root in (0..rows.len() / 2).rev() {
        sift(rows, root, rows.len(), check)?;
    }
    for end in (1..rows.len()).rev() {
        check()?;
        rows.swap(0, end);
        sift(rows, 0, end, check)?;
    }
    duplicates(rows, node, check)
}

fn sift(rows: &mut [BufferedRow], mut root: usize, end: usize, check: Check<'_>) -> Result<()> {
    while root < end / 2 {
        check()?;
        let child = root * 2 + 1;
        let greater = greater_child(rows, child, end);
        if rows[root].identity >= rows[greater].identity {
            return Ok(());
        }
        rows.swap(root, greater);
        root = greater;
    }
    Ok(())
}

fn greater_child(rows: &[BufferedRow], child: usize, end: usize) -> usize {
    let right = child + 1;
    if right < end && rows[right].identity > rows[child].identity {
        right
    } else {
        child
    }
}

fn duplicates(rows: &[BufferedRow], node: &str, check: Check<'_>) -> Result<()> {
    for pair in rows.windows(2) {
        check()?;
        if pair[0].identity == pair[1].identity {
            return Err(operator_error(
                node,
                &format!(
                    "duplicate row identity at event_time_micros={}",
                    pair[0].identity.event_time,
                ),
            ));
        }
    }
    check()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CalcFlowError, operator::rolling::KeyValue};

    fn row(time: i64) -> BufferedRow {
        BufferedRow::new(
            vec![Some(KeyValue::Signed(time % 3))],
            vec![KeyValue::Signed(time)],
            time,
            Vec::new(),
        )
    }

    #[test]
    fn test_cooperative_order_matches_the_existing_canonical_total_order() {
        for times in [
            vec![],
            vec![1],
            vec![4, 1, 3, 0, 2],
            (0..513).rev().collect(),
        ] {
            let mut rows: Vec<_> = times.into_iter().map(row).collect();
            let mut expected: Vec<_> = rows.iter().map(|row| row.identity.clone()).collect();
            expected.sort();
            let observed = allocation_counter::measure(|| {
                sort(&mut rows, "features", &mut || Ok(())).unwrap();
            });
            assert_eq!(
                rows.into_iter().map(|row| row.identity).collect::<Vec<_>>(),
                expected
            );
            assert_eq!(observed.bytes_total, 0);
        }
    }

    #[test]
    fn test_cooperative_order_rejects_duplicates_with_the_actual_node() {
        let error = sort(
            &mut [row(3), row(1), row(3)],
            "physical-features",
            &mut || Ok(()),
        )
        .unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Operator { node_id, message } if node_id == "physical-features" && message.contains("event_time_micros=3"))
        );
    }

    #[test]
    fn test_cooperative_order_preserves_cancellation_identity_during_sort() {
        let mut rows: Vec<_> = (0..513).rev().map(row).collect();
        let mut checks = 0;
        let result = sort(&mut rows, "features", &mut || {
            checks += 1;
            if checks == 16 {
                return Err(CalcFlowError::Cancelled {
                    run_id: "finite-sort".into(),
                });
            }
            Ok(())
        });
        assert_eq!(checks, 16);
        assert!(
            matches!(result, Err(CalcFlowError::Cancelled { run_id }) if run_id == "finite-sort")
        );
    }
}
