use crate::{CalcFlowError, Result};
use std::cmp::Ordering;

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

pub(super) fn sort_unique<'a, T>(
    values: &mut Vec<&'a T>,
    spare: &mut Vec<&'a T>,
    compare: fn(&T, &T) -> Ordering,
    check: Check<'_>,
) -> Result<usize> {
    check()?;
    if spare.capacity() < values.len() {
        return Err(invalid());
    }
    let mut width = 1;
    while width < values.len() {
        merge_level(values, spare, width, compare, check)?;
        std::mem::swap(values, spare);
        width = width.checked_mul(2).ok_or_else(invalid)?;
    }
    deduplicate(values, compare, check)?;
    spare.clear();
    Ok(values.len())
}
fn merge_level<'a, T>(
    values: &[&'a T],
    output: &mut Vec<&'a T>,
    width: usize,
    compare: fn(&T, &T) -> Ordering,
    check: Check<'_>,
) -> Result<()> {
    check()?;
    output.clear();
    let span = width.checked_mul(2).ok_or_else(invalid)?;
    for start in (0..values.len()).step_by(span) {
        let middle = start.saturating_add(width).min(values.len());
        let end = start.saturating_add(span).min(values.len());
        merge(values, output, start..middle, middle..end, compare, check)?;
    }
    Ok(())
}
fn merge<'a, T>(
    values: &[&'a T],
    output: &mut Vec<&'a T>,
    mut left: std::ops::Range<usize>,
    mut right: std::ops::Range<usize>,
    compare: fn(&T, &T) -> Ordering,
    check: Check<'_>,
) -> Result<()> {
    while !left.is_empty() || !right.is_empty() {
        checkpoint(output.len(), check)?;
        let index = if take_left(values, &left, &right, compare) {
            let index = left.start;
            left.start += 1;
            index
        } else {
            let index = right.start;
            right.start += 1;
            index
        };
        output.push(values[index]);
    }
    Ok(())
}
fn take_left<T>(
    values: &[&T],
    left: &std::ops::Range<usize>,
    right: &std::ops::Range<usize>,
    compare: fn(&T, &T) -> Ordering,
) -> bool {
    if right.is_empty() {
        return true;
    }
    !left.is_empty() && compare(values[left.start], values[right.start]) != Ordering::Greater
}
fn deduplicate<T>(
    values: &mut Vec<&T>,
    compare: fn(&T, &T) -> Ordering,
    check: Check<'_>,
) -> Result<()> {
    let mut retained = usize::from(!values.is_empty());
    for index in 1..values.len() {
        checkpoint(index, check)?;
        if compare(values[retained - 1], values[index]) != Ordering::Equal {
            values[retained] = values[index];
            retained += 1;
        }
    }
    values.truncate(retained);
    check()
}
fn checkpoint(position: usize, check: Check<'_>) -> Result<()> {
    if position.is_multiple_of(4_096) {
        check()?;
    }
    Ok(())
}
fn invalid() -> CalcFlowError {
    CalcFlowError::Format {
        message: "rolling checkpoint entity ordering capacity is invalid".into(),
    }
}

#[cfg(test)]
#[path = "order_tests.rs"]
mod tests;
