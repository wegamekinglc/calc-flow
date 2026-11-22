use std::collections::BTreeSet;

pub(super) fn projection<E>(
    items: impl Iterator<Item = Option<usize>>,
    check: &mut dyn FnMut() -> Result<(), E>,
) -> Result<BTreeSet<usize>, E> {
    let mut projection = BTreeSet::new();
    for item in items {
        check()?;
        if let Some(index) = item {
            projection.insert(index);
        }
    }
    Ok(projection)
}
#[cfg(test)]
#[path = "control_tests.rs"]
mod tests;
