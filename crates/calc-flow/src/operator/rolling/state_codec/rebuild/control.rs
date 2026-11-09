use crate::Result;
use std::collections::BTreeSet;

pub(super) fn fold<T, U>(
    items: impl Iterator<Item = T>,
    check: &mut dyn FnMut() -> Result<()>,
    body: impl FnOnce(&mut dyn Iterator<Item = T>) -> Result<U>,
) -> Result<U> {
    let mut failure = None;
    let result = {
        let mut rows = items.map_while(|item| match check() {
            Ok(()) => Some(item),
            Err(error) => {
                failure = Some(error);
                None
            }
        });
        body(&mut rows)
    };
    match failure {
        Some(error) => Err(error),
        None => result,
    }
}
pub(super) fn projection<E>(
    items: impl Iterator<Item = Option<usize>>,
    check: &mut dyn FnMut() -> std::result::Result<(), E>,
) -> std::result::Result<BTreeSet<usize>, E> {
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
