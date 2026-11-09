use crate::Result;

pub(super) fn contains(
    columns: impl Iterator<Item = usize>,
    index: usize,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<bool> {
    for column in columns {
        check()?;
        if column == index {
            return Ok(true);
        }
    }
    Ok(false)
}
#[cfg(test)]
#[path = "columns_tests.rs"]
mod tests;
