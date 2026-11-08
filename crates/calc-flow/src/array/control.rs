use crate::Result;

pub(super) fn checkpoint(index: usize, check: &mut dyn FnMut() -> Result<()>) -> Result<()> {
    if index.is_multiple_of(4096) {
        check()
    } else {
        Ok(())
    }
}

pub(super) fn finish<T>(result: Result<T>, check: &mut dyn FnMut() -> Result<()>) -> Result<T> {
    let value = result?;
    check()?;
    Ok(value)
}
