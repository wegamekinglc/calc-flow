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

// Keep the checkpoint cadence across short lanes as well as long inner loops.
pub(super) fn step(index: &mut usize, check: &mut dyn FnMut() -> Result<()>) -> Result<()> {
    checkpoint(*index, check)?;
    *index = index.wrapping_add(1);
    Ok(())
}
