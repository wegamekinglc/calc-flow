use super::{
    EWMA_BYTES, EwmaOptions, EwmaState, Reader, RestoreLimits, SequenceState, mismatch, word,
};
use crate::{Result, array::rolling::EwmaKernel};

pub(super) fn write(state: &EwmaState, bytes: &mut Vec<u8>) {
    let (count, previous) = state.kernel.parts();
    word(bytes, count);
    word(bytes, previous.to_bits());
}

pub(super) fn read(
    options: EwmaOptions,
    positions: u64,
    reader: &mut Reader<'_>,
    limits: &RestoreLimits<'_>,
) -> Result<SequenceState> {
    if reader.bytes.len() != EWMA_BYTES {
        return Err(mismatch("bytes", "requires exactly 92 EWMA bytes"));
    }
    limits.candidate(24, 32)?;
    let count = reader.u64()?;
    let previous = reader.u64()?;
    validate(positions, count, previous)?;
    Ok(SequenceState::Ewma(EwmaState {
        options,
        limits: limits.state,
        positions,
        kernel: EwmaKernel::from_parts(count, f64::from_bits(previous)),
    }))
}

fn validate(positions: u64, count: u64, previous: u64) -> Result<()> {
    if count > positions {
        return Err(mismatch("valid_count", "exceeds processed positions"));
    }
    if count == 0 && previous != 0 {
        return Err(mismatch(
            "previous",
            "unseeded state requires positive zero",
        ));
    }
    Ok(())
}
