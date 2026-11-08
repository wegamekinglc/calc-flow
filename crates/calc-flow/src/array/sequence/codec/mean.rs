use super::{
    Reader, RestoreLimits, RollingMeanOptions, RollingMeanState, SequenceState, mismatch, word,
};
use crate::{
    Result,
    array::{control, rolling_mean::MeanKernel, rolling_mean_math::FloatSummary},
};

const HEADER_BYTES: usize = 140;
const SLOT_BYTES: usize = 9;

pub(super) fn encoded_bytes(len: usize) -> Result<usize> {
    len.checked_mul(SLOT_BYTES)
        .and_then(|bytes| bytes.checked_add(HEADER_BYTES))
        .ok_or_else(|| mismatch("bytes", "encoded history length overflowed usize"))
}

pub(super) fn encoded_requirements(
    options: RollingMeanOptions,
    length: usize,
) -> Result<(usize, usize)> {
    let len = encoded_history_len(length)?;
    validate_window_len(len, options.window())?;
    let charge = super::super::history_charge(len)
        .map_err(|_| mismatch("history_length", "state charge overflowed usize"))?;
    Ok((charge, scratch_bytes()))
}

fn encoded_history_len(length: usize) -> Result<usize> {
    let bytes = length
        .checked_sub(HEADER_BYTES)
        .ok_or_else(|| mismatch("encoded_bytes", "mean state is shorter than its header"))?;
    if !bytes.is_multiple_of(SLOT_BYTES) {
        return Err(mismatch(
            "encoded_bytes",
            "mean history has a partial encoded slot",
        ));
    }
    Ok(bytes / SLOT_BYTES)
}
fn validate_window_len(len: usize, window: u64) -> Result<()> {
    let len = u64::try_from(len).map_err(|_| mismatch("history_length", "exceeds u64"))?;
    if len > window {
        return Err(mismatch(
            "history_length",
            "encoded history exceeds configured window",
        ));
    }
    Ok(())
}

pub(super) fn write(
    state: &RollingMeanState,
    bytes: &mut Vec<u8>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    write_summary(state.kernel.summary(), bytes);
    let len = u64::try_from(state.kernel.history_len())
        .map_err(|_| mismatch("history_length", "exceeds u64"))?;
    word(bytes, len);
    for (index, value) in state.kernel.history().enumerate() {
        control::checkpoint(index, check)?;
        bytes.push(u8::from(value.is_some()));
        word(bytes, value.map_or(0, f64::to_bits));
    }
    Ok(())
}

fn write_summary(summary: &FloatSummary, bytes: &mut Vec<u8>) {
    word(bytes, summary.valid_count);
    word(bytes, summary.sum.to_bits());
    word(bytes, summary.mean.to_bits());
    word(bytes, summary.m2.to_bits());
    word(bytes, summary.nan_count);
    word(bytes, summary.pos_inf);
    word(bytes, summary.neg_inf);
}

fn read_summary(reader: &mut Reader<'_>) -> Result<FloatSummary> {
    let valid_count = reader.u64()?;
    let sum = f64::from_bits(reader.u64()?);
    let mean = f64::from_bits(reader.u64()?);
    let m2 = f64::from_bits(reader.u64()?);
    let (nan_count, pos_inf, neg_inf) = read_counts(reader)?;
    Ok(FloatSummary {
        valid_count,
        sum,
        mean,
        m2,
        nan_count,
        pos_inf,
        neg_inf,
    })
}

fn read_counts(reader: &mut Reader<'_>) -> Result<(u64, u64, u64)> {
    Ok((reader.u64()?, reader.u64()?, reader.u64()?))
}

fn history_len(reader: &mut Reader<'_>, positions: u64, window: u64) -> Result<usize> {
    let len = reader.u64()?;
    if len != positions.min(window) {
        return Err(mismatch(
            "history_length",
            "does not match processed positions and window",
        ));
    }
    let len =
        usize::try_from(len).map_err(|_| mismatch("history_length", "exceeds local usize"))?;
    if encoded_bytes(len)? != reader.bytes.len() {
        return Err(mismatch(
            "bytes",
            "history length does not match exact segment size",
        ));
    }
    Ok(len)
}

#[derive(Default)]
struct Counts {
    present: u64,
    positive: u64,
    negative: u64,
}

impl Counts {
    fn observe(&mut self, value: Option<f64>) -> Result<()> {
        let Some(value) = value else { return Ok(()) };
        increment(&mut self.present)?;
        if value.is_infinite() {
            let count = if value.is_sign_positive() {
                &mut self.positive
            } else {
                &mut self.negative
            };
            increment(count)?;
        }
        Ok(())
    }

    fn validate(&self, summary: &FloatSummary) -> Result<()> {
        if (self.present, self.positive, self.negative)
            != (summary.valid_count, summary.pos_inf, summary.neg_inf)
            || summary.nan_count != 0
        {
            return Err(mismatch(
                "counts",
                "summary counts do not match normalized history",
            ));
        }
        Ok(())
    }
}

fn increment(count: &mut u64) -> Result<()> {
    *count = count
        .checked_add(1)
        .ok_or_else(|| mismatch("counts", "history count overflowed u64"))?;
    Ok(())
}

fn sample(reader: &mut Reader<'_>) -> Result<Option<f64>> {
    let tag = reader.take::<1>()?[0];
    let bits = reader.u64()?;
    match tag {
        0 => missing(bits),
        1 => present(bits),
        _ => Err(mismatch("history.tag", "requires missing or present")),
    }
}

fn missing(bits: u64) -> Result<Option<f64>> {
    if bits != 0 {
        return Err(mismatch(
            "history.value",
            "missing position requires zero bits",
        ));
    }
    Ok(None)
}

fn present(bits: u64) -> Result<Option<f64>> {
    let value = f64::from_bits(bits);
    if value.is_nan() {
        return Err(mismatch(
            "history.value",
            "present history cannot contain NaN",
        ));
    }
    Ok(Some(value))
}

fn inspect(
    mut reader: Reader<'_>,
    len: usize,
    summary: &FloatSummary,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    let mut counts = Counts::default();
    for index in 0..len {
        control::checkpoint(index, check)?;
        counts.observe(sample(&mut reader)?)?;
    }
    counts.validate(summary)
}

const fn scratch_bytes() -> usize {
    size_of::<FloatSummary>() + size_of::<Counts>() + size_of::<u64>()
}

pub(super) fn read(
    options: RollingMeanOptions,
    positions: u64,
    mut reader: Reader<'_>,
    limits: &RestoreLimits<'_>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<SequenceState> {
    limits.candidate(64, scratch_bytes())?;
    let summary = read_summary(&mut reader)?;
    let len = history_len(&mut reader, positions, options.window())?;
    let charge = super::super::history_charge(len)
        .map_err(|_| mismatch("history_length", "state charge overflowed usize"))?;
    limits.candidate(charge, scratch_bytes())?;
    inspect(reader, len, &summary, check)?;
    restored(options, positions, reader, summary, limits, check)
}

fn restored(
    options: RollingMeanOptions,
    positions: u64,
    mut reader: Reader<'_>,
    summary: FloatSummary,
    limits: &RestoreLimits<'_>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<SequenceState> {
    let len = (reader.bytes.len() - reader.position) / 9;
    let values = history(&mut reader, len, limits, check)?;
    let bytes = super::super::history_charge(values.capacity())
        .map_err(|_| mismatch("history_length", "state charge overflowed usize"))?;
    Ok(SequenceState::Mean(RollingMeanState {
        options,
        limits: limits.state,
        bytes,
        kernel: MeanKernel::from_parts(values, summary, positions),
    }))
}

fn history(
    reader: &mut Reader<'_>,
    len: usize,
    limits: &RestoreLimits<'_>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<Option<f64>>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|error| mismatch("history", &error.to_string()))?;
    let actual = super::super::history_charge(values.capacity())
        .map_err(|_| mismatch("history_length", "state charge overflowed usize"))?;
    limits.candidate(actual, scratch_bytes())?;
    for index in 0..len {
        control::checkpoint(index, check)?;
        values.push(sample(reader)?);
    }
    Ok(values)
}
