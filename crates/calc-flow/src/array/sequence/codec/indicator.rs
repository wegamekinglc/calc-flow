use super::{KernelOptions, Reader, RestoreLimits, SequenceSpec, SequenceState, mismatch, word};
use crate::{
    CalcFlowError, Result,
    array::{
        AtrOptions, RsiOptions,
        indicators::{AtrKernel, RsiKernel},
        sequence::{AtrState, RsiState},
    },
};

const MAGIC: &[u8; 8] = b"CFAIND\0\0";

pub(super) fn shape_rule(kernel: KernelOptions) -> Option<&'static str> {
    match kernel {
        KernelOptions::Rsi(_) => Some("rank_one"),
        KernelOptions::Atr(_) => Some("packed_high_low_close"),
        _ => None,
    }
}

pub(super) fn requirements(
    length: usize,
    encoded: usize,
    retained: usize,
) -> Result<(usize, usize)> {
    if length != encoded {
        return Err(mismatch(
            "encoded_bytes",
            "indicator layout 2 requires its exact fixed segment length",
        ));
    }
    Ok((retained, 32))
}

pub(super) fn encode(
    spec: &SequenceSpec,
    state: &SequenceState,
    mut bytes: Vec<u8>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<u8>> {
    write_header(spec, state.positions(), &mut bytes);
    match state {
        SequenceState::Rsi(state) => write_rsi(state, &mut bytes)?,
        SequenceState::Atr(state) => write_atr(state, &mut bytes)?,
        _ => return Err(mismatch("kernel", "layout 2 requires RSI or ATR")),
    }
    check()?;
    Ok(bytes)
}

fn write_header(spec: &SequenceSpec, positions: u64, bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&spec.markers());
    bytes.extend_from_slice(spec.fingerprint());
    word(bytes, positions);
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    bytes.extend_from_slice(&payload_bytes(spec).to_le_bytes());
    word(bytes, spec.parameters().0);
}

fn prefix(bytes: &mut Vec<u8>, has_previous: bool, count: u64, previous: f64) {
    bytes.extend_from_slice(&[u8::from(has_previous), 0, 0, 0, 0, 0, 0, 0]);
    word(bytes, count);
    word(bytes, previous.to_bits());
}

fn write_rsi(state: &RsiState, bytes: &mut Vec<u8>) -> Result<()> {
    invariants(state.kernel.validate(state.options.period()))?;
    let kernel = &state.kernel;
    prefix(
        bytes,
        kernel.has_previous,
        kernel.seed_count,
        kernel.previous_price,
    );
    word(bytes, kernel.mean_gain.to_bits());
    word(bytes, kernel.mean_loss.to_bits());
    Ok(())
}
fn write_atr(state: &AtrState, bytes: &mut Vec<u8>) -> Result<()> {
    invariants(state.kernel.validate(state.options.period()))?;
    let kernel = &state.kernel;
    prefix(
        bytes,
        kernel.has_previous,
        kernel.seed_count,
        kernel.previous_close,
    );
    word(bytes, kernel.mean_tr.to_bits());
    Ok(())
}

pub(super) fn read_header(spec: &SequenceSpec, reader: &mut Reader<'_>) -> Result<u64> {
    reader.require(MAGIC, "magic")?;
    reader.require(&2_u32.to_le_bytes(), "layout")?;
    reader.require(&1_u32.to_le_bytes(), "numeric_semantics")?;
    reader.require(&spec.markers(), "kernel/dtype/reserved")?;
    read_identity(spec, reader)
}
fn read_identity(spec: &SequenceSpec, reader: &mut Reader<'_>) -> Result<u64> {
    reader.require(spec.fingerprint(), "fingerprint")?;
    let positions = reader.u64()?;
    reader.require(&8_u32.to_le_bytes(), "parameter_bytes")?;
    reader.require(&payload_bytes(spec).to_le_bytes(), "payload_bytes")?;
    reader.require(&spec.parameters().0.to_le_bytes(), "period")?;
    Ok(positions)
}
fn payload_bytes(spec: &SequenceSpec) -> u32 {
    match spec.kernel {
        KernelOptions::Rsi(_) => 40,
        _ => 32,
    }
}

pub(super) fn read_state(
    spec: &SequenceSpec,
    positions: u64,
    reader: Reader<'_>,
    limits: &RestoreLimits<'_>,
) -> Result<SequenceState> {
    let (retained, scratch) = super::encoded_requirements(spec, reader.bytes.len())?;
    limits.candidate(retained, scratch)?;
    match spec.kernel {
        KernelOptions::Rsi(options) => read_rsi(options, positions, reader, limits),
        KernelOptions::Atr(options) => read_atr(options, positions, reader, limits),
        _ => Err(mismatch("kernel", "layout 2 requires RSI or ATR")),
    }
}
fn read_rsi(
    options: RsiOptions,
    positions: u64,
    mut reader: Reader<'_>,
    limits: &RestoreLimits<'_>,
) -> Result<SequenceState> {
    let kernel = RsiKernel {
        positions,
        has_previous: flag(&mut reader)?,
        seed_count: reader.u64()?,
        previous_price: f64::from_bits(reader.u64()?),
        mean_gain: f64::from_bits(reader.u64()?),
        mean_loss: f64::from_bits(reader.u64()?),
    };
    invariants(kernel.validate(options.period()))?;
    Ok(SequenceState::Rsi(RsiState {
        options,
        limits: limits.state,
        kernel,
    }))
}
fn read_atr(
    options: AtrOptions,
    positions: u64,
    mut reader: Reader<'_>,
    limits: &RestoreLimits<'_>,
) -> Result<SequenceState> {
    let kernel = AtrKernel {
        positions,
        has_previous: flag(&mut reader)?,
        seed_count: reader.u64()?,
        previous_close: f64::from_bits(reader.u64()?),
        mean_tr: f64::from_bits(reader.u64()?),
    };
    invariants(kernel.validate(options.period()))?;
    Ok(SequenceState::Atr(AtrState {
        options,
        limits: limits.state,
        kernel,
    }))
}
fn flag(reader: &mut Reader<'_>) -> Result<bool> {
    let flag = reader.take::<8>()?;
    if flag[0] > 1 || flag[1..].iter().any(|&value| value != 0) {
        return Err(mismatch(
            "has_previous/reserved",
            "requires a binary flag and zero reserved bytes",
        ));
    }
    Ok(flag[0] == 1)
}
fn invariants(result: Result<()>) -> Result<()> {
    result.map_err(|error| match error {
        CalcFlowError::InvalidArgument { field, message } => {
            mismatch(field.strip_prefix("state.").unwrap_or(&field), &message)
        }
        error => error,
    })
}
