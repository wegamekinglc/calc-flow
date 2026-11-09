use sha2::{Digest as _, Sha256};

use super::{ArrayStateOptions, AtrState, EwmaState, MacdState, RollingMeanState, RsiState};
use crate::{
    CalcFlowError, Result, RollingNumericalProfile,
    array::{
        ArrayDType, ArrayExecutionOptions, AtrOptions, EwmaOptions, MacdOptions,
        RollingMeanOptions, RsiOptions,
    },
};

mod ewma;
mod indicator;
mod macd;
mod mean;

const MAGIC: &[u8; 8] = b"CFASEQ\0\0";
const VERSION: u32 = 1;
const EWMA_BYTES: usize = 92;

#[derive(Clone, Copy, Eq, PartialEq)]
pub(in crate::array) enum KernelOptions {
    Mean(RollingMeanOptions),
    Ewma(EwmaOptions),
    Rsi(RsiOptions),
    Atr(AtrOptions),
    Macd(MacdOptions),
}

impl KernelOptions {
    fn document(self) -> serde_json::Value {
        match self {
            Self::Mean(options) => {
                serde_json::json!({"function":"rolling_mean", "window":options.window(),
                "min_periods":options.min_periods(), "numerical_profile":profile_name(options.numerical_profile())})
            }
            Self::Ewma(options) => {
                serde_json::json!({"function":"ewma", "span":options.span(), "min_periods":options.min_periods()})
            }
            Self::Rsi(options) => serde_json::json!({"function":"rsi", "period":options.period()}),
            Self::Atr(options) => serde_json::json!({"function":"atr", "period":options.period()}),
            Self::Macd(options) => macd::document(options),
        }
    }
}

fn profile_name(profile: RollingNumericalProfile) -> &'static str {
    match profile {
        RollingNumericalProfile::StableV1 => "stable_v1",
        RollingNumericalProfile::StableV2Preview => "stable_v2",
    }
}

pub(in crate::array) struct SequenceSpec {
    dtype: ArrayDType,
    kernel: KernelOptions,
    fingerprint: [u8; 32],
}

impl SequenceSpec {
    pub(in crate::array) fn new(dtype: ArrayDType, kernel: KernelOptions) -> Result<Self> {
        let dtype_name = match dtype {
            ArrayDType::Float32 => "float32",
            ArrayDType::Float64 => "float64",
            ArrayDType::Boolean => {
                return Err(super::super::invalid(
                    "options.input_dtype",
                    "requires float32 or float64",
                ));
            }
        };
        let mut document = serde_json::json!({
            "identity": "native:array_sequence@1", "input_dtype": dtype_name,
            "kernel": kernel.document(), "layout": 1, "numeric_semantics": 1,
        });
        if let Some(shape) = indicator::shape_rule(kernel) {
            document["layout"] = kernel_layout(kernel).into();
            document["input_shape_rule"] = shape.into();
        }
        let bytes = crate::json::canonical_json(&document)
            .map_err(|error| super::super::invalid("options.kernel", error.to_string()))?;
        Ok(Self {
            dtype,
            kernel,
            fingerprint: Sha256::digest(bytes.as_bytes()).into(),
        })
    }

    pub(in crate::array) const fn fingerprint(&self) -> &[u8; 32] {
        &self.fingerprint
    }

    pub(in crate::array) const fn layout(&self) -> u32 {
        kernel_layout(self.kernel)
    }

    pub(in crate::array) fn validate_input_shape(&self, shape: &[usize]) -> Result<()> {
        if matches!(self.kernel, KernelOptions::Atr(_)) {
            if !matches!(shape, [_, 3]) {
                return Err(super::super::invalid(
                    "input.shape",
                    "ATR sequence input requires [n, 3] in high, low, close order",
                ));
            }
        } else if shape.len() != 1 {
            return Err(super::super::invalid(
                "inputs.input.shape",
                format!(
                    "requires a rank-one sequence; received rank {}",
                    shape.len()
                ),
            ));
        }
        Ok(())
    }

    pub(in crate::array) fn indicator_tag(&self) -> Option<u8> {
        match self.kernel {
            KernelOptions::Rsi(_) => Some(3),
            KernelOptions::Atr(_) => Some(4),
            KernelOptions::Macd(_) => Some(5),
            _ => None,
        }
    }

    pub(in crate::array) const fn dtype(&self) -> ArrayDType {
        self.dtype
    }

    fn markers(&self) -> [u8; 4] {
        let dtype = match self.dtype {
            ArrayDType::Float32 => 1,
            _ => 2,
        };
        match self.kernel {
            KernelOptions::Mean(options) => [
                1,
                dtype,
                if options.numerical_profile() == RollingNumericalProfile::StableV1 {
                    1
                } else {
                    2
                },
                0,
            ],
            KernelOptions::Ewma(_) => [2, dtype, 0, 0],
            KernelOptions::Rsi(_) => [3, dtype, 0, 0],
            KernelOptions::Atr(_) => [4, dtype, 0, 0],
            KernelOptions::Macd(_) => [5, dtype, 0, 0],
        }
    }

    fn parameters(&self) -> (u64, u64) {
        match self.kernel {
            KernelOptions::Mean(options) => (options.window(), options.min_periods()),
            KernelOptions::Ewma(options) => (options.span(), options.min_periods()),
            KernelOptions::Rsi(options) => (options.period(), 0),
            KernelOptions::Atr(options) => (options.period(), 0),
            KernelOptions::Macd(_) => unreachable!("MACD uses its five-parameter header"),
        }
    }

    fn validate_state(&self, state: &SequenceState) -> Result<()> {
        if self.kernel != state.kernel_options() {
            return Err(mismatch("parameters", "state and configured kernel differ"));
        }
        Ok(())
    }
}

pub(in crate::array) enum SequenceState {
    Mean(RollingMeanState),
    Ewma(EwmaState),
    Rsi(RsiState),
    Atr(AtrState),
    Macd(MacdState),
}

impl SequenceState {
    pub(in crate::array) fn new(spec: &SequenceSpec, limits: ArrayStateOptions) -> Result<Self> {
        match spec.kernel {
            KernelOptions::Mean(options) => RollingMeanState::new(options, limits).map(Self::Mean),
            KernelOptions::Ewma(options) => EwmaState::new(options, limits).map(Self::Ewma),
            KernelOptions::Rsi(options) => RsiState::new(options, limits).map(Self::Rsi),
            KernelOptions::Atr(options) => AtrState::new(options, limits).map(Self::Atr),
            KernelOptions::Macd(options) => MacdState::new(options, limits).map(Self::Macd),
        }
    }

    pub(in crate::array) const fn state_bytes(&self) -> usize {
        match self {
            Self::Mean(state) => state.state_bytes(),
            Self::Ewma(state) => state.state_bytes(),
            Self::Rsi(state) => state.state_bytes(),
            Self::Atr(state) => state.state_bytes(),
            Self::Macd(state) => state.state_bytes(),
        }
    }

    pub(in crate::array) fn reset(&mut self) {
        match self {
            Self::Mean(state) => state.reset(),
            Self::Ewma(state) => state.reset(),
            Self::Rsi(state) => state.reset(),
            Self::Atr(state) => state.reset(),
            Self::Macd(state) => state.reset(),
        }
    }

    pub(in crate::array) fn prepare_checked(
        &self,
        input: crate::array::ArrayView<'_>,
        execution: &ArrayExecutionOptions,
        captures: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<super::Prepared<Self>> {
        match self {
            Self::Mean(state) => state
                .prepare_checked(input, execution, captures, check)
                .map(|prepared| wrap(prepared, Self::Mean)),
            Self::Ewma(state) => state
                .prepare_checked(input, execution, captures, check)
                .map(|prepared| wrap(prepared, Self::Ewma)),
            Self::Rsi(state) => state
                .prepare_checked(input, execution, captures, check)
                .map(|prepared| wrap(prepared, Self::Rsi)),
            Self::Atr(state) => state
                .prepare_packed_checked(input, execution, captures, check)
                .map(|prepared| wrap(prepared, Self::Atr)),
            Self::Macd(state) => state
                .prepare_packed_checked(input, execution, captures, check)
                .map(|prepared| wrap(prepared, Self::Macd)),
        }
    }

    fn kernel_options(&self) -> KernelOptions {
        match self {
            Self::Mean(state) => KernelOptions::Mean(state.options),
            Self::Ewma(state) => KernelOptions::Ewma(state.options),
            Self::Rsi(state) => KernelOptions::Rsi(state.options),
            Self::Atr(state) => KernelOptions::Atr(state.options),
            Self::Macd(state) => KernelOptions::Macd(state.options),
        }
    }

    fn positions(&self) -> u64 {
        match self {
            Self::Mean(state) => state.kernel.positions(),
            Self::Ewma(state) => state.positions,
            Self::Rsi(state) => state.kernel.positions,
            Self::Atr(state) => state.kernel.positions,
            Self::Macd(state) => state.kernel.positions,
        }
    }

    pub(super) fn encoded_bytes(&self) -> Result<usize> {
        match self {
            Self::Mean(state) => mean::encoded_bytes(state.kernel.history_len()),
            Self::Ewma(_) => Ok(EWMA_BYTES),
            Self::Rsi(_) => Ok(116),
            Self::Atr(_) => Ok(108),
            Self::Macd(_) => Ok(156),
        }
    }
}

pub(in crate::array) fn mismatch(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: format!("native:array_sequence@1 state.{field}: {message}"),
    }
}

pub(in crate::array) fn admit(execution: &ArrayExecutionOptions, charges: &[usize]) -> Result<()> {
    let peak = charges
        .iter()
        .try_fold(0_usize, |total, bytes| total.checked_add(*bytes))
        .ok_or_else(|| mismatch("peak_bytes", "live byte count overflowed usize"))?;
    if peak > execution.max_allocation_bytes() {
        return Err(mismatch(
            "peak_bytes",
            "live state/encoded bytes exceed the allocation limit",
        ));
    }
    Ok(())
}

pub(in crate::array) fn encode(
    spec: &SequenceSpec,
    state: &SequenceState,
    execution: &ArrayExecutionOptions,
    captures: usize,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<u8>> {
    check()?;
    spec.validate_state(state)?;
    let bytes = allocate(state, execution, captures)?;
    if spec.layout() == 2 {
        return indicator::encode(spec, state, bytes, check);
    }
    if spec.layout() == 3 {
        return macd::encode(spec, state, bytes, check);
    }
    write_legacy(spec, state, bytes, check)
}

fn write_legacy(
    spec: &SequenceSpec,
    state: &SequenceState,
    mut bytes: Vec<u8>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<u8>> {
    write_header(spec, state.positions(), &mut bytes);
    match state {
        SequenceState::Mean(state) => mean::write(state, &mut bytes, check)?,
        SequenceState::Ewma(state) => ewma::write(state, &mut bytes),
        _ => unreachable!("indicator encodings use their versioned writer"),
    }
    check()?;
    Ok(bytes)
}

fn allocate(
    state: &SequenceState,
    execution: &ArrayExecutionOptions,
    captures: usize,
) -> Result<Vec<u8>> {
    let len = state.encoded_bytes()?;
    admit(
        execution,
        &[state.state_bytes(), len, captures, size_of::<u64>()],
    )?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|error| mismatch("bytes", &error.to_string()))?;
    admit(
        execution,
        &[
            state.state_bytes(),
            bytes.capacity(),
            captures,
            size_of::<u64>(),
        ],
    )?;
    Ok(bytes)
}

fn write_header(spec: &SequenceSpec, positions: u64, bytes: &mut Vec<u8>) {
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&VERSION.to_le_bytes());
    bytes.extend_from_slice(&spec.markers());
    bytes.extend_from_slice(spec.fingerprint());
    let (window, minimum) = spec.parameters();
    word(bytes, window);
    word(bytes, minimum);
    word(bytes, positions);
}

fn word(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

#[derive(Clone, Copy)]
struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Result<[u8; N]> {
        let end = self
            .position
            .checked_add(N)
            .ok_or_else(|| mismatch("bytes", "byte offset overflowed"))?;
        let bytes = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| mismatch("bytes", "truncated segment"))?;
        self.position = end;
        bytes
            .try_into()
            .map_err(|_| mismatch("bytes", "invalid field length"))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take()?))
    }

    fn require<const N: usize>(&mut self, expected: &[u8; N], field: &str) -> Result<()> {
        if &self.take::<N>()? != expected {
            return Err(mismatch(field, "unexpected value"));
        }
        Ok(())
    }
}

fn read_header(spec: &SequenceSpec, reader: &mut Reader<'_>) -> Result<u64> {
    if spec.layout() == 2 {
        return indicator::read_header(spec, reader);
    }
    if spec.layout() == 3 {
        return macd::read_header(spec, reader);
    }
    reader.require(MAGIC, "magic")?;
    reader.require(&VERSION.to_le_bytes(), "layout")?;
    reader.require(&VERSION.to_le_bytes(), "numeric_semantics")?;
    reader.require(&spec.markers(), "kernel/dtype/profile")?;
    read_identity(spec, reader)
}

fn read_identity(spec: &SequenceSpec, reader: &mut Reader<'_>) -> Result<u64> {
    reader.require(spec.fingerprint(), "fingerprint")?;
    let (window, minimum) = spec.parameters();
    reader.require(&window.to_le_bytes(), "window_or_span")?;
    reader.require(&minimum.to_le_bytes(), "min_periods")?;
    reader.u64()
}

pub(in crate::array) struct RestoreLimits<'a> {
    pub(in crate::array) state: ArrayStateOptions,
    pub(in crate::array) execution: &'a ArrayExecutionOptions,
    pub(in crate::array) old: usize,
    pub(in crate::array) captures: usize,
    pub(in crate::array) encoded: usize,
}

impl RestoreLimits<'_> {
    fn input(&self, length: usize) -> Result<()> {
        if self.encoded < length {
            return Err(mismatch(
                "encoded_bytes",
                "encoded allocation charge is below the segment length",
            ));
        }
        admit(self.execution, &[self.old, self.encoded, self.captures, 32])
    }

    fn candidate(&self, bytes: usize, scratch: usize) -> Result<()> {
        self.state
            .validate(bytes)
            .map_err(|_| mismatch("max_state_bytes", "decoded state exceeds retained limit"))?;
        admit(
            self.execution,
            &[self.old, bytes, self.encoded, self.captures, scratch],
        )
    }
}

pub(in crate::array) fn preflight_encoded(
    spec: &SequenceSpec,
    length: usize,
    limits: &RestoreLimits<'_>,
) -> Result<()> {
    limits.input(length)?;
    let (candidate, scratch) = encoded_requirements(spec, length)?;
    limits.candidate(candidate, scratch)
}
pub(in crate::array) fn validate_encoded_storage(
    spec: &SequenceSpec,
    length: usize,
    state: ArrayStateOptions,
) -> Result<()> {
    let (candidate, _) = encoded_requirements(spec, length)?;
    state
        .validate(candidate)
        .map_err(|_| mismatch("max_state_bytes", "encoded state exceeds retained limit"))
}
fn encoded_requirements(spec: &SequenceSpec, length: usize) -> Result<(usize, usize)> {
    match spec.kernel {
        KernelOptions::Mean(options) => mean::encoded_requirements(options, length),
        KernelOptions::Ewma(_) => ewma_encoded_requirements(length),
        KernelOptions::Rsi(_) => indicator::requirements(length, 116, 48),
        KernelOptions::Atr(_) => indicator::requirements(length, 108, 40),
        KernelOptions::Macd(_) => macd::requirements(length),
    }
}
fn ewma_encoded_requirements(length: usize) -> Result<(usize, usize)> {
    if length != EWMA_BYTES {
        return Err(mismatch(
            "encoded_bytes",
            "EWMA requires exactly 92 encoded bytes",
        ));
    }
    Ok((24, 32))
}

pub(in crate::array) fn decode(
    spec: &SequenceSpec,
    bytes: &[u8],
    limits: &RestoreLimits<'_>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<SequenceState> {
    check()?;
    limits.input(bytes.len())?;
    let mut reader = Reader { bytes, position: 0 };
    let positions = read_header(spec, &mut reader)?;
    let state = read_state(spec, positions, reader, limits, check)?;
    check()?;
    Ok(state)
}

fn read_state(
    spec: &SequenceSpec,
    positions: u64,
    mut reader: Reader<'_>,
    limits: &RestoreLimits<'_>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<SequenceState> {
    match spec.kernel {
        KernelOptions::Mean(options) => mean::read(options, positions, reader, limits, check),
        KernelOptions::Ewma(options) => ewma::read(options, positions, &mut reader, limits),
        KernelOptions::Rsi(_) | KernelOptions::Atr(_) => {
            indicator::read_state(spec, positions, reader, limits)
        }
        KernelOptions::Macd(options) => macd::read_state(options, positions, reader, limits),
    }
}

fn wrap<T>(
    prepared: super::Prepared<T>,
    constructor: impl FnOnce(T) -> SequenceState,
) -> super::Prepared<SequenceState> {
    super::Prepared {
        candidate: prepared.candidate.map(constructor),
        output: prepared.output,
    }
}

const fn kernel_layout(kernel: KernelOptions) -> u32 {
    match kernel {
        KernelOptions::Macd(_) => 3,
        KernelOptions::Rsi(_) | KernelOptions::Atr(_) => 2,
        _ => 1,
    }
}
