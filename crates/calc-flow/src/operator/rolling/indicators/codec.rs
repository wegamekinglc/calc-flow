use super::{AtrKernel, Group, MacdKernel, RsiKernel, State};
use crate::{Result, operator::checkpoint::state_format};

impl Group {
    pub(in super::super) fn encode(&self, state: &State) -> Result<Vec<u8>> {
        self.validate(state)?;
        let mut bytes = Vec::with_capacity(self.payload_len());
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&[self.tag(), 0, 0, 0]);
        bytes.extend_from_slice(&state.positions().to_le_bytes());
        let (parameters, count) = self.parameters();
        for value in parameters.into_iter().take(count) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        state.append_payload(&mut bytes);
        Ok(bytes)
    }
    pub(in super::super) fn decode(&self, bytes: &[u8], positions: u64) -> Result<State> {
        self.validate_header(bytes, positions)?;
        let (_, count) = self.parameters();
        let state = self.decode_payload(&bytes[16 + count * 8..], positions)?;
        self.validate(&state)?;
        Ok(state)
    }
    fn decode_payload(&self, payload: &[u8], positions: u64) -> Result<State> {
        if matches!(self, Self::Macd(_)) {
            return decode_macd(payload, positions).map(State::Macd);
        }
        let flag = word(payload, 0)?;
        if flag > 1 {
            return Err(mismatch("has_previous must be zero or one"));
        }
        match self {
            Self::Rsi { .. } => decode_rsi(payload, positions, flag == 1).map(State::Rsi),
            Self::Atr { .. } => decode_atr(payload, positions, flag == 1).map(State::Atr),
            Self::Macd(_) => unreachable!("MACD handled before resettable indicator payloads"),
        }
    }
    fn validate_header(&self, bytes: &[u8], positions: u64) -> Result<()> {
        if bytes.len() != self.payload_len() {
            return Err(mismatch("payload length does not match the compiled group"));
        }
        if bytes[..8] != [1, 0, 0, 0, self.tag(), 0, 0, 0] {
            return Err(mismatch(
                "numeric semantics, kernel or reserved bytes differ",
            ));
        }
        if word(bytes, 8)? != positions {
            return Err(mismatch("positions differ from the entity"));
        }
        self.validate_parameters(bytes)
    }
    fn validate_parameters(&self, bytes: &[u8]) -> Result<()> {
        let (parameters, count) = self.parameters();
        for (index, expected) in parameters.into_iter().take(count).enumerate() {
            if word(bytes, 16 + index * 8)? != expected {
                return Err(mismatch("parameters differ from the compiled group"));
            }
        }
        Ok(())
    }
    fn validate(&self, state: &State) -> Result<()> {
        match (self, state) {
            (Self::Rsi { period, .. }, State::Rsi(state)) => state.validate(*period),
            (Self::Atr { period, .. }, State::Atr(state)) => state.validate(*period),
            (Self::Macd(group), State::Macd(state)) => state.validate(group.options),
            _ => return Err(mismatch("kernel does not match the compiled group")),
        }
        .map_err(|error| mismatch(&error.to_string()))
    }
    const fn tag(&self) -> u8 {
        match self {
            Self::Rsi { .. } => 3,
            Self::Atr { .. } => 4,
            Self::Macd(_) => 5,
        }
    }
    fn parameters(&self) -> ([u64; 5], usize) {
        match self {
            Self::Rsi { period, .. } | Self::Atr { period, .. } => ([*period, 0, 0, 0, 0], 1),
            Self::Macd(group) => {
                let options = group.options;
                (
                    [
                        options.fast_span(),
                        options.slow_span(),
                        options.signal_span(),
                        options.min_periods(),
                        options.signal_min_periods(),
                    ],
                    5,
                )
            }
        }
    }
    pub(super) const fn payload_len(&self) -> usize {
        match self {
            Self::Rsi { .. } => 64,
            Self::Atr { .. } => 56,
            Self::Macd(_) => 104,
        }
    }
}
impl State {
    pub(in super::super) const fn positions(&self) -> u64 {
        match self {
            Self::Rsi(state) => state.positions,
            Self::Atr(state) => state.positions,
            Self::Macd(state) => state.positions,
        }
    }
    fn append_payload(&self, bytes: &mut Vec<u8>) {
        match self {
            Self::Rsi(state) => append_words(
                bytes,
                [
                    u64::from(state.has_previous),
                    state.seed_count,
                    state.previous_price.to_bits(),
                    state.mean_gain.to_bits(),
                    state.mean_loss.to_bits(),
                ],
            ),
            Self::Atr(state) => append_words(
                bytes,
                [
                    u64::from(state.has_previous),
                    state.seed_count,
                    state.previous_close.to_bits(),
                    state.mean_tr.to_bits(),
                ],
            ),
            Self::Macd(state) => append_words(
                bytes,
                [
                    state.fast_count,
                    state.fast_value.to_bits(),
                    state.slow_count,
                    state.slow_value.to_bits(),
                    state.signal_count,
                    state.signal_value.to_bits(),
                ],
            ),
        }
    }
}
fn append_words<const N: usize>(bytes: &mut Vec<u8>, words: [u64; N]) {
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
}
fn decode_rsi(bytes: &[u8], positions: u64, has_previous: bool) -> Result<RsiKernel> {
    Ok(RsiKernel {
        positions,
        has_previous,
        seed_count: word(bytes, 8)?,
        previous_price: float(bytes, 16)?,
        mean_gain: float(bytes, 24)?,
        mean_loss: float(bytes, 32)?,
    })
}
fn decode_atr(bytes: &[u8], positions: u64, has_previous: bool) -> Result<AtrKernel> {
    Ok(AtrKernel {
        positions,
        has_previous,
        seed_count: word(bytes, 8)?,
        previous_close: float(bytes, 16)?,
        mean_tr: float(bytes, 24)?,
    })
}
fn decode_macd(bytes: &[u8], positions: u64) -> Result<MacdKernel> {
    Ok(MacdKernel {
        positions,
        fast_count: word(bytes, 0)?,
        fast_value: float(bytes, 8)?,
        slow_count: word(bytes, 16)?,
        slow_value: float(bytes, 24)?,
        signal_count: word(bytes, 32)?,
        signal_value: float(bytes, 40)?,
    })
}
fn word(bytes: &[u8], start: usize) -> Result<u64> {
    let bytes = bytes
        .get(start..start + 8)
        .and_then(|bytes| bytes.try_into().ok())
        .ok_or_else(|| mismatch("truncated indicator word"))?;
    Ok(u64::from_le_bytes(bytes))
}
fn float(bytes: &[u8], start: usize) -> Result<f64> {
    word(bytes, start).map(f64::from_bits)
}
fn mismatch(message: &str) -> crate::CalcFlowError {
    state_format(format!("rolling indicator state: {message}"))
}
