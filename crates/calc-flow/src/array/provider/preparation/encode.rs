use super::source::{self, Latched, Values};
use crate::{
    Result, STATIC_INPUT_DIGEST_VERSION, StaticInputDigest,
    array::{ArrayDType, control},
    batch::native_cell,
};
use sha2::{Digest, Sha256};

const CHUNK_BYTES: usize = 4096;
pub(super) fn workspace(aliases: usize) -> Result<usize> {
    aliases
        .checked_mul(size_of::<Sha256>())
        .and_then(|bytes| bytes.checked_add(CHUNK_BYTES))
        .ok_or_else(|| source::invalid("bytes", "native digest workspace overflowed"))
}
pub(super) fn digests(
    aliases: &[&str],
    value: &Latched<'_>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<StaticInputDigest>> {
    check()?;
    let mut hashers = prefixes(aliases, check)?;
    encode(&mut hashers, value, check)?;
    finish(hashers, check)
}
fn encode(
    hashers: &mut [Sha256],
    value: &Latched<'_>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    let mut writer = Writer {
        hashers,
        bytes: [0; CHUNK_BYTES],
        used: 0,
        check,
    };
    descriptor(&mut writer, value)?;
    cells(&mut writer, value)?;
    writer.flush()
}
fn finish(
    hashers: Vec<Sha256>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<StaticInputDigest>> {
    let mut output = Vec::with_capacity(hashers.len());
    for (index, state) in hashers.into_iter().enumerate() {
        hash_checkpoint(index, check)?;
        output.push(StaticInputDigest {
            digest_version: STATIC_INPUT_DIGEST_VERSION.into(),
            sha256: hex::encode(state.finalize()),
        });
    }
    check()?;
    Ok(output)
}
fn prefixes(aliases: &[&str], check: &mut dyn FnMut() -> Result<()>) -> Result<Vec<Sha256>> {
    let mut hashers = Vec::with_capacity(aliases.len());
    for (index, alias) in aliases.iter().enumerate() {
        hash_checkpoint(index, check)?;
        let mut state = Sha256::new();
        state.update(STATIC_INPUT_DIGEST_VERSION.as_bytes());
        state.update([0, 0x01]);
        state.update(length(alias.len())?.to_be_bytes());
        for chunk in alias.as_bytes().chunks(CHUNK_BYTES) {
            check()?;
            state.update(chunk);
        }
        state.update([0x11]);
        hashers.push(state);
    }
    Ok(hashers)
}
fn descriptor(writer: &mut Writer<'_>, value: &Latched<'_>) -> Result<()> {
    writer.text(value.backend.as_bytes())?;
    writer.write(&[tag(value.dtype)])?;
    shape(writer, value.shape)?;
    writer.usize(source::elements(value.shape)?)
}
fn tag(dtype: ArrayDType) -> u8 {
    match dtype {
        ArrayDType::Boolean => 0x40,
        ArrayDType::Float32 => 0x49,
        ArrayDType::Float64 => 0x4a,
    }
}
fn shape(writer: &mut Writer<'_>, shape: &[u64]) -> Result<()> {
    writer.usize(shape.len())?;
    for dimension in shape {
        writer.write(&dimension.to_be_bytes())?;
    }
    Ok(())
}
fn cells(writer: &mut Writer<'_>, value: &Latched<'_>) -> Result<()> {
    let elements = source::elements(value.shape)?;
    value.validate_flags(elements)?;
    let mut next = 0;
    for index in 0..elements {
        control::checkpoint(index, writer.check)?;
        cell(writer, value, index, &mut next)?;
    }
    value.validate_count(next)
}

fn cell(
    writer: &mut Writer<'_>,
    value: &Latched<'_>,
    index: usize,
    next: &mut usize,
) -> Result<()> {
    if value.is_null(index) {
        writer.write(&[0x30])
    } else {
        scalar(writer, value, *next)?;
        *next += 1;
        Ok(())
    }
}
fn scalar(writer: &mut Writer<'_>, value: &Latched<'_>, index: usize) -> Result<()> {
    writer.write(&[0x31])?;
    match value.values {
        Values::Boolean(values) => boolean_cell(writer, value.dtype, *required(values, index)?),
        Values::Float(values) => float_cell(writer, value.dtype, *required(values, index)?),
    }
}
fn boolean_cell(writer: &mut Writer<'_>, dtype: ArrayDType, value: bool) -> Result<()> {
    if dtype != ArrayDType::Boolean {
        return Err(source::invalid(
            "dtype",
            "carrier family differs from dtype",
        ));
    }
    writer.write(&[u8::from(value)])
}
fn float_cell(writer: &mut Writer<'_>, dtype: ArrayDType, value: f64) -> Result<()> {
    match dtype {
        ArrayDType::Float32 => writer.write(&native_cell::float32(value)),
        ArrayDType::Float64 => writer.write(&native_cell::float64(value)),
        ArrayDType::Boolean => Err(source::invalid(
            "dtype",
            "carrier family differs from dtype",
        )),
    }
}
fn required<T>(values: &[T], index: usize) -> Result<&T> {
    values
        .get(index)
        .ok_or_else(|| source::invalid("values", "carrier does not cover valid cells"))
}
fn length(value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| source::invalid("length", "length exceeds the digest range"))
}
fn hash_checkpoint(index: usize, check: &mut dyn FnMut() -> Result<()>) -> Result<()> {
    if index.is_multiple_of(64) {
        check()?;
    }
    Ok(())
}
struct Writer<'a> {
    hashers: &'a mut [Sha256],
    bytes: [u8; CHUNK_BYTES],
    used: usize,
    check: &'a mut dyn FnMut() -> Result<()>,
}
impl Writer<'_> {
    fn usize(&mut self, value: usize) -> Result<()> {
        self.write(&length(value)?.to_be_bytes())
    }
    fn text(&mut self, bytes: &[u8]) -> Result<()> {
        self.usize(bytes.len())?;
        self.write(bytes)
    }
    fn write(&mut self, mut bytes: &[u8]) -> Result<()> {
        while !bytes.is_empty() {
            let count = bytes.len().min(CHUNK_BYTES - self.used);
            self.bytes[self.used..self.used + count].copy_from_slice(&bytes[..count]);
            self.used += count;
            bytes = &bytes[count..];
            if self.used == CHUNK_BYTES {
                self.flush()?;
            }
        }
        Ok(())
    }
    fn flush(&mut self) -> Result<()> {
        (self.check)()?;
        for (index, state) in self.hashers.iter_mut().enumerate() {
            hash_checkpoint(index, self.check)?;
            state.update(&self.bytes[..self.used]);
        }
        self.used = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::source::{Latched, Values};
    use super::*;
    use crate::array::ArrayDType;

    #[test]
    fn test_shared_encoding_preserves_physical_name_digest_v1_nan_and_signed_zero() {
        let values = [f64::from_bits(0x7ff0_0000_0000_0001), -0.0, f64::INFINITY];
        let view = Latched {
            backend: "native",
            dtype: ArrayDType::Float32,
            shape: &[4],
            nulls: Some(&[false, false, true, false]),
            values: Values::Float(&values),
            retained: 0,
        };
        let actual = digests(&["node_a.beta", "node_b.beta"], &view, &mut || Ok(())).unwrap();
        assert_eq!(
            actual
                .iter()
                .map(|value| value.digest_version.as_str())
                .collect::<Vec<_>>(),
            [STATIC_INPUT_DIGEST_VERSION; 2]
        );
        assert_eq!(
            actual
                .iter()
                .map(|value| value.sha256.as_str())
                .collect::<Vec<_>>(),
            [
                "24e506ba78a537f37f44bb2449e196ef6349bcd794c3a6577044d5488482392f",
                "efd29039e584455346d181157adb294a3cf4b31b450030fbc51a388ca3f5dee4"
            ]
        );
        assert_eq!(values[1].to_bits(), (-0.0_f64).to_bits());
    }

    #[test]
    fn test_encoding_cancellation_checks_long_null_scans_before_a_chunk_fills() {
        let nulls = vec![true; 8193];
        let view = Latched {
            backend: "native",
            dtype: ArrayDType::Float64,
            shape: &[8193],
            nulls: Some(&nulls),
            values: Values::Float(&[]),
            retained: 0,
        };
        let mut checks = 0;
        let result = digests(&["calc.beta"], &view, &mut || {
            checks += 1;
            if checks == 4 {
                Err(crate::CalcFlowError::Cancelled {
                    run_id: "encode".into(),
                })
            } else {
                Ok(())
            }
        });
        assert!(matches!(
            result,
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
        assert_eq!(checks, 4);
    }

    #[test]
    fn test_many_physical_aliases_have_bounded_hash_cancellation_and_checked_workspace() {
        let names = (0..129)
            .map(|index| format!("node_{index}.beta"))
            .collect::<Vec<_>>();
        let aliases = names.iter().map(String::as_str).collect::<Vec<_>>();
        let view = Latched {
            backend: "native",
            dtype: ArrayDType::Boolean,
            shape: &[],
            nulls: None,
            values: Values::Boolean(&[true]),
            retained: 0,
        };
        let mut checks = 0;
        let result = digests(&aliases, &view, &mut || {
            checks += 1;
            if checks == 3 {
                Err(crate::CalcFlowError::Cancelled {
                    run_id: "hash".into(),
                })
            } else {
                Ok(())
            }
        });
        assert!(matches!(
            result,
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
        assert_eq!(
            workspace(aliases.len()).unwrap(),
            4096 + 129 * size_of::<Sha256>()
        );
        assert!(workspace(usize::MAX).is_err());
    }
}
