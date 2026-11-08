#[allow(
    clippy::cast_possible_truncation,
    reason = "digest-v1 float32 carriers use the declared IEEE conversion"
)]
pub(crate) fn float32(value: f64) -> [u8; 4] {
    let bits = if value.is_nan() {
        0x7fc0_0000_u32
    } else {
        (value as f32).to_bits()
    };
    bits.to_be_bytes()
}
pub(crate) fn float64(value: f64) -> [u8; 8] {
    let bits = if value.is_nan() {
        0x7ff8_0000_0000_0000_u64
    } else {
        value.to_bits()
    };
    bits.to_be_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float32_cells_preserve_signed_zero_subnormals_and_infinities() {
        for bits in [
            0x0000_0000,
            0x8000_0000,
            0x0000_0001,
            0x7f7f_ffff,
            0x7f80_0000,
            0xff80_0000,
        ] {
            assert_eq!(float32(f64::from(f32::from_bits(bits))), bits.to_be_bytes());
        }
    }
    #[test]
    fn test_float64_cells_preserve_signed_zero_subnormals_and_infinities() {
        for bits in [
            0x0000_0000_0000_0000,
            0x8000_0000_0000_0000,
            0x0000_0000_0000_0001,
            0x7fef_ffff_ffff_ffff,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
        ] {
            assert_eq!(float64(f64::from_bits(bits)), bits.to_be_bytes());
        }
    }
    #[test]
    fn test_cells_keep_digest_v1_canonical_nan_payloads() {
        for value in [
            f64::from_bits(0x7ff8_1234_5678_9abc),
            f64::from_bits(0xfff8_1234_5678_9abc),
        ] {
            assert_eq!(float32(value), 0x7fc0_0000_u32.to_be_bytes());
            assert_eq!(float64(value), 0x7ff8_0000_0000_0000_u64.to_be_bytes());
        }
    }
}
