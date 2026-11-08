use std::cmp::Ordering;

const LIMBS: usize = (2098 + usize::BITS as usize).div_ceil(64);
const FRACTION_MASK: u64 = (1_u64 << 52) - 1;
const POSITIVE_INFINITY: u8 = 1;
const NEGATIVE_INFINITY: u8 = 2;
const NAN: u8 = 4;

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct CountOverflow;

pub(crate) struct ExactSum {
    positive: [u64; LIMBS],
    negative: [u64; LIMBS],
    count: usize,
    nonfinite: u8,
}

impl ExactSum {
    pub(crate) const fn scratch_bytes() -> usize {
        // Cover the accumulator and conservatively bound finalization arrays.
        size_of::<Self>() + 4 * size_of::<[u64; LIMBS]>()
    }

    pub(crate) const fn new() -> Self {
        Self {
            positive: [0; LIMBS],
            negative: [0; LIMBS],
            count: 0,
            nonfinite: 0,
        }
    }

    pub(crate) fn push(&mut self, value: f64) -> Result<(), CountOverflow> {
        let count = self.count.checked_add(1).ok_or(CountOverflow)?;
        if !value.is_finite() {
            self.nonfinite |= if value.is_nan() {
                NAN
            } else if value.is_sign_negative() {
                NEGATIVE_INFINITY
            } else {
                POSITIVE_INFINITY
            };
            self.count = count;
            return Ok(());
        }
        let bits = value.to_bits();
        let bytes = bits.to_le_bytes();
        let exponent = (u16::from_le_bytes([bytes[6], bytes[7]]) >> 4) & 0x7ff;
        let fraction = bits & FRACTION_MASK;
        let significand = fraction | u64::from(exponent != 0) << 52;
        let shift = exponent.saturating_sub(1);
        let index = usize::from(shift / 64);
        let offset = u32::from(shift % 64);
        let words = if bits >> 63 == 0 {
            &mut self.positive
        } else {
            &mut self.negative
        };
        add_word(words, index, significand << offset);
        if offset != 0 {
            add_word(words, index + 1, significand >> (64 - offset));
        }
        self.count = count;
        Ok(())
    }

    pub(crate) fn sum(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        if let Some(special) = self.special() {
            return Some(special);
        }
        let (negative, words) = self.magnitude();
        Some(round(&words, negative, 0, 1))
    }

    pub(crate) fn mean(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        if let Some(special) = self.special() {
            return Some(special);
        }
        let (negative, words) = self.magnitude();
        let divisor = self.count as u128;
        let (quotient, remainder) = divide(&words, divisor);
        Some(round(&quotient, negative, remainder, divisor))
    }

    fn special(&self) -> Option<f64> {
        if self.nonfinite & NAN != 0
            || self.nonfinite & (POSITIVE_INFINITY | NEGATIVE_INFINITY)
                == POSITIVE_INFINITY | NEGATIVE_INFINITY
        {
            Some(f64::from_bits(0x7ff8_0000_0000_0000))
        } else if self.nonfinite & POSITIVE_INFINITY != 0 {
            Some(f64::INFINITY)
        } else if self.nonfinite & NEGATIVE_INFINITY != 0 {
            Some(f64::NEG_INFINITY)
        } else {
            None
        }
    }

    fn magnitude(&self) -> (bool, [u64; LIMBS]) {
        let order = self.positive.iter().rev().cmp(self.negative.iter().rev());
        match order {
            Ordering::Less => (true, subtract(&self.negative, &self.positive)),
            Ordering::Equal => (false, [0; LIMBS]),
            Ordering::Greater => (false, subtract(&self.positive, &self.negative)),
        }
    }
}

fn add_word(words: &mut [u64; LIMBS], mut index: usize, mut value: u64) {
    while value != 0 {
        let (next, carry) = words[index].overflowing_add(value);
        words[index] = next;
        value = u64::from(carry);
        index += 1;
    }
}

fn subtract(left: &[u64; LIMBS], right: &[u64; LIMBS]) -> [u64; LIMBS] {
    let mut result = [0; LIMBS];
    let mut borrow = false;
    for ((output, left), right) in result.iter_mut().zip(left).zip(right) {
        let (difference, first) = left.overflowing_sub(*right);
        let (difference, second) = difference.overflowing_sub(u64::from(borrow));
        *output = difference;
        borrow = first || second;
    }
    result
}

fn divide(words: &[u64; LIMBS], divisor: u128) -> ([u64; LIMBS], u128) {
    let mut quotient = [0; LIMBS];
    let mut remainder = 0;
    for (output, word) in quotient.iter_mut().zip(words).rev() {
        let dividend = (remainder << 64) | u128::from(*word);
        // The remainder is below divisor, so one quotient limb fits u64.
        *output = u64::try_from(dividend / divisor).expect("one quotient limb fits u64");
        remainder = dividend % divisor;
    }
    (quotient, remainder)
}

fn round(words: &[u64; LIMBS], negative: bool, remainder: u128, divisor: u128) -> f64 {
    let sign = u64::from(negative) << 63;
    let Some(index) = words.iter().rposition(|word| *word != 0) else {
        return f64::from_bits(sign | u64::from(remainder * 2 > divisor));
    };
    let mut highest = index * 64 + 63 - words[index].leading_zeros() as usize;
    let shift = highest.saturating_sub(52);
    let mut significand = shifted_word(words, shift);
    let round_up = if shift == 0 {
        remainder * 2 > divisor || (remainder * 2 == divisor && significand & 1 != 0)
    } else {
        bit(words, shift - 1)
            && (any_below(words, shift - 1) || remainder != 0 || significand & 1 != 0)
    };
    significand += u64::from(round_up);
    if significand == 1 << 53 {
        significand >>= 1;
        highest += 1;
    }
    if highest < 52 {
        return f64::from_bits(sign | significand);
    }
    let exponent = highest - 51;
    if exponent >= 0x7ff {
        return f64::from_bits(sign | 0x7ff0_0000_0000_0000);
    }
    f64::from_bits(sign | (exponent as u64) << 52 | (significand & FRACTION_MASK))
}

fn shifted_word(words: &[u64; LIMBS], shift: usize) -> u64 {
    let index = shift / 64;
    let offset = shift % 64;
    let low = words[index] >> offset;
    if offset == 0 {
        low
    } else {
        low | words.get(index + 1).copied().unwrap_or(0) << (64 - offset)
    }
}

fn bit(words: &[u64; LIMBS], position: usize) -> bool {
    words[position / 64] & (1_u64 << (position % 64)) != 0
}

fn any_below(words: &[u64; LIMBS], bits: usize) -> bool {
    let full = bits / 64;
    let partial = bits % 64;
    words[..full].iter().any(|word| *word != 0)
        || (partial != 0 && words[full] & ((1_u64 << partial) - 1) != 0)
}

#[cfg(test)]
#[path = "exact_fixtures.rs"]
mod fixtures;

#[cfg(test)]
mod tests {
    use super::{ExactSum, fixtures::CASES};

    #[test]
    fn cancellation_preserves_the_small_exact_residual() {
        let mut total = ExactSum::new();
        for value in [1e308, 1.0, -1e308] {
            total.push(value).unwrap();
        }
        assert_eq!(total.sum().unwrap().to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn mean_does_not_overflow_with_a_finite_average() {
        let mut total = ExactSum::new();
        total.push(1e308).unwrap();
        total.push(1e308).unwrap();
        assert_eq!(total.mean().unwrap().to_bits(), 1e308_f64.to_bits());
    }

    #[test]
    fn nonfinite_values_follow_canonical_rules() {
        for values in [
            vec![f64::from_bits(0xfff0_0000_0000_0042), 1.0],
            vec![f64::INFINITY, f64::NEG_INFINITY],
        ] {
            let mut total = ExactSum::new();
            for value in values {
                total.push(value).unwrap();
            }
            assert_eq!(total.sum().unwrap().to_bits(), 0x7ff8_0000_0000_0000);
            assert_eq!(total.mean().unwrap().to_bits(), 0x7ff8_0000_0000_0000);
        }
    }

    #[test]
    fn independently_rounded_fraction_fixtures_match() {
        for (case, &(inputs, sum, mean)) in CASES.iter().enumerate() {
            let mut total = ExactSum::new();
            for &value in inputs {
                total.push(f64::from_bits(value)).unwrap();
            }
            assert_eq!(total.sum().unwrap().to_bits(), sum, "sum case {case}");
            assert_eq!(total.mean().unwrap().to_bits(), mean, "mean case {case}");
        }
    }

    #[test]
    fn empty_input_has_no_value_and_exact_zero_is_positive() {
        let mut total = ExactSum::new();
        assert!(total.sum().is_none());
        assert!(total.mean().is_none());
        for value in [-0.0, 0.0, -1.0, 1.0] {
            total.push(value).unwrap();
        }
        assert_eq!(total.sum().unwrap().to_bits(), 0);
        assert_eq!(total.mean().unwrap().to_bits(), 0);
    }

    #[test]
    fn a_single_infinity_sign_is_preserved() {
        for value in [f64::INFINITY, f64::NEG_INFINITY] {
            let mut total = ExactSum::new();
            for input in [1.0, value, -1.0] {
                total.push(input).unwrap();
            }
            assert_eq!(total.sum().unwrap().to_bits(), value.to_bits());
            assert_eq!(total.mean().unwrap().to_bits(), value.to_bits());
        }
    }

    #[test]
    fn count_overflow_leaves_every_accumulator_field_unchanged() {
        let mut total = ExactSum::new();
        total.push(123.0).unwrap();
        total.count = usize::MAX;
        let prior = (total.positive, total.negative, total.nonfinite);
        assert!(total.push(f64::NAN).is_err());
        assert_eq!(total.count, usize::MAX);
        assert_eq!((total.positive, total.negative, total.nonfinite), prior);
    }

    #[test]
    fn carries_across_high_limbs_preserve_cancellation() {
        let mut total = ExactSum::new();
        for _ in 0..65_536 {
            total.push(f64::MAX).unwrap();
        }
        for _ in 1..65_536 {
            total.push(-f64::MAX).unwrap();
        }
        assert_eq!(total.sum().unwrap().to_bits(), f64::MAX.to_bits());
    }
}
