use super::{AtrKernel, RsiKernel};

#[test]
fn test_restored_invariants_reject_hidden_reset_and_counter_state() {
    let good = RsiKernel::default();
    good.validate(2).unwrap();
    for bad in [
        RsiKernel {
            mean_loss: -0.0,
            ..good
        },
        RsiKernel {
            previous_price: 1.0,
            ..good
        },
        RsiKernel {
            has_previous: true,
            ..good
        },
        RsiKernel {
            positions: 1,
            has_previous: true,
            seed_count: 1,
            ..good
        },
        RsiKernel {
            positions: 4,
            has_previous: true,
            seed_count: 3,
            ..good
        },
        RsiKernel {
            mean_gain: f64::INFINITY,
            ..good
        },
    ] {
        assert!(bad.validate(2).is_err());
    }
    let good = AtrKernel {
        positions: 3,
        has_previous: true,
        seed_count: 2,
        previous_close: -0.0,
        mean_tr: 1.0,
    };
    good.validate(2).unwrap();
    for bad in [
        AtrKernel {
            mean_tr: -1.0,
            ..good
        },
        AtrKernel {
            previous_close: f64::NAN,
            ..good
        },
        AtrKernel {
            seed_count: 0,
            ..good
        },
        AtrKernel {
            has_previous: false,
            ..good
        },
    ] {
        assert!(bad.validate(2).is_err());
    }
}

#[test]
fn test_maximum_period_and_position_checks_remain_constant_space() {
    let mut kernel = RsiKernel {
        positions: u64::MAX - 1,
        has_previous: true,
        seed_count: u64::MAX - 2,
        previous_price: 0.0,
        mean_gain: 1.0,
        mean_loss: 1.0,
    };
    assert!(kernel.push(Some(1.0), u64::MAX, "input").unwrap().is_none());
    assert_eq!(kernel.positions, u64::MAX);
    assert_eq!(kernel.seed_count, u64::MAX - 1);
    kernel.validate(u64::MAX).unwrap();
    let previous = kernel;
    assert!(kernel.push(None, u64::MAX, "input").is_err());
    assert_eq!(kernel.positions, previous.positions);
    assert_eq!(kernel.mean_gain.to_bits(), previous.mean_gain.to_bits());
}

#[test]
fn test_denominator_above_exact_integer_range_has_specified_rounding() {
    let mut kernel = AtrKernel {
        positions: u64::MAX - 1,
        has_previous: true,
        seed_count: (1 << 53) + 1,
        previous_close: 1.0,
        mean_tr: 2.0,
    };
    let output = kernel
        .push(
            [Some(5.0), Some(0.0), Some(2.0)],
            (1 << 53) + 1,
            ["high", "low", "close"],
        )
        .unwrap()
        .unwrap();
    let expected: f64 = 2.0 + (5.0 - 2.0) / 9_007_199_254_740_992.0;
    assert_eq!(output.to_bits(), expected.to_bits());
}
