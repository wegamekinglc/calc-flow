use super::*;
use crate::CalcFlowError;
use datafusion::arrow::array::{Array, PrimitiveArray};
use datafusion::arrow::datatypes::{Float64Type, UInt8Type};

#[test]
fn test_checkpoint_control_arrays_have_prepaid_exact_buffers_and_ieee_values() {
    let source = [
        None,
        Some(-0.0),
        Some(f64::from_bits(0x7ff8_0000_0000_0042)),
        Some(f64::INFINITY),
    ];
    let cost = requirement::<Float64Type>(source.len(), true).unwrap();
    let mut output = None;
    let actual = allocation_counter::measure(|| {
        output = Some(
            primitive::<Float64Type>(source.len(), true, &mut |index| source[index], &mut || {
                Ok(())
            })
            .unwrap(),
        );
    });
    assert!(cost.construction_peak >= actual.bytes_max);
    let output = output.unwrap();
    let output = output
        .as_any()
        .downcast_ref::<PrimitiveArray<Float64Type>>()
        .unwrap();
    assert_eq!(
        output.values().inner().capacity(),
        source.len() * size_of::<f64>()
    );
    assert_eq!(output.nulls().unwrap().buffer().capacity(), 64);
    assert!(output.is_null(0));
    assert_eq!(output.value(0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(output.value(1).to_bits(), (-0.0_f64).to_bits());
    assert_eq!(output.value(2).to_bits(), 0x7ff8_0000_0000_0042);
    assert_eq!(output.value(3).to_bits(), f64::INFINITY.to_bits());
    let normalized = u64::try_from(actual.bytes_current).unwrap()
        - u64::try_from(7 * size_of::<usize>() * 2).unwrap();
    assert_eq!(cost.retained, normalized);
}

#[test]
fn test_checkpoint_control_arrays_cover_empty_valid_and_refuse_unplanned_mask() {
    for rows in [0, 1, 17] {
        let cost = requirement::<UInt8Type>(rows, false).unwrap();
        let mut output = None;
        let actual = allocation_counter::measure(|| {
            output = Some(
                primitive::<UInt8Type>(rows, false, &mut |_| Some(3), &mut || Ok(())).unwrap(),
            );
        });
        assert!(cost.construction_peak >= actual.bytes_max);
        assert!(output.unwrap().nulls().is_none());
    }
    assert!(primitive::<UInt8Type>(1, false, &mut |_| None, &mut || Ok(())).is_err());
    assert!(requirement::<Float64Type>(usize::MAX, true).is_err());
}

#[test]
fn test_checkpoint_control_arrays_check_long_construction_cancel() {
    let mut calls = 0;
    let mut generated = 0;
    let error = primitive::<UInt8Type>(
        8_193,
        false,
        &mut |_| {
            generated += 1;
            Some(1)
        },
        &mut || {
            calls += 1;
            if calls == 2 {
                Err(CalcFlowError::Cancelled {
                    run_id: "control".into(),
                })
            } else {
                Ok(())
            }
        },
    )
    .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    assert_eq!(generated, 4_096);
}
