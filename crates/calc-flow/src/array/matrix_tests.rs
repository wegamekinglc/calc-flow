use super::{ArrayExecutionOptions, NativeArray, invalid, matrix};

#[test]
fn matmul_cancellation_interrupts_a_single_long_dot_product() {
    let left = NativeArray::from_f64(vec![1.0; 8193], vec![1, 8193], None).unwrap();
    let right = NativeArray::from_f64(vec![1.0; 8193], vec![8193, 1], None).unwrap();
    let mut calls = 0;
    let mut check = || {
        calls += 1;
        if calls == 4 {
            Err(invalid("test.cancel", "cancelled"))
        } else {
            Ok(())
        }
    };
    let result = matrix::matmul_checked(
        left.view(),
        right.view(),
        &ArrayExecutionOptions::default(),
        &mut check,
    );
    assert!(result.unwrap_err().to_string().contains("test.cancel"));
    assert_eq!(calls, 4);
}

#[test]
fn transpose_cancellation_interrupts_a_large_copy() {
    let input = NativeArray::from_f32(vec![1.0; 8193], vec![3, 2731], None).unwrap();
    let mut calls = 0;
    let mut check = || {
        calls += 1;
        if calls == 3 {
            Err(invalid("test.cancel", "cancelled"))
        } else {
            Ok(())
        }
    };
    let result = matrix::transpose_checked(&input, &ArrayExecutionOptions::default(), &mut check);
    assert!(result.unwrap_err().to_string().contains("test.cancel"));
    assert_eq!(calls, 3);
}

#[test]
fn matrix_alias_checks_control_before_allocation_and_return() {
    let input = NativeArray::from_f32(vec![1.0], vec![], None).unwrap();
    let limits = ArrayExecutionOptions::default().with_max_elements(0);
    let mut check = || Err(invalid("test.cancel", "cancelled"));
    assert!(
        matrix::reshape_checked(&input, &[1, 1], &limits, &mut check)
            .unwrap_err()
            .to_string()
            .contains("test.cancel")
    );
    let mut calls = 0;
    let mut check = || {
        calls += 1;
        if calls == 2 {
            Err(invalid("test.cancel", "cancelled"))
        } else {
            Ok(())
        }
    };
    assert!(
        matrix::reshape_checked(
            &input,
            &[1, 1],
            &ArrayExecutionOptions::default(),
            &mut check
        )
        .is_err()
    );
    assert_eq!(calls, 2);
}

#[test]
fn axis_reduction_cancellation_interrupts_an_individual_lane() {
    let input = NativeArray::from_f64(vec![1.0; 8193], vec![1, 8193], None).unwrap();
    let mut calls = 0;
    let mut check = || {
        calls += 1;
        if calls == 4 {
            Err(invalid("test.cancel", "cancelled"))
        } else {
            Ok(())
        }
    };
    let result = super::reduction::reduce_checked(
        input.view(),
        super::Reduction::Sum,
        super::ReductionOptions::default().with_axis(Some(1)),
        &ArrayExecutionOptions::default(),
        &mut check,
    );
    assert!(result.unwrap_err().to_string().contains("test.cancel"));
    assert_eq!(calls, 4);
}

#[test]
fn short_dot_monitor_dispatch_is_bounded() {
    let input = NativeArray::from_f64(vec![1.0; 8193], vec![8193, 1], None).unwrap();
    let right = NativeArray::from_f64(vec![1.0], vec![1, 1], None).unwrap();
    let mut calls = 0;
    let mut check = || {
        calls += 1;
        Ok(())
    };
    assert_eq!(
        matrix::matmul_checked(
            input.view(),
            right.view(),
            &ArrayExecutionOptions::default(),
            &mut check
        )
        .unwrap()
        .element_count(),
        8193
    );
    assert!(
        calls <= 10,
        "monitor dispatched {calls} times for short dots"
    );
}

#[test]
fn short_axis_monitor_dispatch_is_bounded() {
    let input = NativeArray::from_f64(vec![1.0; 8193], vec![8193, 1], None).unwrap();
    let mut calls = 0;
    let mut check = || {
        calls += 1;
        Ok(())
    };
    assert_eq!(
        super::reduction::reduce_checked(
            input.view(),
            super::Reduction::Sum,
            super::ReductionOptions::default().with_axis(Some(1)),
            &ArrayExecutionOptions::default(),
            &mut check
        )
        .unwrap()
        .element_count(),
        8193
    );
    assert!(
        calls <= 10,
        "monitor dispatched {calls} times for short lanes"
    );
}

#[test]
fn descriptor_only_axis_requirements_bound_nullable_output_and_exact_scratch() {
    let options = super::ReductionOptions::default()
        .with_axis(Some(1))
        .with_keepdims(true);
    let cost = super::reduction::reduction_requirements(
        super::ArrayDType::Float64,
        &[2, 3],
        true,
        super::Reduction::Mean,
        options,
    )
    .unwrap();
    assert_eq!(
        cost,
        (
            2,
            2 * size_of::<f64>() + 1 + 2 * size_of::<usize>(),
            super::reduction::exact_scratch_bytes()
        )
    );
    let cost = super::reduction::reduction_requirements(
        super::ArrayDType::Float32,
        &[2, 3],
        false,
        super::Reduction::Min,
        options.with_axis(Some(0)).with_keepdims(false),
    )
    .unwrap();
    assert_eq!(cost, (3, 3 * size_of::<f32>() + size_of::<usize>(), 0));
    let full = super::reduction::reduction_requirements(
        super::ArrayDType::Float64,
        &[2, 3],
        true,
        super::Reduction::Sum,
        super::ReductionOptions::default(),
    )
    .unwrap();
    assert_eq!(
        full,
        (
            1,
            size_of::<f64>() + 1,
            super::reduction::exact_scratch_bytes()
        )
    );
}

#[test]
fn descriptor_only_axis_requirements_handle_zero_lanes_and_invalid_descriptors() {
    let options = super::ReductionOptions::default().with_axis(Some(1));
    let empty = super::reduction::reduction_requirements(
        super::ArrayDType::Float64,
        &[2, 0],
        false,
        super::Reduction::Sum,
        options,
    )
    .unwrap();
    assert_eq!(empty, (2, 2 * size_of::<f64>() + 1 + size_of::<usize>(), 0));
    let none = super::reduction::reduction_requirements(
        super::ArrayDType::Float64,
        &[0, 3],
        true,
        super::Reduction::Mean,
        options,
    )
    .unwrap();
    assert_eq!(none, (0, size_of::<usize>(), 0));
    assert!(
        super::reduction::reduction_requirements(
            super::ArrayDType::Boolean,
            &[0, 3],
            false,
            super::Reduction::Sum,
            options
        )
        .unwrap_err()
        .to_string()
        .contains("input.dtype")
    );
    assert!(
        super::reduction::reduction_requirements(
            super::ArrayDType::Float64,
            &[2, 3],
            false,
            super::Reduction::Sum,
            options.with_axis(Some(2))
        )
        .unwrap_err()
        .to_string()
        .contains("options.axis")
    );
    assert!(
        super::reduction::reduction_requirements(
            super::ArrayDType::Float64,
            &[1, 1, 1],
            false,
            super::Reduction::Sum,
            options
        )
        .unwrap_err()
        .to_string()
        .contains("input.shape")
    );
    assert!(
        super::reduction::reduction_requirements(
            super::ArrayDType::Float64,
            &[0, usize::MAX],
            false,
            super::Reduction::Sum,
            options.with_axis(Some(0))
        )
        .unwrap_err()
        .to_string()
        .contains("outputs.output.bytes")
    );
}
