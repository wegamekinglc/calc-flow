use crate::{
    CalcFlowError, Result,
    array::{
        self, ArrayExecutionOptions, ArrayView, EwmaOptions, FloatView, NativeArray,
        ReductionOptions, RollingMeanOptions,
    },
};

fn cancellation() -> CalcFlowError {
    CalcFlowError::Cancelled {
        run_id: "monitor-test".into(),
    }
}
fn stops_on_third_check(run: impl FnOnce(&mut dyn FnMut() -> Result<()>) -> Result<NativeArray>) {
    let mut calls = 0;
    let result = run(&mut || {
        calls += 1;
        if calls == 3 {
            Err(cancellation())
        } else {
            Ok(())
        }
    });
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(calls, 3);
}

#[test]
fn test_controlled_kernels_cancel_inside_a_single_long_node() {
    let values = vec![1.0; 8193];
    let view = FloatView::from_f64(&values);
    let input = ArrayView::from_float(view);
    let options = ArrayExecutionOptions::default();
    stops_on_third_check(|check| array::elementwise::neg_checked(input, &options, check));
    stops_on_third_check(|check| {
        array::elementwise::greater_checked(
            input,
            ArrayView::scalar_f64(Some(0.0)),
            &options,
            check,
        )
    });
    stops_on_third_check(|check| {
        array::elementwise::where_checked(
            ArrayView::scalar_boolean(Some(true)),
            input,
            input,
            &options,
            check,
        )
    });
    stops_on_third_check(|check| {
        array::reduction::sum_checked(view, ReductionOptions::default(), &options, check)
    });
    stops_on_third_check(|check| {
        array::reduction::min_checked(view, ReductionOptions::default(), &options, check)
    });
    stops_on_third_check(|check| {
        array::rolling::ewma::ewma_checked(view, EwmaOptions::new(3).unwrap(), &options, check)
    });
    stops_on_third_check(|check| {
        array::rolling_mean::rolling_mean_checked(
            view,
            RollingMeanOptions::new(5000).unwrap(),
            &options,
            check,
        )
    });
}

#[test]
fn test_stable_refold_checks_cancellation_inside_retained_history() {
    let mut checks = 0;
    let result = array::rolling_mean_math::stable_fold_controlled(
        std::iter::repeat_n(Some(1.0), 8193),
        &mut || {
            checks += 1;
            if checks == 3 {
                Err(cancellation())
            } else {
                Ok(())
            }
        },
        |_| cancellation(),
    );
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
    assert_eq!(checks, 3);
}

#[test]
fn test_ir_v1_runtime_shape_contract_rejects_higher_rank_storage() {
    assert!(super::execute::native_shape(&[1, 2], "inputs.x").is_err());
    assert!(matches!(
        super::execute::native_shape(&[], "inputs.x").unwrap(),
        super::infer::Shape::Scalar
    ));
    assert!(matches!(
        super::execute::native_shape(&[0], "inputs.x").unwrap(),
        super::infer::Shape::Vector(Some(0))
    ));
}

#[test]
fn test_exact_reduction_observes_a_final_check_after_its_scan() {
    let mut checks = 0;
    let result = array::reduction::sum_checked(
        FloatView::from_f64(&[1.0]),
        ReductionOptions::default(),
        &ArrayExecutionOptions::default(),
        &mut || {
            checks += 1;
            if checks == 4 {
                Err(cancellation())
            } else {
                Ok(())
            }
        },
    );
    assert!(matches!(result, Err(CalcFlowError::Cancelled { .. })));
}
