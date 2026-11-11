use super::*;

#[test]
fn test_work_invariant_lanes_reuse_const_core_and_direct_float_writer() {
    let batch = input(64_000, 64, true);
    let mut mean_plan = lane_plan(false);
    mean_plan.groups[0] = TypedGroupPlan::Numeric {
        input_index: 3,
        frame: TypedFrame::Rows(37),
    };
    mean_plan.outputs[0].min_periods = 3;
    mean_plan.numerical_profile = RollingNumericalProfile::StableV2Preview;
    mean_plan.nan_as_value = true;
    let mean = assert_parity(&mean_plan, &batch);

    let finite = input(64_000, 64, false);
    let mut sum_plan = lane_plan(false);
    sum_plan.groups[0] = TypedGroupPlan::Numeric {
        input_index: 3,
        frame: TypedFrame::Duration(2_048),
    };
    sum_plan.outputs[0].kind = TypedOutputKind::Statistic(Statistic::Sum);
    let sum = assert_parity(&sum_plan, &finite);
    assert_eq!(
        (mean.0, mean.1, mean.2 + sum.2),
        (64_000, 64_000, 0),
        "lanes must execute the shared pure mean core and concrete Float64 writers"
    );
}

#[test]
fn test_capability_parallel_shared_rows_preserve_v2_rebase_and_nan_policy() {
    let batch = input(64_000, 64, true);
    let mut plan = lane_plan(true);
    plan.numerical_profile = RollingNumericalProfile::StableV2Preview;
    plan.nan_as_value = true;
    for (group, width) in plan.groups.iter_mut().zip([37, 71]) {
        *group = TypedGroupPlan::Numeric {
            input_index: 3,
            frame: TypedFrame::Rows(width),
        };
    }
    plan.shared_rows = plan.shared_rows_shape();
    assert_parity(&plan, &batch);
    let finite = input(64_000, 64, false);
    assert_parity(&plan, &finite);
}

#[test]
fn test_capability_parallel_executes_arbitrary_groups_and_f64_readouts() {
    let batch = input(64_000, 64, false);
    let mut plan = lane_plan(false);
    plan.groups = vec![
        TypedGroupPlan::Numeric {
            input_index: 3,
            frame: TypedFrame::Duration(2_048),
        },
        TypedGroupPlan::Numeric {
            input_index: 3,
            frame: TypedFrame::Rows(37),
        },
        TypedGroupPlan::Ewma {
            input_index: 3,
            alpha: 0.25,
            cumulative: false,
        },
        TypedGroupPlan::Cumulative {
            input_index: 3,
            statistic: Statistic::Sum,
            storage: OutputStorage::Float64,
        },
    ];
    plan.outputs = [
        TypedOutputKind::Statistic(Statistic::Sum),
        TypedOutputKind::Statistic(Statistic::Variance),
        TypedOutputKind::Ewma,
        TypedOutputKind::Statistic(Statistic::Sum),
    ]
    .into_iter()
    .enumerate()
    .map(|(group, kind)| TypedOutputPlan {
        group,
        kind,
        storage: OutputStorage::Float64,
        min_periods: 1,
        ddof: 0,
    })
    .collect();
    assert_parity(&plan, &batch);
}

#[test]
fn test_capability_parallel_keeps_unprepared_decay_and_non_f64_output_serial() {
    let batch = input(64_000, 64, false);
    for storage in [
        OutputStorage::Count,
        OutputStorage::Int64,
        OutputStorage::Float32,
    ] {
        let mut plan = lane_plan(false);
        plan.outputs[0].storage = storage;
        assert!(!plan.supports_entity_parallel(&batch));
    }
    let mut plan = lane_plan(false);
    plan.groups.push(TypedGroupPlan::Scan {
        input_index: 3,
        kind: ScanKind::Decay,
        window: 37,
    });
    assert!(
        !plan.supports_entity_parallel(&batch),
        "the whole plan must stay on reference/quota-aware decay preparation"
    );
}

#[test]
fn test_capability_parallel_accounts_for_many_output_descriptors() {
    check_lane_builder_allocation([32_000, 32_000], 7, true);
    let batch = input(64_000, 64, false);
    let mut plan = lane_plan(false);
    plan.outputs = vec![plan.outputs[0]; 7];
    assert_parity(&plan, &batch);
}

#[test]
fn test_capability_parallel_skew_uses_existing_duration_work() {
    let batch = input(64_000, 64, false);
    let mut plan = lane_plan(false);
    plan.groups[0] = TypedGroupPlan::Numeric {
        input_index: 3,
        frame: TypedFrame::Duration(1_000_000),
    };
    let mut state = warm(&plan, &batch);
    let entity = Arc::make_mut(&mut state.states[0]);
    entity.transition_count += 256_000;
    let TypedWindowState::Numeric(numeric) = &mut entity.groups[0] else {
        unreachable!()
    };
    for time in (-300_000..-44_000).rev() {
        numeric.samples.values.push_front(TimedSample {
            event_time: time,
            value: None,
        });
    }
    let prepared = plan
        .prepare_ordered_stream_inputs(&state, &batch, "r", None)
        .unwrap();
    let pointer = prepared.fill.stream.state.states.as_ptr();
    let rejected = prepared
        .split_two()
        .err()
        .expect("one entity owns nearly all duration expiry work");
    assert_eq!(rejected.fill.stream.state.states.as_ptr(), pointer);
}

#[test]
fn test_capability_parallel_exact_f64_readout_and_earliest_group_error() {
    let (unsigned, unsigned_batch) = exact_fixture(false);
    assert_parity(&unsigned, &unsigned_batch);
    let (plan, batch) = exact_fixture(true);
    assert_parity(&plan, &batch);
    let mut state = warm(&plan, &batch);
    for entity in state.states.iter_mut().take(2) {
        let TypedWindowState::Exact(exact) = &mut Arc::make_mut(entity).groups[0] else {
            unreachable!()
        };
        exact.accumulator.sum = Some(SumState::Signed(i128::MAX));
    }
    let before = format!("{:?}", state.states);
    let expected = plan
        .prepare_ordered_stream(&state, &batch, "r", None)
        .err()
        .expect("injected sum overflow");
    let prepared = plan
        .prepare_ordered_stream_inputs(&state, &batch, "r", None)
        .unwrap();
    let (seed, requests) = prepared
        .split_two()
        .ok()
        .expect("independent exact means have F64 readouts");
    let mut exits = requests.map(|request| exit(request, &AtomicBool::new(false)));
    for (lane, exit) in exits.iter().enumerate() {
        let NumericLaneOutcome::OrdinaryError(error) = &exit.outcome else {
            panic!("both exact lanes must hit checked arithmetic");
        };
        assert_eq!(error.site.row, u32::try_from(lane).unwrap());
        assert_eq!(error.site.phase, NumericPhase::Group);
        assert_eq!(error.site.ordinal, 0);
    }
    let actual = merge_numeric_results(seed, &mut exits, "job")
        .err()
        .expect("earliest checked error");
    assert_eq!(actual.to_string(), expected.to_string());
    assert_eq!(format!("{:?}", state.states), before);
}

#[test]
fn test_capability_parallel_rejects_incomplete_row_coverage_before_moving_state() {
    let batch = input(64_000, 64, false);
    let plan = lane_plan(false);
    let state = warm(&plan, &batch);
    for missing_entity in [false, true] {
        let mut prepared = plan
            .prepare_ordered_stream_inputs(&state, &batch, "r", None)
            .unwrap();
        if missing_entity {
            prepared.fill.stream.entity_rows.counts.pop();
        } else {
            prepared.fill.stream.entity_rows.counts[0] -= 1;
        }
        let pointer = prepared.fill.stream.state.states.as_ptr();
        let rejected = prepared
            .split_two()
            .err()
            .expect("invalid routing must keep prepared serial ownership");
        assert_eq!(rejected.fill.stream.state.states.as_ptr(), pointer);
    }
}

fn exact_fixture(signed: bool) -> (RollingKernelPlan, RecordBatch) {
    let batch = input(64_000, 64, false);
    let mut columns = batch.columns().to_vec();
    columns[3] = if signed {
        Arc::new(Int64Array::from_iter_values(std::iter::repeat_n(
            1,
            batch.num_rows(),
        )))
    } else {
        Arc::new(UInt64Array::from_iter_values(std::iter::repeat_n(
            u64::MAX,
            batch.num_rows(),
        )))
    };
    let fields = batch
        .schema()
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            if index == 3 {
                Arc::new(Field::new(
                    "price",
                    if signed {
                        DataType::Int64
                    } else {
                        DataType::UInt64
                    },
                    true,
                ))
            } else {
                Arc::clone(field)
            }
        })
        .collect::<Vec<_>>();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
    let mut plan = lane_plan(false);
    plan.groups[0] = if signed {
        TypedGroupPlan::Signed {
            input_index: 3,
            frame: TypedFrame::Rows(37),
        }
    } else {
        TypedGroupPlan::Unsigned {
            input_index: 3,
            frame: TypedFrame::Rows(37),
        }
    };
    (plan, batch)
}

fn assert_parity(plan: &RollingKernelPlan, batch: &RecordBatch) -> (usize, usize, usize) {
    let state = warm(plan, batch);
    let before = format!("{:?}", state.states);
    let expected = plan
        .prepare_ordered_stream(&state, batch, "r", None)
        .unwrap();
    let prepared = plan
        .prepare_ordered_stream_inputs(&state, batch, "r", None)
        .unwrap();
    let (seed, requests) = prepared
        .split_two()
        .ok()
        .expect("independent F64 typed transitions split");
    mean::SHARED_GROUP_UPDATE_VISITS.set(0);
    mean::SHARED_MEAN_READOUT_VISITS.set(0);
    GENERIC_F64_BUILDER_VISITS.set(0);
    let mut exits = requests.map(|request| exit(request, &AtomicBool::new(false)));
    let visits = (
        mean::SHARED_GROUP_UPDATE_VISITS.replace(0),
        mean::SHARED_MEAN_READOUT_VISITS.replace(0),
        GENERIC_F64_BUILDER_VISITS.replace(0),
    );
    let actual = merge_numeric_results(seed, &mut exits, "job").unwrap();
    assert_eq!(
        actual.execution.columns.len(),
        expected.execution.columns.len()
    );
    for (actual, expected) in actual
        .execution
        .columns
        .iter()
        .zip(&expected.execution.columns)
    {
        let actual = actual.as_any().downcast_ref::<Float64Array>().unwrap();
        let expected = expected.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(actual.nulls(), expected.nulls());
        assert!(
            actual
                .values()
                .iter()
                .zip(expected.values())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
    assert_eq!(
        format!("{:?}", actual.execution.state.states),
        format!("{:?}", expected.execution.state.states)
    );
    assert_eq!(format!("{:?}", state.states), before);
    assert_eq!(actual.execution.entity_ids, expected.execution.entity_ids);
    visits
}
