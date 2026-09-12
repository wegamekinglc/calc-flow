use std::{
    panic::AssertUnwindSafe,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use datafusion::arrow::{
    array::StringArray,
    datatypes::{Field, TimeUnit},
};

use super::{entity_parallel::*, *};
use crate::CalcFlowError;

fn lane_plan(dual: bool) -> RollingKernelPlan {
    let groups = if dual {
        vec![
            TypedGroupPlan::Numeric {
                input_index: 3,
                frame: TypedFrame::Rows(5),
            },
            TypedGroupPlan::Numeric {
                input_index: 3,
                frame: TypedFrame::Rows(20),
            },
        ]
    } else {
        vec![TypedGroupPlan::Numeric {
            input_index: 3,
            frame: TypedFrame::Rows(20),
        }]
    };
    let mean = |group| TypedFloatReadout {
        group,
        kind: TypedFloatReadoutKind::Mean,
        min_periods: if dual && group == 0 { 5 } else { 20 },
        ddof: 0,
    };
    RollingKernelPlan {
        version: ROLLING_KERNEL_PLAN_VERSION,
        state_layout_version: 3,
        numerical_profile: RollingNumericalProfile::StableV1,
        selection: KernelSelection::OrderedPrimitive,
        complexity: KernelComplexity::AmortizedConstant,
        event_time_index: 0,
        order_columns: vec![0, 1],
        partition_columns: vec![2],
        sequence_columns: vec![1],
        nan_as_value: false,
        groups,
        outputs: vec![TypedOutputPlan {
            group: 0,
            kind: if dual {
                TypedOutputKind::Difference {
                    left: mean(0),
                    right: mean(1),
                }
            } else {
                TypedOutputKind::Statistic(Statistic::Mean)
            },
            storage: OutputStorage::Float64,
            min_periods: 20,
            ddof: 0,
        }],
        fallback_reason: None,
        estimated_state_bytes_per_entity: 0,
        fingerprint: "two-lane-fixture".into(),
    }
}

fn two_output_plan() -> RollingKernelPlan {
    let mut plan = lane_plan(true);
    plan.outputs = [5, 20]
        .into_iter()
        .enumerate()
        .map(|(group, min_periods)| TypedOutputPlan {
            group,
            kind: TypedOutputKind::Statistic(Statistic::Mean),
            storage: OutputStorage::Float64,
            min_periods,
            ddof: 0,
        })
        .collect();
    plan
}

fn input(rows: usize, entities: usize, special: bool) -> RecordBatch {
    let names = (0..entities)
        .map(|index| format!("e{index:04}"))
        .collect::<Vec<_>>();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("entity", DataType::Utf8, false),
            Field::new("price", DataType::Float64, true),
        ])),
        vec![
            Arc::new(TimestampMicrosecondArray::from_iter_values(
                (0..rows).map(|n| i64::try_from(n).unwrap()),
            )),
            Arc::new(UInt64Array::from_iter_values(
                (0..rows).map(|n| u64::try_from(n).unwrap()),
            )),
            Arc::new(StringArray::from_iter_values(
                (0..rows).map(|n| names[n % entities].as_str()),
            )),
            Arc::new(
                (0..rows)
                    .map(|n| {
                        if !special {
                            return Some(f64::from(u32::try_from(n % 101).unwrap()));
                        }
                        match (n / entities) % 31 {
                            0 => None,
                            1 => Some(f64::NAN),
                            2 => Some(f64::INFINITY),
                            3 => Some(f64::NEG_INFINITY),
                            4..=6 => Some(1e155),
                            7 => Some(-1e155),
                            _ => Some(f64::from(u32::try_from(n % 101).unwrap())),
                        }
                    })
                    .collect::<Float64Array>(),
            ),
        ],
    )
    .unwrap()
}

fn foreign_gap_input(rows: usize, special: bool) -> RecordBatch {
    let batch = input(rows, 16, special);
    let hot_rows = rows * 5 / 8;
    let cold_rows = (rows - hot_rows) / 15;
    let entities = (0..16)
        .chain(std::iter::repeat_n(0, hot_rows - 1))
        .chain((1..16).flat_map(|entity| std::iter::repeat_n(entity, cold_rows - 1)))
        .map(|entity| format!("e{entity:04}"))
        .collect::<Vec<_>>();
    assert_eq!(entities.len(), rows);
    let mut columns = batch.columns().to_vec();
    columns[2] = Arc::new(StringArray::from_iter_values(entities));
    RecordBatch::try_new(batch.schema(), columns).unwrap()
}

fn warm(plan: &RollingKernelPlan, batch: &RecordBatch) -> RollingKernelState {
    let mut state = RollingKernelState::default();
    let prior = batch.slice(0, 1_280);
    let columns = std::iter::once(
        Arc::new(TimestampMicrosecondArray::from_iter_values(-1_280..0)) as ArrayRef,
    )
    .chain(prior.columns().iter().skip(1).cloned())
    .collect();
    let prior = RecordBatch::try_new(prior.schema(), columns).unwrap();
    plan.prepare_ordered_stream(&state, &prior, "r", None)
        .unwrap()
        .commit(&mut state);
    state
}

fn exit(request: NumericLaneRequest, cancellation: &AtomicBool) -> NumericLaneExit {
    let mut progress = LaneProgress::default();
    let outcome = match std::panic::catch_unwind(AssertUnwindSafe(|| {
        run_numeric_lane(request, cancellation, &mut progress)
    })) {
        Ok(Ok(result)) => NumericLaneOutcome::Complete(result),
        Ok(Err(NumericLaneStop::Ordinary(error))) => NumericLaneOutcome::OrdinaryError(error),
        Ok(Err(NumericLaneStop::Cancelled)) => NumericLaneOutcome::Cancelled,
        Err(payload) => NumericLaneOutcome::Panicked(LocatedPanic {
            site: progress.active_site,
            payload,
        }),
    };
    NumericLaneExit {
        outcome,
        work: progress.work,
    }
}

fn pair(
    plan: &RollingKernelPlan,
    state: &RollingKernelState,
    input: &RecordBatch,
) -> (NumericJoinSeed, [NumericLaneRequest; 2]) {
    let prepared = plan
        .prepare_ordered_stream_inputs(state, input, "r", None)
        .unwrap();
    assert!(prepared.scratch_plan().unwrap().accounted_bytes().unwrap() <= ScratchPlan::CAP_BYTES);
    prepared
        .split_two()
        .unwrap_or_else(|_| panic!("balanced fixture must split"))
}

pub(crate) fn entity_parallel_test_pair() -> (NumericJoinSeed, [NumericLaneRequest; 2], ScratchPlan)
{
    let input = input(64_000, 64, false);
    let plan = lane_plan(true);
    let state = warm(&plan, &input);
    let prepared = plan
        .prepare_ordered_stream_inputs(&state, &input, "r", None)
        .unwrap();
    let scratch = prepared.scratch_plan().unwrap();
    let (seed, requests) = prepared
        .split_two()
        .unwrap_or_else(|_| panic!("balanced fixture"));
    (seed, requests, scratch)
}

#[test]
fn entity_parallel_preserves_rows_values_and_complete_touched_state() {
    for dual in [false, true] {
        for special in [false, true] {
            let plan = lane_plan(dual);
            let input = input(64_000, 64, special);
            let state = warm(&plan, &input);
            let prior = format!("{:?}", state.states);
            let expected = plan
                .prepare_ordered_stream(&state, &input, "r", None)
                .unwrap();
            let (seed, requests) = pair(&plan, &state, &input);
            let token = AtomicBool::new(false);
            let mut results = requests.map(|request| exit(request, &token));
            assert_eq!(
                results
                    .iter()
                    .map(|result| result.work.numeric_rows)
                    .sum::<u64>(),
                64_000
            );
            let actual = merge_numeric_results(seed, &mut results, "job").unwrap();
            assert_eq!(actual.entity_ids(), expected.entity_ids());
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
            assert_eq!(format!("{:?}", state.states), prior);
        }
    }
}

#[test]
fn entity_parallel_qualification_checks_frame_profile_columns_and_actual_queues() {
    let input = input(64_000, 64, false);
    let original = lane_plan(false);
    for change in [
        "stable_v2",
        "nan_as_value",
        "w21",
        "duration",
        "three_aliases",
        "wrong_state",
        "queue_w_plus_one",
    ] {
        let mut plan = original.clone();
        let mut state = warm(&plan, &input);
        match change {
            "stable_v2" => plan.numerical_profile = RollingNumericalProfile::StableV2Preview,
            "nan_as_value" => plan.nan_as_value = true,
            "w21" => {
                plan.groups[0] = TypedGroupPlan::Numeric {
                    input_index: 3,
                    frame: TypedFrame::Rows(21),
                }
            }
            "duration" => {
                plan.groups[0] = TypedGroupPlan::Numeric {
                    input_index: 3,
                    frame: TypedFrame::Duration(20),
                }
            }
            "three_aliases" => plan.outputs = vec![plan.outputs[0]; 3],
            "wrong_state" => {
                state.states[0].groups[0] = TypedWindowState::Exact(ExactNumericState::new(
                    TypedFrame::Rows(20),
                    SumClass::Signed,
                    20,
                ));
            }
            _ => match &mut state.states[0].groups[0] {
                TypedWindowState::Numeric(numeric) => numeric.samples.push(0, None),
                _ => unreachable!(),
            },
        }
        let prepared = plan
            .prepare_ordered_stream_inputs(&state, &input, "r", None)
            .unwrap();
        assert!(prepared.scratch_plan().is_none(), "{change}");
    }
}

#[test]
fn entity_parallel_new_entity_full_window_uses_local_validity_and_bounded_queues() {
    let input = input(64_000, 64, false);
    let plan = lane_plan(false);
    let state = RollingKernelState::default();
    let (seed, requests) = pair(&plan, &state, &input);
    let token = AtomicBool::new(false);
    let mut results = requests.map(|request| exit(request, &token));
    let result = merge_numeric_results(seed, &mut results, "job").unwrap();
    let column = &result.execution.columns[0];
    for row in 0..input.num_rows() {
        assert_eq!(column.is_null(row), row / 64 < 19);
    }
    for entity in result.execution.state.states {
        let TypedWindowState::Numeric(numeric) = &entity.groups[0] else {
            unreachable!()
        };
        assert_eq!(numeric.samples.len(), 20);
    }
}

#[test]
fn entity_parallel_small_sparse_and_skewed_inputs_keep_prepared_serial_state() {
    for (rows, entities) in [(1_280, 64), (64_000, 1), (64_000, 8)] {
        let input = input(rows, entities, false);
        let plan = lane_plan(false);
        let state = warm(&plan, &input);
        let prepared = plan
            .prepare_ordered_stream_inputs(&state, &input, "r", None)
            .unwrap();
        assert!(prepared.scratch_plan().is_none());
        let actual = prepared.finish_serial(None).unwrap();
        let expected = plan
            .prepare_ordered_stream(&state, &input, "r", None)
            .unwrap();
        assert_eq!(
            format!("{:?}", actual.execution.state.states),
            format!("{:?}", expected.execution.state.states)
        );
    }
}

#[test]
fn entity_parallel_cancelled_lane_is_separate_and_does_not_count_unattempted_rows() {
    let input = input(64_000, 64, false);
    let plan = lane_plan(true);
    let state = warm(&plan, &input);
    let (_, [request, _]) = pair(&plan, &state, &input);
    let token = AtomicBool::new(false);
    token.store(true, Ordering::SeqCst);
    let actual = exit(request, &token);
    assert!(matches!(actual.outcome, NumericLaneOutcome::Cancelled));
    assert_eq!(actual.work.numeric_rows, 0);
}

#[test]
fn entity_parallel_scratch_uses_checked_capacity_including_entity_containers() {
    assert!(
        ScratchPlan::for_dimensions(64_000, 64, 2, 1)
            .accounted_bytes()
            .unwrap()
            < ScratchPlan::CAP_BYTES
    );
    assert!(
        ScratchPlan::for_dimensions(256_000, 256_000, 2, 1)
            .accounted_bytes()
            .unwrap()
            > ScratchPlan::CAP_BYTES
    );
    assert_eq!(
        ScratchPlan::for_dimensions(usize::MAX, 64, 2, 1).accounted_bytes(),
        None
    );
}

#[test]
fn entity_parallel_two_output_columns_preserve_full_window_and_value_bits() {
    for special in [false, true] {
        let input = input(64_000, 64, special);
        let plan = two_output_plan();
        let state = RollingKernelState::default();
        let expected = plan
            .prepare_ordered_stream(&state, &input, "r", None)
            .unwrap();
        let (seed, requests) = pair(&plan, &state, &input);
        let mut exits = requests.map(|request| exit(request, &AtomicBool::new(false)));
        let actual = merge_numeric_results(seed, &mut exits, "job").unwrap();
        assert_eq!(actual.execution.columns.len(), 2);
        for ((actual, expected), window) in actual
            .execution
            .columns
            .iter()
            .zip(&expected.execution.columns)
            .zip([5, 20])
        {
            let actual = actual.as_any().downcast_ref::<Float64Array>().unwrap();
            let expected = expected.as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(actual.nulls(), expected.nulls());
            for row in 0..input.num_rows() {
                if !special {
                    assert_eq!(actual.is_null(row), row / 64 < window - 1);
                }
                assert_eq!(actual.value(row).to_bits(), expected.value(row).to_bits());
            }
        }
        assert_eq!(
            format!("{:?}", actual.execution.state.states),
            format!("{:?}", expected.execution.state.states)
        );
        assert!(state.states.is_empty());
    }
}

fn allocation_lane_columns(
    plan: &RollingKernelPlan,
    rows: usize,
    count: usize,
    late_null: bool,
) -> Vec<Float64Array> {
    let mut builders = plan.outputs[..count]
        .iter()
        .map(|output| DerivedBuilder::new(*output, rows))
        .collect::<Vec<_>>();
    for builder in &mut builders {
        let DerivedBuilder::Float(values, OutputStorage::Float64) = builder else {
            unreachable!()
        };
        assert_eq!(values.capacity(), rows);
        let null_row = if late_null { rows - 1 } else { 0 };
        for row in 0..rows {
            if row == null_row {
                values.append_null();
            } else {
                values.append_value(42.0);
            }
        }
        assert_eq!(values.capacity(), rows);
    }
    let columns = builders
        .into_iter()
        .map(|builder| match builder {
            DerivedBuilder::Float(mut values, OutputStorage::Float64) => values.finish(),
            _ => unreachable!("qualified lane output is Float64"),
        })
        .collect::<Vec<_>>();
    assert_eq!(columns.len(), count);
    columns
}

fn allocation_lane_buffer_bytes(columns: &[Float64Array], rows: usize, late_null: bool) -> usize {
    let mut bytes = 0;
    for column in columns {
        let bitmap = rows.div_ceil(8).next_multiple_of(64);
        assert_eq!(column.get_buffer_memory_size(), rows * 8 + bitmap);
        assert_eq!(column.len(), rows);
        assert_eq!(column.null_count(), 1);
        assert!(column.is_null(if late_null { rows - 1 } else { 0 }));
        bytes += column.get_buffer_memory_size();
    }
    bytes
}

fn check_lane_builder_allocation(loads: [usize; 2], count: usize, late_null: bool) {
    let plan = two_output_plan();
    // Builders and typed Vecs remain in the current-thread scope through drop.
    let mut typed_array_capacity_bytes = 0;
    let mut retained_buffer_bytes = 0;
    let observed = allocation_counter::measure(|| {
        let columns = loads.map(|rows| {
            let columns = allocation_lane_columns(&plan, rows, count, late_null);
            typed_array_capacity_bytes += columns.capacity() * size_of::<Float64Array>();
            columns
        });
        for (columns, rows) in columns.iter().zip(loads) {
            retained_buffer_bytes += allocation_lane_buffer_bytes(columns, rows, late_null);
        }
    });
    let rows = loads.into_iter().sum::<usize>();
    let allowance = ScratchPlan::for_dimensions(rows, 0, count, 0)
        .accounted_bytes()
        .unwrap()
        - rows * size_of::<LaneRow>();
    let retained_bytes = typed_array_capacity_bytes + retained_buffer_bytes;
    assert!(retained_bytes <= allowance);
    assert!(observed.bytes_max >= u64::try_from(retained_bytes).unwrap());
    assert!(observed.bytes_max <= u64::try_from(allowance).unwrap());
    assert_eq!(observed.bytes_current, 0);
    assert_eq!(observed.count_current, 0);
    eprintln!(
        "lane builders loads={loads:?} outputs={count} late_null={late_null}: \
         current_thread_peak={} accounted_output_allowance={allowance} \
         typed_array_capacity_bytes={typed_array_capacity_bytes} \
         retained_buffer_bytes={retained_buffer_bytes}",
        observed.bytes_max
    );
}

#[test]
fn entity_parallel_lane_builder_allocation_fits_the_declared_capacity_allowance() {
    for loads in [[24_000, 40_000], [32_000, 32_000], [96_000, 160_000]] {
        for count in [1, 2] {
            for late_null in [false, true] {
                check_lane_builder_allocation(loads, count, late_null);
            }
        }
    }
}

#[test]
fn entity_parallel_split_preserves_reserved_container_capacities() {
    for (rows, entities) in [(64_000, 16), (64_000, 64), (256_000, 64)] {
        let input = input(rows, entities, false);
        let plan = two_output_plan();
        let state = RollingKernelState::default();
        let prepared = plan
            .prepare_ordered_stream_inputs(&state, &input, "r", None)
            .unwrap();
        let accounted = prepared.scratch_plan().unwrap().accounted_bytes().unwrap();
        let pointer = prepared.fill.stream.state.states.as_ptr();
        let capacity = prepared.fill.stream.state.states.capacity();
        let (seed, requests) = prepared.split_two().ok().unwrap();
        assert_eq!(seed.execution.state.states.as_ptr(), pointer);
        assert_eq!(seed.execution.state.states.capacity(), capacity);
        let mut container_bytes = 0;
        for request in &requests {
            assert!(request.rows.is_empty());
            assert_eq!(request.rows.capacity(), rows / 2);
            assert_eq!(request.entities.capacity(), request.entities.len());
            container_bytes += request.rows.capacity() * size_of::<LaneRow>()
                + request.entities.capacity() * size_of::<OwnedLaneEntity>();
        }
        let values_and_validity = 2 * (rows * 8 + rows.div_ceil(8) + 128);
        assert!(container_bytes + values_and_validity < accounted);
        assert!(accounted <= ScratchPlan::CAP_BYTES);
    }
}

#[test]
fn entity_parallel_group_cancel_keeps_attempts_and_stops_before_next_group_or_readout() {
    use std::sync::atomic::AtomicUsize;

    let input = input(64_000, 64, true);
    let plan = lane_plan(true);
    let state = warm(&plan, &input);
    for attempted in [255, 256, 257] {
        let (_, [mut request, _]) = pair(&plan, &state, &input);
        let target = u32::try_from(2 * (attempted - 1)).unwrap();
        let token = Arc::new(AtomicBool::new(false));
        let cancel = Arc::clone(&token);
        let after = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&after);
        request.test_hook = Some(Arc::new(move |point| match point {
            NumericTestPoint::BeforeGroup(site) if site.row == target && site.ordinal == 0 => {
                cancel.store(true, Ordering::SeqCst);
            }
            NumericTestPoint::AfterGroup(site) if site.row == target => {
                observed.fetch_add(1, Ordering::Relaxed);
            }
            NumericTestPoint::BeforeGroup(site) | NumericTestPoint::BeforeOutput(site)
                if site.row == target =>
            {
                panic!("cancelled update must not start next group/readout")
            }
            _ => {}
        }));
        let actual = exit(request, &token);
        assert!(matches!(actual.outcome, NumericLaneOutcome::Cancelled));
        assert_eq!(actual.work.numeric_rows, u64::try_from(attempted).unwrap());
        assert_eq!(after.load(Ordering::Relaxed), 1);
    }
}

#[test]
fn entity_parallel_transition_error_counts_only_prior_attempts() {
    let input = input(64_000, 64, false);
    let plan = lane_plan(false);
    let state = warm(&plan, &input);
    let (_, [mut request, _]) = pair(&plan, &state, &input);
    let first = LaneRow {
        original_row: 0,
        lane_entity: 0,
    };
    request.entities[usize::try_from(first.lane_entity).unwrap()]
        .state
        .transition_count = u64::MAX;
    let actual = exit(request, &AtomicBool::new(false));
    let NumericLaneOutcome::OrdinaryError(error) = actual.outcome else {
        panic!("ordinary overflow required")
    };
    assert_eq!(
        error.site,
        NumericSite {
            row: first.original_row,
            phase: NumericPhase::Transition,
            ordinal: 0
        }
    );
    assert_eq!(actual.work.numeric_rows, 0);
    assert!(
        error
            .error
            .to_string()
            .contains("transition count overflowed")
    );
}

#[test]
fn entity_parallel_group_error_is_not_erased_by_the_post_cancel_check() {
    let input = input(64_000, 64, false);
    let plan = lane_plan(false);
    let state = warm(&plan, &input);
    let (_, [mut request, _]) = pair(&plan, &state, &input);
    let TypedWindowState::Numeric(numeric) = &mut request.entities[0].state.groups[0] else {
        unreachable!()
    };
    numeric.accumulator.valid_count = u64::MAX;
    let token = Arc::new(AtomicBool::new(false));
    let cancel = Arc::clone(&token);
    request.test_hook = Some(Arc::new(move |point| {
        if matches!(point, NumericTestPoint::BeforeGroup(_)) {
            cancel.store(true, Ordering::SeqCst);
        }
    }));
    let actual = exit(request, &token);
    let NumericLaneOutcome::OrdinaryError(error) = actual.outcome else {
        panic!("update error precedes post-check")
    };
    assert_eq!(error.site.phase, NumericPhase::Group);
    assert_eq!(actual.work.numeric_rows, 1);
}

#[test]
fn entity_parallel_earlier_panic_resumes_its_original_payload_after_both_lanes() {
    let input = input(64_000, 64, false);
    let plan = lane_plan(false);
    let state = warm(&plan, &input);
    let (seed, [mut first, mut second]) = pair(&plan, &state, &input);
    first.test_hook = Some(Arc::new(|point| {
        if matches!(point, NumericTestPoint::BeforeGroup(_)) {
            std::panic::panic_any(417_u32);
        }
    }));
    second.entities[0].state.transition_count = u64::MAX;
    let token = AtomicBool::new(false);
    let mut results = [exit(first, &token), exit(second, &token)];
    let panic = std::panic::catch_unwind(AssertUnwindSafe(|| {
        merge_numeric_results(seed, &mut results, "job")
    }))
    .err()
    .unwrap();
    assert_eq!(*panic.downcast::<u32>().unwrap(), 417);
    assert_eq!(
        results
            .iter()
            .map(|result| result.work.numeric_rows)
            .sum::<u64>(),
        1
    );
}

#[test]
fn entity_parallel_hot_entity_rejection_reuses_the_same_prepared_candidate() {
    let mut input = input(64_000, 64, false);
    let mut columns = input.columns().to_vec();
    columns[2] = Arc::new(StringArray::from_iter_values((0..64_000).map(|n| {
        if n < 50_000 {
            "hot".to_owned()
        } else {
            format!("e{:04}", n % 63)
        }
    })));
    input = RecordBatch::try_new(input.schema(), columns).unwrap();
    let plan = lane_plan(false);
    let state = warm(&plan, &input);
    let prepared = plan
        .prepare_ordered_stream_inputs(&state, &input, "r", None)
        .unwrap();
    assert!(prepared.scratch_plan().is_some());
    let before = prepared.fill.stream.state.states.as_ptr();
    let rejected = prepared.split_two().err().expect("skew must be serial");
    assert_eq!(rejected.fill.stream.state.states.as_ptr(), before);
    let actual = rejected.finish_serial(None).unwrap();
    assert_eq!(actual.execution.metrics.input_rows, 64_000);
}

#[test]
fn entity_parallel_original_error_order_beats_later_speculative_panic() {
    let input = input(64_000, 64, false);
    let plan = lane_plan(false);
    let state = warm(&plan, &input);
    let (seed, [mut first, mut second]) = pair(&plan, &state, &input);
    first.entities[0].state.transition_count = u64::MAX;
    second.test_hook = Some(Arc::new(|point| {
        if matches!(point, NumericTestPoint::BeforeGroup(_)) {
            panic!("later lane panic");
        }
    }));
    let token = AtomicBool::new(false);
    // Complete the later lane first; merge is ordered by original input sites.
    let later = exit(second, &token);
    let earlier = exit(first, &token);
    let mut results = [earlier, later];
    let error = merge_numeric_results(seed, &mut results, "job")
        .err()
        .unwrap();
    assert!(error.to_string().contains("transition count overflowed"));
    assert_eq!(
        results
            .iter()
            .map(|result| result.work.numeric_rows)
            .sum::<u64>(),
        1
    );
}

#[test]
fn entity_parallel_finish_panic_has_no_stale_row_site() {
    let input = input(64_000, 64, false);
    let plan = lane_plan(false);
    let state = warm(&plan, &input);
    let (_, [mut request, _]) = pair(&plan, &state, &input);
    request.test_hook = Some(Arc::new(|point| {
        if matches!(point, NumericTestPoint::Finish) {
            panic!("finish panic");
        }
    }));
    let actual = exit(request, &AtomicBool::new(false));
    let NumericLaneOutcome::Panicked(panic) = actual.outcome else {
        panic!("panic required")
    };
    assert_eq!(panic.site, None);
    assert_eq!(actual.work.numeric_rows, 32_000);
}

fn assert_route_index_panic_has_no_site(
    mut request: NumericLaneRequest,
    retained_entities: usize,
    completed_rows: u64,
) {
    request.entities.truncate(retained_entities);
    let actual = exit(request, &AtomicBool::new(false));
    let NumericLaneOutcome::Panicked(panic) = actual.outcome else {
        panic!("the missing routed entity must cause an index panic");
    };
    let message = panic.payload.downcast::<String>().unwrap();
    assert_eq!(
        *message,
        format!(
            "index out of bounds: the len is {retained_entities} but the index is {retained_entities}"
        )
    );
    assert_eq!(panic.site, None);
    assert_eq!(actual.work.numeric_rows, completed_rows);
    assert!(!actual.work.overflowed);
}

#[test]
fn entity_parallel_route_panic_on_first_owned_row_has_no_site() {
    let batch = input(64_000, 64, false);
    for plan in [lane_plan(false), lane_plan(true), two_output_plan()] {
        let state = warm(&plan, &batch);
        let (_, [request, _]) = pair(&plan, &state, &batch);
        assert_route_index_panic_has_no_site(request, 0, 0);
    }
}

#[test]
fn entity_parallel_route_panic_after_completed_row_has_no_site() {
    let batch = input(64_000, 64, false);
    for plan in [lane_plan(false), lane_plan(true), two_output_plan()] {
        let state = warm(&plan, &batch);
        let (_, [request, _]) = pair(&plan, &state, &batch);
        // Attempt one is not a periodic cancellation check that could clear a stale site.
        assert_route_index_panic_has_no_site(request, 1, 1);
    }
}

#[test]
fn entity_parallel_route_panic_after_long_foreign_span_has_no_site() {
    let rows = 64_000;
    let prior = input(rows, 16, false);
    let hot_rows = rows * 5 / 8;
    let cold_rows = (rows - hot_rows) / 15;
    let keys = (0..2)
        .chain(std::iter::repeat_n(0, hot_rows - 1))
        .chain(2..16)
        .chain((1..16).flat_map(|entity| std::iter::repeat_n(entity, cold_rows - 1)))
        .map(|entity| format!("e{entity:04}"))
        .collect::<Vec<_>>();
    let mut columns = prior.columns().to_vec();
    columns[2] = Arc::new(StringArray::from_iter_values(keys));
    let batch = RecordBatch::try_new(prior.schema(), columns).unwrap();
    for plan in [lane_plan(false), lane_plan(true), two_output_plan()] {
        let state = warm(&plan, &prior);
        let (_, [_, request]) = pair(&plan, &state, &batch);
        // Lane one completes row one, skips 39,999 hot rows, then fails on attempt one.
        assert_route_index_panic_has_no_site(request, 1, 1);
    }
}

#[test]
fn entity_parallel_route_panic_without_site_precedes_an_earlier_row_error() {
    let batch = input(64_000, 64, false);
    let plan = lane_plan(false);
    let state = warm(&plan, &batch);
    let (seed, [mut first, mut second]) = pair(&plan, &state, &batch);
    first.entities[0].state.transition_count = u64::MAX;
    second.entities.truncate(1);
    let token = AtomicBool::new(false);
    let mut results = [exit(first, &token), exit(second, &token)];
    assert!(matches!(
        results[0].outcome,
        NumericLaneOutcome::OrdinaryError(_)
    ));
    assert_eq!(
        results.each_ref().map(|result| result.work.numeric_rows),
        [0, 1]
    );
    let NumericLaneOutcome::Panicked(original) = &results[1].outcome else {
        panic!("lane one must panic at its missing entity");
    };
    let original_payload = std::ptr::from_ref(original.payload.downcast_ref::<String>().unwrap());
    let resumed = std::panic::catch_unwind(AssertUnwindSafe(|| {
        merge_numeric_results(seed, &mut results, "job")
    }))
    .err()
    .expect("an unlocated route panic must precede the row-zero ordinary error");
    let message = resumed.downcast::<String>().unwrap();
    assert_eq!(std::ptr::from_ref(message.as_ref()), original_payload);
    assert_eq!(
        *message,
        "index out of bounds: the len is 1 but the index is 1"
    );
}

#[test]
fn entity_parallel_merge_checks_global_row_coverage_before_returning_state() {
    let input = input(64_000, 64, false);
    let plan = lane_plan(false);
    let state = warm(&plan, &input);
    let (seed, requests) = pair(&plan, &state, &input);
    let token = AtomicBool::new(false);
    let mut results = requests.map(|request| exit(request, &token));
    let NumericLaneOutcome::Complete(second) = &mut results[1].outcome else {
        unreachable!()
    };
    second.rows[0].original_row = 0;
    assert!(
        merge_numeric_results(seed, &mut results, "job")
            .err()
            .unwrap()
            .to_string()
            .contains("row coverage")
    );
}

#[test]
fn entity_parallel_lazy_rows_reuse_reserved_capacities_and_original_ids() {
    for rows in [64_000, 256_000] {
        let input = foreign_gap_input(rows, false);
        let plan = two_output_plan();
        let state = warm(&plan, &input);
        let prepared = plan
            .prepare_ordered_stream_inputs(&state, &input, "r", None)
            .unwrap();
        let ids_pointer = prepared.fill.stream.entity_ids.as_ptr();
        let ids_capacity = prepared.fill.stream.entity_ids.capacity();
        let (seed, requests) = prepared.split_two().ok().unwrap();
        let loads = [rows * 5 / 8, rows * 3 / 8];
        let reserved = requests.each_ref().map(|request| {
            assert!(
                request.rows.is_empty(),
                "split must defer schedule filling to lanes"
            );
            (request.rows.as_ptr(), request.rows.capacity())
        });
        let token = AtomicBool::new(false);
        let mut results = requests.map(|request| exit(request, &token));
        for (lane, result) in results.iter().enumerate() {
            let NumericLaneOutcome::Complete(result) = &result.outcome else {
                panic!("qualified lane must complete");
            };
            assert_eq!(reserved[lane].1, loads[lane]);
            assert_eq!(result.rows.len(), loads[lane]);
            assert_eq!(result.rows.capacity(), reserved[lane].1);
            assert_eq!(result.rows.as_ptr(), reserved[lane].0);
        }
        let update = merge_numeric_results(seed, &mut results, "job").unwrap();
        assert_eq!(update.execution.entity_ids.as_ptr(), ids_pointer);
        assert_eq!(update.execution.entity_ids.capacity(), ids_capacity);
        assert_eq!(update.execution.entity_ids.len(), rows);
        assert!(
            update
                .execution
                .columns
                .iter()
                .all(|column| column.len() == rows)
        );
    }
}

#[test]
fn entity_parallel_lazy_scan_long_foreign_spans_preserve_values_and_state() {
    for dual in [false, true] {
        let input = foreign_gap_input(64_000, true);
        let plan = lane_plan(dual);
        let state = warm(&plan, &input);
        let expected = plan
            .prepare_ordered_stream(&state, &input, "r", None)
            .unwrap();
        let (seed, requests) = pair(&plan, &state, &input);
        let mut results = requests.map(|request| exit(request, &AtomicBool::new(false)));
        assert_eq!(
            results
                .iter()
                .map(|result| result.work.numeric_rows)
                .sum::<u64>(),
            64_000
        );
        let actual = merge_numeric_results(seed, &mut results, "job").unwrap();
        assert_eq!(actual.entity_ids(), expected.entity_ids());
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
    }
}

#[test]
fn entity_parallel_lazy_foreign_gap_keeps_before_output_cancel_attempt_order() {
    use std::sync::atomic::AtomicUsize;

    for rows in [64_000, 256_000] {
        let input = foreign_gap_input(rows, false);
        let plan = lane_plan(false);
        let state = warm(&plan, &input);
        let (_, [_, mut request]) = pair(&plan, &state, &input);
        let token = Arc::new(AtomicBool::new(false));
        let cancel = Arc::clone(&token);
        let groups = Arc::new(AtomicUsize::new(0));
        let seen = Arc::clone(&groups);
        request.test_hook = Some(Arc::new(move |point| match point {
            NumericTestPoint::BeforeOutput(site) if site.row == 15 => {
                cancel.store(true, Ordering::SeqCst);
            }
            NumericTestPoint::BeforeGroup(site) => {
                assert!(site.row <= 15, "next transition must stop before its group");
                seen.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }));
        let mut progress = LaneProgress::default();
        let outcome = run_numeric_lane(request, &token, &mut progress);
        assert!(matches!(outcome, Err(NumericLaneStop::Cancelled)));
        assert_eq!(groups.load(Ordering::Relaxed), 15);
        assert_eq!(progress.work.numeric_rows, 16);
        assert_eq!(progress.active_site, None);
    }
}

#[test]
fn entity_parallel_lazy_merge_rejects_results_from_different_inputs() {
    let input = input(64_000, 64, false);
    let plan = lane_plan(false);
    let state = warm(&plan, &input);
    let (seed, [first, _]) = pair(&plan, &state, &input);
    let (_, [_, second]) = pair(&plan, &state, &input);
    let token = AtomicBool::new(false);
    let mut results = [exit(first, &token), exit(second, &token)];
    let error = merge_numeric_results(seed, &mut results, "job")
        .err()
        .expect("mixed private inputs must be rejected before ownership recovery");
    assert!(error.to_string().contains("numeric lane inputs differ"));
}

#[test]
fn entity_parallel_null_free_final_values_move_without_reallocation() {
    fn check<const COLUMNS: usize>(rows: usize) {
        let bits = [
            0.0_f64.to_bits(),
            (-0.0_f64).to_bits(),
            f64::INFINITY.to_bits(),
            f64::NEG_INFINITY.to_bits(),
            0x7ff8_0000_0000_0123,
            0x7ff8_0000_0000_0456,
        ];
        let mut same_allocations = false;
        let mut same_bits = false;
        let observed = allocation_counter::measure(|| {
            let values = std::array::from_fn::<_, COLUMNS, _>(|column| {
                let mut values = Vec::with_capacity(rows + 16);
                values
                    .extend((0..rows).map(|row| f64::from_bits(bits[(row + column) % bits.len()])));
                values
            });
            let allocations = values
                .each_ref()
                .map(|values| (values.as_ptr(), values.capacity()));
            let columns = finish_null_free_numeric_columns(values);
            same_allocations = columns.len() == COLUMNS
                && columns
                    .iter()
                    .zip(allocations)
                    .all(|(column, (pointer, capacity))| {
                        let column = column.as_any().downcast_ref::<Float64Array>().unwrap();
                        column.values().as_ptr() == pointer
                            && column.values().inner().capacity() == capacity * size_of::<f64>()
                            && column.len() == rows
                            && column.nulls().is_none()
                    });
            same_bits = columns.iter().enumerate().all(|(ordinal, column)| {
                let column = column.as_any().downcast_ref::<Float64Array>().unwrap();
                column
                    .values()
                    .iter()
                    .enumerate()
                    .all(|(row, value)| value.to_bits() == bits[(row + ordinal) % bits.len()])
            });
        });
        assert!(
            same_allocations,
            "Arrow must own the original spare-capacity Vec"
        );
        assert!(
            same_bits,
            "valid floating-point bit patterns must be copied unchanged"
        );
        assert_eq!(observed.count_current, 0);
        assert_eq!(observed.bytes_current, 0);
        assert!(
            observed.bytes_max <= ((rows + 16) * size_of::<f64>() + 4096) as u64 * COLUMNS as u64
        );
        println!("final Vec rows={rows} columns={COLUMNS}: {observed:?}");
    }
    for rows in [64_000, 256_000] {
        check::<1>(rows);
        check::<2>(rows);
    }
}

#[test]
fn entity_parallel_null_free_merge_allocates_only_final_buffers_and_owners() {
    let mut counts = [0; 2];
    for count in [1, 2] {
        let input = input(64_000, 64, false);
        let plan = if count == 1 {
            lane_plan(false)
        } else {
            two_output_plan()
        };
        let state = warm(&plan, &input);
        let (seed, requests) = pair(&plan, &state, &input);
        let mut exits = requests.map(|request| exit(request, &AtomicBool::new(false)));
        let mut valid_output = false;
        // Inputs predate this scope; only count newly requested merge allocations.
        let observed = allocation_counter::measure(|| {
            let update = merge_numeric_results(seed, &mut exits, "job").unwrap();
            valid_output = update.execution.columns.len() == count
                && update
                    .execution
                    .columns
                    .iter()
                    .all(|column| column.len() == 64_000 && column.null_count() == 0);
        });
        assert!(valid_output);
        counts[count - 1] = observed.count_total;
        println!(
            "merge columns={count}: new_allocations={} new_requested_bytes={}",
            observed.count_total, observed.bytes_total
        );
    }
    // Per column: final Vec, Buffer owner and ArrayRef owner; one output Vec.
    assert!(
        counts[0] <= 4 && counts[1] <= 7,
        "unexpected merge allocations: {counts:?}"
    );
}

fn merge_value_bits(row: usize, column: usize) -> u64 {
    [
        0,
        1 << 63,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0123,
        0x7ff8_0000_0000_0456,
    ][(row + column) % 6]
}

fn merge_slot_is_null(case: usize, row: usize, column: usize) -> bool {
    match case {
        2 => row == 1 || row == 63_999,
        3 => column == 1 && (row == 0 || row == 63_999),
        4 => row.is_multiple_of(if column == 0 { 3 } else { 5 }),
        _ => false,
    }
}

fn complete_result(exit: &mut NumericLaneExit) -> &mut NumericLaneResult {
    let NumericLaneOutcome::Complete(result) = &mut exit.outcome else {
        panic!("fixture lane must complete");
    };
    result
}

fn merge_fixture(count: usize, null_case: usize) -> (NumericJoinSeed, [NumericLaneExit; 2]) {
    use datafusion::arrow::buffer::NullBuffer;

    let input = input(64_000, 64, false);
    let plan = if count == 2 {
        two_output_plan()
    } else {
        lane_plan(false)
    };
    let state = warm(&plan, &input);
    let (mut seed, requests) = pair(&plan, &state, &input);
    seed.set_output_count_for_test(count);
    let mut exits = requests.map(|request| exit(request, &AtomicBool::new(false)));
    for (lane, exit) in exits.iter_mut().enumerate() {
        let result = complete_result(exit);
        let prefix = 3 + lane * 2;
        let columns = (0..count)
            .map(|column| {
                let values = std::iter::repeat_n(-71.0, prefix)
                    .chain(result.rows.iter().map(|row| {
                        if merge_slot_is_null(null_case, row.original_row as usize, column) {
                            -91.0
                        } else {
                            f64::from_bits(merge_value_bits(row.original_row as usize, column))
                        }
                    }))
                    .chain([83.0])
                    .collect::<Vec<_>>();
                let validity = (null_case != 0).then(|| {
                    NullBuffer::from(
                        std::iter::repeat_n(true, prefix)
                            .chain(result.rows.iter().map(|row| {
                                !merge_slot_is_null(null_case, row.original_row as usize, column)
                            }))
                            .chain([true])
                            .collect::<Vec<_>>(),
                    )
                });
                Float64Array::new(values.into(), validity).slice(prefix, result.rows.len())
            })
            .collect();
        *result.columns_for_test() = columns;
    }
    (seed, exits)
}

#[test]
fn entity_parallel_merge_compatibility_preserves_slices_bits_and_nullable_slots() {
    for count in [0, 1, 2, 3] {
        for null_case in 0..=4 {
            let (seed, mut exits) = merge_fixture(count, null_case);
            let input = input(64_000, 64, false);
            let plan = if count == 2 {
                two_output_plan()
            } else {
                lane_plan(false)
            };
            let expected = plan
                .prepare_ordered_stream(&warm(&plan, &input), &input, "r", None)
                .unwrap();
            let actual = merge_numeric_results(seed, &mut exits, "job").unwrap();
            assert_eq!(actual.execution.columns.len(), count);
            assert_eq!(actual.entity_ids(), expected.entity_ids());
            assert_eq!(
                actual.execution.state.entities,
                expected.execution.state.entities
            );
            assert_eq!(
                actual.execution.state.kernel_fingerprint,
                expected.execution.state.kernel_fingerprint
            );
            assert_eq!(
                actual.execution.state.last_identity,
                expected.execution.state.last_identity
            );
            assert_eq!(
                format!("{:?}", actual.execution.state.states),
                format!("{:?}", expected.execution.state.states)
            );
            for (column, array) in actual.execution.columns.iter().enumerate() {
                let array = array.as_any().downcast_ref::<Float64Array>().unwrap();
                assert_eq!(array.data_type(), &DataType::Float64);
                assert_eq!(array.len(), 64_000);
                for row in 0..64_000 {
                    let is_null = merge_slot_is_null(null_case, row, column);
                    assert_eq!(
                        array.is_null(row),
                        is_null,
                        "count={count} case={null_case} row={row} column={column}"
                    );
                    assert_eq!(
                        array.value(row).to_bits(),
                        if is_null {
                            0
                        } else {
                            merge_value_bits(row, column)
                        }
                    );
                }
            }
        }
    }
}

fn corrupt_merge(
    seed: &mut NumericJoinSeed,
    exits: &mut [NumericLaneExit; 2],
    case: usize,
) -> &'static str {
    match case {
        0 => {
            complete_result(&mut exits[0]).columns_for_test().clear();
            complete_result(&mut exits[0]).rows[0].original_row = 9;
            seed.execution.state.entities.insert(vec![255], 64);
            "numeric lane output shape changed"
        }
        1 => {
            let result = complete_result(&mut exits[0]);
            let column = &mut result.columns_for_test()[0];
            *column = column.slice(1, column.len() - 1);
            "numeric lane output shape changed"
        }
        2 => {
            complete_result(&mut exits[1]).rows[0].original_row = 0;
            "numeric lane row coverage or entity identity changed"
        }
        3 => {
            complete_result(&mut exits[0]).rows.swap(0, 1);
            "numeric lane row coverage or entity identity changed"
        }
        4 => {
            let result = complete_result(&mut exits[1]);
            result.rows.pop();
            for column in result.columns_for_test() {
                *column = column.slice(0, column.len() - 1);
            }
            "numeric lane row coverage is incomplete"
        }
        5 => {
            complete_result(&mut exits[0]).rows[0].lane_entity = u32::MAX;
            "numeric lane row coverage or entity identity changed"
        }
        6 => {
            complete_result(&mut exits[0]).rows[0].lane_entity = 1;
            "numeric lane row coverage or entity identity changed"
        }
        7 => {
            let result = complete_result(&mut exits[0]);
            result.rows.push(LaneRow {
                original_row: 64_000,
                lane_entity: 0,
            });
            for column in result.columns_for_test() {
                *column = column.iter().chain([Some(1.0)]).collect();
            }
            seed.execution.state.entities.insert(vec![255], 64);
            "numeric lane row coverage has extra rows"
        }
        8 => {
            let result = complete_result(&mut exits[0]);
            result.entities.push(OwnedLaneEntity {
                local_id: result.entities[0].local_id,
                state: result.entities[0].state.clone(),
            });
            "numeric lane entity slots changed"
        }
        9 => {
            seed.execution.state.entities.insert(vec![255], 64);
            "numeric lane entity slots are incomplete"
        }
        10 => {
            let result = complete_result(&mut exits[0]);
            result.rows.remove(10);
            for column in result.columns_for_test() {
                *column = column
                    .iter()
                    .enumerate()
                    .filter_map(|(row, value)| (row != 10).then_some(value))
                    .collect();
            }
            "numeric lane row coverage or entity identity changed"
        }
        _ => unreachable!(),
    }
}

#[test]
fn entity_parallel_merge_compatibility_keeps_shape_coverage_and_state_error_order() {
    for null_case in [0, 4] {
        for case in 0..=10 {
            let (mut seed, mut exits) = merge_fixture(2, null_case);
            let expected = corrupt_merge(&mut seed, &mut exits, case);
            let error = merge_numeric_results(seed, &mut exits, "job")
                .err()
                .unwrap();
            let CalcFlowError::Internal { message } = error else {
                panic!("internal shape/coverage/state error required");
            };
            assert_eq!(message, expected, "null_case={null_case} corruption={case}");
        }
    }
}

#[test]
fn entity_parallel_merge_compatibility_selects_lane_failures_before_malformed_output() {
    for failure in 0..4 {
        let (seed, mut exits) = merge_fixture(1, 0);
        complete_result(&mut exits[0]).columns_for_test().clear();
        let site = NumericSite {
            row: 1,
            phase: NumericPhase::Output,
            ordinal: 0,
        };
        exits[1].outcome = match failure {
            0 => NumericLaneOutcome::OrdinaryError(TaggedNumericError {
                site,
                error: internal_error("original lane error"),
            }),
            1 => NumericLaneOutcome::Cancelled,
            _ => NumericLaneOutcome::Panicked(LocatedPanic {
                site: (failure == 2).then_some(site),
                payload: Box::new(419_u32),
            }),
        };
        let actual = std::panic::catch_unwind(AssertUnwindSafe(|| {
            merge_numeric_results(seed, &mut exits, "job")
        }));
        match failure {
            0 => assert!(
                matches!(actual.unwrap(), Err(CalcFlowError::Internal { message }) if message == "original lane error")
            ),
            1 => assert!(
                matches!(actual.unwrap(), Err(CalcFlowError::Cancelled { run_id }) if run_id == "job")
            ),
            _ => assert_eq!(*actual.err().unwrap().downcast::<u32>().unwrap(), 419),
        }
    }
}

#[test]
fn entity_parallel_merge_compatibility_drops_owned_allocations_on_error_and_unwind() {
    for null_case in [0, 4] {
        for failure in [2, 8, 9, 11] {
            let mut correct_failure = false;
            // Catch inside measure: the allocator probe itself is not unwind-safe.
            let observed = allocation_counter::measure(|| {
                let (mut seed, mut exits) = merge_fixture(2, null_case);
                let expected = if failure == 11 {
                    seed.panic_before_output_finish_for_test();
                    ""
                } else {
                    corrupt_merge(&mut seed, &mut exits, failure)
                };
                let actual = std::panic::catch_unwind(AssertUnwindSafe(|| {
                    merge_numeric_results(seed, &mut exits, "job")
                }));
                correct_failure = if failure == 11 {
                    actual.err().is_some_and(|payload| {
                        payload.downcast::<u32>().is_ok_and(|value| *value == 673)
                    })
                } else {
                    matches!(actual, Ok(Err(CalcFlowError::Internal { message })) if message == expected)
                };
            });
            assert!(correct_failure, "null_case={null_case} failure={failure}");
            assert_eq!(
                observed.count_current, 0,
                "null_case={null_case} failure={failure}"
            );
            assert_eq!(
                observed.bytes_current, 0,
                "null_case={null_case} failure={failure}"
            );
        }
    }
}

#[test]
fn entity_parallel_null_free_gather_reuses_reserved_values_through_arrow_finish() {
    fn check<const COLUMNS: usize>(rows: usize) {
        let input = input(rows, 64, false);
        let plan = if COLUMNS == 1 {
            lane_plan(false)
        } else {
            two_output_plan()
        };
        let state = warm(&plan, &input);
        let (seed, requests) = pair(&plan, &state, &input);
        let results = requests.map(|request| {
            let NumericLaneOutcome::Complete(result) =
                exit(request, &AtomicBool::new(false)).outcome
            else {
                panic!("fixture must complete");
            };
            result
        });
        let ids = (0..rows).map(|row| row % 64).collect::<Vec<_>>();
        let mut same_allocations = false;
        let observed = allocation_counter::measure(|| {
            let values = std::array::from_fn::<_, COLUMNS, _>(|_| Vec::with_capacity(rows));
            let reserved = values
                .each_ref()
                .map(|values| (values.as_ptr(), values.capacity()));
            let values = gather_null_free_numeric_columns(&ids, &results, values).unwrap();
            same_allocations = values
                .iter()
                .zip(reserved)
                .all(|(values, (pointer, capacity))| {
                    values.as_ptr() == pointer
                        && values.capacity() == capacity
                        && capacity == rows
                        && values.len() == rows
                });
            let columns = finish_null_free_numeric_columns(values);
            same_allocations &=
                columns
                    .iter()
                    .zip(reserved)
                    .all(|(column, (pointer, capacity))| {
                        let column = column.as_any().downcast_ref::<Float64Array>().unwrap();
                        column.values().as_ptr() == pointer
                            && column.values().inner().capacity() == capacity * size_of::<f64>()
                    });
        });
        assert!(same_allocations);
        assert_eq!(observed.count_current, 0);
        assert_eq!(observed.bytes_current, 0);
        assert!(observed.count_total <= (3 * COLUMNS + 1) as u64);
        assert!(observed.bytes_max <= (COLUMNS * (rows * size_of::<f64>() + 4096)) as u64);
        println!("gather/finish rows={rows} columns={COLUMNS}: {observed:?}");
        drop((seed, results));
    }
    println!("merge fixed metadata bytes={NULL_FREE_MERGE_METADATA_BYTES}");
    for rows in [64_000, 256_000] {
        check::<1>(rows);
        check::<2>(rows);
    }
}

type ReadoutTrace = Arc<std::sync::Mutex<Vec<(&'static str, Option<NumericSite>)>>>;

fn readout_trace(request: &mut NumericLaneRequest) -> ReadoutTrace {
    let trace = ReadoutTrace::default();
    let observed = Arc::clone(&trace);
    request.test_hook = Some(Arc::new(move |point| {
        let event = match point {
            NumericTestPoint::BeforeGroup(site) => ("before group", Some(site)),
            NumericTestPoint::AfterGroup(site) => ("after group", Some(site)),
            NumericTestPoint::BeforeOutput(site) => ("before output", Some(site)),
            NumericTestPoint::Finish => ("finish", None),
        };
        if event.1.is_none_or(|site| site.row == 0) {
            observed.lock().unwrap().push(event);
        }
    }));
    trace
}

fn readout_success_trace(groups: usize) -> Vec<(&'static str, Option<NumericSite>)> {
    (0..groups)
        .flat_map(|ordinal| {
            let site = Some(NumericSite {
                row: 0,
                phase: NumericPhase::Group,
                ordinal,
            });
            [("before group", site), ("after group", site)]
        })
        .chain([(
            "before output",
            Some(NumericSite {
                row: 0,
                phase: NumericPhase::Output,
                ordinal: 0,
            }),
        )])
        .chain([("finish", None)])
        .collect()
}

#[test]
fn entity_parallel_readout_compatibility_difference_reads_right_after_left_null() {
    let input = input(64_000, 64, false);
    for left_error in [false, true] {
        let mut plan = lane_plan(true);
        if left_error {
            let TypedOutputKind::Difference { left, .. } = &mut plan.outputs[0].kind else {
                unreachable!();
            };
            left.min_periods = 1;
        }
        let (_, [mut request, _]) = pair(&plan, &RollingKernelState::default(), &input);
        request.entities[0].state.groups.truncate(1);
        if left_error {
            request.entities[0].state.groups[0] =
                TypedWindowState::Extrema(TypedExtremaState::new(TypedFrame::Rows(5), false, 20));
        }
        let trace = readout_trace(&mut request);
        let mut progress = LaneProgress::default();
        let error = run_numeric_lane(request, &AtomicBool::new(false), &mut progress).unwrap_err();
        let NumericLaneStop::Ordinary(error) = error else {
            panic!("the original readout error must be retained");
        };
        let CalcFlowError::Internal { message } = error.error else {
            panic!("private readout errors are internal errors");
        };
        assert_eq!(
            message,
            if left_error {
                "typed fused aggregate readout state mismatch"
            } else {
                "typed fused readout group is out of bounds"
            }
        );
        assert_eq!(
            error.site,
            NumericSite {
                row: 0,
                phase: NumericPhase::Output,
                ordinal: 0,
            }
        );
        assert_eq!(progress.active_site, Some(error.site));
        assert_eq!(progress.work.numeric_rows, 1);
        assert!(!progress.work.overflowed);
        let mut expected = readout_success_trace(1);
        expected.pop();
        assert_eq!(*trace.lock().unwrap(), expected);
    }
}

#[test]
fn entity_parallel_readout_compatibility_keeps_minimum_and_builder_error_order() {
    type AppendMean =
        fn(&mut DerivedBuilder, &[TypedWindowState], &TypedOutputPlan, &str) -> Result<()>;
    let plan = lane_plan(false);
    let states = [TypedWindowState::new(plan.groups[0], 20)];
    for append in [
        DerivedBuilder::append as AppendMean,
        append_numeric_lane_mean_for_test,
    ] {
        let mut output = plan.outputs[0];
        output.storage = OutputStorage::Count;
        output.min_periods = 1;
        let mut builder = DerivedBuilder::new(output, 1);
        append(&mut builder, &states, &output, "r").unwrap();
        output.min_periods = 0;
        assert!(matches!(
            append(&mut builder, &states, &output, "r"),
            Err(CalcFlowError::Internal { message })
                if message == "typed rolling output builder does not match its statistic"
        ));
        output.group = 1;
        output.min_periods = u64::MAX;
        assert!(matches!(
            append(&mut builder, &states, &output, "r"),
            Err(CalcFlowError::Internal { message })
                if message == "typed rolling output group is out of bounds"
        ));
        let column = builder.finish().unwrap();
        assert_eq!(column.len(), 1);
        assert!(column.is_null(0));
    }
}

#[test]
fn entity_parallel_difference_readout_keeps_minimum_and_builder_error_order() {
    type AppendDifference = fn(
        &mut DerivedBuilder,
        &[TypedWindowState],
        TypedFloatReadout,
        TypedFloatReadout,
    ) -> Result<()>;
    let plan = lane_plan(false);
    let states = [TypedWindowState::new(plan.groups[0], 20)];
    let readout = TypedFloatReadout {
        group: 0,
        kind: TypedFloatReadoutKind::Mean,
        min_periods: 1,
        ddof: 0,
    };
    let nonaggregate = [TypedWindowState::Ewma(TypedEwmaState::new(0.5))];
    assert_eq!(read_typed_float(&nonaggregate, readout).unwrap(), None);
    assert!(matches!(
        read_typed_float(&nonaggregate, TypedFloatReadout { min_periods: 0, ..readout }),
        Err(CalcFlowError::Internal { message })
            if message == "typed fused aggregate readout state mismatch"
    ));
    let mut output = plan.outputs[0];
    output.storage = OutputStorage::Count;
    output.kind = TypedOutputKind::Difference {
        left: TypedFloatReadout {
            group: 7,
            ..readout
        },
        right: TypedFloatReadout {
            group: 8,
            ..readout
        },
    };
    let TypedOutputKind::Difference { left, right } = output.kind else {
        unreachable!();
    };
    for append in [
        DerivedBuilder::append_difference as AppendDifference,
        append_numeric_lane_difference_for_test,
    ] {
        let mut builder = DerivedBuilder::new(output, 1);
        assert!(matches!(
            append(&mut builder, &states, left, right),
            Err(CalcFlowError::Internal { message })
                if message == "typed rolling output builder does not match its statistic"
        ));
        assert_eq!(builder.finish().unwrap().len(), 0);

        output.storage = OutputStorage::Float64;
        let mut builder = DerivedBuilder::new(output, 1);
        assert!(matches!(
            append(&mut builder, &states, readout, right),
            Err(CalcFlowError::Internal { message })
                if message == "typed fused readout group is out of bounds"
        ));
        assert_eq!(
            builder.finish().unwrap().len(),
            0,
            "left null cannot be appended before right"
        );

        let mut builder = DerivedBuilder::new(output, 1);
        assert!(matches!(
            append(&mut builder, &nonaggregate, TypedFloatReadout { min_periods: 0, ..readout }, right),
            Err(CalcFlowError::Internal { message })
                if message == "typed fused aggregate readout state mismatch"
        ));
        assert_eq!(builder.finish().unwrap().len(), 0);

        let mut builder = DerivedBuilder::new(output, 1);
        append(&mut builder, &nonaggregate, readout, readout).unwrap();
        let column = builder.finish().unwrap();
        assert_eq!(column.len(), 1);
        assert!(
            column.is_null(0),
            "minimum check precedes a nonaggregate state error"
        );
        output.storage = OutputStorage::Count;
    }
}

#[test]
fn entity_parallel_mean_only_difference_keeps_exact_fallback_and_float_bits() {
    let plan = lane_plan(true);
    let readout = |group, min_periods| TypedFloatReadout {
        group,
        kind: TypedFloatReadoutKind::Mean,
        min_periods,
        ddof: u8::MAX,
    };
    for exact in [false, true] {
        for mean in [
            -0.0,
            1.0e308,
            -1.0e308,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            let mut states = [TypedWindowState::new(
                if exact {
                    TypedGroupPlan::Signed {
                        input_index: 3,
                        frame: TypedFrame::Rows(20),
                    }
                } else {
                    plan.groups[0]
                },
                20,
            )];
            let accumulator = match &mut states[0] {
                TypedWindowState::Numeric(state) => &mut state.accumulator,
                TypedWindowState::Exact(state) => &mut state.accumulator,
                _ => unreachable!(),
            };
            accumulator.valid_count = 1;
            accumulator.mean = mean;
            let before = format!("{states:?}");
            for min_periods in [0, 1, 2] {
                let mut expected = DerivedBuilder::new(plan.outputs[0], 1);
                let mut actual = DerivedBuilder::new(plan.outputs[0], 1);
                let readout = readout(0, min_periods);
                expected
                    .append_difference(&states, readout, readout)
                    .unwrap();
                append_numeric_lane_difference_for_test(&mut actual, &states, readout, readout)
                    .unwrap();
                let expected = expected.finish().unwrap();
                let actual = actual.finish().unwrap();
                let expected = expected.as_any().downcast_ref::<Float64Array>().unwrap();
                let actual = actual.as_any().downcast_ref::<Float64Array>().unwrap();
                assert_eq!(actual.nulls(), expected.nulls());
                assert_eq!(actual.value(0).to_bits(), expected.value(0).to_bits());
                if min_periods <= 1 && !mean.is_finite() {
                    assert!(
                        actual.value(0).is_nan(),
                        "same-group nonfinite difference is not zero"
                    );
                }
                assert_eq!(format!("{states:?}"), before);
            }
        }
    }
}

fn readout_sliced_input() -> RecordBatch {
    let source = input(64_009, 64, false);
    let prices = Arc::new(
        (0..source.num_rows())
            .map(|row| match (row / 64) % 37 {
                0 => None,
                1 => Some(f64::NAN),
                2 => Some(f64::INFINITY),
                3 => Some(f64::NEG_INFINITY),
                4 => Some(-0.0),
                5 => Some(0.0),
                6 => Some(1e155),
                7 => Some(-1e155),
                _ => Some(f64::from(u32::try_from(row % 101).unwrap())),
            })
            .collect::<Float64Array>(),
    );
    let other_prices = Arc::new(
        (0..source.num_rows())
            .map(|row| match (row / 64) % 29 {
                0 => Some(17.0),
                1 => None,
                2 => Some(f64::NEG_INFINITY),
                _ => Some(-f64::from(u32::try_from(row % 83).unwrap())),
            })
            .collect::<Float64Array>(),
    );
    let mut fields = source.schema().fields().to_vec();
    fields.push(Arc::new(Field::new("other_price", DataType::Float64, true)));
    let mut columns = source.columns().to_vec();
    columns[3] = prices.clone();
    columns.push(other_prices.clone());
    let input = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .unwrap()
        .slice(9, 64_000);
    for (index, original) in [(3, &prices), (4, &other_prices)] {
        let slice = input
            .column(index)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(slice.values().as_ptr(), original.values()[9..].as_ptr());
    }
    input
}

fn readout_shape_plan(group_count: usize, difference: bool) -> RollingKernelPlan {
    let mut plan = lane_plan(group_count == 2);
    if group_count == 2 {
        plan.groups[1] = TypedGroupPlan::Numeric {
            input_index: 4,
            frame: TypedFrame::Rows(20),
        };
    }
    let readout = |group| TypedFloatReadout {
        group,
        kind: TypedFloatReadoutKind::Mean,
        min_periods: 1,
        ddof: 0,
    };
    plan.outputs[0].group = group_count - 1;
    plan.outputs[0].kind = if difference {
        TypedOutputKind::Difference {
            left: readout(group_count - 1),
            right: readout(0),
        }
    } else {
        TypedOutputKind::Statistic(Statistic::Mean)
    };
    plan.outputs[0].min_periods = if difference { u64::MAX } else { 2 };
    plan
}

fn check_readout_shape(
    plan: &RollingKernelPlan,
    input: &RecordBatch,
    group_count: usize,
    difference: bool,
    hydrated: bool,
) {
    let state = if hydrated {
        warm(plan, input)
    } else {
        RollingKernelState::default()
    };
    let before = format!("{:?}", state.states);
    let expected = plan
        .prepare_ordered_stream(&state, input, "r", None)
        .unwrap();
    let (seed, [mut first, second]) = pair(plan, &state, input);
    let trace = readout_trace(&mut first);
    let mut progress = LaneProgress::default();
    let first = run_numeric_lane(first, &AtomicBool::new(false), &mut progress).unwrap();
    assert_eq!(progress.active_site, None);
    assert_eq!(progress.work.numeric_rows, 32_000);
    assert!(!progress.work.overflowed);
    assert_eq!(*trace.lock().unwrap(), readout_success_trace(group_count));
    let mut exits = [
        NumericLaneExit {
            outcome: NumericLaneOutcome::Complete(first),
            work: progress.work,
        },
        exit(second, &AtomicBool::new(false)),
    ];
    let actual = merge_numeric_results(seed, &mut exits, "job").unwrap();
    let actual_column = actual.execution.columns[0]
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let expected_column = expected.execution.columns[0]
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(actual_column.nulls(), expected_column.nulls());
    assert!(
        actual_column
            .values()
            .iter()
            .zip(expected_column.values())
            .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
    );
    if group_count == 1 && difference && !hydrated {
        assert!(actual_column.is_valid(128 - 9));
        assert!(
            actual_column.value(128 - 9).is_nan(),
            "Inf - Inf is not zero"
        );
    }
    assert_eq!(actual.entity_ids(), expected.entity_ids());
    assert_eq!(
        format!("{:?}", actual.execution.state.states),
        format!("{:?}", expected.execution.state.states)
    );
    assert_eq!(format!("{:?}", state.states), before);
}

#[test]
fn entity_parallel_readout_compatibility_four_shapes_preserve_slices_bits_and_state() {
    let input = readout_sliced_input();
    for group_count in [1, 2] {
        for difference in [false, true] {
            let plan = readout_shape_plan(group_count, difference);
            for hydrated in [false, true] {
                check_readout_shape(&plan, &input, group_count, difference, hydrated);
            }
        }
    }
}

#[test]
fn entity_parallel_readout_compatibility_short_zip_and_extra_state_keep_sites() {
    let input = input(64_000, 64, false);
    let mut plan = lane_plan(true);
    plan.outputs[0].kind = TypedOutputKind::Statistic(Statistic::Mean);
    plan.outputs[0].min_periods = 1;
    let state = RollingKernelState::default();
    let expected = plan
        .prepare_ordered_stream(&state, &input, "r", None)
        .unwrap();
    let expected_column = expected.execution.columns[0]
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for groups in [0, 1, 3] {
        let (_, [mut request, _]) = pair(&plan, &state, &input);
        if groups < 2 {
            request.entities[0].state.groups.truncate(groups);
        } else {
            let mut extra = TypedEwmaState::new(0.5);
            extra.update(Some(73.0), "r").unwrap();
            request.entities[0]
                .state
                .groups
                .push(TypedWindowState::Ewma(extra));
        }
        let prior_extra =
            (groups == 3).then(|| format!("{:?}", request.entities[0].state.groups[2]));
        let trace = readout_trace(&mut request);
        let mut progress = LaneProgress::default();
        let actual = run_numeric_lane(request, &AtomicBool::new(false), &mut progress);
        if groups == 0 {
            let NumericLaneStop::Ordinary(error) = actual.unwrap_err() else {
                panic!("the missing output group must fail at readout");
            };
            assert!(matches!(error.error, CalcFlowError::Internal { message }
                if message == "typed rolling output group is out of bounds"));
            assert_eq!(
                error.site,
                NumericSite {
                    row: 0,
                    phase: NumericPhase::Output,
                    ordinal: 0
                }
            );
            assert_eq!(progress.active_site, Some(error.site));
            assert_eq!(progress.work.numeric_rows, 1);
            let mut expected = readout_success_trace(0);
            expected.pop();
            assert_eq!(*trace.lock().unwrap(), expected);
            continue;
        }
        let mut actual = actual.unwrap();
        let column = actual.columns_for_test()[0].clone();
        for (offset, row) in actual.rows.iter().enumerate() {
            let row = row.original_row as usize;
            assert_eq!(column.is_valid(offset), expected_column.is_valid(row));
            assert_eq!(
                column.value(offset).to_bits(),
                expected_column.value(row).to_bits()
            );
        }
        let expected_groups = &expected.execution.state.states[0].groups;
        assert_eq!(
            format!("{:?}", actual.entities[0].state.groups[0]),
            format!("{:?}", expected_groups[0])
        );
        assert_eq!(
            actual.entities[0].state.transition_count,
            expected.execution.state.states[0].transition_count
        );
        if let Some(prior) = prior_extra {
            assert_eq!(format!("{:?}", actual.entities[0].state.groups[2]), prior);
        }
        assert_eq!(progress.active_site, None);
        assert_eq!(progress.work.numeric_rows, 32_000);
        assert!(!progress.work.overflowed);
        assert_eq!(*trace.lock().unwrap(), readout_success_trace(groups.min(2)));
    }
}

#[test]
fn entity_parallel_readout_compatibility_non_numeric_state_preserves_first_error() {
    let input = input(64_000, 64, false);
    let mut plan = lane_plan(true);
    plan.outputs[0].kind = TypedOutputKind::Statistic(Statistic::Mean);
    plan.outputs[0].min_periods = 1;
    for exact in [false, true] {
        let (_, [mut request, _]) = pair(&plan, &RollingKernelState::default(), &input);
        request.entities[0].state.groups[0] = TypedWindowState::new(
            if exact {
                TypedGroupPlan::Signed {
                    input_index: 3,
                    frame: TypedFrame::Rows(5),
                }
            } else {
                TypedGroupPlan::Extrema {
                    input_index: 3,
                    frame: TypedFrame::Rows(5),
                    descending: false,
                    storage: OutputStorage::Float64,
                }
            },
            20,
        );
        let trace = readout_trace(&mut request);
        let mut progress = LaneProgress::default();
        let NumericLaneStop::Ordinary(error) =
            run_numeric_lane(request, &AtomicBool::new(false), &mut progress).unwrap_err()
        else {
            panic!("the private state must keep its ordinary error");
        };
        assert!(matches!(error.error, CalcFlowError::Internal { message }
        if message == if exact {
            "typed rolling group state does not match its input plan"
        } else {
            "typed rolling output builder does not match its statistic"
        }));
        assert_eq!(
            error.site,
            NumericSite {
                row: 0,
                phase: if exact {
                    NumericPhase::Group
                } else {
                    NumericPhase::Output
                },
                ordinal: 0
            }
        );
        assert_eq!(progress.active_site, Some(error.site));
        assert_eq!(progress.work.numeric_rows, 1);
        assert!(!progress.work.overflowed);
        let mut expected = readout_success_trace(2);
        expected.truncate(if exact { 1 } else { expected.len() - 1 });
        assert_eq!(*trace.lock().unwrap(), expected);
    }
}

#[test]
fn entity_parallel_compiled_readout_preserves_generic_lane_allocation_bounds() {
    let input = input(64_000, 64, true);
    for group_count in [1, 2] {
        for difference in [false, true] {
            let mut plan = lane_plan(group_count == 2);
            plan.outputs[0].kind = if difference {
                let mean = |group| TypedFloatReadout {
                    group,
                    kind: TypedFloatReadoutKind::Mean,
                    min_periods: 1,
                    ddof: 0,
                };
                TypedOutputKind::Difference {
                    left: mean(0),
                    right: mean(group_count - 1),
                }
            } else {
                TypedOutputKind::Statistic(Statistic::Mean)
            };
            let state = warm(&plan, &input);
            let measure = |generic| {
                allocation_counter::measure(|| {
                    let (seed, [request, other]) = pair(&plan, &state, &input);
                    let mut progress = LaneProgress::default();
                    let result = if generic {
                        run_generic_numeric_lane_for_test(
                            request,
                            &AtomicBool::new(false),
                            &mut progress,
                        )
                    } else {
                        run_numeric_lane(request, &AtomicBool::new(false), &mut progress)
                    };
                    let result = result.unwrap();
                    assert_eq!(progress.active_site, None);
                    assert_eq!(progress.work.numeric_rows, 32_000);
                    drop((seed, result, other));
                })
            };
            let generic = measure(true);
            let compiled = measure(false);
            assert_eq!(compiled.count_total, generic.count_total);
            assert_eq!(compiled.bytes_total, generic.bytes_total);
            assert!(compiled.bytes_max <= generic.bytes_max);
            assert_eq!(compiled.count_current, 0);
            assert_eq!(compiled.bytes_current, 0);
            eprintln!(
                "readout G{group_count} difference={difference}: generic={generic:?} compiled={compiled:?}"
            );
        }
    }
}

fn private_readout_input(
    request: &mut NumericLaneRequest,
    input: &RecordBatch,
    output: &mut TypedOutputPlan,
    signed: bool,
) -> TypedGroupInput {
    if signed {
        for entity in &mut request.entities {
            entity.state.groups[0] = TypedWindowState::new(
                TypedGroupPlan::Signed {
                    input_index: 3,
                    frame: TypedFrame::Rows(20),
                },
                1_000,
            );
        }
        TypedGroupInput::Signed(Int64Array::from(vec![71; input.num_rows()]))
    } else {
        for entity in &mut request.entities {
            let mut extra = Float64NumericState::new(TypedFrame::Rows(5), 20);
            extra
                .update(-1, Some(73.0), RollingNumericalProfile::StableV1, 1, "r")
                .unwrap();
            entity.state.groups.push(TypedWindowState::Numeric(extra));
        }
        output.group = 1;
        TypedGroupInput::Single(
            input
                .column(3)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .clone(),
        )
    }
}

#[test]
fn entity_parallel_compiled_readout_preserves_private_input_and_extra_group_readout() {
    let input = input(64_000, 64, false);
    let plan = lane_plan(false);
    for signed in [false, true] {
        let (seed, [mut request, other]) = pair(&plan, &RollingKernelState::default(), &input);
        drop(other);
        let mut output = plan.outputs[0];
        output.min_periods = 1;
        let column = private_readout_input(&mut request, &input, &mut output, signed);
        let prior_extra = (!signed).then(|| format!("{:?}", request.entities[0].state.groups[1]));
        request.replace_inputs_and_outputs_for_test(vec![column], vec![output]);
        let trace = readout_trace(&mut request);
        let mut progress = LaneProgress::default();
        let mut result = run_numeric_lane(request, &AtomicBool::new(false), &mut progress).unwrap();
        let output = &result.columns_for_test()[0];
        assert_eq!(output.null_count(), 0);
        assert!(
            output
                .values()
                .iter()
                .all(|&value| value.to_bits() == if signed { 71.0_f64 } else { 73.0 }.to_bits())
        );
        if let Some(prior_extra) = prior_extra {
            assert!(
                result
                    .entities
                    .iter()
                    .all(|entity| format!("{:?}", entity.state.groups[1]) == prior_extra)
            );
        } else {
            assert!(result.entities.iter().all(|entity| matches!(
                &entity.state.groups[0], TypedWindowState::Exact(state) if state.accumulator.valid_count == 20
            )));
        }
        assert!(
            result
                .entities
                .iter()
                .all(|entity| entity.state.transition_count == 1_000)
        );
        assert_eq!(progress.work.numeric_rows, 32_000);
        assert_eq!(progress.active_site, None);
        assert_eq!(*trace.lock().unwrap(), readout_success_trace(1));
        drop(seed);
    }
}

#[test]
fn entity_parallel_compiled_readout_fixed_metadata_keeps_checked_capacity_boundaries() {
    let descriptor = size_of::<Option<TypedOutputPlan>>();
    let readout_capture = size_of::<TypedOutputPlan>().max(size_of::<[TypedFloatReadout; 2]>());
    let counters = size_of::<[usize; 2]>();
    let fixed_views = size_of::<[(&Float64Array, &TypedGroupInput); 2]>();
    let required = 2 * (descriptor + readout_capture + counters + fixed_views);
    assert_eq!(COMPILED_LANE_READOUT_METADATA_BYTES, required);
    let fixed = ScratchPlan::for_dimensions(0, 0, 0, 0)
        .accounted_bytes()
        .unwrap();
    assert!(fixed >= required + 96 + NULL_FREE_MERGE_METADATA_BYTES);
    let cap_name = ScratchPlan::CAP_BYTES - fixed;
    assert_eq!(
        ScratchPlan::for_dimensions(0, 0, 0, cap_name).accounted_bytes(),
        Some(ScratchPlan::CAP_BYTES)
    );
    assert_eq!(
        ScratchPlan::for_dimensions(0, 0, 0, cap_name + 1).accounted_bytes(),
        Some(ScratchPlan::CAP_BYTES + 1)
    );
    assert_eq!(
        ScratchPlan::for_dimensions(0, 0, 0, usize::MAX - fixed).accounted_bytes(),
        Some(usize::MAX)
    );
    assert_eq!(
        ScratchPlan::for_dimensions(0, 0, 0, usize::MAX - fixed + 1).accounted_bytes(),
        None
    );
    eprintln!(
        "readout metadata descriptor={descriptor} capture={readout_capture} counters={counters} fixed_views={fixed_views} pair_extra={required} total_zero_dimension={fixed}"
    );
}
