use super::*;

fn split_for_cpu(
    prepared: PreparedOrderedKernel<'_>,
    cpus: usize,
) -> Option<(NumericJoinSeed, Vec<NumericLaneRequest>)> {
    prepared.split_lanes(cpus).ok()
}

#[test]
fn cpu_rolling_one_cpu_keeps_the_prepared_serial_path() {
    let batch = input(128_000, 64, false);
    let plan = lane_plan(false);
    let state = warm(&plan, &batch);
    let prepared = plan
        .prepare_ordered_stream_inputs(&state, &batch, "r", None)
        .unwrap();
    assert!(
        split_for_cpu(prepared, 1).is_none(),
        "one CPU must not launch parallel lanes"
    );
}

#[test]
fn cpu_rolling_eight_lanes_keep_exact_output_and_touched_state() {
    for special in [false, true] {
        let batch = input(128_000, 64, special);
        let plan = lane_plan(true);
        let state = warm(&plan, &batch);
        let expected = plan
            .prepare_ordered_stream(&state, &batch, "r", None)
            .unwrap();
        let prepared = plan
            .prepare_ordered_stream_inputs(&state, &batch, "r", None)
            .unwrap();
        let (seed, requests) = split_for_cpu(prepared, 8).expect("balanced entities split");
        assert_eq!(
            requests.len(),
            8,
            "sufficient work must use the supplied CPU capacity"
        );
        let mut exits = requests
            .into_iter()
            .map(|request| exit(request, &AtomicBool::new(false)))
            .collect::<Vec<_>>();
        let actual = merge_numeric_results(seed, &mut exits, "job").unwrap();
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
                    .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
            );
        }
        assert_eq!(
            format!("{:?}", actual.execution.state.states),
            format!("{:?}", expected.execution.state.states)
        );
        assert_eq!(actual.execution.entity_ids, expected.execution.entity_ids);
    }
}

#[test]
fn cpu_rolling_primary_error_considers_all_eight_lanes() {
    let batch = input(128_000, 64, false);
    let plan = lane_plan(false);
    let state = warm(&plan, &batch);
    let prepared = plan
        .prepare_ordered_stream_inputs(&state, &batch, "r", None)
        .unwrap();
    let (seed, mut requests) = split_for_cpu(prepared, 8).expect("balanced eight-lane fixture");
    for lane in [3, 7] {
        Arc::make_mut(&mut requests[lane].entities[0].state).transition_count = u64::MAX;
    }
    let mut exits = requests
        .into_iter()
        .map(|request| exit(request, &AtomicBool::new(false)))
        .collect::<Vec<_>>();
    for (lane, message) in [(3, "earliest-lane-three"), (7, "later-lane-seven")] {
        let NumericLaneOutcome::OrdinaryError(error) = &mut exits[lane].outcome else {
            panic!("the numeric transition must fail at its original row");
        };
        assert_eq!(error.site.row, u32::try_from(lane).unwrap());
        error.error = internal_error(message);
    }
    let error = merge_numeric_results(seed, &mut exits, "job")
        .err()
        .unwrap();
    let CalcFlowError::Internal { message } = error else {
        panic!("the selected original lane error must be preserved");
    };
    assert_eq!(message, "earliest-lane-three");
    assert!(matches!(
        exits[7].outcome,
        NumericLaneOutcome::OrdinaryError(_)
    ));
}

#[test]
fn cpu_rolling_million_rows_remain_eligible_and_account_large_scratch() {
    let batch = input(1_000_000, 64, false);
    let plan = two_output_plan();
    assert!(
        plan.supports_entity_parallel(&batch),
        "large batches must not have a private 256k ceiling"
    );
    let state = warm(&plan, &batch);
    let prepared = plan
        .prepare_ordered_stream_inputs(&state, &batch, "r", None)
        .unwrap();
    let bytes = prepared
        .scratch_plan()
        .expect("large supported input")
        .accounted_bytes()
        .unwrap();
    assert!(bytes > 16 << 20);
    let (_, requests) = split_for_cpu(prepared, 16).expect("no private 16MiB admission ceiling");
    assert_eq!(requests.len(), 16);
}

#[test]
fn cpu_rolling_schedules_visit_each_input_row_once() {
    let batch = input(64_000, 64, false);
    let plan = lane_plan(false);
    let state = warm(&plan, &batch);
    let (_, requests) = pair(&plan, &state, &batch);
    SCHEDULE_VISITS.set(0);
    let exits = requests.map(|request| exit(request, &AtomicBool::new(false)));
    assert!(
        exits
            .iter()
            .all(|exit| matches!(exit.outcome, NumericLaneOutcome::Complete(_)))
    );
    assert_eq!(
        SCHEDULE_VISITS.get(),
        batch.num_rows(),
        "lanes must visit only their compact ordinal schedules"
    );
}

#[test]
fn cpu_rolling_split_reuses_accumulated_entity_counts() {
    let batch = input(64_000, 64, false);
    let plan = lane_plan(false);
    let state = warm(&plan, &batch);
    let prepared = plan
        .prepare_ordered_stream_inputs(&state, &batch, "r", None)
        .unwrap();
    ENTITY_COUNT_VISITS.set(0);
    let _split = prepared.split_two().ok().expect("balanced fixture");
    assert_eq!(
        ENTITY_COUNT_VISITS.get(),
        64,
        "load planning must inspect distinct counts rather than all input IDs"
    );
}

#[test]
fn cpu_rolling_virtual_sixty_four_cpus_have_no_worker_ceiling() {
    let batch = input(1_000_000, 64, false);
    let plan = lane_plan(false);
    let state = warm(&plan, &batch);
    let prepared = plan
        .prepare_ordered_stream_inputs(&state, &batch, "r", None)
        .unwrap();
    let (_, requests) = split_for_cpu(prepared, 64).expect("large balanced batch can be split");
    assert_eq!(requests.len(), 1_000_000 / 16_000);
    assert!(requests.len() > 32);
    assert!(requests.iter().all(|request| !request.entities.is_empty()));
}
