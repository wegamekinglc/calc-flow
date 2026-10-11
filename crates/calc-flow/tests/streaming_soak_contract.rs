use calc_flow as engine;
use calc_flow::{CalcFlowError, Epoch, Result};
use serde_json::json;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};
#[path = "../benches/support/streaming_soak/contracts.rs"]
mod contracts;
use contracts::*;

fn validate_private_checkpoint_benchmark_report(path: &Path) -> Result<serde_json::Value> {
    let bytes = read_private_checkpoint_benchmark_report(path)?;
    let digest_path = PathBuf::from(format!("{}.sha256", path.display()));
    let expected = std::fs::read_to_string(&digest_path).map_err(|source| CalcFlowError::Io {
        path: digest_path.display().to_string(),
        source,
    })?;
    if !private_checkpoint_benchmark_report_digest_matches(&bytes, expected.trim()) {
        return Err(benchmark_evidence_error(
            "private benchmark report hash does not match its sidecar",
        ));
    }
    let report: serde_json::Value = serde_json::from_slice(&bytes)
        .map_err(|error| benchmark_evidence_error(error.to_string()))?;
    if report["schema"] != "calc-flow.m5-checkpoint-absolute-benchmark.v1" {
        return Err(benchmark_evidence_error(
            "private benchmark report schema is invalid",
        ));
    }
    Ok(report)
}

fn read_private_checkpoint_benchmark_report(path: &Path) -> Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(benchmark_evidence_error(
            "private benchmark report is not a regular file",
        ));
    }
    std::fs::read(path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })
}

fn private_checkpoint_benchmark_report_digest_matches(bytes: &[u8], expected: &str) -> bool {
    expected.len() == 64
        && expected.bytes().all(|byte| byte.is_ascii_hexdigit())
        && sha256_bytes(bytes) == expected
}

fn checkpoint_rss_process_samples(
    generation: usize,
    pid: u32,
    initial_rss_kib: u64,
) -> CheckpointRssProcessSamples {
    let sample_start = generation * CHECKPOINT_SOAK_PROCESS_SAMPLE_COUNT;
    let samples = (0..CHECKPOINT_SOAK_PROCESS_SAMPLE_COUNT)
        .map(|local_index| RssSample {
            elapsed_seconds: elapsed_at_sample(sample_start + local_index),
            rss_kib: initial_rss_kib
                + u64::try_from(local_index.min(CHECKPOINT_RSS_PROCESS_WARMUP_SAMPLES)).unwrap()
                    * 1_024,
        })
        .collect();
    CheckpointRssProcessSamples {
        generation,
        pid,
        samples,
    }
}

fn checkpoint_rss_process_with_stable_window(
    generation: usize,
    pid: u32,
    initial_rss_kib: u64,
    normalized_growth_kib: u64,
    stable_growth_kib: u64,
) -> CheckpointRssProcessSamples {
    let mut process = checkpoint_rss_process_samples(generation, pid, initial_rss_kib);
    for (local_index, sample) in process.samples.iter_mut().enumerate() {
        if local_index >= CHECKPOINT_RSS_WINDOW_LOCAL_RANGE[0] {
            sample.rss_kib = sample.rss_kib.saturating_add(normalized_growth_kib);
        }
        if local_index
            >= CHECKPOINT_RSS_WINDOW_LOCAL_RANGE[0] + CHECKPOINT_RSS_STABLE_SUBWINDOW_SAMPLES
        {
            sample.rss_kib = sample.rss_kib.saturating_add(stable_growth_kib);
        }
    }
    process
}

fn checkpoint_soak_process_report_fixture(
    plan: &CheckpointSoakProcessPlan,
    pid: u32,
) -> CheckpointSoakProcessReport {
    let generation_started_micros = plan.parent_launch_offset_micros;
    let restored_order = u64::try_from(plan.generation).unwrap() * 10;
    let terminal_order = restored_order + 10;
    let restored_cursor_orders = ["left", "right"]
        .into_iter()
        .map(|source_id| {
            (
                source_id.into(),
                (plan.generation > 0).then(|| hex::encode(restored_order.to_be_bytes())),
            )
        })
        .collect();
    let terminal_cursor_orders = ["left", "right"]
        .into_iter()
        .map(|source_id| (source_id.into(), hex::encode(terminal_order.to_be_bytes())))
        .collect();
    let samples: Vec<CheckpointSoakProcessSample> = (plan.sample_start..plan.sample_end)
        .enumerate()
        .map(|(local_index, index)| CheckpointSoakProcessSample {
            index,
            elapsed_micros: generation_started_micros
                + u64::try_from(local_index + 1).unwrap()
                    * u64::try_from(CHECKPOINT_SOAK_CADENCE.as_micros()).unwrap(),
            rss_kib: 100_000,
            task_count: 8,
            maximum_queue_depth: 1,
            maximum_charged_rows: 1,
            maximum_charged_bytes: 1,
            completed_checkpoints: 1,
            failed_checkpoints: 0,
            manifest_count: 2,
            state_bytes: 1,
        })
        .collect();
    CheckpointSoakProcessReport {
        schema: CHECKPOINT_SOAK_PROCESS_SCHEMA.into(),
        commit: plan.commit.clone(),
        executable_sha256: plan.executable_sha256.clone(),
        config_hash: plan.config_hash.clone(),
        run_root: plan.run_root.clone(),
        parent_pid: plan.parent_pid,
        pid,
        generation: plan.generation,
        sample_start: plan.sample_start,
        sample_end: plan.sample_end,
        generation_started_micros,
        sampling_started_micros: generation_started_micros,
        generation_finished_micros: samples
            .last()
            .map_or(generation_started_micros + 250_000, |sample| {
                sample.elapsed_micros + 250_000
            }),
        restored_epoch: (plan.generation > 0).then_some(restored_order),
        restored_cursor_orders,
        terminal_epoch: terminal_order,
        terminal_cursor_orders,
        restored_window_state: plan.generation > 0,
        restored_watermarks: plan.generation > 0,
        restored_progress: plan.generation > 0,
        samples,
        compacted_epochs: vec![terminal_order],
        maximum_manifest_count: 2,
        maximum_state_bytes: 1,
        completed_checkpoints: 1,
        failed_checkpoints: 0,
        source_open_events: 2,
        source_close_events: 2,
        sink_close_events: 2,
        terminal_cause: if plan.final_generation {
            "natural_end".into()
        } else {
            "explicit_cancel".into()
        },
        terminal_tasks: 0,
        terminal_charged_edges: 0,
        terminal_registries: (0, 0),
        source_records: 20,
        output_records: 40,
        duplicate_records: 0,
        missing_records: 0,
        temporary_artifacts: 0,
    }
}

fn checkpoint_restart_soak_metadata_fixture() -> serde_json::Value {
    checkpoint_restart_soak_metadata(
        "abc123",
        "f".repeat(64).as_str(),
        "Linux 1",
        "rustc 1.88",
        "x86_64-unknown-linux-gnu",
        "glibc 2.39",
    )
}

#[test]
fn parses_linux_vm_rss_without_accepting_other_units() {
    assert_eq!(
        parse_vm_rss_kib("Name:\tcalc\nVmRSS:\t  12345 kB\nThreads:\t4\n"),
        Some(12_345)
    );
    assert_eq!(parse_vm_rss_kib("VmRSS: 12 MB"), None);
    assert_eq!(parse_vm_rss_kib("VmSize: 12 kB"), None);
}

#[test]
fn least_squares_reports_mib_per_hour() {
    let samples = [
        RssSample {
            elapsed_seconds: 0.0,
            rss_kib: 10_000,
        },
        RssSample {
            elapsed_seconds: 1_800.0,
            rss_kib: 10_512,
        },
        RssSample {
            elapsed_seconds: 3_600.0,
            rss_kib: 11_024,
        },
    ];
    let slope = least_squares_mib_per_hour(&samples).unwrap();
    assert!((slope - 1.0).abs() < f64::EPSILON);
}

#[test]
fn median_handles_even_and_odd_windows() {
    assert_eq!(median_kib(&[9, 1, 5]), Some(5));
    assert_eq!(median_kib(&[10, 2, 6, 4]), Some(5));
    assert_eq!(median_kib(&[]), None);
}

#[test]
fn rss_gate_allows_only_slope_threshold_to_be_exceeded() {
    assert!(rss_gate_passed(
        MAX_RSS_SLOPE_MIB_PER_HOUR + 1.0,
        100_000,
        100_000,
    ));
}

#[test]
fn rss_gate_allows_only_median_threshold_to_be_exceeded() {
    assert!(rss_gate_passed(
        0.0,
        100_000,
        100_000 + MAX_MEDIAN_GROWTH_KIB + 1,
    ));
}

#[test]
fn rss_gate_fails_when_both_thresholds_are_exceeded() {
    assert!(!rss_gate_passed(
        MAX_RSS_SLOPE_MIB_PER_HOUR + 1.0,
        100_000,
        100_000 + MAX_MEDIAN_GROWTH_KIB + 1,
    ));
}

#[test]
fn evaluates_a_stable_full_sample_set() {
    let stable = (0..SAMPLE_COUNT)
        .map(|index| RssSample {
            elapsed_seconds: elapsed_at_sample(index),
            rss_kib: 100_000,
        })
        .collect::<Vec<_>>();
    assert!(evaluate_rss_gate(&stable).unwrap().passed);
}

#[test]
fn checkpoint_restart_rss_gate_compares_equivalent_lifecycle_phases() {
    let lifecycle_rss = |index: usize, growing: bool| match index {
        40..50 | 80..90 => 90_000,
        50..80 => 120_000,
        90..120 => {
            let growth = if growing {
                u64::try_from(index - 90).unwrap().saturating_mul(2_000)
            } else {
                0
            };
            120_000 + growth
        }
        _ => 100_000,
    };
    let stable = (0..SAMPLE_COUNT)
        .map(|index| RssSample {
            elapsed_seconds: elapsed_at_sample(index),
            rss_kib: lifecycle_rss(index, false),
        })
        .collect::<Vec<_>>();
    let growing = (0..SAMPLE_COUNT)
        .map(|index| RssSample {
            elapsed_seconds: elapsed_at_sample(index),
            rss_kib: lifecycle_rss(index, true),
        })
        .collect::<Vec<_>>();
    let split_processes = |samples: Vec<RssSample>| {
        samples
            .chunks_exact(CHECKPOINT_SOAK_PROCESS_SAMPLE_COUNT)
            .enumerate()
            .map(|(generation, samples)| CheckpointRssProcessSamples {
                generation,
                pid: u32::try_from(generation + 1).unwrap(),
                samples: samples.to_vec(),
            })
            .collect::<Vec<_>>()
    };
    let stable_processes = split_processes(stable.clone());
    let growing_processes = split_processes(growing);

    assert!(!evaluate_rss_gate(&stable).unwrap().passed);
    assert!(
        evaluate_checkpoint_restart_rss_gate(&stable_processes)
            .unwrap()
            .passed
    );
    assert!(
        !evaluate_checkpoint_restart_rss_gate(&growing_processes)
            .unwrap()
            .passed
    );
}

#[test]
fn checkpoint_restart_rss_gate_normalizes_each_process_baseline() {
    let process_samples = [
        checkpoint_rss_process_samples(0, 101, 100_000),
        checkpoint_rss_process_samples(1, 202, 180_000),
        checkpoint_rss_process_samples(2, 303, 260_000),
    ];

    let gate = evaluate_checkpoint_restart_rss_gate(&process_samples).unwrap();

    assert!(gate.passed);
    assert_eq!(gate.processes.len(), CHECKPOINT_SOAK_GENERATIONS);
    assert_eq!(gate.processes[0].pid, 101);
    assert_eq!(gate.processes[1].pid, 202);
    assert_eq!(gate.processes[2].pid, 303);
    assert_eq!(gate.normalized_growth_spread_kib, 0);
    assert!(
        gate.processes
            .iter()
            .all(|process| process.normalized_growth_kib == 0 && process.passed)
    );
}

#[test]
fn checkpoint_restart_rss_gate_accepts_repeatable_converged_windows() {
    let process_samples = [
        checkpoint_rss_process_with_stable_window(0, 101, 100_000, 10 * 1_024, 4 * 1_024),
        checkpoint_rss_process_with_stable_window(1, 202, 180_000, 12 * 1_024, 4 * 1_024),
        checkpoint_rss_process_with_stable_window(2, 303, 260_000, 14 * 1_024, 4 * 1_024),
    ];

    let gate = evaluate_checkpoint_restart_rss_gate(&process_samples).unwrap();

    assert!(gate.passed);
    assert_eq!(gate.normalized_growth_spread_kib, 0);
    assert!(
        gate.processes
            .iter()
            .all(|process| process.normalized_growth_kib == 4 * 1_024 && process.passed)
    );
}

#[test]
fn checkpoint_restart_rss_gate_rejects_nonrepeatable_converged_windows() {
    let process_samples = [
        checkpoint_rss_process_with_stable_window(0, 101, 100_000, 0, 0),
        checkpoint_rss_process_with_stable_window(1, 202, 180_000, 0, 256),
        checkpoint_rss_process_with_stable_window(2, 303, 260_000, 0, 513),
    ];

    let gate = evaluate_checkpoint_restart_rss_gate(&process_samples).unwrap();

    assert!(gate.processes.iter().all(|process| process.passed));
    assert_eq!(gate.normalized_growth_spread_kib, 513);
    assert!(!gate.passed);
}

#[test]
fn checkpoint_restart_rss_gate_ignores_pre_stable_warmup_variance() {
    let process_samples = [
        checkpoint_rss_process_with_stable_window(0, 101, 100_000, 0, 4 * 1_024),
        checkpoint_rss_process_with_stable_window(1, 202, 180_000, 20 * 1_024, 4 * 1_024),
        checkpoint_rss_process_with_stable_window(2, 303, 260_000, 5 * 1_024, 4 * 1_024),
    ];

    let gate = evaluate_checkpoint_restart_rss_gate(&process_samples).unwrap();

    assert!(gate.passed);
    assert_eq!(gate.normalized_growth_spread_kib, 0);
}

#[test]
fn checkpoint_restart_rss_gate_rejects_reused_process_identity() {
    let process_samples = [
        checkpoint_rss_process_samples(0, 101, 100_000),
        checkpoint_rss_process_samples(1, 101, 180_000),
        checkpoint_rss_process_samples(2, 303, 260_000),
    ];

    assert_eq!(evaluate_checkpoint_restart_rss_gate(&process_samples), None);
}

#[test]
fn rss_gate_rejects_incomplete_or_extra_sample_sets() {
    let samples = (0..=SAMPLE_COUNT)
        .map(|index| RssSample {
            elapsed_seconds: elapsed_at_sample(index),
            rss_kib: 100_000,
        })
        .collect::<Vec<_>>();

    assert_eq!(evaluate_rss_gate(&samples[..SAMPLE_COUNT - 1]), None);
    assert_eq!(evaluate_rss_gate(&samples), None);
}

#[test]
fn twenty_minute_soak_contract_has_exact_cadence_and_sample_windows() {
    assert_eq!(CADENCE, Duration::from_secs(10));
    assert_eq!(SAMPLE_COUNT, 120);
    assert_eq!(WARMUP_SAMPLES, 30);
    assert!((elapsed_at_sample(SAMPLE_COUNT - 1) - 1_200.0).abs() < f64::EPSILON);
}

#[tokio::test(start_paused = true)]
async fn absolute_soak_deadlines_do_not_accumulate_sampling_work_delay() {
    let started = tokio::time::Instant::now();
    let mut samples = Vec::with_capacity(SAMPLE_COUNT);

    for index in 0..SAMPLE_COUNT {
        let elapsed_seconds = wait_for_sample_deadline(started, index).await;
        samples.push(RssSample {
            elapsed_seconds,
            rss_kib: 100_000,
        });
        tokio::time::advance(Duration::from_secs(1)).await;
    }

    assert_eq!(samples.len(), SAMPLE_COUNT);
    assert_eq!(observed_timeline_issue(&samples), None);
    assert!((samples.last().unwrap().elapsed_seconds - 1_200.0).abs() < f64::EPSILON);
}

#[test]
fn soak_source_uses_a_zero_cost_batch_for_every_fourth_sequence() {
    for sequence in 0..8 {
        let batch = soak_batch("left", sequence).unwrap();
        if sequence % 4 == 0 {
            assert_eq!(batch.num_rows(), 0);
            assert_eq!(batch.estimated_bytes().unwrap(), 0);
        } else {
            assert_eq!(batch.num_rows(), 1);
            assert!(batch.estimated_bytes().unwrap() > 0);
        }
    }
}

#[test]
fn soak_metadata_is_machine_readable_and_declares_the_slot_contract() {
    let metadata = soak_metadata("abc123", "Linux 1", "rustc 1.88");

    assert_eq!(metadata["schema"], "calc-flow.m4-soak-log.v1");
    assert_eq!(metadata["commit"], "abc123");
    assert_eq!(metadata["target_duration_seconds"], 1_200);
    assert_eq!(metadata["sample_count"], 120);
    assert_eq!(metadata["sink_write_delay_millis"], 500);
    assert_eq!(metadata["warmup_duration_seconds"], 300);
    assert_eq!(
        metadata["deterministic_seed"],
        "sequential-two-source-window-v1"
    );
    assert_eq!(metadata["environment"]["kernel"], "Linux 1");
    assert_eq!(metadata["environment"]["rustc"], "rustc 1.88");
    assert_eq!(
        metadata["boundedness_contract"]["message_slot_limit"],
        "max_rows"
    );
    assert_eq!(metadata["progress_contract"]["source_count"], 2);
    assert_eq!(metadata["state_contract"]["window_size_micros"], 1);
    assert_eq!(metadata["state_contract"]["checkpoint_batch_interval"], 8);
    assert_eq!(metadata["state_contract"]["terminal_live_keys"], 0);
    assert_eq!(
        metadata["progress_contract"]["fence_selection"],
        "all-visible"
    );
    assert_eq!(
        metadata["command"],
        "CALC_FLOW_STREAM_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::twenty_minute_two_source_slow_sink -- --ignored --exact --nocapture"
    );
}

#[test]
fn soak_zero_cost_count_is_derived_from_the_accepted_oracle() {
    let accepted = BTreeMap::from([
        ("left".to_owned(), vec![0, 1, 2, 3, 4]),
        ("right".to_owned(), vec![0, 1, 2, 3]),
    ]);

    assert_eq!(zero_cost_batch_counts(&accepted), (3, 9));
}

#[test]
fn checkpoint_soak_cadence_tolerance_includes_exact_thirteen_second_boundary() {
    let directory = tempfile::tempdir().unwrap();
    let mut plan = checkpoint_soak_process_plans(
        directory.path(),
        &"a".repeat(40),
        &"b".repeat(64),
        CheckpointSoakProcessMode::Standard,
    )
    .remove(0);
    bind_checkpoint_soak_parent_launch(&mut plan, Duration::ZERO).unwrap();
    let mut report = checkpoint_soak_process_report_fixture(&plan, 10_000);
    report.samples[0].elapsed_micros = 13_000_000;

    assert_eq!(
        checkpoint_soak_sample_issue(&report, 0, &report.samples[0], 10_000_000, 3_000_000),
        None,
    );

    report.samples[0].elapsed_micros = 13_000_001;
    assert!(
        checkpoint_soak_sample_issue(&report, 0, &report.samples[0], 10_000_000, 3_000_000)
            .is_some()
    );
}

#[test]
fn checkpoint_soak_cadence_starts_after_child_startup() {
    let directory = tempfile::tempdir().unwrap();
    let mut plan = checkpoint_soak_process_plans(
        directory.path(),
        &"a".repeat(40),
        &"b".repeat(64),
        CheckpointSoakProcessMode::Standard,
    )
    .remove(0);
    bind_checkpoint_soak_parent_launch(&mut plan, Duration::ZERO).unwrap();
    let mut delayed = checkpoint_soak_process_report_fixture(&plan, 10_000);
    let startup_micros = 15_000_000;
    delayed.sampling_started_micros += startup_micros;
    delayed.generation_finished_micros += startup_micros;
    for sample in &mut delayed.samples {
        sample.elapsed_micros += startup_micros;
    }

    validate_checkpoint_soak_report(&plan, &delayed, 0).unwrap();

    let additional_preflight =
        u64::try_from(MAX_CHECKPOINT_SOAK_CHILD_PREFLIGHT.as_micros()).unwrap() - startup_micros;
    delayed.sampling_started_micros += additional_preflight;
    delayed.generation_finished_micros += additional_preflight;
    for sample in &mut delayed.samples {
        sample.elapsed_micros += additional_preflight;
    }
    validate_checkpoint_soak_report(&plan, &delayed, 0).unwrap();

    delayed.sampling_started_micros += 1;
    assert!(validate_checkpoint_soak_report(&plan, &delayed, 0).is_err());
}

#[test]
fn checkpoint_soak_smoke_allows_slow_preflight_with_ordered_timestamps() {
    let directory = tempfile::tempdir().unwrap();
    let mut plan = checkpoint_soak_process_plans(
        directory.path(),
        &"a".repeat(40),
        &"b".repeat(64),
        CheckpointSoakProcessMode::Smoke,
    )
    .remove(0);
    bind_checkpoint_soak_parent_launch(&mut plan, Duration::from_micros(1)).unwrap();
    let mut report = checkpoint_soak_process_report_fixture(&plan, 10_000);
    report.sampling_started_micros += 61_000_000;
    report.generation_finished_micros += 61_000_000;

    validate_checkpoint_soak_report(&plan, &report, 0).unwrap();

    report.sampling_started_micros = 0;
    assert!(validate_checkpoint_soak_report(&plan, &report, 0).is_err());
}

#[test]
fn checkpoint_soak_standard_watchdog_covers_startup_and_settle() {
    let directory = tempfile::tempdir().unwrap();
    let plan = checkpoint_soak_process_plans(
        directory.path(),
        &"a".repeat(40),
        &"b".repeat(64),
        CheckpointSoakProcessMode::Standard,
    )
    .remove(0);
    let samples = u32::try_from(plan.sample_end - plan.sample_start).unwrap();
    let minimum = CHECKPOINT_SOAK_CADENCE * samples
        + MAX_CHECKPOINT_SOAK_CHILD_PREFLIGHT
        + CHECKPOINT_SOAK_CANCEL_ALIGNMENT_TIMEOUT
        + CHECKPOINT_SOAK_SETTLE_TIMEOUT;
    assert!(checkpoint_soak_process_timeout(&plan) > minimum);
}

#[test]
fn checkpoint_soak_cancel_requires_a_fresh_idle_checkpoint() {
    let previous = Epoch::new(7).unwrap();
    let next = Epoch::new(8).unwrap();
    let mut status = engine::CheckpointStatus {
        last_completed_epoch: Some(previous),
        ..engine::CheckpointStatus::default()
    };
    assert!(!checkpoint_soak_fresh_checkpoint_idle(
        &status,
        Some(previous)
    ));
    status.last_completed_epoch = Some(next);
    status.current_epoch = Some(next);
    assert!(!checkpoint_soak_fresh_checkpoint_idle(
        &status,
        Some(previous)
    ));
    status.current_epoch = None;
    assert!(checkpoint_soak_fresh_checkpoint_idle(
        &status,
        Some(previous)
    ));
}

#[test]
#[cfg(target_os = "linux")]
fn checkpoint_soak_evidence_bundle_contains_all_raw_child_documents() {
    let directory = tempfile::tempdir().unwrap();
    let executable_sha256 = "b".repeat(64);
    let mut plans = checkpoint_soak_process_plans(
        directory.path(),
        &"a".repeat(40),
        &executable_sha256,
        CheckpointSoakProcessMode::Standard,
    );
    for plan in &mut plans {
        bind_checkpoint_soak_parent_launch(
            plan,
            Duration::from_secs(u64::try_from(plan.generation).unwrap() * 401),
        )
        .unwrap();
    }
    let reports = plans
        .iter()
        .enumerate()
        .map(|(generation, plan)| {
            checkpoint_soak_process_report_fixture(
                plan,
                10_000 + u32::try_from(generation).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let parent_timings = reports
        .iter()
        .map(|report| CheckpointSoakParentTiming {
            generation: report.generation,
            launch_micros: report.generation_started_micros,
            finish_micros: report.generation_finished_micros + 250_000,
        })
        .collect::<Vec<_>>();
    let evidence = aggregate_checkpoint_soak_processes(
        plans,
        reports,
        vec![0; CHECKPOINT_SOAK_GENERATIONS],
        parent_timings,
    );
    let rss_gate = evaluate_checkpoint_restart_rss_gate(&evidence.rss_process_samples).unwrap();
    let metadata = checkpoint_restart_soak_metadata(
        &"a".repeat(40),
        &executable_sha256,
        "Linux 1",
        "rustc 1.88",
        "x86_64-unknown-linux-gnu",
        "glibc 2.39",
    );
    let summary = checkpoint_soak_process_summary(&"a".repeat(40), &evidence, &rss_gate);

    let bundle = checkpoint_soak_evidence_bundle(&metadata, &summary, &evidence);

    assert_eq!(bundle["schema"], "calc-flow.m5-checkpoint-soak-evidence.v3");
    assert_eq!(bundle["metadata"], metadata);
    assert_eq!(bundle["summary"], summary);
    assert_eq!(bundle["child_plans"].as_array().unwrap().len(), 3);
    assert_eq!(bundle["child_reports"].as_array().unwrap().len(), 3);
    assert_eq!(
        bundle["child_reports"]
            .as_array()
            .unwrap()
            .iter()
            .map(|report| report["samples"].as_array().unwrap().len())
            .sum::<usize>(),
        CHECKPOINT_SOAK_SAMPLE_COUNT,
    );
    assert_eq!(bundle["executable_sha256"], executable_sha256);

    publish_checkpoint_soak_evidence(directory.path(), &metadata, &summary, &evidence).unwrap();
    assert_eq!(
        read_checkpoint_soak_document::<serde_json::Value>(
            &directory.path().join("evidence/metadata.json"),
        )
        .unwrap(),
        metadata,
    );
    assert_eq!(
        read_checkpoint_soak_document::<serde_json::Value>(
            &directory.path().join("evidence/summary.json"),
        )
        .unwrap(),
        summary,
    );
    assert_eq!(
        read_checkpoint_soak_document::<serde_json::Value>(
            &directory.path().join("evidence/bundle.json"),
        )
        .unwrap(),
        bundle,
    );
}

#[test]
fn checkpoint_restart_process_evidence_fails_closed() {
    let directory = tempfile::tempdir().unwrap();
    let mut plans = checkpoint_soak_process_plans(
        directory.path(),
        &"a".repeat(40),
        &"b".repeat(64),
        CheckpointSoakProcessMode::Standard,
    );
    for plan in &mut plans {
        let generation = u64::try_from(plan.generation).unwrap();
        bind_checkpoint_soak_parent_launch(
            plan,
            Duration::from_micros(250_000 + generation * 400_500_000),
        )
        .unwrap();
    }
    let reports = plans
        .iter()
        .enumerate()
        .map(|(generation, plan)| {
            checkpoint_soak_process_report_fixture(
                plan,
                10_000 + u32::try_from(generation).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let exits = vec![0; CHECKPOINT_SOAK_GENERATIONS];
    let parent_timings = reports
        .iter()
        .map(|report| CheckpointSoakParentTiming {
            generation: report.generation,
            launch_micros: report.generation_started_micros,
            finish_micros: report.generation_finished_micros + 250_000,
        })
        .collect::<Vec<_>>();
    validate_checkpoint_soak_process_set(&plans, &reports, &exits, &parent_timings).unwrap();

    assert_checkpoint_soak_valid_timing_variants(&plans, &reports, &exits, &parent_timings);
    assert_checkpoint_soak_rejects_invalid_reports(&plans, &reports, &exits, &parent_timings);
    assert_checkpoint_soak_rejects_invalid_timeline(&plans, &reports, &parent_timings);

    let evidence = aggregate_checkpoint_soak_processes(plans, reports, exits, parent_timings);
    assert_eq!(
        evidence.report.generation_process_ids,
        [10_000, 10_001, 10_002]
    );
}

fn assert_checkpoint_soak_valid_timing_variants(
    plans: &[CheckpointSoakProcessPlan],
    reports: &[CheckpointSoakProcessReport],
    exits: &[i32],
    parent_timings: &[CheckpointSoakParentTiming],
) {
    let mut late_final_exit = parent_timings.to_vec();
    late_final_exit[2].finish_micros = reports[2].generation_finished_micros
        + u64::try_from(CHECKPOINT_SOAK_PROCESS_OVERHEAD_BUDGET.as_micros()).unwrap();
    validate_checkpoint_soak_process_set(plans, reports, exits, &late_final_exit).unwrap();
    late_final_exit[2].finish_micros += 1;
    assert!(validate_checkpoint_soak_process_set(plans, reports, exits, &late_final_exit).is_err());
    late_final_exit[2].finish_micros = parent_timings[2].finish_micros + 100_000_000;
    assert!(validate_checkpoint_soak_process_set(plans, reports, exits, &late_final_exit).is_err());

    let startup_micros = 15_000_000;
    let mut delayed_plans = plans.to_vec();
    let mut delayed_reports = reports.to_vec();
    let mut delayed_timings = parent_timings.to_vec();
    for (generation, ((plan, report), timing)) in delayed_plans
        .iter_mut()
        .zip(&mut delayed_reports)
        .zip(&mut delayed_timings)
        .enumerate()
    {
        if generation > 0 {
            plan.parent_launch_offset_micros += startup_micros;
            plan.config_hash = checkpoint_soak_plan_hash(plan);
            report.config_hash = plan.config_hash.clone();
            report.generation_started_micros += startup_micros;
            timing.launch_micros += startup_micros;
        }
        report.sampling_started_micros += startup_micros;
        report.generation_finished_micros += startup_micros;
        timing.finish_micros += startup_micros;
        for sample in &mut report.samples {
            sample.elapsed_micros += startup_micros;
        }
    }
    validate_checkpoint_soak_process_set(&delayed_plans, &delayed_reports, exits, &delayed_timings)
        .unwrap();

    let mut bounded_scheduler_jitter = reports.to_vec();
    bounded_scheduler_jitter[0].samples[0].elapsed_micros += 2_250_000;
    validate_checkpoint_soak_process_set(plans, &bounded_scheduler_jitter, exits, parent_timings)
        .unwrap();

    let mut abandoned_nonterminal_staging = reports.to_vec();
    abandoned_nonterminal_staging[0].temporary_artifacts = 7;
    validate_checkpoint_soak_process_set(
        plans,
        &abandoned_nonterminal_staging,
        exits,
        parent_timings,
    )
    .unwrap();
}

fn assert_checkpoint_soak_rejects_invalid_reports(
    plans: &[CheckpointSoakProcessPlan],
    reports: &[CheckpointSoakProcessReport],
    exits: &[i32],
    parent_timings: &[CheckpointSoakParentTiming],
) {
    let mut repeated_pid = reports.to_vec();
    repeated_pid[1].pid = repeated_pid[0].pid;
    assert!(matches!(
        validate_checkpoint_soak_process_set(plans, &repeated_pid, exits, parent_timings),
        Err(CalcFlowError::InvalidArgument { message, .. })
            if message.contains("not distinct")
    ));
    let mut bad_exit = exits.to_vec();
    bad_exit[1] = 9;
    assert!(
        validate_checkpoint_soak_process_set(plans, reports, &bad_exit, parent_timings).is_err()
    );

    let mut wrong_identity = reports.to_vec();
    wrong_identity[1].config_hash = "c".repeat(64);
    assert!(
        validate_checkpoint_soak_process_set(plans, &wrong_identity, exits, parent_timings)
            .is_err()
    );
    let mut wrong_head = reports.to_vec();
    wrong_head[1].commit = "d".repeat(40);
    assert!(
        validate_checkpoint_soak_process_set(plans, &wrong_head, exits, parent_timings).is_err()
    );
    let mut wrong_root = reports.to_vec();
    wrong_root[1].run_root.push("different-root");
    assert!(
        validate_checkpoint_soak_process_set(plans, &wrong_root, exits, parent_timings).is_err()
    );
    let mut wrong_generation = reports.to_vec();
    wrong_generation[1].generation = 2;
    assert!(
        validate_checkpoint_soak_process_set(plans, &wrong_generation, exits, parent_timings)
            .is_err()
    );
    let mut in_memory_only = reports.to_vec();
    in_memory_only[1].restored_epoch = None;
    assert!(
        validate_checkpoint_soak_process_set(plans, &in_memory_only, exits, parent_timings)
            .is_err()
    );

    let mut sample_gap = reports.to_vec();
    sample_gap[1].samples.remove(0);
    assert!(
        validate_checkpoint_soak_process_set(plans, &sample_gap, exits, parent_timings).is_err()
    );
    let mut sample_duplicate = reports.to_vec();
    let duplicate = sample_duplicate[0].samples[39].clone();
    sample_duplicate[1].samples.insert(0, duplicate);
    assert!(
        validate_checkpoint_soak_process_set(plans, &sample_duplicate, exits, parent_timings)
            .is_err()
    );

    let mut mistimed_sample = reports.to_vec();
    mistimed_sample[0].samples[5].elapsed_micros += 4_000_000;
    assert!(matches!(
        validate_checkpoint_soak_process_set(
            plans,
            &mistimed_sample,
            exits,
            parent_timings,
        ),
        Err(CalcFlowError::InvalidArgument { message, .. })
            if message.contains(
                "generation 0 sample 5 elapsed 64000000us outside target 60000000us +/-3000000us"
            )
    ));
}

fn assert_checkpoint_soak_rejects_invalid_timeline(
    plans: &[CheckpointSoakProcessPlan],
    reports: &[CheckpointSoakProcessReport],
    parent_timings: &[CheckpointSoakParentTiming],
) {
    let mut unbounded_gap_reports = reports.to_vec();
    let mut unbounded_gap_timings = parent_timings.to_vec();
    let shifted_launch = parent_timings[0].finish_micros
        + u64::try_from(MAX_CHECKPOINT_SOAK_RESTART_GAP.as_micros()).unwrap()
        + 1_000_000;
    let shift = shifted_launch - unbounded_gap_reports[1].generation_started_micros;
    unbounded_gap_reports[1].generation_started_micros += shift;
    unbounded_gap_reports[1].generation_finished_micros += shift;
    for sample in &mut unbounded_gap_reports[1].samples {
        sample.elapsed_micros += shift;
    }
    unbounded_gap_timings[1].launch_micros += shift;
    unbounded_gap_timings[1].finish_micros += shift;
    assert!(matches!(
        validate_checkpoint_soak_timeline(
            plans,
            &unbounded_gap_reports,
            &unbounded_gap_timings,
        ),
        Err(CalcFlowError::InvalidArgument { message, .. })
            if message.contains("restart gap")
    ));

    let mut forged_parent_timings = parent_timings.to_vec();
    forged_parent_timings[1].finish_micros = reports[1].generation_finished_micros - 10_000;
    assert!(matches!(
        validate_checkpoint_soak_timeline(plans, reports, &forged_parent_timings),
        Err(CalcFlowError::InvalidArgument { message, .. })
            if message.contains("does not bound")
    ));
}

#[test]
fn checkpoint_restart_process_document_rejects_missing_and_malformed_reports() {
    let directory = tempfile::tempdir().unwrap();
    let mut plan = checkpoint_soak_process_plans(
        directory.path(),
        &"a".repeat(40),
        &"b".repeat(64),
        CheckpointSoakProcessMode::Standard,
    )
    .remove(0);
    bind_checkpoint_soak_parent_launch(&mut plan, Duration::from_micros(1_234_567)).unwrap();
    let report = checkpoint_soak_process_report_fixture(&plan, 10_000);
    let round_trip = directory.path().join("round-trip.json");
    write_checkpoint_soak_document(&round_trip, &report).unwrap();
    assert_eq!(
        read_checkpoint_soak_document::<CheckpointSoakProcessReport>(&round_trip).unwrap(),
        report,
    );

    let missing = directory.path().join("missing.json");
    let missing_result: Result<CheckpointSoakProcessReport> =
        read_checkpoint_soak_document(&missing);
    assert!(missing_result.is_err());

    let malformed = directory.path().join("malformed.json");
    std::fs::write(&malformed, b"{not-json}").unwrap();
    let malformed_result: Result<CheckpointSoakProcessReport> =
        read_checkpoint_soak_document(&malformed);
    assert!(matches!(
        malformed_result,
        Err(CalcFlowError::InvalidArgument { message, .. })
            if message.contains("malformed")
    ));
}

#[test]
fn checkpoint_soak_child_command_stabilizes_system_allocator() {
    let command = checkpoint_soak_child_command(
        Path::new("/test/checkpoint-soak"),
        Path::new("/test/plan.json"),
    );
    let environment = command
        .get_envs()
        .map(|(name, value)| (name.to_owned(), value.map(std::ffi::OsStr::to_owned)))
        .collect::<BTreeMap<_, _>>();

    assert_eq!(
        environment.get(std::ffi::OsStr::new("MALLOC_ARENA_MAX")),
        Some(&Some(std::ffi::OsString::from("1")))
    );
    assert_eq!(
        environment.get(std::ffi::OsStr::new("MALLOC_TRIM_THRESHOLD_")),
        Some(&Some(std::ffi::OsString::from("0")))
    );
    assert_eq!(
        environment.get(std::ffi::OsStr::new("MALLOC_TOP_PAD_")),
        Some(&Some(std::ffi::OsString::from("0")))
    );
}

#[test]
fn checkpoint_restart_soak_contract_is_exact_and_machine_readable() {
    let metadata = checkpoint_restart_soak_metadata_fixture();

    assert_eq!(metadata["schema"], "calc-flow.m5-checkpoint-soak.v3");
    assert_eq!(metadata["target_duration_seconds"], 1_200);
    assert_eq!(metadata["sample_count"], 120);
    assert_eq!(metadata["cadence_seconds"], 10);
    assert_eq!(metadata["cadence_tolerance_seconds"], 3);
    assert_eq!(metadata["cadence_tolerance_boundary"], "inclusive");
    assert_eq!(metadata["maximum_restart_gap_seconds"], 60);
    assert_eq!(metadata["maximum_child_preflight_seconds"], 60);
    assert_eq!(metadata["maximum_process_overhead_seconds"], 10);
    assert_eq!(metadata["cancel_alignment_timeout_seconds"], 20);
    assert_eq!(
        metadata["cancel_alignment_policy"],
        "fresh_completed_idle_checkpoint"
    );
    assert_eq!(metadata["sampling_clock_start"], "after_running_preflight");
    assert_eq!(metadata["child_preflight_clock_start"], "child_test_entry");
    assert_eq!(
        metadata["timing_source"],
        "parent_std_instant_plus_child_local_instant"
    );
    assert_eq!(metadata["restart_sample_indices"], json!([39, 79]));
    assert_eq!(metadata["restart_kind"], "os_process");
    assert_eq!(metadata["process_generations"], 3);
    assert_eq!(
        metadata["process_sample_ranges"],
        json!([[0, 40], [40, 80], [80, 120]])
    );
    assert_eq!(
        metadata["child_report_schema"],
        CHECKPOINT_SOAK_PROCESS_SCHEMA
    );
    assert_eq!(metadata["source_count"], 2);
    assert_eq!(metadata["transactional_sink_count"], 2);
    assert_eq!(metadata["retained_epochs"], 2);
    assert_eq!(metadata.get("warmup_samples"), None);
    assert_eq!(metadata.get("restart_warmup_samples"), None);
    assert_eq!(
        metadata["rss_gate"]["normalization"],
        "same_pid_baseline_median"
    );
    assert_eq!(metadata["rss_gate"]["process_warmup_samples"], 30);
    assert_eq!(metadata["rss_gate"]["stable_window_samples"], 10);
    assert_eq!(metadata["rss_gate"]["stable_subwindow_samples"], 5);
    assert_eq!(
        metadata["rss_gate"]["slope_local_sample_range"],
        json!([30, 39])
    );
    assert_eq!(
        metadata["rss_gate"]["baseline_local_sample_range"],
        json!([30, 34])
    );
    assert_eq!(
        metadata["rss_gate"]["stable_local_sample_range"],
        json!([35, 39])
    );
    assert_eq!(metadata["rss_gate"]["required_passing_generations"], 3);
    assert_eq!(metadata["rss_gate"]["slope_role"], "informational");
    assert_eq!(metadata["rss_gate"]["maximum_normalized_growth_kib"], 8_192);
    assert_eq!(
        metadata["rss_gate"]["maximum_normalized_growth_spread_kib"],
        512
    );
    assert_eq!(
        metadata["rss_gate"]["spread_calibration"]["controlled_run_spread_kib"],
        json!([92, 208])
    );
    assert_eq!(
        metadata["rss_gate"]["spread_calibration"]["threshold_policy"],
        "next_power_of_two_above_twice_maximum_observed"
    );
    assert_eq!(metadata["state_bytes_limit"], 64 * 1_024 * 1_024);
    assert_eq!(
        metadata["command"],
        "CALC_FLOW_M5_CHECKPOINT_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::twenty_minute_epoch_checkpoint_restart -- --ignored --exact --nocapture"
    );
}

#[test]
fn checkpoint_restart_soak_contract_pins_provenance_and_artifacts() {
    let metadata = checkpoint_restart_soak_metadata_fixture();

    assert_eq!(
        metadata["environment"]["rss_source"],
        "/proc/self/status:VmRSS"
    );
    assert_eq!(
        metadata["environment"]["target"],
        "x86_64-unknown-linux-gnu"
    );
    assert_eq!(metadata["environment"]["libc"], "glibc 2.39");
    assert_eq!(
        metadata["environment"]["allocator_controls"],
        json!({
            "MALLOC_ARENA_MAX": "1",
            "MALLOC_TRIM_THRESHOLD_": "0",
            "MALLOC_TOP_PAD_": "0",
        })
    );
    assert_eq!(metadata["commit"], "abc123");
    assert_eq!(metadata["executable_sha256"], "f".repeat(64));
    assert_eq!(
        metadata["evidence_artifacts"],
        json!({
            "metadata": "evidence/metadata.json",
            "summary": "evidence/summary.json",
            "bundle": "evidence/bundle.json",
            "child_plans": [
                "evidence/plans/generation-0.json",
                "evidence/plans/generation-1.json",
                "evidence/plans/generation-2.json",
            ],
            "child_reports": [
                "evidence/reports/generation-0.json",
                "evidence/reports/generation-1.json",
                "evidence/reports/generation-2.json",
            ],
        })
    );
}

#[tokio::test]
async fn checkpoint_manifest_scan_skips_entry_removed_after_enumeration() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("manifest-race.json");
    tokio::fs::write(&path, b"enumerated before retention")
        .await
        .unwrap();
    let mut entries = tokio::fs::read_dir(directory.path()).await.unwrap();
    let entry = entries.next_entry().await.unwrap().unwrap();
    tokio::fs::remove_file(entry.path()).await.unwrap();

    assert!(
        read_checkpoint_manifest_document(&entry.path())
            .await
            .is_none()
    );
}

#[test]
fn private_benchmark_errors_name_the_candidate_only_gate() {
    let CalcFlowError::InvalidArgument { field, .. } = benchmark_evidence_error("invalid") else {
        panic!("benchmark evidence error must be an invalid argument");
    };

    assert_eq!(field, "CALC_FLOW_M5_CHECKPOINT_BENCHMARK");
}

#[test]
fn private_full_runner_self_overhead_modes_use_identical_sink_pacing() {
    assert_eq!(
        private_full_runner_sink_delay(PrivateFullRunnerMode::NoCheckpoint),
        Duration::ZERO,
    );
    assert_eq!(
        private_full_runner_sink_delay(PrivateFullRunnerMode::CheckpointDisabled),
        private_full_runner_sink_delay(PrivateFullRunnerMode::CheckpointEnabled),
    );
    assert_eq!(
        private_full_runner_sink_delay(PrivateFullRunnerMode::CheckpointEnabled),
        Duration::from_millis(5),
    );
}

#[test]
fn private_benchmark_rejects_missing_or_stale_embedded_source_identity() {
    let candidate = PrivateBenchmarkProvenance {
        commit: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
        tree: "2222222222222222222222222222222222222222".into(),
        clean: true,
        harness_hash: "1".repeat(64),
        config_hash: "2".repeat(64),
        executable: PathBuf::from("candidate-executable"),
        executable_sha256: "3".repeat(64),
        build_identity_hash: "4".repeat(64),
        toolchain_hash: "5".repeat(64),
        environment_hash: "6".repeat(64),
    };

    validate_private_benchmark_source_identity(
        &candidate,
        Some(&candidate.commit),
        Some(&candidate.tree),
    )
    .unwrap();
    assert!(
        validate_private_benchmark_source_identity(&candidate, None, Some(&candidate.tree))
            .is_err()
    );
    assert!(
        validate_private_benchmark_source_identity(
            &candidate,
            Some("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
            Some(&candidate.tree),
        )
        .is_err()
    );
}

#[test]
fn private_benchmark_metadata_reports_honest_absolute_only_m5_cases() {
    let candidate = PrivateBenchmarkProvenance {
        commit: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb".into(),
        tree: "2222222222222222222222222222222222222222".into(),
        clean: true,
        harness_hash: "harness-v2".into(),
        config_hash: "config-v2".into(),
        executable: PathBuf::from("candidate-executable"),
        executable_sha256: "candidate-executable".into(),
        build_identity_hash: "candidate-build".into(),
        toolchain_hash: "toolchain".into(),
        environment_hash: "environment".into(),
    };
    let measurements = M5_PRIVATE_BENCHMARK_CASES
        .into_iter()
        .map(|case| json!({"case": case, "decision": "absolute_only"}))
        .collect::<Vec<_>>();
    let metadata = private_checkpoint_benchmark_metadata(
        &candidate,
        Path::new("target/m5-checkpoint-criterion"),
        &measurements,
    );

    assert_eq!(
        metadata["schema"],
        "calc-flow.m5-checkpoint-absolute-benchmark.v1"
    );
    assert_eq!(metadata["commit"], candidate.commit);
    assert_eq!(
        metadata["absolute_cases"],
        json!([
            "m5/private_path/barrier_cut_single_source",
            "m5/private_path/barrier_cut_two_source_fan_out",
            "m5/private_path/pass_through_two_input_alignment",
            "m5/private_path/window_two_input_alignment",
            "m5/private_path/dirty_window_state_stage",
            "m5/private_path/non_empty_manifest_publication",
            "m5/private_path/retained_delta_compacted_base_restore",
            "m5/private_path/single_transactional_sink_commit",
            "m5/private_path/multi_transactional_sink_commit",
            "m5/private_full_path/no_checkpoint",
            "m5/private_full_path/checkpoint_disabled",
            "m5/private_full_path/checkpoint_enabled",
        ])
    );
    assert_eq!(metadata["comparison"], "none");
    assert_eq!(metadata["topology"]["sources"], 2);
    assert_eq!(metadata["topology"]["sinks"], 2);
    assert_eq!(metadata["command"], CHECKPOINT_BENCHMARK_COMMAND);
    assert!(
        metadata["command"]
            .as_str()
            .unwrap()
            .contains("cargo test --release --locked")
    );
    assert!(
        metadata["command"]
            .as_str()
            .unwrap()
            .contains("CALC_FLOW_M5_PRIVATE_SOURCE_COMMIT=<candidate-commit>")
    );
    assert!(
        metadata["command"]
            .as_str()
            .unwrap()
            .contains("CALC_FLOW_M5_PRIVATE_SOURCE_TREE=<candidate-tree>")
    );
    assert_eq!(metadata["provenance"]["tree"], candidate.tree);
    assert_eq!(metadata["measurements"].as_array().unwrap().len(), 12);
    assert_eq!(metadata["overall_result"], "absolute_only");
}

#[test]
fn private_benchmark_contract_uses_honest_cases_and_batched_setup() {
    assert_eq!(
        M5_PRIVATE_BENCHMARK_CASES.as_slice(),
        [
            "m5/private_path/barrier_cut_single_source",
            "m5/private_path/barrier_cut_two_source_fan_out",
            "m5/private_path/pass_through_two_input_alignment",
            "m5/private_path/window_two_input_alignment",
            "m5/private_path/dirty_window_state_stage",
            "m5/private_path/non_empty_manifest_publication",
            "m5/private_path/retained_delta_compacted_base_restore",
            "m5/private_path/single_transactional_sink_commit",
            "m5/private_path/multi_transactional_sink_commit",
            "m5/private_full_path/no_checkpoint",
            "m5/private_full_path/checkpoint_disabled",
            "m5/private_full_path/checkpoint_enabled",
        ]
    );
    let source = include_str!("../benches/support/streaming_soak.rs");
    assert_eq!(
        source
            .lines()
            .filter(|line| line.trim_start().starts_with("bencher.iter_batched("))
            .count(),
        12
    );
    assert!(source.contains("BatchSize::SmallInput"));
}

#[test]
fn private_absolute_measurement_recomputes_raw_statistics_and_hashes_inputs() {
    let directory = tempfile::tempdir().unwrap();
    let target_root = directory.path();
    let run_root = target_root.join("candidate-run");
    let case_directory = "m5_private_path_barrier_cut_fan_out";
    let candidate_case = run_root.join(case_directory).join("candidate-sha");
    std::fs::create_dir_all(&candidate_case).unwrap();
    std::fs::write(
        candidate_case.join("estimates.json"),
        serde_json::to_vec(&json!({
            "median": {
                "point_estimate": 104.5,
                "confidence_interval": {
                    "confidence_level": 0.95,
                    "lower_bound": 101.0,
                    "upper_bound": 108.0,
                },
            },
        }))
        .unwrap(),
    )
    .unwrap();
    let sample_bytes = serde_json::to_vec(&json!({
        "sampling_mode": "Flat",
        "iters": vec![1.0; 10],
        "times": (100..110).map(f64::from).collect::<Vec<_>>(),
    }))
    .unwrap();
    std::fs::write(candidate_case.join("sample.json"), &sample_bytes).unwrap();

    let measurement = private_benchmark_absolute_measurement(
        "m5/private_path/barrier_cut_fan_out",
        "candidate-sha",
        &run_root,
        target_root,
    )
    .unwrap();

    assert_eq!(measurement["median_ns"], 104.5);
    assert_eq!(measurement["sample_count"], 10);
    assert_eq!(measurement["confidence_level"], 0.95);
    let confidence = measurement["median_confidence_interval_ns"]
        .as_array()
        .unwrap();
    assert!(confidence[0].as_f64().unwrap() <= 104.5);
    assert!(confidence[1].as_f64().unwrap() >= 104.5);
    assert_eq!(measurement["decision"], "absolute_only");
    // The recorded artifact path uses platform separators.
    let sample_path = measurement["artifacts"]["sample"]["path"]
        .as_str()
        .unwrap()
        .replace('\\', "/");
    assert!(sample_path.ends_with("candidate-sha/sample.json"));
    assert_eq!(
        measurement["artifacts"]["sample"]["sha256"],
        sha256_bytes(&sample_bytes)
    );
    assert!(
        measurement["artifacts"]["estimates"]["sha256"]
            .as_str()
            .is_some_and(|digest| digest.len() == 64)
    );
    validate_private_benchmark_artifacts(&measurement, target_root).unwrap();

    std::fs::write(candidate_case.join("estimates.json"), b"{}").unwrap();
    assert!(validate_private_benchmark_artifacts(&measurement, target_root).is_err());
}

#[test]
fn private_benchmark_report_path_is_validated_and_immutable() {
    let directory = tempfile::tempdir().unwrap();
    let metadata = json!({"schema": "calc-flow.m5-checkpoint-absolute-benchmark.v1"});
    assert!(
        write_private_checkpoint_benchmark_metadata(
            directory.path(),
            "../not-a-commit",
            "run-1",
            &metadata,
        )
        .is_err()
    );

    let commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    let path =
        write_private_checkpoint_benchmark_metadata(directory.path(), commit, "run-1", &metadata)
            .unwrap();
    let original = std::fs::read(&path).unwrap();
    assert!(
        write_private_checkpoint_benchmark_metadata(directory.path(), commit, "run-1", &json!({}),)
            .is_err()
    );
    let repeated =
        write_private_checkpoint_benchmark_metadata(directory.path(), commit, "run-2", &metadata)
            .unwrap();
    assert_ne!(path, repeated);
    assert_eq!(std::fs::read(&path).unwrap(), original);
    validate_private_checkpoint_benchmark_report(&path).unwrap();

    std::fs::write(&path, b"{}").unwrap();
    assert!(validate_private_checkpoint_benchmark_report(&path).is_err());
}

#[test]
fn temporary_artifact_oracle_covers_state_manifest_and_both_sinks() {
    let directory = tempfile::tempdir().unwrap();
    let state_root = directory.path().join("state");
    let manifest_root = directory.path().join("manifests");
    let sink_root = directory.path().join("sinks");
    let state_staging = state_root.join("staging/lineage/1/window");
    let state_committed = state_root.join("committed/lineage/window");
    std::fs::create_dir_all(&state_staging).unwrap();
    std::fs::create_dir_all(&state_committed).unwrap();
    std::fs::create_dir_all(&manifest_root).unwrap();
    std::fs::create_dir_all(sink_root.join("sink-a")).unwrap();
    std::fs::create_dir_all(sink_root.join("sink-b")).unwrap();
    std::fs::write(state_staging.join("segment.tmp"), b"temporary state").unwrap();
    std::fs::write(state_committed.join("segment.segment"), b"durable state").unwrap();
    std::fs::write(
        manifest_root.join("manifest-00000000000000000001.json"),
        b"durable manifest",
    )
    .unwrap();
    std::fs::write(manifest_root.join(".tmp-manifest"), b"temporary manifest").unwrap();
    std::fs::write(
        sink_root.join("sink-a/visible-00000000000000000001.json"),
        b"durable sink output",
    )
    .unwrap();
    std::fs::write(
        sink_root.join("sink-a/prepared-1.json"),
        b"prepared sink output",
    )
    .unwrap();
    std::fs::write(sink_root.join("sink-b/.tmp-sink"), b"temporary sink output").unwrap();

    let artifacts = inspect_checkpoint_temporary_artifacts(&state_root, &manifest_root, &sink_root);
    assert_eq!(artifacts.total(), 4);
    assert_eq!(
        artifacts,
        CheckpointTemporaryArtifacts {
            state: 1,
            manifests: 1,
            sink_a: 1,
            sink_b: 1,
        }
    );
}

#[test]
fn manifest_parent_sync_io_support_classifies_platform_and_permission_capability() {
    assert_eq!(
        classify_manifest_parent_sync_io_support(true, true),
        ManifestParentSyncIoSupport::Supported
    );
    assert_eq!(
        classify_manifest_parent_sync_io_support(false, false),
        ManifestParentSyncIoSupport::Unsupported("requires Unix directory permission semantics")
    );
    assert_eq!(
        classify_manifest_parent_sync_io_support(true, false),
        ManifestParentSyncIoSupport::Unsupported(
            "this process bypasses mode 0111 directory restrictions"
        )
    );
}
