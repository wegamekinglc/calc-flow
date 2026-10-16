use super::contracts::*;
#[cfg(target_os = "linux")]
use super::engine::Result;
use serde_json::json;
#[cfg(target_os = "linux")]
use std::path::Path;
use std::{
    collections::{BTreeMap, BTreeSet},
    time::Duration,
};

pub(super) const CADENCE: Duration = Duration::from_secs(10);

pub(super) const SAMPLE_COUNT: usize = 120;

pub(super) const WARMUP_SAMPLES: usize = 30;

pub(super) const FIVE_MINUTE_SAMPLES: usize = 30;

pub(super) const MAX_RSS_SLOPE_MIB_PER_HOUR: f64 = 1.0;

pub(super) const MAX_MEDIAN_GROWTH_KIB: u64 = 8 * 1024;

pub(super) const STANDARD_SOAK_SINK_DELAY: Duration = Duration::from_millis(500);

pub(super) const SECONDS_PER_HOUR: f64 = 60.0 * 60.0;

pub(super) const SOAK_COMMAND: &str = "CALC_FLOW_STREAM_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::twenty_minute_two_source_slow_sink -- --ignored --exact --nocapture";

pub(super) const CHECKPOINT_SOAK_PROCESS_SAMPLE_COUNT: usize =
    CHECKPOINT_SOAK_SAMPLE_COUNT / CHECKPOINT_SOAK_GENERATIONS;

pub(super) const CHECKPOINT_RSS_PROCESS_WARMUP_SAMPLES: usize = 30;

pub(super) const CHECKPOINT_RSS_STABLE_WINDOW_SAMPLES: usize = 10;

pub(super) const CHECKPOINT_RSS_STABLE_SUBWINDOW_SAMPLES: usize =
    CHECKPOINT_RSS_STABLE_WINDOW_SAMPLES / 2;

pub(super) const CHECKPOINT_RSS_WINDOW_LOCAL_RANGE: [usize; 2] = [
    CHECKPOINT_RSS_PROCESS_WARMUP_SAMPLES,
    CHECKPOINT_SOAK_PROCESS_SAMPLE_COUNT - 1,
];

pub(super) const CHECKPOINT_RSS_BASELINE_LOCAL_RANGE: [usize; 2] = [
    CHECKPOINT_RSS_PROCESS_WARMUP_SAMPLES,
    CHECKPOINT_RSS_PROCESS_WARMUP_SAMPLES + CHECKPOINT_RSS_STABLE_SUBWINDOW_SAMPLES - 1,
];

pub(super) const CHECKPOINT_RSS_STABLE_LOCAL_RANGE: [usize; 2] = [
    CHECKPOINT_RSS_BASELINE_LOCAL_RANGE[1] + 1,
    CHECKPOINT_SOAK_PROCESS_SAMPLE_COUNT - 1,
];

pub(super) const CHECKPOINT_RSS_SPREAD_CALIBRATION_HEAD: &str =
    "a09ab3bba24a2db344d9710d4955158cd86ce9d7";

pub(super) const CHECKPOINT_RSS_SPREAD_CALIBRATION_KIB: [u64; 2] = [92, 208];

pub(super) const MAX_CHECKPOINT_RSS_NORMALIZED_GROWTH_SPREAD_KIB: u64 = 512;

pub(super) const CHECKPOINT_SOAK_COMMAND: &str = "CALC_FLOW_M5_CHECKPOINT_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::twenty_minute_epoch_checkpoint_restart -- --ignored --exact --nocapture";

pub(super) const CHECKPOINT_SOAK_SCHEMA: &str = "calc-flow.m5-checkpoint-soak.v3";

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct RssGate {
    pub(super) slope_mib_per_hour: f64,
    pub(super) first_median_kib: u64,
    pub(super) final_median_kib: u64,
    pub(super) passed: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct CheckpointRssProcessGate {
    pub(super) generation: usize,
    pub(super) pid: u32,
    pub(super) baseline_sample_range: [usize; 2],
    pub(super) stable_sample_range: [usize; 2],
    pub(super) slope_mib_per_hour: f64,
    pub(super) baseline_median_kib: u64,
    pub(super) stable_median_kib: u64,
    pub(super) normalized_growth_kib: u64,
    pub(super) passed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct CheckpointRestartRssGate {
    pub(super) processes: Vec<CheckpointRssProcessGate>,
    pub(super) normalized_growth_spread_kib: u64,
    pub(super) passed: bool,
}

pub(super) fn parse_vm_rss_kib(status: &str) -> Option<u64> {
    status.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        if fields.next()? != "VmRSS:" {
            return None;
        }
        let kib = fields.next()?.parse().ok()?;
        (fields.next()? == "kB").then_some(kib)
    })
}

pub(super) fn sample_count_as_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).expect("the bounded soak sample count fits u32"))
}

pub(super) fn rss_kib_as_f64(rss_kib: u64) -> f64 {
    f64::from(u32::try_from(rss_kib).expect("a process VmRSS in KiB fits u32"))
}

pub(super) fn elapsed_at_sample(index: usize) -> f64 {
    sample_count_as_f64(index + 1) * CADENCE.as_secs_f64()
}

pub(super) async fn wait_for_sample_deadline(started: tokio::time::Instant, index: usize) -> f64 {
    let sample_number = u32::try_from(index + 1).expect("the bounded soak sample count fits u32");
    tokio::time::sleep_until(started + CADENCE * sample_number).await;
    started.elapsed().as_secs_f64()
}

pub(super) fn observed_timeline_issue(samples: &[RssSample]) -> Option<&'static str> {
    if samples.len() != SAMPLE_COUNT {
        return Some("sample count differs from the soak contract");
    }
    for (index, sample) in samples.iter().enumerate() {
        if sample.elapsed_seconds < elapsed_at_sample(index) {
            return Some("sample preceded its absolute deadline");
        }
        if index > 0 && sample.elapsed_seconds <= samples[index - 1].elapsed_seconds {
            return Some("observed sample timestamps are not strictly increasing");
        }
    }
    (samples.last()?.elapsed_seconds < TARGET_DURATION.as_secs_f64())
        .then_some("observed timeline ended before the soak target")
}

pub(super) fn least_squares_mib_per_hour(samples: &[RssSample]) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let count = sample_count_as_f64(samples.len());
    let mean_x = samples
        .iter()
        .map(|sample| sample.elapsed_seconds)
        .sum::<f64>()
        / count;
    let mean_y = samples
        .iter()
        .map(|sample| rss_kib_as_f64(sample.rss_kib))
        .sum::<f64>()
        / count;
    let numerator = samples
        .iter()
        .map(|sample| (sample.elapsed_seconds - mean_x) * (rss_kib_as_f64(sample.rss_kib) - mean_y))
        .sum::<f64>();
    let denominator = samples
        .iter()
        .map(|sample| (sample.elapsed_seconds - mean_x).powi(2))
        .sum::<f64>();
    (denominator > 0.0).then_some(numerator / denominator * SECONDS_PER_HOUR / 1_024.0)
}

pub(super) fn median_kib(values: &[u64]) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        Some(sorted[middle - 1].saturating_add(sorted[middle]) / 2)
    } else {
        Some(sorted[middle])
    }
}

pub(super) fn rss_gate_passed(
    slope_mib_per_hour: f64,
    first_median_kib: u64,
    final_median_kib: u64,
) -> bool {
    let slope_exceeded = slope_mib_per_hour > MAX_RSS_SLOPE_MIB_PER_HOUR;
    let median_exceeded = final_median_kib > first_median_kib.saturating_add(MAX_MEDIAN_GROWTH_KIB);
    !(slope_exceeded && median_exceeded)
}

pub(super) fn evaluate_rss_gate(samples: &[RssSample]) -> Option<RssGate> {
    if samples.len() != SAMPLE_COUNT {
        return None;
    }
    let post_warmup = samples.get(WARMUP_SAMPLES..)?;
    let slope_mib_per_hour = least_squares_mib_per_hour(post_warmup)?;
    let (first_median_kib, final_median_kib) = rss_window_medians(post_warmup)?;
    Some(RssGate {
        slope_mib_per_hour,
        first_median_kib,
        final_median_kib,
        passed: rss_gate_passed(slope_mib_per_hour, first_median_kib, final_median_kib),
    })
}

pub(super) fn evaluate_checkpoint_restart_rss_gate(
    process_samples: &[CheckpointRssProcessSamples],
) -> Option<CheckpointRestartRssGate> {
    if process_samples.len() != CHECKPOINT_SOAK_GENERATIONS
        || process_samples
            .iter()
            .map(|process| process.pid)
            .collect::<BTreeSet<_>>()
            .len()
            != CHECKPOINT_SOAK_GENERATIONS
    {
        return None;
    }
    let processes = process_samples
        .iter()
        .enumerate()
        .map(|(generation, process)| evaluate_checkpoint_rss_process(generation, process))
        .collect::<Option<Vec<_>>>()?;
    let minimum_growth = processes
        .iter()
        .map(|process| process.normalized_growth_kib)
        .min()?;
    let maximum_growth = processes
        .iter()
        .map(|process| process.normalized_growth_kib)
        .max()?;
    let normalized_growth_spread_kib = maximum_growth.saturating_sub(minimum_growth);
    Some(CheckpointRestartRssGate {
        passed: normalized_growth_spread_kib <= MAX_CHECKPOINT_RSS_NORMALIZED_GROWTH_SPREAD_KIB
            && processes.iter().all(|process| process.passed),
        processes,
        normalized_growth_spread_kib,
    })
}

fn checkpoint_rss_window(
    samples: &[RssSample],
    range: [usize; 2],
    expected_len: usize,
) -> Option<&[RssSample]> {
    let window = samples.get(range[0]..=range[1])?;
    (window.len() == expected_len).then_some(window)
}

fn checkpoint_rss_window_statistics(samples: &[RssSample]) -> Option<(f64, u64, u64)> {
    let stable_window = checkpoint_rss_window(
        samples,
        CHECKPOINT_RSS_WINDOW_LOCAL_RANGE,
        CHECKPOINT_RSS_STABLE_WINDOW_SAMPLES,
    )?;
    let baseline = checkpoint_rss_window(
        samples,
        CHECKPOINT_RSS_BASELINE_LOCAL_RANGE,
        CHECKPOINT_RSS_STABLE_SUBWINDOW_SAMPLES,
    )?;
    let stable_samples = checkpoint_rss_window(
        samples,
        CHECKPOINT_RSS_STABLE_LOCAL_RANGE,
        CHECKPOINT_RSS_STABLE_SUBWINDOW_SAMPLES,
    )?;
    Some((
        least_squares_mib_per_hour(stable_window)?,
        rss_sample_median(baseline)?,
        rss_sample_median(stable_samples)?,
    ))
}

fn evaluate_checkpoint_rss_process(
    generation: usize,
    process: &CheckpointRssProcessSamples,
) -> Option<CheckpointRssProcessGate> {
    if process.generation != generation
        || process.samples.len() != CHECKPOINT_SOAK_PROCESS_SAMPLE_COUNT
    {
        return None;
    }
    let (slope_mib_per_hour, baseline_median_kib, stable_median_kib) =
        checkpoint_rss_window_statistics(&process.samples)?;
    let normalized_growth_kib = stable_median_kib.saturating_sub(baseline_median_kib);
    let sample_start = generation.checked_mul(CHECKPOINT_SOAK_PROCESS_SAMPLE_COUNT)?;
    Some(CheckpointRssProcessGate {
        generation,
        pid: process.pid,
        baseline_sample_range: [
            sample_start + CHECKPOINT_RSS_BASELINE_LOCAL_RANGE[0],
            sample_start + CHECKPOINT_RSS_BASELINE_LOCAL_RANGE[1],
        ],
        stable_sample_range: [
            sample_start + CHECKPOINT_RSS_STABLE_LOCAL_RANGE[0],
            sample_start + CHECKPOINT_RSS_STABLE_LOCAL_RANGE[1],
        ],
        slope_mib_per_hour,
        baseline_median_kib,
        stable_median_kib,
        normalized_growth_kib,
        passed: normalized_growth_kib <= MAX_MEDIAN_GROWTH_KIB,
    })
}

pub(super) fn rss_window_medians(samples: &[RssSample]) -> Option<(u64, u64)> {
    let first = samples.get(..FIVE_MINUTE_SAMPLES)?;
    let final_start = samples.len().checked_sub(FIVE_MINUTE_SAMPLES)?;
    let final_samples = samples.get(final_start..)?;
    Some((rss_sample_median(first)?, rss_sample_median(final_samples)?))
}

pub(super) fn rss_sample_median(samples: &[RssSample]) -> Option<u64> {
    median_kib(
        &samples
            .iter()
            .map(|sample| sample.rss_kib)
            .collect::<Vec<_>>(),
    )
}

pub(super) fn soak_metadata(commit: &str, kernel: &str, rustc: &str) -> serde_json::Value {
    json!({
        "schema": "calc-flow.m4-soak-log.v1",
        "type": "calc_flow_stream_soak_metadata",
        "runtime_path": "ContinuousRunner/source/progress-driver/union/window/sink/supervisor/reaper",
        "commit": commit,
        "deterministic_seed": "sequential-two-source-window-v1",
        "command": SOAK_COMMAND,
        "environment": {
            "kernel": kernel,
            "rustc": rustc,
            "allocator": "system",
            "rss_source": "/proc/self/status:VmRSS",
        },
        "cadence_seconds": CADENCE.as_secs(),
        "target_duration_seconds": TARGET_DURATION.as_secs(),
        "sink_write_delay_millis": STANDARD_SOAK_SINK_DELAY.as_millis(),
        "sample_count": SAMPLE_COUNT,
        "warmup_samples": WARMUP_SAMPLES,
        "warmup_duration_seconds": (CADENCE
            * u32::try_from(WARMUP_SAMPLES).expect("the bounded soak warmup count fits u32"))
        .as_secs(),
        "boundedness_contract": {
            "message_slot_limit": "max_rows",
            "charged_rows_limit": "max_rows",
            "charged_bytes_limit": "max_bytes",
            "zero_cost_envelopes_consume_message_slots": true,
        },
        "progress_contract": {
            "source_count": 2,
            "watermark_policy": "source-provided",
            "per_binding_inbox_capacity": 64,
            "fence_selection": "all-visible",
            "completion_receipts": true,
            "full_execution_trace": true,
            "gate_cut_evidence": true,
            "settlement_latency_evidence": true,
        },
        "state_contract": {
            "window_size_micros": SOAK_WINDOW_MICROS,
            "checkpoint_batch_interval": SOAK_CHECKPOINT_BATCH_INTERVAL,
            "incremental_arrow_segments": true,
            "compaction_required": true,
            "terminal_live_keys": 0,
            "deterministic_final_aggregate": true,
        },
    })
}

pub(super) fn zero_cost_batch_counts(accepted: &BTreeMap<String, Vec<u64>>) -> (usize, usize) {
    let total = accepted.values().map(Vec::len).sum();
    let zero_cost = accepted
        .values()
        .flatten()
        .filter(|sequence| **sequence % 4 == 0)
        .count();
    (zero_cost, total)
}

pub(super) fn checkpoint_restart_soak_metadata(
    commit: &str,
    executable_sha256: &str,
    kernel: &str,
    rustc: &str,
    target: &str,
    libc: &str,
) -> serde_json::Value {
    json!({
        "schema": CHECKPOINT_SOAK_SCHEMA,
        "commit": commit,
        "executable_sha256": executable_sha256,
        "target_duration_seconds": 1_200,
        "sample_count": CHECKPOINT_SOAK_SAMPLE_COUNT,
        "cadence_seconds": CHECKPOINT_SOAK_CADENCE.as_secs(),
        "cadence_tolerance_seconds": CHECKPOINT_SOAK_CADENCE_TOLERANCE.as_secs(),
        "cadence_tolerance_boundary": "inclusive",
        "maximum_restart_gap_seconds": MAX_CHECKPOINT_SOAK_RESTART_GAP.as_secs(),
        "maximum_child_preflight_seconds": MAX_CHECKPOINT_SOAK_CHILD_PREFLIGHT.as_secs(),
        "maximum_process_overhead_seconds": CHECKPOINT_SOAK_PROCESS_OVERHEAD_BUDGET.as_secs(),
        "cancel_alignment_timeout_seconds": CHECKPOINT_SOAK_CANCEL_ALIGNMENT_TIMEOUT.as_secs(),
        "cancel_alignment_policy": "fresh_completed_idle_checkpoint",
        "sampling_clock_start": "after_running_preflight",
        "child_preflight_clock_start": "child_test_entry",
        "timing_source": "parent_std_instant_plus_child_local_instant",
        "restart_sample_indices": CHECKPOINT_SOAK_RESTART_SAMPLES,
        "restart_kind": "os_process",
        "process_generations": CHECKPOINT_SOAK_GENERATIONS,
        "process_sample_ranges": [[0, 40], [40, 80], [80, 120]],
        "child_report_schema": CHECKPOINT_SOAK_PROCESS_SCHEMA,
        "source_count": 2,
        "transactional_sink_count": 2,
        "retained_epochs": 2,
        "rss_gate": {
            "normalization": "same_pid_baseline_median",
            "process_warmup_samples": CHECKPOINT_RSS_PROCESS_WARMUP_SAMPLES,
            "stable_window_samples": CHECKPOINT_RSS_STABLE_WINDOW_SAMPLES,
            "stable_subwindow_samples": CHECKPOINT_RSS_STABLE_SUBWINDOW_SAMPLES,
            "slope_local_sample_range": CHECKPOINT_RSS_WINDOW_LOCAL_RANGE,
            "baseline_local_sample_range": CHECKPOINT_RSS_BASELINE_LOCAL_RANGE,
            "stable_local_sample_range": CHECKPOINT_RSS_STABLE_LOCAL_RANGE,
            "required_passing_generations": CHECKPOINT_SOAK_GENERATIONS,
            "slope_role": "informational",
            "maximum_normalized_growth_kib": MAX_MEDIAN_GROWTH_KIB,
            "maximum_normalized_growth_spread_kib": MAX_CHECKPOINT_RSS_NORMALIZED_GROWTH_SPREAD_KIB,
            "spread_calibration": {
                "head": CHECKPOINT_RSS_SPREAD_CALIBRATION_HEAD,
                "controlled_run_spread_kib": CHECKPOINT_RSS_SPREAD_CALIBRATION_KIB,
                "threshold_policy": "next_power_of_two_above_twice_maximum_observed",
            },
        },
        "state_bytes_limit": MAX_CHECKPOINT_SOAK_STATE_BYTES,
        "deterministic_seed": "two-source-final-window-restart-v1",
        "environment": {
            "kernel": kernel,
            "rustc": rustc,
            "target": target,
            "libc": libc,
            "allocator": "system",
            "allocator_controls": {
                "MALLOC_ARENA_MAX": CHECKPOINT_SOAK_MALLOC_ARENA_MAX,
                "MALLOC_TRIM_THRESHOLD_": CHECKPOINT_SOAK_MALLOC_TRIM_THRESHOLD,
                "MALLOC_TOP_PAD_": CHECKPOINT_SOAK_MALLOC_TOP_PAD,
            },
            "rss_source": "/proc/self/status:VmRSS",
        },
        "evidence_artifacts": {
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
        },
        "command": CHECKPOINT_SOAK_COMMAND,
    })
}

#[cfg(target_os = "linux")]
pub(super) fn checkpoint_soak_sample_max(
    evidence: &CheckpointSoakProcessEvidence,
    select: impl Fn(&CheckpointSoakProcessSample) -> usize,
) -> usize {
    evidence
        .processes
        .iter()
        .flat_map(|process| process.samples.iter().map(&select))
        .max()
        .unwrap_or(0)
}

#[cfg(target_os = "linux")]
pub(super) fn checkpoint_soak_process_summary(
    commit: &str,
    evidence: &CheckpointSoakProcessEvidence,
    rss_gate: &CheckpointRestartRssGate,
) -> serde_json::Value {
    let report = &evidence.report;
    json!({
        "schema": CHECKPOINT_SOAK_SCHEMA,
        "type": "calc_flow_m5_checkpoint_soak_summary",
        "commit": commit,
        "executable_sha256": evidence.plans[0].executable_sha256,
        "target_duration_seconds": 1_200,
        "sample_count": evidence.rss_samples.len(),
        "restart_sample_indices": CHECKPOINT_SOAK_RESTART_SAMPLES,
        "restarts": report.restarts,
        "processes": evidence.processes.iter().zip(&evidence.parent_timings).map(|(process, timing)| json!({
            "generation": process.generation,
            "pid": process.pid,
            "exit_code": report.generation_exit_codes[process.generation],
            "sample_range": [process.sample_start, process.sample_end],
            "generation_started_seconds": Duration::from_micros(
                process.generation_started_micros,
            ).as_secs_f64(),
            "sampling_started_seconds": Duration::from_micros(
                process.sampling_started_micros,
            ).as_secs_f64(),
            "generation_finished_seconds": Duration::from_micros(
                process.generation_finished_micros,
            ).as_secs_f64(),
            "parent_launch_seconds": Duration::from_micros(timing.launch_micros).as_secs_f64(),
            "parent_finish_seconds": Duration::from_micros(timing.finish_micros).as_secs_f64(),
            "restored_epoch": process.restored_epoch,
            "terminal_epoch": process.terminal_epoch,
            "config_hash": process.config_hash,
        })).collect::<Vec<_>>(),
        "completed_epochs": report.completed_epochs,
        "compacted_window_epochs": report.compacted_window_epochs,
        "resource_bounds": {
            "maximum_task_count": checkpoint_soak_sample_max(evidence, |sample| sample.task_count),
            "maximum_queue_depth": checkpoint_soak_sample_max(
                evidence,
                |sample| sample.maximum_queue_depth,
            ),
            "maximum_charged_rows": checkpoint_soak_sample_max(
                evidence,
                |sample| sample.maximum_charged_rows,
            ),
            "maximum_charged_bytes": checkpoint_soak_sample_max(
                evidence,
                |sample| sample.maximum_charged_bytes,
            ),
            "maximum_manifest_count": report.maximum_manifest_count,
            "maximum_state_bytes": report.maximum_state_bytes,
            "state_bytes_limit": MAX_CHECKPOINT_SOAK_STATE_BYTES,
        },
        "conservation": {
            "source_records": report.source_records,
            "output_records": report.output_records,
            "duplicate_records": report.duplicate_records,
            "missing_records": report.missing_records,
        },
        "rss": {
            "normalization": "same_pid_baseline_median",
            "process_warmup_samples": CHECKPOINT_RSS_PROCESS_WARMUP_SAMPLES,
            "stable_window_samples": CHECKPOINT_RSS_STABLE_WINDOW_SAMPLES,
            "stable_subwindow_samples": CHECKPOINT_RSS_STABLE_SUBWINDOW_SAMPLES,
            "slope_local_sample_range": CHECKPOINT_RSS_WINDOW_LOCAL_RANGE,
            "required_passing_generations": CHECKPOINT_SOAK_GENERATIONS,
            "slope_role": "informational",
            "maximum_normalized_growth_kib": MAX_MEDIAN_GROWTH_KIB,
            "maximum_normalized_growth_spread_kib": MAX_CHECKPOINT_RSS_NORMALIZED_GROWTH_SPREAD_KIB,
            "spread_calibration": {
                "head": CHECKPOINT_RSS_SPREAD_CALIBRATION_HEAD,
                "controlled_run_spread_kib": CHECKPOINT_RSS_SPREAD_CALIBRATION_KIB,
                "threshold_policy": "next_power_of_two_above_twice_maximum_observed",
            },
            "normalized_growth_spread_kib": rss_gate.normalized_growth_spread_kib,
            "processes": rss_gate.processes.iter().map(|process| json!({
                "generation": process.generation,
                "pid": process.pid,
                "baseline_sample_range": process.baseline_sample_range,
                "stable_sample_range": process.stable_sample_range,
                "slope_mib_per_hour": process.slope_mib_per_hour,
                "baseline_median_kib": process.baseline_median_kib,
                "stable_median_kib": process.stable_median_kib,
                "normalized_growth_kib": process.normalized_growth_kib,
                "passed": process.passed,
            })).collect::<Vec<_>>(),
            "passed": rss_gate.passed,
        },
        "temporary_artifacts": report.temporary_artifacts,
        "terminal_tasks": report.terminal_tasks,
        "terminal_charged_edges": report.terminal_charged_edges,
        "terminal_registries": report.terminal_registries,
        "source_open_events": evidence.processes.iter()
            .map(|process| process.source_open_events).sum::<usize>(),
        "source_close_events": evidence.processes.iter()
            .map(|process| process.source_close_events).sum::<usize>(),
        "sink_close_events": evidence.processes.iter()
            .map(|process| process.sink_close_events).sum::<usize>(),
    })
}

#[cfg(target_os = "linux")]
pub(super) fn checkpoint_soak_evidence_bundle(
    metadata: &serde_json::Value,
    summary: &serde_json::Value,
    evidence: &CheckpointSoakProcessEvidence,
) -> serde_json::Value {
    json!({
        "schema": "calc-flow.m5-checkpoint-soak-evidence.v3",
        "type": "calc_flow_m5_checkpoint_soak_evidence",
        "commit": metadata["commit"],
        "executable_sha256": metadata["executable_sha256"],
        "metadata": metadata,
        "summary": summary,
        "child_plans": evidence.plans,
        "child_reports": evidence.processes,
        "parent_timings": evidence.parent_timings,
    })
}

#[cfg(target_os = "linux")]
pub(super) fn publish_checkpoint_soak_evidence(
    run_root: &Path,
    metadata: &serde_json::Value,
    summary: &serde_json::Value,
    evidence: &CheckpointSoakProcessEvidence,
) -> Result<()> {
    let evidence_root = run_root.join("evidence");
    write_checkpoint_soak_document(&evidence_root.join("metadata.json"), metadata)?;
    write_checkpoint_soak_document(&evidence_root.join("summary.json"), summary)?;
    write_checkpoint_soak_document(
        &evidence_root.join("bundle.json"),
        &checkpoint_soak_evidence_bundle(metadata, summary, evidence),
    )
}

const _: () = assert!(CHECKPOINT_RSS_STABLE_WINDOW_SAMPLES.is_multiple_of(2));
const _: () = assert!(
    CHECKPOINT_RSS_PROCESS_WARMUP_SAMPLES + CHECKPOINT_RSS_STABLE_WINDOW_SAMPLES
        == CHECKPOINT_SOAK_PROCESS_SAMPLE_COUNT
);
const _: () = assert!(MAX_CHECKPOINT_RSS_NORMALIZED_GROWTH_SPREAD_KIB < MAX_MEDIAN_GROWTH_KIB);
