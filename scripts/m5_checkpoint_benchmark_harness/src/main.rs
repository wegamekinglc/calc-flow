use std::{
    any::Any,
    fs::{File, OpenOptions},
    hint::black_box,
    io::Write as _,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::Instant,
};

use calc_flow::{Batch, BatchMetadata, EdgeBudget, ExternalPayload, StreamMessage, edge_channel};
use serde_json::json;
use sha2::{Digest, Sha256};

const CASE: &str = "m5/common/stream_channel_data_roundtrip";
const SAMPLE_COUNT: usize = 30;
const WARMUP_ITERATIONS: usize = 5_000;
const ITERATIONS_PER_SAMPLE: usize = 20_000;
const SOURCE_COMMIT: &str = env!("CALC_FLOW_M5_SOURCE_COMMIT");
const SOURCE_TREE: &str = env!("CALC_FLOW_M5_SOURCE_TREE");
const EXPECTED_HARNESS_SHA256: &str = env!("CALC_FLOW_M5_HARNESS_SHA256");
const RUN_LABEL: &str = env!("CALC_FLOW_M5_RUN_LABEL");
const WORKLOAD_CONTRACT: &str = "edge_channel;rows=128;budget_rows=1024;budget_bytes=1048576;warmup=5000;iterations_per_sample=20000;samples=30";

#[derive(Debug)]
struct Payload;

impl ExternalPayload for Payload {
    fn backend(&self) -> &'static str {
        "m5-common-benchmark"
    }

    fn len(&self) -> usize {
        128
    }

    fn estimated_bytes(&self) -> usize {
        1_024
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn sha256(bytes: impl AsRef<[u8]>) -> String {
    hex::encode(Sha256::digest(bytes.as_ref()))
}

fn git(arguments: &[&str]) -> String {
    let output = Command::new("git").args(arguments).output().unwrap();
    assert!(output.status.success(), "git command failed");
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

fn bootstrap_median_interval(samples: &[f64]) -> [f64; 2] {
    const RESAMPLES: usize = 8_192;
    let mut state = 0x4231_4331_4232_4332_u64;
    let sample_count = u64::try_from(samples.len()).unwrap();
    let mut medians = Vec::with_capacity(RESAMPLES);
    for _ in 0..RESAMPLES {
        let mut resample = Vec::with_capacity(samples.len());
        for _ in samples {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let index = usize::try_from(state % sample_count).unwrap();
            resample.push(samples[index]);
        }
        medians.push(median(&mut resample));
    }
    medians.sort_by(f64::total_cmp);
    [medians[RESAMPLES / 40], medians[RESAMPLES * 39 / 40]]
}

async fn dispatch(
    sender: &mut calc_flow::EdgeSender,
    receiver: &mut calc_flow::EdgeReceiver,
    message: &StreamMessage,
) {
    sender.send(message.clone()).await.unwrap();
    black_box(receiver.recv().await.unwrap().unwrap());
}

fn harness_hash() -> String {
    let mut digest = Sha256::new();
    for (path, bytes) in [
        ("Cargo.toml", include_bytes!("../Cargo.toml").as_slice()),
        ("src/main.rs", include_bytes!("main.rs").as_slice()),
    ] {
        digest.update(path.as_bytes());
        digest.update(u64::try_from(bytes.len()).unwrap().to_be_bytes());
        digest.update(bytes);
    }
    hex::encode(digest.finalize())
}

fn create_report(path: &Path, bytes: &[u8]) {
    assert!(path.is_absolute(), "output path must be absolute");
    let parent = path.parent().unwrap();
    let mut report = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .unwrap();
    report.write_all(bytes).unwrap();
    report.sync_all().unwrap();
    let digest_path = PathBuf::from(format!("{}.sha256", path.display()));
    let mut digest = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(digest_path)
        .unwrap();
    digest
        .write_all(format!("{}\n", sha256(bytes)).as_bytes())
        .unwrap();
    digest.sync_all().unwrap();
    File::open(parent).unwrap().sync_all().unwrap();
}

fn main() {
    assert_eq!(git(&["rev-parse", "HEAD"]), SOURCE_COMMIT);
    assert_eq!(git(&["rev-parse", "HEAD^{tree}"]), SOURCE_TREE);
    assert_eq!(harness_hash(), EXPECTED_HARNESS_SHA256);
    assert!(["B1", "C1", "B2", "C2"].contains(&RUN_LABEL));
    let output = PathBuf::from(std::env::var_os("CALC_FLOW_M5_COMMON_OUTPUT").unwrap());
    let executable = std::env::current_exe().unwrap().canonicalize().unwrap();
    let executable_sha256 = sha256(std::fs::read(&executable).unwrap());
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let budget = EdgeBudget {
        max_rows: 1_024,
        max_bytes: 1 << 20,
    };
    let (mut sender, mut receiver) = edge_channel("m5-common", budget).unwrap();
    let message =
        StreamMessage::data(Batch::external(Arc::new(Payload), BatchMetadata::default()).unwrap());
    runtime.block_on(async {
        for _ in 0..WARMUP_ITERATIONS {
            dispatch(&mut sender, &mut receiver, &message).await;
        }
    });
    let mut samples = Vec::with_capacity(SAMPLE_COUNT);
    for _ in 0..SAMPLE_COUNT {
        let started = Instant::now();
        runtime.block_on(async {
            for _ in 0..ITERATIONS_PER_SAMPLE {
                dispatch(&mut sender, &mut receiver, &message).await;
            }
        });
        samples.push(
            started.elapsed().as_secs_f64() * 1_000_000_000.0
                / f64::from(u32::try_from(ITERATIONS_PER_SAMPLE).unwrap()),
        );
    }
    assert!(
        samples
            .iter()
            .all(|sample| sample.is_finite() && *sample > 0.0)
    );
    let confidence = bootstrap_median_interval(&samples);
    let raw_samples = samples.clone();
    let median = median(&mut samples);
    assert!(confidence[0] <= median && median <= confidence[1]);
    let report = json!({
        "schema": "calc-flow.m5-common-benchmark-run.v1",
        "label": RUN_LABEL,
        "case": CASE,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "harness_sha256": harness_hash(),
        "workload_sha256": sha256(WORKLOAD_CONTRACT),
        "workload_contract": WORKLOAD_CONTRACT,
        "executable": executable,
        "executable_sha256": executable_sha256,
        "sample_count": SAMPLE_COUNT,
        "confidence_level": 0.95,
        "raw_samples_ns": raw_samples,
        "median_ns": median,
        "median_confidence_interval_ns": confidence,
        "timing_scope": {
            "included": "send, reserve, enqueue, receive, release for one immutable data message",
            "excluded": ["runtime construction", "channel fixture", "batch fixture", "warmup", "report encoding and I/O"],
            "remaining_dilution": "Tokio block_on entry is included once per measured sample",
        },
    });
    create_report(&output, &serde_json::to_vec_pretty(&report).unwrap());
}
