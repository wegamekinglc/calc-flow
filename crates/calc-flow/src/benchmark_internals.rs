//! Opt-in, data-only fixtures for native performance acceptance.

mod dispatch_trace;
pub use dispatch_trace::DispatchTraceOwner;

use std::{
    collections::{BTreeMap, BTreeSet},
    io::Cursor,
    path::{Component, Path},
    time::Instant,
};

use datafusion::arrow::{ipc::reader::FileReader, record_batch::RecordBatch};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{CalcFlowError, Result};

#[derive(Clone)]
pub(crate) enum EventKind {
    Data(crate::Batch),
    Watermark(crate::EventTime),
    Idle,
    End,
}

#[derive(Clone)]
pub(crate) struct FixtureEvent {
    pub(crate) input: String,
    pub(crate) kind: EventKind,
    pub(crate) id: String,
    pub(crate) chunk: Option<u64>,
}

pub(crate) struct Fixture {
    pub(crate) manifest: Value,
    pub(crate) project: crate::ProjectSpec,
    pub(crate) preload: Vec<FixtureEvent>,
    pub(crate) processing: Vec<FixtureEvent>,
    pub(crate) terminal: Vec<FixtureEvent>,
    pub(crate) expected: Vec<RecordBatch>,
    pub(crate) barriers: BTreeMap<String, usize>,
    pub(crate) dispatch_replay: Option<DispatchReplay>,
}

pub(crate) struct DispatchReplay {
    pub(crate) envelope: Value,
    pub(crate) sha256: String,
}

pub(crate) fn invalid(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "benchmark_fixture".into(),
        message: message.into(),
    }
}

/// Observes the running benchmark process's affinity and thread settings.
///
/// # Errors
/// Returns an error when Linux process affinity cannot be read or parsed.
pub fn process_observations() -> Result<Value> {
    #[cfg(target_os = "linux")]
    let cpus = {
        let status = std::fs::read_to_string("/proc/self/status")
            .map_err(|error| invalid(format!("process status: {error}")))?;
        let allowed = status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))
            .ok_or_else(|| invalid("process affinity is unavailable"))?
            .trim();
        let mut cpus = BTreeSet::new();
        for group in allowed.split(',') {
            let (first, last) = group.split_once('-').unwrap_or((group, group));
            let first = first
                .parse::<u32>()
                .map_err(|_| invalid("invalid process affinity"))?;
            let last = last
                .parse::<u32>()
                .map_err(|_| invalid("invalid process affinity"))?;
            if first > last {
                return Err(invalid("invalid process affinity range"));
            }
            cpus.extend(first..=last);
        }
        cpus.into_iter().collect::<Vec<_>>()
    };
    #[cfg(not(target_os = "linux"))]
    let cpus = Vec::<u32>::new();
    let environment = [
        "TOKIO_WORKER_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "RAYON_NUM_THREADS",
        "POLARS_MAX_THREADS",
    ]
    .into_iter()
    .map(|key| (key.to_owned(), json!(std::env::var(key).ok())))
    .collect::<serde_json::Map<_, _>>();
    Ok(
        json!({"observed_cpu_affinity":cpus,"observed_thread_environment":environment,"process_id":std::process::id()}),
    )
}

pub(crate) fn with_process_observations(mut observation: Value) -> Result<Value> {
    observation
        .as_object_mut()
        .ok_or_else(|| invalid("native observation must be an object"))?
        .extend(
            process_observations()?
                .as_object()
                .expect("process observation object")
                .clone(),
        );
    Ok(observation)
}

/// Closes and awaits every gather task owned by a direct benchmark job.
///
/// # Errors
/// Returns an error if any owned task fails while draining.
pub async fn settle_job(job: &crate::StreamJobContext) -> Result<()> {
    let failures = job.gather_owner().close_and_drain().await;
    if failures.is_empty() {
        Ok(())
    } else {
        Err(invalid(format!(
            "owned gather cleanup failures: {failures:?}"
        )))
    }
}

/// Checks the fixed retained Join checkpoint lifecycle with four million quotes.
///
/// This correctness-only diagnostic observes reservations and refunds in the
/// default unbounded runtime pool. It does not report performance measurements.
///
/// # Errors
/// Returns an error if the diagnostic fixture cannot be constructed. Lifecycle
/// failures are retained in the returned data together with cleanup evidence.
pub async fn join_checkpoint_history_4m() -> Result<Value> {
    crate::operator::run_checkpoint_history_diagnostic().await
}

/// Reads a sealed fixture and verifies its project before native compilation.
///
/// # Errors
/// Rejects malformed manifests, unavailable checkpoint modes, unsafe relative
/// paths, unreadable files, and content digest mismatches.
pub fn validate_matched_fixture(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).map_err(|error| invalid(format!("fixture read: {error}")))?;
    let manifest: Value = serde_json::from_slice(&bytes)
        .map_err(|error| invalid(format!("fixture JSON: {error}")))?;
    if manifest["schema"] != "calc-flow.matched-native-fixture.v1" {
        return Err(invalid("unsupported fixture schema"));
    }
    validate_fixture_mode(&manifest)?;
    read_sealed(
        path.parent()
            .ok_or_else(|| invalid("fixture requires a parent directory"))?,
        &manifest["project"],
        "project",
    )?;
    Ok(manifest)
}

fn validate_fixture_mode(manifest: &Value) -> Result<()> {
    let enabled = manifest["checkpoint_enabled"]
        .as_bool()
        .ok_or_else(|| invalid("checkpoint_enabled must be boolean"))?;
    if !enabled && !cfg!(feature = "benchmark-checkpoint-off") {
        return Err(invalid(
            "checkpoint-off capability is unavailable in this binary",
        ));
    }
    Ok(())
}

/// Replays a sealed fixture through the ordinary compiled native operator chain.
///
/// `check_only` admits tiny correctness fixtures. Measurement mode requires the
/// fixed million-row, 1024-row-message workload.
///
/// # Errors
/// Rejects invalid fixture identity, unsupported graph shapes, unavailable
/// checkpoint modes, invalid progress, oracle mismatches, or cleanup failures.
pub async fn run_matched_fixture(path: &Path, check_only: bool) -> Result<Value> {
    let setup_start = Instant::now();
    let path = path.to_owned();
    let fixture = tokio::task::spawn_blocking(move || load_fixture(&path, check_only))
        .await
        .map_err(|error| invalid(format!("fixture loader task: {error}")))??;
    let observation =
        crate::runtime::streaming::benchmark_replay::run(fixture, check_only, setup_start).await?;
    with_process_observations(observation)
}

/// Runs the closed matched-fixture benchmark command.
///
/// # Errors
/// Rejects unsupported arguments before fixture I/O, then propagates replay errors.
pub async fn run_fixture_cli(arguments: &[String]) -> Result<Value> {
    match arguments {
        [flag, rest @ ..] if flag == "--matched-fixture" => run_matched_cli(rest).await,
        [flag, path] if flag == "--prove-ordering" => run_ordering_fixture(path).await,
        _ => Err(invalid_cli()),
    }
}

fn invalid_cli() -> CalcFlowError {
    invalid("matched fixture arguments require --matched-fixture PATH [--check]")
}

async fn run_matched_cli(arguments: &[String]) -> Result<Value> {
    match arguments {
        [path, trace_flag, trace] if trace_flag == "--dispatch-trace" => {
            run_dispatch_fixture(path, trace, false).await
        }
        [path, trace_flag, trace, check]
            if trace_flag == "--dispatch-trace" && check == "--check" =>
        {
            run_dispatch_fixture(path, trace, true).await
        }
        [path] => run_matched_fixture(Path::new(path), false).await,
        [path, check] if check == "--check" => run_matched_fixture(Path::new(path), true).await,
        _ => Err(invalid_cli()),
    }
}

async fn run_ordering_fixture(path: &str) -> Result<Value> {
    let path = std::path::PathBuf::from(path);
    let fixture = tokio::task::spawn_blocking(move || load_fixture(&path, true))
        .await
        .map_err(|error| invalid(format!("fixture loader task: {error}")))??;
    with_process_observations(
        crate::runtime::streaming::benchmark_replay::prove_ordering(fixture).await?,
    )
}

async fn run_dispatch_fixture(path: &str, trace: &str, check_only: bool) -> Result<Value> {
    let setup_start = Instant::now();
    let path = std::path::PathBuf::from(path);
    let trace = std::path::PathBuf::from(trace);
    let fixture = tokio::task::spawn_blocking(move || {
        let mut fixture = load_fixture(&path, check_only)?;
        let bytes = std::fs::read(trace)
            .map_err(|error| invalid(format!("dispatch envelope read: {error}")))?;
        let envelope = serde_json::from_slice(&bytes)
            .map_err(|error| invalid(format!("dispatch envelope JSON: {error}")))?;
        fixture.dispatch_replay = Some(DispatchReplay {
            envelope,
            sha256: hex::encode(Sha256::digest(&bytes)),
        });
        Ok::<_, CalcFlowError>(fixture)
    })
    .await
    .map_err(|error| invalid(format!("dispatch fixture loader: {error}")))??;
    with_process_observations(
        crate::runtime::streaming::benchmark_replay::run(fixture, check_only, setup_start).await?,
    )
}

pub(crate) fn fingerprint(value: &Value) -> Result<String> {
    Ok(hex::encode(Sha256::digest(
        crate::canonical_json(value)?.as_bytes(),
    )))
}

fn load_fixture(path: &Path, check_only: bool) -> Result<Fixture> {
    let manifest = validate_matched_fixture(path)?;
    let root = path
        .parent()
        .ok_or_else(|| invalid("fixture requires a parent"))?;
    let geometry = fixture_geometry(&manifest, check_only)?;
    load_fixture_contents(root, manifest, geometry)
}

#[derive(Clone, Copy)]
struct FixtureGeometry {
    join: bool,
    rows: u64,
    batch_rows: u64,
}

fn fixed_workload_mismatch(rows: u64, batch_rows: u64, check_only: bool) -> bool {
    rows == 0 || batch_rows == 0 || (!check_only && (rows != 1_000_000 || batch_rows != 1024))
}

fn fixture_row_count(manifest: &Value, join: bool) -> Result<u64> {
    manifest[if join { "rows" } else { "rows_per_side" }]
        .as_u64()
        .ok_or_else(|| invalid("rows_per_side must be positive"))
}

fn fixture_geometry(manifest: &Value, check_only: bool) -> Result<FixtureGeometry> {
    let join = manifest["scenario"] == "join_sealed";
    if !join && manifest["scenario"] != "asof_join" {
        return Err(invalid("unsupported matched fixture scenario"));
    }
    let rows = fixture_row_count(manifest, join)?;
    let batch_rows = manifest["batch_rows"]
        .as_u64()
        .ok_or_else(|| invalid("batch_rows must be positive"))?;
    if fixed_workload_mismatch(rows, batch_rows, check_only) {
        return Err(invalid("fixture does not match the fixed workload size"));
    }
    Ok(FixtureGeometry {
        join,
        rows,
        batch_rows,
    })
}

fn load_events(root: &Path, manifest: &Value) -> Result<Value> {
    let events: Value = serde_json::from_slice(&read_sealed(root, &manifest["events"], "events")?)
        .map_err(|error| invalid(format!("events JSON: {error}")))?;
    if manifest["canonical_native_events_sha256"].as_str() != Some(fingerprint(&events)?.as_str()) {
        return Err(invalid("canonical native events SHA256 mismatch"));
    }
    Ok(events)
}

fn load_fixture_contents(
    root: &Path,
    manifest: Value,
    geometry: FixtureGeometry,
) -> Result<Fixture> {
    let FixtureGeometry {
        rows,
        batch_rows,
        join,
    } = geometry;
    let project = crate::import_project_json(&read_sealed(root, &manifest["project"], "project")?)?;
    let inputs = load_inputs(root, &manifest, rows, batch_rows, join)?;
    let expected = load_expected(root, &manifest, rows)?;
    let events = load_events(root, &manifest)?;
    assemble_fixture(manifest, project, &inputs, expected, &events, geometry)
}

fn assemble_fixture(
    manifest: Value,
    project: crate::ProjectSpec,
    inputs: &BTreeMap<String, Vec<RecordBatch>>,
    expected: Vec<RecordBatch>,
    events: &Value,
    geometry: FixtureGeometry,
) -> Result<Fixture> {
    let FixtureGeometry {
        rows,
        batch_rows,
        join,
    } = geometry;
    let mut phases = prepare_events(&manifest, events, inputs, batch_rows, join)?;
    let processing = phases.remove("processing").expect("parsed phase");
    let barriers = prepare_barriers(events, &processing, rows, batch_rows, join)?;
    let preload = phases.remove("preload").expect("parsed phase");
    validate_preload(&preload, join)?;
    let terminal = phases.remove("terminal").expect("parsed phase");
    validate_terminal(&terminal, inputs)?;
    Ok(Fixture {
        manifest,
        project,
        preload,
        processing,
        terminal,
        expected,
        barriers,
        dispatch_replay: None,
    })
}

fn validate_preload(preload: &[FixtureEvent], join: bool) -> Result<()> {
    if join
        && (preload.len() != 2
            || preload.iter().any(|event| event.input != "right")
            || !matches!(preload[0].kind, EventKind::Data(_))
            || !matches!(preload[1].kind, EventKind::Watermark(_)))
    {
        return Err(invalid(
            "static Join preload requires right Data then sealed Watermark",
        ));
    }
    Ok(())
}

fn validate_terminal(
    terminal: &[FixtureEvent],
    inputs: &BTreeMap<String, Vec<RecordBatch>>,
) -> Result<()> {
    if terminal.len() != inputs.len()
        || terminal
            .iter()
            .map(|event| event.input.clone())
            .collect::<BTreeSet<_>>()
            != inputs.keys().cloned().collect()
    {
        return Err(invalid("every fixture input must end exactly once"));
    }
    Ok(())
}

fn load_inputs(
    root: &Path,
    manifest: &Value,
    rows: u64,
    batch_rows: u64,
    join: bool,
) -> Result<BTreeMap<String, Vec<RecordBatch>>> {
    let records = input_records(manifest)?;
    let mut inputs = BTreeMap::new();
    let mut input_identity = serde_json::Map::new();
    for (name, record) in records {
        let input_rows = input_row_count(manifest, name, rows, batch_rows, join)?;
        let batches = load_input(root, record, name, input_rows)?;
        input_identity.insert(
            name.clone(),
            json!({"sha256":record["sha256"],"rows":record["rows"]}),
        );
        inputs.insert(name.clone(), batches);
    }
    if manifest["inputs_sha256"].as_str() != Some(fingerprint(&input_identity.into())?.as_str()) {
        return Err(invalid("input identity SHA256 mismatch"));
    }
    Ok(inputs)
}

fn input_records(manifest: &Value) -> Result<&serde_json::Map<String, Value>> {
    let records = manifest["inputs"]
        .as_object()
        .ok_or_else(|| invalid("fixture inputs must be an object"))?;
    if records.len() != 2 {
        return Err(invalid("ASOF fixture requires exactly two inputs"));
    }
    Ok(records)
}

fn input_row_count(
    manifest: &Value,
    name: &str,
    rows: u64,
    batch_rows: u64,
    join: bool,
) -> Result<u64> {
    if join && name == "right" {
        manifest["dimension_rows"]
            .as_u64()
            .filter(|rows| *rows > 0 && *rows <= batch_rows)
            .ok_or_else(|| invalid("invalid static dimension row count"))
    } else {
        Ok(rows)
    }
}

fn load_input(
    root: &Path,
    record: &Value,
    name: &str,
    input_rows: u64,
) -> Result<Vec<RecordBatch>> {
    let batches = read_ipc(&read_sealed(root, record, name)?)?;
    if batches
        .iter()
        .map(|batch| batch.num_rows() as u64)
        .sum::<u64>()
        != input_rows
        || record["rows"].as_u64() != Some(input_rows)
    {
        return Err(invalid(format!("input {name} has inconsistent row count")));
    }
    Ok(batches)
}

fn load_expected(root: &Path, manifest: &Value, rows: u64) -> Result<Vec<RecordBatch>> {
    let expected = read_ipc(&read_sealed(
        root,
        &manifest["expected_output"],
        "expected_output",
    )?)?;
    if expected
        .iter()
        .map(|batch| batch.num_rows() as u64)
        .sum::<u64>()
        != rows
        || manifest["expected_output"]["rows"].as_u64() != Some(rows)
    {
        return Err(invalid("expected output row count mismatch"));
    }
    Ok(expected)
}

fn prepare_events(
    manifest: &Value,
    events: &Value,
    inputs: &BTreeMap<String, Vec<RecordBatch>>,
    batch_rows: u64,
    join: bool,
) -> Result<BTreeMap<&'static str, Vec<FixtureEvent>>> {
    let mut partial = serde_json::Map::new();
    let mut preparation = EventPreparation {
        inputs,
        batch_rows,
        join,
        seen: BTreeSet::new(),
        ids: BTreeSet::new(),
    };
    let mut phases = BTreeMap::new();
    for phase in ["preload", "processing", "terminal"] {
        let (grouped, prepared) = preparation.phase(events, phase)?;
        partial.insert(phase.into(), grouped);
        phases.insert(phase, prepared);
    }
    let expected_indices = inputs
        .iter()
        .flat_map(|(name, batches)| (0..batches.len()).map(move |index| (name.clone(), index)))
        .collect::<BTreeSet<_>>();
    if preparation.seen != expected_indices {
        return Err(invalid("fixture omits input record batches"));
    }
    partial.insert("barriers".into(), events["barriers"].clone());
    if manifest["shared_partial_order_sha256"].as_str()
        != Some(fingerprint(&partial.into())?.as_str())
    {
        return Err(invalid("shared partial order SHA256 mismatch"));
    }
    Ok(phases)
}

struct EventPreparation<'a> {
    inputs: &'a BTreeMap<String, Vec<RecordBatch>>,
    batch_rows: u64,
    join: bool,
    seen: BTreeSet<(String, usize)>,
    ids: BTreeSet<String>,
}

impl EventPreparation<'_> {
    fn phase(&mut self, events: &Value, phase: &str) -> Result<(Value, Vec<FixtureEvent>)> {
        let raw = events[phase]
            .as_array()
            .ok_or_else(|| invalid(format!("missing {phase} events")))?;
        if !self.join && phase == "preload" && !raw.is_empty() {
            return Err(invalid("ASOF preloading is outside the matched contract"));
        }
        let mut grouped = serde_json::Map::new();
        let mut prepared = Vec::new();
        for event in raw {
            prepared.push(self.event(event, phase, &mut grouped)?);
        }
        Ok((grouped.into(), prepared))
    }

    fn identity<'a>(&mut self, event: &'a Value) -> Result<(&'a str, &'a str)> {
        let input = event["input"]
            .as_str()
            .filter(|input| self.inputs.contains_key(*input))
            .ok_or_else(|| invalid("unknown fixture input"))?;
        let id = event["event_id"]
            .as_str()
            .ok_or_else(|| invalid("missing event identity"))?;
        if !self.ids.insert(id.to_owned()) {
            return Err(invalid("duplicate event identity"));
        }
        Ok((input, id))
    }

    fn event(
        &mut self,
        event: &Value,
        phase: &str,
        grouped: &mut serde_json::Map<String, Value>,
    ) -> Result<FixtureEvent> {
        let (input, id) = self.identity(event)?;
        grouped
            .entry(input.to_owned())
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .expect("created array")
            .push(event.clone());
        let kind = prepare_kind(
            event,
            input,
            phase,
            self.inputs,
            &mut self.seen,
            self.batch_rows,
            self.join,
        )?;
        let chunk = event["chunk_id"].as_u64();
        if phase == "processing" && chunk.is_none() {
            return Err(invalid("processing event has no chunk identity"));
        }
        Ok(FixtureEvent {
            input: input.to_owned(),
            kind,
            id: id.to_owned(),
            chunk,
        })
    }
}

fn data_record(
    index: &Value,
    input: &str,
    inputs: &BTreeMap<String, Vec<RecordBatch>>,
    seen: &mut BTreeSet<(String, usize)>,
) -> Result<RecordBatch> {
    let index = index
        .as_u64()
        .and_then(|index| usize::try_from(index).ok())
        .ok_or_else(|| invalid("invalid data batch index"))?;
    if !seen.insert((input.to_owned(), index)) {
        return Err(invalid("duplicate input record batch"));
    }
    Ok(inputs[input]
        .get(index)
        .ok_or_else(|| invalid("data batch index out of range"))?
        .clone())
}

fn data_group(
    event: &Value,
    input: &str,
    inputs: &BTreeMap<String, Vec<RecordBatch>>,
    seen: &mut BTreeSet<(String, usize)>,
) -> Result<Vec<RecordBatch>> {
    let indices = event["batches"]
        .as_array()
        .filter(|indices| !indices.is_empty())
        .ok_or_else(|| invalid("missing data batch group"))?;
    let mut group = Vec::new();
    for index in indices {
        group.push(data_record(index, input, inputs, seen)?);
    }
    Ok(group)
}

fn prepare_data(
    event: &Value,
    input: &str,
    inputs: &BTreeMap<String, Vec<RecordBatch>>,
    seen: &mut BTreeSet<(String, usize)>,
    batch_rows: u64,
) -> Result<EventKind> {
    let group = data_group(event, input, inputs, seen)?;
    let batch = crate::Batch::table(group, crate::BatchMetadata::default())?;
    if batch.num_rows() == 0 || batch.num_rows() as u64 > batch_rows {
        return Err(invalid("data group exceeds fixed message rows"));
    }
    Ok(EventKind::Data(batch))
}

fn is_data_phase(phase: &str, join: bool) -> bool {
    phase == "processing" || (join && phase == "preload")
}

fn prepare_kind(
    event: &Value,
    input: &str,
    phase: &str,
    inputs: &BTreeMap<String, Vec<RecordBatch>>,
    seen: &mut BTreeSet<(String, usize)>,
    batch_rows: u64,
    join: bool,
) -> Result<EventKind> {
    Ok(match event["kind"].as_str() {
        Some("data") if is_data_phase(phase, join) => {
            prepare_data(event, input, inputs, seen, batch_rows)?
        }
        Some("watermark") if is_data_phase(phase, join) => {
            EventKind::Watermark(crate::EventTime::from_micros(
                event["micros"]
                    .as_i64()
                    .ok_or_else(|| invalid("invalid watermark"))?,
            ))
        }
        Some("end") if phase == "terminal" => EventKind::End,
        _ => return Err(invalid("control event is outside its approved phase")),
    })
}

fn prepare_barriers(
    events: &Value,
    processing: &[FixtureEvent],
    rows: u64,
    batch_rows: u64,
    join: bool,
) -> Result<BTreeMap<String, usize>> {
    let mut coverage = RendezvousCoverage {
        barriers: BTreeMap::new(),
        covered: BTreeSet::new(),
        row_count: usize::try_from(rows).map_err(|_| invalid("rows exceed platform capacity"))?,
        previous_rows: 0,
    };
    for (chunk_index, barrier) in events["barriers"]
        .as_array()
        .ok_or_else(|| invalid("missing rendezvous barriers"))?
        .iter()
        .enumerate()
    {
        prepare_barrier(
            barrier,
            processing,
            chunk_index,
            FixtureGeometry {
                join,
                rows,
                batch_rows,
            },
            &mut coverage,
        )?;
    }
    if coverage.previous_rows != coverage.row_count
        || coverage.covered != processing.iter().map(|event| event.id.clone()).collect()
    {
        return Err(invalid("incomplete rendezvous coverage"));
    }
    Ok(coverage.barriers)
}

fn barrier_event_ids(barrier: &Value, join: bool) -> Result<(u64, Vec<String>)> {
    let chunk = barrier["chunk_id"]
        .as_u64()
        .ok_or_else(|| invalid("invalid barrier chunk"))?;
    let after = barrier["after_events"]
        .as_array()
        .filter(|after| after.len() == if join { 2 } else { 4 })
        .ok_or_else(|| invalid("barrier requires two data and two watermark events"))?;
    let ids = after
        .iter()
        .map(|id| {
            id.as_str()
                .map(str::to_owned)
                .ok_or_else(|| invalid("invalid barrier event identity"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((chunk, ids))
}

fn select_barrier_chunk<'a>(
    processing: &'a [FixtureEvent],
    chunk: u64,
    chunk_index: usize,
    ids: &[String],
    join: bool,
) -> Result<Vec<&'a FixtureEvent>> {
    let selected = processing
        .iter()
        .filter(|event| event.chunk == Some(chunk))
        .collect::<Vec<_>>();
    if selected.len() != (if join { 2 } else { 4 })
        || selected.iter().map(|event| &event.id).ne(ids.iter())
    {
        return Err(invalid("barrier does not follow its four canonical events"));
    }
    let width = if join { 2 } else { 4 };
    if chunk != chunk_index as u64
        || processing
            .iter()
            .skip(chunk_index * width)
            .take(width)
            .map(|event| &event.id)
            .ne(ids.iter())
    {
        return Err(invalid("rendezvous chunks must be contiguous"));
    }
    Ok(selected)
}

fn prepare_barrier(
    barrier: &Value,
    processing: &[FixtureEvent],
    chunk_index: usize,
    geometry: FixtureGeometry,
    coverage: &mut RendezvousCoverage,
) -> Result<()> {
    let FixtureGeometry {
        rows,
        batch_rows,
        join,
    } = geometry;
    let (chunk, ids) = barrier_event_ids(barrier, join)?;
    let selected = select_barrier_chunk(processing, chunk, chunk_index, &ids, join)?;
    let remaining = rows.saturating_sub(chunk * batch_rows);
    let expected_rows = usize::try_from(remaining.min(batch_rows))
        .map_err(|_| invalid("batch size exceeds platform capacity"))?;
    validate_chunk(&selected, join, expected_rows)?;
    coverage.add(&ids, barrier, expected_rows)
}

struct RendezvousCoverage {
    barriers: BTreeMap<String, usize>,
    covered: BTreeSet<String>,
    row_count: usize,
    previous_rows: usize,
}

impl RendezvousCoverage {
    fn add(&mut self, ids: &[String], barrier: &Value, expected_rows: usize) -> Result<()> {
        for id in ids {
            if !self.covered.insert(id.clone()) {
                return Err(invalid("overlapping rendezvous barriers"));
            }
        }
        let output_rows = barrier["output_rows"]
            .as_u64()
            .and_then(|rows| usize::try_from(rows).ok())
            .ok_or_else(|| invalid("invalid rendezvous output count"))?;
        if output_rows != self.previous_rows + expected_rows || output_rows > self.row_count {
            return Err(invalid("invalid rendezvous row progression"));
        }
        self.previous_rows = output_rows;
        self.barriers
            .insert(ids.last().expect("four ids").clone(), output_rows);
        Ok(())
    }
}

fn validate_chunk(selected: &[&FixtureEvent], join: bool, expected_rows: usize) -> Result<()> {
    let mut by_input = BTreeMap::<&str, Vec<&FixtureEvent>>::new();
    for event in selected {
        by_input.entry(&event.input).or_default().push(event);
    }
    validate_chunk_inputs(&by_input, join)?;
    if expected_rows == 0 || by_input.values().any(|group| !matches!(&group[0].kind, EventKind::Data(batch) if batch.num_rows() == expected_rows)) {
            return Err(invalid("processing message geometry differs from full chunks and final tail"));
        }
    Ok(())
}

fn invalid_chunk_input(group: &[&FixtureEvent]) -> bool {
    group.len() != 2
        || !matches!(group[0].kind, EventKind::Data(_))
        || !matches!(group[1].kind, EventKind::Watermark(_))
}

fn validate_chunk_inputs(by_input: &BTreeMap<&str, Vec<&FixtureEvent>>, join: bool) -> Result<()> {
    if by_input.len() != (if join { 1 } else { 2 })
        || (join && !by_input.contains_key("left"))
        || by_input.values().any(|group| invalid_chunk_input(group))
    {
        return Err(invalid(
            "each processing input requires Data then Watermark",
        ));
    }
    Ok(())
}

fn read_ipc(bytes: &[u8]) -> Result<Vec<RecordBatch>> {
    FileReader::try_new(Cursor::new(bytes), None)
        .map_err(|error| invalid(format!("Arrow IPC: {error}")))?
        .map(|batch| batch.map_err(|error| invalid(format!("Arrow IPC batch: {error}"))))
        .collect()
}

pub(crate) fn read_sealed(root: &Path, record: &Value, name: &str) -> Result<Vec<u8>> {
    let relative = Path::new(
        record["path"]
            .as_str()
            .ok_or_else(|| invalid(format!("missing {name} path")))?,
    );
    if relative
        .components()
        .any(|part| !matches!(part, Component::Normal(_)))
        || relative.as_os_str().is_empty()
    {
        return Err(invalid(format!("unsafe {name} path")));
    }
    let bytes = std::fs::read(root.join(relative))
        .map_err(|error| invalid(format!("{name} read: {error}")))?;
    let digest = hex::encode(Sha256::digest(&bytes));
    if record["sha256"].as_str() != Some(digest.as_str()) {
        return Err(invalid(format!("{name} SHA256 mismatch")));
    }
    Ok(bytes)
}
