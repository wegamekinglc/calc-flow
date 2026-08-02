use std::{
    any::Any,
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fmt::{self, Display},
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    thread,
};

use allocation_counter::{AllocationInfo, measure};
use async_trait::async_trait;
use calc_flow::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, DataFusionConfig, Edge, ExecutionOptions,
    ExpressionOperator, ExternalPayload, JsonMap, Operator, OperatorContext, PipelineBuilder, Port,
    PortEndpoint, RunResult, SqlOperator, UdfRegistry,
};
use datafusion::arrow::{array::Int64Array, record_batch::RecordBatch};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const FIXED_BASELINE_SHA: &str = "2ac7e97c1549baf0e97849d5823f65e7dd298e99";
const ALLOCATION_COUNTER_CHECKSUM: &str =
    "beb9e990c0a33699f1984d85a6abead615ccc72dd8130bf3e15dcabe2ca149c9";
const SCHEMA_VERSION: u32 = 1;
const FIXED_WARMUP_DISPATCHES: u64 = 1_000;
const FIXED_MEASURED_DISPATCHES: u64 = 10_000;
const FIXED_REPETITIONS: usize = 10;
const FROZEN_FILES: [&str; 3] = [
    "Cargo.lock",
    "crates/calc-flow/Cargo.toml",
    "crates/calc-flow/benches/allocation_regression.rs",
];
const DATAFUSION_CONFIG: DataFusionConfig = DataFusionConfig {
    batch_size: 1_024,
    target_partitions: 1,
};

#[derive(Debug)]
struct HarnessError(String);

impl Display for HarnessError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for HarnessError {}

type HarnessResult<T> = Result<T, HarnessError>;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Role {
    Baseline,
    Candidate,
}

impl Role {
    fn parse(value: &str) -> HarnessResult<Self> {
        match value {
            "baseline" => Ok(Self::Baseline),
            "candidate" => Ok(Self::Candidate),
            _ => Err(HarnessError(format!(
                "--role must be baseline or candidate, got {value:?}"
            ))),
        }
    }
}

#[derive(Debug)]
enum Invocation {
    Measure(MeasureOptions),
    Compare {
        baseline: PathBuf,
        candidate: PathBuf,
    },
    ValidateNoiseFloor {
        before_relative_mad_percent: f64,
        after_relative_mad_percent: f64,
        declared_noise_floor_percent: f64,
    },
}

#[derive(Debug)]
struct MeasureOptions {
    warmup_dispatches: u64,
    measured_dispatches: u64,
    repetitions: usize,
    role: Role,
    output: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct FrozenFileEvidence {
    cargo_lock_sha256: String,
    crate_manifest_sha256: String,
    benchmark_sha256: String,
    allocation_counter_version: String,
    allocation_counter_registry_checksum: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ToolchainEvidence {
    rustc_vv: String,
    cargo_version: String,
    host: String,
    target: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PowerSupplyEvidence {
    name: String,
    kind: String,
    online: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct EnvironmentEvidence {
    uname_a: String,
    lscpu: String,
    cpu_governors: Vec<String>,
    power_supplies: Vec<PowerSupplyEvidence>,
    virtualization: String,
    container_markers: String,
    background_load_policy: String,
    cargo_build_jobs: String,
    calc_flow_benchmark_scale: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MeasurementThreadEvidence {
    id: String,
    #[serde(deserialize_with = "deserialize_required_option")]
    name: Option<String>,
    current_thread_runtime: bool,
    unchanged_for_all_measurements: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct RawAllocationInfo {
    count_total: u64,
    count_current: i64,
    count_max: u64,
    bytes_total: u64,
    bytes_current: i64,
    bytes_max: u64,
}

impl From<AllocationInfo> for RawAllocationInfo {
    fn from(info: AllocationInfo) -> Self {
        Self {
            count_total: info.count_total,
            count_current: info.count_current,
            count_max: info.count_max,
            bytes_total: info.bytes_total,
            bytes_current: info.bytes_current,
            bytes_max: info.bytes_max,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[expect(
    clippy::struct_field_names,
    reason = "the JSON contract names all four normalized fields by their denominator"
)]
struct NormalizedAllocationInfo {
    calls_per_dispatch: f64,
    bytes_per_dispatch: f64,
    calls_per_node_dispatch: f64,
    bytes_per_node_dispatch: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RepetitionReport {
    repetition_index: usize,
    requested_dispatches: u64,
    completed_dispatches: u64,
    measurement_thread_id: String,
    on_expected_thread: bool,
    raw: RawAllocationInfo,
    normalized: NormalizedAllocationInfo,
    output_assertion_passed: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    invalid_reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "the evidence schema records independent assertion and stability results"
)]
struct CaseReport {
    name: String,
    payload: String,
    workload_fingerprint: String,
    plan_fingerprint: String,
    compiled_node_count: usize,
    compiled_operator_variants: Vec<String>,
    configured_datafusion: DataFusionConfig,
    #[serde(deserialize_with = "deserialize_required_option")]
    compiled_datafusion: Option<DataFusionConfig>,
    requires_datafusion: bool,
    metric_assertion: String,
    output_assertion: String,
    warmup_dispatches: u64,
    requested_dispatches: u64,
    repetitions_requested: usize,
    repetitions: Vec<RepetitionReport>,
    stable_count_total: bool,
    stable_bytes_total: bool,
    valid: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    invalid_reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AllocationReport {
    schema_version: u32,
    role: Role,
    valid: bool,
    #[serde(deserialize_with = "deserialize_required_option")]
    invalid_reason: Option<String>,
    product_sha: String,
    fixed_baseline_sha: String,
    harness_commit_sha: String,
    frozen_files: FrozenFileEvidence,
    git_status_short: String,
    toolchain: ToolchainEvidence,
    environment: EnvironmentEvidence,
    measurement_thread: MeasurementThreadEvidence,
    cases: Vec<CaseReport>,
}

#[derive(Debug, Serialize)]
struct ComparisonRepetition {
    repetition_index: usize,
    baseline_count_total: u64,
    candidate_count_total: u64,
    count_delta: i64,
    baseline_bytes_total: u64,
    candidate_bytes_total: u64,
    bytes_delta: i64,
    calls_per_dispatch_delta: f64,
    bytes_per_dispatch_delta: f64,
    calls_per_node_dispatch_delta: f64,
    bytes_per_node_dispatch_delta: f64,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct ComparisonCase {
    name: String,
    repetitions: Vec<ComparisonRepetition>,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct ComparisonReport {
    valid: bool,
    passed: bool,
    invalid_reason: Option<String>,
    baseline_product_sha: String,
    candidate_product_sha: String,
    harness_commit_sha: String,
    cases: Vec<ComparisonCase>,
}

#[derive(Debug)]
struct BenchmarkExternalPayload {
    rows: usize,
}

impl ExternalPayload for BenchmarkExternalPayload {
    fn backend(&self) -> &'static str {
        "allocation-regression-benchmark"
    }

    fn len(&self) -> usize {
        self.rows
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct PassthroughOperator {
    inputs: [Port; 1],
    outputs: [Port; 1],
}

impl PassthroughOperator {
    fn new(kind: BatchKind) -> HarnessResult<Self> {
        Ok(Self {
            inputs: [Port::new("input", kind, true, None).map_err(harness_error)?],
            outputs: [Port::new("output", kind, true, None).map_err(harness_error)?],
        })
    }
}

#[async_trait]
impl Operator for PassthroughOperator {
    fn name(&self) -> &'static str {
        "allocation-regression-passthrough"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        BTreeMap::new()
    }

    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _context: &OperatorContext<'_>,
    ) -> calc_flow::Result<BTreeMap<String, Batch>> {
        let output = inputs
            .get("input")
            .cloned()
            .ok_or_else(|| CalcFlowError::Internal {
                message: "allocation-regression passthrough input is missing".into(),
            })?;
        Ok(BTreeMap::from([("output".into(), output)]))
    }
}

#[derive(Clone, Copy)]
enum OutputExpectation {
    ExternalPayloadIdentity,
    TableIdentity,
    ExpressionValues,
    SqlValues,
}

impl OutputExpectation {
    const fn description(self) -> &'static str {
        match self {
            Self::ExternalPayloadIdentity => "single terminal external payload Arc::ptr_eq",
            Self::TableIdentity => "terminal schema, array, and value-buffer pointer identity",
            Self::ExpressionValues => "single plus_one Int64 column equals 1..64",
            Self::SqlValues => "single doubled Int64 column equals 0,2,..126",
        }
    }
}

struct PreparedCase {
    name: &'static str,
    payload: &'static str,
    logical_variants: &'static [&'static str],
    expected_node_count: usize,
    expected_metric_node: Option<&'static str>,
    expectation: OutputExpectation,
    plan: calc_flow::ExecutionPlan,
    input_name: String,
    terminal_outputs: Vec<String>,
    input: Batch,
    workload_fingerprint: String,
}

impl PreparedCase {
    fn finish(
        name: &'static str,
        payload: &'static str,
        logical_variants: &'static [&'static str],
        expected_metric_node: Option<&'static str>,
        expectation: OutputExpectation,
        plan: calc_flow::ExecutionPlan,
        input: Batch,
    ) -> HarnessResult<Self> {
        let expected_node_count = logical_variants.len();
        let actual_node_count = plan.topological_order().len();
        if actual_node_count != expected_node_count {
            return Err(HarnessError(format!(
                "{name}: expected {expected_node_count} compiled nodes, got {actual_node_count}"
            )));
        }
        let expected_datafusion = expected_metric_node.is_some();
        if plan.requires_datafusion() != expected_datafusion {
            return Err(HarnessError(format!(
                "{name}: requires_datafusion mismatch"
            )));
        }
        if plan.datafusion_config() != expected_datafusion.then_some(DATAFUSION_CONFIG) {
            return Err(HarnessError(format!(
                "{name}: compiled DataFusion configuration mismatch"
            )));
        }
        let input_names = plan.external_inputs().keys().cloned().collect::<Vec<_>>();
        if input_names.len() != 1 {
            return Err(HarnessError(format!(
                "{name}: expected one external input, got {input_names:?}"
            )));
        }
        let terminal_outputs = plan.external_outputs().keys().cloned().collect::<Vec<_>>();
        let expected_outputs = match expectation {
            OutputExpectation::TableIdentity if expected_node_count == 4 => 3,
            _ => 1,
        };
        if terminal_outputs.len() != expected_outputs {
            return Err(HarnessError(format!(
                "{name}: expected {expected_outputs} terminal outputs, got {terminal_outputs:?}"
            )));
        }
        let workload_fingerprint = workload_fingerprint(
            name,
            payload,
            logical_variants,
            actual_node_count,
            expected_metric_node,
            expectation.description(),
            plan.fingerprint(),
        )?;
        Ok(Self {
            name,
            payload,
            logical_variants,
            expected_node_count,
            expected_metric_node,
            expectation,
            plan,
            input_name: input_names.into_iter().next().expect("length checked"),
            terminal_outputs,
            input,
            workload_fingerprint,
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum FailureCode {
    WrongThread,
    DispatchError,
    MetricAssertion,
    OutputAssertion,
}

impl FailureCode {
    const fn description(self) -> &'static str {
        match self {
            Self::WrongThread => "measurement future moved off the calling thread",
            Self::DispatchError => "ExecutionPlan::execute returned an error",
            Self::MetricAssertion => "DataFusion metric assertion failed",
            Self::OutputAssertion => "output value or identity assertion failed",
        }
    }
}

fn main() {
    #[cfg(test)]
    if env::args_os().len() == 1 {
        if let Err(error) = run_regression_tests() {
            eprintln!("allocation_regression regression tests: {error}");
            std::process::exit(1);
        }
        return;
    }

    if let Err(error) = run() {
        eprintln!("allocation_regression: {error}");
        std::process::exit(1);
    }
}

fn run() -> HarnessResult<()> {
    match parse_invocation(env::args().skip(1))? {
        Invocation::Measure(options) => run_measurement(&options),
        Invocation::Compare {
            baseline,
            candidate,
        } => compare_reports(&baseline, &candidate),
        Invocation::ValidateNoiseFloor {
            before_relative_mad_percent,
            after_relative_mad_percent,
            declared_noise_floor_percent,
        } => validate_timing_noise_floor(
            before_relative_mad_percent,
            after_relative_mad_percent,
            declared_noise_floor_percent,
        ),
    }
}

fn parse_invocation(arguments: impl Iterator<Item = String>) -> HarnessResult<Invocation> {
    let arguments = arguments
        .filter(|argument| argument != "--bench")
        .collect::<Vec<_>>();
    if arguments.first().map(String::as_str) == Some("--compare") {
        if arguments.len() != 3 {
            return Err(HarnessError(
                "--compare requires exactly BASELINE_JSON CANDIDATE_JSON".into(),
            ));
        }
        return Ok(Invocation::Compare {
            baseline: arguments[1].clone().into(),
            candidate: arguments[2].clone().into(),
        });
    }
    if arguments.first().map(String::as_str) == Some("--validate-noise-floor") {
        if arguments.len() != 4 {
            return Err(HarnessError(
                "--validate-noise-floor requires exactly BEFORE_RELATIVE_MAD_PERCENT AFTER_RELATIVE_MAD_PERCENT DECLARED_NOISE_FLOOR_PERCENT".into(),
            ));
        }
        return Ok(Invocation::ValidateNoiseFloor {
            before_relative_mad_percent: parse_nonnegative_f64(
                "BEFORE_RELATIVE_MAD_PERCENT",
                &arguments[1],
            )?,
            after_relative_mad_percent: parse_nonnegative_f64(
                "AFTER_RELATIVE_MAD_PERCENT",
                &arguments[2],
            )?,
            declared_noise_floor_percent: parse_nonnegative_f64(
                "DECLARED_NOISE_FLOOR_PERCENT",
                &arguments[3],
            )?,
        });
    }

    let mut warmup_dispatches = None;
    let mut measured_dispatches = None;
    let mut repetitions = None;
    let mut role = None;
    let mut output = None;
    let mut cases = None;
    let mut index = 0;
    while index < arguments.len() {
        let flag = arguments[index].as_str();
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| HarnessError(format!("{flag} requires a value")))?;
        match flag {
            "--warmup-dispatches" => {
                warmup_dispatches = Some(parse_exact_u64(flag, value, FIXED_WARMUP_DISPATCHES)?);
            }
            "--measured-dispatches" => {
                measured_dispatches =
                    Some(parse_exact_u64(flag, value, FIXED_MEASURED_DISPATCHES)?);
            }
            "--repetitions" => {
                repetitions = Some(parse_exact_usize(flag, value, FIXED_REPETITIONS)?);
            }
            "--cases" => cases = Some(value.clone()),
            "--role" => role = Some(Role::parse(value)?),
            "--output" => output = Some(PathBuf::from(value)),
            _ => return Err(HarnessError(format!("unknown argument {flag:?}"))),
        }
        index += 2;
    }
    if cases.as_deref() != Some("all-existing-data") {
        return Err(HarnessError(
            "--cases must be exactly all-existing-data".into(),
        ));
    }
    Ok(Invocation::Measure(MeasureOptions {
        warmup_dispatches: require_option(warmup_dispatches, "--warmup-dispatches")?,
        measured_dispatches: require_option(measured_dispatches, "--measured-dispatches")?,
        repetitions: require_option(repetitions, "--repetitions")?,
        role: require_option(role, "--role")?,
        output: require_option(output, "--output")?,
    }))
}

fn parse_exact_u64(flag: &str, value: &str, expected: u64) -> HarnessResult<u64> {
    let parsed = value
        .parse::<u64>()
        .map_err(|error| HarnessError(format!("{flag} must be an integer: {error}")))?;
    if parsed != expected {
        return Err(HarnessError(format!(
            "{flag} must be exactly {expected}, got {parsed}"
        )));
    }
    Ok(parsed)
}

fn parse_exact_usize(flag: &str, value: &str, expected: usize) -> HarnessResult<usize> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| HarnessError(format!("{flag} must be an integer: {error}")))?;
    if parsed != expected {
        return Err(HarnessError(format!(
            "{flag} must be exactly {expected}, got {parsed}"
        )));
    }
    Ok(parsed)
}

fn parse_nonnegative_f64(name: &str, value: &str) -> HarnessResult<f64> {
    let parsed = value
        .parse::<f64>()
        .map_err(|error| HarnessError(format!("{name} must be a number: {error}")))?;
    if !parsed.is_finite() || parsed < 0.0 {
        return Err(HarnessError(format!(
            "{name} must be finite and non-negative, got {parsed}"
        )));
    }
    Ok(parsed)
}

fn validate_timing_noise_floor(
    before_relative_mad_percent: f64,
    after_relative_mad_percent: f64,
    declared_noise_floor_percent: f64,
) -> HarnessResult<()> {
    let expected = 1.0_f64.max(2.0 * before_relative_mad_percent.max(after_relative_mad_percent));
    if declared_noise_floor_percent.to_bits() != expected.to_bits() {
        return Err(HarnessError(format!(
            "timing noise floor must be max(1%, 2 * max(before relative MAD, after relative MAD)) = {expected}%, got {declared_noise_floor_percent}%"
        )));
    }
    println!(
        "validated timing noise floor: max(1%, 2 * max({before_relative_mad_percent}%, {after_relative_mad_percent}%)) = {expected}%"
    );
    Ok(())
}

fn require_option<T>(value: Option<T>, name: &str) -> HarnessResult<T> {
    value.ok_or_else(|| HarnessError(format!("missing required argument {name}")))
}

fn build_cases() -> HarnessResult<Vec<PreparedCase>> {
    Ok(vec![
        external_payload_one_node()?,
        external_table_one_node()?,
        external_table_three_way_fan_out()?,
        builtin_expression_one_node()?,
        builtin_sql_one_node()?,
    ])
}

fn external_payload_one_node() -> HarnessResult<PreparedCase> {
    let plan = PipelineBuilder::new("allocation-regression-external-payload")
        .map_err(harness_error)?
        .with_datafusion_config(DATAFUSION_CONFIG)
        .add_node(
            "passthrough",
            Box::new(PassthroughOperator::new(BatchKind::Array)?),
        )
        .map_err(harness_error)?
        .compile(&UdfRegistry::new().snapshot())
        .map_err(harness_error)?;
    let input = Batch::external(
        Arc::new(BenchmarkExternalPayload { rows: 1_000 }),
        BatchMetadata::default(),
    )
    .map_err(harness_error)?;
    PreparedCase::finish(
        "external_payload_one_node",
        "external payload, 1000 rows",
        &["External"],
        None,
        OutputExpectation::ExternalPayloadIdentity,
        plan,
        input,
    )
}

fn external_table_one_node() -> HarnessResult<PreparedCase> {
    let plan = PipelineBuilder::new("allocation-regression-external-table")
        .map_err(harness_error)?
        .with_datafusion_config(DATAFUSION_CONFIG)
        .add_node(
            "passthrough",
            Box::new(PassthroughOperator::new(BatchKind::Table)?),
        )
        .map_err(harness_error)?
        .compile(&UdfRegistry::new().snapshot())
        .map_err(harness_error)?;
    PreparedCase::finish(
        "external_table_one_node",
        "Arrow table value=0..63",
        &["External"],
        None,
        OutputExpectation::TableIdentity,
        plan,
        table_input()?,
    )
}

fn external_table_three_way_fan_out() -> HarnessResult<PreparedCase> {
    let mut builder = PipelineBuilder::new("allocation-regression-external-table-fan-out")
        .map_err(harness_error)?
        .with_datafusion_config(DATAFUSION_CONFIG)
        .add_node(
            "root",
            Box::new(PassthroughOperator::new(BatchKind::Table)?),
        )
        .map_err(harness_error)?;
    for leaf in ["leaf_a", "leaf_b", "leaf_c"] {
        builder = builder
            .add_node(leaf, Box::new(PassthroughOperator::new(BatchKind::Table)?))
            .map_err(harness_error)?
            .connect(Edge::new(
                PortEndpoint::new("root", "output").map_err(harness_error)?,
                PortEndpoint::new(leaf, "input").map_err(harness_error)?,
            ))
            .map_err(harness_error)?;
    }
    let plan = builder
        .compile(&UdfRegistry::new().snapshot())
        .map_err(harness_error)?;
    PreparedCase::finish(
        "external_table_three_way_fan_out",
        "Arrow table value=0..63, three terminal branches",
        &["External", "External", "External", "External"],
        None,
        OutputExpectation::TableIdentity,
        plan,
        table_input()?,
    )
}

fn builtin_expression_one_node() -> HarnessResult<PreparedCase> {
    let operator =
        ExpressionOperator::new("expression", "plus_one = value + 1", vec![], None, vec![])
            .map_err(harness_error)?;
    let plan = PipelineBuilder::new("allocation-regression-expression")
        .map_err(harness_error)?
        .with_datafusion_config(DATAFUSION_CONFIG)
        .add_node("expression", Box::new(operator))
        .map_err(harness_error)?
        .compile(&UdfRegistry::new().snapshot())
        .map_err(harness_error)?;
    PreparedCase::finish(
        "builtin_expression_one_node",
        "Arrow table value=0..63",
        &["Expression"],
        Some("expression"),
        OutputExpectation::ExpressionValues,
        plan,
        table_input()?,
    )
}

fn builtin_sql_one_node() -> HarnessResult<PreparedCase> {
    let operator = SqlOperator::new(
        "sql",
        "SELECT value * 2 AS doubled FROM input",
        vec!["input".into()],
        vec![],
    )
    .map_err(harness_error)?;
    let plan = PipelineBuilder::new("allocation-regression-sql")
        .map_err(harness_error)?
        .with_datafusion_config(DATAFUSION_CONFIG)
        .add_node("sql", Box::new(operator))
        .map_err(harness_error)?
        .compile(&UdfRegistry::new().snapshot())
        .map_err(harness_error)?;
    PreparedCase::finish(
        "builtin_sql_one_node",
        "Arrow table input.value=0..63",
        &["Sql"],
        Some("sql"),
        OutputExpectation::SqlValues,
        plan,
        table_input()?,
    )
}

fn table_input() -> HarnessResult<Batch> {
    let record = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from_iter_values(0..64)) as Arc<dyn datafusion::arrow::array::Array>,
    )])
    .map_err(harness_error)?;
    Batch::table(vec![record], BatchMetadata::default()).map_err(harness_error)
}

fn workload_fingerprint(
    name: &str,
    payload: &str,
    variants: &[&str],
    node_count: usize,
    metric_node: Option<&str>,
    output_assertion: &str,
    plan_fingerprint: &str,
) -> HarnessResult<String> {
    let descriptor = serde_json::to_vec(&serde_json::json!({
        "name": name,
        "payload": payload,
        "compiled_operator_variants": variants,
        "compiled_node_count": node_count,
        "datafusion_config": DATAFUSION_CONFIG,
        "metric_node": metric_node,
        "output_assertion": output_assertion,
        "plan_fingerprint": plan_fingerprint,
    }))
    .map_err(harness_error)?;
    Ok(hex::encode(Sha256::digest(descriptor)))
}

fn run_measurement(options: &MeasureOptions) -> HarnessResult<()> {
    validate_measurement_scale(options)?;
    let provenance = collect_provenance(options.role)?;
    let cases = build_cases()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(harness_error)?;
    let measurement_thread = thread::current();
    let measurement_thread_id = measurement_thread.id();
    let measurement_thread_label = format!("{measurement_thread_id:?}");
    let mut case_reports = Vec::with_capacity(cases.len());
    let mut invalid_reason = None;

    for case in &cases {
        let report = measure_case(
            case,
            &runtime,
            measurement_thread_id,
            &measurement_thread_label,
            options,
        );
        if !report.valid {
            invalid_reason.clone_from(&report.invalid_reason);
            case_reports.push(report);
            break;
        }
        case_reports.push(report);
    }

    let valid = invalid_reason.is_none() && case_reports.len() == cases.len();
    if !valid && invalid_reason.is_none() {
        invalid_reason = Some("not all five fixed cases completed".into());
    }
    let report = AllocationReport {
        schema_version: SCHEMA_VERSION,
        role: options.role,
        valid,
        invalid_reason: invalid_reason.clone(),
        product_sha: provenance.product_sha,
        fixed_baseline_sha: FIXED_BASELINE_SHA.into(),
        harness_commit_sha: provenance.harness_commit_sha,
        frozen_files: provenance.frozen_files,
        git_status_short: provenance.git_status_short,
        toolchain: provenance.toolchain,
        environment: provenance.environment,
        measurement_thread: MeasurementThreadEvidence {
            id: measurement_thread_label,
            name: measurement_thread.name().map(str::to_owned),
            current_thread_runtime: true,
            unchanged_for_all_measurements: case_reports.iter().all(|case| {
                case.repetitions
                    .iter()
                    .all(|repetition| repetition.on_expected_thread)
            }),
        },
        cases: case_reports,
    };
    let output = if options.output.is_absolute() {
        options.output.clone()
    } else {
        provenance.repo_root.join(&options.output)
    };
    write_json(&output, &report)?;
    if valid {
        println!(
            "wrote valid {:?} allocation report to {}",
            options.role,
            output.display()
        );
        Ok(())
    } else {
        Err(HarnessError(format!(
            "allocation run is invalid: {}",
            invalid_reason.unwrap_or_else(|| "unknown reason".into())
        )))
    }
}

fn validate_measurement_scale(options: &MeasureOptions) -> HarnessResult<()> {
    if options.warmup_dispatches != FIXED_WARMUP_DISPATCHES
        || options.measured_dispatches != FIXED_MEASURED_DISPATCHES
        || options.repetitions != FIXED_REPETITIONS
    {
        return Err(HarnessError(format!(
            "measurement scale must be exactly warmups={FIXED_WARMUP_DISPATCHES}, dispatches={FIXED_MEASURED_DISPATCHES}, repetitions={FIXED_REPETITIONS}"
        )));
    }
    Ok(())
}

struct Provenance {
    repo_root: PathBuf,
    product_sha: String,
    harness_commit_sha: String,
    frozen_files: FrozenFileEvidence,
    git_status_short: String,
    toolchain: ToolchainEvidence,
    environment: EnvironmentEvidence,
}

fn collect_provenance(role: Role) -> HarnessResult<Provenance> {
    let repo_root = PathBuf::from(command_output("git", &["rev-parse", "--show-toplevel"])?);
    let head = command_output_in(&repo_root, "git", &["rev-parse", "HEAD"])?;
    let git_status_short = command_output_in(&repo_root, "git", &["status", "--short"])?;
    if !git_status_short.is_empty() {
        return Err(HarnessError(format!(
            "tracked worktree must be clean before measurement, status: {git_status_short:?}"
        )));
    }
    let harness_commit_sha = command_output_in(
        &repo_root,
        "git",
        &[
            "log",
            "-1",
            "--format=%H",
            "--",
            FROZEN_FILES[0],
            FROZEN_FILES[1],
            FROZEN_FILES[2],
        ],
    )?;
    if harness_commit_sha.is_empty() {
        return Err(HarnessError(
            "could not resolve the harness-only commit".into(),
        ));
    }
    command_output_in(
        &repo_root,
        "git",
        &[
            "merge-base",
            "--is-ancestor",
            FIXED_BASELINE_SHA,
            &harness_commit_sha,
        ],
    )?;
    command_output_in(
        &repo_root,
        "git",
        &["merge-base", "--is-ancestor", &harness_commit_sha, &head],
    )?;
    let changed = command_output_in(
        &repo_root,
        "git",
        &[
            "diff",
            "--name-only",
            FIXED_BASELINE_SHA,
            &harness_commit_sha,
        ],
    )?;
    let actual = changed.lines().collect::<BTreeSet<_>>();
    let expected = FROZEN_FILES.into_iter().collect::<BTreeSet<_>>();
    if actual != expected || (role == Role::Baseline && head != harness_commit_sha) {
        return Err(HarnessError(format!(
            "harness must descend from the fixed product SHA and change exactly the three frozen files; changed={actual:?}, head={head}, harness={harness_commit_sha}, role={role:?}"
        )));
    }

    let rustc_vv = command_output("rustc", &["-Vv"])?;
    let cargo_version = command_output("cargo", &["-V"])?;
    let host = rustc_vv
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .ok_or_else(|| HarnessError("rustc -Vv did not report a host".into()))?
        .to_owned();
    let target = env::var("CARGO_BUILD_TARGET").unwrap_or_else(|_| host.clone());
    let lockfile = fs::read_to_string(repo_root.join(FROZEN_FILES[0])).map_err(harness_error)?;
    let registry_checksum = allocation_counter_checksum(&lockfile)?;
    if registry_checksum != ALLOCATION_COUNTER_CHECKSUM {
        return Err(HarnessError(format!(
            "allocation-counter registry checksum mismatch: {registry_checksum}"
        )));
    }
    let frozen_files = FrozenFileEvidence {
        cargo_lock_sha256: sha256_file(repo_root.join(FROZEN_FILES[0]))?,
        crate_manifest_sha256: sha256_file(repo_root.join(FROZEN_FILES[1]))?,
        benchmark_sha256: sha256_file(repo_root.join(FROZEN_FILES[2]))?,
        allocation_counter_version: "0.8.1".into(),
        allocation_counter_registry_checksum: registry_checksum,
    };
    Ok(Provenance {
        repo_root,
        product_sha: match role {
            Role::Baseline => FIXED_BASELINE_SHA.into(),
            Role::Candidate => head,
        },
        harness_commit_sha,
        frozen_files,
        git_status_short,
        toolchain: ToolchainEvidence {
            rustc_vv,
            cargo_version,
            host,
            target,
        },
        environment: collect_environment(),
    })
}

fn allocation_counter_checksum(lockfile: &str) -> HarnessResult<String> {
    lockfile
        .split("[[package]]")
        .find(|package| {
            package.contains("name = \"allocation-counter\"")
                && package.contains("version = \"0.8.1\"")
        })
        .and_then(|package| {
            package
                .lines()
                .find_map(|line| line.trim().strip_prefix("checksum = \""))
        })
        .and_then(|value| value.strip_suffix('"'))
        .map(str::to_owned)
        .ok_or_else(|| HarnessError("Cargo.lock lacks allocation-counter 0.8.1 checksum".into()))
}

fn collect_environment() -> EnvironmentEvidence {
    EnvironmentEvidence {
        uname_a: best_effort_command("uname", &["-a"]),
        lscpu: best_effort_command("lscpu", &[]),
        cpu_governors: cpu_governors(),
        power_supplies: power_supplies(),
        virtualization: best_effort_command("systemd-detect-virt", &[]),
        container_markers: format!(
            "/.dockerenv={}, /run/.containerenv={}, /proc/1/cgroup={}",
            Path::new("/.dockerenv").exists(),
            Path::new("/run/.containerenv").exists(),
            fs::read_to_string("/proc/1/cgroup")
                .unwrap_or_else(|error| format!("unavailable: {error}"))
                .trim()
        ),
        background_load_policy: env::var("DAL38_BACKGROUND_LOAD_POLICY").unwrap_or_else(|_| {
            "shared runner; no benchmark-related work intentionally run concurrently".into()
        }),
        cargo_build_jobs: env::var("CARGO_BUILD_JOBS").unwrap_or_else(|_| "unset".into()),
        calc_flow_benchmark_scale: env::var("CALC_FLOW_BENCHMARK_SCALE")
            .unwrap_or_else(|_| "unset".into()),
    }
}

fn cpu_governors() -> Vec<String> {
    let mut governors = BTreeSet::new();
    if let Ok(entries) = fs::read_dir("/sys/devices/system/cpu") {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.strip_prefix("cpu").is_some_and(|suffix| {
                !suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit())
            }) {
                let path = entry.path().join("cpufreq/scaling_governor");
                if let Ok(governor) = fs::read_to_string(path) {
                    governors.insert(governor.trim().to_owned());
                }
            }
        }
    }
    if governors.is_empty() {
        governors.insert("unavailable".into());
    }
    governors.into_iter().collect()
}

fn power_supplies() -> Vec<PowerSupplyEvidence> {
    let mut supplies = Vec::new();
    if let Ok(entries) = fs::read_dir("/sys/class/power_supply") {
        for entry in entries.flatten() {
            let path = entry.path();
            let kind = fs::read_to_string(path.join("type"))
                .unwrap_or_else(|error| format!("unavailable: {error}"))
                .trim()
                .to_owned();
            let online = fs::read_to_string(path.join("online"))
                .unwrap_or_else(|error| format!("unavailable: {error}"))
                .trim()
                .to_owned();
            supplies.push(PowerSupplyEvidence {
                name: entry.file_name().to_string_lossy().into_owned(),
                kind,
                online,
            });
        }
    }
    supplies.sort_by(|left, right| left.name.cmp(&right.name));
    supplies
}

fn measure_case(
    case: &PreparedCase,
    runtime: &tokio::runtime::Runtime,
    expected_thread: thread::ThreadId,
    measurement_thread_label: &str,
    options: &MeasureOptions,
) -> CaseReport {
    for _ in 0..options.warmup_dispatches {
        if let Err(failure) = runtime.block_on(dispatch_once(case, expected_thread)) {
            return invalid_case_report(
                case,
                options,
                format!("warm-up failed: {}", failure.description()),
            );
        }
    }

    measure(|| {});
    let mut repetitions = Vec::with_capacity(options.repetitions);
    let mut stable_count_total = true;
    let mut stable_bytes_total = true;
    let mut expected_raw: Option<RawAllocationInfo> = None;
    let mut case_invalid_reason = None;

    for repetition_index in 0..options.repetitions {
        let mut completed_dispatches = 0_u64;
        let mut failure = None;
        let info = measure(|| {
            runtime.block_on(async {
                for _ in 0..options.measured_dispatches {
                    match dispatch_once(case, expected_thread).await {
                        Ok(()) => completed_dispatches += 1,
                        Err(code) => {
                            failure = Some(code);
                            break;
                        }
                    }
                }
            });
        });
        let raw = RawAllocationInfo::from(info);
        if let Some(expected) = expected_raw {
            stable_count_total &= raw.count_total == expected.count_total;
            stable_bytes_total &= raw.bytes_total == expected.bytes_total;
        } else {
            expected_raw = Some(raw);
        }
        let mut invalid_reason = failure.map(|code| code.description().to_owned());
        if completed_dispatches != options.measured_dispatches && invalid_reason.is_none() {
            invalid_reason = Some(format!(
                "completed {completed_dispatches} dispatches, expected {}",
                options.measured_dispatches
            ));
        }
        if !stable_count_total || !stable_bytes_total {
            invalid_reason = Some("count_total or bytes_total changed between repetitions".into());
        }
        let normalized = normalize(raw, options.measured_dispatches, case.expected_node_count);
        let output_assertion_passed = failure.is_none();
        repetitions.push(RepetitionReport {
            repetition_index,
            requested_dispatches: options.measured_dispatches,
            completed_dispatches,
            measurement_thread_id: measurement_thread_label.to_owned(),
            on_expected_thread: !matches!(failure, Some(FailureCode::WrongThread)),
            raw,
            normalized,
            output_assertion_passed,
            invalid_reason: invalid_reason.clone(),
        });
        if invalid_reason.is_some() {
            case_invalid_reason = invalid_reason;
            break;
        }
    }

    let valid = case_invalid_reason.is_none()
        && repetitions.len() == options.repetitions
        && stable_count_total
        && stable_bytes_total;
    CaseReport {
        name: case.name.into(),
        payload: case.payload.into(),
        workload_fingerprint: case.workload_fingerprint.clone(),
        plan_fingerprint: case.plan.fingerprint().into(),
        compiled_node_count: case.expected_node_count,
        compiled_operator_variants: case
            .logical_variants
            .iter()
            .map(|variant| (*variant).to_owned())
            .collect(),
        configured_datafusion: DATAFUSION_CONFIG,
        compiled_datafusion: case.plan.datafusion_config(),
        requires_datafusion: case.plan.requires_datafusion(),
        metric_assertion: metric_assertion_description(case.expected_metric_node),
        output_assertion: case.expectation.description().into(),
        warmup_dispatches: options.warmup_dispatches,
        requested_dispatches: options.measured_dispatches,
        repetitions_requested: options.repetitions,
        repetitions,
        stable_count_total,
        stable_bytes_total,
        valid,
        invalid_reason: case_invalid_reason,
    }
}

fn invalid_case_report(
    case: &PreparedCase,
    options: &MeasureOptions,
    reason: String,
) -> CaseReport {
    CaseReport {
        name: case.name.into(),
        payload: case.payload.into(),
        workload_fingerprint: case.workload_fingerprint.clone(),
        plan_fingerprint: case.plan.fingerprint().into(),
        compiled_node_count: case.expected_node_count,
        compiled_operator_variants: case
            .logical_variants
            .iter()
            .map(|variant| (*variant).to_owned())
            .collect(),
        configured_datafusion: DATAFUSION_CONFIG,
        compiled_datafusion: case.plan.datafusion_config(),
        requires_datafusion: case.plan.requires_datafusion(),
        metric_assertion: metric_assertion_description(case.expected_metric_node),
        output_assertion: case.expectation.description().into(),
        warmup_dispatches: options.warmup_dispatches,
        requested_dispatches: options.measured_dispatches,
        repetitions_requested: options.repetitions,
        repetitions: Vec::new(),
        stable_count_total: false,
        stable_bytes_total: false,
        valid: false,
        invalid_reason: Some(reason),
    }
}

fn metric_assertion_description(expected_metric_node: Option<&str>) -> String {
    expected_metric_node.map_or_else(
        || "no DataFusion metrics".into(),
        |node| format!("exactly one DataFusion metric for node {node}"),
    )
}

async fn dispatch_once(
    case: &PreparedCase,
    expected_thread: thread::ThreadId,
) -> Result<(), FailureCode> {
    if thread::current().id() != expected_thread {
        return Err(FailureCode::WrongThread);
    }
    let inputs = BTreeMap::from([(case.input_name.clone(), case.input.clone())]);
    let result = case
        .plan
        .execute(inputs, ExecutionOptions::default())
        .await
        .map_err(|_| FailureCode::DispatchError)?;
    if !metrics_match(&result, case.expected_metric_node) {
        return Err(FailureCode::MetricAssertion);
    }
    if !outputs_match(case, &result) {
        return Err(FailureCode::OutputAssertion);
    }
    if thread::current().id() != expected_thread {
        return Err(FailureCode::WrongThread);
    }
    Ok(())
}

fn metrics_match(result: &RunResult, expected_metric_node: Option<&str>) -> bool {
    match expected_metric_node {
        None => result.datafusion_metrics.is_empty(),
        Some(node) => {
            result.datafusion_metrics.len() == 1
                && result.datafusion_metrics[0].node_id.as_deref() == Some(node)
        }
    }
}

fn outputs_match(case: &PreparedCase, result: &RunResult) -> bool {
    if result.outputs.len() != case.terminal_outputs.len() {
        return false;
    }
    match case.expectation {
        OutputExpectation::ExternalPayloadIdentity => {
            let Some(output) = case
                .terminal_outputs
                .first()
                .and_then(|name| result.outputs.get(name))
            else {
                return false;
            };
            match (case.input.external_payload(), output.external_payload()) {
                (Ok(input), Ok(output)) => Arc::ptr_eq(input, output),
                _ => false,
            }
        }
        OutputExpectation::TableIdentity => case.terminal_outputs.iter().all(|name| {
            result
                .outputs
                .get(name)
                .is_some_and(|output| table_identity_matches(&case.input, output))
        }),
        OutputExpectation::ExpressionValues => {
            transformed_values_match(case, result, "plus_one", 1_i64..=64)
        }
        OutputExpectation::SqlValues => {
            transformed_values_match(case, result, "doubled", (0_i64..64).map(|value| value * 2))
        }
    }
}

fn table_identity_matches(input: &Batch, output: &Batch) -> bool {
    let (Ok(input), Ok(output)) = (input.table_payload(), output.table_payload()) else {
        return false;
    };
    if input.batches().len() != 1 || output.batches().len() != 1 {
        return false;
    }
    let input_record = &input.batches()[0];
    let output_record = &output.batches()[0];
    if input_record.num_columns() != 1 || output_record.num_columns() != 1 {
        return false;
    }
    let (Some(input_values), Some(output_values)) = (
        input_record.column(0).as_any().downcast_ref::<Int64Array>(),
        output_record
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>(),
    ) else {
        return false;
    };
    Arc::ptr_eq(input.schema(), output.schema())
        && Arc::ptr_eq(input_record.column(0), output_record.column(0))
        && input_values.values().as_ptr() == output_values.values().as_ptr()
}

fn transformed_values_match(
    case: &PreparedCase,
    result: &RunResult,
    column: &str,
    expected: impl Iterator<Item = i64>,
) -> bool {
    let Some(output) = case
        .terminal_outputs
        .first()
        .and_then(|name| result.outputs.get(name))
    else {
        return false;
    };
    let Ok(table) = output.table_payload() else {
        return false;
    };
    if table.batches().len() != 1 {
        return false;
    }
    let Some(values) = table.batches()[0]
        .column_by_name(column)
        .and_then(|array| array.as_any().downcast_ref::<Int64Array>())
    else {
        return false;
    };
    values.values().iter().copied().eq(expected)
}

#[expect(
    clippy::cast_precision_loss,
    reason = "normalized floating values are informational; raw integers determine the verdict"
)]
fn normalize(
    raw: RawAllocationInfo,
    dispatches: u64,
    compiled_nodes: usize,
) -> NormalizedAllocationInfo {
    let dispatches = dispatches as f64;
    let node_dispatches = dispatches * compiled_nodes as f64;
    NormalizedAllocationInfo {
        calls_per_dispatch: raw.count_total as f64 / dispatches,
        bytes_per_dispatch: raw.bytes_total as f64 / dispatches,
        calls_per_node_dispatch: raw.count_total as f64 / node_dispatches,
        bytes_per_node_dispatch: raw.bytes_total as f64 / node_dispatches,
    }
}

fn compare_reports(baseline_path: &Path, candidate_path: &Path) -> HarnessResult<()> {
    let baseline = read_report(baseline_path)?;
    let candidate = read_report(candidate_path)?;
    if let Err(error) = validate_comparison_identity(&baseline, &candidate) {
        let comparison = ComparisonReport {
            valid: false,
            passed: false,
            invalid_reason: Some(error.to_string()),
            baseline_product_sha: baseline.product_sha,
            candidate_product_sha: candidate.product_sha,
            harness_commit_sha: baseline.harness_commit_sha,
            cases: Vec::new(),
        };
        println!(
            "{}",
            serde_json::to_string_pretty(&comparison).map_err(harness_error)?
        );
        return Err(error);
    }
    let mut case_reports = Vec::with_capacity(baseline.cases.len());
    let mut all_passed = true;
    for (baseline_case, candidate_case) in baseline.cases.iter().zip(&candidate.cases) {
        let mut repetitions = Vec::with_capacity(baseline_case.repetitions.len());
        let mut case_passed = true;
        for (baseline_rep, candidate_rep) in baseline_case
            .repetitions
            .iter()
            .zip(&candidate_case.repetitions)
        {
            let passed = candidate_rep.raw.count_total <= baseline_rep.raw.count_total
                && candidate_rep.raw.bytes_total <= baseline_rep.raw.bytes_total;
            case_passed &= passed;
            repetitions.push(ComparisonRepetition {
                repetition_index: baseline_rep.repetition_index,
                baseline_count_total: baseline_rep.raw.count_total,
                candidate_count_total: candidate_rep.raw.count_total,
                count_delta: signed_delta(
                    candidate_rep.raw.count_total,
                    baseline_rep.raw.count_total,
                ),
                baseline_bytes_total: baseline_rep.raw.bytes_total,
                candidate_bytes_total: candidate_rep.raw.bytes_total,
                bytes_delta: signed_delta(
                    candidate_rep.raw.bytes_total,
                    baseline_rep.raw.bytes_total,
                ),
                calls_per_dispatch_delta: candidate_rep.normalized.calls_per_dispatch
                    - baseline_rep.normalized.calls_per_dispatch,
                bytes_per_dispatch_delta: candidate_rep.normalized.bytes_per_dispatch
                    - baseline_rep.normalized.bytes_per_dispatch,
                calls_per_node_dispatch_delta: candidate_rep.normalized.calls_per_node_dispatch
                    - baseline_rep.normalized.calls_per_node_dispatch,
                bytes_per_node_dispatch_delta: candidate_rep.normalized.bytes_per_node_dispatch
                    - baseline_rep.normalized.bytes_per_node_dispatch,
                passed,
            });
        }
        all_passed &= case_passed;
        case_reports.push(ComparisonCase {
            name: baseline_case.name.clone(),
            repetitions,
            passed: case_passed,
        });
    }
    let comparison = ComparisonReport {
        valid: true,
        passed: all_passed,
        invalid_reason: None,
        baseline_product_sha: baseline.product_sha,
        candidate_product_sha: candidate.product_sha,
        harness_commit_sha: baseline.harness_commit_sha,
        cases: case_reports,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&comparison).map_err(harness_error)?
    );
    if all_passed {
        Ok(())
    } else {
        Err(HarnessError(
            "candidate has a positive allocation count or byte delta".into(),
        ))
    }
}

fn validate_comparison_identity(
    baseline: &AllocationReport,
    candidate: &AllocationReport,
) -> HarnessResult<()> {
    let expected_provenance = collect_provenance(Role::Candidate)?;
    let expected_cases = build_cases()?;
    validate_report(
        baseline,
        Role::Baseline,
        FIXED_BASELINE_SHA,
        &expected_provenance,
        &expected_cases,
    )?;
    validate_report(
        candidate,
        Role::Candidate,
        &expected_provenance.product_sha,
        &expected_provenance,
        &expected_cases,
    )?;
    Ok(())
}

fn validate_report(
    report: &AllocationReport,
    expected_role: Role,
    expected_product_sha: &str,
    expected_provenance: &Provenance,
    expected_cases: &[PreparedCase],
) -> HarnessResult<()> {
    let label = format!("{expected_role:?}").to_ascii_lowercase();
    if report.schema_version != SCHEMA_VERSION {
        return Err(HarnessError(format!(
            "{label} schema_version must be {SCHEMA_VERSION}, got {}",
            report.schema_version
        )));
    }
    if report.role != expected_role {
        return Err(HarnessError(format!("{label} role is incorrect")));
    }
    if !report.valid || report.invalid_reason.is_some() {
        return Err(HarnessError(format!(
            "{label} report must be valid with no invalid_reason"
        )));
    }
    if report.product_sha != expected_product_sha {
        return Err(HarnessError(format!("{label} product SHA is incorrect")));
    }
    if report.fixed_baseline_sha != FIXED_BASELINE_SHA {
        return Err(HarnessError(format!(
            "{label} fixed baseline SHA is incorrect"
        )));
    }
    if report.harness_commit_sha != expected_provenance.harness_commit_sha {
        return Err(HarnessError(format!("{label} harness commit is incorrect")));
    }
    if report.frozen_files != expected_provenance.frozen_files {
        return Err(HarnessError(format!(
            "{label} frozen-file or dependency identity is incorrect"
        )));
    }
    if !report.git_status_short.is_empty() {
        return Err(HarnessError(format!(
            "{label} report was produced from a dirty tracked worktree"
        )));
    }
    if report.toolchain != expected_provenance.toolchain {
        return Err(HarnessError(format!("{label} toolchain is incorrect")));
    }
    if report.measurement_thread.id.is_empty()
        || !report.measurement_thread.current_thread_runtime
        || !report.measurement_thread.unchanged_for_all_measurements
    {
        return Err(HarnessError(format!(
            "{label} measurement thread evidence is invalid"
        )));
    }
    if report.cases.len() != expected_cases.len() {
        return Err(HarnessError(format!(
            "{label} must contain exactly {} cases",
            expected_cases.len()
        )));
    }
    let unique_names = report
        .cases
        .iter()
        .map(|case| case.name.as_str())
        .collect::<BTreeSet<_>>();
    if unique_names.len() != expected_cases.len() {
        return Err(HarnessError(format!(
            "{label} contains duplicate or missing cases"
        )));
    }
    for (case, expected_case) in report.cases.iter().zip(expected_cases) {
        validate_case_report(&label, case, expected_case, &report.measurement_thread.id)?;
    }
    Ok(())
}

fn validate_case_report(
    report_label: &str,
    case: &CaseReport,
    expected: &PreparedCase,
    measurement_thread_id: &str,
) -> HarnessResult<()> {
    let expected_variants = expected
        .logical_variants
        .iter()
        .map(|variant| (*variant).to_owned())
        .collect::<Vec<_>>();
    let expected_metric_assertion = metric_assertion_description(expected.expected_metric_node);
    let expected_output_assertion = expected.expectation.description();
    if case.name != expected.name
        || case.payload != expected.payload
        || case.plan_fingerprint != expected.plan.fingerprint()
        || case.compiled_node_count != expected.expected_node_count
        || case.compiled_operator_variants != expected_variants
        || case.configured_datafusion != DATAFUSION_CONFIG
        || case.compiled_datafusion != expected.plan.datafusion_config()
        || case.requires_datafusion != expected.plan.requires_datafusion()
        || case.metric_assertion != expected_metric_assertion
        || case.output_assertion != expected_output_assertion
    {
        return Err(HarnessError(format!(
            "{report_label} case {} does not match the compiled fixed workload contract",
            case.name
        )));
    }
    let report_variants = case
        .compiled_operator_variants
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let recomputed_workload_fingerprint = workload_fingerprint(
        &case.name,
        &case.payload,
        &report_variants,
        case.compiled_node_count,
        expected.expected_metric_node,
        &case.output_assertion,
        &case.plan_fingerprint,
    )?;
    if case.workload_fingerprint != expected.workload_fingerprint
        || case.workload_fingerprint != recomputed_workload_fingerprint
    {
        return Err(HarnessError(format!(
            "{report_label} case {} workload fingerprint is not reproducible",
            case.name
        )));
    }
    if case.warmup_dispatches != FIXED_WARMUP_DISPATCHES
        || case.requested_dispatches != FIXED_MEASURED_DISPATCHES
        || case.repetitions_requested != FIXED_REPETITIONS
        || case.repetitions.len() != FIXED_REPETITIONS
    {
        return Err(HarnessError(format!(
            "{report_label} case {} does not use the fixed measurement scale",
            case.name
        )));
    }
    if !case.valid
        || case.invalid_reason.is_some()
        || !case.stable_count_total
        || !case.stable_bytes_total
    {
        return Err(HarnessError(format!(
            "{report_label} case {} validity or stability evidence failed",
            case.name
        )));
    }

    let mut stable_totals = None;
    for (expected_index, repetition) in case.repetitions.iter().enumerate() {
        validate_repetition(
            report_label,
            case,
            repetition,
            expected_index,
            measurement_thread_id,
        )?;
        let totals = (repetition.raw.count_total, repetition.raw.bytes_total);
        if stable_totals.is_some_and(|expected_totals| expected_totals != totals) {
            return Err(HarnessError(format!(
                "{report_label} case {} raw totals are unstable",
                case.name
            )));
        }
        stable_totals = Some(totals);
    }
    Ok(())
}

fn validate_repetition(
    report_label: &str,
    case: &CaseReport,
    repetition: &RepetitionReport,
    expected_index: usize,
    measurement_thread_id: &str,
) -> HarnessResult<()> {
    if repetition.repetition_index != expected_index
        || repetition.requested_dispatches != FIXED_MEASURED_DISPATCHES
        || repetition.completed_dispatches != FIXED_MEASURED_DISPATCHES
    {
        return Err(HarnessError(format!(
            "{report_label} case {} repetition {expected_index} index or dispatch count is invalid",
            case.name
        )));
    }
    if repetition.measurement_thread_id != measurement_thread_id || !repetition.on_expected_thread {
        return Err(HarnessError(format!(
            "{report_label} case {} repetition {expected_index} ran off-thread",
            case.name
        )));
    }
    if !repetition.output_assertion_passed || repetition.invalid_reason.is_some() {
        return Err(HarnessError(format!(
            "{report_label} case {} repetition {expected_index} output or validity assertion failed",
            case.name
        )));
    }
    validate_raw_allocation(report_label, &case.name, expected_index, repetition.raw)?;
    let recomputed = normalize(
        repetition.raw,
        FIXED_MEASURED_DISPATCHES,
        case.compiled_node_count,
    );
    if !normalized_equal(&repetition.normalized, &recomputed) {
        return Err(HarnessError(format!(
            "{report_label} case {} repetition {expected_index} normalized allocation identity is incorrect",
            case.name
        )));
    }
    Ok(())
}

fn validate_raw_allocation(
    report_label: &str,
    case_name: &str,
    repetition_index: usize,
    raw: RawAllocationInfo,
) -> HarnessResult<()> {
    let current_is_zero = raw.count_current == 0 && raw.bytes_current == 0;
    let peaks_fit_totals = raw.count_max <= raw.count_total && raw.bytes_max <= raw.bytes_total;
    let zero_counts_are_consistent =
        raw.count_total != 0 || (raw.count_max == 0 && raw.bytes_total == 0 && raw.bytes_max == 0);
    if !current_is_zero || !peaks_fit_totals || !zero_counts_are_consistent {
        return Err(HarnessError(format!(
            "{report_label} case {case_name} repetition {repetition_index} has inconsistent raw AllocationInfo fields"
        )));
    }
    Ok(())
}

fn normalized_equal(left: &NormalizedAllocationInfo, right: &NormalizedAllocationInfo) -> bool {
    left.calls_per_dispatch.to_bits() == right.calls_per_dispatch.to_bits()
        && left.bytes_per_dispatch.to_bits() == right.bytes_per_dispatch.to_bits()
        && left.calls_per_node_dispatch.to_bits() == right.calls_per_node_dispatch.to_bits()
        && left.bytes_per_node_dispatch.to_bits() == right.bytes_per_node_dispatch.to_bits()
}

fn signed_delta(candidate: u64, baseline: u64) -> i64 {
    let delta = i128::from(candidate) - i128::from(baseline);
    i64::try_from(delta).unwrap_or_else(|_| {
        if delta.is_negative() {
            i64::MIN
        } else {
            i64::MAX
        }
    })
}

fn read_report(path: &Path) -> HarnessResult<AllocationReport> {
    let bytes = fs::read(path)
        .map_err(|error| HarnessError(format!("failed to read {}: {error}", path.display())))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| HarnessError(format!("invalid report {}: {error}", path.display())))
}

fn deserialize_required_option<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

fn write_json(path: &Path, report: &AllocationReport) -> HarnessResult<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(harness_error)?;
    }
    let bytes = serde_json::to_vec_pretty(report).map_err(harness_error)?;
    fs::write(path, bytes)
        .map_err(|error| HarnessError(format!("failed to write {}: {error}", path.display())))
}

#[cfg(test)]
type TestReport = serde_json::Value;

#[cfg(test)]
type TestMutation = (&'static str, TestReport, TestReport);

#[cfg(test)]
fn run_regression_tests() -> HarnessResult<()> {
    let repo_root = PathBuf::from(command_output("git", &["rev-parse", "--show-toplevel"])?);
    let directory = repo_root
        .join("target/allocation-regression")
        .join(format!("regression-tests-{}", std::process::id()));
    fs::create_dir_all(&directory).map_err(harness_error)?;
    let original_baseline = synthetic_test_report(Role::Baseline)?;
    let original_candidate = synthetic_test_report(Role::Candidate)?;
    let positive = test_comparison_status(
        &directory,
        "fixed-contract-positive",
        &original_baseline,
        &original_candidate,
    )?;
    let mut failures = Vec::new();
    if !positive.status.success() {
        failures.push(format!(
            "fixed-contract-positive: comparison failed: {}",
            String::from_utf8_lossy(&positive.stderr)
        ));
    }
    if test_short_measurement_status(&directory)?.success() {
        failures.push("short 1/1/1 measurement unexpectedly exited zero".into());
    }
    for (name, baseline, candidate) in test_mutations(&original_baseline, &original_candidate)? {
        expect_test_comparison_failure(&mut failures, &directory, name, &baseline, &candidate)?;
    }
    test_noise_floor_statuses(&mut failures)?;
    if failures.is_empty() {
        println!("allocation_regression regression tests passed");
        Ok(())
    } else {
        Err(HarnessError(failures.join("\n")))
    }
}

#[cfg(test)]
fn synthetic_test_report(role: Role) -> HarnessResult<TestReport> {
    let provenance = collect_provenance(role)?;
    let measurement_thread_id = "ThreadId(1)".to_owned();
    let cases = build_cases()?
        .into_iter()
        .enumerate()
        .map(|(case_index, case)| synthetic_test_case(case_index, case, &measurement_thread_id))
        .collect();
    serde_json::to_value(AllocationReport {
        schema_version: SCHEMA_VERSION,
        role,
        valid: true,
        invalid_reason: None,
        product_sha: provenance.product_sha,
        fixed_baseline_sha: FIXED_BASELINE_SHA.into(),
        harness_commit_sha: provenance.harness_commit_sha,
        frozen_files: provenance.frozen_files,
        git_status_short: provenance.git_status_short,
        toolchain: provenance.toolchain,
        environment: provenance.environment,
        measurement_thread: MeasurementThreadEvidence {
            id: measurement_thread_id,
            name: Some("main".into()),
            current_thread_runtime: true,
            unchanged_for_all_measurements: true,
        },
        cases,
    })
    .map_err(harness_error)
}

#[cfg(test)]
fn synthetic_test_case(
    case_index: usize,
    case: PreparedCase,
    measurement_thread_id: &str,
) -> CaseReport {
    let raw = RawAllocationInfo {
        count_total: 100 + case_index as u64,
        count_current: 0,
        count_max: 1,
        bytes_total: 1_000 + case_index as u64,
        bytes_current: 0,
        bytes_max: 64,
    };
    CaseReport {
        name: case.name.into(),
        payload: case.payload.into(),
        workload_fingerprint: case.workload_fingerprint,
        plan_fingerprint: case.plan.fingerprint().into(),
        compiled_node_count: case.expected_node_count,
        compiled_operator_variants: case
            .logical_variants
            .iter()
            .map(|variant| (*variant).to_owned())
            .collect(),
        configured_datafusion: DATAFUSION_CONFIG,
        compiled_datafusion: case.plan.datafusion_config(),
        requires_datafusion: case.plan.requires_datafusion(),
        metric_assertion: metric_assertion_description(case.expected_metric_node),
        output_assertion: case.expectation.description().into(),
        warmup_dispatches: FIXED_WARMUP_DISPATCHES,
        requested_dispatches: FIXED_MEASURED_DISPATCHES,
        repetitions_requested: FIXED_REPETITIONS,
        repetitions: (0..FIXED_REPETITIONS)
            .map(|repetition_index| RepetitionReport {
                repetition_index,
                requested_dispatches: FIXED_MEASURED_DISPATCHES,
                completed_dispatches: FIXED_MEASURED_DISPATCHES,
                measurement_thread_id: measurement_thread_id.to_owned(),
                on_expected_thread: true,
                raw,
                normalized: normalize(raw, FIXED_MEASURED_DISPATCHES, case.expected_node_count),
                output_assertion_passed: true,
                invalid_reason: None,
            })
            .collect(),
        stable_count_total: true,
        stable_bytes_total: true,
        valid: true,
        invalid_reason: None,
    }
}

#[cfg(test)]
fn test_comparison_status(
    directory: &Path,
    name: &str,
    baseline: &TestReport,
    candidate: &TestReport,
) -> HarnessResult<std::process::Output> {
    let baseline_path = directory.join(format!("{name}-baseline.json"));
    let candidate_path = directory.join(format!("{name}-candidate.json"));
    fs::write(
        &baseline_path,
        serde_json::to_vec_pretty(baseline).map_err(harness_error)?,
    )
    .map_err(harness_error)?;
    fs::write(
        &candidate_path,
        serde_json::to_vec_pretty(candidate).map_err(harness_error)?,
    )
    .map_err(harness_error)?;
    Command::new(env::current_exe().map_err(harness_error)?)
        .arg("--compare")
        .arg(baseline_path)
        .arg(candidate_path)
        .output()
        .map_err(harness_error)
}

#[cfg(test)]
fn expect_test_comparison_failure(
    failures: &mut Vec<String>,
    directory: &Path,
    name: &str,
    baseline: &TestReport,
    candidate: &TestReport,
) -> HarnessResult<()> {
    if test_comparison_status(directory, name, baseline, candidate)?
        .status
        .success()
    {
        failures.push(format!("{name}: comparison unexpectedly exited zero"));
    }
    Ok(())
}

#[cfg(test)]
fn test_short_measurement_status(directory: &Path) -> HarnessResult<std::process::ExitStatus> {
    Command::new(env::current_exe().map_err(harness_error)?)
        .args([
            "--warmup-dispatches",
            "1",
            "--measured-dispatches",
            "1",
            "--repetitions",
            "1",
            "--cases",
            "all-existing-data",
            "--role",
            "candidate",
            "--output",
        ])
        .arg(directory.join("short-measurement.json"))
        .status()
        .map_err(harness_error)
}

#[cfg(test)]
fn test_mutations(
    baseline: &TestReport,
    candidate: &TestReport,
) -> HarnessResult<Vec<TestMutation>> {
    use serde_json::{Value, json};

    let mut mutations = vec![
        test_baseline_mutation(
            "wrong-fixed-baseline",
            baseline,
            candidate,
            "/fixed_baseline_sha",
            json!("wrong-baseline"),
        )?,
        test_paired_mutation(
            "paired-forged-schema",
            baseline,
            candidate,
            "/schema_version",
            json!(99),
        )?,
        test_paired_mutation(
            "paired-forged-harness",
            baseline,
            candidate,
            "/harness_commit_sha",
            json!("forged-harness"),
        )?,
        test_paired_mutation(
            "paired-forged-file-hash",
            baseline,
            candidate,
            "/frozen_files/benchmark_sha256",
            json!("forged-file-hash"),
        )?,
        test_paired_mutation(
            "paired-forged-dependency-checksum",
            baseline,
            candidate,
            "/frozen_files/allocation_counter_registry_checksum",
            json!("forged-dependency-checksum"),
        )?,
        test_paired_mutation(
            "paired-short-report-scale",
            baseline,
            candidate,
            "/cases/0/warmup_dispatches",
            json!(1),
        )?,
        test_candidate_mutation(
            "wrong-product-sha",
            baseline,
            candidate,
            "/product_sha",
            json!("forged-product"),
        )?,
        test_paired_mutation(
            "paired-forged-workload-fingerprint",
            baseline,
            candidate,
            "/cases/0/workload_fingerprint",
            json!("forged-workload"),
        )?,
    ];
    mutations.extend(test_execution_mutations(baseline, candidate)?);

    let mut duplicate = candidate.clone();
    duplicate["cases"][4] = duplicate["cases"][0].clone();
    mutations.push(("duplicate-and-missing-case", baseline.clone(), duplicate));
    let mut missing_repetition = candidate.clone();
    missing_repetition["cases"][0]["repetitions"]
        .as_array_mut()
        .expect("fixture repetitions are an array")
        .pop();
    mutations.push(("missing-repetition", baseline.clone(), missing_repetition));
    let mut invalid_case = candidate.clone();
    invalid_case["cases"][0]["valid"] = Value::Bool(false);
    invalid_case["cases"][0]["invalid_reason"] = json!("injected failure");
    mutations.push(("invalid-case", baseline.clone(), invalid_case));
    let mut missing_field = candidate.clone();
    missing_field
        .as_object_mut()
        .expect("fixture report is an object")
        .remove("invalid_reason");
    mutations.push((
        "missing-required-schema-field",
        baseline.clone(),
        missing_field,
    ));
    let mut unknown_field = candidate.clone();
    unknown_field["attacker_controlled"] = json!("ignored");
    mutations.push(("unknown-schema-field", baseline.clone(), unknown_field));
    Ok(mutations)
}

#[cfg(test)]
fn test_execution_mutations(
    baseline: &TestReport,
    candidate: &TestReport,
) -> HarnessResult<Vec<TestMutation>> {
    use serde_json::json;

    [
        (
            "zero-completed",
            "/cases/0/repetitions/0/completed_dispatches",
            json!(0),
        ),
        (
            "short-completed",
            "/cases/0/repetitions/0/completed_dispatches",
            json!(FIXED_MEASURED_DISPATCHES - 1),
        ),
        (
            "off-thread",
            "/cases/0/repetitions/0/on_expected_thread",
            json!(false),
        ),
        (
            "failed-output-assertion",
            "/cases/0/repetitions/0/output_assertion_passed",
            json!(false),
        ),
        (
            "unstable-measurement-thread",
            "/measurement_thread/unchanged_for_all_measurements",
            json!(false),
        ),
        (
            "unstable-case-totals",
            "/cases/0/stable_count_total",
            json!(false),
        ),
        (
            "malformed-raw-allocation-info",
            "/cases/0/repetitions/0/raw/count_current",
            json!(1),
        ),
        (
            "forged-normalized-allocation",
            "/cases/0/repetitions/0/normalized/calls_per_dispatch",
            json!(999.0),
        ),
        (
            "malformed-repetition-index",
            "/cases/0/repetitions/0/repetition_index",
            json!(9),
        ),
    ]
    .into_iter()
    .map(|(name, pointer, value)| {
        test_candidate_mutation(name, baseline, candidate, pointer, value)
    })
    .collect()
}

#[cfg(test)]
fn test_baseline_mutation(
    name: &'static str,
    baseline: &TestReport,
    candidate: &TestReport,
    pointer: &str,
    value: TestReport,
) -> HarnessResult<TestMutation> {
    let mut mutated = baseline.clone();
    set_test_pointer(&mut mutated, pointer, value)?;
    Ok((name, mutated, candidate.clone()))
}

#[cfg(test)]
fn test_candidate_mutation(
    name: &'static str,
    baseline: &TestReport,
    candidate: &TestReport,
    pointer: &str,
    value: TestReport,
) -> HarnessResult<TestMutation> {
    let mut mutated = candidate.clone();
    set_test_pointer(&mut mutated, pointer, value)?;
    Ok((name, baseline.clone(), mutated))
}

#[cfg(test)]
fn test_paired_mutation(
    name: &'static str,
    baseline: &TestReport,
    candidate: &TestReport,
    pointer: &str,
    value: TestReport,
) -> HarnessResult<TestMutation> {
    let mut baseline = baseline.clone();
    let mut candidate = candidate.clone();
    set_test_pointer(&mut baseline, pointer, value.clone())?;
    set_test_pointer(&mut candidate, pointer, value)?;
    Ok((name, baseline, candidate))
}

#[cfg(test)]
fn set_test_pointer(
    report: &mut TestReport,
    pointer: &str,
    value: TestReport,
) -> HarnessResult<()> {
    *report
        .pointer_mut(pointer)
        .ok_or_else(|| HarnessError(format!("test fixture lacks pointer {pointer}")))? = value;
    Ok(())
}

#[cfg(test)]
fn test_noise_floor_statuses(failures: &mut Vec<String>) -> HarnessResult<()> {
    let valid = Command::new(env::current_exe().map_err(harness_error)?)
        .args(["--validate-noise-floor", "0.25", "0.75", "1.5"])
        .output()
        .map_err(harness_error)?;
    if !valid.status.success() {
        failures.push(format!(
            "correct timing noise floor rejected: {}",
            String::from_utf8_lossy(&valid.stderr)
        ));
    }
    let invalid = Command::new(env::current_exe().map_err(harness_error)?)
        .args(["--validate-noise-floor", "0.25", "0.75", "1.0"])
        .status()
        .map_err(harness_error)?;
    if invalid.success() {
        failures.push("wrong timing noise floor unexpectedly exited zero".into());
    }
    Ok(())
}

fn sha256_file(path: impl AsRef<Path>) -> HarnessResult<String> {
    let mut file = fs::File::open(path).map_err(harness_error)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8 * 1_024];
    loop {
        let read = file.read(&mut buffer).map_err(harness_error)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn command_output(program: &str, arguments: &[&str]) -> HarnessResult<String> {
    command_output_from(None, program, arguments)
}

fn command_output_in(directory: &Path, program: &str, arguments: &[&str]) -> HarnessResult<String> {
    command_output_from(Some(directory), program, arguments)
}

fn command_output_from(
    directory: Option<&Path>,
    program: &str,
    arguments: &[&str],
) -> HarnessResult<String> {
    let mut command = Command::new(program);
    if let Some(directory) = directory {
        command.current_dir(directory);
    }
    let output = command
        .args(arguments)
        .output()
        .map_err(|error| HarnessError(format!("failed to run {program}: {error}")))?;
    if !output.status.success() {
        return Err(HarnessError(format!(
            "{program} {arguments:?} failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn best_effort_command(program: &str, arguments: &[&str]) -> String {
    command_output(program, arguments).unwrap_or_else(|error| format!("unavailable: {error}"))
}

fn harness_error(error: impl Display) -> HarnessError {
    HarnessError(error.to_string())
}
