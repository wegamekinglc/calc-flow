# Array Benchmark Phase 2 Evidence

## Evidence outcome

Phase 2 measured the Rust execution components named by the Phase 1 handoff.
The production source under measurement was commit
`73b9b22598ef80e225628a25ca0c3105b36b0f67`; the only uncommitted source used
by Criterion was the benchmark-local instrumentation in
`crates/calc-flow/benches/core.rs`.

The direct provider and plan measurements came from one Python benchmark
report. They are compatible within that report, but they are not compatible
with the Phase 1 environment because the Python and JAX versions differ.
Nothing in this handoff subtracts or aggregates across those environments.

## Environment

| Item                         | Value                                                              |
| ---------------------------- | ------------------------------------------------------------------ |
| Production source SHA        | `73b9b22598ef80e225628a25ca0c3105b36b0f67`                         |
| Host                         | WSL2 Linux `5.15.167.4-microsoft-standard-WSL2`, x86-64            |
| CPU                          | 13th Gen Intel Core i9-13900HX, 32 logical CPUs                    |
| Rust                         | `rustc 1.88.0 (6b00bc388 2025-06-23)`                              |
| Cargo                        | `cargo 1.88.0 (873a06493 2025-05-10)`                              |
| Criterion                    | `0.8.2`                                                            |
| Python                       | `3.13.9`                                                           |
| NumPy                        | `2.5.1`                                                            |
| JAX / JAXlib                 | `0.10.2` / `0.10.2`                                                |
| JAX configuration            | CPU, x64 disabled                                                  |
| PyArrow                      | `24.0.0`                                                           |
| pytest-benchmark             | `5.2.3`                                                            |
| Maturin                      | `1.14.1`                                                           |
| Machine fingerprint          | `fe554238f5c55b49c8d1961066a86debe87eb538f7fbb6fc19a72f926e264a56` |
| NumPy dependency fingerprint | `89608534f832980a53970e51a9da75a93163ab04de339229ab748945341c1e2e` |
| JAX dependency fingerprint   | `554af159aa7730c1c25683357257c61bb6cdcb0e4e984285a0e4c748accd14a5` |

As in Phase 1, the WSL2 guest exposed no scaling governor, ACPI platform
profile, Intel P-state status, or `powerprofilesctl` profile. Host power mode
could not be observed or fixed.

## Commands

The Rust benchmark used the repository's release benchmark profile:

```bash
CARGO_TARGET_DIR=target/cargo CARGO_BUILD_JOBS=1 \
  cargo bench -p calc-flow --bench core
```

The current Python binding was installed in release mode because no generated
native module remained after Phase 1:

```bash
env -u CONDA_PREFIX \
  VIRTUAL_ENV="$PWD/.venv" \
  CARGO_TARGET_DIR="$PWD/target/cargo" \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  .venv/bin/maturin develop --release
```

The provider and plan layers were then collected together in one report:

```bash
env -u CONDA_PREFIX \
  UV_CACHE_DIR="$PWD/target/uv-cache" \
  CALC_FLOW_BENCHMARK_SCALE=overhead \
  JAX_PLATFORMS=cpu \
  uv run --extra benchmark pytest \
  benchmarks/test_array_provider.py benchmarks/test_array_plan.py \
  -q --benchmark-only \
  --benchmark-json=target/benchmark-results/array-phase2.json
```

The Python command passed all 16 cases in 32.09 seconds. The JSON report SHA-256
was `6b36584a3e4dcabe6e8de4a20edc41d5bb0b7fe27c02b26d863c1b7ec26e091d`.

## Criterion measurements

Criterion's slope point estimates and 95% confidence intervals are reported
in microseconds. The first, fourth, and sixth rows are existing controls; the
other rows are the Phase 2 attribution cases.

| Case                                             | Point estimate (us) | 95% lower (us) | 95% upper (us) |
| ------------------------------------------------ | ------------------- | -------------- | -------------- |
| `compile/expression`                             | 10.789              | 10.743         | 10.838         |
| `execute/datafusion_runtime_new`                 | 30.257              | 30.030         | 30.531         |
| `execute/datafusion_runtime_new_register_udfs`   | 30.913              | 30.689         | 31.156         |
| `execute/expression_1024_rows`                   | 394.174             | 384.570        | 404.419        |
| `execute/external_passthrough_1000_rows`         | 38.063              | 37.619         | 38.595         |
| `json/canonical_nested`                          | 3.533               | 3.513          | 3.558          |

The stable empty UDF registration list adds a diagnostic `0.656 us` to runtime
construction. The external passthrough leaves a diagnostic `7.150 us` after
subtracting the combined runtime-construction and empty-registration point
estimate. Those differences are derived from same-run Criterion cases and do
not have independent confidence intervals.

The passthrough operator and its immutable 1,000-row external payload are
benchmark-local. The plan is compiled once, and every measured iteration calls
the public `ExecutionPlan::execute` with default options. Its time therefore
includes transaction capture, input/output validation, graph routing,
run-scoped DataFusion setup, node timing, and result assembly, while excluding
Python and compilation.

## Contract-v2 validation

`target/benchmark-results/array-phase2.json` contains eight
`provider_boundary` entries and eight `plan_end_to_end` entries. Validation
found no missing fields, invalid nested identity shapes, duplicate logical
cases, non-canonical fingerprints, or workload projection errors.

Each provider/plan pair has the same machine and dependency fingerprints and
the same workload identity after removing only `scope`. The scope is expected
to differ and is therefore included in distinct workload fingerprints. No
cross-report or cross-revision timing was subtracted.

## Same-report attribution

For each scenario/backend pair, the attribution denominator is:

```text
remaining plan overhead = plan_end_to_end mean - provider_boundary mean
```

This removes the direct provider callback measured in the same report and
leaves the compatible plan-layer overhead. It is an attribution denominator,
not an assertion that independently measured components are additive. The
DataFusion setup and external passthrough measurements overlap, so their
percentages must not be summed.

| Scenario                    | Backend | Provider us | Provider CoV | Plan us   | Plan CoV | Gap us    | Runtime new | External passthrough |
| --------------------------- | ------- | ----------- | ------------ | --------- | -------- | --------- | ----------- | -------------------- |
| array_elementwise           | jax     | 116.899     | 106.295%     | 194.856   | 31.698%  | 77.957    | 38.812%     | 48.826%              |
| array_matrix_multiplication | jax     | 67.856      | 180.549%     | 136.703   | 60.983%  | 68.848    | 43.948%     | 55.286%              |
| array_mean                  | jax     | 65.749      | 125.330%     | 145.901   | 54.474%  | 80.152    | 37.750%     | 47.489%              |
| array_transpose_reshape     | jax     | 68.312      | 25.253%      | 138.226   | 22.828%  | 69.915    | 43.277%     | 54.442%              |
| array_elementwise           | numpy   | 22.419      | 33.049%      | 71.040    | 31.155%  | 48.621    | 62.230%     | 78.285%              |
| array_matrix_multiplication | numpy   | 15.224      | 27.888%      | 63.817    | 24.134%  | 48.593    | 62.266%     | 78.330%              |
| array_mean                  | numpy   | 14.488      | 39.207%      | 65.371    | 28.229%  | 50.883    | 59.464%     | 74.805%              |
| array_transpose_reshape     | numpy   | 17.376      | 35.586%      | 67.610    | 25.108%  | 50.234    | 60.233%     | 75.772%              |

The pytest-benchmark distributions are noisy, so these values are diagnostic
point estimates rather than a Phase 1 acceptance comparison. Task 7 defines no
CoV rejection rule. The gate is nevertheless not marginal: the lower
Criterion confidence bound for runtime creation is above 30 us, and its
smallest point-estimate share across all eight denominators is 37.750%.

## Core design gate

| Component                                        | Absolute time (us) | Share of gaps   | At least 10 us | At least 20% | Gate result |
| ------------------------------------------------ | ------------------ | --------------- | -------------- | ------------ | ----------- |
| DataFusion runtime construction                  | 30.257             | 37.750%-62.266% | Yes            | Yes          | Trigger     |
| Runtime construction plus empty UDF registration | 30.913             | 38.568%-63.615% | Yes            | Yes          | Trigger     |
| Empty UDF registration increment                 | 0.656              | 0.818%-1.349%   | No             | No           | Below gate  |
| Public external passthrough execution            | 38.063             | 47.489%-78.330% | Yes            | Yes          | Trigger     |
| Passthrough residual after combined setup        | 7.150              | 8.921%-14.715%  | No             | No           | Below gate  |

The independently isolated component that crosses both thresholds is
run-scoped DataFusion runtime setup. A separate reviewed design would have to
cover semantics, concurrency, cancellation, rollback, metrics, checkpointing,
and API impact. This task does not propose or implement cached sessions,
unchecked execution, skipped transactions, alternate callback routes, or any
other production fast path.

## Design decision

New core design required: run-scoped DataFusion setup
