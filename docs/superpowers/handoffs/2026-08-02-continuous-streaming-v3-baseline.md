# Continuous-Streaming v3 — M0.3 Correctness and Performance Baseline

Captured for Task M0.3 of the
[continuous-streaming v3 plan](../plans/2026-08-02-continuous-streaming-v3.md).
Measurement and documentation only; no production code was changed.

> **M1 rebuild warning.** Milestone M1 rewrites both benchmark harnesses
> (`crates/calc-flow/benches/core.rs` and
> `crates/calc-flow/benches/allocation_regression.rs`, per plan tasks M1.1,
> M2.5, and M4.2). Every number in this document is a **pre-M1 reference
> point only**. After M1 lands, both the Criterion baseline and the
> allocation baseline MUST be rebuilt from the rewritten harnesses; the
> thresholds must not be silently carried over or relaxed.

> **Amendment (2026-08-03).** This document was extended to reference the
> scheduled-CI benchmark artifacts of run `30732416522` alongside the local
> captures: new §1.1 (commit delta audit), §5.2 and §5.3 (corroborating CI
> reference numbers), §6.1 (why the local allocation capture is the sole
> allocation source of truth), §8.1 (M7 regression-gate attribution method),
> and Appendix A (full per-scale CI tables). The locally captured numbers in
> §5 and §6 remain the pre-M1 reference baseline; the CI numbers are the
> corroborating fleet reference, labeled by commit and machine class.

## 1. Baseline identity

| Item                              | Value                                                                                                                                                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Baseline commit (HEAD of `main`)  | `e9c4127d2023a4640e4f65085ef2a1b715cbf3f6`                                                                                                                                                                         |
| Commit subject                    | `Merge pull request #75 from wegamekinglc/fix/streaming-research-plan-review`                                                                                                                                      |
| Capture date                      | 2026-08-03 (local, CST)                                                                                                                                                                                            |
| Criterion baseline name           | `continuous-streaming-v3-m0`                                                                                                                                                                                       |
| Allocation frozen product SHA     | `2ac7e97c1549baf0e97849d5823f65e7dd298e99`                                                                                                                                                                         |
| Allocation harness commit         | `fe34d7dcd5bfd66c9e97c79d540380f58ee1a04d`                                                                                                                                                                         |
| Working-tree state during capture | Tracked files clean at `e9c4127`; only untracked M0 spec/API artifacts under `.codex/artifacts/` (not compiled or measured)                                                                                        |
| CI fleet reference run            | Scheduled `benchmarks.yml` run `30732416522` (2026-08-02, `schedule` trigger, success) at commit `3bae1a6d109f4c21df8f2251b47669a6e4d34666` — <https://github.com/wegamekinglc/calc-flow/actions/runs/30732416522> |

Every baseline number below is tied to commit `e9c4127`. The allocation
measurement additionally records its own provenance evidence (frozen-file
SHA-256 values, toolchain, environment) inside the JSON reports. The CI
reference numbers in §5.2, §5.3, and Appendix A are tied to commit
`3bae1a6`; §1.1 audits the delta between the two commits.

### 1.1 Commit delta audit: CI reference `3bae1a6` vs baseline `e9c4127`

The CI reference run predates the local baseline by 32 commits
(`git log 3bae1a6..e9c4127`). Both commits descend from the allocation
frozen product SHA `2ac7e97` (`2ac7e97` → `3bae1a6` → `e9c4127`).
The delta is **not** "docs changes plus the allocation-harness merge" as
initially expected — it also contains the DAL-38 runtime-envelope refactor
series, which touches runtime source. This is flagged prominently here:

| Group                                            | Commits                                                                                                                                                                   | Runtime-affecting?                                                                                                                  |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Runtime-envelope refactor series (DAL-38)        | `5cace7a`, `b902658`, `4222777`, `c6d9b93`, `f777fa0` (+ merges `2ae6b65` #57, `5b31005` #66)                                                                             | **YES** — touches `crates/calc-flow/src/runtime/envelope.rs`, `runtime/mod.rs`, `pipeline.rs`, `pipeline/control.rs`, `operator.rs` |
| Allocation harness, CI wiring, and harness tests | `2174698`, `bb08d0e`, `f3af9f9`, `bc78beb`, `89f1a2d`, `b2f0316`, `4f64103`, `18ba3e7`, `0835864`, `eaac319`, `6ddbb9e`, `fe34d7d`, `0c8ae5b`, `f880fb8` (#72), `8922c5d` | No — `benches/`, `.github/workflows/`, `scripts/`, harness-only                                                                     |
| Docs and streaming plan                          | `ba08b2d` (#70), `0905b22` (#71), `cb2321e`, `1dceb9e` (#73), `1417c37`, `1d8bc75`, `8360529` (#74), `959cbe0`, `eeac0a4`, `e9c4127` (#75)                                | No — `docs/`, `AGENTS.md`, `CLAUDE.md`, `.claude/agents/`                                                                           |

Files changed across the whole delta (`git diff --name-only 3bae1a6..e9c4127`):
`.claude/agents/cf-critic.md`, `.codacy.yml`, `.github/workflows/ci.yml`,
`AGENTS.md`, `CHANGELOG.md`, `CLAUDE.md`, `Cargo.lock`,
`crates/calc-flow/Cargo.toml`, `crates/calc-flow/benches/allocation_regression.rs`,
`crates/calc-flow/src/operator.rs`, `crates/calc-flow/src/pipeline.rs`,
`crates/calc-flow/src/pipeline/control.rs`,
`crates/calc-flow/src/pipeline/runtime_envelope_tests.rs`,
`crates/calc-flow/src/pipeline/signal_allocation_tests.rs`,
`crates/calc-flow/src/runtime/envelope.rs`,
`crates/calc-flow/src/runtime/mod.rs`, `docs/README.md`,
`docs/introduction.md`, `docs/research/2026-08-02-arroyo-risingwave-streaming-research.md`,
`docs/runtime-envelope.md`, `docs/superpowers/plans/2026-08-02-continuous-streaming-v3.md`,
`scripts/run_rust_tests.py`, `scripts/test_release_config.py`,
`scripts/test_run_rust_tests.py`.

**Conclusion.** The runtime-envelope series is a crate-private refactor
(the envelope is an internal contract per `docs/runtime-envelope.md`), and
three independent neutrality checks hold at `e9c4127`:

1. The allocation comparison `2ac7e97` → `e9c4127` (§6), which spans the
   entire series, shows **zero allocation drift** on all five fixed
   workloads.
2. Every verification surface in §4 is green at `e9c4127`.
3. The series' own review/test commits landed before the harness merge
   `f880fb8`, which froze the post-series allocation profile as the gate
   reference.

Nevertheless, the series is runtime code: the CI reference numbers
(`3bae1a6`) and the local baseline numbers (`e9c4127`) describe **different
code states on different machines**, which is one more reason the two data
sets must never be compared directly (§8.1). For M7 attribution purposes
the delta is treated as performance-neutral-by-evidence, not
performance-neutral-by-assumption; any future doubt is resolved by a
paired same-machine rerun, not by cross-referencing §5 against §5.2.

## 2. Machine and toolchain

This is a **virtualized, shared development workstation (WSL2)**, not a
dedicated benchmark runner. See §9 for the noise policy this implies.

| Item                  | Value                                                                                     |
| --------------------- | ----------------------------------------------------------------------------------------- |
| CPU model             | 13th Gen Intel(R) Core(TM) i9-13900HX                                                     |
| Topology              | 1 socket, 16 physical cores, 2 threads/core, 32 logical CPUs                              |
| Total memory          | 32,669,504 kB (≈ 31.2 GiB) visible to the WSL2 guest                                      |
| Swap                  | 64 GiB (≈ 53 GiB free during capture)                                                     |
| OS / kernel           | Linux `5.15.167.4-microsoft-standard-WSL2` (`#1 SMP Tue Nov 5 00:21:55 UTC 2024`), x86_64 |
| Virtualization        | `wsl` (systemd-detect-virt)                                                               |
| CPU governor          | Unavailable in the WSL2 guest                                                             |
| Power supplies        | `AC1`, `BAT1` present; no frequency control exposed                                       |
| rustc                 | `rustc 1.88.0 (6b00bc388 2025-06-23)`, host `x86_64-unknown-linux-gnu`, LLVM 20.1.5       |
| cargo                 | `cargo 1.88.0 (873a06493 2025-05-10)`                                                     |
| cargo-llvm-cov        | `0.8.7`                                                                                   |
| uv                    | `0.11.19 (x86_64-unknown-linux-gnu)`                                                      |
| Python (project venv) | `3.13.9` (interpreter at `/home/wegamekinglc/anaconda3/bin/python3`)                      |
| maturin               | `1.14.1`                                                                                  |
| ruff                  | `0.16.0`                                                                                  |
| node                  | `v20.20.2`                                                                                |
| npm                   | `10.8.2`                                                                                  |
| criterion (dev-dep)   | `0.8.0` with `async_tokio`                                                                |
| allocation-counter    | `0.8.1` (pinned `=0.8.1`)                                                                 |
| DataFusion            | `=54.0.0`                                                                                 |

## 3. Target directory layout

All build and measurement artifacts stayed under the repository `target/`
tree; scratch git worktrees lived under `/tmp` and were removed after the
runs.

| Path                                       | Contents                                                                                                                                                                                                                  |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `target/` (default)                        | fmt/clippy/test/doc builds (`target/debug`, `target/doc`)                                                                                                                                                                 |
| `target/llvm-cov-target/`                  | cargo-llvm-cov instrumented build and coverage data                                                                                                                                                                       |
| `target/cargo/`                            | `CARGO_TARGET_DIR` for all bench builds; Criterion output at `target/cargo/criterion/<case>/continuous-streaming-v3-m0/`                                                                                                  |
| `target/cargo/release/deps/`               | Prebuilt bench binaries (`core-*`, `allocation_regression-*`)                                                                                                                                                             |
| `target/allocation-regression/`            | `baseline-2ac7e97.json`, `candidate-e9c4127.json`, `compare-m0.json`                                                                                                                                                      |
| `target/m0-baseline-logs/`                 | Per-surface logs and `criterion-estimates.json`                                                                                                                                                                           |
| `/tmp/cf-alloc-harness` (removed)          | Detached worktree at `fe34d7d` for the `--role baseline` run                                                                                                                                                              |
| `/tmp/cf-alloc-candidate` (removed)        | Detached worktree at `e9c4127` for the `--role candidate` run                                                                                                                                                             |
| `target/benchmark-results/ci-30732416522/` | Mined CI artifacts: four pytest-benchmark JSONs, extracted Criterion estimates (`criterion-estimates.json`), and the full Criterion report tree (`criterion-report/`) — local copies of the 90-day-retention CI artifacts |

## 4. Verification surface results

All commands from the Commands section of `AGENTS.md` were run at `e9c4127`.
**Every surface passes.** Local-environment retries that were needed are
listed explicitly in §7; none indicate a repository defect.

### 4.1 Rust core

| Check           | Command                                                                     | Result | Key numbers                                                                                                                                                                       |
| --------------- | --------------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dependency sync | `uv sync --extra dev`                                                       | PASS   | 43 packages resolved, environment already current                                                                                                                                 |
| Format          | `cargo fmt --all --check`                                                   | PASS   | No diffs                                                                                                                                                                          |
| Lints           | `cargo clippy --workspace --all-targets --all-features -- -D warnings`      | PASS   | Zero warnings                                                                                                                                                                     |
| Rust tests      | `uv run python scripts/run_rust_tests.py`                                   | PASS   | 335 tests, 0 failed: 266 `calc-flow` (41 lib + 225 integration) + 69 `calc-flow-python` lib tests (serial, 1.79 s); core test-profile compile 20 m 06 s under parallel build load |
| Coverage        | `cargo llvm-cov --workspace --all-features --fail-under-lines 90`           | PASS   | **Lines 90.77 %** (10,723/11,813) ≥ 90 % floor; regions 88.63 %; functions 81.55 %                                                                                                |
| Rustdoc         | `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps` | PASS   | Generated without warnings                                                                                                                                                        |

### 4.2 Python binding and adapters

| Check               | Command                                           | Result | Key numbers                                                                  |
| ------------------- | ------------------------------------------------- | ------ | ---------------------------------------------------------------------------- |
| Native module build | `uv run maturin develop`                          | PASS   | `calc_flow-2.0.0` editable install rebuilt at `e9c4127` (abi3 wheel, cp313)  |
| Python tests        | `JAX_PLATFORMS=cpu uv run pytest python/tests -q` | PASS   | **426 passed, 0 failed** in 11.61 s (re-run against the fresh native module) |
| Lint                | `uv run ruff check .`                             | PASS   | No findings                                                                  |
| Format              | `uv run ruff format --check .`                    | PASS   | No diffs                                                                     |

### 4.3 Studio backend

| Check                    | Command                                                                             | Result | Key numbers                                                             |
| ------------------------ | ----------------------------------------------------------------------------------- | ------ | ----------------------------------------------------------------------- |
| Backend tests + coverage | `cd web-ui/backend && uv run --project . --extra dev pytest --cov=calc_flow_studio` | PASS   | **150 passed, 4 skipped** in 11.87 s; coverage **93.88 %** ≥ 85 % floor |

### 4.4 Studio frontend and generated contract

| Check                | Command                                                                                                                          | Result | Key numbers                                                                     |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------- |
| Dependency install   | `npm ci`                                                                                                                         | PASS   | 165 packages, audited 166                                                       |
| Build                | `npm run build`                                                                                                                  | PASS   | `tsc -b && vite build`, built in 230 ms                                         |
| Unit tests           | `npm test` (`vitest run`)                                                                                                        | PASS   | **18 files / 182 tests passed**, duration 3.58 s                                |
| Generated-file drift | `npm run sync:api` + SHA-256 comparison of `web-ui/openapi.json`, `web-ui/src/api/schema.d.ts`, `schemas/project-v2.schema.json` | PASS   | All three files byte-identical after regeneration; working tree left unmodified |

`npm run test:e2e` (Playwright) was not part of the M0.3 command list and
was not run.

### 4.5 Supply-chain helpers

`cargo audit`, `cargo deny --locked check`, and
`python -m unittest scripts.test_*` are in the `AGENTS.md` command groups
but outside the M0.3 surface list; they were not run here. They remain part
of the M7 release gate.

## 5. Criterion baseline (`benches/core.rs`)

Command (binaries pre-built into `target/cargo`, run on an idle machine,
1-minute load average ≈ 0.2):

```bash
CARGO_TARGET_DIR=$PWD/target/cargo \
  cargo bench --locked -p calc-flow --bench core -- \
  --save-baseline continuous-streaming-v3-m0
```

Criterion 0.8 defaults: 3 s warm-up + 5 s measurement per case, 100 samples.
The named baseline is stored at
`target/cargo/criterion/<case>/continuous-streaming-v3-m0/estimates.json`
(verified byte-identical to the `new/` analysis of the same run).
All intervals are Criterion-reported 95 % confidence intervals. Note that
the Criterion console headline for `b.iter()` cases prints the **slope**
statistic; all three location statistics are recorded below.

| Case                                           | Mean (point) [95 % CI]     | Median (point) [95 % CI]   | Slope (point) [95 % CI]    | Std dev  | Outliers |
| ---------------------------------------------- | -------------------------- | -------------------------- | -------------------------- | -------- | -------- |
| `compile/expression`                           | 11.374 µs [11.258, 11.510] | 11.222 µs [11.159, 11.318] | 11.232 µs [11.165, 11.305] | 0.655 µs | 6 %      |
| `execute/expression_1024_rows`                 | 374.70 µs [371.17, 378.58] | 371.00 µs [367.35, 373.43] | 376.10 µs [370.50, 382.52] | 19.13 µs | 7 %      |
| `execute/datafusion_runtime_new`               | 18.696 ns [18.584, 18.812] | 18.601 ns [18.456, 18.701] | 18.611 ns [18.508, 18.720] | 0.591 ns | 2 %      |
| `execute/datafusion_runtime_new_register_udfs` | 54.826 ns [54.494, 55.174] | 54.575 ns [54.081, 54.973] | 54.817 ns [54.423, 55.231] | 1.757 ns | 3 %      |
| `execute/external_passthrough_1000_rows`       | 2.0229 µs [2.0105, 2.0362] | 2.0131 µs [2.0017, 2.0342] | 1.9920 µs [1.9808, 2.0051] | 0.066 µs | 4 %      |
| `execute/external_plan_table_requirement`      | 236.57 ps [233.11, 240.27] | 230.27 ps [228.73, 232.21] | 238.08 ps [234.55, 242.04] | 0.018 ns | 22 %     |
| `json/canonical_nested`                        | 3.5184 µs [3.5022, 3.5354] | 3.5056 µs [3.4879, 3.5228] | 3.5193 µs [3.4963, 3.5449] | 0.085 µs | 4 %      |

Reading notes:

- `execute/datafusion_runtime_new` at ≈ 18.7 ns confirms that
  `DataFusionRuntime::new` is a lazy constructor (see the 2026-07-18
  lazy-DataFusion-runtime handoff); eager session creation cost is not in
  this case. If M2 changes runtime initialization, a separate eager-path
  case must be added rather than comparing against this number.
- `execute/external_plan_table_requirement` is a sub-nanosecond predicate;
  its 22 % outlier fraction reflects timer resolution on WSL2, not work.
- The machine was otherwise idle during the measurement window, but this is
  still a shared WSL2 workstation; treat these as same-machine-only
  reference points (§9).

### 5.1 Coverage advisory against the plan's M0.3 list

Plan task M0.3 names five areas: expression, SQL, external passthrough,
DataFusion runtime creation, checkpoint persistence. Current
`benches/core.rs` covers expression (compile + execute), external
passthrough, and DataFusion runtime creation. Two gaps:

1. **No SQL execution timing case.** SQL appears only as an allocation case
   (`builtin_sql_one_node`) in `allocation_regression.rs`. A
   `execute/sql_1024_rows` Criterion case belongs in `benches/core.rs`
   when M2.5 touches that file (the plan already schedules core.rs changes
   in M2.5 and M4.2).
2. **No checkpoint persistence timing case.** Checkpoint
   serialization/atomic-write/recovery timing exists only in the
   pytest-benchmark suite (`benchmarks/test_runtime.py`). A Criterion case
   for checkpoint commit (write + rename) belongs in `benches/core.rs`
   alongside the M4/M5 state-backend work, so M5's manifest-v3 rewrite can
   be compared against a native Rust baseline.

Both are advisory; adding benchmark cases is scheduled milestone work, not
part of this measurement-only task.

### 5.2 Corroborating CI reference — Criterion at `3bae1a6` (run `30732416522`)

Source: artifact `benchmark-rust-30732416522` (full `target/criterion`
report uploaded by the scheduled `benchmarks.yml` workflow, `cargo bench -p
calc-flow`, 90-day retention) from
<https://github.com/wegamekinglc/calc-flow/actions/runs/30732416522>.
Estimates extracted from each case's `new/estimates.json`; preserved copy
under `target/benchmark-results/ci-30732416522/`. These numbers describe
commit `3bae1a6` on the CI runner — a **different code state** (§1.1) on a
**different machine class** than the §5 local baseline. They corroborate
orders of magnitude only; they must never be compared directly against §5
(§8.1).

CI runner identity (from the workflow and the artifacts' `machine_info`):

| Item               | Value                                                                      |
| ------------------ | -------------------------------------------------------------------------- |
| Workflow           | `.github/workflows/benchmarks.yml`, `schedule` trigger (cron `17 3 * * *`) |
| Runner label       | `ubuntu-latest` (GitHub-hosted)                                            |
| VM / kernel        | Azure VM, Linux `6.17.0-1020-azure`                                        |
| CPU                | AMD EPYC 9V74 80-Core Processor, 4 vCPUs visible, 2.87 GHz                 |
| Memory             | 16 GB (GitHub-hosted standard runner specification)                        |
| Python             | 3.13.14 (local baseline used 3.13.9 — dependency fingerprints differ)      |
| Artifact retention | 90 days (`benchmark-rust-*`, `benchmark-<scale>-*`)                        |

Side-by-side presentation of the seven shared Criterion cases, labeled by
commit and machine class (mean point estimate [95 % CI]):

| Case                                           | Local `e9c4127` — WSL2, i9-13900HX (§5) | CI `3bae1a6` — ubuntu-latest, EPYC 4 vCPU |
| ---------------------------------------------- | --------------------------------------- | ----------------------------------------- |
| `compile/expression`                           | 11.374 µs [11.258, 11.510]              | 17.455 µs [17.420, 17.499]                |
| `execute/expression_1024_rows`                 | 374.70 µs [371.17, 378.58]              | 522.76 µs [521.71, 524.59]                |
| `execute/datafusion_runtime_new`               | 18.696 ns [18.584, 18.812]              | 31.158 ns [30.796, 31.534]                |
| `execute/datafusion_runtime_new_register_udfs` | 54.826 ns [54.494, 55.174]              | 84.733 ns [84.564, 84.922]                |
| `execute/external_passthrough_1000_rows`       | 2.0229 µs [2.0105, 2.0362]              | 3.3677 µs [3.3590, 3.3780]                |
| `execute/external_plan_table_requirement`      | 236.57 ps [233.11, 240.27]              | 503.30 ps [500.19, 507.08]                |
| `json/canonical_nested`                        | 3.5184 µs [3.5022, 3.5354]              | 4.4530 µs [4.4458, 4.4631]                |

The CI runner is consistently slower per case (≈ 1.3–1.7×), consistent with
a 4-vCPU shared Azure VM versus a 32-thread desktop-class CPU; the ratio is
a machine property, not evidence about the code delta in either direction.

### 5.3 Corroborating CI reference — pytest-benchmark scales at `3bae1a6`

Source: artifacts `benchmark-{overhead,small,standard,nightly}-30732416522`
(pytest-benchmark JSON, 40 cases per scale, 14 measurement-scope groups per
scale, `commit_info.id = 3bae1a6d109f4c21…`, `dirty = false`). Preserved
under `target/benchmark-results/ci-30732416522/<scale>.json`. Full
per-case tables are in Appendix A.

| Scale      | Rows / elements / matrix    | Cases | Artifact preserved at                                   |
| ---------- | --------------------------- | ----- | ------------------------------------------------------- |
| `overhead` | 1,000 / 1,000 / 16          | 40    | `target/benchmark-results/ci-30732416522/overhead.json` |
| `small`    | 10,000 / 10,000 / 64        | 40    | `target/benchmark-results/ci-30732416522/small.json`    |
| `standard` | 100,000 / 100,000 / 256     | 40    | `target/benchmark-results/ci-30732416522/standard.json` |
| `nightly`  | 1,000,000 / 1,000,000 / 512 | 40    | `target/benchmark-results/ci-30732416522/nightly.json`  |

pytest-benchmark reports min/mean/median/std-dev/rounds rather than
Criterion-style confidence intervals; a 95 % interval can be reconstructed
per case as `mean ± 1.96 · stddev / √rounds` when the M7 gate needs one
(§8.1). The JSONs also carry the contract-v2 machine/dependency/workload
fingerprints that gate any future comparison.

## 6. Allocation regression frozen baseline (`benches/allocation_regression.rs`)

The harness is a custom provenance-locked binary (not Criterion). The
frozen baseline is defined by `FIXED_BASELINE_SHA`
(`2ac7e97c1549baf0e97849d5823f65e7dd298e99`); per CI wiring
(`.github/workflows/ci.yml`, `rust-core` job), the baseline report is
produced on demand from a detached worktree at the harness commit
(`fe34d7d`), and a candidate report from the revision under test. This
baseline capture replicates that wiring locally with the same fixed
measurement scale (1,000 warm-up + 10,000 measured dispatches × 10
repetitions; counts asserted stable across repetitions).

Provenance of this capture:

| Item                                              | Value                                                                                                                                                                                  |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline report                                   | `target/allocation-regression/baseline-2ac7e97.json` (role `baseline`, product SHA `2ac7e97`, measured in `/tmp/cf-alloc-harness` at `fe34d7d`)                                        |
| Candidate report                                  | `target/allocation-regression/candidate-e9c4127.json` (role `candidate`, product SHA `e9c4127`, measured in `/tmp/cf-alloc-candidate`)                                                 |
| Comparison                                        | `target/allocation-regression/compare-m0.json` — **valid, passed** (candidate ≤ baseline on every case and repetition: zero allocation drift between the frozen product and `e9c4127`) |
| Frozen `Cargo.lock` SHA-256                       | `4e7069ec6f20f828a59bf39a7c1f2e0fcaff74cb61095133ff63c4abb227458f`                                                                                                                     |
| Frozen `crates/calc-flow/Cargo.toml` SHA-256      | `1d9a6c60cbc25d16d9cee09bde7a7756f4f08f3c85aeb3fd8e305419ad4c8eee`                                                                                                                     |
| Frozen `benches/allocation_regression.rs` SHA-256 | `63f334634c0b16a5ce34c0b7b8ef0c8654da0263a084ad5fd68bebc845e68cea`                                                                                                                     |
| allocation-counter                                | `0.8.1`, registry checksum `beb9e990c0a33699f1984d85a6abead615ccc72dd8130bf3e15dcabe2ca149c9` (matches the harness `ALLOCATION_COUNTER_CHECKSUM` constant)                             |
| Report schema version                             | `1`                                                                                                                                                                                    |

Frozen baseline values (identical in the candidate report; per-case totals
over 10,000 measured dispatches, stable across all 10 repetitions):

| Case                               | `count_total` | `bytes_total` | calls / dispatch | bytes / dispatch | calls / node-dispatch | bytes / node-dispatch |
| ---------------------------------- | ------------- | ------------- | ---------------- | ---------------- | --------------------- | --------------------- |
| `external_payload_one_node`        | 500,001       | 136,770,064   | 50.0001          | 13,677.0064      | 50.0001               | 13,677.0064           |
| `external_table_one_node`          | 500,001       | 136,750,064   | 50.0001          | 13,675.0064      | 50.0001               | 13,675.0064           |
| `external_table_three_way_fan_out` | 1,230,001     | 255,630,064   | 123.0001         | 25,563.0064      | 30.7500               | 6,390.7516            |
| `builtin_expression_one_node`      | 34,090,001    | 3,726,780,064 | 3,409.0001       | 372,678.0064     | 3,409.0001            | 372,678.0064          |
| `builtin_sql_one_node`             | 31,040,001    | 3,456,420,064 | 3,104.0001       | 345,642.0064     | 3,104.0001            | 345,642.0064          |

Allocation counts are deterministic integers (the harness asserts equality
across repetitions), so machine timing noise does not affect them; the
provenance lock (clean worktree, frozen-file hashes, toolchain equality) is
what comparability depends on. **M1 rewrites this harness; these values are
pre-M1 reference points and the baseline must be re-established from the
rewritten harness, not grandfathered.**

### 6.1 Why this local capture is the sole allocation source of truth

The CI benchmark artifacts cannot cover the allocation baseline, for two
independent reasons:

1. **The harness did not exist at the CI run's commit.**
   `benches/allocation_regression.rs` and its CI wiring landed in merge
   `f880fb8` (harness commit `fe34d7d`), which sits in the
   `3bae1a6..e9c4127` delta (§1.1) — *after* the scheduled run
   `30732416522`. The `benchmark-rust-*` artifact at `3bae1a6` contains
   only the Criterion `core` bench, and no retroactive CI run can produce
   an allocation report for a commit that lacks the bench target.
2. **The harness's provenance lock pins more than the code.** A valid
   report requires a clean worktree at the harness commit, byte-identical
   frozen files, the pinned `allocation-counter = 0.8.1`, and — for any
   comparison — an identical toolchain (`rustc -Vv`, cargo version, host,
   target) in both reports. Reports from other machines or toolchains are
   rejected by `--compare` before any number is examined.

The local capture in §6 was produced at `e9c4127` with the CI-identical
command lines and worktree layout, which is exactly what plan task M0.3
requires ("记录 `allocation_regression` 当前的冻结分配基线"). It is
therefore the **sole** allocation source of truth for the v3 program. Any
future allocation comparison must re-run the harness on a matching
toolchain against `target/allocation-regression/baseline-2ac7e97.json`;
CI or third-party numbers cannot substitute.

## 7. Explicit list of failing / anomalous items

No repository check fails at `e9c4127`. Three first-attempt failures were
caused by this local environment and were recovered by replicating the CI
environment exactly; they are recorded faithfully for future operators:

1. **`cargo llvm-cov` attempt 1 — FAILED (exit 127).** The compiled PyO3
   test binary could not load `libpython3.13.so.1.0`
   (`error while loading shared libraries`). CI passes because
   `actions/setup-python` exports the interpreter library path.
   **Recovery:** re-ran with
   `LD_LIBRARY_PATH=/home/wegamekinglc/anaconda3/lib` (the project venv's
   interpreter library directory).
2. **`cargo llvm-cov` attempt 2 — HUNG** on
   `pipeline::tests::async_deadline_crossing_rolls_back_state_and_skips_downstream`
   with default parallel test threads. CI sets `RUST_TEST_THREADS: 1`
   job-wide in `.github/workflows/ci.yml`; `AGENTS.md` does not mention
   this requirement for local coverage runs.
   **Recovery:** re-ran with `RUST_TEST_THREADS=1` (CI-equivalent); full
   suite then passed. **Follow-up (not delivered by this PR):** the
   `RUST_TEST_THREADS=1` requirement for local `cargo llvm-cov` runs should
   be documented in `AGENTS.md`; this wrap-up PR intentionally leaves
   `AGENTS.md` unchanged, so the fix is owned by a separate docs follow-up
   PR or by milestone M1.
3. **`uv run maturin develop` attempt 1 — FAILED (exit 1).** Maturin
   refuses to run when both `VIRTUAL_ENV` (set by `uv run`) and
   `CONDA_PREFIX` (set by this shell's active conda base) are present.
   **Recovery:** re-ran with `CONDA_PREFIX` unset
   (`env -u CONDA_PREFIX uv run maturin develop`). The first
   `pytest python/tests` run (426 passed) executed against the previously
   installed native module; after the successful rebuild, the suite was
   re-run against the fresh `e9c4127` native module: 426 passed.

Additional observations:

- The llvm-cov **text table's leading columns are Regions, then Functions,
  then Lines**. The 90 % floor gates on Lines (90.77 % here), not on the
  88.63 % Regions figure. Probed empirically: the gate fails at 99 and
  passes at 85 against the same data.
- Criterion 0.8's console headline for `b.iter()` cases is the **slope**
  statistic; the saved `estimates.json` additionally holds mean, median,
  std-dev, and median-absolute-deviation, each with 95 % CIs. All are
  recorded in §5.
- During the candidate allocation step, a `uv run python` invocation from
  inside `/tmp/cf-alloc-candidate` created a throwaway `.venv` in that
  scratch worktree (uv project discovery). No effect on measurements; the
  scratch worktrees were removed afterwards.

## 8. Noise environment statement and regression gate rule

- Machine: shared WSL2 development workstation (virtualized; background
  IDE/indexer processes present but mostly idle; 1-minute load ≈ 0.2 during
  the Criterion window and ≈ 0.4 before the allocation windows).
- Repetitions: Criterion 100 samples/case (3 s warm-up + 5 s measurement);
  allocation harness 10 repetitions × 10,000 dispatches with stability
  assertions.
- Reduction: this document records Criterion's own point estimates with
  95 % CIs (mean, median, slope); no cross-run reduction was needed for a
  single baseline capture.
- Per plan task M0.3 and M7.1: the performance regression gate is **5 %**,
  but a regression claim is only valid with **same-machine paired data and
  confidence-interval support** — never by comparing single point
  estimates, and never by comparing against this document from a different
  machine, dependency set, or power state (contract-v2 fingerprint rule).

### 8.1 M7 regression-gate attribution method

Attribution of any future performance change MUST follow this order:

1. **Primary — CI-to-CI within the same scheduled workflow.** Compare a
   later `benchmarks.yml` run's artifacts against an earlier main-branch
   run's artifacts (e.g., run `30732416522`) at the same scale and bench
   target. Same workflow, same `ubuntu-latest` runner class, same
   measurement scopes; the pytest-benchmark JSONs carry contract-v2
   fingerprints that must match. For Criterion cases, use the saved
   `estimates.json` confidence intervals directly; for pytest-benchmark
   cases, reconstruct a 95 % interval as `mean ± 1.96 · stddev / √rounds`.
   Per `benchmarks/README.md`, no CI gate applies until at least 20
   comparable main-branch samples exist on stable runners — until then
   these artifacts are informational, and the 5 % gate requires
   confidence-interval support from paired runs, never a single point
   estimate.
2. **Secondary — local paired same-machine measurement.** Used when CI
   data is insufficient (change too fresh for fleet data, or a path the
   fleet does not exercise): interleaved baseline/branch runs, at least
   two repetitions each plus a same-ref spread measurement, per-case **min**
   reduction, regression bar at 2× the same-ref spread, matching
   contract-v2 fingerprints. This is the discipline §5 was captured under.
3. **Absolute prohibition.** Numbers from this WSL2 development
   workstation (§5, §6) and CI runner numbers (§5.2, §5.3, Appendix A)
   must **never** be compared directly. They differ in machine class
   (desktop i9-13900HX under WSL2 vs shared 4-vCPU AMD EPYC Azure VM),
   dependency fingerprints (e.g., Python 3.13.9 vs 3.13.14), and — for the
   specific `3bae1a6` ↔ `e9c4127` pairing — in code state as well (§1.1).
   A ratio computed across these axes attributes machine and version
   differences to code, which is exactly the false-regression mode this
   baseline exists to prevent.

## 9. Reproduction commands

```bash
# Verification surfaces (see AGENTS.md; local env notes from §7 apply)
uv sync --extra dev
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
uv run python scripts/run_rust_tests.py
RUST_TEST_THREADS=1 LD_LIBRARY_PATH="$(uv run python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')" \
  cargo llvm-cov --workspace --all-features --fail-under-lines 90
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
env -u CONDA_PREFIX uv run maturin develop
JAX_PLATFORMS=cpu uv run pytest python/tests -q
uv run ruff check . && uv run ruff format --check .
(cd web-ui/backend && uv run --project . --extra dev pytest --cov=calc_flow_studio)
(cd web-ui && npm ci && npm run build && npm test)

# Criterion baseline (build first, then measure on an idle machine)
CARGO_TARGET_DIR=$PWD/target/cargo cargo bench --locked -p calc-flow --no-run
CARGO_TARGET_DIR=$PWD/target/cargo cargo bench --locked -p calc-flow --bench core -- \
  --save-baseline continuous-streaming-v3-m0

# Allocation frozen baseline + candidate + compare (CI-identical wiring)
REPO=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow
git worktree add --detach /tmp/cf-alloc-harness fe34d7dcd5bfd66c9e97c79d540380f58ee1a04d
git worktree add --detach /tmp/cf-alloc-candidate e9c4127d2023a4640e4f65085ef2a1b715cbf3f6
(cd /tmp/cf-alloc-harness && CARGO_TARGET_DIR=$REPO/target/cargo \
  cargo bench --locked -p calc-flow --bench allocation_regression -- \
  --warmup-dispatches 1000 --measured-dispatches 10000 --repetitions 10 \
  --cases all-existing-data --role baseline \
  --output $REPO/target/allocation-regression/baseline-2ac7e97.json)
(cd /tmp/cf-alloc-candidate && CARGO_TARGET_DIR=$REPO/target/cargo \
  cargo bench --locked -p calc-flow --bench allocation_regression -- \
  --warmup-dispatches 1000 --measured-dispatches 10000 --repetitions 10 \
  --cases all-existing-data --role candidate \
  --output $REPO/target/allocation-regression/candidate-e9c4127.json)
(cd /tmp/cf-alloc-candidate && CARGO_TARGET_DIR=$REPO/target/cargo \
  cargo bench --locked -p calc-flow --bench allocation_regression -- \
  --compare $REPO/target/allocation-regression/baseline-2ac7e97.json \
            $REPO/target/allocation-regression/candidate-e9c4127.json)
git worktree remove --force /tmp/cf-alloc-harness
git worktree remove --force /tmp/cf-alloc-candidate
```

## Appendix A. CI pytest-benchmark tables (run 30732416522, commit 3bae1a6)

Source artifacts: benchmark-{overhead,small,standard,nightly}-30732416522 from
<https://github.com/wegamekinglc/calc-flow/actions/runs/30732416522> (90-day retention;
preserved copies under target/benchmark-results/ci-30732416522/). CI runner:
ubuntu-latest, AMD EPYC 9V74, 4 vCPU, Azure kernel 6.17.0-1020-azure, Python 3.13.14.
Times are per-loop statistics reported by pytest-benchmark (min / mean / std dev / rounds).
Case names have the redundant scale prefix inside the parameter brackets elided.

### A.1 Scale `overhead` (40 cases, commit `3bae1a6`, CI runner)

| Case                                                             | Min       | Mean      | Std dev   | Rounds |
| ---------------------------------------------------------------- | --------- | --------- | --------- | ------ |
| `test_backend_kernel[array_elementwise-numpy]`                   | 4.91 us   | 5.15 us   | 1.24 us   | 3124   |
| `test_backend_kernel[array_elementwise-jax]`                     | 45.35 us  | 57.68 us  | 8.14 us   | 1824   |
| `test_backend_kernel[array_mean-numpy]`                          | 4.35 us   | 4.54 us   | 948.88 ns | 7729   |
| `test_backend_kernel[array_mean-jax]`                            | 11.88 us  | 14.28 us  | 4.14 us   | 3686   |
| `test_backend_kernel[array_matrix_multiplication-numpy]`         | 2.27 us   | 2.41 us   | 753.80 ns | 12043  |
| `test_backend_kernel[array_matrix_multiplication-jax]`           | 11.22 us  | 12.33 us  | 3.71 us   | 2796   |
| `test_backend_kernel[array_transpose_reshape-numpy]`             | 2.81 us   | 2.97 us   | 727.08 ns | 13863  |
| `test_backend_kernel[array_transpose_reshape-jax]`               | 89.18 us  | 97.35 us  | 12.88 us  | 1611   |
| `test_batch_ownership[numpy]`                                    | 7.79 us   | 8.29 us   | 2.07 us   | 6053   |
| `test_batch_ownership[jax]`                                      | 15.17 us  | 16.24 us  | 3.06 us   | 3985   |
| `test_plan_end_to_end[array_elementwise-numpy]`                  | 90.83 us  | 100.68 us | 11.06 us  | 1750   |
| `test_plan_end_to_end[array_elementwise-jax]`                    | 254.16 us | 271.76 us | 15.69 us  | 859    |
| `test_plan_end_to_end[array_mean-numpy]`                         | 70.51 us  | 76.51 us  | 8.97 us   | 1924   |
| `test_plan_end_to_end[array_mean-jax]`                           | 146.43 us | 157.03 us | 13.58 us  | 1221   |
| `test_plan_end_to_end[array_matrix_multiplication-numpy]`        | 63.59 us  | 71.37 us  | 14.04 us  | 1941   |
| `test_plan_end_to_end[array_matrix_multiplication-jax]`          | 143.75 us | 154.02 us | 12.01 us  | 1192   |
| `test_plan_end_to_end[array_transpose_reshape_diagnostic-numpy]` | 76.25 us  | 83.34 us  | 10.35 us  | 1694   |
| `test_plan_end_to_end[array_transpose_reshape_diagnostic-jax]`   | 232.17 us | 253.12 us | 33.13 us  | 898    |
| `test_provider_boundary[array_elementwise-numpy]`                | 59.82 us  | 64.74 us  | 6.86 us   | 2403   |
| `test_provider_boundary[array_elementwise-jax]`                  | 207.37 us | 222.61 us | 16.09 us  | 1065   |
| `test_provider_boundary[array_mean-numpy]`                       | 36.07 us  | 39.95 us  | 5.98 us   | 2711   |
| `test_provider_boundary[array_mean-jax]`                         | 104.55 us | 127.74 us | 23.08 us  | 1441   |
| `test_provider_boundary[array_matrix_multiplication-numpy]`      | 31.58 us  | 34.37 us  | 5.45 us   | 3107   |
| `test_provider_boundary[array_matrix_multiplication-jax]`        | 102.23 us | 110.52 us | 11.37 us  | 1407   |
| `test_provider_boundary[array_transpose_reshape-numpy]`          | 43.44 us  | 47.32 us  | 6.40 us   | 2907   |
| `test_provider_boundary[array_transpose_reshape-jax]`            | 189.23 us | 203.82 us | 16.03 us  | 1009   |
| `test_projection_and_calculated_column[overhead]`                | 633.36 us | 667.55 us | 33.87 us  | 689    |
| `test_filter_selectivity[overhead]`                              | 640.41 us | 671.96 us | 20.30 us  | 727    |
| `test_group_by_aggregation[overhead]`                            | 995.11 us | 1.024 ms  | 26.80 us  | 445    |
| `test_join_cardinality[overhead]`                                | 1.298 ms  | 1.336 ms  | 24.96 us  | 340    |
| `test_window_function[overhead]`                                 | 1.280 ms  | 1.448 ms  | 565.22 us | 358    |
| `test_builtin_versus_registered_udf[builtin]`                    | 752.21 us | 789.95 us | 17.85 us  | 597    |
| `test_builtin_versus_registered_udf[registered_udf]`             | 970.27 us | 1.228 ms  | 877.46 us | 459    |
| `test_execution_configuration[1024-1]`                           | 743.21 us | 996.58 us | 303.54 us | 617    |
| `test_execution_configuration[8192-2]`                           | 743.01 us | 780.77 us | 21.51 us  | 620    |
| `test_repeated_compiled_plan_execution[overhead]`                | 755.44 us | 784.75 us | 15.34 us  | 596    |
| `test_graph_fan_out[overhead]`                                   | 2.723 ms  | 2.775 ms  | 34.57 us  | 176    |
| `test_checkpoint_json_serialization[overhead]`                   | 5.76 us   | 5.98 us   | 888.52 ns | 17761  |
| `test_checkpoint_atomic_write[overhead]`                         | 1.019 ms  | 1.337 ms  | 346.06 us | 186    |
| `test_checkpoint_recovery_load[overhead]`                        | 356.79 us | 397.97 us | 24.57 us  | 1094   |

### A.2 Scale `small` (40 cases, commit `3bae1a6`, CI runner)

| Case                                                             | Min       | Mean      | Std dev   | Rounds |
| ---------------------------------------------------------------- | --------- | --------- | --------- | ------ |
| `test_backend_kernel[array_elementwise-numpy]`                   | 21.07 us  | 21.82 us  | 2.06 us   | 6703   |
| `test_backend_kernel[array_elementwise-jax]`                     | 46.27 us  | 67.42 us  | 31.99 us  | 1539   |
| `test_backend_kernel[array_mean-numpy]`                          | 6.21 us   | 6.54 us   | 1.32 us   | 7402   |
| `test_backend_kernel[array_mean-jax]`                            | 11.56 us  | 17.19 us  | 6.00 us   | 2504   |
| `test_backend_kernel[array_matrix_multiplication-numpy]`         | 15.43 us  | 16.32 us  | 10.95 us  | 7334   |
| `test_backend_kernel[array_matrix_multiplication-jax]`           | 23.14 us  | 46.36 us  | 13.27 us  | 2622   |
| `test_backend_kernel[array_transpose_reshape-numpy]`             | 4.51 us   | 4.80 us   | 1.22 us   | 11816  |
| `test_backend_kernel[array_transpose_reshape-jax]`               | 91.58 us  | 101.18 us | 14.14 us  | 1465   |
| `test_batch_ownership[numpy]`                                    | 11.21 us  | 12.19 us  | 1.98 us   | 5812   |
| `test_batch_ownership[jax]`                                      | 15.71 us  | 16.84 us  | 3.02 us   | 3971   |
| `test_plan_end_to_end[array_elementwise-numpy]`                  | 116.90 us | 126.82 us | 10.84 us  | 1645   |
| `test_plan_end_to_end[array_elementwise-jax]`                    | 300.66 us | 328.53 us | 27.74 us  | 806    |
| `test_plan_end_to_end[array_mean-numpy]`                         | 72.07 us  | 80.80 us  | 29.43 us  | 1962   |
| `test_plan_end_to_end[array_mean-jax]`                           | 142.31 us | 153.22 us | 12.24 us  | 1242   |
| `test_plan_end_to_end[array_matrix_multiplication-numpy]`        | 80.91 us  | 87.76 us  | 9.68 us   | 1749   |
| `test_plan_end_to_end[array_matrix_multiplication-jax]`          | 142.75 us | 153.01 us | 11.01 us  | 1240   |
| `test_plan_end_to_end[array_transpose_reshape_diagnostic-numpy]` | 80.26 us  | 87.95 us  | 9.64 us   | 1993   |
| `test_plan_end_to_end[array_transpose_reshape_diagnostic-jax]`   | 237.69 us | 255.97 us | 17.12 us  | 923    |
| `test_provider_boundary[array_elementwise-numpy]`                | 82.99 us  | 89.31 us  | 9.15 us   | 2165   |
| `test_provider_boundary[array_elementwise-jax]`                  | 255.00 us | 273.43 us | 16.94 us  | 968    |
| `test_provider_boundary[array_mean-numpy]`                       | 38.87 us  | 44.45 us  | 9.40 us   | 2863   |
| `test_provider_boundary[array_mean-jax]`                         | 101.56 us | 109.68 us | 9.97 us   | 1616   |
| `test_provider_boundary[array_matrix_multiplication-numpy]`      | 49.23 us  | 54.38 us  | 6.84 us   | 2797   |
| `test_provider_boundary[array_matrix_multiplication-jax]`        | 100.03 us | 110.14 us | 22.31 us  | 1625   |
| `test_provider_boundary[array_transpose_reshape-numpy]`          | 49.67 us  | 54.40 us  | 7.23 us   | 2487   |
| `test_provider_boundary[array_transpose_reshape-jax]`            | 191.98 us | 207.46 us | 15.85 us  | 990    |
| `test_projection_and_calculated_column[small]`                   | 639.76 us | 674.12 us | 23.85 us  | 518    |
| `test_filter_selectivity[small]`                                 | 654.37 us | 710.27 us | 70.69 us  | 625    |
| `test_group_by_aggregation[small]`                               | 1.050 ms  | 1.082 ms  | 24.96 us  | 437    |
| `test_join_cardinality[small]`                                   | 1.357 ms  | 1.417 ms  | 41.90 us  | 345    |
| `test_window_function[small]`                                    | 2.671 ms  | 2.744 ms  | 42.45 us  | 180    |
| `test_builtin_versus_registered_udf[builtin]`                    | 765.95 us | 803.84 us | 20.09 us  | 528    |
| `test_builtin_versus_registered_udf[registered_udf]`             | 1.179 ms  | 1.224 ms  | 81.58 us  | 375    |
| `test_execution_configuration[1024-1]`                           | 779.87 us | 821.39 us | 20.90 us  | 584    |
| `test_execution_configuration[8192-2]`                           | 923.70 us | 997.04 us | 26.52 us  | 459    |
| `test_repeated_compiled_plan_execution[small]`                   | 768.66 us | 805.58 us | 23.67 us  | 588    |
| `test_graph_fan_out[small]`                                      | 2.818 ms  | 2.887 ms  | 42.10 us  | 168    |
| `test_checkpoint_json_serialization[small]`                      | 5.73 us   | 6.03 us   | 1.31 us   | 18403  |
| `test_checkpoint_atomic_write[small]`                            | 1.008 ms  | 1.141 ms  | 309.14 us | 174    |
| `test_checkpoint_recovery_load[small]`                           | 365.77 us | 412.21 us | 22.41 us  | 1104   |

### A.3 Scale `standard` (40 cases, commit `3bae1a6`, CI runner)

| Case                                                             | Min       | Mean      | Std dev   | Rounds |
| ---------------------------------------------------------------- | --------- | --------- | --------- | ------ |
| `test_backend_kernel[array_elementwise-numpy]`                   | 859.03 us | 894.01 us | 14.46 us  | 391    |
| `test_backend_kernel[array_elementwise-jax]`                     | 98.18 us  | 147.13 us | 24.92 us  | 709    |
| `test_backend_kernel[array_mean-numpy]`                          | 26.78 us  | 27.51 us  | 2.04 us   | 4820   |
| `test_backend_kernel[array_mean-jax]`                            | 26.72 us  | 42.90 us  | 10.93 us  | 1962   |
| `test_backend_kernel[array_matrix_multiplication-numpy]`         | 428.31 us | 445.14 us | 10.71 us  | 882    |
| `test_backend_kernel[array_matrix_multiplication-jax]`           | 270.42 us | 320.44 us | 42.06 us  | 938    |
| `test_backend_kernel[array_transpose_reshape-numpy]`             | 184.39 us | 196.60 us | 4.92 us   | 1864   |
| `test_backend_kernel[array_transpose_reshape-jax]`               | 110.96 us | 141.06 us | 18.50 us  | 929    |
| `test_batch_ownership[numpy]`                                    | 32.25 us  | 33.26 us  | 2.84 us   | 3290   |
| `test_batch_ownership[jax]`                                      | 13.63 us  | 14.35 us  | 1.58 us   | 3324   |
| `test_plan_end_to_end[array_elementwise-numpy]`                  | 299.86 us | 314.59 us | 12.13 us  | 794    |
| `test_plan_end_to_end[array_elementwise-jax]`                    | 487.57 us | 602.74 us | 80.02 us  | 506    |
| `test_plan_end_to_end[array_mean-numpy]`                         | 74.41 us  | 78.33 us  | 7.10 us   | 1535   |
| `test_plan_end_to_end[array_mean-jax]`                           | 102.09 us | 133.08 us | 26.94 us  | 1063   |
| `test_plan_end_to_end[array_matrix_multiplication-numpy]`        | 530.19 us | 551.08 us | 41.22 us  | 563    |
| `test_plan_end_to_end[array_matrix_multiplication-jax]`          | 362.84 us | 408.50 us | 27.19 us  | 720    |
| `test_plan_end_to_end[array_transpose_reshape_diagnostic-numpy]` | 263.54 us | 277.92 us | 12.24 us  | 887    |
| `test_plan_end_to_end[array_transpose_reshape_diagnostic-jax]`   | 227.87 us | 258.27 us | 20.39 us  | 662    |
| `test_provider_boundary[array_elementwise-numpy]`                | 279.50 us | 290.38 us | 10.04 us  | 599    |
| `test_provider_boundary[array_elementwise-jax]`                  | 450.29 us | 552.53 us | 88.83 us  | 426    |
| `test_provider_boundary[array_mean-numpy]`                       | 52.65 us  | 55.54 us  | 18.39 us  | 2156   |
| `test_provider_boundary[array_mean-jax]`                         | 54.54 us  | 80.51 us  | 14.34 us  | 1316   |
| `test_provider_boundary[array_matrix_multiplication-numpy]`      | 497.72 us | 516.89 us | 37.95 us  | 633    |
| `test_provider_boundary[array_matrix_multiplication-jax]`        | 322.53 us | 358.57 us | 22.73 us  | 866    |
| `test_provider_boundary[array_transpose_reshape-numpy]`          | 241.06 us | 252.16 us | 8.80 us   | 1024   |
| `test_provider_boundary[array_transpose_reshape-jax]`            | 209.45 us | 237.10 us | 18.24 us  | 756    |
| `test_projection_and_calculated_column[standard]`                | 710.64 us | 748.63 us | 21.98 us  | 396    |
| `test_filter_selectivity[standard]`                              | 784.46 us | 832.65 us | 25.28 us  | 483    |
| `test_group_by_aggregation[standard]`                            | 1.571 ms  | 1.630 ms  | 27.87 us  | 275    |
| `test_join_cardinality[standard]`                                | 2.058 ms  | 2.135 ms  | 42.41 us  | 179    |
| `test_window_function[standard]`                                 | 20.417 ms | 20.701 ms | 229.27 us | 24     |
| `test_builtin_versus_registered_udf[builtin]`                    | 842.27 us | 886.76 us | 37.46 us  | 470    |
| `test_builtin_versus_registered_udf[registered_udf]`             | 2.897 ms  | 3.032 ms  | 64.78 us  | 156    |
| `test_execution_configuration[1024-1]`                           | 1.112 ms  | 1.159 ms  | 88.90 us  | 366    |
| `test_execution_configuration[8192-2]`                           | 1.047 ms  | 1.127 ms  | 62.54 us  | 321    |
| `test_repeated_compiled_plan_execution[standard]`                | 840.70 us | 940.55 us | 190.87 us | 505    |
| `test_graph_fan_out[standard]`                                   | 3.400 ms  | 3.475 ms  | 77.27 us  | 114    |
| `test_checkpoint_json_serialization[standard]`                   | 5.83 us   | 6.16 us   | 827.49 ns | 8509   |
| `test_checkpoint_atomic_write[standard]`                         | 842.19 us | 923.53 us | 85.36 us  | 184    |
| `test_checkpoint_recovery_load[standard]`                        | 315.75 us | 362.99 us | 24.54 us  | 1140   |

### A.4 Scale `nightly` (40 cases, commit `3bae1a6`, CI runner)

| Case                                                             | Min        | Mean       | Std dev   | Rounds |
| ---------------------------------------------------------------- | ---------- | ---------- | --------- | ------ |
| `test_backend_kernel[array_elementwise-numpy]`                   | 3.668 ms   | 3.906 ms   | 125.91 us | 111    |
| `test_backend_kernel[array_elementwise-jax]`                     | 403.28 us  | 494.40 us  | 211.78 us | 184    |
| `test_backend_kernel[array_mean-numpy]`                          | 196.86 us  | 204.53 us  | 11.05 us  | 955    |
| `test_backend_kernel[array_mean-jax]`                            | 88.09 us   | 126.29 us  | 15.13 us  | 1446   |
| `test_backend_kernel[array_matrix_multiplication-numpy]`         | 3.058 ms   | 3.128 ms   | 169.78 us | 143    |
| `test_backend_kernel[array_matrix_multiplication-jax]`           | 1.489 ms   | 1.539 ms   | 27.08 us  | 223    |
| `test_backend_kernel[array_transpose_reshape-numpy]`             | 900.47 us  | 922.42 us  | 20.31 us  | 476    |
| `test_backend_kernel[array_transpose_reshape-jax]`               | 213.66 us  | 251.01 us  | 30.36 us  | 833    |
| `test_batch_ownership[numpy]`                                    | 273.64 us  | 287.86 us  | 15.28 us  | 323    |
| `test_batch_ownership[jax]`                                      | 15.19 us   | 16.18 us   | 2.77 us   | 3846   |
| `test_plan_end_to_end[array_elementwise-numpy]`                  | 2.348 ms   | 2.489 ms   | 87.21 us  | 123    |
| `test_plan_end_to_end[array_elementwise-jax]`                    | 3.316 ms   | 3.713 ms   | 731.06 us | 82     |
| `test_plan_end_to_end[array_mean-numpy]`                         | 265.26 us  | 282.27 us  | 19.37 us  | 749    |
| `test_plan_end_to_end[array_mean-jax]`                           | 196.79 us  | 230.08 us  | 21.04 us  | 1023   |
| `test_plan_end_to_end[array_matrix_multiplication-numpy]`        | 3.211 ms   | 3.313 ms   | 79.73 us  | 121    |
| `test_plan_end_to_end[array_matrix_multiplication-jax]`          | 1.599 ms   | 1.686 ms   | 121.70 us | 223    |
| `test_plan_end_to_end[array_transpose_reshape_diagnostic-numpy]` | 1.040 ms   | 1.075 ms   | 26.69 us  | 349    |
| `test_plan_end_to_end[array_transpose_reshape_diagnostic-jax]`   | 353.93 us  | 405.78 us  | 32.33 us  | 566    |
| `test_provider_boundary[array_elementwise-numpy]`                | 2.162 ms   | 2.364 ms   | 96.80 us  | 131    |
| `test_provider_boundary[array_elementwise-jax]`                  | 3.266 ms   | 3.368 ms   | 53.27 us  | 119    |
| `test_provider_boundary[array_mean-numpy]`                       | 234.04 us  | 246.47 us  | 14.45 us  | 885    |
| `test_provider_boundary[array_mean-jax]`                         | 147.14 us  | 179.33 us  | 19.59 us  | 1158   |
| `test_provider_boundary[array_matrix_multiplication-numpy]`      | 3.166 ms   | 3.248 ms   | 59.70 us  | 130    |
| `test_provider_boundary[array_matrix_multiplication-jax]`        | 1.560 ms   | 1.624 ms   | 74.90 us  | 266    |
| `test_provider_boundary[array_transpose_reshape-numpy]`          | 1.007 ms   | 1.040 ms   | 22.65 us  | 362    |
| `test_provider_boundary[array_transpose_reshape-jax]`            | 315.08 us  | 354.92 us  | 20.26 us  | 653    |
| `test_projection_and_calculated_column[nightly]`                 | 1.359 ms   | 1.435 ms   | 42.83 us  | 247    |
| `test_filter_selectivity[nightly]`                               | 2.162 ms   | 2.330 ms   | 63.11 us  | 195    |
| `test_group_by_aggregation[nightly]`                             | 6.248 ms   | 6.494 ms   | 1.161 ms  | 78     |
| `test_join_cardinality[nightly]`                                 | 14.136 ms  | 14.667 ms  | 243.45 us | 35     |
| `test_window_function[nightly]`                                  | 258.645 ms | 260.675 ms | 2.114 ms  | 3      |
| `test_builtin_versus_registered_udf[builtin]`                    | 1.566 ms   | 1.642 ms   | 45.54 us  | 119    |
| `test_builtin_versus_registered_udf[registered_udf]`             | 22.948 ms  | 23.139 ms  | 114.65 us | 21     |
| `test_execution_configuration[1024-1]`                           | 3.852 ms   | 4.076 ms   | 144.34 us | 101    |
| `test_execution_configuration[8192-2]`                           | 2.293 ms   | 2.479 ms   | 192.15 us | 184    |
| `test_repeated_compiled_plan_execution[nightly]`                 | 1.503 ms   | 1.583 ms   | 44.23 us  | 286    |
| `test_graph_fan_out[nightly]`                                    | 8.330 ms   | 8.821 ms   | 1.157 ms  | 34     |
| `test_checkpoint_json_serialization[nightly]`                    | 5.69 us    | 5.93 us    | 1.05 us   | 19173  |
| `test_checkpoint_atomic_write[nightly]`                          | 968.87 us  | 1.087 ms   | 200.34 us | 193    |
| `test_checkpoint_recovery_load[nightly]`                         | 370.36 us  | 415.33 us  | 23.95 us  | 971    |
