# Verification

[Documentation](README.md) / 5.1 Verification

Run commands from the repository root unless a working directory is shown.
[AGENTS.md](../AGENTS.md#commands) maintains the complete required command
groups and toolchain rules. Keep build, coverage, release, and cache outputs
under `target/` in a constrained checkout.

## Documentation and examples

Prepare the environment using [getting started](getting-started.md). For an
existing installation, `--no-sync` keeps its installed native package when
running examples:

```bash
uv run --no-sync python scripts/run_examples.py
python -m unittest scripts.test_run_examples scripts.test_release_config
uv run --no-sync ruff check examples
uv run --no-sync ruff format --check examples
git diff --check
```

The runner discovers numbered Python programs in order, runs the maintained
Rust user-example list, and stops with the first failing program's exit code.
Schema exporters are separate tools because generation updates a tracked
artifact. See [the example inventory](../examples/README.md).

Check every relative Markdown link and heading anchor in changed documents,
including links from the repository and example indexes. Confirm that each
function guide links to a real program or clearly identifies a configuration
fragment. Expected values should be checked by executable assertions; elapsed
time and platform-specific debug formatting are not fixed output contracts.

## Implementation surfaces

The repository command groups cover:

- Rust formatting, all-target/all-feature Clippy, the core/PyO3 test harness,
  rustdoc, and workspace coverage.
- Python native installation, binding/adapter tests, and Ruff.
- Studio backend tests with its independent coverage floor.
- Frontend dependency installation, API generation, build, unit tests,
  Playwright workflows, and dependency audit.
- Supply-chain policy and packaging/release helper tests.

The Rust test harness must inherit the same managed Python environment used
for PyO3 compilation, including NumPy, PyArrow, and interpreter library paths.
Connector coverage requires running Kafka, PostgreSQL, and ClickHouse plus
the environment variables listed in [AGENTS.md](../AGENTS.md#commands).
Missing services are a blocked verification gate, not a passing coverage run.

After checks, confirm generated contracts and whitespace:

```bash
git diff --exit-code -- schemas/project-v3.schema.json web-ui/openapi.json web-ui/src/api/schema.d.ts
git diff --check
```

## Dependency audit scope

The explicit `cargo audit` waiver for `RUSTSEC-2026-0235` is lockfile-only:
the optional `rkyv` dependency recorded through `rust_decimal` is not enabled
in the workspace build. Confirm the scope with
`cargo tree --workspace --all-features -i rkyv`; an enabled dependency path
requires a new security assessment. This waiver does not authorize compiled
or shipped vulnerable code. The maintained audit command is in
[AGENTS.md](../AGENTS.md#commands); the separate scoped dependency-policy
exceptions are recorded in [deny.toml](../deny.toml).

## Streaming stress and soak checks

The streaming tests exercise bounded zero-cost traffic, admission, cancellation,
graceful drain, resource closure, and task/reaper convergence. The checkpoint
fault/restart matrix uses the public runner facade and observes actual fault,
cancellation, and checkpoint probes.

The opt-in Linux soak harnesses each measure 1,200 seconds at a ten-second
cadence, collecting 120 RSS samples. Run them explicitly:

```bash
CALC_FLOW_STREAM_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::twenty_minute_two_source_slow_sink -- --ignored --exact --nocapture
CALC_FLOW_M5_CHECKPOINT_SOAK=1 cargo test -p calc-flow --lib runtime::streaming::soak::twenty_minute_epoch_checkpoint_restart -- --ignored --exact --nocapture
```

The first harness includes a 300-second warm-up. The checkpoint restart
harness runs three sequential child processes sharing a filesystem state root
and verifies cursor, epoch, output, watermark, and cleanup continuity.
These are verification procedures, not claims that a particular revision has
passed them.

## Performance and releases

Use the [benchmark suite](benchmark-suite.md), [SQL measurements](sql-datafusion-performance.md),
and [warm-stream measurements](warm-stream-performance.md) for timing work.
Do not run benchmarks alongside builds or tests. Preserve raw failed and
inconclusive results as well as successful ones.

For release changes, additionally build the core wheel, sdist, crate, and
Studio wheel, inspect each artifact, install wheels in clean environments,
and perform the smoke checks in the [release guide](python-release.md).

Next: [benchmark suite](benchmark-suite.md).
