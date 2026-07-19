# All-Scale Benchmark CI Design

## Goal

Expand the pull-request and main-branch benchmark smoke job from the
`overhead` scale to every scale supported by the benchmark suite:
`overhead`, `small`, `standard`, and `nightly`.

The scheduled benchmark workflow already runs these four scales. This change
keeps the CI workflow aligned with that established scale set while leaving
benchmark results informational.

## Workflow design

Change the `benchmark-smoke` job in `.github/workflows/ci.yml` to a matrix job
with `fail-fast: false` and one runner per scale. Set
`CALC_FLOW_BENCHMARK_SCALE` from `matrix.scale` at job scope and retain
`JAX_PLATFORMS=cpu`.

Each runner will:

1. install the benchmark dependencies once;
2. run the complete Python benchmark suite for its selected scale;
3. write `benchmark-results/<scale>.json`; and
4. upload an artifact named `benchmark-smoke-<scale>`.

The job and step names will include or describe the selected scale so a slow
or failed scale can be identified directly in the Actions UI. The job will
remain dependent on `lint-and-test`, and benchmark timing deltas will remain
non-gating.

## Alternatives considered

- A single runner could loop over all four scales. This avoids repeated setup
  but creates one long job, weakens failure isolation, and cannot run scales in
  parallel.
- The scheduled and CI workflows could be refactored into a reusable workflow.
  That removes duplication but expands the change beyond the requested CI
  coverage update.

The matrix approach matches the existing scheduled workflow and gives the
clearest results with the smallest focused change.

## Validation

Add a focused release-configuration test that reads the CI workflow and
asserts that the benchmark smoke matrix contains the same ordered scale names
as `benchmarks.support.SCALES`. The test will also verify scale-specific JSON
and artifact naming, preventing the two workflow surfaces from drifting back
to overhead-only coverage.

Run the focused test first, then the release-helper unit tests, formatting and
lint checks, and workflow diff hygiene. No benchmark timing threshold will be
introduced.
