# Unified benchmark suite

## Scope

Use DAL's complete-result tables, retained evidence and same-runner base/head
comparison pattern. Preserve existing Python scales and specialized benchmark
contracts. Add seven engine row scales (10 through 10,000,000), actual native
StreamingRunner execution, warm-state append measurements, SQL and direct
DataFusion/Polars/TA-Lib reference implementations.

## Acceptance

- Every non-documentation Linux PR/main CI runs the complete suite through one
  reusable workflow. Scheduled and manual runs call the same suite; tuning-only
  SQL experiments remain separately labeled, supplemental experiments.
- One final summary lists every result, including errors and missing cases.
  Raw samples, exact refs, release-wheel hashes, dependency/machine identity,
  dimensions, scopes, thread settings and correctness evidence are retained.
- New engine and warm cases compare base/head with two rounds of ten
  interleaved samples on one runner. A regression gate requires both rounds
  to exceed +5%. Existing fixture/criterion suites retain their original timing
  boundaries; their block-level version comparisons are explicitly informational.
- External libraries are correctness-checked references, never mislabeled as
  historical regression baselines. Unsupported workload/backend combinations
  are explicit in the catalog. No missing dependency is silently skipped.
- Baseline failures, dropped cases, missing artifacts, incompatible fingerprints,
  nonfinite samples and incorrect outputs fail closed. All result tables are
  published even when measurement or regression validation fails.
- Do not change production engine code, scheduler defaults or existing numerical
  acceptance tolerances. Do not overwrite the dirty primary checkout.

## Work sequence

1. RED: catalog completeness, regression math, partial/failure reports and CI
   wiring tests; engine oracle tests across warm-up boundaries.
2. Implement shared catalog, release-bound worker, engine/warm adapters and report.
3. Normalize all existing suites, wire reusable CI, and retain full evidence.
4. Run focused and full regression tests, dependency/style/workflow checks and
   representative real measurements including the largest row scale.
