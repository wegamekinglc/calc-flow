# Branch review: benchmark workflow test inventory

Branch: `fix/benchmark-workflow-inventory`.
Base: `54406159002026dca0899d545d54869840e0cbb9`.
Reviewed the local workflow, routing regression test, and benchmark guide diff.
No PR or additional remote CI status was queried.

## Summary

The reported preparation failures came from two script-test modules missing
from the workflow inventory. The fix runs the standard-library isolation tests
in preparation and comparison support tests once in `warm-10`, after the
candidate wheel is available. The latter use a dedicated environment populated
from the existing RisingWave lock, with required imports checked before test
discovery.

## Findings

No blocking finding remains. The isolation module requires only the standard
library. The comparison module's direct and transitive imports are covered by
the dedicated lock and installed candidate wheel. Explicit interpreter paths
keep these dependencies separate from the measurement environment, and the
preflight prevents missing psycopg from silently skipping the comparison module.
The catalog contains one `warm-10` shard, and artifact download precedes the
new support-test step.

The comparison tests exercise fixtures, saved results, mocked service cleanup,
and a read-only CPU-description subprocess. They do not launch RisingWave or
collect performance samples. The guide accurately describes the routing and
environment boundaries. The regression checks both module inventory and the
comparison step's interpreter, dependency lock, wheel installation, imports,
and once-only condition.

## Verification evidence

- Expected failures were recorded in
  `target/latest-benchmark-workflow-red.log` and
  `target/latest-benchmark-environments-red.log` before implementation.
- The previously failing catalog/workflow group reported 217 tests, OK with
  one existing skip; workflow and isolation checks passed all 27 tests.
- The existing RisingWave lock installed all 12 dependencies in the separate
  test environment. The exact current-main Linux CI wheel
  `calc_flow_python-2026.10.27-cp313-abi3-linux_x86_64.whl` was installed without
  resolving additional dependencies after its artifact archive hash was verified.
- Required imports passed, followed by all nine comparison tests with no skips.
  Evidence is in `target/latest-benchmark-comparison-green.log` and
  `target/latest-benchmark-wheel-install.log`.
- The author reports Ruff, formatting, YAML parsing, schema-drift, and whitespace
  checks passed. The reviewer reused this evidence without duplicating tests.
- No performance measurements, full native builds, or full regression/coverage
  matrix was run for this workflow-routing fix.

## Verdict

**Approve** for the reviewed local fix. No unaddressed correctness, dependency
routing, scope, or documentation issue remains. Local validation does not
establish a successful new hosted benchmark run or merge readiness.
