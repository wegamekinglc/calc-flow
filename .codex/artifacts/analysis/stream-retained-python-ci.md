# PR368 Python collection and static-check remediation

The benchmark-suite workflow now explicitly runs the retained-state adapter
module. Its script counterpart uses `unittest.TestCase`, is registered once in
the script inventory, and retains every former parametrized case as a subtest.
The workflow contracts reject missing adapter coverage and zero unittest
collection. Script assertions use unittest methods rather than removable
`assert` statements.

The Polars references use the approved fresh multiprocessing worker, verify its
distinct PID and actual one-thread pool, and cover all eight reference scenarios.
Interval SQL retains the same eleven equality-join branches for inclusive
offsets from -5 to +5 seconds; every query branch is a static string literal.

Catalog membership, sample evidence, checkpoint setup, and replay-prefix
feeding are decomposed into focused functions. Their scope, batch-size,
checkpoint, cap, exact-cursor, and durable acknowledgement checks remain strict.
Static Join proof awaits a causally newer emitted-row marker after throughput
timing. Prefix cancellation and completed recovery also require that marker
before accepting zero retained/evicted left counters.

## Validation

- Collection RED: zero retained script tests, absent adapter workflow path,
  and missing exact-once script inventory entry (3 failures).
- Causal Join RED: stale snapshots incorrectly accepted later retained and
  evicted quote rows; successful case read only the stale snapshot (3 failures).
- Workflow, retained script, catalog/report, and measure checks: 79 passed.
  After test-only decomposition, the retained module's 11 methods passed.
- Interval references, checkpoint duration/acknowledgement, and cancellation:
  14 functional tests passed. The six Join/interval/ASOF recovery variants were
  rechecked after adding causal Join proof and passed.
- Causal snapshots and fresh Polars references: 6 passed. All 21 retained
  adapter cases are collected by pytest; the seven other collected cases were
  not rerun because their implementation was unchanged.
- Radon complexity: catalog IDs 1, variant cases 5, evidence validation 4,
  checkpoint cut 4, recovery completion 6; changed helpers/tests are at most 8.
- Bandit B603/B608 passed for SQL and fresh worker tests; B101 passed for
  retained script tests. No ignores or analyzer configurations changed.
- Scoped Ruff lint, complexity-rule scan, format, contract drift, and whitespace
  checks passed. No Rust build, performance measurement, remote mutation, or
  sealed performance report revision was performed.

Remote CI and Codacy results remain separate handoff gates.
