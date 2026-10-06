## Branch Review: ASOF output source ranges (A3)

**Author:** Cheng Li | **Branch:**
`feature/stream-asof-output-ranges-main` → `main` | **Files:** 7

**Base:** `eccb26973811bc476f0944b977ddedf8564b0237`

**Reviewed source head:** `9983cfd0c3d096da7a60c3a96ae03b785f5ae955`

**Source tree:** `78065d9fe80635e3f6d64831215acacd9ad3e572`

The source consists of `4d10a5cb` plus the minimal complexity refactor
`9983cfd0`. The final increment received separate source/evidence review.

### Summary

This implements FR15's remaining output-planning work. Canonical left runs
resolve their source once; an inline pending span coalesces physical row
ranges before recording the existing range estimate. Left row-position
storage and eager row-sized span capacity are removed. Right candidate
references, gather behavior, source-registration order, backing charges,
prefix commits and checkpoint bytes preserve their existing contracts.

The review read the changed Rust source and new tests, the existing
`OutputRun`/payload ownership and selected-range estimation paths, backing
refusal tests, prefix commit/cancellation checks, code-style, repository
guidance, the acceleration specification and current ASOF guide. It is
limited to this A3 branch; no remote review or merge was performed.

### Build and Test Results

- Rust: **Passed scoped author checks.** The two intended RED failures were
  followed by GREEN. The serial ASOF run passed all 302 existing cases and
  three new cases; its sole new fixture failure was resolved by the exact
  matrix rerun. Final focused output-range tests passed 5/5, including the
  130-cut matrix and explicit unsupported-payload rejection. Integration
  checks passed 3 boundary, 8 property and 5 resource tests. Core lib/tests
  Clippy with `-D warnings`, fmt, generated-contract and whitespace checks
  passed.
- Python: **Not touched**; not run.
- Studio backend: **Not touched**; not run.
- Studio frontend: **Not touched**; not run.
- New failures: One new matrix fixture asserted pool refund before its
  owned gather service stopped. The isolated service/runtime shutdown fix
  passed the exact rerun; production lifecycle and budgets were not changed.
  Clippy's range-style finding was fixed by naming the exclusive end, with
  the same half-open range, and the focused tests and lint passed afterward.
- Regressions: None identified in the reviewed source or resolved test
  evidence.

The reviewer ran only read-only source/diff inspection. Native build, tests,
timing and coverage were not run by the reviewer because the author owns the
shared cache. There are no PR check runs or prior remote reviews for this
branch handoff. Required CI and the combined Rust 90% coverage gate remain
unverified locally; this report makes no merge-ready or performance claim.
The author confirmed zero owned native/Cargo/rustc processes and returned the
shared cache after the final checks.

Author commands used the explicitly handed-off root `CARGO_TARGET_DIR`,
`CARGO_BUILD_JOBS=2`, unchanged debug settings and serial Rust tests:

```bash
cargo test --locked -p calc-flow --lib \
  operator::asof::finalize::output_ranges:: -- --test-threads=1
cargo test --locked -p calc-flow --lib operator::asof:: -- --test-threads=1
cargo test --locked -p calc-flow \
  --test stream_asof_join_properties --test stream_asof_join_resources \
  --test stream_asof_join_boundaries -- --test-threads=1
cargo clippy --locked -p calc-flow --lib --tests -- -D warnings
cargo fmt --all --check
git diff --check
```

The ASOF serial command is the initial 305/306 result, with the diagnosed
new fixture failure kept visible above. The exact corrected matrix and final
five-test focused rerun complete that evidence; unchanged passing cases were
not repeated. RED measured `(1,024, 1,024)` left planning visits against
`(1, 1)` and 57,344 constructor bytes against the 16,896-byte bound. The
separate unchanged reservation assertion was 278,528 bytes.

The added Lizard threshold is satisfied by changed functions: candidate
consumers are 8/8/8, `left_run_source` is 2 and the new counters are 1 each.
`lizard -l rust -C 8 -w` on the finalizer, output planner and new range tests
exited 0. Reported higher-complexity functions in unchanged workspace/output
code are outside this changed-function gate and were not suppressed.
The final helper preserves the sole source lookup at run offset zero;
the cursor extension preserves the original borrowed iterator, preallocated
capacity and key/cursor ownership. Cancellation checks retain their positions.

### Blocking Issues

None. The intermediate complexity finding was corrected and the final scoped
tests, lint and complexity check passed.

### Style Issues

None. Mutation stays inside the
owned private builder; caller Arrow arrays remain read-only. No public API,
dependency, unsafe-code allowance, executor I/O or new lifecycle machinery
is introduced.

### Test Coverage

- Production matching exercises binary search, monotonic cursor and supplied
  parallel-candidate paths. A 1,024-row canonical run proves one left source
  visit and one range-estimate visit, with the same right null references,
  row count and span order.
- Constructor tests measure actual allocations at 0, 1, 17, 1,024 and
  64,000 rows and separately assert the unchanged `256 * rows + 16 KiB`
  reservation. Fragmented reverse-row tests measure vector growth at
  0, 1, 2, 3, 7, 17, 1,024 and 64,000 rows against the original charge.
- An independent 130-cut matrix compares exact Arrow output and order plus
  per-row slice byte sums. It covers cropped sources, repeated and fragmented
  rows, missing/repeated right candidates, nullable UTF-8/large UTF-8 and
  binary/large binary values, 7/19-column schemas, full, both-sided,
  one-sided and zero-column projections.
- Dictionary/list/struct payloads continue to fail explicit schema validation
  on either side. This preserves the existing flat-payload contract.
- Existing boundary, generated watermark/restore, stalled/hot-key/wide,
  checkpoint allocation and atomic refusal tests supply directly affected
  integration coverage. Existing ASOF tests cover accepted-prefix recovery,
  cancellation, output-byte refusal and backing-owner retirement.

`OutputRun::Chunk` uses one fixed payload reference; physical row positions
may vary, so source reuse is valid while span coalescing still requires exact
adjacency. Legacy runs remain one row each. Variable-width estimates merge
only contiguous ranges from the same source; fixed-width estimates count
selected rows. A partial final run flushes its pending span before buffer
credit is obtained and materialization starts.

At most one emitted span exists per selected row. Span vector growth plus
right positions fits the unchanged per-row planning credit; the fixed base
covers inline state and tiny-vector minimum capacity. Source descriptor,
type-header, full backing-owner and buffer-growth charges are unchanged.
Neither unselected columns nor hidden slice backing lose their charges.
Sink acceptance remains the commit boundary and cancellation does not alter
the prefix inventories or remaining state.

### Documentation Consistency

The [single analysis artifact](../analysis/stream-asof-output-ranges.md)
records the private scope, actual RED/GREEN evidence, allocation funding,
fixture correction and compatibility checks. The CHANGELOG entry accurately
describes output planning rather than claiming a new materialization path or
measured throughput gain. The current ASOF guide already specifies spans,
selected-column workspace, full-array sharing guards, bounded chunk retry,
accepted-prefix commits and cancellation ownership; no additional public
documentation change is needed for this private optimization.

Read-only comparison found no changes to generated contracts, ASOF
admission/checkpoint code, generic gather ownership or historical v1 fixtures.
The `CFASRW10` layout and accounting-10 implementation remain outside this
diff. Reviewer `git diff --check` on the final source commit passed; the
review artifact's whitespace, final newline and local links also passed.

### Verdict

**Approve** for source head `9983cfd0c3d096da7a60c3a96ae03b785f5ae955`
and tree `78065d9fe80635e3f6d64831215acacd9ad3e572`.

No blocking correctness, funding, style, test or documentation issue remains
in this source review. Required remote CI/coverage and the parent's separate
performance/publication work remain outside this local approval.
