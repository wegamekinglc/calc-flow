# DAL-517 independent finite inner join acceptance

Tested product HEAD: `4c1416449f16c7ab7cb19fa225a55072f8f3065c`.
Branch: `feature/dal-517-finite-inner-join`.
PR: https://github.com/wegamekinglc/calc-flow/pull/601 (base `main`).

The assigned checkout was clean on entry and was reused as explicitly requested.
This increment adds only two independent test files and this report. Product code,
existing assertions, frozen note, dependencies, baselines and team definitions are
unchanged. The published test-only commit is recorded in the issue handoff.

## Evidence reused and coverage gaps selected

Read the [frozen contract](../api-notes/finite-inner-join.md), the
[implementation and repair evidence](finite-inner-join.md), repository
`AGENTS.md`, `cf-tester` and `code-style`, production admission/matching/finite
and public declaration APIs, and nearby tests before writing tests.
The identity brief's DAL-specific `.codex/references/` files and dal-public
component do not exist in this repository.

Reused, without rerunning the suites: the original 64 distinct Rust / 303 Python
checks and repair's 6 Rust / 81 Python checks. These include the Rust DF/SQL
six-pair oracle in both modes, old @1/@2 identities, actual managed restart and
next-input equivalence, project reload, analysis/latest gates and core examples
12/13 plus the new finite example compilation. Those are earlier evidence,
not fresh counts from this independent run.

The remaining evidence gaps were exact negative/zero and nullable-key results
through public interfaces, explicit 12-triple nested/fan-out results with source
read counts, cumulative byte/row limits on both sides over multiple RecordBatches,
and an empty-side retry immediately after each separate failure.

## Independent oracles and tests added

`crates/calc-flow/tests/finite_inner_join_independent.rs` adds five tests:

- Native nullable key/time rows and a different key leave exactly
  `(1,13),(1,80),(1,81),(2,13),(2,80),(2,81)`. Both sides contain multiple
  RecordBatches. Exact prefixed schema preserves type/order/nullability.
- Two left rows against one right row succeed with two pairs; two right
  RecordBatches produce four pairs and fail a match budget of three.
- With the opposite side empty, one admitted row succeeds and two separate
  one-row RecordBatches fail, on either side, under either a one-row budget or
  an exact one-row byte budget. The independent V1 byte calculation is
  `64 + (1 + 4 + 4 + 8) + 16 + 3*(1+8) = 124` bytes.
- Every match/side-budget failure immediately retries left-empty, right-empty,
  both-empty and a fresh singleton on the same compiled plan, without reset.
  Empty-side probes use both old time 10 and new time 100 so stale rows would
  still be eligible to match. Schema failure and pre-cancelled execution each
  receive the same immediate retry sequence after a successful execution.
- Rust's public table input rejects nullable event time at declaration, with
  `inputs.<name>.event_time` and non-null UTC microsecond diagnostics. This
  precedes either DF or SQL join declaration; it is not a delayed compile claim.

`python/tests/test_finite_inner_join_independent.py` adds 24 parametrized cases:

- DF/SQL, batch/stream each independently check literal six pairs with NULL and
  different keys; zero bounds against times 9/10/11; negative left time -10
  against -13/-12/-7/-6, producing only IDs 7 and 12. Input Arrow rows remain
  unchanged. No other frontend supplies expected values.
- DF/SQL each export three unique join nodes: the shared ordered first join,
  its nested join and a distinct zero-bound join. First output and fan-out each
  equal the six literal pairs; nesting with two matching third-source rows
  yields 12 explicit triples; the distinct join is empty and unrelated output
  survives. Execute in batch and a real stream job without collecting an
  intermediate relation. Source iterator starts are exactly a=1/b=1/c=1;
  data reads are a=1/b=2/c=2.
- DF/SQL each reuse a cached public finite plan and the same Runtime: success,
  four-pair/two-RecordBatch failure at budget three, each empty side, both-empty,
  then only new pair (500,600). Cache lookup returns the identical plan.
- Both DF/SQL, both modes and each ingress reject nullable time at the expected
  structured public path. Native nullable-time drop remains a separate test.

Stream oracles concatenate every append output at normal EOF with disabled
watermarks; they do not assume cross-source emission order. Node count plus
source read observations establish the exported shared-owner topology and
single source consumption; no heap-level owner allocation instrumentation is
claimed.

## Provenance

Before any Python product test, re-imported using the recorded interpreter:

```sh
export CF_PY=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/python
export CF_SITE=target/finite-inner-join/site
export PYTHONPATH="$CF_SITE"
export CARGO_BUILD_JOBS=2
export CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/pr-repair-20261112/cargo-debug
```

The interpreter is CPython 3.13. The executable and Cargo artifact cache happen
to live in the development checkout; candidate Python/native code loads solely
from this assigned checkout's `target/finite-inner-join/site/calc_flow/`.
No candidate package is imported from that other checkout.

Compared `provenance-codacy.json` with current HEAD, `sys.executable`,
`calc_flow.__file__`, `calc_flow._native.__file__` and actual file SHA-256s:
all 1202 recorded source/manifest files match and all 58 loaded Python/stub
files match this checkout byte-for-byte. The loaded native SHA-256 is exactly:

`2a6d537354830e3d1dc993ac16f73a7e5359cf10bd1352b4f82c9659b186e4a7`

Reused that native built here after the repair; no product changes require a
rebuild for this test-only increment. Local captures are under
`target/finite-inner-join/independent/`; the committed tests/report are the
reviewable deliverables.

## Commands and results

Commands run at the assigned checkout root with the environment above:

```sh
"$CF_PY" -m pytest python/tests/test_finite_inner_join_independent.py -q -W error
cargo test --locked -p calc-flow --test finite_inner_join_independent
cargo test --locked -p calc-flow --lib operator::join::tests::null_event_time_and_null_key_rows_are_counted_not_stored -- --exact
cargo clippy --locked -p calc-flow --test finite_inner_join_independent -- -D warnings
rustfmt --edition 2024 --check crates/calc-flow/tests/finite_inner_join_independent.rs
"$CF_PY" -m ruff check python/tests/test_finite_inner_join_independent.py
"$CF_PY" -m ruff format --check python/tests/test_finite_inner_join_independent.py
lizard -C 8 -w crates/calc-flow/tests/finite_inner_join_independent.rs python/tests/test_finite_inner_join_independent.py
uvx --offline --from lizard==1.26.0 lizard -C 8 -w crates/calc-flow/tests/finite_inner_join_independent.rs python/tests/test_finite_inner_join_independent.py
"$CF_PY" scripts/verify_complexity_gates.py
git diff --check
```

Final result: **5 new Rust tests passed**, **24 new Python cases passed**, and
**1 existing native stream NULL test passed** (3189 unrelated unit tests filtered
out). Selected-target Clippy passes with warnings denied. The existing unit
test binary required a four-minute compilation; only the named test ran.
Lizard 1.23.0 and pinned 1.26.0 report no function above eight; maximum is seven
in each file. Complexity ratchet and changed-file Ruff/format checks pass.

Running existing tests, authoring tests and repairing failures are separate:
only the single named existing native stream NULL test is rerun. The rest are
new independent tests. Initial fixture iterations found a prohibited same-name
derived Python column and Rust's earlier-than-assumed public input rejection;
the fixtures now reflect those existing contracts. The complexity ratchet
caught six pytest parameters; they are grouped into one case parameter.
Clippy required iterator collection instead of explicit FromIterator calls.
No product behavior was repaired and no existing assertion was modified;
there is no manufactured pre-implementation RED claim.

## Verdict and limits

No product blocker found in the requested independent acceptance scope.
Full suites, coverage, performance and Windows were
not run. Reused checkpoint/identity/example evidence remains attributed to its
original run. No new proof is claimed for every possible ingress schedule,
in-flight cancellation, RSS bounds, public nullable-time support, broader SQL
or non-inner joins. Required hosted checks and independent reviewer acceptance
remain outstanding; a prior pending CI snapshot is not green evidence.

No delegation or merge. DAL-517 remains `in_progress`; the coordinator arranges
the focused reviewer handoff on the published head.
