# DAL-345 CI failure recovery

This batch starts at approved Join head
`64e712cc5dc4f29f4604b79ac399623f8d11eb57`, targeting main/base
`40cddf67cdddbc49f9febb74f6c5463f08251b6b`. Source for the correction is the
commit containing this report. The two resolved Join funding findings and their
independent approval retain their original source provenance.

## Actual failures and root cause

The completed base-push Python jobs failed in
[Linux workspace error transparency](https://github.com/wegamekinglc/calc-flow/actions/runs/38818949674/job/116719799042)
and [Windows ASOF state-limit transparency](https://github.com/wegamekinglc/calc-flow/actions/runs/38818949763/job/116711806030).
They exposed a generic upstream operator failure without the downstream ASOF
reason/detail. The supplied completed job logs were fetched with `gh api
repos/wegamekinglc/calc-flow/actions/jobs/<job-id>/logs`.

The ingress preparation task owned the physical receiver. Operator failure
cancelled and joined those tasks before supervisor failure publication, dropping
the receiver and waking an upstream blocked sender. That EdgeClosed could enter
the primary scheduling round before convergence, sort before the ASOF failure by
task ID and become the first public error. The affected ingress source blob is
`8e8b755f1d7e934295e2d47a5eb4e5592e36625d` in both base and approved head.

Before correction, running the two reported Python tests with two workers gave
**1 failed, 1 passed** in 0.78s: the state-limit test exposed
`operator "quotes" execution failed` with `reason_code=None`. The workspace
test passed in that local run; its actual failure remains evidenced by CI.
A new Rust test with a real ASOF state-limit failure and a blocked upstream
reproduced **two primary errors**, upstream TaskId(0) EdgeClosed before downstream
TaskId(1) AsofStateLimitExceeded, with the original detail still present second.

Preparation now borrows its receiver. Completion returns the receiver even on
cancellation or contained panic, and cleanup restores it into the operator's
retained frame. Existing supervisor publication and convergence precede dropping
that frame. The failure ordering protocol, independent primary errors, original
reason/detail and cleanup error priority stay intact. No Python error-selection
heuristic, public API, logical fee, pool limit or checkpoint format changed.

## Focused validation

Rust 1.88.0; `CARGO_BUILD_JOBS=2`, `CARGO_PROFILE_DEV_DEBUG=0`,
`CARGO_PROFILE_TEST_DEBUG=0`. Native wheels were built under `target/` using a
task-local CPython 3.13.9 environment with PyArrow 24.0.0 and pytest-xdist.
CI used CPython 3.13.16; Windows was diagnosed from its supplied log.

```bash
cargo test --locked -p calc-flow --features benchmark-checkpoint-diagnostic \
  --lib test_asof_admission_cleanup_preserves_reason_before_upstream_edge_close \
  -- --test-threads=1 --nocapture
target/debug/deps/calc_flow-abab16b2e36df81b error_propagation_tests \
  --skip test_asof_admission_cleanup_preserves_reason_before_upstream_edge_close \
  --test-threads=1
target/debug/deps/calc_flow-abab16b2e36df81b ingress_admission --test-threads=1

target/dal345-venv/bin/maturin build --locked --profile dev \
  --interpreter target/dal345-venv/bin/python --out target/dal345-wheels
uv pip install --python target/dal345-venv/bin/python --reinstall --no-deps \
  target/dal345-wheels/calc_flow_python-2026.10.13-cp313-cp313-linux_x86_64.whl
target/dal345-venv/bin/python -m pytest -q -n 2 \
  python/tests/test_asof_stream_error_transparency.py \
  python/tests/test_asof_stream_results.py::test_asof_state_limit_error_is_explicit_and_emits_no_wrong_match \
  python/tests/test_asof_stream_results.py::test_asof_source_rejects_null_identity_before_operator_admission

cargo clippy --locked -p calc-flow --lib --tests --all-features -- -D warnings
cargo fmt --all --check
git diff --check
git diff --exit-code -- Cargo.lock schemas/project-v3.schema.json
```

Results: **16 Rust tests passed** (new regression 1, prior propagation cases 4,
affected ingress cases 11). The new case keeps only ASOF as primary, preserves
the exact native detail, retains EdgeClosed as secondary and emits no match.
Panic cleanup additionally verifies the physical receiver stays alive until
the retained frame drops. Backpressure, read-ahead, cancellation, admission
priority, checkpoint continuation and independent failure identities pass.
The **6 Python tests pass** in 13.23s, including both reported failures and
convenience/explicit success and failure paths. Core Clippy, fmt and drift checks
pass. Lizard 1.23.0: 33 functions across the touched runtime/test modules, max CCN 7.
The rebuilt wheel SHA256 is
`f12bc4bfe38b911cb3bdfab58d452a2da9c89bbb4f6980e85659ab4639163a50`.

## Coverage baseline recovery

[Coverage baseline job](https://github.com/wegamekinglc/calc-flow/actions/runs/38829651552/job/116753370202)
correctly rejected an incomplete base. Its only flag is rust; the Python suite
failed before report/artifact upload. Base push run `38818949674`, attempt 1,
has rust-coverage artifact `12016315336` and no python-coverage artifact.
[Coveralls build 82398704](https://coveralls.io/builds/82398704) measures the exact
base SHA with 94.08244343636414% rust coverage; its rust flag reports
`full_number=38818949674.1`. The -0.1% aggregate/-0.07% rust comparison statuses
are failure. A high percentage alone does not complete this baseline.

Merged PR #450 head `1b85368c2e4588523e6d92c9f9e8dacd567e48de` has complete
successful reports, but its full tree `59394972682c6779f59d75cadb49ec6f606e1e66`
differs from base tree `75376486724f9b5fb22598067bb804e5213048ae`. It cannot be an
alias under the current same-tree policy. The resolver, policy tests, workflows,
Coveralls settings, status values and coverage floors are unchanged.

After root-cause diagnosis and passing focused correction checks, exactly one
rerun of the failed base Python job was requested through the ordinary API:

```bash
gh api --method POST \
  repos/wegamekinglc/calc-flow/actions/jobs/116719799042/rerun
```

The request succeeded. This runs the immutable old base and its original suite
assertions/90% floor; it does not apply this fix to old source. The goal is a
genuine missing Python measurement on the exact base/run, followed by the
unchanged resolver's flag/report identity checks. No retry loop or CI watch was
started. Completion and restored-baseline validation remain a coordinator handoff.
Only a real successful measurement can restore the baseline. If the diagnosed
old-source race fails again, stop retrying and retain the blocker for explicit
baseline-recovery coordination; do not alter statuses or select a different tree.

Once both exact-base reports exist, rerun the failed PR coverage-baseline job
and its dependent coverage/gate jobs through the normal Actions rerun mechanism.
The resolver already accepts complete direct same-SHA flags even when their
comparison status decreased (covered by its existing policy test); no change to
that rule is proposed. If the base remains incomplete, #453 remains blocked from
merge. Original Join performance limits, including development CPU +20–21% and
unmeasured release/#443 targets, remain unchanged. Full regression, cross-platform
and combined 90% coverage remain CI responsibilities.
