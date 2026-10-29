# Codebase cleanup review

**Branch:** `fix/codebase-cleanup`

**Base:** `origin/main` at `b5ae227e48ac9bb8a81583a23485e2c6064cee52`

**Target:** final working-tree changes, including the new `restore_input.rs` and
`test_symbolic_compile_cache.py`. This is a local branch review, not a PR review.

## Verdict

**Approve the scoped local changes: no remaining blocking findings.** Review
covered preparation caching and registration isolation, stream/collection plan
ownership, join-recovery extraction, trusted batch-input validation, test cleanup,
and documentation consistency. No merge authority or full-CI certification is
implied.

## Resolved finding

The first cache implementation could repopulate invalidated preparation after
same-thread registration from a provider's native-compilation validation callback.
`RLock` permits that reentrancy. The reviewer reproduced it through a matrix
program's actual `_SymbolicMatrixProvider.validate`: revision changed from 3 to 4,
yet the next batch compilation returned the same cached plan.

The final monotonic invalidation generation prevents stale insertion after either
new-provider registration or lifecycle registration that leaves record count
unchanged. It covers batch/collect factories and cold/warm stream document paths.
Legal reentrant registration still returns the current result without recaching
invalidated work. Seven regressions were red before this correction and green
afterward. No unresolved blocker remains.

## Reviewed boundaries

- Runtime ownership and invalidation isolate cache entries; mode, declarations,
  and lateness distinguish preparation. Locking covers preparation and native
  compilation; failed cold compilation publishes no cache entry.
- Stream compilation returns a fresh owning plan; convenience collection retains
  fresh native compilation. Public lowering preserves validation/error ordering.
- Shared recovery input keeps field/drop order, leases, credits, partial copies,
  and cancellation boundaries. Extracted accounting formulas are unchanged.
- External inputs remain validated before lifecycle work. Dynamic internal edges,
  external operator outputs, public operator calls, and replay retain validation.
  The trusted expression method has one production caller in the batch dispatcher.
- Test changes preserve public contracts and strengthen output/fingerprint checks.
  Private imports now target their owning modules. Documentation matches compact
  append SQL, snapshot SQL restrictions, and the revised cache scope.

## Verification

The reviewer independently reproduced the resolved defect, reviewed final source
and tests, and confirmed whitespace and generated-schema stability. Test results
below were supplied by implementing agents/task owner; unchanged passing groups
were not repeated by the reviewer.

Python used worktree source in `target/python-overlay` with the published
`2026.10.27` binding, not a rebuilt wheel. The final cache command was:

```bash
PYTHONPATH=/home/wegamekinglc/.codex/worktrees/codebase-cleanup/calc-flow/target/python-overlay /home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/python -m pytest python/tests/test_symbolic_compile_cache.py -q
```

Final cache module: **29 passed**. Reentrancy/template subset: **7 expected
failures before the fix, then 7 passed**, included in those 29. Adjacent groups
reported **127 passed**, earlier **98 passed**; cleanup groups reported **49
passed** and **53 passed, 1 skipped** (Kafka feature). Groups overlap and are not
summed. Changed-file Ruff, formatting, and complexity-baseline checks passed.

Rust changes were newly compiled in this worktree with Rust 1.94/debug and
`CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target`:

```bash
cargo test -p calc-flow --lib operator::join::metadata_validation::schema:: -- --test-threads=1
cargo test -p calc-flow --test operator --test pipeline_execute -- --test-threads=1
cargo test -p calc-flow --test pipeline_execute expression_graph_rejects_schema_drift_from_external_upstream -- --exact --test-threads=1
cargo clippy -p calc-flow --lib --test operator --test pipeline_execute -- -D warnings
```

Recovery/schema: **27 passed before and after**. Operator: **27 passed**. Pipeline:
**25 existing cases passed**; an initially invalid new fixture was corrected and
its focused rerun **passed**. Thus **80 distinct Rust cases** have passing evidence;
unchanged cases were not rerun. Limited Clippy passed in **2m03s** and changed Rust
formatting passed. Owned build/test processes finished.

Changed-document links, anchors, final newlines, and spec references passed task
owner checks. Reviewer checks confirmed `git diff --check` and unchanged
`schemas/project-v3.schema.json`. No full-workspace regression, coverage,
benchmark, binding build, push, or remote CI query was performed for this review.
No measured speedup is claimed. Required remote checks remain necessary before an
authorized merge.
