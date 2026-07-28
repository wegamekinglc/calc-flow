# DAL-5 5.2 Option C Acceptance Audit

Authority:
`.codex/artifacts/api-notes/dal-5-5-2-main-reconciliation.md`.

Baseline: `f5282fdc4965ed687df736400d8c0717db51e992`.

| Row | Surface | Result | Evidence |
| --- | --- | --- | --- |
| 1 | Python value object | Pass | Native frozen `ExecutionOptions`; omitted, `{}`, and explicit `None` settings tests |
| 2 | Settings copy | Pass | Root/nested one-pass `Mapping`, defensive getter, mutation, and alias tests |
| 3 | Strict JSON | Pass | Exact scalar/list rules, duplicate/surrogate/range/depth/cycle/path/redaction tests |
| 4 | Deadline | Pass | Positive/negative/zero offset normalization and microseconds `0`, `1`, `999999` |
| 5 | Deadline errors | Pass | Naive/invalid/tz hook/range failures have fixed messages and no cause/context |
| 6 | Omitted options | Pass | Omitted, `None`, and empty options retain default execution behavior |
| 7 | Native plumbing | Pass | Rust/PyO3/Python full suites pass with the retained native owner |
| 8 | Opted-in sync provider | Pass | Public and mapped callbacks receive defensive `ProviderContext` copies |
| 9 | Opted-in async provider | Pass | Async context delivery plus cancellation linearization tests pass |
| 10 | Legacy provider | Pass | Default exact `False` remains the two-argument one-shot ABI |
| 11 | Concurrent isolation | Pass | Same-plan A/B barrier test and manually authenticated pending lock test |
| 12 | Expired deadline | Pass | Sync/async zero-provider-entry tests remain green |
| 13 | Cooperative deadline | Pass | Post-provider deadline check wins over provider error |
| 14 | Rollback | Pass | Recovery/input/snapshot/marker priority, restore counter, GC, and run C reuse |
| 15 | Asyncio cancellation | Pass | Result/error-wins, cancellation-wins, repeated cancellation, queued marker tests |
| 16 | Registration | Pass | Exact bool, no partial registration, and revision isolation tests |
| 17 | Worker reconstruction | Pass | Immutable whole-tuple preflight; public/mapped missing/false/true; zero mutation |
| 18 | 5.1 stability | Pass | Capability suites and frozen source projection checks pass |
| 19 | Schema stability | Pass | Eight named checkpoint/project/REST/OpenAPI/TypeScript paths have zero baseline diff |
| 20 | Type surface | Pass | Native classes, positional constructor, `ProviderContext`, and `accepts_context` retained |
| 21 | Documentation | Pass | Strict Mapping, any-aware UTC normalization, race precedence, Studio limit distinction |
| 22 | Full gates | Pass | Rust/Python/Studio/frontend/e2e/coverage/security/release/wheel matrices pass locally |

## Frozen non-change projection

The following paths are byte-identical to the baseline:

- `crates/calc-flow/src/checkpoint.rs`
- `python/calc_flow/capabilities.py`
- `schemas/project-v2.schema.json`
- `web-ui/backend/src/calc_flow_studio/app.py`
- `web-ui/backend/src/calc_flow_studio/models.py`
- `web-ui/openapi.json`
- `web-ui/src/api/schema.d.ts`
- `web-ui/src/api/decoders.ts`

The worker continues to call `plan.execute(batches)` without project/REST
settings or deadlines. `serialized`, `lazyBuiltin`, `unavailable`, exact
selection, and the lazy built-in set are unchanged.

## Environment-only reruns

- The first PyO3 test invocation could not locate `libpython3.13.so.1.0`;
  the repository's existing Anaconda library path produced `69 passed`.
- Coverage CLIs were absent. Lock-version tools were installed under the
  isolated worktree and coverage passed at `91.14%` lines.
- The first isolated `uv build` inherited no rustup default from the sdist
  sandbox. Explicitly passing the repository's pinned Rust `1.88.0` toolchain
  built both the sdist and ABI3 wheel successfully.
- Frontend commands used installed Node `20.20.2`; system Node `20.18.0` is
  below the lockfile engine requirement.
