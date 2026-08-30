# SCE-13 Static Array Snapshot - Critic Critique

## Target

- API note: `.codex/artifacts/api/sce-13-static-array-snapshot.md`
- Evidence head: PR #214 at `f882d2133e5805e80b729089877baed91bb0b890`
- Base: `origin/main@760eea4`
- Controlling contracts: `docs/introduction.md`, `.codex/artifacts/specs/symbolic-computation-contract.md` D10-D12, and `.codex/artifacts/analysis/sce-11-static-stream-inputs-review-record.md`

## Verdict

**Revise**

The manually approved owned two-type surface is implementable and internally coherent. `#[non_exhaustive]` plus private snapshot fields gives the intended downstream match/construction boundary; the detached deep copy, read-only accessors, path-first errors, transient worker lifetime, and success-only cache rule are sufficient without exposing the latched payload or creating a shared crate. Two blocking omissions can still let PR #214 land with an inherited contract violation or with tests that do not prove the reviewer-critical behavior.

## Findings

### Blocking Issues

- **The note omits SCE-11's mandatory declaration-kind versus port-kind validation for the first consumer** - The SCE-11 review record explicitly binds four follow-up obligations to the first consuming operator/provider; obligation 4 requires cross-validating a static declaration's kind against the graph port kind (`.codex/artifacts/analysis/sce-11-static-stream-inputs-review-record.md:55-70`). The current project validator gathers only unconnected port names and checks that the declaration names one of them (`crates/calc-flow/src/config.rs:1861-1883`); it does not retain or compare the port's `BatchKind`. Consequently, an array declaration can still name a table port (and vice versa), compile, and fail only after the new consumer starts using the value. The revised note says SCE-11 otherwise remains intact (`sce-13-static-array-snapshot.md:281-284`) but never assigns this activated obligation or adds an acceptance test. This is a compile-time correctness gap, not a request to broaden the public snapshot surface.
  - **Suggested fix:** Add a normative implementation requirement that project validation compares `StaticInputSpec::{Table, Array}` with the exact unconnected port kind and fails at `static_inputs[index].kind` before plan construction/provider work. Add both mismatch directions to `crates/calc-flow/tests/project_connector_compile.rs`. Keep the snapshot API unchanged.

- **The directed gates do not execute or deterministically prove the required negative API and Tokio behavior** - Test 5 requires `compile_fail` rustdoc examples for absent `Debug` and private construction/mutation (`sce-13-static-array-snapshot.md:318-320`), but the only concrete documentation gate is `cargo doc -p calc-flow --no-deps` (`sce-13-static-array-snapshot.md:292-300`); `cargo doc` renders documentation and does not run doctests. The implementation can therefore retain the current `Clone, Debug` derives or accidentally expose fields while every listed command passes. Test 7 also refers to an “existing large-static Tokio progress probe” (`sce-13-static-array-snapshot.md:326-329`), but no such test exists at the evidence head; the current placement occurs synchronously before `spawn_blocking` (`crates/calc-flow-python/src/provider.rs:385-413,459-506`). A timing/large-input check is not a deterministic proof that snapshot cloning, Python-list construction, NumPy reshape, and JAX placement all begin inside the worker.
  - **Suggested fix:** Add an explicit focused `cargo test -p calc-flow --doc ...` command (or an equivalent external compile-fail harness) to the required gate. Replace the nonexistent “existing probe” with a deterministic test contract: a test-only placement gate signals only after the blocking worker enters placement, holds that worker while a current-thread Tokio task advances, and then releases it. The test must also prove cache absence on placement error/cancellation and cache installation only after a successful, non-cancelled worker return.

### Significant Concerns

- **All-false null bitmaps are not covered at the consuming-provider boundary** - The semantics correctly preserve `Some(all_false)` and reject only when a logical null is present (`sce-13-static-array-snapshot.md:141-145,219-220`). The provider matrix asks only for generic “null rejection” (`sce-13-static-array-snapshot.md:321-325`). An implementation that rejects every `Some(bitmap)` could satisfy that negative test while rejecting a valid latched array.
  - **Suggested fix:** Add one placement success case for `Some(all_false)` (and `Some(&[])` on a zero-sized array) plus one mixed-mask rejection case that verifies the exact `.nulls` path.

- **The additive-evolution statement needs the standard auto-trait qualification** - The approved owned fields and variant carriers automatically make both public types observable as `Send`, `Sync`, `Unpin`, and unwind-safe traits. Adding a future private field or enum variant is compatible only if it preserves the already-observable auto-trait set; `#[non_exhaustive]` does not protect that dimension (`sce-13-static-array-snapshot.md:243-260`). This does not block the current concrete representation, whose fields are compatible with those traits.
  - **Suggested fix:** Qualify the future-evolution sentence accordingly. Also turn “avoid freezing” (`sce-13-static-array-snapshot.md:124-125`) into an explicit implementation instruction to remove the current unnecessary `Clone` derive along with `Debug`; otherwise PR #214 silently freezes more trait surface than the advertised minimum.

- **The owned-copy cost is explicit but should not be called bounded without naming what is bounded** - The note correctly freezes an O(n) detached copy per public call, places the runtime copy once per job/static input inside `spawn_blocking`, and deliberately excludes that internal clone from `static_placement_bytes` (`sce-13-static-array-snapshot.md:134-188`). The bounded property is lifetime/frequency, not peak bytes: during placement the latch, snapshot vectors, Python list, and provider array can coexist.
  - **Suggested fix:** In rustdoc/API reference, say “one transient O(n) snapshot clone per placed static input” and explain that the metric counts logical provider transfer, not peak memory or the internal clone. No new metric or benchmark is required by this note.

- **Redaction tests should explicitly cover null-position secrecy** - Test 8 names raw-value and address sentinels but not the separately forbidden null positions (`sce-13-static-array-snapshot.md:196-205,330-332`).
  - **Suggested fix:** Use a distinctive mixed mask and assert that final errors contain the fixed path-first null message without indices or a rendered bitmap.

### Minor / Style Notes

- The selected artifact path is `.codex/artifacts/api/`, while established notes live under `.codex/artifacts/api-notes/`. The issue explicitly chose the former, so this is not blocking; the checked-in note, review handoff, and blob-hash witness must simply use one canonical path consistently.
- `#[non_exhaustive]` on the private-field snapshot struct is redundant for current construction privacy but useful as an explicit evolution signal. It is correct to keep.

## Counter-Proposals

Do not change the approved public shape. Keep the owned `StaticArraySnapshot`, non-exhaustive `StaticArrayValues`, five accessors, and owned `Batch` method exactly as written. Amend only the inherited project-kind requirement and the executable/deterministic gates; the significant concerns can be folded into the same one-time correction without another design loop.

## Axis Audit

- **Correctness:** Snapshot materialization, compact carriers, empty arrays, float32 carrier semantics, and fail-closed wildcard handling are sound. Declaration/port kind validation is missing.
- **Hidden assumptions:** The note assumes static declaration kind already agrees with the graph port and assumes a Tokio progress probe already exists; neither is true at the evidence head.
- **Missing edge cases:** All-false masks and null-position redaction need consumer coverage. Empty arrays, non-latched arrays, tables, carrier mismatch, future variants, and overflow are otherwise addressed.
- **Backwards compatibility:** The two crate-root types and method are additive in 4.x; deleting `#[doc(hidden)]`, adding `#[non_exhaustive]`, and removing unreleased PR-only derives are safe before merge. Project/checkpoint/Python/OpenAPI contracts remain unchanged. Preserve observable auto traits during future evolution.
- **Performance:** One transient O(n) snapshot copy per static input per job is the approved cost. Placement remains off the Tokio task and is cached once; no repeated benchmark is needed.
- **Surface and ergonomics:** The five-accessor owned shape matches the manual approval and is sufficient for PyO3. No additional typed accessors or constructors are needed.
- **Test plan:** Proportional after the executable doctest gate, deterministic worker gate, project-kind tests, and all-false mask case are added.
- **Risk and scope:** The seam is load-bearing but isolated. Neither blocker requires a shared payload crate, serialized surface, or API redesign.

## Questions for the Author

1. Will the one-time correction explicitly assign the SCE-11 F7/obligation-4 project-kind validation to SCE-13?
2. What exact executable command will run the negative API examples, and what deterministic test seam will replace the nonexistent large-static progress probe?
