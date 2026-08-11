# Public Continuous Streaming A6 - Critic Critique

## Target

- Spec: `.codex/artifacts/specs/a6-public-continuous-runtime.md`
- API note: `.codex/artifacts/api-notes/a6-public-continuous-runtime.md`
- Baseline: `main@70f7e3d1e9306c419a0b2358527ec888c2ed9934`
- Reviewed spec SHA-256:
  `bf60b2f401df35ef500c27b27db7861fd2fc1df3040495428c31c985f6180b72`
- Reviewed API-note SHA-256:
  `f5305d1786b5b388397c544317beb2ce28ae4cb489988fe0a072ab9600ff31e7`

## Verdict

**Proceed with caveats**

All six required axes have zero blockers. The second revision closes the two
remaining failures without reopening the previously approved checkpoint,
recovery, capability, or cutover contracts. The package is ready for the
authorized documentation-only A6-01 assembly, review, and draft PR; it is not
evidence that the runtime implementation already exists.

## Six-Axis Conclusions

- **Ownership — Zero blockers.** Rust keeps consuming `start(self)` and one
  non-cloneable job. The sole checkpoint constructor now accepts one local
  managed root, derives `state` and `manifests` internally, and retains both
  complete-root and lineage leases through settled cleanup
  (`.codex/artifacts/api-notes/a6-public-continuous-runtime.md:334`). Python and
  PyO3 expose the same one-directory boundary with no backend or two-root
  escape hatch.
- **Checkpoint completion — Zero blockers.** `absent`, `installed-unknown`,
  and `durable` remain distinct; manual success follows parent-directory sync,
  while installed-unknown preserves sink intent, advances no durable epoch,
  and permits both post-crash persistence outcomes
  (`.codex/artifacts/api-notes/a6-public-continuous-runtime.md:730`).
- **Recovery — Zero blockers.** Every canonical candidate is validated, an
  invalid lower or higher epoch poisons the lineage, all-valid sparse recovery
  selects the highest epoch, and terminal recovery exposes only fresh
  sink-scoped evidence
  (`.codex/artifacts/api-notes/a6-public-continuous-runtime.md:779`).
- **Capability proof — Zero blockers.** Exact pause/report/seek is explicit in
  identity and status, proof is derived per reachable output, and the test plan
  includes a declared-exact source that ignores its restored cursor
  (`.codex/artifacts/api-notes/a6-public-continuous-runtime.md:799`).
- **Redaction — Zero blockers.** Every fallible owner/runner/job method has one
  failure shape, `CalcFlowError::Streaming(StreamingError)`; construction is
  private, category mapping is exhaustive, Rust formatting and source chains
  retain only safe fields, and Python exceptions discard raw cause/context
  (`.codex/artifacts/api-notes/a6-public-continuous-runtime.md:499` and
  `.codex/artifacts/api-notes/a6-public-continuous-runtime.md:1407`).
- **Cutover — Zero blockers.** The atomic inventory covers crate exports,
  native registration, Python exports/stubs, Studio-private v2 persistence,
  examples, benchmarks, active docs, generated-contract stability, and clean
  package/release smoke. The A6-01 PR is explicitly documentation-only and
  cannot claim runtime completion.

The connector signatures continue to borrow or own immutable `Batch` values,
consistent with `docs/introduction.md:91`. They do not move table calculation
outside DataFusion, cross raw tables/arrays over runner boundaries, or add a
Python expression evaluator.

## Findings

### Blocking Issues

None.

### Significant Concerns

None.

### Minor / Style Notes

- **Exercise hierarchical managed-root contention during implementation.** A
  complete-root lease should reject not only equal canonical roots but also a
  second managed root nested under the first owner's `state` or `manifests`
  subtree. Otherwise the baseline manifest scanner rejects the nested owner's
  directories as non-files (`crates/calc-flow/src/state/transaction.rs:920`).
  This does not require another contract revision: “exclusive lease covering
  the complete canonical managed root” already controls the result. Add an
  ancestor/descendant managed-root subprocess case beside the specified
  same-root and symlink cases.

## Counter-Proposals

None. The single-root owner and single safe-error conversion seam are the
smallest coherent corrections to the baseline.

## Questions for the Author

None.
