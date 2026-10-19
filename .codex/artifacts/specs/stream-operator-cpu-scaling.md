# Streaming operator CPU scaling

The accepted Join implementation at `e35ff1190e8eba08ad3c3135299bbca6ac029698`
is the baseline. Extend its useful principles to existing streaming calculations:
CPU-derived concurrency, enough work per dispatch, reuse of preparation facts,
immutable worker input, deterministic owner publication, and owned cancellation.

## Scope

- ASOF: let explicit edge and workspace budgets determine the finalization
  prefix, decouple temporary computation routing from persistent key shards,
  and keep the persisted shard layout and checkpoint identities unchanged.
- Rolling mean and dual-SMA: replace the two-lane numeric owner with dynamically
  planned lanes, preserve whole-entity arithmetic order, reuse existing entity
  counts, and build stable schedules in one pass. Large batches must be eligible;
  remove private size ceilings while retaining checked owned scratch accounting
  and the existing explicit logical state budget.
- Cross-section exact-time mean: retain complete time-group boundaries from the
  existing order proof, compute independent groups in the managed CPU service,
  and assemble the original output on the owner. Keep each group's arithmetic
  order, output geometry, rollback, and checkpoint wire format.
- Row-local SQL: reuse the already proven immutable compiled-expression path
  for repeated nonaggregate SQL with no selected UDFs. Ineligible plans continue
  through DataFusion. Stateful aggregate/reduction semantics are unchanged.

Pure column projections already share Arrow buffers. Simple vectorized
expressions already cache immutable physical expressions; do not add speculative
thread splitting to their cheap kernels. Window and cumulative SQL aggregates
use ordered state transitions and compensated floating-point accumulators. The
safe boundary for those is a complete independent state slot, not an arbitrary
row chunk; this change does not replace their accumulators with partial sums.

## Required invariants and checks

Each behavior starts with a focused failing test. Validate CPU/work/entity
selection, actual worker overlap, stable schedules and routes, output and
metadata equality, original float bits, deterministic failures, cancellation,
abandoned futures, explicit budget admission, full refunds, and state recovery
where the changed path owns state. No new public API or project schema drift.
Keep persistent user inputs read-only. Never dispatch barriers or state commits
to independent workers.

Use focused module tests, scoped Clippy/format checks, and independent specialist
review. Keep generated native libraries and all measurement/build artifacts under
the root `target/` tree. Do not run the full workspace or coverage matrix locally.

## Performance evidence

Before acquisition, freeze the exact clean candidate SHA, exact baseline,
matching release builds and harness identities, selected cases, sample count,
resource limits and total budget. The proposed five configurations cover large
ASOF, rolling, cross-section mean, repeated row-local SQL, and a single-CPU
rolling control. Select the smallest geometry exercising the changed paths;
one-million-row cases are needed for the removed large-batch ceilings.

Use maintained full-output oracles and paired statistics: two rounds of ten
alternating AB/BA pairs per configuration, one excluded warmup per worker,
200 formal observations and 20 warmups. Acquisition, fixture setup, startup,
validation and cleanup share 600 seconds. Estimate and report release builds
separately. Stop on interference, identity/oracle/protocol error or deadline;
preserve partial evidence and do not retry or expand for a favorable verdict.

Report each selected case and uncertainty. Separate same-resource version
comparisons from descriptive CPU ratios. Do not relabel older Join release
artifacts or timing results as measurements of this candidate.
