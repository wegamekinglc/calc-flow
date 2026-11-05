# Inline grouped integer SUM state

The frozen comparison accepts the candidate. It removes duplicate per-group
integer SUM storage and repeated field-metadata construction. All three selected
cases have an exact maintained `improved` verdict after both rounds, complete
correctness checks, required profiles, source/helper post-seals, and cleanup.

| Case             | Baseline P50 ms | Candidate P50 ms | Change  | Verdict  |
|------------------|-----------------|------------------|---------|----------|
| Dense, one CPU   | 263.467         | 189.067          | -28.24% | improved |
| Dense, four CPUs | 250.388         | 172.311          | -31.18% | improved |
| Sparse, one CPU  | 27.236          | 23.442           | -13.93% | improved |

These are pooled wall-time medians within this cohort. The maintained verdict
uses each round's paired-median intervals, retained in the
[sealed report](sum-inline-state-20261105/README.md). The prior experiment's
absolute timings belong to a different cohort and are not combined with these.

Each dense run has 10,000 seed rows plus twenty 100,000-row input batches;
the sparse run has the same seeds plus twenty 100-row inputs. Each retains
twenty-one complete snapshots over 10,000 groups. The primary timer starts
after IPC preload and covers graph construction, planning, execution, retained
output collection, natural end, and cleanup. Oracles run after that timer.
These measurements are whole-run elapsed times, not per-row or per-batch latency.

## Scope and mechanism

Exactly one trusted built-in, non-distinct, unordered grouped SUM with an
Int64 or UInt64 result stores a compact typed nullable integer. Its result
comes from that same state. The existing DataFusion accumulator still owns
arithmetic, history reduction, filtering, and null/wrapping behavior.

The immutable bound plan caches only the type and NULL template. This avoids
reconstructing DataFusion field metadata for each group. It never caches a
committed sum. Generic aggregates retain their existing state/result vectors;
stack state views and borrowed values adapt seeding and export. The thread
model, public configuration, project schema, and checkpoint wire layout remain
unchanged.

Candidates remain isolated until output acceptance. The focused controls cover
Int64/UInt64 NULL-tail and sparse updates over 8,193 groups, wrapping, held
outputs, discarded updates, restore continuation, refusal/retry, cancellation,
and global/generic paths. Eight new and nineteen affected existing Rust tests
passed; strict scoped Clippy, formatting, and complexity checks passed.

## Component allocation evidence

The synchronous probe creates and finalizes 256 existing-key candidates and
keeps them alive through the counter. Fixture/native arrays and its retained
candidate buffer are prepared outside this first counter.

| Metric            | Legacy | Candidate |
|-------------------|--------|-----------|
| Total allocations | 1792   | 256       |
| Held allocations  | 1024   | 256       |
| Held bytes        | 104448 | 4096      |

The mixed SUM+COUNT control remains at 3,840 total allocations, 1,536 held
allocations, and 145,408 held bytes. Group, Candidate, and map-row layouts are
unchanged at 96, 128, and 136 bytes on this host; target-relative tests compare
against the old-equivalent layout on every platform. The 992-byte plan fits
the existing 4,096-byte structural charge.

A separate counter includes full candidate-map, group-buffer, and index
capacity. Its inline peak is 111,648 bytes against 407,552 prepaid bytes;
the generic mixed case is 252,960 against 452,608. Those are different query
shapes, not a legacy-versus-candidate memory comparison. Exact-credit and
one-byte-short refusal tests preserve committed capture and accounting.

These counters establish the component change, not whole-job allocation or
RSS. The IPC test checks legacy Vec-equivalent and inline representations
through the shared unchanged writer, not a separately generated historical
golden checkpoint. Unchanged descriptors/policy, import validation, continuation,
and existing recovery controls provide the additional compatibility evidence.

## Resources and profiles

| Case             | CPU P50 ms, baseline to candidate | Whole-process peak RSS P50 MiB, baseline to candidate |
|------------------|-----------------------------------|-------------------------------------------------------|
| Dense, one CPU   | 263.453 to 186.243                | 231.06 to 227.19                                      |
| Dense, four CPUs | 297.202 to 219.231                | 243.88 to 239.95                                      |
| Sparse, one CPU  | 27.236 to 23.443                  | 70.52 to 66.50                                        |

Average active cores in the four-CPU case are 1.189 and 1.280 respectively.
CPU work decreases along with wall time. Whole-process RSS includes IPC inputs
and twenty-one retained complete outputs; it is not an isolated operator
working set, a state budget, or a new engine memory guarantee.

Both dense single-CPU profiles passed the five-window and timestamp/loss gates.
The baseline has 1,288 selected user-CPU samples and the candidate 878;
unresolved leaves account for 374 and 184 samples. All 3,016 and 2,441 raw
timestamps match the imported profile clock, with maximum error below
`6e-8` milliseconds. Nearly all selected CPU samples belong to one worker
in each profile.

Group destruction's inclusive share falls from 14.75% to 4.78%, candidate
construction from 14.44% to 8.20%, and scalar destruction from 2.87% to 0.80%.
The maintained disjoint commit bucket falls from 14.67% to 4.90%. These
observations support the allocation hypothesis alongside the component tests;
inclusive counts overlap and do not measure separate phase wall times.
Missing or inlined symbols do not imply zero cost.

The candidate still spends substantial sampled CPU in chunk updates (49.77%),
transaction finishing (22.67%), and expression work (21.07%) under the
maintained disjoint classifier. Row encoding has 125 baseline versus 129
candidate inclusive samples; its larger percentage after the total shrinks
does not establish a regression. The complete stacks and attribution limits
are retained in the [profile evidence](sum-inline-state-20261105/profile-summary.json).

## Fixed comparison

The comparison uses three cases: dense single CPU, dense four CPUs
`[0,2,4,6]`, and sparse single CPU with retained history. Each has two rounds
of ten alternating pairs: 120 formal observations and twelve warmups. Both
products require a dense single-CPU profile with five accepted processing
windows, excluding the first of six jobs. All twenty-one snapshots are checked
against the independent oracle.

The 600-second budget includes preparation, guards, startup, warmups, correctness,
profiling/decoding, resources, statistics, publication, and cleanup. Builds are
reported separately. One acquisition completed in **102.02959463198204 seconds**,
including all those phases, with zero retries or additional observations.
All 144 workload jobs (120 formal, twelve warmup, twelve profile) pass the
full-snapshot oracle. All 135 owned invocation receipts confirm process cleanup.
The [manifest](sum-inline-state-20261105/evidence-manifest.json) records the
stopped clock and all eight sealed-file hashes. The maintained
paired-median rule must report improvement for at least one dense target and
no confirmed regression for any selected case. Inconclusive timings establish
neither equivalence nor improvement.

Generic/UInt64/nullable/checkpoint latency, finite execution, hot-key workloads,
and other thread counts are unmeasured. The selected workloads keep one
DataFusion partition and default input partitioning, using the identical audited
common657 harness. The previous rejected parallel experiment remains sealed.

## Source and build identity

The exact baseline is main
`222f4e2d88f56b5b4ce635abab24eceed732da6d`. The old af400 product differs
in compiled source and was rejected for reuse. A fresh baseline build took
597.861 seconds and produced binary SHA-256
`c99c5bb268f320d71f4efb957d2ac4a709ddc231751536836f1d64a9c0e2c83b`.

The public candidate is
[74a1eaf2](https://github.com/wegamekinglc/calc-flow/commit/74a1eaf2183adfd04fcb1aeecd655c6df73ed16f),
tree `04b554c37f0afe8e40fa8e821b9c6b808418f558`, identical to the reviewed
source/tool tree. A separate frozen worktree owns its build and common657-only
overlay. Source inventories, actual fresh compiler records, dependency proof,
binary receipts, helper seals, and final protocol passed the acquisition gate
and required post-acquisition checks.

The candidate build completed in 548.642 seconds, producing binary SHA-256
`8378ba7b8683780868604ce70568aa626f61e0abcd1742ec84f050b95aef1ed3`.
The core dependency record includes the new production state module. Both builds
use the same pinned compiler, lockfile, feature set, and audited harness.

Both core and example compiler records are genuinely `fresh=false`; public Git
blob comparisons cover 1,004 baseline and 1,008 candidate source inputs, with
the common657 overlay as the sole exception. The live source and binary seals
remain valid after acquisition. The frozen
[protocol](sum-inline-state-20261105.protocol.json) SHA-256 is
`1bb73519f7e11e686fbe562fa6490597e272ccb9bf6fad758faf1f43d4c9f6c4`.
Its baseline/candidate build receipt hashes are `3f2dba8c` and `49f7a092`.
The [complete evidence](sum-inline-state-20261105/complete-evidence.json.gz)
retains full receipts, source inventories, actual compiler/dependency proof,
all formal/warmup records and snapshot oracles, resource observations, commands,
profile clocks, helper/native post-seals, and cleanup. Large original perf files
remain at recorded local paths with hashes; imported full stacks are public.

## Upstream integration after acquisition

After acquisition, main advanced to
`49525978f0a4731ee2e2ad8ce01faefff86a2f4a`, adding the explicit opt-in
`auto_rows` partition policy. The delivery integrates that commit, including its
Cargo configuration, generated schema, Python surface, documentation, and tests.
The unchanged SUM implementation and reviewed measurement helpers still match
public candidate `74a1eaf2`.

The common657 workload explicitly selected Fixed mode and one target partition.
Source review finds that selected decision path unchanged by the new policy;
the default also remains Fixed. The integrated product and `auto_rows` mode
were not performance remeasured. Every timing in this report remains bound to
the frozen comparison of `222f4e2d` and `74a1eaf2`. Final CI must cover the
combined delivery; the sealed measurements and original build/source identities
remain unchanged.

## Next bounded hypothesis

The accepted change lowers serial state-management cost without adding workers.
The remaining row-key conversion and Arrow filtering work are candidates for a
separate study of redundant string-key encoding or filter copies. Their
inclusive shares overlap with other frames and promise no independent speedup.
No further prototype, cases, profiles, or samples were run using this campaign's
unused budget.
