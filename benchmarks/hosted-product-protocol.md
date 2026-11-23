# Public single-row Program input mapping

Compare the single-input ordinary-dict lookup with its reviewed wheel's actual
first parent on the public Program.session().push path. The input is an ordinary
dict keyed by the one logical source, containing the already formed native Batch.
The stage removes two key sets and a second equality comparison on valid input.
The fixture, worker and clock come from the paired session record export study.
Cold release builds are separate.

The single timed workload is WarmSessionScenario with dual_sma_spread, 4,096
warm rows and one appended row, one entity, exact projection, checkpointing off,
one native worker and one pinned CPU. Every fresh process prepares the input,
requests forced GC, then times a public push with the explicit Watermark through
the returned named Arrow table. Preserve the warm-history and new-row digests,
numeric and ordering oracle, optimized physical plan, zero pending rows on clean
finish, and the argument/native/result phase probe. Preparation and verification
surround the individual timer and count toward the job budget.

Run two independently counterbalanced rounds with 10 AB/BA pairs in each and
four excluded warmups in total. Use the existing exact per-round paired
intervals, pooled medians, p95 and maintained 5% regression rule. The interim
targets remain 80 µs p50 and 120 µs p95. Report all 40 raw timings and four
warmups. Use a new hosted Ubuntu runner, the established three-second preflight,
per-second CPU and SMT sibling, memory and paging guard, and single-threaded
BLAS. Stop on the first resource, oracle, provenance or deadline failure.

The stage budget is 600 seconds across this one workload. Allow 120 seconds
for source, CI build and artifact selection/verification outside the runner and
480 seconds from hosted job start through verified wheel downloads, matched
ABI3 CPython 3.13 package installs, samples, cleanup and upload, reserving 45
seconds for final cleanup. The prior hosted study took 58 seconds end to end.
Record actual checkout/tree, exact release job and image, archive checksum,
wheel and native hashes from each separately tested package. No second formal
collection is allowed. The public time measures Python and native runtime together.

The final reviewed adapter is 594009d7. The release smoke job checks out
68a91699, whose first parent is 6a52ff2e. Their complete Git diff consists
of session.py and its focused test module. Only session.py ships in the wheel.
Verify its SHA-256 against the corresponding Git blob on each arm's installed
package before collection. The main job for that first parent was superseded
before its wheel upload. A dispatch pins the same parent and reuses each step
of its official smoke job. Record separately built native module identities.
