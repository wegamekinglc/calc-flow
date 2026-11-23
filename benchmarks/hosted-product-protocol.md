# Public rolling SUM after checking iterable event time on Arrow buffers

Compare the reviewed input-order validator to its actual first parent on one
fresh Linux runner. Use the exact public TableExpr.stream clock, fixture and
independent numerical oracle from PR 660. A phase study on the single-window
wheel in Actions run 45132619300 completed seven observations. The three
ordinary observations had median 53.944 ms. The three instrumented ones had
median 56.422 ms. Timestamp validation consumed 4.290 ms across 100 batches
in a typical instrumented observation. This motivates the present comparison.

Each fresh run uses 100,000 time-adjacent rows, 1,000 entities, 100 batches
and one 60-second Float64 SUM. Pin one CPU and one native worker, single
threaded BLAS. Wall time starts before declaration and ends after collecting
all output Arrow tables and exiting the async context. Validate every value,
124 nulls, lower-open time boundaries, row order, exact schema and metadata,
and a single physical rolling node after every observation. Retain output
tables through the end of each timed observation. Fixture SHA-256 is
36847ac453bd58a3a0a918d993ec726d56bbddb8c3af6e8362ca940999286ab0.

Run two independent rounds with 10 alternating AB/BA pairs each, plus one
excluded warmup per arm per round. Preserve all samples, exact per-round
paired intervals and the maintained 5% regression criterion. The full public
p50 goal is 12 ms. Assess this stage through the paired Python results.

Use matching official optimized Linux CPython 3.13 ABI3 wheels, verified
from their workflow/run and successful smoke jobs. Require the same build
image/version, exact product checkout/tree, archive SHA-256, wheel/module
hashes and installed bytes. Build time is separate. Allow up to 75 seconds
for external selection and verification and 105 from actual hosted job start,
including fetching wheels and dependencies, startup, fixture, preflight,
all workers, cleanup and upload. Stop collecting after 60 seconds to retain
45 seconds for cleanup and upload. This 180-second cap fits the 480-second
remainder of a 600-second feature study after its phase reservation. The
previous complete job took 43 seconds. Preserve the maintained three-second
preflight, every one-second pressure sample and the final interval. Stop on
any correctness, resource, setup or time-budget failure and preserve partial
results. There is one formal collection on a new VM.

The independent review replaced an iterator-wide comparison with a check of
the previous-to-first boundary followed by Rust's contiguous slice sort test.
The earlier candidate and its parent had no hosted samples or warmups. Charge
its full external preparation reservation of 75 seconds and cancel its obsolete
parent build. The final official wheel is built at synthetic merge 52babf51,
whose first parent is 86781a39. Only the five reviewed product/test files differ.
The phase reservation of 120 seconds, previous external reservation of 75 seconds
and final collection's 180 seconds total 375 of the 600-second stage allowance.
