# Public 10k-group changelog for the single native group hash

Compare only the PR 649 group lookup change, using the official fresh public
21-batch sparse regression case already used for PRs 633 and 642. Bind the
actual tested CI first parent and candidate, or two artifact source trees
whose only product difference is the reviewed PR. Before acquisition verify
same Ubuntu Linux CI image, release profile, toolchain, CPython 3.13 ABI3,
actual checked-out source tree, immutable artifact, wheel and module hashes.
If a comparable pair or an attributable boundary cannot be established,
report it without timing.

Two independent ten-pair rounds with alternating execution and one excluded
warmup per arm per round, independent workers, one CPU/Tokio/Rayon/BLAS
thread. The timer covers declaration, stream start, 21 input batches,
collection of all Arrow tables, EOF and cleanup. All observations must pass
the existing independent every-prefix/schema/reconstruction oracle: 12,000
input rows including 10,000 seed keys; 13,200 changes in output (10,000
initially, then twenty batches of 160 changes). Report the exact maintained
round intervals, raw arrays, P50/P95 and 5% verdict; the 15 ms individual
increment target has a different timer. Keep the prior single- and multi-
stage results distinct. The hash lookup and allocation counters are native
measurements on different clocks.

Use fixed audited collector 016934e2f625797a4b23b05b467de4cda50c2401 on a
fresh hosted runner with the same 3-second preflight and 1-second machine
gates, including the last sample. No formal retries after resource, timing
or correctness stops. Total budget 600s: provenance/selection/local
verification <=120s and hosted <=480s including setup, downloads, oracle,
process startup, cleanup and upload. Reserve the final 45s; the external
guard cancels by 435s of actual job time. Builds separately accounted.
