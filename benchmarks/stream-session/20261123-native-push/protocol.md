# Native single-row push registration

Registered before sampling on 2026-11-23. The baseline is main at
`4c3cc98e2ebaa8834b8f12931c33da64856daaae`. The candidate uses that
base with the four source and test files from #674 and #675. Its Git tree
is `487172223b97746bcbd8a97fc0159cc5fabf3971`. Compare the combined
change. Both workers use identical release settings, compiler, lockfile,
vendor contents, graph, data, reference and measurement code. Allow
12–25 minutes of build time per worker separately from sampling.

Use one public Rust `StreamSession` on the maintained single-symbol
`dual_sma_spread` graph, with a 5-row mean minus a 20-row mean. Both
means admit the first price. Warm the session with 4,096 rows in one push,
then submit 64 consecutive, individually timed one-row slices. Supply each
push with the explicit watermark equal to its last row's UTC microsecond
timestamp. Construct the envelope and watermark before timing. Start
immediately before `StreamSession::push` and stop on return of the owned
`SessionOutput`. Verify all five fields including schema, row order, null
positions and values after the clock, plus original input and empty final
flush. Use the maintained Python workload and independent oracle only to
prepare offline Arrow and graph inputs. No Python call occurs in the timed
worker.

Run two rounds of ten alternating AB/BA process pairs, one excluded
warmup process per side before each round. Pin every worker to the same
logical CPU with one Tokio thread. Start on a host without active builds.
Use the repository resource preflight to choose the quietest CPU and its
SMT siblings after three seconds, then sample every second. Admission
requires at most 20% CPU and sibling usage. Sampling requires sibling
usage below 50%. At all times the resource checks require average host CPU
below 50%, available memory at least 10 GiB, fewer than 256 major faults
and fewer than 1,024 swapped pages per observation interval. Stop on the
first resource, deadline, seal or correctness failure, retain partial
evidence and settle the worker. Allow 300 seconds total including process
startup, fixture generation, binary and fixture verification, resource
preflight, warmup, sampling, oracle and cleanup. The controller stops
acquisition at 285 seconds to allow for cleanup.

Use each process's median of 64 calls as the independent paired
observation. Apply the maintained two-round exact paired-median
interval and 5% regression rule. Report descriptive pooled P50 and P95
for 1,280 individual appends per side. The absolute goals are a pooled
P50 at most 10 µs and a pooled P95 at most 25 µs. This one workload
measures the combined changes at the public Rust entry point.
