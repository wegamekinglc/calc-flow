# Bounded ready-prefix settlement in Rust

One case, `balanced_512`, measures the existing release
`stream_asof_perf` worker at main `85af0b53e0a38e19cb472aff18c1ccfd84048912`
and the independently reviewed ready-prefix revision based on that main.
Seal both rebuilt binaries and write the exact candidate SHA in
`arms.json` before acquisition. Use the same locked toolchain, dependency
cache, absolute checkout path, bench profile and benchmark-internals
feature. Compare full binary hashes and require an identical harness,
Cargo lock, validator, statistics, and resource monitor.

This case settles 512 pending left rows against 512 retained right rows
spread across 32 keys. No restore. The watermark makes every left row
ready. The callback exercises the ready-prefix selection, latest-per-key
lookup, Arrow materialization, collector, state commit and eviction.
This small balanced case includes the rest of settlement and does not
isolate the prefix calculation or represent sparse readiness. The before
and after checkpoint captures, fixture setup, full-row validation pass,
worker startup and native harness allocation reporting outside the
callback do not enter the reported latency. The worker's allocation
instrumentation remains enabled on both arms during the callback.

Use the maintained ASOF worker protocol. Each new process first runs a
separate full Arrow oracle on every one of 512 rows, then times one
settlement with the usual row, budget, watermark, retained-state and
eviction checks. Each round excludes a warmup per arm then alternates
ten adjacent AB/BA process pairs. Two rounds yield 20 formal pairs and
four excluded processes. Keep all raw ASOF_READY/ASOF_SAMPLE records,
source and dependency fingerprints and resource intervals. Report pooled
P50/P95 and exact paired intervals from repository paired_round. Apply
its maintained five-percent classification separately from the prospective
10% reduction in pooled native P50. The 20 process observations per arm
give only a descriptive P95. Do not claim a Python or full-runner effect.

Limit acquisition to 240 seconds including binary verification, resource
preflight, fixture work, subprocess startup, all checks, and cleanup.
The inner controller uses 228 seconds to leave room for shutdown; a
process-level timeout bounds the full run. Build time is recorded separately.
Pin all processes to the single qualifying CPU selected by the maintained
three-second resource preflight, with one Tokio worker. Use the unchanged
sparse_changelog_ci._resource gate at one-second intervals as workers run.
Coordinate with concurrent local builders and measurements. On any resource,
integrity or budget failure stop and retain partial evidence. No reruns.
