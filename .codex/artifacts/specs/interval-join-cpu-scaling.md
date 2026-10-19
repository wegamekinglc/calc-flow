# Interval join CPU scaling

Continue the accepted streaming CPU changes at `8d4f04fc5427c7da6abfcecf12d3f4c500beb9e6`.
The interval join keeps inclusive event-time bounds, stable pair ordering, exact
multiplicity, logical byte accounting, checkpoint recovery, and cancellation.
No public configuration changes or fixed concurrency caps are introduced.

## Changes

- Retain optional prepaid window bounds from parallel count for reuse during fill.
  Cache identity must bind the immutable index, ingress, bounds, and partition;
  memory refusal falls back to the existing cursor walk. Partial work and
  cancellation release all credits, and optional cache memory yields to mandatory
  pair allocation. Deferred output remains compatible with moved probe inputs;
  a pool other than the default unbounded implementation releases its optional
  cache before mandatory state preparation.
- Check a repeated incoming row's constant output width once per contiguous run.
  Preserve the generic path for nullable, variable-width, and nested values.
- Check sparse replacement eligibility once per source owner in each newly
  appended retained chunk, preserving first-seen queue order and row accounting.

Every change starts with an observed focused failure. Verify directly affected
probe, output, funding, cancellation, state retention, and recovery tests, followed
by scoped formatting/lint and independent specialist review.

## Measurement

Reuse the sealed default release of baseline `8d4f04f`; build one clean-source
candidate default release separately (estimated ten minutes). The baseline and
candidate use the maintained interval oracle, exact cursor replay, accepted
watermarks, owning completion, resource guards, and paired comparison rules.
Explicitly coalesce both fixtures before slicing, use a fixed independently
planned nonnullable result schema, and allocate fresh checkpoint directories.
The ready timer ends after expected output and Arrow concatenation; subsequent
watermarks, EOF, oracle checks, and cleanup count toward the wall-clock budget.

Five configurations are fixed before acquisition:

| Rows per side | Rows per batch | CPUs | Purpose                               |
|---------------|----------------|------|---------------------------------------|
| 262144        | 262144         | 32   | Full 32-partition probe and fanout    |
| 262144        | 262144         | 8    | Scaling control                       |
| 262144        | 262144         | 1    | Serial control                        |
| 262144        | 65536          | 32   | Continuing batches and retained state |
| 4096          | 4096           | 32   | Small serial-path regression control  |

Each configuration has two rounds of ten alternating AB/BA pairs and one
excluded warmup per fresh arm/round worker: 200 formal samples and 20 warmups.
Initial baseline diagnostics and final measurement share 600 seconds including
setup, fixtures, oracles, startup, and cleanup; builds are separate. Initial
setup and diagnostic attempts consumed 36.582191670 seconds, leaving a fixed
563-second final allowance with ten seconds reserved for cleanup. Failed setup
attempts are retained and excluded from timing conclusions. Do not extend cases,
samples, or budget after an unfavorable or inconclusive outcome. A stopped run
reports completed cases and uncertainty without restarting measurement.

Plans, exact-source release manifests, harness hashes, failed setup evidence,
raw samples, CPU diagnostics, and final report live under the repository
`target/performance/interval-join-cpu-scaling/` directory.
