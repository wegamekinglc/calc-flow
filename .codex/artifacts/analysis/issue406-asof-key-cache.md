# Issue 406: reuse ASOF right-key admission dictionary IDs

PR #407 merged as `acd10e0e`. The follow-up workflow fix is PR #408;
its complete benchmark is run `38158920926`, against pre-merge main
`588726c8`. Runtime performance targets remain unmet; that workflow fix
changes no runtime source.

A new diagnostic profile reused measured product `67df5dac`, native SHA-256
`573bd1e185edad150a967ae2c457f3549624f15b19c8cd79f19a9de643c8ce8c`.
Its runtime sources match merged main; the only crate diff is an integration
test. Binary and execution-controller identities remain distinct.
Four preregistered OFF cases each ran one warmup and five processing windows
at 4,000 Hz. Total acquisition, correctness, analysis and sealing took
26.477 / 180 seconds; build cost was zero. All owned processes settled
without forced kills. WSL2 host isolation remains unverified.

- `row-1/join_sealed/1000000/batch-64000/off/cpu-1`: 1904 samples; native stacks 1815; windows [384, 378, 412, 365, 365].
- `row-1/join_sealed/1000000/batch-64000/off/cpu-32`: 2001 samples; native stacks 1852; windows [387, 402, 406, 393, 413].
- `row-3/join_retained/4000000/batch-64000/off/cpu-1`: 16734 samples; native stacks 16011; windows [3500, 3498, 3325, 3203, 3208].
- `row-6/asof/1000000/batch-64000/off/cpu-1`: 5209 samples; native stacks 5000; windows [1013, 1036, 1050, 1067, 1043].

Join admission append has 14.65% / 12.04% inclusive samples at 1 / 32 CPUs,
and probe-key preparation 17.44% / 16.79%. ASOF admission preparation has
44.62%, identity construction 20.79%, and installation 21.96%.
These inclusive regions overlap and must not be added. The profile supplies
no speedup estimate or acceptance verdict. Evidence and frozen plan are under
`target/issue406-postmerge-profile/`; all raw profiles and symbol sidecars are retained.

## Selected increment

ASOF already builds distinct string encodings and per-row dictionary IDs.
Right admission still probes `InputKeys` for every accepted row and expands
hashes into a per-row vector. Cache the resolved admission-key index and hash
per distinct dictionary entry, scoped to one fully accepted record batch.
Count every admitted row in right capacities even when its key lookup is reused.

Keep the existing path for mixed-late, non-string, high-cardinality input and
optional cache admission refusal. Reserve cache storage before allocation,
never retain a cache across independently encoded record batches, and preserve
identity/duplicate validation, row order, resident-key ownership and cancellation.
Default memory remains unbounded; explicit caller budgets remain authoritative.

The focused RED must observe 4,096 interner probes for 64 distinct keys where
the optimization requires only 64. Add controls for independently ordered batch
dictionaries, mixed late rows, explicit budget refusal and wide key ownership.
Use the existing identity/lifecycle regressions; do not relax validation to
reach a timing goal. Measure only affected ASOF paths and necessary controls
with fixed pairs after implementation. Full eight-row and OFF/LOW/ON acceptance
continues separately; no target is marked achieved by this source increment.

## Focused validation

The initial test failed with exactly 4,096 interner calls against the required
64. The implementation passes all 63 admission tests when run serially, plus
the canonical string dictionary hashing/sliced-row test. The new narrow and
wide-key checks each observe 64 calls with unchanged canonical identities and
right capacities; all tracked pool reservations retire after use.

The first local 32-thread admission run exhausted the existing bounded process
infrastructure and aborted. This was diagnosed before the serial rerun; no
resource bound or test was weakened. Final specialist source review approves
the implementation. Library Clippy, scoped formatting and generated-contract
checks pass. Timings and final CI are still outstanding.
