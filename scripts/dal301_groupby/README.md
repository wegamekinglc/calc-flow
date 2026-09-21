# DAL-301 two-case diagnostic

After incremental review, the assigned performance investigator can dispatch the
existing **Benchmarks** workflow once, using the reviewed PR branch:

```bash
gh workflow run benchmarks.yml --repo wegamekinglc/calc-flow \
  --ref perf/dal-301-asof-prepare-cost -f mode=dal301-groupby
```

`benchmarks.yml` is already registered on `main`. GitHub supports selecting another
branch with [`--ref`](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).
The branch supplies this mode; merging the diagnostic first is unnecessary. The
default `standard` mode and scheduled jobs retain their existing scope. Diagnostic
mode skips all three regular benchmark jobs. It does not alter PR/release gates.

This entry has not been dispatched during implementation. The subsequent diagnostic
assignment requires one terminal run result and its artifacts, including failed or
skipped preflight outcomes. Do not retry it automatically or dispatch a full matrix.

## Fixed scope and provenance

Only `calc-flow-stream/group_by`, N=10000 and N=100000, is accepted. For each size,
the predetermined comparisons are A/A, B/B, then A/B; each has two rounds of ten
alternating AB/BA pairs. The existing catalog, worker oracle, timing path and
paired-median classification retain the +5% threshold and `inconclusive` result.
There are 240 timed observations if all six comparisons complete. Profile samples
are separate and never enter these comparisons.

| Side | Product ref                                | Original artifact, run 35596885420 |
|------|--------------------------------------------|------------------------------------|
| A    | `a594ad697a57947237dd289f3a1bc2272ef099e8` | `10636839917`                      |
| B    | `b7e92cc58bc5000be0db5e6ec9a2decec052feb5` | `10637665772`                      |

`contract.py` pins their wheel/native hashes. The existing release loader verifies
the downloaded bytes; each worker also verifies its loaded native module. Missing
artifacts, different seals, unequal inputs or failed correctness stop collection.
The current PR head identifies the harness and never substitutes for product B.
The common harness hash, diagnostic file hashes, checkout SHA, dependencies,
thread settings, input Arrow IPC/hash and every request/response are retained.

The collector requires Python 3.13.15 and actual GitHub-hosted Linux with four
logical CPUs and affinity to all four. WSL, a larger machine restricted with
taskset, and an unqualified runner fail preflight. Catalog thread settings remain
unchanged. Resource traces run every 250 ms, separately for pairing and profiling;
an observed native build overlap invalidates the window.

## Separate profile builds

Two build jobs use the fixed A/B source refs before the sampling job starts. They
retain Rust 1.88.0, locked dependencies and release optimization. Only those isolated
checkouts receive a recorded maturin `strip=false` overlay plus release debug level
1, `strip=false` and `-C force-frame-pointers=yes`. Product defaults stay unchanged.
Manifests retain source hashes before/after the overlay, build argv/environment,
compiler, wheel/native hashes, ELF build-id and symbol inventory. Profile manifests
cannot identify the old stripped native modules as these new builds.

The sampling runner must provide `perf`, `readelf` and `nm`. Preflight records tool
versions, permissions and symbol resolution. `perf record` uses a fixed 49 Hz
`cpu-clock` event with frame-pointer unwinding and `--strict-freq`; it never changes
permissions, retries at another frequency or substitutes a high-overhead profiler.
See the [perf record options](https://man7.org/linux/man-pages/man1/perf-record.1.html).
Each side/size gets one 10-second recording window and at most 200 oracle-checked
workloads; a workload already started may finish after the recording window.

Keep raw `perf.data`, decoded and raw event dumps, command logs/exit codes, native
maps before/after, and resource traces. Missing resolved window frames, lost events
or throttling fail validation. Perf does not provide a direct sampling-lag metric
here; that gap is explicit, with raw timestamps retained. Low frequency is a
predeclared configuration, not proof of negligible overhead. These stacks include
preparation outside the paired timer and require causal interpretation; they are
not a replacement performance comparison.

## Evidence and failure interpretation

The always-uploaded artifacts are `dal301-profile-A-<run>-<attempt>`,
`dal301-profile-B-<run>-<attempt>` and `dal301-evidence-<run>-<attempt>`.
They retain step outcomes even if installation, artifact download or a build fails.
Build/collector shell wrappers retain full stdout/stderr and the original exit
code. Worker IPC, partial results and exit files survive collector failures.
Also download the run's raw GitHub job logs: action bootstrap/cache/download logs
belong to GitHub and are not reconstructed by the collector.

`outcome.json: completed` means the diagnostic collected valid evidence. It does
not mean no regression: inspect every comparison's classification in `results.json`.
A failed profile does not discard completed sealed-wheel results. A successful new
run cannot explain the old Rust installation conflict or close the old group_by
regressions by itself. Existing ASOF evidence keeps its original head; historical
release `inconclusive`, SQL sameHEAD RSS failure, ABBA informational status and the
unmerged PR #319 harness boundary all remain.

For a local selection-only check, without building or sampling:

```bash
python -m scripts.dal301_groupby plan --output target/dal301-plan
python -m unittest scripts.test_dal301_groupby scripts.test_rust_toolchain_diagnostics
```
