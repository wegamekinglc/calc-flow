# Issue #363: A1 and progress/retained performance evidence

Measured on 2026-10-06. **A1 has no confirmed material speedup in this matrix.**
The ordinary ASOF P50 observations are −3.66% at 1M rows/64k batches and −1.91%
at 100k rows/1,024 batches. Their paired intervals do not establish a repeatable
gain above 5%. Corrected overlapping input is still **inconclusive**, with a
+5.81% P50 observation; this is not evidence of zero regression.

Phase 0.2 proves the static lookup benchmark reaches 1M emitted rows with zero
left retention and zero left eviction. Its changed timing scope prevents a
native optimization claim. Retained interval and actual checkpoint/recovery
coverage pass their complete row oracles. Large interval cost is substantial:
about 45–54 seconds and roughly 8 GiB worker RSS for 10,998,080 output rows.

## Exact scope and provenance

- Baseline: foundation [PR #365](https://github.com/wegamekinglc/calc-flow/pull/365),
  `9b1535bcf4a9cdf394477da03c9718412a1092b1`.
- A1: [PR #367](https://github.com/wegamekinglc/calc-flow/pull/367), corrected
  `f81a9f87c6907782e90f531ba498d6a31f529673`. Its native hash differs from the
  baseline. The rejected `c1ab9b48188bf6b9ce489e7867ea670ac54a720d` wheel, source,
  drivers and regression measurements remain preserved.
- Progress: [PR #366](https://github.com/wegamekinglc/calc-flow/pull/366),
  `94383f5bb8ef9050113726fab39e8c379019af36`.
- Retained harness: [PR #368](https://github.com/wegamekinglc/calc-flow/pull/368),
  **exact measured revision** `159c10c319a7e7c5995a75a871222e4c9fb6bb09`.
  This pure harness change uses the sealed `94383f5` native wheel; the verifier
  confirms its build-source SHA-256 equals the progress release. The wheel is
  not relabeled as a release built from `159c10c`. Later helper extraction or
  other PR revisions were **not measured** here.

| Role         | Exact commit                             | Native SHA-256                                                   | Wheel SHA-256                                                    |
|--------------|------------------------------------------|------------------------------------------------------------------|------------------------------------------------------------------|
| Foundation   | 9b1535bcf4a9cdf394477da03c9718412a1092b1 | b5b671d656f4121505ec345ed3ed7f135492ef252d051513ef7a9d6ad6615c96 | 7940da9214c2d4ec32364e833055d4ace09fa052288a0acb72a8b49f588c4a04 |
| Rejected A1  | c1ab9b48188bf6b9ce489e7867ea670ac54a720d | 8cc0e25953580adcedfc5303590b8fc5b1e5f8e9b79c0b4c78e4f360b7a8e276 | 4a8363a7f6a218cd34e6770126301b241c5c1bd39960888b32c3d780c153fbf6 |
| Corrected A1 | f81a9f87c6907782e90f531ba498d6a31f529673 | 46c81dcbbe55e6b18051fcb72b6b827f09a79579f8cf1d85948d486c19879934 | 427fc3edc2aeb80eac795bda390c260cd19586adde5c2bf7a38b4178f2f34f03 |
| Progress     | 94383f5bb8ef9050113726fab39e8c379019af36 | 10ca280623470d67f266940ca3d5bca0a76086def8e8cfff1ec066e94d349122 | 1f2215ca897e6c0c886e55adb61cc4805fdb6fd8ca8cd230f5b8493a6e03f6e4 |

Each release is clean-source, `--release --locked --features pyo3/abi3-py313`.
Cargo lock SHA-256 is
`84789d29f2cb7342b0297169caf03c8f14227645656d34956ed9bc9aaa0c4840`.
Rust is 1.88.0; Maturin commands, logs, native/build-source hashes, source
identities and wheel seals are retained in `releases/`. Corrected A1 source
mtimes were refreshed before building in an independent release cache; this
prevents stale artifacts after switching source snapshots. Measured modules
were loaded only from the corresponding sealed wheel install.

## Environment and method

WSL2, Intel i9-13900HX, 32 logical CPUs, affinity 0–31, Linux
5.15.167.4-microsoft-standard-WSL2. Tokio/Polars each use 32 threads;
OMP/OpenBLAS/MKL use one. Python 3.13.9, NumPy 2.5.2, PyArrow 24.0.0,
DataFusion 54.0.0, Polars 1.44.2, TA-Lib 0.7.1 and JAX/JAXlib 0.11.1.
Windows host power mode is unavailable from the WSL guest. These are shared
machine observations, with background editing/app processes; they are not
portable bare-metal estimates.

Comparable cases use the maintained collector's **two rounds of ten alternating
AB/BA pairs**, with fresh workers per round and untimed full-oracle warmups.
The exact median intervals conservatively cover 97.85% for ten pairs. The
documented gate reports regression only when both lower bounds exceed +5%;
an `improved` verdict requires both upper bounds below −5%. Minima remain
diagnostic and do not replace this gate. This is the unified paired contract,
not the standalone same-ref/minimum diagnostic.

The coordinated team window had no Cargo builds or tests during corrected
overlap (11:41:42–11:41:50 UTC) or matrix (11:44:21–11:46:43 UTC).
User-owned external C++ compilation was observed at **11:52:56 UTC**, after
these comparisons completed. Retained results carry a loaded functional/cost
label, and the team quiet window was released before its small/checkpoint
proofs finished. External user processes were not stopped. Raw timestamps and
the observed process snapshot are retained in `noise/`.

`verify.py` independently rechecks wheel/native/source/lock seals, fresh PID
uniqueness, original IPC sample counts, full oracle results, preparation
dimensions, completion proofs, sample arrays and recomputed statistics.
`VERIFIED.json` records machine/dependency/workload SHA-256 fingerprints.
Comparable cases match; the v5→v6 static Join diagnostic deliberately differs
in workload identity. The older rejected A1 source fingerprints are rebuilt
from its immutable clean snapshot, rather than added to its original report.
There are 47 completed workers across retained accepted evidence and the
preserved rejected A1 run; interrupted pilot workers are separate.

## A1: common v5 harness, native revision comparison

Both revisions use exactly the same fixture, oracle, enqueue schedule and
`ready-enqueue-to-arrow/interleaved-inputs-v5` scope for ordinary cases.
Compilation, construction, startup, validation, EOF and shutdown are untimed.
Arrow output materialization is timed. Projection supplies a control floor.

| Case                | Rows/batch       | Base P50/P95 ms | Head P50/P95 ms | P50 delta | Round 1 median [CI] % | Round 2 median [CI] % | Verdict                 |
|---------------------|------------------|-----------------|-----------------|-----------|-----------------------|-----------------------|-------------------------|
| Overlap             | 64,000/64,000    | 37.316/39.780   | 39.483/43.728   | +5.81%    | +6.03 [+1.16, +14.35] | +6.12 [+2.23, +10.06] | inconclusive            |
| Contiguous pipeline | 64,000/64,000    | 28.617/31.555   | 26.631/29.917   | -6.94%    | -5.81 [-10.68, -1.41] | -5.37 [-12.26, +0.12] | no-confirmed-regression |
| ASOF                | 1,000,000/64,000 | 374.156/387.704 | 360.451/397.726 | -3.66%    | -3.39 [-4.61, +0.32]  | -2.26 [-4.77, -0.96]  | no-confirmed-regression |
| Projection          | 1,000,000/64,000 | 6.613/7.466     | 6.285/7.860     | -4.95%    | -5.86 [-17.47, +6.00] | -1.15 [-12.52, +5.53] | inconclusive            |
| ASOF                | 100,000/1,024    | 84.045/92.925   | 82.441/86.210   | -1.91%    | -2.15 [-4.19, +0.29]  | +0.16 [-8.40, +1.90]  | no-confirmed-regression |
| Projection          | 100,000/1,024    | 30.723/34.435   | 31.361/37.542   | +2.08%    | +2.74 [-3.91, +4.15]  | +2.11 [-5.89, +18.44] | inconclusive            |

The two extra 64k diagnostic fixtures queue two 32k left chunks before their
watermark. Contiguous chunks exercise long merged runs; even/odd row chunks
interleave heads and exercise one-row runs. Both use the same right fixture,
state limits and `pipelined-*-v1` schedule on baseline and candidate. Besides
the full payload oracle, output sequence must equal the canonical sequence
without sorting. All warmup and measured oracles passed. These diagnostics
currently live in the evidence driver, not the maintained scheduled catalog.

The rejected c1 overlap run was 37.807→
50.840 ms (+34.47%). Its two CI ranges
were [+22.42, +44.90]%
and [+36.39, +45.62]%,
which confirmed regression. The corrected run used the same fixture and was
not selected from faster observations. Keep the corrected positive,
inconclusive interval when assessing residual risk.

At 100k/1,024, the raw ASOF/projection P50 ratio is 2.74× baseline
and 2.63× candidate. **The 2× harness-overhead/operator-time target
is unverified**: these are total ASOF and projection times; matching native
operator time is not isolated. Projection subtraction is not real overhead.

## Phase 0.2: progress and settled static lookup

Common v6 ASOF/projection controls use both wheels under the same scope:

| Case       | Rows/batch       | Base P50/P95 ms | Head P50/P95 ms | P50 delta | Round 1 median [CI] % | Round 2 median [CI] % | Verdict                 |
|------------|------------------|-----------------|-----------------|-----------|-----------------------|-----------------------|-------------------------|
| asof_join  | 1,000,000/64,000 | 371.493/389.588 | 357.830/370.565 | -3.68%    | -2.71 [-8.05, -1.67]  | -4.11 [-7.80, -2.03]  | no-confirmed-regression |
| projection | 1,000,000/64,000 | 6.245/6.808     | 6.440/8.846     | +3.13%    | +3.95 [-4.46, +16.87] | +1.35 [-4.71, +18.34] | inconclusive            |

Static Join is **new coverage** under settled-dimension v6: candidate P50/P95
655.113/675.864 ms,
20 measured samples, 1M exact output rows. Every completed sample records
`emitted_match_rows=1_000_000`, left `retained_rows=0`, `retained_bytes=0` and
`evicted_rows=0`; no state/match-limit failure. Right dimension progress is
acknowledged before the timed left feed. Status calls are outside the timer.

The immediate post-timer status reports 960k emitted rows while the sink
already owns all 1M rows. That snapshot precedes the last metric flush;
the separately captured completed status proves the final 1M total.
Both are preserved. On completion the right dimension is cleared at EOF;
its 64 evictions are separate from the zero left-eviction invariant.

The v5→v6 Join diagnostic observes
689.054→640.639 ms
(-7.03%). Paired changes are
-7.15 [-12.49, -3.48]% and
-6.02 [-10.70, -4.09]%.
**This is changed-scope evidence, not a native kernel speedup.** Baseline v5
retains/evicts the first 64k left batch before settling; its final left
eviction total is 64,000. Its status lacks the newer progress fields, so no
fabricated compatible progress-field comparison is used.

## Retained regimes: loaded functional and cost diagnostics

| Retained case            | Rows/batch       | Measured samples        | P50/P95            | Evidence                                                  |
|--------------------------|------------------|-------------------------|--------------------|-----------------------------------------------------------|
| Interval                 | 1,000,000/64,000 | 2; one completed worker | 51.520/53.604 s    | 10,998,080 exact output rows; loaded cost diagnostic      |
| Interval                 | 100,000/1,024    | 20; two fresh workers   | 614.540/628.501 ms | 1,098,080 exact output rows; candidate-only new coverage  |
| ASOF checkpoint/recovery | 100,000/64,000   | 20; two fresh workers   | 215.990/248.590 ms | 64,000-row nonterminal prefix; epoch 2; recovery verified |

The 1M interval pilot passed the full 10,998,080-row oracle, with 46.914s warmup
and 44.997s first measured cost. Its initial 20-sample plan was bounded after
observing ~45s/sample. The reduced attempt completed two measured samples in
one worker, 49.204s and 53.836s; the second worker was interrupted after
external load appeared. All attempts remain preserved; the pilot is excluded
from the aggregate above, and **two-round/20-sample evidence is insufficient**.
There is no paired or idle performance verdict for this case.

The completed 1M samples report exactly 10,998,080 emitted rows, left/right
retained 320 rows each, evicted 999,680 each, and zero limit failures.
Retained native state is 41,280 bytes per side. In contrast, worker RSS after
samples is about 7.75–7.80 GiB, and process VmHWM is 8.89 GiB. Process memory
includes fixtures, complete output and oracle work; it is not isolated kernel
memory. A planned 20-sample run costs about 16 minutes plus startup/oracles
and needs materially more memory than the static Join case.

Every small interval sample verifies all 1,098,080 output rows, with zero
maximum absolute error. Every checkpoint sample verifies all 100k rows,
nonterminal prefix=64k, published epoch=2 and `recovery=verified` through exact
cursor replay. The 215.990ms median includes a deliberate 100ms pacing delay,
checkpoint acknowledgement, cancellation, restart/recovery and Arrow output;
it is **not steady-state throughput**. No native optimization delta is claimed.

These checkpoint observations satisfy this harness's durable recovery proof,
not the entire specialized lifecycle evidence gate: phase durations and
quantiles, checkpoint bytes and complete diagnostic RSS samples were not
collected through `verify_stream_lifecycle_evidence.py`. That gate, the cold
matrix and full scheduled matrix remain **unverified** here.

## Coverage and remaining limits

1. Add the overlapping and contiguous pending-left fixtures to maintained
   engine coverage. Ordinary per-batch watermarks do not fully exercise merged
   multi-chunk traversal; preserve short-run cases alongside large contiguous
   runs when evaluating successor/gallop changes.
2. Add a matched native ASOF/projection operator-time measurement for the 2×
   overhead target. Current total-time ratios cannot establish it.
3. Retained interval coverage now exposes its output/memory cost, and should
   guide Join materialization work. Compare future native changes under this
   same retained harness and dimensions before assigning a runtime gain.

Measurements are tied to the exact revisions above and do not certify later
changes. They do not substitute for CI correctness, coverage or specialist
review. At handoff, PR #367 Linux lib tests have a post-drop `pool.reserved()`
assertion failure (39,552 versus 0) under diagnosis; Windows passing does not
resolve it. This performance evidence is not a merge-ready declaration.

## Raw artifacts and reproduction

Local output root: `/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-asof-runs/target/issue363-asof-runs-perf`.

- `overlap/results.json`, `matrix/results.json`: original paired arrays,
  P50/P95, exact per-round intervals, environments and complete samples.
- `coverage/results.json`: original small/checkpoint sample and lifecycle proof.
- `large-cost.json`, `coverage-loaded-large-attempt/`: two complete large
  samples and stopped second-worker attempt, explicitly loaded/insufficient.
- `initial-c1-a1/`, `coverage-cost-pilot/`: retained rejected A1 and cost pilot.
- `releases/`, `sources/`, `installed/`: source/native/wheel seals and owned
  immutable source checkouts/loaded sites. Wheels remain local release assets.
- `VERIFIED.json`, `verify.py`, `SYSTEM.json`, `noise/`: independent checks,
  normalized SHA-256 identities and observed host load.
- `measure.py`, `worker.py`, `coverage.py`: exact drivers, preserved hashes and
  untimed status proofs. `write_report.py` derives this report from raw JSON.

Use the lock-matching `target/benchmark-venv/bin/python` from the investigation
worktree. Run `measure.py --only overlap` or `--only matrix` only in a new
destination under a separately coordinated quiet window; existing destinations
are preserved evidence. `coverage.py` uses `sources/retained` on `PYTHONPATH`.
This report was written after all owned benchmark processes exited; no
measurement was repeated to replace unchanged passing observations.

## Repository-verifiable raw observations

The following original timing arrays and identity/archival metadata are embedded
for PR review, so recomputing the tables does not depend on ignored local
`target/` paths. Every timing value is in seconds; list positions retain the
paired observation order, and the two lists retain the two independent rounds.
The one-round loaded large interval has no paired classification. The archive
contains original IPC/oracle/completion evidence beyond this timing appendix.

```json
{
  "provenance": {
    "archive": {
      "archive": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-asof-runs/target/issue363-asof-runs-perf/issue363-asof-runs-evidence.tar.gz",
      "archive_sha256": "bce39f8fbec8eabd6976b34c61dee79e54a60a4822ecb1148693b8ba06f67aea",
      "archive_bytes": 670349,
      "files": 1484,
      "report_sha256": "bc10802460fbd79913956f3a152f4677af20ee670b2cf5885f42615f32ea6ece",
      "native_wheels_in_archive": false,
      "wheel_assets": "Local releases/; exact seals in archive and independently checked",
      "full_source_snapshots": "Local sources/; archived harness files match exact git identities"
    },
    "archive_note": "The raw archive was sealed before adding this appendix. Its earlier Markdown snapshot excludes this appendix; original raw JSON/arrays are unchanged. report_sha256 in ARCHIVE.json identifies that earlier snapshot, not this final PR document.",
    "machine": {
      "collected_utc": "2026-10-06T12:03:13.264957+00:00",
      "node": "chengli-i9",
      "platform": "Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.43",
      "cpu": "{\n   \"lscpu\": [\n      {\n         \"field\": \"Architecture:\",\n         \"data\": \"x86_64\"\n      },{\n         \"field\": \"CPU op-mode(s):\",\n         \"data\": \"32-bit, 64-bit\"\n      },{\n         \"field\": \"Address sizes:\",\n         \"data\": \"39 bits physical, 48 bits virtual\"\n      },{\n         \"field\": \"Byte Order:\",\n         \"data\": \"Little Endian\"\n      },{\n         \"field\": \"CPU(s):\",\n         \"data\": \"32\"\n      },{\n         \"field\": \"On-line CPU(s) list:\",\n         \"data\": \"0-31\"\n      },{\n         \"field\": \"Vendor ID:\",\n         \"data\": \"GenuineIntel\"\n      },{\n         \"field\": \"Model name:\",\n         \"data\": \"13th Gen Intel(R) Core(TM) i9-13900HX\"\n      },{\n         \"field\": \"CPU family:\",\n         \"data\": \"6\"\n      },{\n         \"field\": \"Model:\",\n         \"data\": \"183\"\n      },{\n         \"field\": \"Thread(s) per core:\",\n         \"data\": \"2\"\n      },{\n         \"field\": \"Core(s) per socket:\",\n         \"data\": \"16\"\n      },{\n         \"field\": \"Socket(s):\",\n         \"data\": \"1\"\n      },{\n         \"field\": \"Stepping:\",\n         \"data\": \"1\"\n      },{\n         \"field\": \"BogoMIPS:\",\n         \"data\": \"4838.39\"\n      },{\n         \"field\": \"Flags:\",\n         \"data\": \"fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology tsc_reliable nonstop_tsc cpuid pni pclmulqdq vmx ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves avx_vnni umip waitpkg gfni vaes vpclmulqdq rdpid movdiri movdir64b fsrm md_clear serialize flush_l1d arch_capabilities\"\n      },{\n         \"field\": \"Virtualization:\",\n         \"data\": \"VT-x\"\n      },{\n         \"field\": \"Hypervisor vendor:\",\n         \"data\": \"Microsoft\"\n      },{\n         \"field\": \"Virtualization type:\",\n         \"data\": \"full\"\n      },{\n         \"field\": \"L1d cache:\",\n         \"data\": \"768 KiB (16 instances)\"\n      },{\n         \"field\": \"L1i cache:\",\n         \"data\": \"512 KiB (16 instances)\"\n      },{\n         \"field\": \"L2 cache:\",\n         \"data\": \"32 MiB (16 instances)\"\n      },{\n         \"field\": \"L3 cache:\",\n         \"data\": \"36 MiB (1 instance)\"\n      },{\n         \"field\": \"Vulnerability Gather data sampling:\",\n         \"data\": \"Not affected\"\n      },{\n         \"field\": \"Vulnerability Itlb multihit:\",\n         \"data\": \"Not affected\"\n      },{\n         \"field\": \"Vulnerability L1tf:\",\n         \"data\": \"Not affected\"\n      },{\n         \"field\": \"Vulnerability Mds:\",\n         \"data\": \"Not affected\"\n      },{\n         \"field\": \"Vulnerability Meltdown:\",\n         \"data\": \"Not affected\"\n      },{\n         \"field\": \"Vulnerability Mmio stale data:\",\n         \"data\": \"Not affected\"\n      },{\n         \"field\": \"Vulnerability Reg file data sampling:\",\n         \"data\": \"Vulnerable: No microcode\"\n      },{\n         \"field\": \"Vulnerability Retbleed:\",\n         \"data\": \"Mitigation; Enhanced IBRS\"\n      },{\n         \"field\": \"Vulnerability Spec rstack overflow:\",\n         \"data\": \"Not affected\"\n      },{\n         \"field\": \"Vulnerability Spec store bypass:\",\n         \"data\": \"Mitigation; Speculative Store Bypass disabled via prctl and seccomp\"\n      },{\n         \"field\": \"Vulnerability Spectre v1:\",\n         \"data\": \"Mitigation; usercopy/swapgs barriers and __user pointer sanitization\"\n      },{\n         \"field\": \"Vulnerability Spectre v2:\",\n         \"data\": \"Mitigation; Enhanced / Automatic IBRS; IBPB conditional; RSB filling; PBRSB-eIBRS SW sequence; BHI BHI_DIS_S\"\n      },{\n         \"field\": \"Vulnerability Srbds:\",\n         \"data\": \"Not affected\"\n      },{\n         \"field\": \"Vulnerability Tsx async abort:\",\n         \"data\": \"Not affected\"\n      }\n   ]\n}\n",
      "power_mode": "Windows host power mode unavailable from this WSL2 guest; not independently verified",
      "threads": {
        "tokio": 32,
        "polars": 32,
        "OMP": 1,
        "OPENBLAS": 1,
        "MKL": 1
      },
      "machine_sha256": "5de5f8abb0849b1ca9596527c2ac2cb3aa32415841a14dc2fca807ea8807f5c4"
    },
    "dependency_identity": {
      "python": "3.13.9",
      "numpy": "2.5.2",
      "pyarrow": "24.0.0",
      "platform": "Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.43",
      "logical_cpus": 32,
      "cpu_affinity": [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31
      ],
      "tokio_worker_threads": "32",
      "packages": {
        "datafusion": "54.0.0",
        "polars": "1.44.2",
        "TA-Lib": "0.7.1",
        "jax": "0.11.1",
        "jaxlib": "0.11.1"
      },
      "polars_threads": 32,
      "thread_environment": {
        "POLARS_MAX_THREADS": "32",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_FLAGS": null
      }
    },
    "fingerprints": [
      {
        "report": "initial-c1-a1/overlap",
        "id": "engines/64000/calc-flow-stream/asof_join/batch-64000/overlap",
        "compatible": true,
        "source_fingerprints": "reconstructed-from-sealed-clean-snapshot",
        "unique_fingerprints": [
          {
            "dependency_sha256": "6380048f9ec667274e816d52d0f30a9a42df481b9c16c3783349a2a5ce8e3c69",
            "machine_sha256": "57212897187071d36033d45de0d65e356b063167f66fa9566b777ff709612f41",
            "workload_sha256": "cecd6c48afbb3f7b80550b1eacbb1eff6fa9a0582b4d88074a580e464f7cb9ae"
          }
        ]
      },
      {
        "report": "overlap",
        "id": "engines/64000/calc-flow-stream/asof_join/batch-64000/overlap",
        "compatible": true,
        "source_fingerprints": "captured-in-original-report",
        "unique_fingerprints": [
          {
            "dependency_sha256": "6380048f9ec667274e816d52d0f30a9a42df481b9c16c3783349a2a5ce8e3c69",
            "machine_sha256": "57212897187071d36033d45de0d65e356b063167f66fa9566b777ff709612f41",
            "workload_sha256": "8dc11821cbb2570eef68cd6d0f2269f31502e996bdf51094a39f291361659ea3"
          }
        ]
      },
      {
        "report": "matrix",
        "id": "engines/64000/calc-flow-stream/asof_join/batch-64000/pipelined",
        "compatible": true,
        "source_fingerprints": "captured-in-original-report",
        "unique_fingerprints": [
          {
            "dependency_sha256": "6380048f9ec667274e816d52d0f30a9a42df481b9c16c3783349a2a5ce8e3c69",
            "machine_sha256": "57212897187071d36033d45de0d65e356b063167f66fa9566b777ff709612f41",
            "workload_sha256": "583d576f273a516a2cdce83992a08ca59b33dba67ceb216a4a664bbe3f2f473e"
          }
        ]
      },
      {
        "report": "matrix",
        "id": "engines/1000000/calc-flow-stream/asof_join/batch-64000/standard",
        "compatible": true,
        "source_fingerprints": "captured-in-original-report",
        "unique_fingerprints": [
          {
            "dependency_sha256": "6380048f9ec667274e816d52d0f30a9a42df481b9c16c3783349a2a5ce8e3c69",
            "machine_sha256": "57212897187071d36033d45de0d65e356b063167f66fa9566b777ff709612f41",
            "workload_sha256": "1e4bdccae9243b145e8c316aa734e184eeec62009710af546545500c18c44712"
          }
        ]
      },
      {
        "report": "matrix",
        "id": "engines/1000000/calc-flow-stream/projection/batch-64000/standard",
        "compatible": true,
        "source_fingerprints": "captured-in-original-report",
        "unique_fingerprints": [
          {
            "dependency_sha256": "6380048f9ec667274e816d52d0f30a9a42df481b9c16c3783349a2a5ce8e3c69",
            "machine_sha256": "57212897187071d36033d45de0d65e356b063167f66fa9566b777ff709612f41",
            "workload_sha256": "99a5602dfaedd65d91a031f6f23a2c5d7cea1abf93f69ff5078b27465bf08665"
          }
        ]
      },
      {
        "report": "matrix",
        "id": "engines/100000/calc-flow-stream/asof_join/batch-1024/standard",
        "compatible": true,
        "source_fingerprints": "captured-in-original-report",
        "unique_fingerprints": [
          {
            "dependency_sha256": "6380048f9ec667274e816d52d0f30a9a42df481b9c16c3783349a2a5ce8e3c69",
            "machine_sha256": "57212897187071d36033d45de0d65e356b063167f66fa9566b777ff709612f41",
            "workload_sha256": "b62e4ea2fb481ea9d927098fa3580f64d1fbb8e53a85ff557aa3a30a3b58caf1"
          }
        ]
      },
      {
        "report": "matrix",
        "id": "engines/100000/calc-flow-stream/projection/batch-1024/standard",
        "compatible": true,
        "source_fingerprints": "captured-in-original-report",
        "unique_fingerprints": [
          {
            "dependency_sha256": "6380048f9ec667274e816d52d0f30a9a42df481b9c16c3783349a2a5ce8e3c69",
            "machine_sha256": "57212897187071d36033d45de0d65e356b063167f66fa9566b777ff709612f41",
            "workload_sha256": "26ccc7fe233be42291c82b069bcec0a83a77f88b0745c355d9ff724135a8c3af"
          }
        ]
      },
      {
        "report": "matrix",
        "id": "engines/1000000/calc-flow-stream/asof_join/batch-64000/standard",
        "compatible": true,
        "source_fingerprints": "captured-in-original-report",
        "unique_fingerprints": [
          {
            "dependency_sha256": "6380048f9ec667274e816d52d0f30a9a42df481b9c16c3783349a2a5ce8e3c69",
            "machine_sha256": "57212897187071d36033d45de0d65e356b063167f66fa9566b777ff709612f41",
            "workload_sha256": "323f491f17d43445c1a96872eb34e02a44a9b74167c245629bb8c14885209446"
          }
        ]
      },
      {
        "report": "matrix",
        "id": "engines/1000000/calc-flow-stream/projection/batch-64000/standard",
        "compatible": true,
        "source_fingerprints": "captured-in-original-report",
        "unique_fingerprints": [
          {
            "dependency_sha256": "6380048f9ec667274e816d52d0f30a9a42df481b9c16c3783349a2a5ce8e3c69",
            "machine_sha256": "57212897187071d36033d45de0d65e356b063167f66fa9566b777ff709612f41",
            "workload_sha256": "1004b6280e5fcd933dc57016b909b146966261fb84757c9d0909be07137c5cbe"
          }
        ]
      },
      {
        "report": "matrix",
        "id": "engines/1000000/calc-flow-stream/join/batch-64000/standard",
        "compatible": true,
        "source_fingerprints": "captured-in-original-report",
        "unique_fingerprints": [
          {
            "dependency_sha256": "6380048f9ec667274e816d52d0f30a9a42df481b9c16c3783349a2a5ce8e3c69",
            "machine_sha256": "57212897187071d36033d45de0d65e356b063167f66fa9566b777ff709612f41",
            "workload_sha256": "3202f8fde94d62db40ffe008181b8ac1d57478ef4ec7150e23c9c8de4ad02c78"
          }
        ]
      },
      {
        "report": "matrix",
        "id": "engines/1000000/calc-flow-stream/join/batch-64000/standard",
        "compatible": false,
        "source_fingerprints": "captured-in-original-report",
        "unique_fingerprints": [
          {
            "dependency_sha256": "6380048f9ec667274e816d52d0f30a9a42df481b9c16c3783349a2a5ce8e3c69",
            "machine_sha256": "57212897187071d36033d45de0d65e356b063167f66fa9566b777ff709612f41",
            "workload_sha256": "536645eb2293528a1513605048a88e713bbea86764be0fabea6e15798cc7e57d"
          },
          {
            "dependency_sha256": "6380048f9ec667274e816d52d0f30a9a42df481b9c16c3783349a2a5ce8e3c69",
            "machine_sha256": "57212897187071d36033d45de0d65e356b063167f66fa9566b777ff709612f41",
            "workload_sha256": "60b919044c0cb28dddd823a5457012c2227d9153aa11f4cc6921392b422a8f31"
          }
        ]
      },
      {
        "report": "coverage",
        "id": "engines/100000/calc-flow-stream/interval_join/batch-1024",
        "compatible": false,
        "source_fingerprints": null,
        "unique_fingerprints": []
      },
      {
        "report": "coverage",
        "id": "engines/100000/calc-flow-stream/asof_join/batch-64000/checkpoint-100ms-duration-recovery",
        "compatible": false,
        "source_fingerprints": null,
        "unique_fingerprints": []
      }
    ],
    "raw_sample_count_in_appendix": 462,
    "completed_workers_verified": 47
  },
  "cases": [
    {
      "id": "engines/64000/calc-flow-stream/asof_join/batch-64000/overlap",
      "scope": "ready-enqueue-to-arrow/pipelined-overlap-v1",
      "rows": 64000,
      "batch_rows": 64000,
      "baseline": [
        [
          0.037053671,
          0.039230863,
          0.0377821,
          0.038797704,
          0.042350195,
          0.039188995,
          0.039901266,
          0.03664757,
          0.038200605,
          0.037928681
        ],
        [
          0.035033807,
          0.037831211,
          0.036417729,
          0.034216095,
          0.036431568,
          0.036501232,
          0.037843872,
          0.03877702,
          0.035166439,
          0.036781263
        ]
      ],
      "candidate": [
        [
          0.050843085,
          0.048845,
          0.052844748,
          0.047769556,
          0.051844102,
          0.05678515,
          0.0484857,
          0.052496592,
          0.05083763,
          0.056832097
        ],
        [
          0.053461343,
          0.0525037,
          0.049945126,
          0.049826444,
          0.051334035,
          0.050005451,
          0.051858675,
          0.049330717,
          0.049684613,
          0.050165731
        ]
      ],
      "experiment": "a1-overlap",
      "timing_unit": "seconds",
      "comparison": "interleaved",
      "source_report": "initial-c1-a1/overlap/results.json",
      "result": {
        "head_p50": 0.0508403575,
        "head_p95": 0.05678749735,
        "head_min": 0.047769556,
        "head_max": 0.056832097,
        "rows_per_second": 1258842.4461806745,
        "samples": 20,
        "base_p50": 0.037806655499999994,
        "change_percent": 34.47462312554996,
        "round_changes": [
          35.14769869963749,
          37.96458579392737
        ],
        "round_min_changes": [
          30.34849513896829,
          44.17401225943522
        ],
        "round_intervals": [
          {
            "median": 35.14769869963749,
            "low": 22.417622870449595,
            "high": 44.90075593926306,
            "coverage": 0.978515625
          },
          {
            "median": 37.96458579392737,
            "low": 36.38936487852522,
            "high": 45.622824580069675,
            "coverage": 0.978515625
          }
        ],
        "verdict": "regression"
      },
      "driver_sha256": "c60e372daac2a97f535363a38b889256befdaef1a9933658d3427386d64d346c",
      "worker_sha256": "c2d4f7b7f79dbe2a0fad6c4c881cde2739f8982a76285e8716a3a20dc0570e28",
      "correctness": true
    },
    {
      "id": "engines/64000/calc-flow-stream/asof_join/batch-64000/overlap",
      "scope": "ready-enqueue-to-arrow/pipelined-overlap-v1",
      "rows": 64000,
      "batch_rows": 64000,
      "baseline": [
        [
          0.037988889,
          0.039718885,
          0.036753228,
          0.037323245,
          0.036055262,
          0.039329979,
          0.039114755,
          0.034942535,
          0.036089781,
          0.039714533
        ],
        [
          0.03572028,
          0.038843332,
          0.035614643,
          0.036316363,
          0.037308408,
          0.038268564,
          0.037858405,
          0.036807814,
          0.036141882,
          0.040941732
        ]
      ],
      "candidate": [
        [
          0.038430838,
          0.043682062,
          0.039492797,
          0.039014979,
          0.04460707,
          0.041142879,
          0.040101149,
          0.039957318,
          0.039339099,
          0.039592147
        ],
        [
          0.039901558,
          0.040332597,
          0.039197976,
          0.039269879,
          0.041001201,
          0.039123184,
          0.03947282,
          0.038114797,
          0.039022752,
          0.037392262
        ]
      ],
      "experiment": "a1-overlap",
      "timing_unit": "seconds",
      "comparison": "interleaved",
      "source_report": "overlap/results.json",
      "result": {
        "head_p50": 0.0394828085,
        "head_p95": 0.043728312400000004,
        "head_min": 0.037392262,
        "head_max": 0.04460707,
        "rows_per_second": 1620958.6509024554,
        "samples": 20,
        "base_p50": 0.037315826499999996,
        "change_percent": 5.807139230856917,
        "round_changes": [
          6.031707841316159,
          6.117675950889955
        ],
        "round_min_changes": [
          9.982970611605623,
          4.991258792064834
        ],
        "round_intervals": [
          {
            "median": 6.031707841316159,
            "low": 1.1633638456760353,
            "high": 14.35151456527124,
            "coverage": 0.978515625
          },
          {
            "median": 6.117675950889955,
            "low": 2.2332167990416485,
            "high": 10.06140367601045,
            "coverage": 0.978515625
          }
        ],
        "verdict": "inconclusive"
      },
      "driver_sha256": "15232f35259003d0988badb03a6aeb65340a66ea8a4738b8c00cb9dea6c085a8",
      "worker_sha256": "9df94f7ec148efc46c18b30f010ce9d2ed66bb24f94c00e3677c58da8528ccdd",
      "correctness": true
    },
    {
      "id": "engines/64000/calc-flow-stream/asof_join/batch-64000/pipelined",
      "scope": "ready-enqueue-to-arrow/pipelined-contiguous-v1",
      "rows": 64000,
      "batch_rows": 64000,
      "baseline": [
        [
          0.031452829,
          0.030663054,
          0.027481631,
          0.028529102,
          0.025865037,
          0.029293922,
          0.030336547,
          0.033502829,
          0.03028659,
          0.02924446
        ],
        [
          0.028705585,
          0.030146076,
          0.027787504,
          0.026649711,
          0.027559883,
          0.028017017,
          0.02690835,
          0.028947024,
          0.027008688,
          0.027972986
        ]
      ],
      "candidate": [
        [
          0.028093377,
          0.029827701,
          0.026340056,
          0.025715382,
          0.026578556,
          0.028418442,
          0.027123972,
          0.031001988,
          0.029859623,
          0.024682758
        ],
        [
          0.025187457,
          0.025869069,
          0.026190906,
          0.026682529,
          0.027420143,
          0.026288112,
          0.027445843,
          0.026407355,
          0.025659246,
          0.026894774
        ]
      ],
      "experiment": "a1-pipelined",
      "timing_unit": "seconds",
      "comparison": "interleaved",
      "source_report": "matrix/results.json",
      "result": {
        "head_p50": 0.0266305425,
        "head_p95": 0.02991674125,
        "head_min": 0.024682758,
        "head_max": 0.031001988,
        "rows_per_second": 2403255.585198837,
        "samples": 20,
        "base_p50": 0.0286173435,
        "change_percent": -6.942646510847517,
        "round_changes": [
          -5.8092615795305775,
          -5.371033295026123
        ],
        "round_min_changes": [
          -4.570954219010015,
          -5.486941303040771
        ],
        "round_intervals": [
          {
            "median": -5.8092615795305775,
            "low": -10.680921579422954,
            "high": -1.4097559348873578,
            "coverage": 0.978515625
          },
          {
            "median": -5.371033295026123,
            "low": -12.255900724545409,
            "high": 0.1231458007180608,
            "coverage": 0.978515625
          }
        ],
        "verdict": "no-confirmed-regression"
      },
      "driver_sha256": "15232f35259003d0988badb03a6aeb65340a66ea8a4738b8c00cb9dea6c085a8",
      "worker_sha256": "9df94f7ec148efc46c18b30f010ce9d2ed66bb24f94c00e3677c58da8528ccdd",
      "correctness": true
    },
    {
      "id": "engines/1000000/calc-flow-stream/asof_join/batch-64000/standard",
      "scope": "ready-enqueue-to-arrow/interleaved-inputs-v5",
      "rows": 1000000,
      "batch_rows": 64000,
      "baseline": [
        [
          0.374396313,
          0.375038501,
          0.384664601,
          0.370856117,
          0.374000325,
          0.371734746,
          0.431594562,
          0.379274662,
          0.366984421,
          0.367196881
        ],
        [
          0.366879323,
          0.380735907,
          0.363032924,
          0.364502532,
          0.368059864,
          0.375611488,
          0.369519437,
          0.385394201,
          0.374311569,
          0.377899609
        ]
      ],
      "candidate": [
        [
          0.359543081,
          0.363638555,
          0.367950395,
          0.35377403,
          0.360034621,
          0.372930144,
          0.396852139,
          0.414320586,
          0.357391046,
          0.358720974
        ],
        [
          0.363347417,
          0.359012776,
          0.358332323,
          0.354230739,
          0.360607259,
          0.37716732,
          0.360294348,
          0.372593018,
          0.356461523,
          0.37125078
        ]
      ],
      "experiment": "a1-standard",
      "timing_unit": "seconds",
      "comparison": "interleaved",
      "source_report": "matrix/results.json",
      "result": {
        "head_p50": 0.3604508035,
        "head_p95": 0.39772556135,
        "head_min": 0.35377403,
        "head_max": 0.414320586,
        "rows_per_second": 2774303.706053467,
        "samples": 20,
        "base_p50": 0.374155947,
        "change_percent": -3.6629495294377756,
        "round_changes": [
          -3.3869078641910355,
          -2.2606724129052846
        ],
        "round_min_changes": [
          -3.5997143867859127,
          -2.4246244398483108
        ],
        "round_intervals": [
          {
            "median": -3.3869078641910355,
            "low": -4.606122487120789,
            "high": 0.3215728453858313,
            "coverage": 0.978515625
          },
          {
            "median": -2.2606724129052846,
            "low": -4.768766845141248,
            "high": -0.9626887585594379,
            "coverage": 0.978515625
          }
        ],
        "verdict": "no-confirmed-regression"
      },
      "driver_sha256": "15232f35259003d0988badb03a6aeb65340a66ea8a4738b8c00cb9dea6c085a8",
      "worker_sha256": "9df94f7ec148efc46c18b30f010ce9d2ed66bb24f94c00e3677c58da8528ccdd",
      "correctness": true
    },
    {
      "id": "engines/1000000/calc-flow-stream/projection/batch-64000/standard",
      "scope": "ready-enqueue-to-arrow/interleaved-inputs-v5",
      "rows": 1000000,
      "batch_rows": 64000,
      "baseline": [
        [
          0.007292794,
          0.006014329,
          0.007108932,
          0.007939912,
          0.005910609,
          0.007329873,
          0.006219428,
          0.006272831,
          0.006006111,
          0.006539498
        ],
        [
          0.006059368,
          0.006631794,
          0.0066508,
          0.00662272,
          0.006018397,
          0.007259457,
          0.007441535,
          0.006603364,
          0.00612234,
          0.006812036
        ]
      ],
      "candidate": [
        [
          0.006222342,
          0.006374985,
          0.006789983,
          0.006282558,
          0.005888893,
          0.006049267,
          0.005769371,
          0.005997419,
          0.006674492,
          0.006051484
        ],
        [
          0.005959653,
          0.007991876,
          0.006606523,
          0.005793655,
          0.00591987,
          0.006288322,
          0.007853329,
          0.006897557,
          0.006380324,
          0.006333834
        ]
      ],
      "experiment": "a1-standard",
      "timing_unit": "seconds",
      "comparison": "interleaved",
      "source_report": "matrix/results.json",
      "result": {
        "head_p50": 0.00628544,
        "head_p95": 0.007860256349999999,
        "head_min": 0.005769371,
        "head_max": 0.007991876,
        "rows_per_second": 159097851.54261276,
        "samples": 20,
        "base_p50": 0.0066130419999999995,
        "change_percent": -4.953877504482806,
        "round_changes": [
          -5.861451860748567,
          -1.1514182570817155
        ],
        "round_min_changes": [
          -2.3895676401535004,
          -3.734250166614128
        ],
        "round_intervals": [
          {
            "median": -5.861451860748567,
            "low": -17.47105304553026,
            "high": 5.996612423430769,
            "coverage": 0.978515625
          },
          {
            "median": -1.1514182570817155,
            "low": -12.518496931774258,
            "high": 5.533723888955699,
            "coverage": 0.978515625
          }
        ],
        "verdict": "inconclusive"
      },
      "driver_sha256": "15232f35259003d0988badb03a6aeb65340a66ea8a4738b8c00cb9dea6c085a8",
      "worker_sha256": "9df94f7ec148efc46c18b30f010ce9d2ed66bb24f94c00e3677c58da8528ccdd",
      "correctness": true
    },
    {
      "id": "engines/100000/calc-flow-stream/asof_join/batch-1024/standard",
      "scope": "ready-enqueue-to-arrow/interleaved-inputs-v5",
      "rows": 100000,
      "batch_rows": 1024,
      "baseline": [
        [
          0.083799036,
          0.080698189,
          0.081824842,
          0.081346247,
          0.086394909,
          0.084291408,
          0.081326242,
          0.085990384,
          0.083108017,
          0.084588107
        ],
        [
          0.080880567,
          0.093255194,
          0.092907511,
          0.08456102,
          0.086077331,
          0.081429379,
          0.082021351,
          0.082551836,
          0.091742362,
          0.085583228
        ]
      ],
      "candidate": [
        [
          0.081567728,
          0.080932159,
          0.082494688,
          0.080619662,
          0.081518265,
          0.081841253,
          0.080715678,
          0.082386813,
          0.081748644,
          0.081538466
        ],
        [
          0.081357472,
          0.085421105,
          0.083497123,
          0.082594674,
          0.086114156,
          0.082972698,
          0.082505486,
          0.082777833,
          0.085625586,
          0.088040427
        ]
      ],
      "experiment": "a1-standard",
      "timing_unit": "seconds",
      "comparison": "interleaved",
      "source_report": "matrix/results.json",
      "result": {
        "head_p50": 0.08244075049999999,
        "head_p95": 0.08621046955,
        "head_min": 0.080619662,
        "head_max": 0.088040427,
        "rows_per_second": 1212992.3538238532,
        "samples": 20,
        "base_p50": 0.084045222,
        "change_percent": -1.9090573643793962,
        "round_changes": [
          -2.1491797300000726,
          0.1582725231313975
        ],
        "round_min_changes": [
          -0.09730949476450812,
          0.5896410197025359
        ],
        "round_intervals": [
          {
            "median": -2.1491797300000726,
            "low": -4.190667412300431,
            "high": 0.28993215696575536,
            "coverage": 0.978515625
          },
          {
            "median": 0.1582725231313975,
            "low": -8.40069991168535,
            "high": 1.8952852385132335,
            "coverage": 0.978515625
          }
        ],
        "verdict": "no-confirmed-regression"
      },
      "driver_sha256": "15232f35259003d0988badb03a6aeb65340a66ea8a4738b8c00cb9dea6c085a8",
      "worker_sha256": "9df94f7ec148efc46c18b30f010ce9d2ed66bb24f94c00e3677c58da8528ccdd",
      "correctness": true
    },
    {
      "id": "engines/100000/calc-flow-stream/projection/batch-1024/standard",
      "scope": "ready-enqueue-to-arrow/interleaved-inputs-v5",
      "rows": 100000,
      "batch_rows": 1024,
      "baseline": [
        [
          0.029705768,
          0.02950901,
          0.035629233,
          0.031253085,
          0.030325053,
          0.030456372,
          0.032449343,
          0.030127318,
          0.03072054,
          0.029984286
        ],
        [
          0.034372447,
          0.033346246,
          0.031290634,
          0.032293062,
          0.032660279,
          0.030295103,
          0.031568753,
          0.030196559,
          0.030726352,
          0.030282347
        ]
      ],
      "candidate": [
        [
          0.030568238,
          0.029388839,
          0.030706771,
          0.030030901,
          0.031346174,
          0.031354505,
          0.031367889,
          0.034329605,
          0.03199399,
          0.030754089
        ],
        [
          0.032349135,
          0.030230291,
          0.030143199,
          0.038249046,
          0.033502126,
          0.032014531,
          0.031708979,
          0.033712477,
          0.03750433,
          0.030779484
        ]
      ],
      "experiment": "a1-standard",
      "timing_unit": "seconds",
      "comparison": "interleaved",
      "source_report": "matrix/results.json",
      "result": {
        "head_p50": 0.031361197,
        "head_p95": 0.0375415658,
        "head_min": 0.029388839,
        "head_max": 0.038249046,
        "rows_per_second": 3188653.800427324,
        "samples": 20,
        "base_p50": 0.030723446,
        "change_percent": 2.07577952030511,
        "round_changes": [
          2.7353651567140647,
          2.1096296492352784
        ],
        "round_min_changes": [
          -0.40723494281915684,
          -0.17670887600140794
        ],
        "round_intervals": [
          {
            "median": 2.7353651567140647,
            "low": -3.910602745296987,
            "high": 4.145272185970694,
            "coverage": 0.978515625
          },
          {
            "median": 2.1096296492352784,
            "low": -5.88643572568458,
            "high": 18.44354059704838,
            "coverage": 0.978515625
          }
        ],
        "verdict": "inconclusive"
      },
      "driver_sha256": "15232f35259003d0988badb03a6aeb65340a66ea8a4738b8c00cb9dea6c085a8",
      "worker_sha256": "9df94f7ec148efc46c18b30f010ce9d2ed66bb24f94c00e3677c58da8528ccdd",
      "correctness": true
    },
    {
      "id": "engines/1000000/calc-flow-stream/asof_join/batch-64000/standard",
      "scope": "ready-enqueue-to-arrow/bounded-feeds-v6",
      "rows": 1000000,
      "batch_rows": 64000,
      "baseline": [
        [
          0.391646463,
          0.376526228,
          0.372173368,
          0.373419527,
          0.368877069,
          0.374036946,
          0.370161529,
          0.367181605,
          0.358071413,
          0.385854866
        ],
        [
          0.363625929,
          0.385707966,
          0.389480157,
          0.364398615,
          0.36878107,
          0.369373077,
          0.378677427,
          0.376494385,
          0.366537163,
          0.37081269
        ]
      ],
      "candidate": [
        [
          0.35327765,
          0.370234224,
          0.363277262,
          0.352205254,
          0.360110503,
          0.365788982,
          0.358670233,
          0.363699818,
          0.347244525,
          0.354803727
        ],
        [
          0.343295433,
          0.353537057,
          0.359111863,
          0.356990464,
          0.349506721,
          0.361892688,
          0.376846603,
          0.366290973,
          0.351124176,
          0.355957707
        ]
      ],
      "experiment": "progress-compatible",
      "timing_unit": "seconds",
      "comparison": "interleaved",
      "source_report": "matrix/results.json",
      "result": {
        "head_p50": 0.3578303485,
        "head_p95": 0.37056484295,
        "head_min": 0.343295433,
        "head_max": 0.376846603,
        "rows_per_second": 2794620.4233149327,
        "samples": 20,
        "base_p50": 0.37149302900000003,
        "change_percent": -3.6777757409816814,
        "round_changes": [
          -2.7069897963673872,
          -4.105544104841874
        ],
        "round_min_changes": [
          -3.023667237015648,
          -5.591046836486746
        ],
        "round_intervals": [
          {
            "median": -2.7069897963673872,
            "low": -8.047362295024163,
            "high": -1.6710665903465372,
            "coverage": 0.978515625
          },
          {
            "median": -4.105544104841874,
            "low": -7.797135092558771,
            "high": -2.025158157371598,
            "coverage": 0.978515625
          }
        ],
        "verdict": "no-confirmed-regression"
      },
      "driver_sha256": "15232f35259003d0988badb03a6aeb65340a66ea8a4738b8c00cb9dea6c085a8",
      "worker_sha256": "9df94f7ec148efc46c18b30f010ce9d2ed66bb24f94c00e3677c58da8528ccdd",
      "correctness": true
    },
    {
      "id": "engines/1000000/calc-flow-stream/projection/batch-64000/standard",
      "scope": "ready-enqueue-to-arrow/bounded-feeds-v6",
      "rows": 1000000,
      "batch_rows": 64000,
      "baseline": [
        [
          0.007620724,
          0.006169698,
          0.006169402,
          0.005953508,
          0.006321981,
          0.006261162,
          0.006325925,
          0.006764739,
          0.006252156,
          0.005832533
        ],
        [
          0.006340148,
          0.005861376,
          0.005816456,
          0.005972341,
          0.005884463,
          0.006519884,
          0.006493355,
          0.006496874,
          0.005969383,
          0.006237012
        ]
      ],
      "candidate": [
        [
          0.008816657,
          0.009393882,
          0.00643969,
          0.006259097,
          0.006501676,
          0.005982169,
          0.006116402,
          0.005654025,
          0.006471925,
          0.006816282
        ],
        [
          0.005989845,
          0.005828908,
          0.006883292,
          0.006166817,
          0.006967948,
          0.006475097,
          0.006321813,
          0.006917755,
          0.006440682,
          0.005943464
        ]
      ],
      "experiment": "progress-compatible",
      "timing_unit": "seconds",
      "comparison": "interleaved",
      "source_report": "matrix/results.json",
      "result": {
        "head_p50": 0.006440186,
        "head_p95": 0.00884551825,
        "head_min": 0.005654025,
        "head_max": 0.009393882,
        "rows_per_second": 155275018.45443594,
        "samples": 20,
        "base_p50": 0.006244584,
        "change_percent": 3.1323463660669626,
        "round_changes": [
          3.9480983832359984,
          1.3511730999823168
        ],
        "round_min_changes": [
          -3.0605570512845737,
          0.214082252148029
        ],
        "round_intervals": [
          {
            "median": 3.9480983832359984,
            "low": -4.455930065377645,
            "high": 16.866582666570416,
            "coverage": 0.978515625
          },
          {
            "median": 1.3511730999823168,
            "low": -4.706548584482451,
            "high": 18.341684352120936,
            "coverage": 0.978515625
          }
        ],
        "verdict": "inconclusive"
      },
      "driver_sha256": "15232f35259003d0988badb03a6aeb65340a66ea8a4738b8c00cb9dea6c085a8",
      "worker_sha256": "9df94f7ec148efc46c18b30f010ce9d2ed66bb24f94c00e3677c58da8528ccdd",
      "correctness": true
    },
    {
      "id": "engines/1000000/calc-flow-stream/join/batch-64000/standard",
      "scope": "ready-enqueue-to-arrow/bounded-feeds-v6",
      "rows": 1000000,
      "batch_rows": 64000,
      "baseline": [],
      "candidate": [
        [
          0.666496069,
          0.664495073,
          0.652404356,
          0.653639586,
          0.643605829,
          0.654204317,
          0.648916119,
          0.687857695,
          0.666558816,
          0.66123832
        ],
        [
          0.672671629,
          0.639420692,
          0.64085806,
          0.656021731,
          0.652402971,
          0.648413994,
          0.675232551,
          0.661272398,
          0.663825015,
          0.640019711
        ]
      ],
      "experiment": "progress-new-coverage",
      "timing_unit": "seconds",
      "comparison": "new",
      "source_report": "matrix/results.json",
      "result": {
        "head_p50": 0.655113024,
        "head_p95": 0.6758638082,
        "head_min": 0.639420692,
        "head_max": 0.687857695,
        "rows_per_second": 1526454.1588475578,
        "samples": 20,
        "base_p50": null,
        "change_percent": null,
        "round_changes": [],
        "round_min_changes": [],
        "round_intervals": [],
        "verdict": "new-coverage"
      },
      "driver_sha256": "15232f35259003d0988badb03a6aeb65340a66ea8a4738b8c00cb9dea6c085a8",
      "worker_sha256": "9df94f7ec148efc46c18b30f010ce9d2ed66bb24f94c00e3677c58da8528ccdd",
      "correctness": true
    },
    {
      "id": "engines/1000000/calc-flow-stream/join/batch-64000/standard",
      "scope": "v5-to-v6-static-dimension-diagnostic",
      "rows": 1000000,
      "batch_rows": 64000,
      "baseline": [
        [
          0.691230261,
          0.677339031,
          0.686506994,
          0.693812831,
          0.684223659,
          0.721417577,
          0.679556949,
          0.723668746,
          0.701253683,
          0.684842732
        ],
        [
          0.686588041,
          0.692132773,
          0.723644124,
          0.684887077,
          0.700370284,
          0.710205632,
          0.70682485,
          0.669137862,
          0.668836036,
          0.686878547
        ]
      ],
      "candidate": [
        [
          0.666084591,
          0.66867722,
          0.662613684,
          0.651025785,
          0.630737795,
          0.631347108,
          0.635453126,
          0.623109182,
          0.639538083,
          0.624884512
        ],
        [
          0.64319927,
          0.658097456,
          0.682301594,
          0.654364909,
          0.625403423,
          0.629873563,
          0.633837127,
          0.641739063,
          0.652037805,
          0.62987271
        ]
      ],
      "experiment": "progress-changed-scope",
      "timing_unit": "seconds",
      "comparison": "interleaved",
      "source_report": "matrix/results.json",
      "result": {
        "head_p50": 0.640638573,
        "head_p95": 0.6693584387,
        "head_min": 0.623109182,
        "head_max": 0.682301594,
        "rows_per_second": 1560942.5378761885,
        "samples": 20,
        "base_p50": 0.689054404,
        "change_percent": -7.026416306019268,
        "round_changes": [
          -7.153549804501158,
          -6.016289542635922
        ],
        "round_min_changes": [
          -8.006307996150309,
          -6.493760901363876
        ],
        "round_intervals": [
          {
            "median": -7.153549804501158,
            "low": -12.485205777014208,
            "high": -3.480417564398486,
            "coverage": 0.978515625
          },
          {
            "median": -6.016289542635922,
            "low": -10.70388945856533,
            "high": -4.0946418602150425,
            "coverage": 0.978515625
          }
        ],
        "verdict": "no-confirmed-regression"
      },
      "driver_sha256": "15232f35259003d0988badb03a6aeb65340a66ea8a4738b8c00cb9dea6c085a8",
      "worker_sha256": "9df94f7ec148efc46c18b30f010ce9d2ed66bb24f94c00e3677c58da8528ccdd",
      "correctness": true
    },
    {
      "id": "engines/100000/calc-flow-stream/interval_join/batch-1024",
      "scope": "ready-enqueue-to-arrow/exact-cursor-batch-1024-v1",
      "rows": 100000,
      "batch_rows": 1024,
      "baseline": [],
      "candidate": [
        [
          0.60583532,
          0.604627989,
          0.614371089,
          0.61461032,
          0.648531797,
          0.620333473,
          0.624007676,
          0.615551437,
          0.61246598,
          0.625682806
        ],
        [
          0.609516815,
          0.627046028,
          0.614170614,
          0.614929912,
          0.627276262,
          0.604938674,
          0.627446388,
          0.613275303,
          0.612091307,
          0.614469312
        ]
      ],
      "experiment": "retained-newcoverage",
      "timing_unit": "seconds",
      "comparison": "new",
      "source_report": "coverage/results.json",
      "result": {
        "head_p50": 0.614539816,
        "head_p95": 0.62850065845,
        "head_min": 0.604627989,
        "head_max": 0.648531797,
        "rows_per_second": 162723.38650226695,
        "samples": 20,
        "base_p50": null,
        "change_percent": null,
        "round_changes": [],
        "round_min_changes": [],
        "round_intervals": [],
        "verdict": "new-coverage"
      },
      "collector_sha256": "2926d215a2741f9fd153d6568bfce6d0be5f253f71e9b866a22615a0c5cea2af",
      "harness_sha256": "c724f0a77c9e871d5b73299feaa8e69a37abca1341002df31cfee66bc2cfd1a7",
      "environment_state": "loaded-functional-cost-diagnostic",
      "correctness": true
    },
    {
      "id": "engines/100000/calc-flow-stream/asof_join/batch-64000/checkpoint-100ms-duration-recovery",
      "scope": "ready-enqueue-checkpoint-100ms-ack-recover-to-arrow-v1",
      "rows": 100000,
      "batch_rows": 64000,
      "baseline": [],
      "candidate": [
        [
          0.248253261,
          0.247187087,
          0.254987817,
          0.229620709,
          0.193148863,
          0.197118359,
          0.215178482,
          0.21680251,
          0.210314937,
          0.203425977
        ],
        [
          0.209165539,
          0.217203494,
          0.212055614,
          0.231696825,
          0.214701464,
          0.213747486,
          0.226027181,
          0.21126812,
          0.22051853,
          0.217939866
        ]
      ],
      "experiment": "retained-newcoverage",
      "timing_unit": "seconds",
      "comparison": "new",
      "source_report": "coverage/results.json",
      "result": {
        "head_p50": 0.215990496,
        "head_p95": 0.2485899888,
        "head_min": 0.193148863,
        "head_max": 0.254987817,
        "rows_per_second": 462983.33422966907,
        "samples": 20,
        "base_p50": null,
        "change_percent": null,
        "round_changes": [],
        "round_min_changes": [],
        "round_intervals": [],
        "verdict": "new-coverage"
      },
      "collector_sha256": "2926d215a2741f9fd153d6568bfce6d0be5f253f71e9b866a22615a0c5cea2af",
      "harness_sha256": "c724f0a77c9e871d5b73299feaa8e69a37abca1341002df31cfee66bc2cfd1a7",
      "environment_state": "loaded-functional-cost-diagnostic",
      "correctness": true
    },
    {
      "id": "engines/1000000/calc-flow-stream/interval_join",
      "scope": "ready-enqueue-to-arrow/retained-interval-v1",
      "experiment": "loaded-large-interval",
      "source_report": "large-cost.json",
      "timing_unit": "seconds",
      "baseline": [],
      "candidate": [
        [
          49.203676654,
          53.835937244
        ]
      ],
      "sample_count": 2,
      "complete_workers": 1,
      "verdict": "candidate-loaded-cost-diagnostic; insufficient-two-rounds",
      "p50_seconds": 51.519806949,
      "p95_seconds": 53.604324214500004,
      "collector_sha256": "e1098d952232c444a844d7b4a70b08a9d6b1bfdae77185cf9e34a9b465808db3"
    }
  ]
}
```
