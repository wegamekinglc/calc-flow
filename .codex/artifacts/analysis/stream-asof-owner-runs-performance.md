# Stream ASOF owner-run performance evidence

The final A2 stage observes a 6.18% lower median for 1M/64k ASOF and 2.01%
for 100k/1024. The 1M second-round interval does not establish a material
improvement above 5%; neither does the small-batch comparison. Pipelined
contiguous and overlap medians decrease 7.99% and 4.00%, respectively, with
insufficient two-round evidence for the material-improvement gate. Projection
controls are inconclusive. These are measured point estimates, rather than
confirmed gains exceeding 5% or a proof that every case is regression-free.

This is incremental A2 evidence for
[PR #370](https://github.com/wegamekinglc/calc-flow/pull/370), using the final
dependency-integrated J1 candidate as baseline. Earlier 416/29b builds were
preparation artifacts and were never treated as the final measured candidate.
No A3, A4 or A5 implementation or performance is included.

### Final compatible comparison and controls

The clean release comparison is `f1ef46bbf52fc39dfb820b6c143f1c8b2fabea23` →
`22fcb739581cbd53b59e4380b625e6e1230bfb65`. Both incorporate the identical paid ASOF retirement repair,
reviewed status helper, retained benchmark foundation and dependency versions.
Only the native A2 stage differs. The common Python measuring
harness is the candidate's immutable clean snapshot. Its SHA-256 is
`6593983e766921f3c73bbdcef0c3aaad011f93d41463467481c9dc7b6d9ddff5`. The J1 candidate wheel is reused byte-for-byte
as the A2 baseline; it is not rebuilt. The final source commits precede the
later report-only PR update.

All cases have two rounds of ten alternating AB/BA pairs, with separate workers
per round. The exact paired order-statistic intervals and ±5% gates use the
maintained suite policy, not minimum timings. Full payload oracles are outside
the clock. Each sample starts a fresh ready job with empty state; planning,
startup, EOF/retirement cleanup and verification are excluded. Ordinary ready
batch size is explicitly named `fixture_batch_rows`; original input event
attestations prove actual data batches. Replay variants retain `batch_rows`
and every maintained seven-dimensional adapter evidence gate. No validator
was weakened to accept malformed replay evidence.

|  Case                                                    |  Base p50 ms  |  Head p50 ms  |  Head p95 ms  |  p50 change  |  R1 paired interval %  |  R2 paired interval %  |  Verdict                  |
|----------------------------------------------------------|---------------|---------------|---------------|--------------|------------------------|------------------------|---------------------------|
|  64000/calc-flow-stream/asof_join/batch-64000/overlap    |  39.720450    |  38.131734    |  39.681537    |  -4.000%     |  [-8.317, -0.287]      |  [-7.646, +0.608]      |  no-confirmed-regression  |
|  64000/calc-flow-stream/asof_join/batch-64000/pipelined  |  25.722514    |  23.667081    |  25.349948    |  -7.991%     |  [-9.597, -2.872]      |  [-14.126, -4.473]     |  no-confirmed-regression  |
|  1000000/calc-flow-stream/asof_join/batch-64000          |  357.543963   |  335.438075   |  341.317677   |  -6.183%     |  [-8.797, -5.217]      |  [-9.522, -4.329]      |  no-confirmed-regression  |
|  1000000/calc-flow-stream/projection/batch-64000         |  6.372246     |  6.481345     |  7.153004     |  +1.712%     |  [-6.897, +14.279]     |  [-10.088, +14.288]    |  inconclusive             |
|  100000/calc-flow-stream/asof_join/batch-1024            |  81.142234    |  79.514511    |  81.580247    |  -2.006%     |  [-5.534, +0.035]      |  [-4.168, -0.785]      |  no-confirmed-regression  |
|  100000/calc-flow-stream/projection/batch-1024           |  33.612685    |  33.233262    |  39.095407    |  -1.129%     |  [-5.281, +9.700]      |  [-5.890, +1.040]      |  inconclusive             |

The point estimates describe this run. A confirmed material improvement
requires both paired upper bounds below −5%; none of these final E2E cases
meets that condition. Join/ASOF observations must not inherit the much larger
bounded native-progress speedups. Projection controls have broad intervals
whose upper bounds exceed +5%, so their verdict is **inconclusive**. No case
has a confirmed regression, but the overall E2E absence of a regression is
not established for those controls. There is no selective repeat or minimum
selection to obtain a favorable result.

### Final worker memory and host limits

|  Case                                                    |  Version    |  Final RSS median MiB  |  Round worker HWM MiB  |
|----------------------------------------------------------|-------------|------------------------|------------------------|
|  64000/calc-flow-stream/asof_join/batch-64000/overlap    |  baseline   |  421.059               |  430.293, 430.973      |
|  64000/calc-flow-stream/asof_join/batch-64000/overlap    |  candidate  |  416.348               |  428.172, 422.008      |
|  64000/calc-flow-stream/asof_join/batch-64000/pipelined  |  baseline   |  426.965               |  426.211, 437.219      |
|  64000/calc-flow-stream/asof_join/batch-64000/pipelined  |  candidate  |  414.150               |  424.023, 413.789      |
|  1000000/calc-flow-stream/asof_join/batch-64000          |  baseline   |  693.971               |  759.918, 746.156      |
|  1000000/calc-flow-stream/asof_join/batch-64000          |  candidate  |  698.820               |  746.742, 768.797      |
|  1000000/calc-flow-stream/projection/batch-64000         |  baseline   |  472.861               |  507.367, 499.148      |
|  1000000/calc-flow-stream/projection/batch-64000         |  candidate  |  472.559               |  494.938, 511.094      |
|  100000/calc-flow-stream/asof_join/batch-1024            |  baseline   |  347.193               |  356.000, 354.723      |
|  100000/calc-flow-stream/asof_join/batch-1024            |  candidate  |  345.559               |  347.645, 358.023      |
|  100000/calc-flow-stream/projection/batch-1024           |  baseline   |  366.229               |  371.273, 372.527      |
|  100000/calc-flow-stream/projection/batch-1024           |  candidate  |  365.979               |  372.328, 371.004      |

The Python worker RSS includes NumPy/PyArrow/native imports, immutable input,
retained owners, gathered Arrow outputs and full oracle sorting. Two worker
HWMs per version are descriptive memory observations, not a statistically
confirmed RSS change or per-index allocation count. Both version workers
coexist for E2E alternating sampling; their peaks must not be confused with
the sequential single-resident native retained-state probe.

The team-coordinated quiet WSL2 run was 2026-10-06T14:33:53.432899+00:00 to
2026-10-06T14:34:52.092733+00:00; one-minute load-average observations ranged
0.384–1.036.
CPU is i9-13900HX, 32 logical CPUs, affinity 0–31, 32 Tokio and Polars threads,
and OMP/OpenBLAS/MKL limited to one. Actual loaded native module SHA and
dependency/thread identities were checked on all workers. Windows-host power
mode and background scheduling remain unobservable. The broad projection
intervals bound the strength of the conclusion despite team quiet.

### Raw evidence and independent verification

The independent checker verified 240 exact timed values,
24 excluded warmups, 24 unique
worker PIDs, 312 raw IPC responses, all full payload
oracles, actual source-event batch shapes, every causal/completed state gate,
worker exit code zero, and both paired intervals. It rederived the statistics
without importing the suite statistics implementation. It also confirmed the
shared f1 wheel and matching final harness/machine/dependency fingerprints.

Raw paths: `target/issue363-asof-owner-runs-perf/final/matrix-final`.
Checker: J1 target `final/verify.py`. The combined archive lives in the J1
target at `final/issue363-final-e2e-evidence.tar.gz`,
SHA-256 `e78793b6ab41f0d638fde3440312fef307fdab1bb677484905764922521a9c0c`, 330,235 bytes. It contains both
completed matrices, the failed warmup attempt, exact per-stage driver/worker
copies, hello preflight, independent verification and three release/build
manifests/logs. It excludes source snapshots, wheels, binaries, mutable Cargo
cache and these subsequently written Markdown reports. The appendices embed
the arrays, seals and worker proofs for review without ignored target paths.

The first J1 attempt failed immediately after a valid warmup because the
ordinary ready case incorrectly used the replay-reserved `batch_rows` field.
The maintained gate correctly rejected missing replay adapter evidence.
There were zero measured pairs. That attempt remains at J1 `final/matrix`,
with its exact driver copy in `final/attempts/warmup-dimensions`; the corrected
full run uses `matrix-final`. Ordinary actual 1024-row feeds were independently
verified as 97×1024+672 rows for each 100k input, never a default 64k feed.
The final A2 driver checks this attestation before the first measured pair.
The J1 driver's previously loaded exact bytes are preserved separately;
adding the A2 gate did not change a running J1 worker or its recorded hash.

### ASOF workload, order oracle and unverified targets

Standard ASOF uses identical 1M-row 64k feeds and 100k-row 1024 feeds on
both inputs, zero tolerance, 64 entities and full sequence/value oracles.
The 64k diagnostics preload the same reference, then pipeline two 32k left
batches: contiguous halves or even/odd overlapping time ranges. Both preserve
all output work, complete payload validation and exact canonical sequence
order. These diagnostics were declared before the comparison and retained
regardless of their result.

The matched raw small-batch ASOF/projection p50 ratio is 2.414× on baseline
and 2.393× on candidate. This ratio includes actual native ASOF work and
queue/output costs. It does not isolate harness overhead, and subtraction
of projection time is not true overhead. The 2× operator/floor target remains
unverified without a matched independent operator timer. The future A4 1M
100ms target is not an A2 gate and is not reached by this 335.438ms observation.

The changed hot path collects finalized left prefixes and reuses owner
bundles across selected rows. Existing same-ref release cases cover normal,
small, contiguous and overlapping batches. Allocation counts, unique-key
fragmentation sweeps, restore/capture, checkpoint cost and worker allocation
peaks need additional maintained coverage. RSS here includes outputs/oracles
and cannot establish one-owner-per-batch allocations. Source correctness and
complexity tests remain distinct from this timing evidence.

### Exact final arrays, identities and release manifests

```json
{
  "stage": "a2",
  "started_utc": "2026-10-06T14:33:53.432899+00:00",
  "finished_utc": "2026-10-06T14:34:52.092733+00:00",
  "harness": {
    "git_sha": "22fcb739581cbd53b59e4380b625e6e1230bfb65",
    "git_clean": true,
    "source_sha256": "a9cb90664b78e13c4df17fefc5b72ddc3a87f7f93527a4944cc2aff22d1bae2a",
    "cargo_lock_sha256": "84789d29f2cb7342b0297169caf03c8f14227645656d34956ed9bc9aaa0c4840"
  },
  "harness_sha256": "6593983e766921f3c73bbdcef0c3aaad011f93d41463467481c9dc7b6d9ddff5",
  "driver_sha256": "ec7d763da5598aaf73a314ed39445a847d7fef50221d045cf939be0c25a21dca",
  "worker_sha256": "2008f3653561450952ca62f095e9b66a9a440bdcffc61d57359d37b7c956f69f",
  "environment": {
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
  "environment_sha256": "3ffec2325519909fc4073839832f785bfa76ef9990c286fa292e114d32c5a8c4",
  "releases": {
    "baseline": {
      "contract": "benchmark-release-v1",
      "source": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/final/sources/j1-candidate",
      "git_sha": "f1ef46bbf52fc39dfb820b6c143f1c8b2fabea23",
      "git_clean": true,
      "source_sha256": "75d2592fc33e0fb746b8f99bdd6a26122c6671ea1e22fd56c7a4ded34d26be54",
      "cargo_lock_sha256": "84789d29f2cb7342b0297169caf03c8f14227645656d34956ed9bc9aaa0c4840",
      "wheel": "calc_flow_python-2026.9.25-cp313-abi3-linux_x86_64.whl",
      "wheel_sha256": "8c2e520186d1403c5e15f0f8839287e433e185ae5e32b6233d139c0db6d8ae25",
      "native_sha256": "4e9d884d26ff1931868019b4b400db16628fc2f2dfacee3f82b7e749cd8f16d2",
      "build_profile": "release",
      "command": [
        "python",
        "-m",
        "maturin",
        "build",
        "--release",
        "--locked",
        "--features",
        "pyo3/abi3-py313",
        "--out",
        "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/final/releases/j1-candidate"
      ],
      "rustc": "rustc 1.88.0 (6b00bc388 2025-06-23)\nbinary: rustc\ncommit-hash: 6b00bc3880198600130e1cf62b8f8a93494488cc\ncommit-date: 2025-06-23\nhost: x86_64-unknown-linux-gnu\nrelease: 1.88.0\nLLVM version: 20.1.5",
      "python": "3.13.9",
      "python_executable": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/benchmark-venv/bin/python",
      "wheel_path": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/final/releases/j1-candidate/calc_flow_python-2026.9.25-cp313-abi3-linux_x86_64.whl"
    },
    "candidate": {
      "contract": "benchmark-release-v1",
      "source": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/final/sources/a2-candidate",
      "git_sha": "22fcb739581cbd53b59e4380b625e6e1230bfb65",
      "git_clean": true,
      "source_sha256": "a9cb90664b78e13c4df17fefc5b72ddc3a87f7f93527a4944cc2aff22d1bae2a",
      "cargo_lock_sha256": "84789d29f2cb7342b0297169caf03c8f14227645656d34956ed9bc9aaa0c4840",
      "wheel": "calc_flow_python-2026.9.25-cp313-abi3-linux_x86_64.whl",
      "wheel_sha256": "22f1316e66666b837c62241211372c52132f16c4f95cbfcfdd6d264f529141bb",
      "native_sha256": "188d898adf533f8f6b7f66f1dc5c7b1ef953621dbda13f140c295a0a5b9be62e",
      "build_profile": "release",
      "command": [
        "python",
        "-m",
        "maturin",
        "build",
        "--release",
        "--locked",
        "--features",
        "pyo3/abi3-py313",
        "--out",
        "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/final/releases/a2-candidate"
      ],
      "rustc": "rustc 1.88.0 (6b00bc388 2025-06-23)\nbinary: rustc\ncommit-hash: 6b00bc3880198600130e1cf62b8f8a93494488cc\ncommit-date: 2025-06-23\nhost: x86_64-unknown-linux-gnu\nrelease: 1.88.0\nLLVM version: 20.1.5",
      "python": "3.13.9",
      "python_executable": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/benchmark-venv/bin/python",
      "wheel_path": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/final/releases/a2-candidate/calc_flow_python-2026.9.25-cp313-abi3-linux_x86_64.whl"
    }
  },
  "archive": {
    "path": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/final/issue363-final-e2e-evidence.tar.gz",
    "sha256": "e78793b6ab41f0d638fde3440312fef307fdab1bb677484905764922521a9c0c",
    "size_bytes": 330235,
    "contents": "Both completed final matrices, failed warmup attempt, exact per-stage drivers, hello preflight, independent checker, verification and three build/release manifests/logs. Excludes source snapshots, wheels, binaries, mutable Cargo cache and subsequently written Markdown."
  },
  "cases": [
    {
      "id": "engines/64000/calc-flow-stream/asof_join/batch-64000/overlap",
      "family": "engines",
      "backend": "calc-flow-stream",
      "scenario": "asof_join",
      "rows": 64000,
      "fixture_batch_rows": 64000,
      "diagnostic_variant": "overlap",
      "scope": "ready-enqueue-to-arrow/pipelined-overlap-v1",
      "status": "ok",
      "correctness": true,
      "comparison": "interleaved",
      "baseline": [
        [
          0.037251104,
          0.041093147,
          0.041409878,
          0.041235333,
          0.038961719,
          0.039563191,
          0.040003091,
          0.038737988,
          0.037812777,
          0.040649146
        ],
        [
          0.037470908,
          0.039233145,
          0.040936349,
          0.039199249,
          0.040609359,
          0.041350768,
          0.039877708,
          0.039541739,
          0.040689488,
          0.039376911
        ]
      ],
      "candidate": [
        [
          0.036142692,
          0.036504851,
          0.039715501,
          0.038634428,
          0.039679749,
          0.03855324,
          0.037382056,
          0.036809487,
          0.037704086,
          0.037268251
        ],
        [
          0.038031324,
          0.039471605,
          0.037393553,
          0.036605814,
          0.038848016,
          0.038189211,
          0.038175861,
          0.038406982,
          0.038715854,
          0.038087606
        ]
      ],
      "result": {
        "head_p50": 0.0381317335,
        "head_p95": 0.0396815366,
        "head_min": 0.036142692,
        "head_max": 0.039715501,
        "rows_per_second": 1678392.09303191,
        "samples": 20,
        "base_p50": 0.0397204495,
        "change_percent": -3.9997432556748813,
        "round_changes": [
          -4.535020887960839,
          -4.302474232461406
        ],
        "round_min_changes": [
          -2.9755144975032177,
          -2.308708398526127
        ],
        "round_intervals": [
          {
            "median": -4.535020887960839,
            "low": -8.317259604912719,
            "high": -0.28744516701325606,
            "coverage": 0.978515625
          },
          {
            "median": -4.302474232461406,
            "low": -7.645703218861621,
            "high": 0.6078024078875144,
            "coverage": 0.978515625
          }
        ],
        "verdict": "no-confirmed-regression"
      }
    },
    {
      "id": "engines/64000/calc-flow-stream/asof_join/batch-64000/pipelined",
      "family": "engines",
      "backend": "calc-flow-stream",
      "scenario": "asof_join",
      "rows": 64000,
      "fixture_batch_rows": 64000,
      "diagnostic_variant": "pipelined",
      "scope": "ready-enqueue-to-arrow/pipelined-contiguous-v1",
      "status": "ok",
      "correctness": true,
      "comparison": "interleaved",
      "baseline": [
        [
          0.025428031,
          0.025659342,
          0.025835941,
          0.025169215,
          0.025823475,
          0.025339719,
          0.024798644,
          0.026440538,
          0.025711296,
          0.026062154
        ],
        [
          0.027662874,
          0.02488011,
          0.025848364,
          0.025733732,
          0.026189111,
          0.025620307,
          0.024826137,
          0.025870764,
          0.025121062,
          0.026256118
        ]
      ],
      "candidate": [
        [
          0.023859689,
          0.023196699,
          0.022037678,
          0.024446325,
          0.023419627,
          0.023591755,
          0.023664857,
          0.024793246,
          0.02435004,
          0.025584166
        ],
        [
          0.02353808,
          0.023669306,
          0.024692106,
          0.023941291,
          0.022489526,
          0.023148766,
          0.025337621,
          0.022670543,
          0.023230955,
          0.023739189
        ]
      ],
      "result": {
        "head_p50": 0.0236670815,
        "head_p95": 0.02534994825,
        "head_min": 0.022037678,
        "head_max": 0.025584166,
        "rows_per_second": 2704177.952824475,
        "samples": 20,
        "base_p50": 0.025722514000000002,
        "change_percent": -7.990791646570794,
        "round_changes": [
          -6.198971980778917,
          -8.555029986549522
        ],
        "round_min_changes": [
          -11.133536172381042,
          -9.4118992415131
        ],
        "round_intervals": [
          {
            "median": -6.198971980778917,
            "low": -9.597451875422214,
            "high": -2.8721197701239376,
            "coverage": 0.978515625
          },
          {
            "median": -8.555029986549522,
            "low": -14.12642452811782,
            "high": -4.47323474708109,
            "coverage": 0.978515625
          }
        ],
        "verdict": "no-confirmed-regression"
      }
    },
    {
      "id": "engines/1000000/calc-flow-stream/asof_join/batch-64000/standard",
      "family": "engines",
      "backend": "calc-flow-stream",
      "scenario": "asof_join",
      "rows": 1000000,
      "fixture_batch_rows": 64000,
      "diagnostic_variant": "standard",
      "scope": "ready-enqueue-to-arrow/interleaved-inputs-v5",
      "status": "ok",
      "correctness": true,
      "comparison": "interleaved",
      "baseline": [
        [
          0.356259177,
          0.359774678,
          0.35724336,
          0.34385498,
          0.35955359,
          0.356603327,
          0.35579502,
          0.373023573,
          0.364884815,
          0.372893673
        ],
        [
          0.362955825,
          0.378015895,
          0.356073756,
          0.352139459,
          0.350208748,
          0.365634083,
          0.36127887,
          0.356117105,
          0.355113284,
          0.357844566
        ]
      ],
      "candidate": [
        [
          0.326283323,
          0.341005881,
          0.330509078,
          0.323509803,
          0.340467185,
          0.331919988,
          0.339116498,
          0.335483722,
          0.336673596,
          0.340091196
        ],
        [
          0.347241804,
          0.338089949,
          0.328202786,
          0.320109801,
          0.335392429,
          0.340086375,
          0.338451259,
          0.328627808,
          0.321299357,
          0.330096373
        ]
      ],
      "result": {
        "head_p50": 0.3354380755,
        "head_p95": 0.34131767715,
        "head_min": 0.320109801,
        "head_max": 0.347241804,
        "rows_per_second": 2981176.1783733466,
        "samples": 20,
        "base_p50": 0.357543963,
        "change_percent": -6.182704726579324,
        "round_changes": [
          -7.2026423038613565,
          -7.736716017475265
        ],
        "round_min_changes": [
          -5.916789979310466,
          -8.594573142987272
        ],
        "round_intervals": [
          {
            "median": -7.2026423038613565,
            "low": -8.796737347699645,
            "high": -5.216819900815805,
            "coverage": 0.978515625
          },
          {
            "median": -7.736716017475265,
            "low": -9.522011291472833,
            "high": -4.329458274984299,
            "coverage": 0.978515625
          }
        ],
        "verdict": "no-confirmed-regression"
      }
    },
    {
      "id": "engines/1000000/calc-flow-stream/projection/batch-64000/standard",
      "family": "engines",
      "backend": "calc-flow-stream",
      "scenario": "projection",
      "rows": 1000000,
      "fixture_batch_rows": 64000,
      "diagnostic_variant": "standard",
      "scope": "ready-enqueue-to-arrow/interleaved-inputs-v5",
      "status": "ok",
      "correctness": true,
      "comparison": "interleaved",
      "baseline": [
        [
          0.006008496,
          0.00671349,
          0.006347021,
          0.006648209,
          0.006385049,
          0.006035348,
          0.006556372,
          0.00644383,
          0.00736985,
          0.005980716
        ],
        [
          0.006781121,
          0.006123166,
          0.006659068,
          0.006081744,
          0.005879661,
          0.005869295,
          0.006702805,
          0.006627312,
          0.006359444,
          0.005904946
        ]
      ],
      "candidate": [
        [
          0.007126427,
          0.006488559,
          0.006115343,
          0.006816983,
          0.005944647,
          0.006601402,
          0.006158317,
          0.006590776,
          0.006272964,
          0.006834696
        ],
        [
          0.006318936,
          0.006474131,
          0.00644526,
          0.006614624,
          0.006719741,
          0.00765796,
          0.006026623,
          0.005811446,
          0.006257139,
          0.006739328
        ]
      ],
      "result": {
        "head_p50": 0.0064813449999999995,
        "head_p95": 0.007153003650000001,
        "head_min": 0.005811446,
        "head_max": 0.00765796,
        "rows_per_second": 154288963.17045307,
        "samples": 20,
        "base_p50": 0.0063722464999999995,
        "change_percent": 1.7120885075616599,
        "round_changes": [
          -0.5350096273960381,
          2.0615236250645594
        ],
        "round_min_changes": [
          -0.6030883258793751,
          -0.9856209306228547
        ],
        "round_intervals": [
          {
            "median": -0.5350096273960381,
            "low": -6.897394209504116,
            "high": 14.278892360045203,
            "coverage": 0.978515625
          },
          {
            "median": 2.0615236250645594,
            "low": -10.088045228825838,
            "high": 14.287898571023083,
            "coverage": 0.978515625
          }
        ],
        "verdict": "inconclusive"
      }
    },
    {
      "id": "engines/100000/calc-flow-stream/asof_join/batch-1024/standard",
      "family": "engines",
      "backend": "calc-flow-stream",
      "scenario": "asof_join",
      "rows": 100000,
      "fixture_batch_rows": 1024,
      "diagnostic_variant": "standard",
      "scope": "ready-enqueue-to-arrow/interleaved-inputs-v5",
      "status": "ok",
      "correctness": true,
      "comparison": "interleaved",
      "baseline": [
        [
          0.084437296,
          0.08276544,
          0.079938459,
          0.080204952,
          0.085092264,
          0.081488909,
          0.081242275,
          0.080141821,
          0.080588079,
          0.084070103
        ],
        [
          0.079816261,
          0.082675728,
          0.081042193,
          0.082096557,
          0.080442808,
          0.080226824,
          0.07973767,
          0.079717846,
          0.082187598,
          0.085710182
        ]
      ],
      "candidate": [
        [
          0.079764839,
          0.08008964,
          0.076106272,
          0.081164254,
          0.079406723,
          0.079431634,
          0.079354105,
          0.080169627,
          0.077858539,
          0.080469003
        ],
        [
          0.079720073,
          0.081550882,
          0.079215963,
          0.08091836,
          0.07856493,
          0.079597389,
          0.07855644,
          0.077551897,
          0.078232547,
          0.082138175
        ]
      ],
      "result": {
        "head_p50": 0.0795145115,
        "head_p95": 0.08158024665000001,
        "head_min": 0.076106272,
        "head_max": 0.082138175,
        "rows_per_second": 1257632.0738636495,
        "samples": 20,
        "base_p50": 0.08114223400000001,
        "change_percent": -2.006011444052691,
        "round_changes": [
          -3.310009470804337,
          -1.8674131515935055
        ],
        "round_min_changes": [
          -4.7939215340640935,
          -2.7170189721383076
        ],
        "round_intervals": [
          {
            "median": -3.310009470804337,
            "low": -5.533641200447715,
            "high": 0.03469599224603659,
            "coverage": 0.978515625
          },
          {
            "median": -1.8674131515935055,
            "low": -4.1675410279726215,
            "high": -0.7845692607749255,
            "coverage": 0.978515625
          }
        ],
        "verdict": "no-confirmed-regression"
      }
    },
    {
      "id": "engines/100000/calc-flow-stream/projection/batch-1024/standard",
      "family": "engines",
      "backend": "calc-flow-stream",
      "scenario": "projection",
      "rows": 100000,
      "fixture_batch_rows": 1024,
      "diagnostic_variant": "standard",
      "scope": "ready-enqueue-to-arrow/interleaved-inputs-v5",
      "status": "ok",
      "correctness": true,
      "comparison": "interleaved",
      "baseline": [
        [
          0.034372031,
          0.033538018,
          0.035282203,
          0.03386919,
          0.031943446,
          0.032958212,
          0.03497015,
          0.035785246,
          0.041043467,
          0.032497544
        ],
        [
          0.032041281,
          0.033687352,
          0.033005369,
          0.036079523,
          0.034474625,
          0.032581787,
          0.033330932,
          0.032117612,
          0.032996852,
          0.03510688
        ]
      ],
      "candidate": [
        [
          0.032556755,
          0.033273143,
          0.03343614,
          0.032734904,
          0.033208952,
          0.033980599,
          0.039086942,
          0.039256246,
          0.036936427,
          0.032278865
        ],
        [
          0.031860884,
          0.032828605,
          0.032972932,
          0.033257572,
          0.034713915,
          0.032920481,
          0.033675393,
          0.034063156,
          0.032355617,
          0.033039176
        ]
      ],
      "result": {
        "head_p50": 0.033233262,
        "head_p95": 0.039095407199999994,
        "head_min": 0.031860884,
        "head_max": 0.039256246,
        "rows_per_second": 3009033.5399516304,
        "samples": 20,
        "base_p50": 0.033612685,
        "change_percent": -1.1288089600696938,
        "round_changes": [
          -0.7313423328162094,
          -0.3306461344018352
        ],
        "round_min_changes": [
          1.0500401240367108,
          -0.5630143189343739
        ],
        "round_intervals": [
          {
            "median": -0.7313423328162094,
            "low": -5.281259056236731,
            "high": 9.69952812396484,
            "coverage": 0.978515625
          },
          {
            "median": -0.3306461344018352,
            "low": -5.889740130709409,
            "high": 1.0395194100311311,
            "coverage": 0.978515625
          }
        ],
        "verdict": "inconclusive"
      }
    }
  ]
}
```

### Every final worker's oracle, RSS and input attestation

The two rounds have separate workers. Each worker contains one excluded
warmup and ten validated measured outputs; HWM includes setup and full-output
oracles outside the timer. Native stage RSS and these Python worker RSS values
are different measurement boundaries.

```json
{
  "status": "verified",
  "workers": 24,
  "timed_samples": 240,
  "warmup_samples": 24,
  "raw_responses": 312,
  "raw_results_sha256": "0b3c6abf75d32e40674d3f000178fa936edea8914025cd42e7a85b28c2e67459",
  "load_average_range": [
    0.38427734375,
    1.03564453125
  ],
  "post_timer_emitted_counts": [],
  "fixtures": {
    "engines/64000/calc-flow-stream/asof_join/batch-64000/overlap": {"source_scope": "ready-enqueue-to-arrow/bounded-feeds-v6", "source_root": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/final/sources/a2-candidate", "requested_batch_rows": 64000, "variant": "overlap", "input_rows": 64000, "entities": 64, "fixture_batch_rows": [64000], "stream_data_rows": {"reference.input": [64000], "quotes.input": [32000, 32000]}},
    "engines/64000/calc-flow-stream/asof_join/batch-64000/pipelined": {"source_scope": "ready-enqueue-to-arrow/bounded-feeds-v6", "source_root": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/final/sources/a2-candidate", "requested_batch_rows": 64000, "variant": "pipelined", "input_rows": 64000, "entities": 64, "fixture_batch_rows": [64000], "stream_data_rows": {"reference.input": [64000], "quotes.input": [32000, 32000]}},
    "engines/1000000/calc-flow-stream/asof_join/batch-64000/standard": {"source_scope": "ready-enqueue-to-arrow/bounded-feeds-v6", "source_root": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/final/sources/a2-candidate", "requested_batch_rows": 64000, "variant": "standard", "input_rows": 1000000, "entities": 64, "fixture_batch_rows": [64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 40000], "stream_data_rows": {"reference.input": [64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 40000], "quotes.input": [64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 40000]}},
    "engines/1000000/calc-flow-stream/projection/batch-64000/standard": {"source_scope": "ready-enqueue-to-arrow/bounded-feeds-v6", "source_root": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/final/sources/a2-candidate", "requested_batch_rows": 64000, "variant": "standard", "input_rows": 1000000, "entities": 64, "fixture_batch_rows": [64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 40000], "stream_data_rows": {"input": [64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 64000, 40000]}},
    "engines/100000/calc-flow-stream/asof_join/batch-1024/standard": {"source_scope": "ready-enqueue-to-arrow/bounded-feeds-v6", "source_root": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/final/sources/a2-candidate", "requested_batch_rows": 1024, "variant": "standard", "input_rows": 100000, "entities": 64, "fixture_batch_rows": [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 672], "stream_data_rows": {"reference.input": [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 672], "quotes.input": [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 672]}},
    "engines/100000/calc-flow-stream/projection/batch-1024/standard": {"source_scope": "ready-enqueue-to-arrow/bounded-feeds-v6", "source_root": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/final/sources/a2-candidate", "requested_batch_rows": 1024, "variant": "standard", "input_rows": 100000, "entities": 64, "fixture_batch_rows": [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 672], "stream_data_rows": {"input": [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 672]}}
  },
  "workers_evidence": [
    {"case": "engines/64000/calc-flow-stream/asof_join/batch-64000/overlap", "round": 0, "side": "baseline", "pid": 1178909, "measured_samples": 10, "warmups": 1, "oracle_rows": 64000, "all_oracles_passed": true, "prepare_rss_bytes": 369971200, "final_rss_bytes": 440512512, "hwm_bytes": 451194880, "fixture_id": "engines/64000/calc-flow-stream/asof_join/batch-64000/overlap"},
    {"case": "engines/64000/calc-flow-stream/asof_join/batch-64000/overlap", "round": 0, "side": "candidate", "pid": 1178911, "measured_samples": 10, "warmups": 1, "oracle_rows": 64000, "all_oracles_passed": true, "prepare_rss_bytes": 371515392, "final_rss_bytes": 439762944, "hwm_bytes": 448970752, "fixture_id": "engines/64000/calc-flow-stream/asof_join/batch-64000/overlap"},
    {"case": "engines/64000/calc-flow-stream/asof_join/batch-64000/overlap", "round": 1, "side": "baseline", "pid": 1179940, "measured_samples": 10, "warmups": 1, "oracle_rows": 64000, "all_oracles_passed": true, "prepare_rss_bytes": 373252096, "final_rss_bytes": 442511360, "hwm_bytes": 451907584, "fixture_id": "engines/64000/calc-flow-stream/asof_join/batch-64000/overlap"},
    {"case": "engines/64000/calc-flow-stream/asof_join/batch-64000/overlap", "round": 1, "side": "candidate", "pid": 1179942, "measured_samples": 10, "warmups": 1, "oracle_rows": 64000, "all_oracles_passed": true, "prepare_rss_bytes": 367865856, "final_rss_bytes": 433381376, "hwm_bytes": 442507264, "fixture_id": "engines/64000/calc-flow-stream/asof_join/batch-64000/overlap"},
    {"case": "engines/64000/calc-flow-stream/asof_join/batch-64000/pipelined", "round": 0, "side": "baseline", "pid": 1180982, "measured_samples": 10, "warmups": 1, "oracle_rows": 64000, "all_oracles_passed": true, "prepare_rss_bytes": 370647040, "final_rss_bytes": 437293056, "hwm_bytes": 446914560, "fixture_id": "engines/64000/calc-flow-stream/asof_join/batch-64000/pipelined"},
    {"case": "engines/64000/calc-flow-stream/asof_join/batch-64000/pipelined", "round": 0, "side": "candidate", "pid": 1180984, "measured_samples": 10, "warmups": 1, "oracle_rows": 64000, "all_oracles_passed": true, "prepare_rss_bytes": 364863488, "final_rss_bytes": 434982912, "hwm_bytes": 444620800, "fixture_id": "engines/64000/calc-flow-stream/asof_join/batch-64000/pipelined"},
    {"case": "engines/64000/calc-flow-stream/asof_join/batch-64000/pipelined", "round": 1, "side": "baseline", "pid": 1182013, "measured_samples": 10, "warmups": 1, "oracle_rows": 64000, "all_oracles_passed": true, "prepare_rss_bytes": 373264384, "final_rss_bytes": 458117120, "hwm_bytes": 458457088, "fixture_id": "engines/64000/calc-flow-stream/asof_join/batch-64000/pipelined"},
    {"case": "engines/64000/calc-flow-stream/asof_join/batch-64000/pipelined", "round": 1, "side": "candidate", "pid": 1182015, "measured_samples": 10, "warmups": 1, "oracle_rows": 64000, "all_oracles_passed": true, "prepare_rss_bytes": 354983936, "final_rss_bytes": 433553408, "hwm_bytes": 433889280, "fixture_id": "engines/64000/calc-flow-stream/asof_join/batch-64000/pipelined"},
    {"case": "engines/1000000/calc-flow-stream/asof_join/batch-64000/standard", "round": 0, "side": "baseline", "pid": 1183045, "measured_samples": 10, "warmups": 1, "oracle_rows": 1000000, "all_oracles_passed": true, "prepare_rss_bytes": 571412480, "final_rss_bytes": 734896128, "hwm_bytes": 796831744, "fixture_id": "engines/1000000/calc-flow-stream/asof_join/batch-64000/standard"},
    {"case": "engines/1000000/calc-flow-stream/asof_join/batch-64000/standard", "round": 0, "side": "candidate", "pid": 1183047, "measured_samples": 10, "warmups": 1, "oracle_rows": 1000000, "all_oracles_passed": true, "prepare_rss_bytes": 582393856, "final_rss_bytes": 721203200, "hwm_bytes": 783015936, "fixture_id": "engines/1000000/calc-flow-stream/asof_join/batch-64000/standard"},
    {"case": "engines/1000000/calc-flow-stream/asof_join/batch-64000/standard", "round": 1, "side": "baseline", "pid": 1184078, "measured_samples": 10, "warmups": 1, "oracle_rows": 1000000, "all_oracles_passed": true, "prepare_rss_bytes": 588271616, "final_rss_bytes": 720465920, "hwm_bytes": 782401536, "fixture_id": "engines/1000000/calc-flow-stream/asof_join/batch-64000/standard"},
    {"case": "engines/1000000/calc-flow-stream/asof_join/batch-64000/standard", "round": 1, "side": "candidate", "pid": 1184080, "measured_samples": 10, "warmups": 1, "oracle_rows": 1000000, "all_oracles_passed": true, "prepare_rss_bytes": 577921024, "final_rss_bytes": 744329216, "hwm_bytes": 806141952, "fixture_id": "engines/1000000/calc-flow-stream/asof_join/batch-64000/standard"},
    {"case": "engines/1000000/calc-flow-stream/projection/batch-64000/standard", "round": 0, "side": "baseline", "pid": 1185135, "measured_samples": 10, "warmups": 1, "oracle_rows": 1000000, "all_oracles_passed": true, "prepare_rss_bytes": 497152000, "final_rss_bytes": 500142080, "hwm_bytes": 532013056, "fixture_id": "engines/1000000/calc-flow-stream/projection/batch-64000/standard"},
    {"case": "engines/1000000/calc-flow-stream/projection/batch-64000/standard", "round": 0, "side": "candidate", "pid": 1185137, "measured_samples": 10, "warmups": 1, "oracle_rows": 1000000, "all_oracles_passed": true, "prepare_rss_bytes": 484147200, "final_rss_bytes": 487043072, "hwm_bytes": 518979584, "fixture_id": "engines/1000000/calc-flow-stream/projection/batch-64000/standard"},
    {"case": "engines/1000000/calc-flow-stream/projection/batch-64000/standard", "round": 1, "side": "baseline", "pid": 1185988, "measured_samples": 10, "warmups": 1, "oracle_rows": 1000000, "all_oracles_passed": true, "prepare_rss_bytes": 490876928, "final_rss_bytes": 491520000, "hwm_bytes": 523395072, "fixture_id": "engines/1000000/calc-flow-stream/projection/batch-64000/standard"},
    {"case": "engines/1000000/calc-flow-stream/projection/batch-64000/standard", "round": 1, "side": "candidate", "pid": 1185990, "measured_samples": 10, "warmups": 1, "oracle_rows": 1000000, "all_oracles_passed": true, "prepare_rss_bytes": 499056640, "final_rss_bytes": 503984128, "hwm_bytes": 535920640, "fixture_id": "engines/1000000/calc-flow-stream/projection/batch-64000/standard"},
    {"case": "engines/100000/calc-flow-stream/asof_join/batch-1024/standard", "round": 0, "side": "baseline", "pid": 1186840, "measured_samples": 10, "warmups": 1, "oracle_rows": 100000, "all_oracles_passed": true, "prepare_rss_bytes": 361897984, "final_rss_bytes": 365719552, "hwm_bytes": 373293056, "fixture_id": "engines/100000/calc-flow-stream/asof_join/batch-1024/standard"},
    {"case": "engines/100000/calc-flow-stream/asof_join/batch-1024/standard", "round": 0, "side": "candidate", "pid": 1186842, "measured_samples": 10, "warmups": 1, "oracle_rows": 100000, "all_oracles_passed": true, "prepare_rss_bytes": 353964032, "final_rss_bytes": 358514688, "hwm_bytes": 364531712, "fixture_id": "engines/100000/calc-flow-stream/asof_join/batch-1024/standard"},
    {"case": "engines/100000/calc-flow-stream/asof_join/batch-1024/standard", "round": 1, "side": "baseline", "pid": 1187717, "measured_samples": 10, "warmups": 1, "oracle_rows": 100000, "all_oracles_passed": true, "prepare_rss_bytes": 355864576, "final_rss_bytes": 362397696, "hwm_bytes": 371953664, "fixture_id": "engines/100000/calc-flow-stream/asof_join/batch-1024/standard"},
    {"case": "engines/100000/calc-flow-stream/asof_join/batch-1024/standard", "round": 1, "side": "candidate", "pid": 1187719, "measured_samples": 10, "warmups": 1, "oracle_rows": 100000, "all_oracles_passed": true, "prepare_rss_bytes": 359583744, "final_rss_bytes": 366174208, "hwm_bytes": 375414784, "fixture_id": "engines/100000/calc-flow-stream/asof_join/batch-1024/standard"},
    {"case": "engines/100000/calc-flow-stream/projection/batch-1024/standard", "round": 0, "side": "baseline", "pid": 1188594, "measured_samples": 10, "warmups": 1, "oracle_rows": 100000, "all_oracles_passed": true, "prepare_rss_bytes": 374157312, "final_rss_bytes": 383365120, "hwm_bytes": 389308416, "fixture_id": "engines/100000/calc-flow-stream/projection/batch-1024/standard"},
    {"case": "engines/100000/calc-flow-stream/projection/batch-1024/standard", "round": 0, "side": "candidate", "pid": 1188596, "measured_samples": 10, "warmups": 1, "oracle_rows": 100000, "all_oracles_passed": true, "prepare_rss_bytes": 372490240, "final_rss_bytes": 384446464, "hwm_bytes": 390414336, "fixture_id": "engines/100000/calc-flow-stream/projection/batch-1024/standard"},
    {"case": "engines/100000/calc-flow-stream/projection/batch-1024/standard", "round": 1, "side": "baseline", "pid": 1189447, "measured_samples": 10, "warmups": 1, "oracle_rows": 100000, "all_oracles_passed": true, "prepare_rss_bytes": 373313536, "final_rss_bytes": 384671744, "hwm_bytes": 390623232, "fixture_id": "engines/100000/calc-flow-stream/projection/batch-1024/standard"},
    {"case": "engines/100000/calc-flow-stream/projection/batch-1024/standard", "round": 1, "side": "candidate", "pid": 1189449, "measured_samples": 10, "warmups": 1, "oracle_rows": 100000, "all_oracles_passed": true, "prepare_rss_bytes": 375332864, "final_rss_bytes": 383066112, "hwm_bytes": 389025792, "fixture_id": "engines/100000/calc-flow-stream/projection/batch-1024/standard"}
  ]
}
```

### Exact final ready-worker source

The original maintained replay worker runs the retained interval case;
this wrapper runs ordinary ready and declared pipelined cases. Status capture
and proof validation occur after the maintained timer stops.

```python
"""Owned source snapshots, explicit layout and untimed status evidence."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa

from benchmarks import engine_comparison, engine_stream
from calc_flow import Batch, Cursor, Data
from scripts.benchmark_suite import catalog, worker

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--batch-rows", type=int, required=True)
parser.add_argument(
    "--variant", choices=("standard", "overlap", "pipelined"), default="standard"
)
args = parser.parse_args()
catalog.BATCH_ROWS = args.batch_rows
engine_comparison.BATCH_ROWS = args.batch_rows
engine_stream.BATCH_ROWS = args.batch_rows
last_join_status = None
causal_join_status = None
last_job = None
original_measure = engine_stream._measure_ready
original_quote_progress = engine_stream._wait_static_quote_progress


async def attested_quote_progress(job, expected_rows):
    """Keep the pre-wait snapshot and verify one causally complete snapshot."""

    global last_join_status, causal_join_status
    last_join_status = engine_stream._static_join_status(job)
    status = await original_quote_progress(job, expected_rows)
    if status["emitted_match_rows"] < expected_rows or any(
        status["left"][field]
        for field in ("retained_rows", "retained_bytes", "evicted_rows")
    ):
        raise ValueError("causally complete static Join state proof differs")
    causal_join_status = status
    return status


if not hasattr(engine_stream, "_wait_static_quote_progress"):
    raise ValueError("J1 requires the sealed causal-status common harness")
engine_stream._wait_static_quote_progress = attested_quote_progress


async def diagnostic_measure(sources, sink, streams, job, *, static_join=False):
    global last_join_status, causal_join_status, last_job
    last_join_status = causal_join_status = None
    last_job = job
    if args.variant == "standard":
        result = await original_measure(
            sources, sink, streams, job, static_join=static_join
        )
    else:
        await asyncio.wait_for(
            asyncio.gather(*(source.ready.wait() for source in sources.values())),
            timeout=30,
        )
        engine_stream._require_ready_sources(sources, sink)
        started = time.perf_counter_ns()
        for name in ("reference.input", "quotes.input"):
            for event in streams[name]:
                await sources[name].push(event)
        await asyncio.wait_for(sink.complete.wait(), timeout=600)
        if sink.rows != sink.expected_rows:
            raise ValueError("pipelined diagnostic output count differs")
        table = pa.concat_tables(sink.tables)
        result = table, (time.perf_counter_ns() - started) / 1e9
    if static_join and (last_join_status is None or causal_join_status is None):
        raise ValueError("static Join lacks post-timer and causal status proof")
    return result


engine_stream._measure_ready = diagnostic_measure
OriginalEngineCase = engine_comparison.EngineCase


class DiagnosticEngineCase(OriginalEngineCase):
    def __init__(self, case, root):
        super().__init__(case, root)
        if args.variant == "standard":
            return
        if case["scenario"] != "asof_join" or case["rows"] != 64_000:
            raise ValueError("pipelined variants require the declared 64k ASOF fixture")
        table = self.data.table
        if args.variant == "overlap":
            parts = tuple(
                table.take(pa.array(np.arange(parity, table.num_rows, 2)))
                for parity in (0, 1)
            )
        else:
            parts = (table.slice(0, 32_000), table.slice(32_000))
        reference = engine_stream.stream_events(table, self.data.entities)
        left = tuple(
            Data(
                Batch.from_pyarrow(part),
                Cursor((index + 1).to_bytes(8, "big"), {"rows": (index + 1) * 32_000}),
            )
            for index, part in enumerate(parts)
        )
        self.streams = {
            "reference.input": reference,
            "quotes.input": (*left, reference[-2], None),
        }

    def _stream(self):
        # Extra dimensions are evidence metadata; use the ready runner rather
        # than the maintained replay/checkpoint variant for these cases.
        self.count += 1
        plan = engine_stream.stream_plan(
            self.case["scenario"],
            self.data.table,
            self.data.dimension,
            batch_rows=args.batch_rows,
        )
        return self.loop.run_until_complete(
            engine_stream.run_stream(
                plan,
                self.streams,
                self.root / f"sample-{self.count}",
                self.expected.num_rows,
                static_join=self.case["scenario"] == "join",
            )
        )

    def validate(self, result):
        correctness = super().validate(result)
        if args.variant != "standard" and not result["sequence"].cast(
            self.expected["sequence"].type
        ).equals(self.expected["sequence"]):
            raise ValueError("pipelined ASOF output changed canonical row order")
        return correctness

    def sample(self):
        sample = super().sample()
        if self.case["scenario"] == "join":
            statuses = tuple(last_job.status()["stream_joins"].values())
            if len(statuses) != 1:
                raise ValueError("completed static Join lacks one status")
            final = statuses[0]
            if final["emitted_match_rows"] != self.expected.num_rows:
                raise ValueError("completed Join emitted count differs from output")
            sample["completed_join_status"] = final
        return sample


engine_comparison.EngineCase = DiagnosticEngineCase
dispatch = worker.dispatch


def attested_dispatch(message, active, root):
    response, active = dispatch(message, active, root)
    if (
        message["operation"] in ("prepare", "sample")
        and active.case["scenario"] == "join"
    ):
        sample = response["warmup"] if message["operation"] == "prepare" else response
        sample["post_timer_join_status"] = last_join_status
        sample["causal_join_status"] = causal_join_status
    if message["operation"] == "prepare":
        attestation = {
            "source_scope": catalog.STREAM_SCOPE,
            "source_root": str(Path(catalog.__file__).resolve().parents[2]),
            "requested_batch_rows": args.batch_rows,
            "variant": args.variant,
            "input_rows": active.data.table.num_rows,
            "entities": active.data.entities,
            "fixture_batch_rows": [
                batch.num_rows for batch in active.data.table.to_batches()
            ],
            "stream_data_rows": {
                name: [
                    event.batch.num_rows for event in events if isinstance(event, Data)
                ]
                for name, events in getattr(active, "streams", {}).items()
            },
        }
        (root / "workload.json").write_text(json.dumps(attestation, indent=2) + "\n")
    return response, active


worker.dispatch = attested_dispatch
worker.main(args.root)
```
