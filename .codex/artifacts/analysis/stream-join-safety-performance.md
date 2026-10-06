# Stream Join J1 performance evidence

## Result and measured stage

The J1.1–J1.5 stage makes warmed no-expiry and sparse-expiry Join progress
substantially faster at 1M and 4M retained rows. All four compatible native
operator comparisons satisfy the two-round paired improvement gate. Actual
resident memory increases; logical FlatV1 charges stay fixed.

These measurements compare the clean progress baseline
`94383f5bb8ef9050113726fab39e8c379019af36` against the clean J1 stage
`fb7b0157f8607a205ea38b95aa84cce8f5fa661b`. They are stage evidence for
[PR #369](https://github.com/wegamekinglc/calc-flow/pull/369), rather than a
measurement of its later integrated shutdown/runtime fixes. The final native
wheel and final common-harness E2E comparisons remain pending. No A2, J1.6,
checkpoint/restore, full-state expiration, or whole-pipeline gain is inferred
from these operator timings.

## Timings and uncertainty

Units are microseconds. No-op samples report the mean duration of eight
increasing watermark callbacks within one clock interval; sparse samples
report one callback expiring exactly 64 rows. Both refs run the same fixture,
thread count, release configuration and oracle. Setup, state insertion,
warmups, status reads, RSS reads and payload validation are outside timing.

|  Case                     |  Base p50 µs  |  Head p50 µs  |  Head p95 µs  |  p50 change   |  Round 1 change interval %  |  Round 2 change interval %  |  Verdict   |
|---------------------------|---------------|---------------|---------------|---------------|-----------------------------|-----------------------------|------------|
|  retained-1000000/noop    |  17140.121    |  0.401938     |  0.577500     |  -99.997655%  |  [-99.998898, -99.996966]   |  [-99.998422, -99.997177]   |  improved  |
|  retained-1000000/sparse  |  22197.261    |  21.998500    |  29.922900    |  -99.900895%  |  [-99.908032, -99.874825]   |  [-99.911554, -99.896421]   |  improved  |
|  retained-4000000/noop    |  76659.634    |  0.503125     |  0.587250     |  -99.999344%  |  [-99.999632, -99.999260]   |  [-99.999608, -99.999229]   |  improved  |
|  retained-4000000/sparse  |  95159.790    |  23.096500    |  29.582700    |  -99.975729%  |  [-99.976451, -99.973784]   |  [-99.977808, -99.973081]   |  improved  |

Each case has two independent rounds of ten alternating AB/BA pairs: 20
observations per version. Both modes share the same prepared worker in each
pair; the modes are not independent workloads. Native processes run
sequentially, with only one retained state resident at a time. Every worker
exits before the next version starts. The 1M preflight is excluded from the
timing arrays. All attempts are retained; there is no minimum-based selection.

The maintained suite computes per-pair percentage changes and an exact
order-statistic interval for their median. With ten pairs, the interval has
97.8515625% nominal coverage, at least the requested 95%. Improvement requires
both upper bounds below −5%; regression requires both lower bounds above +5%.
The p50 change is a ratio of aggregate medians and need not equal either
round's median pair change. The sub-microsecond no-op values are averages of
eight calls and include callback/future/loop work; timer granularity and shared
host scheduling limit their interpretation. These are not job throughput
figures or claims that an application becomes thousands of times faster.

The candidate's observed 1M→4M p50 ratio is 1.252× for no-op
progress and 1.050× for fixed-64 expiration. This resolves
bounded progress on the declared retained fixture. It is a scaling diagnostic
across two sizes, not a compatible same-workload regression classification.

## Workload, oracle and boundaries

The public `StreamJoinOperator` retains N left rows, one unique eight-byte
UTF-8 key per row, equal far-future timestamps, and amount 7. The right side
starts empty. Seed insertion uses 64k-row batches. Before each measurement,
untimed warmups advance right progress and expire 64 freshly inserted hot
rows. The timed no-op uses eight monotonically increasing right watermarks
below every cold row's expiry. The timed sparse update follows untimed
insertion of another 64 hot rows and expires exactly those identities.
Checkpoint capture and compaction are absent; pending cold upserts are retained.

Each preparation and sample asserts N cold rows remain, right retained rows
and emitted output are zero, logical charges are exactly 128×N bytes, and no
state/match limit fails. Sparse input is proved to contain N+64 rows before
progress; cumulative eviction is 64 after warmup and 128 after measurement.
After timing, 64 right-side canary keys spanning the retained key range produce
exactly 64 matches. The Rust oracle checks both keys, both timestamps, both
amounts, uniqueness and the expected marker set. This is a complete oracle
for the 64-row canary and zero-output timed callbacks; it does not validate
every cold row's payload individually or checkpoint encoding.

The independent raw checker verified 80 measured
workers plus two preflight workers, 410 raw IPC
responses, unique process IDs, sequential AB/BA order, exit code zero, every
state/canary invariant, every raw timing and every paired interval. It reads
individual IPC/result/exit files and recomputes intervals without importing
the suite statistics implementation.

## Memory tradeoff

|  Retained rows  |  Version          |  Logical FlatV1 bytes  |  Prepare RSS p50 MiB  |  Worker HWM p50 MiB  |  Worker HWM max MiB  |
|-----------------|-------------------|------------------------|-----------------------|----------------------|----------------------|
|  1,000,000      |  baseline         |  128,000,000           |  825.322              |  883.736             |  887.152             |
|  1,000,000      |  candidate        |  128,000,000           |  969.787              |  1054.881            |  1057.070            |
|  1,000,000      |  head minus base  |  0                     |  +144.465             |  —                   |  —                   |
|  4,000,000      |  baseline         |  512,000,000           |  3167.139             |  3336.064            |  3338.027            |
|  4,000,000      |  candidate        |  512,000,000           |  3783.084             |  4022.705            |  4024.789            |
|  4,000,000      |  head minus base  |  0                     |  +615.945             |  —                   |  —                   |

These are native process measurements, including 32 Tokio worker threads,
Arrow data, retained state, dirty log and allocation retention. The preparation
RSS is read after seeding and warmup. HWM includes the post-timer canary,
which can build the retained-key cache; it is not the live-state logical
charge. No Python fixture or accumulated 11M-row output is resident in this
operator probe. The expiry/upsert indexes consume additional memory
proportional to retained rows and pending changes. This evidence supports a
CPU/memory tradeoff, not an RSS reduction or the accounting of individual
index allocations.

The 1M preflight's peak was 1,102,770,176
bytes. The conservative 4M projection, including canary peak, was
4,947,951,616 bytes against
30,359,822,336 available bytes and a 70% limit.
The observed maximum was 4,220,297,216 bytes; minimum available memory over
all responses was 26,372,534,272 bytes. Only one state was resident at once.
The preflight forecast was 304.7
seconds; the complete run took approximately 339 seconds including preflight,
seeding, canary validation and process cleanup. Untimed lifecycle costs are
diagnostic and are not counted as streaming throughput.

## Target status and coverage

- J1 bounded retained-progress scaling: the observed 1.252×
  and 1.050× are within the specification's 4.4× planning
  limit for this fixture. Whole retained-stream 1M→4M throughput scaling
  remains unverified.
- Zero no-op retained visits and expired-only work are structural correctness
  targets. The J1 source's focused instrumented tests record these properties;
  this release probe does not instrument visit counts. Timings alone do not
  prove a zero count.
- Final static Join 1M and retained interval 100k/1024 E2E comparisons remain
  pending the final sealed wheel, reviewed causal-status harness and quiet
  window. Immediate timer-cutoff status is retained as a lag diagnostic;
  causal verification must wait for emitted output and check the same snapshot.
- J1.6 compaction latency, capture/restore cost, allocation counts, cold runs,
  and scheduled full-suite/lifecycle evidence remain unverified. The existing
  10k Rust benchmark cannot replace these retained-size cases.
- Highest-priority additional coverage is the fixed retained-progress fixture
  in a maintained Rust operator benchmark. Follow with actual expired-density
  sweeps, checkpoint/capture pressure, and matched final-wheel retained-stream
  oracles. Preserve native RSS and worker peaks separately from logical gauges.

## Environment and provenance

The team-coordinated quiet run was on WSL2, Intel i9-13900HX, 32 logical CPUs,
affinity 0–31 and 32 Tokio threads, from 2026-10-06T13:42:19.570989+00:00 to
2026-10-06T13:47:58.240853+00:00. Linux load-average samples ranged from 0.335 to
1.481. Team builds/tests were paused; quiet coordination does not prove a
dedicated bare-metal host or eliminate all Windows/WSL noise. CPU governor
and power mode were not exposed. The parent launched the unchanged driver;
the agent performed only lightweight source preparation during the run.

The native probe links against clean release core rlibs built with
`cargo build --release --locked -p calc-flow --lib`, no crate features, and
Rust 1.88.0. The Python 3.13.9 controller is not timed. These are sealed native
release-core comparisons, distinct from the separately prepared abi3-py313
wheels. Identical probe source, Cargo lock, compiler, DataFusion/Tokio/Serde
rlib hashes and machine fingerprints were checked. Each runtime rlib was
copied out of the mutable warm cache; rebuilt baseline bytes matched the
original linked hash. Source trees stayed clean and touched source mtimes
forced correct cache freshness.

|  Identity                |  Baseline                                                          |  Candidate                                                         |
|--------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|
|  Git commit              |  94383f5bb8ef9050113726fab39e8c379019af36                          |  fb7b0157f8607a205ea38b95aa84cce8f5fa661b                          |
|  Clean source SHA-256    |  d80d94feee83cf737ef3a67399bc4b939be9823a99d9c5e7b8db70a130c8a93e  |  d8e09f9cb83aacf50938171e71f87ac68858f6a084fb1fd6d3ea0c5e8d4cad8c  |
|  Probe binary SHA-256    |  3b53ebb6b7009975c2041a134ad360b6584b7f2be29cf49c76efc7f9c383b252  |  92a17e30e11e4a1030ca49275b9ea8fe56a7ff420739b92b2082f224db8ce8d3  |
|  calc-flow rlib SHA-256  |  685d4ff5bc5fb288c83972f969d4655dcbb35eea0c550023f136fc6e30774c6c  |  e35f2634a9a1b63b9346b247c2e1f5ef97220e15847116ee1aa510524f28cdcf  |

- Machine SHA-256: `868afb7a001ef232d3d03208f8d4774ccb0a3d5825b29c7224d6b6d5045060bd`.
- Dependency SHA-256: `623aae9cd482af680c615a6bfb992e95875fc1f91c171507a984c768ba6adf73`.
- Probe fixture SHA-256: `e33e05eb9829beed2817107dfc5365fa985862ba234661255f8b308aad807b05`.
- Measurement driver SHA-256: `41696bca13f8292c39ca3e15c6c3cb24268fff812abbca71784c9e43c0a58cbd`.
- Cargo lock SHA-256: `84789d29f2cb7342b0297169caf03c8f14227645656d34956ed9bc9aaa0c4840`.
- Raw result SHA-256: `59f0d72346279c7cfa37b84a19e882f0d22977f03c15c0cedad30714731a932a`.
- Raw evidence archive SHA-256: `55301bdf8ea1293f5bccc06cc5560224501f63647638fbedd4dc6d38efdfde91`, 265,403 bytes.

The archive at
`target/issue363-join-safety-perf/issue363-join-safety-native-evidence.tar.gz`
contains raw IPC/results/exits, build/release metadata, fixture, driver and
checker. It excludes wheels, binaries, rlibs and this subsequently written
Markdown report. Local artifacts are ignored by Git; the appendices below
embed exact arrays, fingerprints, proof summaries and source so PR readers
can independently inspect the measurement without those ignored paths.

Raw directories are `target/issue363-join-safety-perf/native-progress` and
`target/issue363-join-safety-perf/micro-releases`. Run the lightweight checker:

```bash
python target/issue363-join-safety-perf/verify_native_progress.py
```

Overall verdict for the sealed J1 stage's tested progress paths: confirmed
improvement with increased RSS. Final integrated E2E verdict remains pending.

## Appendix A: exact paired arrays and machine metadata

Durations are seconds; no-op arrays contain eight-call means. The arrays are
aligned by round and pair index. The result fields can be recomputed from the
arrays using the repository's paired interval contract.

```json
{
  "started_utc": "2026-10-06T13:42:19.570989+00:00",
  "finished_utc": "2026-10-06T13:47:58.240853+00:00",
  "machine": {
    "platform": "Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.43",
    "uname": [
      "Linux",
      "chengli-i9",
      "5.15.167.4-microsoft-standard-WSL2",
      "#1 SMP Tue Nov 5 00:21:55 UTC 2024",
      "x86_64",
      ""
    ],
    "cpu_model": "13th Gen Intel(R) Core(TM) i9-13900HX",
    "cpu_count": 32,
    "affinity": [
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
    "governors": [],
    "thread_environment": {
      "OMP_NUM_THREADS": null,
      "OPENBLAS_NUM_THREADS": null,
      "MKL_NUM_THREADS": null,
      "TOKIO_WORKER_THREADS": null
    },
    "tokio_threads": 32,
    "python": "3.13.9 | packaged by Anaconda, Inc. | (main, Oct 21 2025, 19:16:10) [GCC 11.2.0]"
  },
  "machine_sha256": "868afb7a001ef232d3d03208f8d4774ccb0a3d5825b29c7224d6b6d5045060bd",
  "dependencies_sha256": "623aae9cd482af680c615a6bfb992e95875fc1f91c171507a984c768ba6adf73",
  "cases": [
    {
      "id": "native-progress/retained-1000000/noop",
      "rows": 1000000,
      "scope": "native-operator-progress/noop-v1",
      "workload_sha256": "0182d38790659d4b6fca3906079490b838044de4fd9dd7a5ec088bcb6b75d711",
      "comparison": "interleaved",
      "correctness": true,
      "baseline": [
        [
          0.017188804375,
          0.017056861125,
          0.017637020875,
          0.016588821125,
          0.016970811625,
          0.016456809375,
          0.017149641375,
          0.017341992625,
          0.017174219375,
          0.016737266875
        ],
        [
          0.0172660605,
          0.017856900125,
          0.0171306005,
          0.0169183405,
          0.016978606,
          0.017376998625,
          0.01715094875,
          0.01708569225,
          0.016584236,
          0.017567398375
        ]
      ],
      "candidate": [
        [
          5.215e-07,
          4.34e-07,
          2.99125e-07,
          2.49e-07,
          1.87e-07,
          3.42e-07,
          1.56625e-07,
          5.10625e-07,
          5.715e-07,
          3.9925e-07
        ],
        [
          3.25125e-07,
          4.99e-07,
          3.25e-07,
          6.915e-07,
          4.5125e-07,
          2.74125e-07,
          4.30375e-07,
          4.04625e-07,
          2.5425e-07,
          4.96e-07
        ]
      ],
      "result": {
        "head_p50": 4.019375e-07,
        "head_p95": 5.775000000000001e-07,
        "head_min": 1.56625e-07,
        "head_max": 6.915e-07,
        "rows_per_second": null,
        "samples": 20,
        "base_p50": 0.0171401209375,
        "change_percent": -99.9976549902917,
        "round_changes": [
          -99.99776821872679,
          -99.99756122701177
        ],
        "round_min_changes": [
          -99.99904826630465,
          -99.99846691761985
        ],
        "round_intervals": [
          {
            "median": -99.99776821872679,
            "low": -99.9988981080921,
            "high": -99.99696604843116,
            "coverage": 0.978515625
          },
          {
            "median": -99.99756122701177,
            "low": -99.99842248361806,
            "high": -99.99717658819245,
            "coverage": 0.978515625
          }
        ],
        "verdict": "improved"
      }
    },
    {
      "id": "native-progress/retained-1000000/sparse",
      "rows": 1000000,
      "scope": "native-operator-progress/sparse-v1",
      "workload_sha256": "049d06cac044483414f32412a6c541219c5ec0afeda51e2d0279c9c94a425659",
      "comparison": "interleaved",
      "correctness": true,
      "baseline": [
        [
          0.023865319,
          0.02357172,
          0.021712837,
          0.023173924,
          0.021940896,
          0.022706486,
          0.022076061,
          0.022182574,
          0.022215592,
          0.021809481
        ],
        [
          0.021987248,
          0.0230114,
          0.021920694,
          0.022634393,
          0.022211947,
          0.02245039,
          0.02149409,
          0.022084334,
          0.021799976,
          0.022803451
        ]
      ],
      "candidate": [
        [
          2.2387e-05,
          2.9506e-05,
          2.268e-05,
          2.1894e-05,
          2.4285e-05,
          2.087e-05,
          2.0303e-05,
          3.7844e-05,
          2.1437e-05,
          2.3481e-05
        ],
        [
          2.3709e-05,
          2.2164e-05,
          1.9388e-05,
          2.1861e-05,
          1.9253e-05,
          2.3254e-05,
          2.2103e-05,
          2.1757e-05,
          2.1529e-05,
          2.1471e-05
        ]
      ],
      "result": {
        "head_p50": 2.1998500000000003e-05,
        "head_p95": 2.9922900000000004e-05,
        "head_min": 1.9253e-05,
        "head_max": 3.7844e-05,
        "rows_per_second": null,
        "samples": 20,
        "base_p50": 0.0221972605,
        "change_percent": -99.90089542806419,
        "round_changes": [
          -99.89952518617656,
          -99.90244954652536
        ],
        "round_min_changes": [
          -99.90649310359582,
          -99.91042654050486
        ],
        "round_intervals": [
          {
            "median": -99.89952518617656,
            "low": -99.90803160038378,
            "high": -99.87482457792643,
            "coverage": 0.978515625
          },
          {
            "median": -99.90244954652536,
            "low": -99.91155389514583,
            "high": -99.89642050761701,
            "coverage": 0.978515625
          }
        ],
        "verdict": "improved"
      }
    },
    {
      "id": "native-progress/retained-4000000/noop",
      "rows": 4000000,
      "scope": "native-operator-progress/noop-v1",
      "workload_sha256": "397409ea7cc785bde750ba7666c1f4b75779c6905e4c94b3c17275efe824e591",
      "comparison": "interleaved",
      "correctness": true,
      "baseline": [
        [
          0.08060001125,
          0.079577026625,
          0.07861189975,
          0.07680460175,
          0.076842796875,
          0.07692383625,
          0.08407384475,
          0.074822713875,
          0.074321973375,
          0.075568285375
        ],
        [
          0.07511995725,
          0.0788049385,
          0.07651466625,
          0.074359592375,
          0.074474127875,
          0.0745882235,
          0.074122564375,
          0.077073444375,
          0.07795030575,
          0.074429704125
        ]
      ],
      "candidate": [
        [
          5.8375e-07,
          4.49375e-07,
          5.62375e-07,
          3.57375e-07,
          5.1425e-07,
          4.28375e-07,
          1.93375e-07,
          5.5375e-07,
          5.53125e-07,
          2.78125e-07
        ],
        [
          5.52875e-07,
          6.5375e-07,
          3.38e-07,
          4.92e-07,
          5.35e-07,
          3.69625e-07,
          2.5775e-07,
          3.02125e-07,
          5.565e-07,
          5.74e-07
        ]
      ],
      "result": {
        "head_p50": 5.03125e-07,
        "head_p95": 5.872500000000001e-07,
        "head_min": 1.93375e-07,
        "head_max": 6.5375e-07,
        "rows_per_second": null,
        "samples": 20,
        "base_p50": 0.076659634,
        "change_percent": -99.99934368979638,
        "round_changes": [
          -99.99938303606723,
          -99.999312216974
        ],
        "round_min_changes": [
          -99.99973981449736,
          -99.99965226513388
        ],
        "round_intervals": [
          {
            "median": -99.99938303606723,
            "low": -99.99963195539158,
            "high": -99.99925991724795,
            "coverage": 0.978515625
          },
          {
            "median": -99.999312216974,
            "low": -99.99960800376518,
            "high": -99.99922880252346,
            "coverage": 0.978515625
          }
        ],
        "verdict": "improved"
      }
    },
    {
      "id": "native-progress/retained-4000000/sparse",
      "rows": 4000000,
      "scope": "native-operator-progress/sparse-v1",
      "workload_sha256": "6faeea1dedeb777462ab179b4bd04bd7fec0c1eb4bdd03f9a902c805b0d88e67",
      "comparison": "interleaved",
      "correctness": true,
      "baseline": [
        [
          0.105934903,
          0.097526158,
          0.099106579,
          0.094760852,
          0.096750165,
          0.097625746,
          0.095247295,
          0.096263176,
          0.090818396,
          0.094591309
        ],
        [
          0.093480425,
          0.095490929,
          0.095851229,
          0.093321997,
          0.092407777,
          0.095072284,
          0.093197174,
          0.097709904,
          0.093358036,
          0.093209404
        ]
      ],
      "candidate": [
        [
          2.5408e-05,
          2.2966e-05,
          2.5982e-05,
          2.4068e-05,
          2.491e-05,
          1.9734e-05,
          2.2741e-05,
          2.3227e-05,
          2.2412e-05,
          2.9503e-05
        ],
        [
          2.284e-05,
          3.1097e-05,
          2.0986e-05,
          2.387e-05,
          2.4875e-05,
          2.2491e-05,
          2.3581e-05,
          2.1684e-05,
          2.1825e-05,
          2.2045e-05
        ]
      ],
      "result": {
        "head_p50": 2.3096499999999998e-05,
        "head_p95": 2.95827e-05,
        "head_min": 1.9734e-05,
        "head_max": 3.1097e-05,
        "rows_per_second": null,
        "samples": 20,
        "base_p50": 0.09515978950000001,
        "change_percent": -99.9757287189039,
        "round_changes": [
          -99.97559676766207,
          -99.97595517023694
        ],
        "round_min_changes": [
          -99.97827092211583,
          -99.97728979023054
        ],
        "round_intervals": [
          {
            "median": -99.97559676766207,
            "low": -99.97645144598026,
            "high": -99.97378377877416,
            "coverage": 0.978515625
          },
          {
            "median": -99.97595517023694,
            "low": -99.9778077767838,
            "high": -99.97308126998878,
            "coverage": 0.978515625
          }
        ],
        "verdict": "improved"
      }
    }
  ]
}
```

## Appendix B: every completed worker's proof and RSS

Each record's columns are declared below. Warmup and measured sparse eviction
counts are cumulative. Both samples and the pre-canary snapshot preserve
exact N/128×N left gauges, zero right rows and zero timed output; completed
canaries emit exactly 64 rows. `all_oracles_passed` is backed by the independent
raw verification described above. HWM includes the untimed canary.

```json
{
  "columns": ["phase", "round_zero_based", "pair_zero_based", "rows", "side", "pid", "flat_v1_charge_bytes", "evicted_after_sparse", "canary_output_rows", "prepare_rss_bytes", "worker_hwm_bytes", "all_oracles_passed"],
  "records": [
    ["preflight", -1, -1, 1000000, "baseline", 1149571, 128000000, 128, 64, 865861632, 927207424, true],
    ["preflight", -1, -1, 1000000, "candidate", 1149605, 128000000, 128, 64, 1012084736, 1102770176, true],
    ["measured", 0, 0, 1000000, "baseline", 1149639, 128000000, 128, 64, 865406976, 926527488, true],
    ["measured", 0, 0, 1000000, "candidate", 1149709, 128000000, 128, 64, 1016602624, 1106046976, true],
    ["measured", 0, 1, 1000000, "baseline", 1149778, 128000000, 128, 64, 866578432, 928325632, true],
    ["measured", 0, 1, 1000000, "candidate", 1149743, 128000000, 128, 64, 1018433536, 1107206144, true],
    ["measured", 0, 2, 1000000, "baseline", 1149812, 128000000, 128, 64, 869511168, 930246656, true],
    ["measured", 0, 2, 1000000, "candidate", 1149846, 128000000, 128, 64, 1019256832, 1108418560, true],
    ["measured", 0, 3, 1000000, "baseline", 1149914, 128000000, 128, 64, 866213888, 927244288, true],
    ["measured", 0, 3, 1000000, "candidate", 1149880, 128000000, 128, 64, 1017683968, 1105616896, true],
    ["measured", 0, 4, 1000000, "baseline", 1149948, 128000000, 128, 64, 865648640, 926666752, true],
    ["measured", 0, 4, 1000000, "candidate", 1149982, 128000000, 128, 64, 1016410112, 1106464768, true],
    ["measured", 0, 5, 1000000, "baseline", 1150050, 128000000, 128, 64, 863465472, 924549120, true],
    ["measured", 0, 5, 1000000, "candidate", 1150016, 128000000, 128, 64, 1015455744, 1106378752, true],
    ["measured", 0, 6, 1000000, "baseline", 1150086, 128000000, 128, 64, 865230848, 926871552, true],
    ["measured", 0, 6, 1000000, "candidate", 1150120, 128000000, 128, 64, 1016197120, 1106321408, true],
    ["measured", 0, 7, 1000000, "baseline", 1150189, 128000000, 128, 64, 865497088, 926687232, true],
    ["measured", 0, 7, 1000000, "candidate", 1150154, 128000000, 128, 64, 1016594432, 1105907712, true],
    ["measured", 0, 8, 1000000, "baseline", 1150223, 128000000, 128, 64, 865054720, 926130176, true],
    ["measured", 0, 8, 1000000, "candidate", 1150257, 128000000, 128, 64, 1017634816, 1106067456, true],
    ["measured", 0, 9, 1000000, "baseline", 1150326, 128000000, 128, 64, 865574912, 926576640, true],
    ["measured", 0, 9, 1000000, "candidate", 1150291, 128000000, 128, 64, 1017610240, 1105768448, true],
    ["measured", 1, 0, 1000000, "baseline", 1150381, 128000000, 128, 64, 865464320, 926478336, true],
    ["measured", 1, 0, 1000000, "candidate", 1150415, 128000000, 128, 64, 1018658816, 1107976192, true],
    ["measured", 1, 1, 1000000, "baseline", 1150483, 128000000, 128, 64, 865419264, 927137792, true],
    ["measured", 1, 1, 1000000, "candidate", 1150449, 128000000, 128, 64, 1016938496, 1106120704, true],
    ["measured", 1, 2, 1000000, "baseline", 1150525, 128000000, 128, 64, 865464320, 926539776, true],
    ["measured", 1, 2, 1000000, "candidate", 1150559, 128000000, 128, 64, 1017532416, 1106104320, true],
    ["measured", 1, 3, 1000000, "baseline", 1150627, 128000000, 128, 64, 865304576, 927084544, true],
    ["measured", 1, 3, 1000000, "candidate", 1150593, 128000000, 128, 64, 1016049664, 1105711104, true],
    ["measured", 1, 4, 1000000, "baseline", 1150661, 128000000, 128, 64, 863870976, 925704192, true],
    ["measured", 1, 4, 1000000, "candidate", 1150695, 128000000, 128, 64, 1016922112, 1106169856, true],
    ["measured", 1, 5, 1000000, "baseline", 1150780, 128000000, 128, 64, 865390592, 926662656, true],
    ["measured", 1, 5, 1000000, "candidate", 1150730, 128000000, 128, 64, 1016262656, 1105584128, true],
    ["measured", 1, 6, 1000000, "baseline", 1150848, 128000000, 128, 64, 863080448, 924258304, true],
    ["measured", 1, 6, 1000000, "candidate", 1150904, 128000000, 128, 64, 1016868864, 1106124800, true],
    ["measured", 1, 7, 1000000, "baseline", 1150972, 128000000, 128, 64, 865337344, 926683136, true],
    ["measured", 1, 7, 1000000, "candidate", 1150938, 128000000, 128, 64, 1016287232, 1106358272, true],
    ["measured", 1, 8, 1000000, "baseline", 1151006, 128000000, 128, 64, 865062912, 925868032, true],
    ["measured", 1, 8, 1000000, "candidate", 1151040, 128000000, 128, 64, 1016090624, 1106345984, true],
    ["measured", 1, 9, 1000000, "baseline", 1151109, 128000000, 128, 64, 867729408, 928743424, true],
    ["measured", 1, 9, 1000000, "candidate", 1151074, 128000000, 128, 64, 1018044416, 1105965056, true],
    ["measured", 0, 0, 4000000, "baseline", 1151143, 512000000, 128, 64, 3322597376, 3499384832, true],
    ["measured", 0, 0, 4000000, "candidate", 1151200, 512000000, 128, 64, 3967090688, 4218011648, true],
    ["measured", 0, 1, 4000000, "baseline", 1151292, 512000000, 128, 64, 3320176640, 3497160704, true],
    ["measured", 0, 1, 4000000, "candidate", 1151257, 512000000, 128, 64, 3966898176, 4218462208, true],
    ["measured", 0, 2, 4000000, "baseline", 1151329, 512000000, 128, 64, 3319136256, 3496292352, true],
    ["measured", 0, 2, 4000000, "candidate", 1151385, 512000000, 128, 64, 3967049728, 4218331136, true],
    ["measured", 0, 3, 4000000, "baseline", 1151461, 512000000, 128, 64, 3319373824, 3496357888, true],
    ["measured", 0, 3, 4000000, "candidate", 1151426, 512000000, 128, 64, 3966672896, 4218138624, true],
    ["measured", 0, 4, 4000000, "baseline", 1151518, 512000000, 128, 64, 3320926208, 3497877504, true],
    ["measured", 0, 4, 4000000, "candidate", 1151553, 512000000, 128, 64, 3966930944, 4218253312, true],
    ["measured", 0, 5, 4000000, "baseline", 1151626, 512000000, 128, 64, 3321008128, 3498237952, true],
    ["measured", 0, 5, 4000000, "candidate", 1151588, 512000000, 128, 64, 3965440000, 4216324096, true],
    ["measured", 0, 6, 4000000, "baseline", 1151660, 512000000, 128, 64, 3320737792, 3497590784, true],
    ["measured", 0, 6, 4000000, "candidate", 1151695, 512000000, 128, 64, 3966685184, 4218068992, true],
    ["measured", 0, 7, 4000000, "baseline", 1151781, 512000000, 128, 64, 3322454016, 3499610112, true],
    ["measured", 0, 7, 4000000, "candidate", 1151729, 512000000, 128, 64, 3967107072, 4217909248, true],
    ["measured", 0, 8, 4000000, "baseline", 1151818, 512000000, 128, 64, 3321077760, 3497930752, true],
    ["measured", 0, 8, 4000000, "candidate", 1151858, 512000000, 128, 64, 3967094784, 4218085376, true],
    ["measured", 0, 9, 4000000, "baseline", 1151931, 512000000, 128, 64, 3320979456, 3497832448, true],
    ["measured", 0, 9, 4000000, "candidate", 1151897, 512000000, 128, 64, 3969044480, 4220297216, true],
    ["measured", 1, 0, 4000000, "baseline", 1151965, 512000000, 128, 64, 3320848384, 3498045440, true],
    ["measured", 1, 0, 4000000, "candidate", 1152059, 512000000, 128, 64, 3966484480, 4218077184, true],
    ["measured", 1, 1, 4000000, "baseline", 1152129, 512000000, 128, 64, 3320819712, 3497299968, true],
    ["measured", 1, 1, 4000000, "candidate", 1152095, 512000000, 128, 64, 3966648320, 4217704448, true],
    ["measured", 1, 2, 4000000, "baseline", 1152234, 512000000, 128, 64, 3319242752, 3496300544, true],
    ["measured", 1, 2, 4000000, "candidate", 1152270, 512000000, 128, 64, 3966996480, 4218392576, true],
    ["measured", 1, 3, 4000000, "baseline", 1152340, 512000000, 128, 64, 3321253888, 3498565632, true],
    ["measured", 1, 3, 4000000, "candidate", 1152305, 512000000, 128, 64, 3966824448, 4218421248, true],
    ["measured", 1, 4, 4000000, "baseline", 1152377, 512000000, 128, 64, 3323121664, 3500175360, true],
    ["measured", 1, 4, 4000000, "candidate", 1152411, 512000000, 128, 64, 3967086592, 4218355712, true],
    ["measured", 1, 5, 4000000, "baseline", 1152485, 512000000, 128, 64, 3321131008, 3498188800, true],
    ["measured", 1, 5, 4000000, "candidate", 1152446, 512000000, 128, 64, 3966730240, 4217970688, true],
    ["measured", 1, 6, 4000000, "baseline", 1152519, 512000000, 128, 64, 3321131008, 3498188800, true],
    ["measured", 1, 6, 4000000, "candidate", 1152562, 512000000, 128, 64, 3966840832, 4218507264, true],
    ["measured", 1, 7, 4000000, "baseline", 1152631, 512000000, 128, 64, 3321339904, 3498569728, true],
    ["measured", 1, 7, 4000000, "candidate", 1152596, 512000000, 128, 64, 3966697472, 4217782272, true],
    ["measured", 1, 8, 4000000, "baseline", 1152666, 512000000, 128, 64, 3320991744, 3498233856, true],
    ["measured", 1, 8, 4000000, "candidate", 1152700, 512000000, 128, 64, 3966861312, 4218294272, true],
    ["measured", 1, 9, 4000000, "baseline", 1152770, 512000000, 128, 64, 3320815616, 3498188800, true],
    ["measured", 1, 9, 4000000, "candidate", 1152734, 512000000, 128, 64, 3966803968, 4218007552, true]
  ]
}
```

## Appendix C: original release manifests

```json
{
  "baseline": {
    "contract": "issue363-native-release-core-probe-v1",
    "source": {
      "git_sha": "94383f5bb8ef9050113726fab39e8c379019af36",
      "git_clean": true,
      "source_sha256": "d80d94feee83cf737ef3a67399bc4b939be9823a99d9c5e7b8db70a130c8a93e",
      "cargo_lock_sha256": "84789d29f2cb7342b0297169caf03c8f14227645656d34956ed9bc9aaa0c4840"
    },
    "build_profile": "release",
    "cargo_command": [
      "cargo",
      "build",
      "--release",
      "--locked",
      "--manifest-path",
      "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/sources/baseline/Cargo.toml",
      "-p",
      "calc-flow",
      "--lib",
      "--message-format=json"
    ],
    "rustc_command": [
      "rustc",
      "--edition=2024",
      "-C",
      "opt-level=3",
      "-D",
      "warnings",
      "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/progress_probe.rs",
      "-o",
      "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/micro-releases/baseline/progress-probe",
      "-L",
      "dependency=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/deps",
      "--extern",
      "calc_flow=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/libcalc_flow.rlib",
      "--extern",
      "datafusion=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/deps/libdatafusion-6add09e86c412710.rlib",
      "--extern",
      "tokio=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/deps/libtokio-d37ab8cdaa9b0ce5.rlib",
      "--extern",
      "serde_json=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/deps/libserde_json-efdfce692aae30c3.rlib"
    ],
    "rustc": "rustc 1.88.0 (6b00bc388 2025-06-23)\nbinary: rustc\ncommit-hash: 6b00bc3880198600130e1cf62b8f8a93494488cc\ncommit-date: 2025-06-23\nhost: x86_64-unknown-linux-gnu\nrelease: 1.88.0\nLLVM version: 20.1.5\n",
    "probe_source_sha256": "e33e05eb9829beed2817107dfc5365fa985862ba234661255f8b308aad807b05",
    "binary_sha256": "3b53ebb6b7009975c2041a134ad360b6584b7f2be29cf49c76efc7f9c383b252",
    "dependencies": {
      "calc_flow": {
        "path": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/micro-releases/baseline/libcalc_flow.rlib",
        "sha256": "685d4ff5bc5fb288c83972f969d4655dcbb35eea0c550023f136fc6e30774c6c",
        "source_cache_path": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/libcalc_flow.rlib"
      },
      "datafusion": {
        "path": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/deps/libdatafusion-6add09e86c412710.rlib",
        "sha256": "c8f51700d6e854e9f1744f087271cc23ad871b39402799e2f1274979fb25fce4"
      },
      "tokio": {
        "path": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/deps/libtokio-d37ab8cdaa9b0ce5.rlib",
        "sha256": "eebd8526f67beb27a1f54f470d96d7be7885f2444de7d15ef46932e386e26f57"
      },
      "serde_json": {
        "path": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/deps/libserde_json-efdfce692aae30c3.rlib",
        "sha256": "fdae5275f4a8f1fac3563c2dfdf3f531d6f8bafed31cb23744ccbf9f96f3ec1a"
      }
    },
    "core_preserve_build_json": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/micro-releases/baseline/cargo-preserve.jsonl"
  },
  "candidate": {
    "contract": "issue363-native-release-core-probe-v1",
    "source": {
      "git_sha": "fb7b0157f8607a205ea38b95aa84cce8f5fa661b",
      "git_clean": true,
      "source_sha256": "d8e09f9cb83aacf50938171e71f87ac68858f6a084fb1fd6d3ea0c5e8d4cad8c",
      "cargo_lock_sha256": "84789d29f2cb7342b0297169caf03c8f14227645656d34956ed9bc9aaa0c4840"
    },
    "build_profile": "release",
    "cargo_command": [
      "cargo",
      "build",
      "--release",
      "--locked",
      "--manifest-path",
      "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/sources/candidate/Cargo.toml",
      "-p",
      "calc-flow",
      "--lib",
      "--message-format=json"
    ],
    "rustc_command": [
      "rustc",
      "--edition=2024",
      "-C",
      "opt-level=3",
      "-D",
      "warnings",
      "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/progress_probe.rs",
      "-o",
      "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/micro-releases/candidate/progress-probe",
      "-L",
      "dependency=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/deps",
      "--extern",
      "calc_flow=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/libcalc_flow.rlib",
      "--extern",
      "datafusion=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/deps/libdatafusion-6add09e86c412710.rlib",
      "--extern",
      "tokio=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/deps/libtokio-d37ab8cdaa9b0ce5.rlib",
      "--extern",
      "serde_json=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/deps/libserde_json-efdfce692aae30c3.rlib"
    ],
    "rustc": "rustc 1.88.0 (6b00bc388 2025-06-23)\nbinary: rustc\ncommit-hash: 6b00bc3880198600130e1cf62b8f8a93494488cc\ncommit-date: 2025-06-23\nhost: x86_64-unknown-linux-gnu\nrelease: 1.88.0\nLLVM version: 20.1.5\n",
    "probe_source_sha256": "e33e05eb9829beed2817107dfc5365fa985862ba234661255f8b308aad807b05",
    "binary_sha256": "92a17e30e11e4a1030ca49275b9ea8fe56a7ff420739b92b2082f224db8ce8d3",
    "dependencies": {
      "calc_flow": {
        "path": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.worktrees/stream-join-safety/target/issue363-join-safety-perf/micro-releases/candidate/libcalc_flow.rlib",
        "sha256": "e35f2634a9a1b63b9346b247c2e1f5ef97220e15847116ee1aa510524f28cdcf",
        "source_cache_path": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/libcalc_flow.rlib"
      },
      "datafusion": {
        "path": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/deps/libdatafusion-6add09e86c412710.rlib",
        "sha256": "c8f51700d6e854e9f1744f087271cc23ad871b39402799e2f1274979fb25fce4"
      },
      "tokio": {
        "path": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/deps/libtokio-d37ab8cdaa9b0ce5.rlib",
        "sha256": "eebd8526f67beb27a1f54f470d96d7be7885f2444de7d15ef46932e386e26f57"
      },
      "serde_json": {
        "path": "/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.claude/worktrees/join-polars-analysis/target/release-wheel/cargo/release/deps/libserde_json-efdfce692aae30c3.rlib",
        "sha256": "fdae5275f4a8f1fac3563c2dfdf3f531d6f8bafed31cb23744ccbf9f96f3ec1a"
      }
    }
  }
}
```

## Appendix D: exact native fixture

```rust
//! Same public operator workload for both release cores; no runtime source edits.
#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, Write};
use std::sync::Arc;
use std::time::{Duration, Instant};

use calc_flow::{
    Batch, BatchMetadata, CancellationToken, EdgeCollector, EventTime, IngressProgress,
    IngressProgressSnapshot, IngressState, JoinStateLimits, JoinTimeBounds, JsonMap,
    OperatorMetadata, StreamJobContext, StreamJoinOperator, StreamJoinSpec, StreamOperator,
    StreamOperatorContext,
};
use datafusion::arrow::array::{Int64Array, StringArray, TimestampMicrosecondArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use datafusion::arrow::record_batch::RecordBatch;
use serde_json::{Value, json};

const BASE: i64 = 100_000_000;
const EXTENSION: i64 = 60_000_000;
const COLD: i64 = BASE + 3_600_000_000;
const SEED_BATCH: usize = 64_000;
const SPARSE: usize = 64;
const NOOP_CALLS: usize = 8;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("account_id", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
        Field::new("amount", DataType::Int64, false),
    ]))
}

fn batch(keys: Vec<String>, timestamp: i64) -> Batch {
    let count = keys.len();
    let record = RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(StringArray::from(keys)),
            Arc::new(TimestampMicrosecondArray::from(vec![timestamp; count])),
            Arc::new(Int64Array::from(vec![7; count])),
        ],
    )
    .unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn context(job: &StreamJobContext, watermark: i64) -> StreamOperatorContext<'_> {
    StreamOperatorContext::with_ingress_progress(
        job,
        "match",
        None,
        IngressProgressSnapshot::new(BTreeMap::from([(
            "right".into(),
            IngressProgress::new(
                IngressState::Active,
                Some(EventTime::from_micros(watermark)),
            ),
        )])),
    )
}

fn rss() -> Value {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let lines: Vec<_> = status
        .lines()
        .filter(|line| line.starts_with("VmRSS:") || line.starts_with("VmHWM:"))
        .collect();
    json!(lines)
}

struct Fixture {
    operator: StreamJoinOperator,
    collector: EdgeCollector,
    job: StreamJobContext,
    rows: usize,
    charged_bytes: u64,
    watermark: i64,
    sparse_id: usize,
    evicted: u64,
}

impl Fixture {
    async fn prepare(rows: usize) -> Self {
        assert!((64..=4_000_000).contains(&rows));
        let spec = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "ts",
            "ts",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
            JoinStateLimits::new(4_000_128, 4 * 1024 * 1024 * 1024, 100_000_000).unwrap(),
        )
        .unwrap();
        let mut operator = StreamJoinOperator::new("match", schema(), schema(), spec).unwrap();
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        let job = StreamJobContext::new(
            1,
            "issue363-progress-probe",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        for start in (0..rows).step_by(SEED_BATCH) {
            let end = (start + SEED_BATCH).min(rows);
            let keys = (start..end).map(|index| format!("C{index:07}")).collect();
            operator
                .process_data(
                    "left",
                    batch(keys, COLD),
                    &StreamOperatorContext::new(&job, "match", None),
                    &mut collector,
                )
                .await
                .unwrap();
            assert!(collector.drain("output").is_empty());
        }
        let status = operator.status();
        assert_eq!(status.left.retained_rows, rows as u64);
        assert_eq!(status.right.retained_rows, 0);
        assert_eq!(status.emitted_match_rows, 0);
        assert_eq!(status.left.retained_bytes % rows as u64, 0);
        let mut fixture = Self {
            operator,
            collector,
            job,
            rows,
            charged_bytes: status.left.retained_bytes,
            watermark: BASE + EXTENSION,
            sparse_id: 0,
            evicted: 0,
        };
        fixture.noop(false).await;
        fixture.sparse(false).await;
        fixture
    }

    fn proof(&self) -> Value {
        let status = self.operator.status();
        assert_eq!(status.left.retained_rows, self.rows as u64);
        assert_eq!(status.left.retained_bytes, self.charged_bytes);
        assert_eq!(status.left.evicted_rows, self.evicted);
        assert_eq!(status.right.retained_rows, 0);
        assert_eq!(status.emitted_match_rows, 0);
        assert_eq!(status.state_limit_failures, 0);
        assert_eq!(status.match_limit_failures, 0);
        json!({"status":status, "retained_rows_expected":self.rows,
            "flat_v1_bytes_expected":self.charged_bytes, "rss":rss(), "correctness":true})
    }

    async fn noop(&mut self, timed: bool) -> Value {
        let contexts: Vec<_> = (1..=NOOP_CALLS)
            .map(|index| context(&self.job, self.watermark + index as i64))
            .collect();
        let start = timed.then(Instant::now);
        for progress in &contexts {
            self.operator
                .on_ingress_progress("right", progress)
                .await
                .unwrap();
        }
        let elapsed = start.map(|value| value.elapsed().as_secs_f64());
        self.watermark += NOOP_CALLS as i64;
        assert!(self.collector.drain("output").is_empty());
        json!({"kind":"noop", "operator_calls":NOOP_CALLS, "total_seconds":elapsed,
            "seconds":elapsed.map(|value|value / NOOP_CALLS as f64), "proof":self.proof()})
    }

    async fn sparse(&mut self, timed: bool) -> Value {
        let timestamp = self.watermark - EXTENSION + 1_000;
        assert!(timestamp < COLD);
        let keys = (self.sparse_id..self.sparse_id + SPARSE)
            .map(|index| format!("H{index:07}"))
            .collect();
        self.operator
            .process_data(
                "left",
                batch(keys, timestamp),
                &context(&self.job, self.watermark),
                &mut self.collector,
            )
            .await
            .unwrap();
        assert!(self.collector.drain("output").is_empty());
        let before = self.operator.status();
        assert_eq!(before.left.retained_rows, (self.rows + SPARSE) as u64);
        assert_eq!(
            before.left.retained_bytes,
            self.charged_bytes + SPARSE as u64 * (self.charged_bytes / self.rows as u64)
        );
        let next = timestamp + EXTENSION + 1;
        let progress = context(&self.job, next);
        let start = timed.then(Instant::now);
        self.operator
            .on_ingress_progress("right", &progress)
            .await
            .unwrap();
        let elapsed = start.map(|value| value.elapsed().as_secs_f64());
        self.watermark = next;
        self.sparse_id += SPARSE;
        self.evicted += SPARSE as u64;
        json!({"kind":"sparse", "operator_calls":1, "expired_rows":SPARSE,
            "seconds":elapsed, "before":before, "proof":self.proof()})
    }

    async fn finish(&mut self) -> Value {
        let markers: BTreeSet<_> = (0..64)
            .map(|index| format!("C{:07}", index * (self.rows - 1) / 63))
            .collect();
        assert_eq!(markers.len(), 64);
        let before = self.proof();
        self.operator
            .process_data(
                "right",
                batch(markers.iter().cloned().collect(), COLD),
                &context(&self.job, self.watermark),
                &mut self.collector,
            )
            .await
            .unwrap();
        let output = self.collector.drain("output");
        let mut observed = BTreeSet::new();
        let mut rows = 0;
        for output_batch in output {
            for record in output_batch
                .as_data()
                .expect("canary output must contain data")
                .table_payload()
                .unwrap()
                .batches()
            {
                let left = record
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                let right = record
                    .column(3)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                for row in 0..record.num_rows() {
                    assert_eq!(left.value(row), right.value(row));
                    assert!(observed.insert(left.value(row).to_owned()));
                    for column in [1, 4] {
                        assert_eq!(
                            record
                                .column(column)
                                .as_any()
                                .downcast_ref::<TimestampMicrosecondArray>()
                                .unwrap()
                                .value(row),
                            COLD
                        );
                    }
                    for column in [2, 5] {
                        assert_eq!(
                            record
                                .column(column)
                                .as_any()
                                .downcast_ref::<Int64Array>()
                                .unwrap()
                                .value(row),
                            7
                        );
                    }
                    rows += 1;
                }
            }
        }
        assert_eq!(rows, 64);
        assert_eq!(observed, markers);
        assert_eq!(self.operator.status().left.retained_rows, self.rows as u64);
        json!({"state":"completed", "canary_rows":rows, "payload_oracle":true,
            "pre_canary_proof":before, "post_canary_status":self.operator.status()})
    }
}

fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(32)
        .enable_all()
        .build()
        .unwrap();
    let mut active: Option<Fixture> = None;
    for line in std::io::stdin().lock().lines() {
        let request: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let response = match request["operation"].as_str().unwrap() {
            "hello" => json!({"protocol":"issue363-native-progress-v1",
                "pid":std::process::id(), "threads":32, "seed_batch_rows":SEED_BATCH,
                "noop_calls":NOOP_CALLS, "sparse_rows":SPARSE}),
            "prepare" => {
                assert!(active.is_none());
                active = Some(
                    runtime.block_on(Fixture::prepare(request["rows"].as_u64().unwrap() as usize)),
                );
                active.as_ref().unwrap().proof()
            }
            "sample" => match request["kind"].as_str().unwrap() {
                "noop" => runtime.block_on(active.as_mut().unwrap().noop(true)),
                "sparse" => runtime.block_on(active.as_mut().unwrap().sparse(true)),
                _ => panic!("unknown progress sample"),
            },
            "finish" => {
                let result = runtime.block_on(active.as_mut().unwrap().finish());
                active = None;
                result
            }
            _ => panic!("unknown operation"),
        };
        println!("{}", serde_json::to_string(&response).unwrap());
        std::io::stdout().flush().unwrap();
    }
}
```
