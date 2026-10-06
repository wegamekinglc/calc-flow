# Issue 363 phase 0 performance evidence

PR [#365](https://github.com/wegamekinglc/calc-flow/pull/365) implements phases
0.1, 0.5 and 0.7. It changes the benchmark harness and reference coverage; it
does not change the native engine. This report measures the implementation at
`9b1535bcf4a9cdf394477da03c9718412a1092b1` against `49d346df15c1ad69af0abfaf9e2ee09cbe80fe49`.

Both clean release wheels have exactly the same native SHA-256 and native
source fingerprint. The native performance gain attributable to this PR is
therefore **zero**. Timing variation in the common-harness experiment measures
the environment's noise. The separate v4-to-v5 experiment measures changed
harness overhead and remains **new coverage**, with no native speedup claim.

In the v4-to-v5 harness diagnostic, 1M-row ASOF P50 changes from
386.239 to 368.516 ms (-4.59%). At 100,000 rows and 1,024-row batches,
it changes from 202.050 to 83.063 ms (-58.89%, 2.43x shorter).

## Measurement contract and environment

- Machine: 13th Gen Intel(R) Core(TM) i9-13900HX, Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.43,
  32 logical CPUs. CPU and memory details, affinity and actual dependencies are
  preserved in `results.json` and each worker's `hello` observation.
- The machine is WSL2/shared. Team builds and tests were paused throughout the
  accepted collection. Process observations are retained. This coordination
  cannot prove independent pairs, constant CPU frequency or an isolated host.
- Native cases use two fresh worker pairs and ten alternating AB/BA pairs per
  round. Setup, one warmup per worker, oracles, startup and shutdown stay outside
  the maintained ready-enqueue-to-Arrow timer. Every output passes the existing
  full-row oracle; all 480 measured samples and 48 warmups passed.
- Thread identities are checked against the sealed wheel: Tokio environment
  32; actual Polars pool 32 for native/32T cases and 1 for 1T cases; BLAS 1.
  Measurement uses the existing lock-matching investigation benchmark venv,
  not the root development venv with dependency drift.
- P50/P95 describe complete workloads across 20 observations per side. They
  are not per-batch latency quantiles. Each round's change is the median of ten
  aligned percentage changes. The maintained exact order-statistic interval
  has 97.85% coverage for ten independent pairs; the suite's two-round +5%
  verdict is retained without replacing it with minima.
- The first attempt encountered an unrelated external C++ build. The entire
  attempt, including eight completed cases and partial workers, is preserved
  under `attempt-1-external-load/` and excluded. A full 14-case rerun was decided
  before inspecting or selecting replacement results.

## Common v5 harness: same-native noise control

Both wheels run the candidate's v5 harness. All dimensions, dependencies,
machine/thread identities and loaded native hashes match. This diagnostic
establishes a current same-native noise control, not a production speedup.
The baseline's canonical v4 catalog still classifies these stream cases as
new coverage in a normal unified-suite run.
Some short-duration control intervals cross +5%; those comparisons remain
inconclusive and do not establish timing equivalence. None supplies a native
optimization claim or a confirmed regression.

| Workload             | Batch rows | Base P50 ms | Base P95 ms | Head P50 ms | Head P95 ms | Round 1 change [CI]     | Round 2 change [CI]     | Diagnostic verdict      |
|----------------------|------------|-------------|-------------|-------------|-------------|-------------------------|-------------------------|-------------------------|
| asof_join 1,000,000  | 64,000     | 374.126     | 385.807     | 373.917     | 388.340     | +0.41% [-2.63, +2.51]   | +0.02% [-3.63, +3.01]   | no-confirmed-regression |
| projection 1,000,000 | 64,000     | 6.528       | 9.186       | 6.442       | 7.739       | -4.53% [-26.38, +15.81] | +3.56% [-31.70, +13.73] | inconclusive            |
| join 1,000,000       | 64,000     | 696.678     | 727.688     | 683.755     | 715.976     | -1.26% [-3.19, +0.85]   | +0.52% [-6.13, +1.62]   | no-confirmed-regression |
| asof_join 100,000    | 1,024      | 84.687      | 90.421      | 86.604      | 92.769      | +3.84% [-1.23, +8.03]   | +2.26% [-5.68, +9.36]   | inconclusive            |
| projection 100,000   | 1,024      | 31.189      | 32.842      | 31.447      | 32.228      | +0.01% [-3.78, +3.30]   | -1.36% [-5.47, +2.45]   | no-confirmed-regression |

## v4 to v5 harness: changed-scope diagnostic

Each wheel runs its own revision's harness here. Source-provided fixture,
batch layout and full output oracle match, but the old ASOF harness polls
status whereas v5 waits for sink delivery. These scopes differ. Paired
statistics are descriptive harness-overhead evidence; their diagnostic verdicts
must not be presented as compatible native-engine version verdicts.
Projection and Join remain unchanged timing controls in this experiment.

| Workload             | Batch rows | Base P50 ms | Base P95 ms | Head P50 ms | Head P95 ms | Round 1 change [CI]      | Round 2 change [CI]      | Diagnostic verdict      |
|----------------------|------------|-------------|-------------|-------------|-------------|--------------------------|--------------------------|-------------------------|
| asof_join 1,000,000  | 64,000     | 386.239     | 407.162     | 368.516     | 383.261     | -4.62% [-5.53, -1.15]    | -4.13% [-6.52, -1.36]    | no-confirmed-regression |
| projection 1,000,000 | 64,000     | 6.415       | 8.608       | 6.129       | 7.748       | -2.90% [-14.77, +11.99]  | -3.79% [-18.63, +15.81]  | inconclusive            |
| join 1,000,000       | 64,000     | 685.345     | 724.999     | 685.893     | 715.302     | +1.07% [-1.73, +2.82]    | -0.12% [-4.25, +1.53]    | no-confirmed-regression |
| asof_join 100,000    | 1,024      | 202.050     | 205.816     | 83.063      | 86.818      | -58.55% [-59.99, -56.53] | -59.21% [-60.35, -58.03] | improved                |
| projection 100,000   | 1,024      | 31.276      | 35.596      | 31.554      | 38.527      | -0.92% [-13.09, +9.81]   | +0.70% [-4.36, +9.65]    | inconclusive            |

## Small-batch acceptance target

At 100,000 rows and 1,024-row batches, candidate ASOF P50 divided by candidate
projection P50 is **2.754x** in the common-harness run and **2.632x** in the
cross-scope diagnostic. These are raw total-workload ratios.
The original v4 ratio in the cross-scope experiment is 6.460x.

The phase 0.1 target concerns harness overhead relative to the projection
floor above a matched operator time. This collection does not isolate a
same-shape native operator time, so that overhead acceptance gate remains
**unverified**. A total ASOF/projection ratio greater than two does not establish
a harness-overhead gate failure; subtracting the projection time does not
measure the actual ASOF harness overhead either.

The separate status-forbidden probe processes the full 100,000-row ASOF
workload at this batch size with `StreamingJob.status` replaced by a raising
function. It passes its output oracle. This probe is not included among the
timing samples; accepted measured workers have no status instrumentation.

## Polars references at 1M rows

Each reference has two fresh workers and ten observations per round, actual
pool-size validation and the same full-row output oracle. These newly
registered single-thread references have no historical version verdict.

| Workload  | Actual threads | P50 ms | P95 ms | Samples |
|-----------|----------------|--------|--------|---------|
| join      | 1              | 17.487 | 18.856 | 20      |
| asof_join | 1              | 63.986 | 68.430 | 20      |
| join      | 32             | 8.333  | 14.980 | 20      |
| asof_join | 32             | 27.223 | 33.671 | 20      |

## Sealed provenance and raw artifacts

| Field                 | Baseline                                                           | Candidate                                                          |
|-----------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|
| Git commit            | `49d346df15c1ad69af0abfaf9e2ee09cbe80fe49`                         | `9b1535bcf4a9cdf394477da03c9718412a1092b1`                         |
| Clean source at build | `true`                                                             | `true`                                                             |
| Build profile         | `release`                                                          | `release`                                                          |
| Native-source SHA-256 | `4ab268c96b25ff2afab10d5725734917d5eb7769a0445fa2605cbe564318d595` | `4ab268c96b25ff2afab10d5725734917d5eb7769a0445fa2605cbe564318d595` |
| Native SHA-256        | `b5b671d656f4121505ec345ed3ed7f135492ef252d051513ef7a9d6ad6615c96` | `b5b671d656f4121505ec345ed3ed7f135492ef252d051513ef7a9d6ad6615c96` |
| Wheel SHA-256         | `d75d5592cf889dd9a2d9d9f02a87188f3b8e4d91c4ed3de7c6f6ec708509a19b` | `7940da9214c2d4ec32364e833055d4ace09fa052288a0acb72a8b49f588c4a04` |

Both wheels were rebuilt from clean exact revisions with the maintained
`scripts.benchmark_suite build` command and release/locked/abi3-py313 recipe;
the existing investigation Cargo cache was reused. No unmanifested old wheel
was relabeled as a sealed build. Original `build.json`, `release.json`, commands
and build logs remain beside each wheel.

All paths below are relative to the evidence worktree:

- `target/issue363-evidence-perf/results.json`: all original rounds, samples,
  correctness evidence, identities, exact intervals and descriptive summaries.
- `target/issue363-evidence-perf/verification.json`: independent verification of
  sealed native hashes, fresh-worker protocol, actual batch layouts, complete
  sampling/oracle inventory, and statistics reproduced from original samples.
- `target/issue363-evidence-perf/common-v5-runtime-control/`,
  `cross-scope-harness-diagnostic/` and `external/` beneath the same evidence
  root: worker commands/PIDs, original IPC JSONL, workload attestations and stderr.
- `target/issue363-evidence-perf/status-forbidden-probe.json`: untimed full-workload
  no-status proof.
- `target/issue363-evidence-perf/releases/{baseline,candidate}/`: sealed wheels,
  manifests and original build logs.
- `target/issue363-evidence-perf/{measure,worker,verify}.py`: reproduction driver,
  explicit batch-layout shim and evidence checker; all use the maintained
  adapters, oracle, isolated worker lifecycle and paired statistics.
- `target/issue363-evidence-perf/quiet-process-observations.jsonl`: process snapshots.
- `target/issue363-evidence-perf/raw-evidence.tar.gz`: portable diagnostic/raw
  worker evidence and build provenance, excluding large wheels and Cargo caches.
  SHA-256: `6c73d84a994f3f9a23c495c445da580dd87c6d8f25fb5dc7415d31ef1baf295a`.

Results SHA-256: `1ded5083c98cbd0d5c73538384da194b5e3df9917c7807155023ffa696b7ac00`.
Common harness SHA-256: `992ebc825beeb72a86887c85d83858f460cec5f9c37497e0773a7e078cb61d2c`.
Collector SHA-256: `b86198b1a75999d36f777ea6225db67ef2152e57d3931047209f035310cf1a64`.
Batch-layout shim SHA-256: `6efe226e4599522c667d02a13aa95a9323f52538b10af5782dcd5f3826172041`.

Reproduction uses the same benchmark interpreter recorded in each original
worker command. From a clean checkout at the sealed candidate revision:

```bash
PYTHONPATH="$PWD" <benchmark-python> target/issue363-evidence-perf/measure.py
PYTHONPATH="$PWD" <benchmark-python> target/issue363-evidence-perf/verify.py
```

Use fresh output directories; retain original attempts rather than overwriting
them. Source checkouts and sealed manifests must retain their exact revisions.

## Coverage and remaining limits

The changed ASOF sink-delivery path is exercised by the maintained native
engine cases, including 1M rows here. Polars 1T/32T paths both have real measured
references and actual pool checks. The explicit 1,024-row batch-size cases are
local named diagnostics. Subsequent benchmark-coverage work adds maintained
catalog dimensions for this regime. A same-shape native operator benchmark is
also needed to verify the overhead target independently of total workload ratios.

This report does not validate Join status/progress phase 0.2, checkpoint-on
throughput, J1/J2 or ASOF admission/finalization optimizations. No Rust runtime
hot path changes in PR #365 require an additional Rust benchmark. Full CI,
cross-platform coverage and later commits do not inherit a verdict here.

## Original timing arrays

These are original seconds, retained without rounding, so both round intervals
can be independently reproduced. Full worker identities and output correctness
remain in the raw artifacts above.

```json
{
  "cases": [
    {
      "experiment": "common-v5-runtime-control",
      "id": "engines/1000000/calc-flow-stream/asof_join/batch-64000",
      "baseline_seconds": [
        [
          0.366514979,
          0.382375065,
          0.381391911,
          0.372204421,
          0.378608848,
          0.373116259,
          0.37008408,
          0.367081709,
          0.361914629,
          0.372822048
        ],
        [
          0.385691479,
          0.377373078,
          0.375135987,
          0.363992162,
          0.388007117,
          0.379081989,
          0.38256678,
          0.382791625,
          0.37027144,
          0.362989223
        ]
      ],
      "candidate_seconds": [
        [
          0.374060038,
          0.372304467,
          0.370415541,
          0.369209108,
          0.374766478,
          0.383623076,
          0.374935925,
          0.368984445,
          0.371006812,
          0.373936447
        ],
        [
          0.364377002,
          0.383843519,
          0.368896421,
          0.365563274,
          0.373924123,
          0.39399376,
          0.388042551,
          0.3813259,
          0.363989277,
          0.373908973
        ]
      ]
    },
    {
      "experiment": "common-v5-runtime-control",
      "id": "engines/1000000/calc-flow-stream/projection/batch-64000",
      "baseline_seconds": [
        [
          0.005388232,
          0.008836806,
          0.007176527,
          0.007131505,
          0.00813329,
          0.005965998,
          0.006961247,
          0.005942122,
          0.006777512,
          0.006218076
        ],
        [
          0.015817873,
          0.006111818,
          0.006219907,
          0.005545298,
          0.00627768,
          0.007944116,
          0.005665778,
          0.008612206,
          0.00692886,
          0.005930975
        ]
      ],
      "candidate_seconds": [
        [
          0.007388247,
          0.00650549,
          0.00556229,
          0.007081153,
          0.005978199,
          0.006349102,
          0.006440539,
          0.005819981,
          0.007849305,
          0.00578252
        ],
        [
          0.006084811,
          0.007733685,
          0.006320284,
          0.006063437,
          0.006623382,
          0.006666202,
          0.006443724,
          0.005882154,
          0.007020287,
          0.006692771
        ]
      ]
    },
    {
      "experiment": "common-v5-runtime-control",
      "id": "engines/1000000/calc-flow-stream/join/batch-64000",
      "baseline_seconds": [
        [
          0.727169135,
          0.696812913,
          0.699833865,
          0.676834153,
          0.734158011,
          0.670370193,
          0.677710228,
          0.715530825,
          0.698286921,
          0.670912891
        ],
        [
          0.696543422,
          0.677421886,
          0.70215181,
          0.666938088,
          0.664276986,
          0.678062688,
          0.694191144,
          0.721068212,
          0.727347566,
          0.716355233
        ]
      ],
      "candidate_seconds": [
        [
          0.703993894,
          0.702716234,
          0.67961642,
          0.660617059,
          0.691942447,
          0.674501656,
          0.676900477,
          0.693733062,
          0.701895895,
          0.684794475
        ],
        [
          0.70301809,
          0.676774417,
          0.713561248,
          0.669909158,
          0.668269435,
          0.684749811,
          0.669828478,
          0.674010519,
          0.682759317,
          0.76184871
        ]
      ]
    },
    {
      "experiment": "common-v5-runtime-control",
      "id": "engines/100000/calc-flow-stream/asof_join/batch-1024",
      "baseline_seconds": [
        [
          0.085505974,
          0.08781927,
          0.086993485,
          0.084165121,
          0.082857655,
          0.081700392,
          0.089536736,
          0.085208101,
          0.079777045,
          0.087110719
        ],
        [
          0.095468268,
          0.086197013,
          0.090155049,
          0.082152348,
          0.082068352,
          0.080189868,
          0.086347,
          0.079495249,
          0.080285051,
          0.080691887
        ]
      ],
      "candidate_seconds": [
        [
          0.084451028,
          0.086770329,
          0.089622298,
          0.091380603,
          0.086438303,
          0.088261963,
          0.092542414,
          0.090324262,
          0.085596645,
          0.084753158
        ],
        [
          0.090042696,
          0.097077875,
          0.084789098,
          0.083853266,
          0.086390581,
          0.082157233,
          0.090790358,
          0.086932748,
          0.080218529,
          0.080101496
        ]
      ]
    },
    {
      "experiment": "common-v5-runtime-control",
      "id": "engines/100000/calc-flow-stream/projection/batch-1024",
      "baseline_seconds": [
        [
          0.030518752,
          0.030654067,
          0.030981366,
          0.030872583,
          0.032044522,
          0.03175079,
          0.032536463,
          0.032680034,
          0.032090683,
          0.03237349
        ],
        [
          0.031153618,
          0.030252969,
          0.031267042,
          0.030998194,
          0.03117279,
          0.030728261,
          0.031683881,
          0.035925011,
          0.030784648,
          0.031204745
        ]
      ],
      "candidate_seconds": [
        [
          0.030569743,
          0.03243289,
          0.031743011,
          0.03189239,
          0.031998784,
          0.032020465,
          0.031538387,
          0.031144112,
          0.030877106,
          0.031619016
        ],
        [
          0.029647847,
          0.030993371,
          0.029963377,
          0.031354955,
          0.029468247,
          0.03038463,
          0.031683989,
          0.031892409,
          0.03029266,
          0.032217484
        ]
      ]
    },
    {
      "experiment": "cross-scope-harness-diagnostic",
      "id": "engines/1000000/calc-flow-stream/asof_join/batch-64000",
      "baseline_seconds": [
        [
          0.372532473,
          0.39688307,
          0.388615216,
          0.373551727,
          0.391964636,
          0.392149148,
          0.408336593,
          0.374486978,
          0.372050261,
          0.379887366
        ],
        [
          0.38349923,
          0.407100626,
          0.388002447,
          0.38486455,
          0.385275922,
          0.387221236,
          0.399338249,
          0.387202326,
          0.370133581,
          0.379815448
        ]
      ],
      "candidate_seconds": [
        [
          0.368253599,
          0.378839279,
          0.369880003,
          0.373379092,
          0.370270173,
          0.381432526,
          0.368779303,
          0.364094826,
          0.354179121,
          0.362060735
        ],
        [
          0.374781312,
          0.373245136,
          0.365546731,
          0.366757677,
          0.365658004,
          0.373443107,
          0.418002456,
          0.361944115,
          0.365095969,
          0.366572469
        ]
      ]
    },
    {
      "experiment": "cross-scope-harness-diagnostic",
      "id": "engines/1000000/calc-flow-stream/projection/batch-64000",
      "baseline_seconds": [
        [
          0.007107971,
          0.006031521,
          0.008631025,
          0.005999903,
          0.006405246,
          0.006301992,
          0.006194936,
          0.006068005,
          0.007118604,
          0.006583651
        ],
        [
          0.005820172,
          0.00613137,
          0.00636553,
          0.007656101,
          0.00675161,
          0.006191259,
          0.006662376,
          0.00860678,
          0.007589617,
          0.00642491
        ]
      ],
      "candidate_seconds": [
        [
          0.006306224,
          0.005868591,
          0.006066033,
          0.005813809,
          0.006257552,
          0.00741166,
          0.006625483,
          0.005867986,
          0.006066988,
          0.007373073
        ],
        [
          0.006734915,
          0.00599314,
          0.007663026,
          0.006042106,
          0.005866703,
          0.007170151,
          0.006054259,
          0.009352618,
          0.006175485,
          0.006083
        ]
      ]
    },
    {
      "experiment": "cross-scope-harness-diagnostic",
      "id": "engines/1000000/calc-flow-stream/join/batch-64000",
      "baseline_seconds": [
        [
          0.706456018,
          0.708584525,
          0.730070987,
          0.681198398,
          0.668829536,
          0.666716506,
          0.684960335,
          0.694431087,
          0.67193581,
          0.672711206
        ],
        [
          0.688534951,
          0.693170861,
          0.686627659,
          0.688869767,
          0.675894677,
          0.663585643,
          0.655167058,
          0.685729614,
          0.664687931,
          0.724731784
        ]
      ],
      "candidate_seconds": [
        [
          0.714942384,
          0.696298725,
          0.702435567,
          0.682934103,
          0.676169488,
          0.673658471,
          0.722131441,
          0.689366455,
          0.690905908,
          0.688852491
        ],
        [
          0.689747508,
          0.706062772,
          0.657417591,
          0.671423511,
          0.663612805,
          0.660818008,
          0.661671062,
          0.691592428,
          0.674826129,
          0.682026561
        ]
      ]
    },
    {
      "experiment": "cross-scope-harness-diagnostic",
      "id": "engines/100000/calc-flow-stream/asof_join/batch-1024",
      "baseline_seconds": [
        [
          0.201295324,
          0.193840428,
          0.169570677,
          0.203096531,
          0.204475397,
          0.202276615,
          0.201092749,
          0.200172138,
          0.201269557,
          0.205504683
        ],
        [
          0.199280339,
          0.204889667,
          0.20353643,
          0.201619887,
          0.202990245,
          0.201822722,
          0.201670888,
          0.204383177,
          0.202652842,
          0.21174045
        ]
      ],
      "candidate_seconds": [
        [
          0.087494135,
          0.083308798,
          0.08614524,
          0.084494983,
          0.085000665,
          0.081303669,
          0.083103106,
          0.080223989,
          0.080533807,
          0.080864692
        ],
        [
          0.081299339,
          0.086782891,
          0.083023512,
          0.084624208,
          0.080492959,
          0.083772402,
          0.081683601,
          0.081415094,
          0.084602346,
          0.082542377
        ]
      ]
    },
    {
      "experiment": "cross-scope-harness-diagnostic",
      "id": "engines/100000/calc-flow-stream/projection/batch-1024",
      "baseline_seconds": [
        [
          0.035415584,
          0.036988694,
          0.031093454,
          0.030698985,
          0.034717299,
          0.031459112,
          0.035522548,
          0.030967636,
          0.032294594,
          0.033487479
        ],
        [
          0.031088466,
          0.031014771,
          0.030863344,
          0.030708549,
          0.03046359,
          0.029620442,
          0.032008182,
          0.030875134,
          0.035252138,
          0.031639992
        ]
      ],
      "candidate_seconds": [
        [
          0.031605161,
          0.030131576,
          0.034144031,
          0.031503659,
          0.033294263,
          0.039448245,
          0.030871072,
          0.031099489,
          0.035430586,
          0.03272702
        ],
        [
          0.038478444,
          0.029880932,
          0.033841547,
          0.030979655,
          0.030620654,
          0.03246684,
          0.030611987,
          0.030458244,
          0.031197992,
          0.032754966
        ]
      ]
    },
    {
      "experiment": "external-new-coverage",
      "id": "engines/1000000/polars-1t/join/batch-64000",
      "baseline_seconds": [],
      "candidate_seconds": [
        [
          0.019304065,
          0.017538228,
          0.01750055,
          0.016959669,
          0.017662023,
          0.018381371,
          0.017134147,
          0.01726924,
          0.017674314,
          0.017345109
        ],
        [
          0.017332997,
          0.017298022,
          0.018063339,
          0.018832662,
          0.017971428,
          0.017474392,
          0.017256854,
          0.017139707,
          0.01739459,
          0.017873457
        ]
      ]
    },
    {
      "experiment": "external-new-coverage",
      "id": "engines/1000000/polars-1t/asof_join/batch-64000",
      "baseline_seconds": [],
      "candidate_seconds": [
        [
          0.062992055,
          0.062200472,
          0.061253959,
          0.064520552,
          0.062721142,
          0.068927675,
          0.063219937,
          0.064829713,
          0.065746935,
          0.067608533
        ],
        [
          0.062214382,
          0.063176461,
          0.061790462,
          0.065851582,
          0.064908182,
          0.065389503,
          0.063452185,
          0.06294129,
          0.067719465,
          0.068403733
        ]
      ]
    },
    {
      "experiment": "external-new-coverage",
      "id": "engines/1000000/polars/join/batch-64000",
      "baseline_seconds": [],
      "candidate_seconds": [
        [
          0.008314776,
          0.009128924,
          0.007158507,
          0.007538481,
          0.008351384,
          0.006962113,
          0.007779751,
          0.007824928,
          0.007408336,
          0.008825274
        ],
        [
          0.010052133,
          0.010053585,
          0.007904241,
          0.00700399,
          0.007189265,
          0.017766679,
          0.013300919,
          0.013726513,
          0.010417542,
          0.014833206
        ]
      ]
    },
    {
      "experiment": "external-new-coverage",
      "id": "engines/1000000/polars/asof_join/batch-64000",
      "baseline_seconds": [],
      "candidate_seconds": [
        [
          0.05693917,
          0.028929398,
          0.024311028,
          0.026515227,
          0.023703286,
          0.032446229,
          0.023349724,
          0.021944839,
          0.030065846,
          0.031787591
        ],
        [
          0.02901105,
          0.025449963,
          0.025541659,
          0.027576363,
          0.026869446,
          0.023162634,
          0.024223265,
          0.029101539,
          0.030893526,
          0.030538384
        ]
      ]
    }
  ]
}
```
