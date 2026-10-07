# A3 fragmented-output fixed-50 follow-up evidence

The fixed-count follow-up completed all 400 primary timings. Full fragmented
output remains statistically inconclusive under the maintained +5% rule;
projected fragmented output has the statistical label no-confirmed-regression.
Observed elapsed-time p50 increased 4.07% and 2.28%, respectively. Neither case
establishes a greater-than-5% improvement or regression.

The environment verdict is **inconclusive**. The saved launch snapshot contains
a new external C++ compiler after the renewed quiet grant. Its exit timestamp
was not observed, so overlap with primary timings is unknown and continuous idle
is not attested. The compiler entry was saved but not inspected before workers
started. The statistical labels below are retained observations; projected
output does not supply an idle regression-clearance proof. The full case also
retains a +5.1503% upper confidence bound. No automatic retry, sample deletion,
optional stopping or pooling with the original 440 observations occurred.

## Fixed scope and observed timings

Only the original full-fragment and projected-fragment shapes at 1M rows and
64,000-row fixture batches were repeated. Each case has two rounds of 50
alternating adjacent AB/BA pairs, with one warmup per fresh worker. Eight workers
is the total, with at most the active round's two revisions resident. Every
revision has 100 measured samples per case. Requests, rounds and cases were
serialized. Source odd/even fragmentation, all-column oracle, output order,
primary ready-enqueue-to-Arrow timer and original 256B/row plus 16KiB workspace
fees were unchanged. Full output has 12 columns; projection has four. The
original eleven-case report remains separate and unchanged.

| Case (1M/64k)      | Base p50 ms | Head p50 ms | p50 delta | Round A median [CI] % | Round B median [CI] % | Statistical label       |
|--------------------|-------------|-------------|-----------|-----------------------|-----------------------|-------------------------|
| full-fragment      | 588.601     | 612.569     | +4.07%    | +3.17% [+1.97, +4.52] | +4.56% [+3.37, +5.15] | inconclusive            |
| projected-fragment | 525.074     | 537.047     | +2.28%    | +1.95% [+1.41, +2.58] | +2.99% [+2.05, +3.80] | no-confirmed-regression |

Positive percentages mean slower elapsed time. The ratio of aggregate p50s
is descriptive; paired confidence intervals concern each round's median of
per-pair percentage changes. Maintained deterministic binomial median intervals
use ranks 18/33 for N=50, nominal iid coverage 0.9671608624357315. Both lower
bounds above `5 + 1e-12` mean regression; otherwise any upper bound above that
threshold means inconclusive; otherwise both upper bounds below its negative
mean improved; remaining outcomes mean no-confirmed-regression. Larger N and
alternation do not establish independence or eliminate WSL2 host noise.

| Case               | Base p95 ms | Base p99 ms | Head p95 ms | Head p99 ms |
|--------------------|-------------|-------------|-------------|-------------|
| full-fragment      | 656.382     | 753.010     | 659.442     | 712.362     |
| projected-fragment | 540.471     | 561.642     | 559.869     | 565.013     |

Tail quantiles are descriptive estimates from 100 samples per revision/case, not tail confidence bounds.

## Functional, lifecycle and resource evidence

The independent verifier reconstructed all 400 timing values from IPC, checked
all 408 measured/warmup all-row/full-payload/canonical-order oracles and terminal
v3 manifest/EOF proofs, and validated 424 resource guards. All eight fresh PIDs
are distinct from the original 44 and exited with strict integer code zero;
none remains in `/proc`. The wrapper has no error, live-owned PID or cleanup
error. Post-run hashing of both installed native files also matches every
worker hello and the sealed release identities. All observed worker Swap is
zero. These sources have no replay; terminal manifests and EOF do not prove a
durable restart or J1 active-compaction latency.

Per round, every IPC observation and both workers' guard journals are merged by
PID to obtain maximum HWM, against global minimum raw MemAvailable. The same
maxima populate the reported memory and conservative RAM gate. Maxima may occur
at different moments, so their sum is an upper bound rather than an observed
simultaneous total. Linux kB means 1024 bytes, with no import-floor deduction.

| Case               | Round | Base HWM bytes | Head HWM bytes | Sum HWM bytes | Sum ×1.25 bytes | 70% min available bytes |
|--------------------|-------|----------------|----------------|---------------|-----------------|-------------------------|
| full-fragment      | 1     | 2342862848     | 2313080832     | 4655943680    | 5819929600      | 19094078259             |
| full-fragment      | 2     | 2203807744     | 2215399424     | 4419207168    | 5524008960      | 19296648806             |
| projected-fragment | 1     | 2511405056     | 2438668288     | 4950073344    | 6187591680      | 18735511961             |
| projected-fragment | 2     | 2524864512     | 2441752576     | 4966617088    | 6208271360      | 18719323750             |

All four round-wide gates pass. The largest individual worker HWM is
2,524,864,512 bytes, and the largest conservative pair sum is 4,966,617,088 bytes.
No RSS-reduction or changed logical-funding claim follows from these values.
Whole matrix wall is 312.906851s; wrapper monotonic wall is
314.906997s; external GNU time caller wall is 5:14.95 and
exit status zero. Wrapper wall begins after seal validation and attempt-file
creation, includes imports/install/warmup/measurement/oracles/cleanup, and
excludes interpreter bootstrap and seal validation. Caller wall includes them.
The initial readiness delay is excluded from both. No forecast was a deadline.

## Environment and noise boundary

The platform is WSL2 Linux 5.15.167.4 on the recorded i9-13900HX host, 32 logical
CPUs with affinity 0–31. Tokio and Polars use 32 threads; OMP/OpenBLAS/MKL use one.
All worker environments match: CPython 3.13.9, NumPy 2.5.2, PyArrow 24.0.0,
DataFusion 54.0.0, Polars 1.44.2, TA-Lib 0.7.1 and JAX/JAXlib 0.11.1. Observed
one-minute Linux load ranged 0.499–2.126; load average cannot establish that individual
primary regions were idle, and host-side activity is not fully observable.

The first readiness check refused to start because an external performance
process group 1365487 was active. Limited process-only snapshots retained that
roughly seven-minute deferral; no worker, warmup, native import or build occurred
during the wait. That group's parent and children were observed gone at
21:05:47 UTC. Root renewed the team quiet grant at 21:06:40 UTC.

The saved 21:07:04.153209 UTC launch snapshot nevertheless contains a different
external group 1375610 with C++/cc1plus PIDs 1375702/1375703; cc1plus was observed
running at 97.6% CPU. The first hello request was 21:07:06.208817 UTC; warmup
preparation ran 21:07:06.900827–21:07:11.057140 UTC, and the first measured sample
request was 21:07:11.058919 UTC. A primary timer is inside that request's scope;
no extra instrumentation was added to locate its absolute boundaries. The new
external compiler's exit time and later activity were not observed. It was gone
at the post-run check. This evidence establishes launch load and unknown timer
overlap, not confirmed contamination of a specific sample or proof of idle.
No external process was killed and no external file was read.

Wrapper UTC boundaries are 2026-10-06T21:07:04.255571+00:00 to
2026-10-06T21:12:19.144578+00:00. All native workers exited before statistical checking,
file hashing and archiving. Team build/test windows resumed only after the
explicit owned-process-zero handoff. A new measurement would require a separate
scope and a fresh inspected readiness gate; it is not inferred or executed here.

## Frozen identities and tooling

| Identity                    | Baseline                                                         | Candidate                                                        |
|-----------------------------|------------------------------------------------------------------|------------------------------------------------------------------|
| Source commit               | eccb26973811bc476f0944b977ddedf8564b0237                         | 764843e634ae1a1da7a5b010095c3017349a25f4                         |
| Source SHA256               | a9cb90664b78e13c4df17fefc5b72ddc3a87f7f93527a4944cc2aff22d1bae2a | fe7fdb4506b9e4e348afc29401e5b6555cb09107ecbce461e3e3d2feb42922bf |
| Actual loaded native SHA256 | 06b152baf350be5d4ac21d9d8d4d5ca9481ef740e8d7d1ef62268098d44b3390 | e32eae86ab6a6b51445ba4a84ec6893e7ba4631d26c39bb5f57e20e6b4f0e889 |
| Wheel SHA256                | f68f4ac605a288d9ec63c1ffe61bfb1852d7fdbb33f6e18aec668425d27b75a8 | ab38ee224ee306e1ce8e6524419a6f45cd34170612a80d9e7dc149f327b07e69 |
| Cargo.lock SHA256           | 84789d29f2cb7342b0297169caf03c8f14227645656d34956ed9bc9aaa0c4840 | 84789d29f2cb7342b0297169caf03c8f14227645656d34956ed9bc9aaa0c4840 |

Both immutable release snapshots and wheels are the ones from the original A3
measurement. Candidate review-document commit 764843e6 has production inputs
identical to approved source 9983cfd0. Builds used Rust 1.88.0, Maturin 1.15.0,
CPython 3.13.9, `--release --locked`, default connector-file plus
`pyo3/abi3-py313`, empty RUSTFLAGS and jobs=2. No release was rebuilt for this
follow-up. Wheel/source contents remain in the original archive referenced below.

Version 2 fixes strict exit-code acceptance, startup process/log ownership and
cross-observer whole-round memory extrema. Three focused assert-reject tests
were RED before correction; 27 stdlib fake-process/synthetic checks were GREEN
afterwards. Tooling and ranks/count protocol were independently approved before
execution. Private shielded spawning uses the original executable, arguments,
working directory, environment, pipes and thread settings. Cleanup waits for a
late-created handle and an already started peer, including repeated cancellation.
The original worker, timer and paired driver files were not edited. These tool
checks prove ownership/acceptance behavior, not performance gains.

| Evidence                       | SHA256                                                           |
|--------------------------------|------------------------------------------------------------------|
| Fixed50 tooling seal v2        | 01dd72c09662026a6675c4b7fd4aec9529dfa464a51378f2accd80d6cf2ba066 |
| Sampling fingerprint           | dd2876a21f1f3958042062c86dfb4ec6614cff0d8cf23e272d5e220aa80dae7d |
| Wrapper                        | 040261b84281dba06910a93f200aab63a01fe92a4d175c35ecee0c26dcf0cc06 |
| Independent verifier           | c3f98d466de872a08ef0e693d53fd8be927698dd730419a4642b73e4103d1d3d |
| Original paired driver         | f597597daf678b51c003ac1c09595a354db0ae874ca8e9fbe93b436f3d5f2e5e |
| Original worker                | 51a5f50a23895b87c8c01045044762390bbc0c1ff7654f9a8a09879aa38ce47f |
| Common maintained harness      | 6593983e766921f3c73bbdcef0c3aaad011f93d41463467481c9dc7b6d9ddff5 |
| New raw matrix                 | ca0215a25698d9e29f27569a9e200b8b36e42b0daabd9285c424fb01a6f8fa3c |
| New independent full proof     | aeb588237fb0155e461faa47136eef7cb12f1eeb15d30830dd77f890b54d900c |
| Machine fingerprint            | 2e679e469194ad79c6c31f550fbe728c88f2a3425045bd3d69389c08a08ffc0d |
| Dependency fingerprint         | 6380048f9ec667274e816d52d0f30a9a42df481b9c16c3783349a2a5ce8e3c69 |
| Thread-settings fingerprint    | 0a54b5c2d6a9090cd50c76f83e49721b3db517ec2ac5c8e353792f5590eed617 |
| Full workload fingerprint      | 2ced5b006663de770a110476940583f8dcc3d6ad8ef1b4104c9c2772d2aaf557 |
| Projected workload fingerprint | 334dda702f517435043ec2c872dd429c647681644e6a1b12c4acf30886544b99 |

Machine fingerprint uses the common observed platform, logical CPU count and
affinity. Dependency fingerprint uses Python/NumPy/PyArrow/package versions;
thread settings are frozen separately. These canonical sorted JSON structures
and the matching complete environment fingerprint are preserved in
`fragment-followup-final-fingerprints-v1.json` and the independent full proof.
Both revisions match these comparison fingerprints; native/source hashes differ
intentionally. The new sampling fingerprint differs from the original N=10
contract and the rejected tooling v1; no old/new classification is pooled.

## Commands and preserved evidence

The actual invocation used repository-relative arguments at the repository root;
launch metadata stores their equivalent absolute normalization. GNU time and the
wrapper argv journals retain the actual invocation and independent wall scopes:

```bash
/usr/bin/time -v \
  -o target/issue363-asof-output-range-perf/fragment-followup-command-time-attempt1.txt \
  env PYTHONDONTWRITEBYTECODE=1 \
  .claude/worktrees/join-polars-analysis/target/benchmark-venv/bin/python \
  target/issue363-asof-output-range-perf/fragment_followup_v1.py \
  --seal target/issue363-asof-output-range-perf/fragment-followup-tooling-seal-v2.json \
  --destination target/issue363-asof-output-range-perf/fragment-followup-attempt1 \
  --quiet-granted \
  > target/issue363-asof-output-range-perf/fragment-followup-command-attempt1.log 2>&1
```

After every worker exited:

```bash
PYTHONDONTWRITEBYTECODE=1 \
.claude/worktrees/join-polars-analysis/target/benchmark-venv/bin/python \
  target/issue363-asof-output-range-perf/verify_fragment_followup_v1.py \
  target/issue363-asof-output-range-perf/fragment-followup-attempt1 \
  --seal target/issue363-asof-output-range-perf/fragment-followup-tooling-seal-v2.json \
  > target/issue363-asof-output-range-perf/fragment-followup-attempt1/verified-v1.json \
  2> target/issue363-asof-output-range-perf/fragment-followup-verifier-attempt1.log
```

The executed attempt is immutable; do not reuse that destination. The new small
archive includes every new verifier-consumed raw file, all 408 terminal manifests,
IPC/guards/exit/workload/command logs, actual command/wall/readiness/noise proofs,
new tooling and its original correction archive. All 536 tar members were read
and checked against their size/hash manifest after writing. The later Markdown,
installed duplicate native libraries and zero-byte checkpoint lock files are
excluded. No rejected native attempt or sample is discarded; the only refusal
was the preserved pre-start environment refusal. The original large archive,
report and 440 observations remain unchanged.

```json
{
  "contract": "issue363-a3-fragment-followup-evidence-archive-v1",
  "created_utc": "2026-10-06T21:18:11.946050+00:00",
  "archive_path": "target/issue363-asof-output-range-perf/evidence-a3-fragment-followup-attempt1-v1.tar.gz",
  "archive_sha256": "f49611766bb741669394395e30c0bc7f66fcf9f737f815fa647480b6736f3f5f",
  "archive_bytes": 606167,
  "files_manifest_sha256": "068a879e01f67aedc275d5fd0397708adb5dc08d157ebd72449e41d132d4cc97",
  "file_count_excluding_manifest": 535,
  "archive_contents": "complete fixed400 raw IPC/status/resource guards/command/exit/workload/terminal manifests; fixed50 tooling and v1 correction archive; commands/readiness/noise/exit/native-check/provenance proofs; unchanged original driver/worker/maintained-statistics and release metadata",
  "excluded": [
    "later supplemental Markdown",
    "installed duplicate wheels/native modules",
    "mutable Cargo outputs",
    "old wheels/source archives already preserved in evidence-a3-v1.tar.gz",
    "empty checkpoint lock files not consumed by verifier"
  ],
  "original_archive_reference": {
    "archive_path": "target/issue363-asof-output-range-perf/evidence-a3-v1.tar.gz",
    "archive_sha256": "f02bba3b7bd3c759b08110e3db3505482da0b9f032db39ad9674184af40d5010",
    "not_rewritten": true
  },
  "startup_noise": "new external cc1plus in launch snapshot; primary-timer overlap unknown; continuous idle not attested",
  "all_owned_native_processes": 0
}
```

## Exact paired arrays in seconds

These are the new independently reconstructed arrays, in chronological pair
order, not sorted or pooled. Each round has exactly 50 values per revision.
Maintained statistics can be recomputed directly from these records without
relying on ignored target-directory paths.

### full-fragment

```json
{
  "round_1": {
    "baseline": [
      0.578296199, 0.590961595, 0.570655138, 0.583374473, 0.58393997, 0.595563606, 0.599812085, 0.584028331,
      0.580840111, 0.619555186, 0.602671215, 0.62261963, 0.607786489, 0.605168287, 0.603138688, 0.596271901,
      0.606583926, 0.588402937, 0.594534574, 0.606561137, 0.626322078, 0.614945297, 0.608005551, 0.608177256,
      0.588799652, 0.580314294, 0.578254533, 0.572130001, 0.584534298, 0.56822464, 0.57360636, 0.571837816,
      0.570754909, 0.595103081, 0.606672977, 0.614152487, 0.61917433, 0.598242605, 0.600816262, 0.601598089,
      0.589392661, 0.65481518, 0.633243426, 0.66533655, 0.752861275, 0.660061461, 0.718826281, 0.647690648,
      0.767711036, 0.656188708
    ],
    "candidate": [
      0.61300284, 0.602026395, 0.596274257, 0.623247129, 0.624549434, 0.607900101, 0.611502433, 0.615953448,
      0.602606855, 0.614745617, 0.621524639, 0.630311523, 0.636243311, 0.617849343, 0.638195694, 0.614943493,
      0.632991309, 0.63739838, 0.63485733, 0.622209786, 0.629498192, 0.628753643, 0.619738223, 0.648002457,
      0.592358509, 0.598977425, 0.597931907, 0.605715479, 0.607278759, 0.60270139, 0.603772366, 0.594432758,
      0.629503231, 0.606816129, 0.616334792, 0.619330678, 0.618403915, 0.625304991, 0.638395544, 0.61836856,
      0.617954276, 0.630804837, 0.64324636, 0.658910404, 0.697799548, 0.693157623, 0.669550094, 0.654126265,
      0.728669376, 0.712196932
    ]
  },
  "round_2": {
    "baseline": [
      0.609704585, 0.583445989, 0.558359775, 0.593574131, 0.602142131, 0.592994249, 0.569377704, 0.576858924,
      0.577938509, 0.58252769, 0.580095596, 0.57588654, 0.596349492, 0.603432595, 0.594439721, 0.56538286,
      0.591365341, 0.596258642, 0.58466493, 0.584061121, 0.576894124, 0.580569482, 0.570615803, 0.591287449,
      0.576042804, 0.57871258, 0.580100681, 0.587144241, 0.582327399, 0.590294337, 0.570121973, 0.570796178,
      0.587710361, 0.566516975, 0.573381868, 0.582221635, 0.587919162, 0.578453377, 0.572924397, 0.583737891,
      0.595544363, 0.578231628, 0.568881928, 0.608953592, 0.587149589, 0.593963581, 0.578066898, 0.603749949,
      0.579191965, 0.575195859
    ],
    "candidate": [
      0.61699766, 0.603981197, 0.584483448, 0.602268115, 0.622912054, 0.63435967, 0.613830825, 0.602880046,
      0.628196999, 0.615798648, 0.613794635, 0.605659627, 0.631450314, 0.599420054, 0.613765129, 0.610764228,
      0.608668855, 0.589467753, 0.61215147, 0.590294998, 0.606606001, 0.614157533, 0.597790905, 0.601519786,
      0.612986041, 0.598582383, 0.624460234, 0.611049399, 0.593090877, 0.595611597, 0.614452989, 0.618448873,
      0.590999041, 0.595138372, 0.606521808, 0.595223771, 0.604137665, 0.606594397, 0.605250202, 0.61199546,
      0.60381301, 0.60898716, 0.59461356, 0.592662478, 0.606934338, 0.605684149, 0.602122935, 0.59860405,
      0.609135798, 0.601640815
    ]
  }
}
```

### projected-fragment

```json
{
  "round_1": {
    "baseline": [
      0.532309381, 0.525139057, 0.518161288, 0.532683062, 0.526791838, 0.5366815, 0.531053804, 0.540234558,
      0.535364165, 0.5248588, 0.515463579, 0.517215986, 0.529194844, 0.52121772, 0.516355806, 0.527823965,
      0.527368634, 0.549332503, 0.534870091, 0.53198924, 0.538560006, 0.528755446, 0.537412678, 0.508120663,
      0.529575509, 0.532503345, 0.516628159, 0.517715138, 0.536749381, 0.544959306, 0.530907855, 0.523409593,
      0.528240047, 0.524911666, 0.516401192, 0.523960939, 0.516055133, 0.525194222, 0.537019414, 0.518878699,
      0.52512937, 0.520329274, 0.519887225, 0.524330288, 0.522764264, 0.516796086, 0.519912221, 0.524129773,
      0.517913774, 0.514569772
    ],
    "candidate": [
      0.564955989, 0.552035149, 0.530818421, 0.547999623, 0.535949759, 0.53576883, 0.533493982, 0.545732029,
      0.54036331, 0.532274779, 0.53323337, 0.537802043, 0.564265776, 0.548687137, 0.526904617, 0.532653254,
      0.555497847, 0.550503104, 0.54281914, 0.535902123, 0.554429122, 0.533849366, 0.551279333, 0.523252705,
      0.542064412, 0.541720377, 0.531712506, 0.527284645, 0.552398247, 0.544448925, 0.533978391, 0.528517738,
      0.524750018, 0.543511226, 0.535373109, 0.532638967, 0.528357218, 0.533720057, 0.534212224, 0.528038183,
      0.530836543, 0.532261567, 0.532419726, 0.548308728, 0.529958798, 0.520188247, 0.525741527, 0.537639816,
      0.538796061, 0.548355886
    ]
  },
  "round_2": {
    "baseline": [
      0.526992686, 0.536105974, 0.515052793, 0.52097759, 0.533897134, 0.52649713, 0.524202428, 0.531061594,
      0.567929936, 0.519130984, 0.519806026, 0.538039764, 0.53076746, 0.529650359, 0.529452582, 0.519705483,
      0.530267295, 0.547574046, 0.522241407, 0.525640872, 0.531672594, 0.525953563, 0.52981711, 0.513816148,
      0.527092352, 0.516243378, 0.52203955, 0.516003885, 0.561578689, 0.536558891, 0.526631068, 0.517030638,
      0.526484569, 0.526586887, 0.515818217, 0.51788795, 0.522934227, 0.50964258, 0.538648487, 0.513975621,
      0.521102315, 0.520439152, 0.521545747, 0.515639891, 0.525019043, 0.522765361, 0.51648532, 0.515584102,
      0.525890423, 0.516070117
    ],
    "candidate": [
      0.537797907, 0.53996835, 0.532897467, 0.550461408, 0.548960204, 0.55844916, 0.538920515, 0.530248284,
      0.560197027, 0.559852057, 0.53783564, 0.539215517, 0.551588808, 0.549775801, 0.527180386, 0.551096565,
      0.553447744, 0.524571151, 0.570654478, 0.538569692, 0.5289138, 0.563820767, 0.531994861, 0.528561752,
      0.551366183, 0.545196669, 0.521878993, 0.534376147, 0.523907001, 0.535702885, 0.536454046, 0.529913854,
      0.542683397, 0.546581038, 0.538235767, 0.528471009, 0.532842801, 0.535779294, 0.535255109, 0.532729484,
      0.524811298, 0.542554398, 0.538952033, 0.537937191, 0.526222003, 0.535256147, 0.543175032, 0.541507083,
      0.542813435, 0.531051444
    ]
  }
}
```

The structural contiguous source/range reduction recorded in the original A3
analysis remains a separate allocation/complexity result. This follow-up covers
the unresolved fragmented output path and supplies full functional/cost evidence;
it does not establish a material pipeline gain or remove full-fragment regression
uncertainty. A matched output-planning microbenchmark remains the most useful
coverage for attributing source/range work separately from matching/gather/runner
cost. Durability, nested/dictionary schemas, additional payload widths and J1
active asynchronous-compaction behavior are outside this fixed follow-up.
