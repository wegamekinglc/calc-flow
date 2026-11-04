# sum-batch-parallel-20261104

Collection status: **completed**.
Performance acceptance: **False**.

| Case               | Baseline P50 ms | Candidate P50 ms | Change  | Verdict      |
|--------------------|-----------------|------------------|---------|--------------|
| stock-stream-1cpu  | 353.969         | 361.425          | +2.11%  | inconclusive |
| stock-stream-4cpu  | 376.225         | 435.468          | +15.75% | regression   |
| stock-stream-8cpu  | 375.070         | 444.834          | +18.60% | regression   |
| sparse-stream-1cpu | 33.471          | 34.572           | +3.29%  | inconclusive |
| hotkey-stream-8cpu | 112.540         | 127.855          | +13.61% | inconclusive |

| Case               | Round | Paired median | Median interval    | Coverage  |
|--------------------|-------|---------------|--------------------|-----------|
| stock-stream-1cpu  | 1     | -0.19%        | [-3.15%, +9.48%]   | 97.85156% |
| stock-stream-1cpu  | 2     | +2.67%        | [-5.24%, +9.19%]   | 97.85156% |
| stock-stream-4cpu  | 1     | +14.42%       | [+11.82%, +24.32%] | 97.85156% |
| stock-stream-4cpu  | 2     | +19.31%       | [+9.89%, +28.62%]  | 97.85156% |
| stock-stream-8cpu  | 1     | +18.64%       | [+11.62%, +28.31%] | 97.85156% |
| stock-stream-8cpu  | 2     | +19.11%       | [+16.02%, +24.19%] | 97.85156% |
| sparse-stream-1cpu | 1     | +1.01%        | [-3.27%, +6.30%]   | 97.85156% |
| sparse-stream-1cpu | 2     | +6.00%        | [-1.09%, +17.25%]  | 97.85156% |
| hotkey-stream-8cpu | 1     | +14.04%       | [+10.91%, +24.74%] | 97.85156% |
| hotkey-stream-8cpu | 2     | +10.67%       | [-0.88%, +27.93%]  | 97.85156% |

| Case               | Side      | P95 ms  | CPU P50 ms | Cores P50 | Peak RSS P50 MiB |
|--------------------|-----------|---------|------------|-----------|------------------|
| stock-stream-1cpu  | baseline  | 382.673 | 353.956    | 1.000     | 231.46           |
| stock-stream-1cpu  | candidate | 385.788 | 357.069    | 1.000     | 231.58           |
| stock-stream-4cpu  | baseline  | 413.297 | 427.090    | 1.149     | 247.27           |
| stock-stream-4cpu  | candidate | 477.599 | 548.670    | 1.254     | 268.45           |
| stock-stream-8cpu  | baseline  | 393.570 | 428.307    | 1.152     | 247.67           |
| stock-stream-8cpu  | candidate | 468.909 | 587.846    | 1.324     | 271.14           |
| sparse-stream-1cpu | baseline  | 36.328  | 33.471     | 1.000     | 70.61            |
| sparse-stream-1cpu | candidate | 39.230  | 34.572     | 1.000     | 70.61            |
| hotkey-stream-8cpu | baseline  | 120.255 | 172.644    | 1.541     | 236.18           |
| hotkey-stream-8cpu | candidate | 148.516 | 204.072    | 1.645     | 244.32           |

Five fixed cases, two rounds of ten alternating pairs, and one 600-second
budget include fixture, preflight, warmup, full 21-snapshot oracles,
profiles,
decode, analysis, publication and cleanup. Builds are recorded
separately.
The actual reused baseline product is af400, byte-compatible with
merged c9afa;
it is never relabeled as a fresh c9afa build. All arms use audited
common657
instrumentation from public841 only in permanently frozen native
source trees.

All five cases must finish, no selected case may regress, and dense4
or dense8
must be improved under the maintained paired-median/5% rule for
acceptance.
Inconclusive does not establish equivalence or speedup. No retries
are allowed.
All streams retain one DataFusion partition; the CPU work allowance
is separate.
Eight logical CPUs represent four reported cores on WSL2; host load
is unknown.
100k source rows filter to 80k SQL rows: ten legacy SQL chunks,
admitting at most five local units with eight CPUs.
The seed and sparse batches remain below parallel admission. Hot-key
input maps
90/100 business rows to one existing key while preserving the 10k
seed tail.
Common657 requires non-null input; nullable semantics are covered by
Rust tests.

Accepted profiles: 2.
SQL worker evidence verified: True.
The protocol requires both 8CPU profiles to contain five verified
processing windows, excluding
the first of six jobs. SQL worker samples are attributed by exact
frozen frame
rules and per-window TID. These samples describe CPU shares, not
phase wall
time or proof of elapsed speedup; unknown/inlined frames remain
disclosed.

[summary.json](summary.json) and [samples.csv](samples.csv) retain
all timing
intervals, CPU/RSS, fixture hashes and accepted profiles. Every
warmup/formal/
profile observation, source/build seal, command, snapshot/schema
digest and
Resource receipts: [complete evidence](complete-evidence.json.gz).
[Profile summary](profile-summary.json) and
[targets](profile-targets.json) retain
CPU/thread attribution; [manifest](evidence-manifest.json) seals
budget/status.

Stopped reason: None
