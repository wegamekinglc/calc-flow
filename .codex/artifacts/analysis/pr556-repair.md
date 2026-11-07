# PR 556 repair and resumed performance acceptance

This follow-up repairs the two Codacy findings and mainline conflicts on
[PR 556](https://github.com/wegamekinglc/calc-flow/pull/556). It resumes the
four controls declared by the [weekly review](weekly-review-20261107.md);
the earlier stop acquired no performance observations. The original dirty
stock-comparison worktree remains untouched.

## Source and corrections

Current baseline is main `2193f8ee7fbdc75c2bb6c1da35b21fcb491b8b86`.
The prepared candidate is `7a2463a89a2f4340a64cbdec90c97d7db05097ba`,
tree `bdb13822a0102b15c6d57b98f5222cda8e2d4fbb`. Its merge retains main's
effective-partition cache keys and table-map rebinding, together with this
PR's per-query native audit. Ordinary fixed-P4 plan caching remains enabled;
the experimental input-partition guard remains in main and candidate.

- The materialized-input test now separates record preparation from its
  assertions. Lizard reports complexity 6 for the test and 5 for its helper,
  below the configured limit of 8. Row counts, values, caller metadata and
  settled-job assertions remain intact.
- The exception serialization test uses a named pickle round trip. Only
  bytes serialized immediately from its own test exception are loaded. A
  documented single-line `nosemgrep` suppression covers that trusted test
  operation; no production deserializer or global scanning exclusion was
  introduced. Copy/deepcopy and actual pickle coverage remain.
- Conflicts in `CHANGELOG.md` and `datafusion.rs` are resolved against the
  latest main. Automatically merged Python documentation retains both sets
  of changes. Schema output has no drift relative to this main.

## Focused verification

The independently rebuilt unit-test binary passes 40 tests: seven input
partition tests, five template tests and 28 native join tests. Build time was
312.54 seconds, outside performance acquisition. Six relevant source hashes
match before and after execution. The unchanged ASOF accounting module was
not rerun merely to attach a new binary hash.

The two test refactors pass their five focused cases using a documented
current-Python/previous-candidate-native overlay. Ruff and format checks pass;
this early overlay does not substitute for the exact rebuilt candidate wheel.
Rust formatting, whitespace and schema checks pass after integration.

The exact rebuilt candidate wheel passes all three affected Python modules:
102 tests in 2.15 seconds. Import paths and native SHA-256 are verified before
pytest. The wheel SHA-256 is
`5cbbdd60c49e9f6b21abb93258503ec705ef00516b569cf32784b15b794f5647`;
the native module SHA-256 is
`e8193ec5f1ee931e5c86a2e92eb2bc7f6c2f0b6df02b4be20dca1bf715a8bb5a`.

Source head `7a2463a8` passes all 33 CI contexts, including Codacy, coverage,
Python, Rust and cross-platform gates (2026-11-07 07:15:34 UTC snapshot).
Runtime performance is unmeasured because both resource preflights were
rejected before any worker started. The
[independent review](../reviews/pr556-repair.md) owns the acceptance verdict;
a green source-head result is not a claim about an unverified later head.

## Declared comparison

The baseline and candidate are rebuilt as clean-source release wheels with
the same locked dependencies and options. Builds are separate from one
600-second acquisition. The first candidate preparation was withdrawn after
the driver was mistakenly launched before its cache copy completed. That
copy failed, the owned build was terminated and reaped, and a fresh directory
was prepared by one sequential copy-then-build script. No performance
observation came from the discarded preparation.

Actual baseline and candidate builds take 2,418.07 and 2,545.64 seconds;
both exceed the 15–20-minute estimates while other native builds share the
machine. These are preparation costs, not workload timing or evidence that
the repair improves or regresses runtime performance.

The fixed selection remains finite native ASOF and interval SQL at 1,000
rows, projected stream ASOF at 10,000 rows with 1,024-row batches/checkpoint
off, and a complete public Program lifecycle over a 1,000-row list input
split 640/360. The maintained paired collector uses two rounds of ten
alternating AB/BA pairs, one warmup and three timed samples per fresh process:
160 observations, 160 warmups and 480 timed samples if complete. Each paired
value is the median of those three samples. The maintained joint two-round
5% regression rule is unchanged.

Workers use CPU 0 and one Tokio/Arrow/external-pool thread; controller and
supervisor use CPU 31. Installation, fixtures, validation, warmup, startup,
statistics, postflight and cleanup belong to the original 600 seconds, with
interruption at 585 seconds. Helper/source/wheel/native fingerprints are
checked before and after collection. A native-build guard rejects concurrent
Cargo/rustc/Maturin/linker processes at preflight, every observation boundary
and postflight. Boundary observations do not continuously isolate the host;
WSL host load and power remain unobservable.

Do not repeat timing acquisitions or increase samples in search of a
favorable verdict. The explicitly authorized zero-data resource resumption
is disclosed below. Report incomplete and inconclusive results explicitly.
Canonical equal-output controls cannot characterize the old wrong-result
CTE/alias shortcut, experimental input partitions, or unimplemented future
optimization proposals.

## Resource episodes

The first resumed acquisition actually started, then stopped in preflight:
the selected CPU was below its limit, but guest mean CPU use exceeded 10%.
Rust/Cargo had exited; a C++ build in another repository was active. The
complete supervisor episode took 8.952 seconds, with zero workers, warmups,
timed samples or completed cases. All owned processes exited without signals
or survivors. The original protocol and failed-preflight records are retained
unchanged, rather than reclassified as an acquisition that never started.

The user explicitly requested waiting for other tasks to finish naturally
before measurement. A separately recorded resource resumption is therefore
prepared, conditional on an observed change in resource state. It retains the
four cases, fixed samples, CPU placement, full oracles, provenance checks and
statistics. Its only changes strengthen the native-build guard to include
observed C++ build/test drivers and reduce its supervisor budget to 590 seconds
(interrupt at 575; final cleanup at 590). Combined active time is capped at
598.953 seconds including the first episode. Resource-waiting wall time is
reported separately; the two episodes are not represented as one continuous
600-second acquisition. There is at most one scientific acquisition. If this
resumption is rejected or interrupted, preserve the partial evidence and make
no further automatic attempt.

The resource-resumed episode was also rejected in preflight: CPU 0 measured
33%, above the unchanged 10% limit. Its complete supervisor took 4.537 seconds,
with no worker, warmup, timed sample or completed case. All owned processes
exited; neither episode required a signal or left a survivor. Combined active
time was 13.489 seconds. The resource wait between the first episode's end and
the second episode's start was 1,417.933 seconds; this wait is separate from
active acquisition and from the earlier build/preparation interval.

The second protocol SHA-256 is
`382de1ab169249609bdcdcb3da069abec914a08576f3f7b485516aac660ddf6d`.
No further automatic attempt was made. None of the four runtime controls was
entered, and there is no timing verdict, confirmed speedup or evidence of
benchmark equivalence. Source fixes and their green CI do not waive the
performance requirement. Keep the PR as a draft and do not merge.

Both preflight protocols, controller summaries, launch/cleanup receipts,
release identities, test logs and exact local adapters are retained in the
[evidence archive](pr556-repair-evidence.tar.gz), with a
[per-file digest index](pr556-repair-evidence-index.json). The
[independent review](../reviews/pr556-repair.md) records the source approval
limits separately from the unresolved performance acceptance.
