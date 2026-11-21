# Join production phase profile

## Outcome

A single sealed `benchmark-internals` observation identifies `run_route` as the dominant remaining ordinary Join cost. This is a diagnostic sample, **not a P50 or a performance acceptance result**.

| Stage             | Wall time   | Share of measured time | Events |
|-------------------|------------:|-----------------------:|-------:|
| `run_route`       | 460.377 ms  | 65.7%                  | 2      |
| `run_materialize` | 140.155 ms  | 20.0%                  | 246    |
| `run_probe`       | 79.144 ms   | 11.3%                  | 246    |
| `run_admit`       | 11.400 ms   | 1.6%                   | 2      |
| `run_build`       | 0.466 ms    | 0.1%                   | 246    |
| `run_commit`      | 0.373 ms    | 0.1%                   | 2      |
| `run_emit`        | 0.244 ms    | <0.1%                  | 2      |
| `run_merge`       | 0.013 ms    | <0.1%                  | 2      |

The ready-source-to-Arrow observation took 701.569 ms with instrumentation enabled. It must not be compared directly with the default-feature P50. The complete 1,000,000-row oracle passed with zero maximum absolute error, and the job completed naturally with no errors and no unfinished trace events. Trace capacity was 16,384 with no overflow.

The raw trace is retained locally at `target/p5-join-stage-profile-42d5ab31-v2/join-cpu-trace.json`; `result.json` contains the compact stage summary, ready markers, output oracle, completed job status and native identities.

## Candidate and harness

- Runtime revision: `42d5ab31`, with `benchmark-internals` explicitly enabled.
- Wheel SHA-256: `e617f146bd91e7f13908ddb8143b9f4ba6ff813a1feddadeb060a4186eae8821`.
- Loaded native SHA-256: `789e7aeb0952a050a65f46da04056095e857ac4cd039c31b6e5c21512e456430`.
- Feature release build: 718.772 seconds, separate from diagnosis.
- Diagnostic wall time including fixture and trace publication: 2.108 seconds.

`profile.py` reuses the maintained `run_stream`, sources, sink, timer and full `EngineCase` oracle. It only adds the job-owned start/marker/take calls and does not duplicate runner logic. The EngineCase loop drives execution and is closed outside the loop. One earlier script-lifecycle failure was corrected before this valid observation; it was not a runtime Join failure and was not used to select a favorable timing.

The stage-summary unit test and scoped Ruff checks passed. The next experiment caches group IDs only for Join's explicitly opted-in, logically identical key runs. Default KeyStore routing retains its original ownership behavior, and every rollback invalidates the cache.
