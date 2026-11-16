# Rolling-kernel G1–G3 baseline

Date: 2026-11-16. Source: `f4faceaa` (P2.1 harness); `f09d4e45` changes only
Rust formatting and does not alter the measured behavior. Binary SHA-256:
`1b27105bf3607eee87cc997376fc628bc0622d6a9b4cbac8db291ca27d0bb77d`.

The sealed release evidence binary was built once with
`cargo test --release -p calc-flow --lib --features benchmark-internals --no-run`
and copied to `target/rolling-kernel-v2-baseline-f4faceaa` outside the repository.
Build time was reported separately. The host is WSL2; no other build or benchmark
ran during measurement.

## Protocol

Five pre-registered cases at the rolling-operator dispatch boundary:
`1x1-dual`, `99x1-dual`, `6400x64-dual`, `64x64-dual-tick`, and
`6400x64-dual-lag1`. Ten alternating AB/BA rounds warmed both recorder arms and
collected 30 timed dispatches per case per arm. All ten rounds completed in
14.8 s of the 590 s budget, with no unmeasured cases or errors.

The baseline and candidate arms intentionally used the **same sealed binary**.
This run registers absolute baseline values; its 1.00x cross-arm ratios are not
performance verdicts. Raw baseline rounds are under
[baseline-f4faceaa-rounds/](baseline-f4faceaa-rounds/). The controller output is
[baseline-f4faceaa.json](baseline-f4faceaa.json), rendered as
[baseline-f4faceaa.md](baseline-f4faceaa.md).

## Absolute values

| Gate | Case and arm                    | Baseline value | Target          |
|------|---------------------------------|---------------:|-----------------|
| G1   | `1x1-dual`, recorder absent     | 6.264 µs       | ≤ 1 µs          |
| G2   | `6400x64-dual`, recorder present | 19.577 ns/row  | ≤ 4 ns/row      |
| G3   | `6400x64-dual`, recorder absent | 452.568 µs     | ≤ 100 µs        |

G2 is recorder-present `numeric_update` nanoseconds divided by 6,400 × 2 window
rows. G3 uses the complete rolling dispatch boundary and is therefore not
interchangeable with earlier reports that included or excluded a projection
under different build and harness configurations.
