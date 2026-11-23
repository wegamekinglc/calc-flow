# Reproduce the native session push comparison

Run from the repository containing this report. Use Rust 1.94.0 and a Linux
environment with `taskset`, GNU `timeout`, Python 3.11 or newer and the
repository's installed `calc-flow-python` plus benchmark dependencies. The
recorded fixture used Python 3.13.9, PyArrow 24.0.0 and NumPy 2.5.1. The
controller regenerates and verifies the three file digests before starting
the workers. Apply the [registration](protocol.md) resource and budget
requirements.

Reconstruct the candidate's four changed files from the two published
heads. Create the source directories in a fresh, ignored `target/`
subdirectory. The worker overlay is identical on both sides.

```bash
set -eu
repo=$(git rev-parse --show-toplevel)
study="$repo/benchmarks/stream-session/20261123-native-push"
run="$repo/target/native-session-push-reproduction"
base=4c3cc98e2ebaa8834b8f12931c33da64856daaae
admission=2a968331b75e5f6bca38694eabfea464887245bd
cost=49683169b9a53e6bcc0ca202bcb7aed010b624a7
mkdir "$run"
git fetch origin refs/pull/674/head refs/pull/675/head
git worktree add --detach "$run/base" "$base"
git worktree add --detach "$run/head" "$base"
git -C "$run/head" restore --source="$admission" --staged --worktree -- \
    crates/calc-flow/src/runtime/streaming/session.rs \
    crates/calc-flow/src/runtime/streaming/session/tests.rs
git -C "$run/head" restore --source="$cost" --staged --worktree -- \
    crates/calc-flow/src/batch.rs crates/calc-flow/tests/batch_cost.rs
test "$(git -C "$run/head" write-tree)" = 487172223b97746bcbd8a97fc0159cc5fabf3971
for side in base head; do
    cp "$study/worker.rs" "$run/$side/crates/calc-flow/examples/native_session_sample.rs"
    (cd "$run/$side" && CARGO_BUILD_JOBS=2 CARGO_INCREMENTAL=0 \
        CARGO_TARGET_DIR="$run/cargo-$side" cargo build --locked --release \
            -p calc-flow --example native_session_sample)
    cp "$run/cargo-$side/release/examples/native_session_sample" "$run/$side-worker"
done
```

Run the two sealed release executables, on a host whose builds have
finished. Set `PYTHONPATH` to include this repository for the maintained
benchmark helpers. Use a fresh output directory for the study. The shell
deadline includes Python import, fixture generation and cleanup. Build
time is recorded separately.

```bash
base_sha=$(sha256sum "$run/base-worker" | cut -d' ' -f1)
head_sha=$(sha256sum "$run/head-worker" | cut -d' ' -f1)
PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 timeout --kill-after=2 298 \
    python "$study/collect.py" "$run/evidence" \
        "$run/base-worker" "$base_sha" "$run/head-worker" "$head_sha"
```

Each raw worker record contains all 64 nanosecond samples and the validation
flags. Pair `-1` labels each round's two excluded warmups. The controller
also saves per-process logs, resource readings, completion status and the
statistics derived from the measured pairs. The percentile targets and
paired verdict use different observation units, as specified in the
registration.
