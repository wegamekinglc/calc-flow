# Public stream composition verification, 2026-11-24

An async Python application executes an interval join over facts and quotes, a backward ASOF join to rates, and an integer grouped SUM changelog. The run used the local debug build of the Rust core and PyO3 extension. Checkpointing was disabled. This record covers output and state correctness across the composition. It contains no latency measurement.

The two input variants have six quote rows with valid keys. The second interleaves two additional null-key rows. Each variant retains four right-side rows after initial admission and emits six interval pairs. Five find an ASOF rate and one has no match. The script checks the inclusive tolerance boundary and a timestamp tie in the rate history. It applies every retract and upsert in received order, checks old values on retraction, and compares the reconstructed groups after each rate watermark. It also drains the stream through completion and checks for unexpected output.

| Rate watermark (µs) | Pending ASOF rows | Reconstructed groups           |
| ------------------- | ----------------- | ------------------------------ |
| 20                  | 4                 | 1 → 3, 2 → 5                   |
| 21                  | 3                 | 1 → 10, 2 → 5                  |
| 28                  | 1                 | 1 → 21, 2 → 18                 |
| 29                  | 0                 | 1 → 21, 2 → 18, 9 → null       |

The original script and its two lines of stdout are archived byte for byte. Each line contains the Arrow version, loaded native module path, input geometry, counters, received changelog rows, and reconstructed state. The native path records where the original verification loaded its extension. [Script](proof.py) · [Raw stdout](observations.ndjson)

The executed full Git tree was `80826dace2a0a4a654ca83e8daabd48f375bffc0`. The final integration tree was `1789c977c1e2618d1d2ad6aed6f53439e8bd410b`. Between those trees a test in the final `#[cfg(test)] mod tests` of `operator/join/production.rs` changed. Both identifiers describe local integration trees. The commands below reconstruct the executed tree from the public PR commits and verify the full Git tree before building. The provenance file records the constituent revisions, Rust core and binding trees, Python package tree, toolchain, Cargo profile, and file digests. [Provenance](provenance.json)

## Reproduce on Linux x86-64

Start in the evidence checkout with Rust 1.94.0, CPython 3.13.9 and Git. The script checks that the interpreter loaded the newly built extension from the adjacent runtime directory. The original build used eight Cargo jobs. An environment where CPython uses a nonstandard shared library directory may also require that directory on `LD_LIBRARY_PATH`.

```bash
set -eu
repo=$(git rev-parse --show-toplevel)
evidence="$repo/benchmarks/evidence/20261124-relational-pipeline-proof"
source_dir="$repo/target/relational-composition-reproduction"
git worktree add --detach "$source_dir" dc1fd149032c586aeb5334095f6862d935b1c8fd
cd "$source_dir"
for revision in \
    c2598d6e9c362558727d5aa8781b36cb7bb72381 \
    f3d88fb63a5df3f41d5bcc1aa6a3c58a3ab165a7 \
    1c1b25a4dfdb705dfb174edc32337d912f5f87dc \
    f34040ea135004021bbc1f9ca83b186d2edb8d0f
do
    git fetch origin "$revision"
done
git merge --no-ff --no-commit c2598d6e f3d88fb6 1c1b25a4 f34040ea
test "$(git write-tree)" = 80826dace2a0a4a654ca83e8daabd48f375bffc0
run_dir="$PWD/target/proof"
mkdir -p "$run_dir/runtime"
python3.13 -m venv "$run_dir/venv"
export PYO3_PYTHON="$run_dir/venv/bin/python"
"$PYO3_PYTHON" -m pip install pyarrow==24.0.0 numpy==2.5.1 typing-extensions==4.16.0
export CARGO_TARGET_DIR="$run_dir/cargo"
CARGO_PROFILE_DEV_DEBUG=0 CARGO_BUILD_JOBS=8 cargo build --locked -p calc-flow-python
cp -R python/calc_flow "$run_dir/runtime/"
cp "$CARGO_TARGET_DIR/debug/libcalc_flow_python.so" "$run_dir/runtime/calc_flow/_native.so"
cp "$evidence/proof.py" "$run_dir/proof.py"
PYTHONPATH="$run_dir/runtime" "$PYO3_PYTHON" "$run_dir/proof.py" > "$run_dir/observations.ndjson"
"$PYO3_PYTHON" - "$evidence/observations.ndjson" "$run_dir/observations.ndjson" <<'PY'
import json
import sys
from pathlib import Path

def observations(path):
    result = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert [row["input_geometry"] for row in result] == ["dense_admission", "mixed_null_keys"]
    for row in result:
        del row["native"]  # The absolute library path depends on the checkout.
    return result

assert observations(sys.argv[1]) == observations(sys.argv[2])
PY
```
