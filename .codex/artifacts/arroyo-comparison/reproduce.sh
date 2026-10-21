#!/usr/bin/env bash
set -euo pipefail

# Run from a dedicated calc-flow checkout at the recorded baseline.
expected_ref=dab98a559314bd46d866ed7a6489abcdd3810afe
test "$(git rev-parse HEAD)" = "$expected_ref"
git diff --exit-code
artifact_dir=.codex/artifacts/arroyo-comparison
run_dir=target/arroyo-comparison-reproduction
test ! -e "$run_dir"
mkdir -p "$run_dir/bin"
uv venv --python 3.13 "$run_dir/venv"
uv pip install --python "$run_dir/venv/bin/python" -r "$artifact_dir/requirements.txt"
gh release download v0.15.0 --repo ArroyoSystems/arroyo \
  --pattern arroyo-linux-x86_64.tar.gz --dir "$run_dir"
printf '%s  %s\n' \
  8aba5032e2d1ce6c8af1ba4fa9c0050912a78fe98280f73e20f68e391171f048 \
  "$run_dir/arroyo-linux-x86_64.tar.gz" | sha256sum --check
tar xzf "$run_dir/arroyo-linux-x86_64.tar.gz" -C "$run_dir/bin"
CARGO_BUILD_JOBS=8 "$run_dir/venv/bin/maturin" build --release --locked \
  --interpreter "$run_dir/venv/bin/python" --out "$run_dir/wheels" \
  > "$run_dir/build.log" 2>&1
uv pip install --python "$run_dir/venv/bin/python" "$run_dir"/wheels/*.whl
uv pip freeze --python "$run_dir/venv/bin/python" > "$run_dir/dependencies.txt"
sha256sum "$run_dir"/wheels/*.whl "$run_dir/bin/arroyo-linux-x86_64/arroyo" \
  > "$run_dir/binary-hashes.txt"
"$run_dir/venv/bin/python" "$artifact_dir/measure.py" \
  --arroyo "$run_dir/bin/arroyo-linux-x86_64/arroyo" \
  --output "$run_dir/preflight" --rows 2048 --pairs 1 --gaps 0
taskset -c 0-1 "$run_dir/venv/bin/python" "$artifact_dir/measure.py" \
  --arroyo "$run_dir/bin/arroyo-linux-x86_64/arroyo" \
  --output "$run_dir/results" --rows 32768 --pairs 5 --cpus 4-7 --budget 600 \
  > "$run_dir/acquisition.log" 2>&1
