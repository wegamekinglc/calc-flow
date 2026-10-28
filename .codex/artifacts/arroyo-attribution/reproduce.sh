#!/usr/bin/env bash
set -euo pipefail

expected_ref=b9a10d4fa4b109b810d2f5839951912b92e597f8
test "$(git rev-parse HEAD)" = "$expected_ref"
git diff --exit-code HEAD --
artifact=.codex/artifacts/arroyo-attribution
run_dir=target/arroyo-attribution-reproduction
test ! -e "$run_dir"
mkdir -p "$run_dir/bin"
uv venv --python 3.13 "$run_dir/venv"
uv pip install --python "$run_dir/venv/bin/python" \
  -r .codex/artifacts/arroyo-comparison/requirements.txt
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
"$run_dir/venv/bin/python" "$artifact/probe.py" --output "$run_dir/probes"
taskset -c 0-1 "$run_dir/venv/bin/python" "$artifact/run.py" collect \
  --arroyo "$run_dir/bin/arroyo-linux-x86_64/arroyo" \
  --output "$run_dir/results" --pairs 5 --profiles --budget 550 \
  > "$run_dir/acquisition.log" 2>&1
"$run_dir/venv/bin/python" "$artifact/audit.py" "$run_dir" > "$run_dir/audit.json"
"$run_dir/venv/bin/python" "$artifact/analyze.py" "$run_dir"
