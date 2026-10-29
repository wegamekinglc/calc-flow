#!/usr/bin/env bash
set -euo pipefail

test "$(git rev-parse HEAD)" = b5ae227e48ac9bb8a81583a23485e2c6064cee52
git diff --exit-code HEAD --
artifact=.codex/artifacts/arrow-ipc-audit
run_dir=target/arrow-ipc-reproduction
test ! -e "$run_dir"
mkdir -p "$run_dir"
uv venv --python 3.13 "$run_dir/venv"
uv pip install --python "$run_dir/venv/bin/python" \
  -r .codex/artifacts/arroyo-comparison/requirements.txt
if test -n "${DAL417_WHEEL:-}"; then
  printf '%s  %s\n' \
    c7353872880b2c511aab3d290aff181795809e3818007159379d0f9a02f07753 \
    "$DAL417_WHEEL" | sha256sum --check
  uv pip install --python "$run_dir/venv/bin/python" "$DAL417_WHEEL"
else
  CARGO_BUILD_JOBS=8 "$run_dir/venv/bin/maturin" build --release --locked \
    --interpreter "$run_dir/venv/bin/python" --out "$run_dir/wheels" \
    > "$run_dir/build.log" 2>&1
  uv pip install --python "$run_dir/venv/bin/python" "$run_dir"/wheels/*.whl
fi
"$run_dir/venv/bin/python" -c \
  'import hashlib,calc_flow._native as n; print(hashlib.sha256(open(n.__file__, "rb").read()).hexdigest())' \
  > "$run_dir/native.sha256"
native_sha=$(cat "$run_dir/native.sha256")
taskset -c 0-1 "$run_dir/venv/bin/python" "$artifact/collect.py" \
  --output "$run_dir/results" --pairs 5 --budget 500 --native-sha "$native_sha" \
  > "$run_dir/acquisition.log" 2>&1
"$run_dir/venv/bin/python" "$artifact/audit_results.py" "$run_dir/results"
"$run_dir/venv/bin/python" "$artifact/verify_columns.py" "$run_dir/results"
"$run_dir/venv/bin/python" "$artifact/audit_inventory.py" "$run_dir/static-inventory.json"
"$run_dir/venv/bin/python" "$artifact/probe_boundaries.py" "$run_dir/timer-probe.json"
