#!/usr/bin/env bash
# Requires Linux x86_64, uv, gh authentication and unoccupied single_node ports.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."

comparison_root=target/risingwave-reproduction
comparison_archive=risingwave-v3.1.0-x86_64-unknown-linux.tar.gz
comparison_wheel=calc_flow_python-2026.10.21-cp313-abi3-linux_x86_64.whl
mkdir -p "$comparison_root/tools" "$comparison_root/ci-wheel"

uv venv --python 3.13.9 "$comparison_root/venv"
uv pip install --python "$comparison_root/venv/bin/python" \
  -r benchmarks/risingwave/requirements.lock
uv python install 3.12.13
comparison_python312=$(uv python find 3.12.13)
comparison_python_lib=$("$comparison_python312" -c \
  'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

if [[ ! -f "$comparison_root/tools/$comparison_archive" ]]; then
  gh release download v3.1.0 --repo risingwavelabs/risingwave \
    --pattern "$comparison_archive" --dir "$comparison_root/tools"
fi
if [[ ! -f "$comparison_root/ci-wheel/$comparison_wheel" ]]; then
  gh run download 39640281625 --repo wegamekinglc/calc-flow \
    --name python-distributions --dir "$comparison_root/ci-wheel"
fi
gh api repos/wegamekinglc/calc-flow/actions/runs/39640281625 \
  > "$comparison_root/ci-run.json"

# Check original artifacts before extracting or installing executable code.
"$comparison_root/venv/bin/python" - "$comparison_root" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
pins = {
    "tools/risingwave-v3.1.0-x86_64-unknown-linux.tar.gz":
        "3e42052148bcd5cf9df076f983005f7164a8ee77d83128ab51e02489c9a7e629",
    "ci-wheel/calc_flow_python-2026.10.21-cp313-abi3-linux_x86_64.whl":
        "1f5d7fe8e57e2a118856857d6d961301e94a3d1865392c592e26a7922cb6c695",
}
for name, wanted in pins.items():
    actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
    if actual != wanted:
        raise SystemExit(f"Artifact hash mismatch: {name}")
    print(f"Verified {name}: {actual}")
PY

tar -xzf "$comparison_root/tools/$comparison_archive" -C "$comparison_root/tools"
uv pip install --python "$comparison_root/venv/bin/python" \
  "$comparison_root/ci-wheel/$comparison_wheel"
"$comparison_root/venv/bin/ruff" check scripts/risingwave_comparison.py
"$comparison_root/venv/bin/ruff" format --check scripts/risingwave_comparison.py

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ARROW_NUM_THREADS=4 TOKIO_WORKER_THREADS=4 \
  "$comparison_root/venv/bin/python" -m scripts.risingwave_comparison \
  --risingwave "$comparison_root/tools/risingwave" \
  --python312-lib "$comparison_python_lib" \
  --wheel "$comparison_root/ci-wheel/$comparison_wheel" \
  --ci-run "$comparison_root/ci-run.json" \
  --output "$comparison_root/results"
