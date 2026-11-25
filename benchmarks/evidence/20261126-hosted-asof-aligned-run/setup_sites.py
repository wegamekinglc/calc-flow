SCRIPT = """from scripts.benchmark_suite.worker import environment
import json,sys
from pathlib import Path
from calc_flow import _native
import calc_flow
print(json.dumps({
    **environment(),
    "executable": sys.executable,
    "native_path": str(Path(_native.__file__).resolve()),
    "package_path": str(Path(calc_flow.__file__).resolve()),
}))"""
