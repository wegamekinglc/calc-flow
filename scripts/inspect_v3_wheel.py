"""Inspect x86-64-v3 native code for vectorized P2/P4 numeric loops."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

_DESIGNATED = ("fill_mean_rows", "fill_sum_rows", "keyed_state::")
_VECTOR = re.compile(r"\b(v(?:add|sub|mul|div|min|max|mov|pack|perm|blend)[a-z0-9]+)\b")
_SCALAR = {"vmulsd", "vmulss", "vaddsd", "vaddss", "vmovsd", "vmovss", "vzeroupper"}


def find_vector_instructions(disassembly: str) -> list[str]:
    """Return AVX instructions only from designated P2/P4 functions."""
    selected: list[str] = []
    designated = False
    current_symbol = ""
    for line in disassembly.splitlines():
        symbol = re.search(r"<([^>]+)>:", line)
        if symbol is not None:
            current_symbol = symbol.group(1)
            designated = any(name in current_symbol for name in _DESIGNATED)
            continue
        if designated:
            match = _VECTOR.search(line)
            if match is not None and match.group(1) not in _SCALAR:
                selected.append(f"{current_symbol}: {line}")
    if not selected:
        raise ValueError("no vector instructions in a designated P2/P4 numeric loop")
    return selected


def inspect_v3_wheel(wheel: Path) -> list[str]:
    """Disassemble the wheel's single native module and inspect numeric loops."""
    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(wheel) as archive:
            natives = [
                name
                for name in archive.namelist()
                if name.startswith("calc_flow/_native")
                and (name.endswith(".so") or name.endswith(".pyd"))
            ]
            if len(natives) != 1:
                raise ValueError("v3 wheel must contain exactly one native module")
            native = archive.extract(natives[0], directory)
        result = subprocess.run(
            ["objdump", "-d", native], check=True, capture_output=True, text=True
        )
    return find_vector_instructions(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    for instruction in inspect_v3_wheel(args.wheel):
        print(instruction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
