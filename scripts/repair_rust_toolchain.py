"""Remove the pinned CI toolchain when a hosted runner has an incomplete copy."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

TOOLCHAIN = "1.88.0"
HOST = "x86_64-unknown-linux-gnu"
METADATA = ("multirust-config.toml", "rust-installer-version")


def repair_incomplete_toolchain(rustup_home: Path) -> bool:
    """Let rustup replace a partial 1.88.0 install before adding components."""

    root = rustup_home / "toolchains" / f"{TOOLCHAIN}-{HOST}"
    metadata = root / "lib" / "rustlib"
    if not root.is_dir() or all((metadata / name).is_file() for name in METADATA):
        return False
    subprocess.run(  # nosec B603 B607 -- fixed, repository-owned rustup command
        ["rustup", "toolchain", "uninstall", TOOLCHAIN], check=True
    )
    return True


if __name__ == "__main__":
    home = Path(os.environ.get("RUSTUP_HOME", str(Path.home() / ".rustup")))
    repair_incomplete_toolchain(home)
