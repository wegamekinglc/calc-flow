"""Closed benchmark-only dependency addition; retain all product lock entries."""

from __future__ import annotations

import tomllib

LEGACY_FEATURES = (
    "\n[features]\nbenchmark-internals = []\n"
    'benchmark-checkpoint-off = ["benchmark-internals"]\n'
)
CLOCK_MANIFEST = (
    "\n[target.'cfg(target_os = \"linux\")'.dependencies]\n"
    'nix = { version = "=0.27.1", default-features = false, '
    'features = ["time"], optional = true }\n'
    '\n[features]\nbenchmark-internals = ["dep:nix"]\n'
    'benchmark-checkpoint-off = ["benchmark-internals"]\n'
)


def clock_lock(contents: str) -> str:
    """Add only the existing pinned nix package to calc-flow's dependency list."""
    packages = tomllib.loads(contents).get("package", [])
    _validate_nix_package(packages)
    sections = contents.split("[[package]]")
    indices = [
        index
        for index, section in enumerate(sections[1:], 1)
        if tomllib.loads(section).get("name") == "calc-flow"
    ]
    if len(indices) != 1:
        raise ValueError("benchmark clock requires one product lock entry")
    index = indices[0]
    section = sections[index]
    dependencies = tomllib.loads(section).get("dependencies", [])
    if "nix" in dependencies:
        return contents
    anchor = ' "parking_lot",\n'
    if section.count(anchor) != 1:
        raise ValueError("unrecognized product dependency lock layout")
    sections[index] = section.replace(anchor, ' "nix",\n' + anchor, 1)
    return "[[package]]".join(sections)


def _validate_nix_package(packages: list[dict]) -> None:
    nix = [package for package in packages if package["name"] == "nix"]
    if len(nix) != 1 or (
        nix[0].get("version"),
        nix[0].get("source"),
        nix[0].get("checksum"),
    ) != (
        "0.27.1",
        "registry+https://github.com/rust-lang/crates.io-index",
        "2eb04e9c688eff1c89d72b407f168cf79bb9e867a9d3323ed6c01519eb9cc053",
    ):
        raise ValueError("benchmark clock requires the exact existing nix lock package")
