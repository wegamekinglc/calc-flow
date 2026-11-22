"""Distribution exclusivity for portable and x86-64-v3 native wheels."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

_DISTRIBUTIONS = ("calc-flow-python", "calc-flow-python-v3")


def ensure_single_distribution() -> None:
    """Reject environments where both native distributions overwrite one package."""
    installed = []
    for name in _DISTRIBUTIONS:
        try:
            version(name)
        except PackageNotFoundError:
            continue
        installed.append(name)
    if len(installed) > 1:
        raise ImportError(
            "calc-flow-python and calc-flow-python-v3 are mutually exclusive; "
            "install exactly one in a clean environment"
        )
