from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from unittest.mock import patch

import pytest

from calc_flow import _distribution


def _installed_versions(installed: set[str]):
    def version(name: str) -> str:
        if name not in installed:
            raise PackageNotFoundError(name)
        return "2026.11.14"

    return version


@pytest.mark.parametrize(
    "installed",
    [
        set(),
        {"calc-flow-python"},
        {"calc-flow-python-v3"},
    ],
)
def test_single_distribution_is_accepted(installed: set[str]) -> None:
    with patch(
        "calc_flow._distribution.version",
        side_effect=_installed_versions(installed),
    ):
        _distribution.ensure_single_distribution()


def test_portable_and_v3_distributions_are_mutually_exclusive() -> None:
    installed = {"calc-flow-python", "calc-flow-python-v3"}
    with (
        patch(
            "calc_flow._distribution.version",
            side_effect=_installed_versions(installed),
        ),
        pytest.raises(ImportError, match="mutually exclusive.*clean environment"),
    ):
        _distribution.ensure_single_distribution()
