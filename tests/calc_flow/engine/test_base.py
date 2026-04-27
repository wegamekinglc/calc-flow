from __future__ import annotations

import pytest

from calc_flow.engine.base import Engine


def test_engine_is_abstract() -> None:
    with pytest.raises(TypeError):
        Engine()
