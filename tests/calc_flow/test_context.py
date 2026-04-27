from __future__ import annotations

from calc_flow.context import Context


def test_context_get_set() -> None:
    ctx = Context()
    ctx["key"] = "value"
    assert ctx["key"] == "value"


def test_context_init_kwargs() -> None:
    ctx = Context(a=1, b=2)
    assert len(ctx) == 2
    assert dict(ctx) == {"a": 1, "b": 2}


def test_context_delete() -> None:
    ctx = Context(x=1)
    del ctx["x"]
    assert "x" not in ctx
    assert len(ctx) == 0


def test_context_iter() -> None:
    ctx = Context(x=1, y=2)
    assert set(ctx) == {"x", "y"}


def test_context_repr() -> None:
    ctx = Context(x=1)
    assert repr(ctx) == "Context(['x'])"
