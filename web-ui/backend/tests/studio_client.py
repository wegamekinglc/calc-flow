"""Authenticated local TestClient for ordinary Studio route tests."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient as FastAPITestClient


class TestClient(FastAPITestClient):
    def __init__(self, app: FastAPI, **kwargs: Any) -> None:
        kwargs.setdefault("base_url", "http://127.0.0.1")
        super().__init__(app, **kwargs)
        self.headers["X-Calc-Flow-Token"] = app.state.launch_token
