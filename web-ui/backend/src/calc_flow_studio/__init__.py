"""Local FastAPI service for Calc Flow Studio."""

from calc_flow_studio.app import create_app, serve

__all__ = ["create_app", "serve"]
