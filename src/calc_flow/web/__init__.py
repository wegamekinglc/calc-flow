"""Optional local FastAPI service for Calc Flow projects and previews."""

from calc_flow.web.app import create_app, serve

__all__ = ["create_app", "serve"]
