"""Keep both embedded project contracts identical to the native JSON Schema."""

from __future__ import annotations

from calc_flow import ProjectDocument
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from calc_flow_studio.models import ProjectCreateRequest


def project_openapi(app: FastAPI) -> dict:
    if app.openapi_schema is None:
        schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
        # FastAPI drops null defaults when serializing its OpenAPI model.
        for model in (ProjectCreateRequest, ProjectDocument):
            component = model.__name__
            schema["components"]["schemas"][component] = _restore_null_defaults(
                model.model_json_schema(), schema["components"]["schemas"][component]
            )
        app.openapi_schema = schema
    return app.openapi_schema


def _restore_null_defaults(canonical: object, rendered: object) -> object:
    if isinstance(canonical, dict) and isinstance(rendered, dict):
        return _restore_mapping_defaults(canonical, rendered)
    if isinstance(canonical, list) and isinstance(rendered, list):
        return [
            _restore_null_defaults(left, right)
            for left, right in zip(canonical, rendered, strict=True)
        ]
    return rendered


def _restore_mapping_defaults(canonical: dict, rendered: dict) -> dict:
    result = {
        key: _restore_null_defaults(canonical.get(key), value)
        for key, value in rendered.items()
    }
    if "default" in canonical and canonical["default"] is None:
        return {**result, "default": None}
    return result
