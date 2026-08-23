from __future__ import annotations

import json
import math
from typing import Any

from pydantic import GetJsonSchemaHandler, RootModel, ValidationError, model_validator
from pydantic_core import CoreSchema, PydanticCustomError

from calc_flow import _native

type JSONValue = (
    None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
)
_MAX_JSON_DEPTH = 32


def _json_error(message: str) -> ValueError:
    return ValueError(message)


def _parse_issue_path(path: str) -> tuple[str | int, ...]:
    dotted = path.replace("[", ".").replace("]", "")
    parts: list[str | int] = []
    for segment in dotted.split("."):
        if segment == "":
            continue
        parts.append(int(segment) if segment.isdigit() else segment)
    return tuple(parts)


def _native_issues(error: Exception) -> tuple[dict[str, str], ...]:
    raw = getattr(error, "issues", ())
    return tuple(
        issue
        for issue in raw
        if isinstance(issue, dict)
        and isinstance(issue.get("path"), str)
        and isinstance(issue.get("code"), str)
        and isinstance(issue.get("message"), str)
    )


def _structured_project_error(issues: tuple[dict[str, str], ...]) -> ValidationError:
    return ValidationError.from_exception_data(
        "ProjectDocument",
        [
            {
                "type": PydanticCustomError(issue["code"], issue["message"]),
                "loc": _parse_issue_path(issue["path"]),
                "input": None,
            }
            for issue in issues
        ],
    )


def _validate_json_value(value: object) -> None:
    pending: list[tuple[object, int, frozenset[int]]] = [(value, 0, frozenset())]
    while pending:
        current, depth, ancestors = pending.pop()
        if depth > _MAX_JSON_DEPTH:
            raise _json_error(
                f"project exceeds the maximum JSON depth of {_MAX_JSON_DEPTH}"
            )
        if current is None or isinstance(current, (bool, str)):
            continue
        if isinstance(current, int):
            if not -(2**63) <= current <= 2**64 - 1:
                raise _json_error("project integer is outside the portable JSON range")
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise _json_error("project JSON numbers must be finite")
            continue
        if type(current) is dict:
            identity = id(current)
            if identity in ancestors:
                raise _json_error("project contains a cycle")
            nested_ancestors = ancestors | {identity}
            for key, child in current.items():
                if not isinstance(key, str):
                    raise _json_error("project JSON object keys must be strings")
                pending.append((child, depth + 1, nested_ancestors))
            continue
        if type(current) is list:
            identity = id(current)
            if identity in ancestors:
                raise _json_error("project contains a cycle")
            nested_ancestors = ancestors | {identity}
            pending.extend((child, depth + 1, nested_ancestors) for child in current)
            continue
        raise _json_error(
            f"project contains a non-JSON value of type {type(current).__name__}"
        )


def _canonicalize(value: object) -> dict[str, JSONValue]:
    _validate_json_value(value)
    try:
        encoded = json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
        canonical = _native.validate_project_json(encoded)
    except _native.ConfigError as error:
        issues = _native_issues(error)
        if issues:
            raise _structured_project_error(issues) from error
        raise _json_error(str(error)) from error
    except Exception as error:
        raise _json_error(str(error)) from error
    parsed = json.loads(canonical)
    if not isinstance(parsed, dict):
        raise _json_error("project root must be a JSON object")
    return parsed


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _json_error(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _openapi_project_schema(component_name: str) -> dict[str, Any]:
    reference_prefix = f"#/components/schemas/{component_name}/$defs/"

    def rewrite_references(value: object) -> object:
        if type(value) is dict:
            rewritten = {key: rewrite_references(item) for key, item in value.items()}
            reference = rewritten.get("$ref")
            if isinstance(reference, str) and reference.startswith("#/$defs/"):
                rewritten["$ref"] = reference_prefix + reference.removeprefix(
                    "#/$defs/"
                )
            return rewritten
        if type(value) is list:
            return [rewrite_references(item) for item in value]
        return value

    schema = rewrite_references(json.loads(_native.project_json_schema()))
    assert isinstance(schema, dict)
    return schema


class ProjectDocument(RootModel[dict[str, JSONValue]]):
    @model_validator(mode="before")
    @classmethod
    def _validate_with_rust(cls, value: object) -> dict[str, JSONValue]:
        return _canonicalize(value)

    @classmethod
    def model_validate_json(
        cls,
        json_data: str | bytes | bytearray,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> ProjectDocument:
        del extra
        try:
            parsed = json.loads(json_data, object_pairs_hook=_reject_duplicate_pairs)
        except RecursionError as cause:
            validation_error = _json_error(
                "project JSON exceeds the maximum nesting depth"
            )
            raise ValidationError.from_exception_data(
                cls.__name__,
                [
                    {
                        "type": "value_error",
                        "loc": (),
                        "input": json_data,
                        "ctx": {"error": validation_error},
                    }
                ],
            ) from cause
        except (TypeError, ValueError, UnicodeDecodeError) as error:
            raise ValidationError.from_exception_data(
                cls.__name__,
                [
                    {
                        "type": "value_error",
                        "loc": (),
                        "input": json_data,
                        "ctx": {"error": ValueError(str(error))},
                    }
                ],
            ) from error
        return cls.model_validate(
            parsed,
            strict=strict,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> dict[str, Any]:
        del core_schema, handler
        return _openapi_project_schema(cls.__name__)

    @classmethod
    def model_json_schema(cls, *args: object, **kwargs: object) -> dict[str, Any]:
        del args, kwargs
        return json.loads(_native.project_json_schema())

    def canonical_json(self) -> str:
        encoded = json.dumps(
            self.root, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
        return _native.validate_project_json(encoded)
