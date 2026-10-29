from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any, Union

from calc_flow import _native
from calc_flow._compat import TypeAliasType, dataclass

JSONValue = TypeAliasType(
    "JSONValue",
    Union[None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"]],
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


class ProjectValidationError(ValueError):
    """Invalid project data, with stable engine codes and field locations."""

    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self._entries = tuple(dict(entry) for entry in entries)
        message = "\n".join(
            f"{'.'.join(map(str, entry['loc'])) or 'project'}: {entry['msg']}"
            for entry in entries
        )
        super().__init__(message)

    def errors(
        self,
        *,
        include_url: bool = True,
        include_context: bool = True,
        include_input: bool = True,
    ) -> list[dict[str, Any]]:
        """Return independent issue records; no external error URLs are needed."""
        del include_url, include_context
        return [
            {
                key: value
                for key, value in entry.items()
                if include_input or key != "input"
            }
            for entry in self._entries
        ]

    def error_count(self) -> int:
        return len(self._entries)


def _structured_project_error(
    issues: tuple[dict[str, str], ...],
) -> ProjectValidationError:
    return ProjectValidationError(
        [
            {
                "type": issue["code"],
                "msg": issue["message"],
                "loc": _parse_issue_path(issue["path"]),
                "input": None,
            }
            for issue in issues
        ]
    )


def _value_validation_error(error: Exception) -> ProjectValidationError:
    return ProjectValidationError(
        [
            {
                "type": "value_error",
                "loc": (),
                "msg": f"Value error, {error}",
                "input": None,
            }
        ]
    )


def _validate_json_scalar(value: object) -> None:
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, int):
        if not -(2**63) <= value <= 2**64 - 1:
            raise _json_error("project integer is outside the portable JSON range")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _json_error("project JSON numbers must be finite")
        return
    raise _json_error(
        f"project contains a non-JSON value of type {type(value).__name__}"
    )


def _validate_json_value(value: object) -> None:
    pending: list[tuple[object, int, frozenset[int]]] = [(value, 0, frozenset())]
    while pending:
        current, depth, ancestors = pending.pop()
        if depth > _MAX_JSON_DEPTH:
            raise _json_error(
                f"project exceeds the maximum JSON depth of {_MAX_JSON_DEPTH}"
            )
        if type(current) is not dict and type(current) is not list:
            _validate_json_scalar(current)
            continue
        identity = id(current)
        if identity in ancestors:
            raise _json_error("project contains a cycle")
        nested_ancestors = ancestors | {identity}
        if type(current) is dict:
            for key, child in current.items():
                if not isinstance(key, str):
                    raise _json_error("project JSON object keys must be strings")
                pending.append((child, depth + 1, nested_ancestors))
        else:
            pending.extend((child, depth + 1, nested_ancestors) for child in current)


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
    if not isinstance(schema, dict):
        raise ValueError("native project schema must be a JSON object")
    return schema


@dataclass(slots=True, init=False)
class ProjectDocument:
    """An owned project-v3 document validated by the native runtime."""

    root: dict[str, JSONValue]

    def __init__(self, root: object) -> None:
        try:
            self.root = _canonicalize(root)
        except ProjectValidationError:
            raise
        except ValueError as error:
            raise _value_validation_error(error) from error

    @classmethod
    def model_validate(
        cls,
        value: object,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> ProjectDocument:
        del strict, extra, context, by_alias, by_name
        return cls(value.root if isinstance(value, cls) else value)

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
            error = _json_error("project JSON exceeds the maximum nesting depth")
            raise _value_validation_error(error) from cause
        except (TypeError, ValueError, UnicodeDecodeError) as error:
            raise _value_validation_error(error) from error
        return cls.model_validate(
            parsed,
            strict=strict,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: object, handler: object
    ) -> dict[str, Any]:
        del core_schema, handler
        return _openapi_project_schema(cls.__name__)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: object, handler: object
    ) -> dict[str, Any]:
        """Describe the optional Pydantic protocol without importing its packages."""
        del source_type, handler
        return {
            "type": "function-plain",
            "function": {"type": "no-info", "function": cls.model_validate},
            "serialization": {
                "type": "function-plain",
                "function": cls.model_dump,
                "info_arg": False,
            },
        }

    @classmethod
    def model_json_schema(cls, *args: object, **kwargs: object) -> dict[str, Any]:
        del args, kwargs
        return json.loads(_native.project_json_schema())

    def model_dump(self, *, mode: str = "python") -> dict[str, JSONValue]:
        """Return a defensive copy of this data-only root mapping."""
        if mode not in ("python", "json"):
            raise ValueError("mode must be python or json")
        _validate_json_value(self.root)
        return deepcopy(self.root)

    def model_dump_json(
        self, *, indent: int | None = None, ensure_ascii: bool = False
    ) -> str:
        """Serialize project data without invoking arbitrary object hooks."""
        _validate_json_value(self.root)
        return json.dumps(
            self.root,
            allow_nan=False,
            ensure_ascii=ensure_ascii,
            indent=indent,
            separators=(",", ":") if indent is None else None,
        )

    def canonical_json(self) -> str:
        _validate_json_value(self.root)
        encoded = json.dumps(
            self.root, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
        return _native.validate_project_json(encoded)
