from __future__ import annotations

import json
import math
from typing import Any, Union

from calc_flow import _native
from calc_flow._compat import TypeAliasType

JSONValue = TypeAliasType(
    "JSONValue",
    Union[None, bool, int, float, str, list["JSONValue"], dict[str, "JSONValue"]],
)
_MAX_JSON_DEPTH = 32


class ProjectValidationError(ValueError):
    __slots__ = ("_errors", "title")

    def __init__(self, title: str, errors: list[dict[str, Any]]) -> None:
        self.title = title
        self._errors = tuple(error.copy() for error in errors)
        super().__init__()

    def errors(
        self,
        *,
        include_url: bool = True,
        include_context: bool = True,
        include_input: bool = True,
    ) -> list[dict[str, Any]]:
        del include_url
        excluded = set()
        if not include_context:
            excluded.add("ctx")
        if not include_input:
            excluded.add("input")
        return [
            {key: value for key, value in error.items() if key not in excluded}
            for error in self._errors
        ]

    def error_count(self) -> int:
        return len(self._errors)

    def __str__(self) -> str:
        count = len(self._errors)
        lines = [
            f"{count} validation error{'' if count == 1 else 's'} for {self.title}"
        ]
        for error in self._errors:
            location = ".".join(str(part) for part in error["loc"]) or "__root__"
            lines.append(f"{location}\n  {error['msg']} [type={error['type']}]")
        return "\n".join(lines)


def _json_error(message: str) -> ValueError:
    return ValueError(message)


def _project_value_error(
    title: str, message: str, input_value: object
) -> ProjectValidationError:
    return ProjectValidationError(
        title,
        [
            {
                "type": "value_error",
                "loc": (),
                "msg": f"Value error, {message}",
                "input": input_value,
                "ctx": {"error": ValueError(message)},
            }
        ],
    )


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


def _structured_project_error(
    issues: tuple[dict[str, str], ...],
) -> ProjectValidationError:
    return ProjectValidationError(
        "ProjectDocument",
        [
            {
                "type": issue["code"],
                "loc": _parse_issue_path(issue["path"]),
                "msg": issue["message"],
                "input": None,
            }
            for issue in issues
        ],
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
    assert isinstance(schema, dict)
    return schema


def _exclude_none_values(value: object) -> object:
    if type(value) is dict:
        return {
            key: _exclude_none_values(item)
            for key, item in value.items()
            if item is not None
        }
    if type(value) is list:
        return [_exclude_none_values(item) for item in value]
    return value


class ProjectDocument:
    __slots__ = ("root",)

    def __init__(self, root: dict[str, JSONValue]) -> None:
        self.root = self._validated_root(root)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ProjectDocument):
            return NotImplemented
        return self.root == other.root

    @classmethod
    def _validated_root(cls, value: object) -> dict[str, JSONValue]:
        try:
            return _canonicalize(value)
        except ProjectValidationError:
            raise
        except (TypeError, ValueError, RecursionError) as error:
            raise _project_value_error(cls.__name__, str(error), value) from error

    @classmethod
    def _from_validated_root(cls, root: dict[str, JSONValue]) -> ProjectDocument:
        document = object.__new__(cls)
        document.root = root
        return document

    @classmethod
    def model_validate(
        cls,
        obj: object,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> ProjectDocument:
        del strict, extra, from_attributes, context, by_alias, by_name
        if isinstance(obj, cls):
            return obj
        return cls._from_validated_root(cls._validated_root(obj))

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
        except RecursionError as error:
            raise _project_value_error(
                cls.__name__,
                "project JSON exceeds the maximum nesting depth",
                json_data,
            ) from error
        except (TypeError, ValueError, UnicodeDecodeError) as error:
            raise _project_value_error(cls.__name__, str(error), json_data) from error
        return cls.model_validate(
            parsed,
            strict=strict,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
        )

    @staticmethod
    def _dump_root(
        root: dict[str, JSONValue],
        include: object,
        exclude: object,
        exclude_none: bool,
    ) -> dict[str, JSONValue]:
        if include is not None or exclude is not None:
            raise TypeError("ProjectDocument does not support include/exclude filters")
        if exclude_none:
            filtered = _exclude_none_values(root)
            assert isinstance(filtered, dict)
            return filtered
        return root

    def model_dump(
        self,
        *,
        mode: str = "python",
        include: object = None,
        exclude: object = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_computed_fields: bool = False,
        round_trip: bool = False,
        warnings: bool | str = True,
        fallback: object = None,
        serialize_as_any: bool = False,
        polymorphic_serialization: bool | None = None,
    ) -> dict[str, JSONValue]:
        del context, by_alias, exclude_unset, exclude_defaults
        del exclude_computed_fields, round_trip, warnings, fallback
        del serialize_as_any, polymorphic_serialization
        if mode not in {"python", "json"}:
            raise ValueError("mode must be 'python' or 'json'")
        root = self._dump_root(self.root, include, exclude, exclude_none)
        encoded = json.dumps(root, allow_nan=False, separators=(",", ":"))
        return json.loads(encoded)

    def model_dump_json(
        self,
        *,
        indent: int | None = None,
        ensure_ascii: bool = False,
        include: object = None,
        exclude: object = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        exclude_computed_fields: bool = False,
        round_trip: bool = False,
        warnings: bool | str = True,
        fallback: object = None,
        serialize_as_any: bool = False,
        polymorphic_serialization: bool | None = None,
    ) -> str:
        del context, by_alias, exclude_unset, exclude_defaults
        del exclude_computed_fields, round_trip, warnings, fallback
        del serialize_as_any, polymorphic_serialization
        root = self._dump_root(self.root, include, exclude, exclude_none)
        return json.dumps(
            root,
            indent=indent,
            ensure_ascii=ensure_ascii,
            allow_nan=False,
            separators=(",", ":") if indent is None else None,
        )

    @classmethod
    def model_json_schema(cls, *args: object, **kwargs: object) -> dict[str, Any]:
        del args, kwargs
        return json.loads(_native.project_json_schema())

    def canonical_json(self) -> str:
        encoded = json.dumps(
            self.root, allow_nan=False, separators=(",", ":"), sort_keys=True
        )
        return _native.validate_project_json(encoded)
