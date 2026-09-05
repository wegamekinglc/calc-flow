"""Shared HTTP error shapes and request-body limits for the v3 API."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request, status

from calc_flow_studio.models import ProjectInvalidResponse

MAX_PROJECT_IMPORT_BYTES = 10 * 1024 * 1024

# Response documentation shared by every project-mutating route: either the
# ValidationReport envelope or the standard request-validation error list.
PROJECT_INVALID_422 = {
    422: {
        "description": "Invalid project document or request",
        "model": ProjectInvalidResponse,
    }
}

# Stable stream Join validation codes (spec FR58); the 422 envelope for
# malformed Join input carries these instead of a flattened message.
JOIN_ISSUE_CODES = frozenset(
    {
        "unsupported_join_type",
        "invalid_time_bound",
        "invalid_join_limit",
        "invalid_join_keys",
        "incompatible_key_type",
        "invalid_event_time",
        "invalid_output_prefix",
    }
)


def http_error(code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=code, detail=detail)


def invalid_report_detail(issues: list[dict[str, str]]) -> dict[str, object]:
    return {
        "kind": "invalid",
        "valid": False,
        "issues": issues,
        "fingerprint": None,
    }


def native_issue_dicts(error: Exception) -> list[dict[str, str]]:
    raw = getattr(error, "issues", ())
    return [
        issue
        for issue in raw
        if isinstance(issue, dict)
        and isinstance(issue.get("path"), str)
        and isinstance(issue.get("code"), str)
        and isinstance(issue.get("message"), str)
    ]


def join_validation_error_detail(error: Any) -> dict[str, object] | None:
    errors = error.errors()
    join_errors = [entry for entry in errors if entry.get("type") in JOIN_ISSUE_CODES]
    if not join_errors or len(join_errors) != len(errors):
        return None
    return invalid_report_detail(
        [
            {
                "path": join_issue_path(entry.get("loc", ())),
                "code": str(entry["type"]),
                "message": str(entry["msg"]),
            }
            for entry in join_errors
        ]
    )


def join_issue_path(loc: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for part in loc:
        if part == "body" and not parts:
            continue
        if isinstance(part, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{part}]"
            else:
                parts.append(str(part))
        else:
            parts.append(str(part))
    return ".".join(parts)


def native_error(error: Exception, *, operation: str) -> HTTPException:
    message = str(error)
    if operation in {"get", "delete"} and (
        isinstance(error, KeyError) or "not found" in message
    ):
        return http_error(status.HTTP_404_NOT_FOUND, message)
    if operation == "create" and "already exists" in message:
        return http_error(status.HTTP_409_CONFLICT, message)
    issues = native_issue_dicts(error)
    if issues:
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=invalid_report_detail(issues),
        )
    return http_error(status.HTTP_422_UNPROCESSABLE_CONTENT, message)


async def bounded_request_body(request: Request) -> bytes:
    declared_lengths = request.headers.getlist("content-length")
    if len(declared_lengths) > 1:
        raise http_error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "project import must contain at most one Content-Length header",
        )
    if declared_lengths:
        try:
            length = int(declared_lengths[0])
        except ValueError as error:
            raise http_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "project import Content-Length must be an integer",
            ) from error
        if length < 0 or length > MAX_PROJECT_IMPORT_BYTES:
            raise http_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"project import exceeds the {MAX_PROJECT_IMPORT_BYTES} byte limit",
            )

    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > MAX_PROJECT_IMPORT_BYTES:
            raise http_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"project import exceeds the {MAX_PROJECT_IMPORT_BYTES} byte limit",
            )
        content.extend(chunk)
    return bytes(content)
