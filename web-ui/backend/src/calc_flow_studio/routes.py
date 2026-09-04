"""Route registrations for the /api/v3 surface.

Each registration function closes over the dependencies it needs from
``create_app`` so route behavior is unchanged while ``app.py`` stays a
small factory.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Protocol
from urllib.parse import quote

from calc_flow import CalcFlowError, ProjectDocument, project_json_schema
from calc_flow.store import (
    export_project_json,
    export_project_yaml,
    import_project_json,
    import_project_yaml,
)
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import TypeAdapter, ValidationError
from starlette.concurrency import run_in_threadpool

from calc_flow_studio.models import (
    CapabilitiesResponse,
    JobCreateRequest,
    JobResponse,
    ProjectCreateRequest,
    ProjectSummary,
    ResourceLimits,
    RunEvent,
    RunStatus,
    ValidationReport,
)
from calc_flow_studio.run_manager import (
    CapabilitySnapshotError,
    RunManagerError,
)
from calc_flow_studio.web_errors import (
    PROJECT_INVALID_422,
    bounded_request_body,
    http_error,
    invalid_report_detail,
    native_error,
)

API_PREFIX = "/api/v3"


class ProjectStoreProtocol(Protocol):
    async def create(self, project: ProjectDocument) -> None: ...

    async def put(self, project: ProjectDocument) -> None: ...

    async def get(self, project_id: str) -> ProjectDocument: ...

    async def list(self) -> list[ProjectDocument]: ...

    async def delete(self, project_id: str) -> None: ...


class RuntimeProtocol(Protocol):
    def catalog(self) -> list[dict[str, object]]: ...

    def validation_report(self, project_json: str) -> dict[str, object]: ...


class RunManagerProtocol(Protocol):
    def capabilities(self) -> CapabilitiesResponse: ...

    def submit_job(self, project: ProjectDocument) -> JobResponse: ...

    def get_job(self, job_id: str) -> JobResponse: ...

    def wait_for_events(
        self, run_id: str, *, after_sequence: int, timeout: float
    ) -> tuple[tuple[RunEvent, ...], RunStatus]: ...

    def cancel_job(self, job_id: str) -> JobResponse: ...

    def shutdown(self) -> None: ...

    def list_jobs(self) -> tuple[JobResponse, ...]: ...

    def trigger_checkpoint(self, run_id: str) -> JobResponse: ...

    def shutdown_job(self, run_id: str) -> JobResponse: ...

    def resource_limits(self) -> ResourceLimits: ...


_VALIDATION_REPORT_ADAPTER = TypeAdapter(ValidationReport)


def register_capability_routes(
    app: FastAPI, runtime: RuntimeProtocol, run_manager: RunManagerProtocol
) -> None:
    """Attach the catalog, capability, schema, and resource-limit routes."""

    @app.get(f"{API_PREFIX}/catalog")
    def get_catalog() -> list[dict[str, object]]:
        return runtime.catalog()

    @app.get(
        f"{API_PREFIX}/capabilities",
        response_model=CapabilitiesResponse,
    )
    def get_capabilities() -> CapabilitiesResponse:
        try:
            return run_manager.capabilities()
        except CapabilitySnapshotError as error:
            raise http_error(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                str(error),
            ) from error
        except RunManagerError as error:
            raise http_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "runtime capability snapshot is unavailable for this session",
            ) from error

    @app.get(f"{API_PREFIX}/schema/project")
    def get_project_schema() -> dict[str, object]:
        schema = json.loads(project_json_schema())
        assert isinstance(schema, dict)
        return schema

    @app.get(
        f"{API_PREFIX}/resource-limits",
        response_model=ResourceLimits,
    )
    async def get_resource_limits() -> ResourceLimits:
        return run_manager.resource_limits()


def _project_summary(project: ProjectDocument) -> ProjectSummary:
    root = project.root
    graph = root["graph"]
    if not isinstance(graph, dict):
        raise ValueError("validated project graph must be an object")
    nodes = graph["nodes"]
    if not isinstance(nodes, list):
        raise ValueError("validated project graph nodes must be an array")
    return ProjectSummary(
        id=str(root["id"]),
        name=str(root["name"]),
        description=str(root["description"]),
        node_count=len(nodes),
    )


async def _runtime_validation_report(
    project: ProjectDocument, runtime: RuntimeProtocol
) -> ValidationReport:
    try:
        report = await run_in_threadpool(
            runtime.validation_report, project.canonical_json()
        )
    except CalcFlowError as error:
        raise native_error(error, operation="validate") from error
    normalized = dict(report) if isinstance(report, dict) else report
    if isinstance(normalized, dict) and "kind" not in normalized:
        valid = normalized.get("valid")
        if isinstance(valid, bool):
            normalized["kind"] = "valid" if valid else "invalid"
    try:
        return _VALIDATION_REPORT_ADAPTER.validate_python(normalized)
    except ValidationError as error:
        first = error.errors(
            include_input=False,
            include_url=False,
        )[0]
        location = ".".join(str(part) for part in first["loc"]) or "<root>"
        raise http_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "runtime validation report violates the v1 contract at "
            f"{location}: {first['msg']}",
        ) from error


def register_project_routes(
    app: FastAPI, projects: ProjectStoreProtocol, runtime: RuntimeProtocol
) -> None:
    """Attach the project CRUD, import/export, and validation routes."""

    async def stored_project(project_id: str) -> ProjectDocument:
        try:
            return await projects.get(project_id)
        except (CalcFlowError, KeyError) as error:
            raise native_error(error, operation="get") from error

    async def validate_for_storage(project: ProjectDocument) -> None:
        report = await _runtime_validation_report(project, runtime)
        if report.valid is True:
            return
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=invalid_report_detail(
                [
                    {
                        "path": issue.path,
                        "code": issue.code,
                        "message": issue.message,
                    }
                    for issue in report.issues
                ]
            ),
        )

    @app.get(f"{API_PREFIX}/projects", response_model=tuple[ProjectSummary, ...])
    async def list_projects() -> tuple[ProjectSummary, ...]:
        try:
            stored = await projects.list()
        except CalcFlowError as error:
            raise native_error(error, operation="list") from error
        summaries = (_project_summary(project) for project in stored)
        return tuple(sorted(summaries, key=lambda item: item.id))

    @app.post(
        f"{API_PREFIX}/projects",
        response_model=ProjectDocument,
        status_code=status.HTTP_201_CREATED,
        responses=PROJECT_INVALID_422,
    )
    async def create_project(request: ProjectCreateRequest) -> ProjectDocument:
        project = request.to_project()
        await validate_for_storage(project)
        try:
            await projects.create(project)
        except CalcFlowError as error:
            raise native_error(error, operation="create") from error
        return project

    @app.post(
        f"{API_PREFIX}/projects/import",
        response_model=ProjectDocument,
        status_code=status.HTTP_201_CREATED,
        responses=PROJECT_INVALID_422,
    )
    async def import_project(
        request: Request,
        format_name: str = Query(alias="format", pattern="^(json|yaml)$"),
        replace: bool = False,
    ) -> ProjectDocument:
        content = await bounded_request_body(request)
        parser = import_project_json if format_name == "json" else import_project_yaml
        try:
            project = await run_in_threadpool(parser, content)
            await validate_for_storage(project)
            if replace:
                await projects.put(project)
            else:
                await projects.create(project)
        except CalcFlowError as error:
            operation = "put" if replace else "create"
            raise native_error(error, operation=operation) from error
        return project

    @app.get(f"{API_PREFIX}/projects/{{project_id}}", response_model=ProjectDocument)
    async def get_project(project_id: str) -> ProjectDocument:
        return await stored_project(project_id)

    @app.put(
        f"{API_PREFIX}/projects/{{project_id}}",
        response_model=ProjectDocument,
        responses=PROJECT_INVALID_422,
    )
    async def put_project(
        project_id: str, request: ProjectCreateRequest
    ) -> ProjectDocument:
        project = request.to_project()
        if project.root["id"] != project_id:
            raise http_error(
                status.HTTP_409_CONFLICT,
                "path project ID does not match the document",
            )
        await validate_for_storage(project)
        try:
            await projects.put(project)
        except CalcFlowError as error:
            raise native_error(error, operation="put") from error
        return project

    @app.delete(
        f"{API_PREFIX}/projects/{{project_id}}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_project(project_id: str) -> Response:
        try:
            await projects.delete(project_id)
        except (CalcFlowError, KeyError) as error:
            raise native_error(error, operation="delete") from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/export")
    async def export_project(
        project_id: str,
        format_name: str = Query(
            default="json", alias="format", pattern="^(json|yaml)$"
        ),
    ) -> PlainTextResponse:
        project = await stored_project(project_id)
        serializer = (
            export_project_json if format_name == "json" else export_project_yaml
        )
        try:
            document = await run_in_threadpool(serializer, project)
        except CalcFlowError as error:
            raise native_error(error, operation="export") from error
        media_type = "application/json" if format_name == "json" else "application/yaml"
        filename = quote(f"{project_id}.{format_name}", safe="")
        return PlainTextResponse(
            document,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{filename}"},
        )

    @app.post(
        f"{API_PREFIX}/projects/{{project_id}}/validate",
        response_model=ValidationReport,
    )
    async def validate_stored_project(project_id: str) -> ValidationReport:
        project = await stored_project(project_id)
        return await _runtime_validation_report(project, runtime)


def register_job_routes(
    app: FastAPI, projects: ProjectStoreProtocol, run_manager: RunManagerProtocol
) -> None:
    """Attach the continuous-job lifecycle and event-stream routes."""

    async def stored_project(project_id: str) -> ProjectDocument:
        try:
            return await projects.get(project_id)
        except (CalcFlowError, KeyError) as error:
            raise native_error(error, operation="get") from error

    @app.post(
        f"{API_PREFIX}/jobs",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_job(request: JobCreateRequest) -> JobResponse:
        project = await stored_project(request.project_id)
        try:
            return await run_in_threadpool(run_manager.submit_job, project)
        except KeyError as error:
            raise http_error(status.HTTP_404_NOT_FOUND, str(error)) from error
        except (RuntimeError, ValueError) as error:
            raise http_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)
            ) from error

    @app.get(f"{API_PREFIX}/jobs", response_model=tuple[JobResponse, ...])
    async def list_jobs() -> tuple[JobResponse, ...]:
        return await run_in_threadpool(run_manager.list_jobs)

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}", response_model=JobResponse)
    async def get_job(job_id: str) -> JobResponse:
        try:
            return await run_in_threadpool(run_manager.get_job, job_id)
        except KeyError as error:
            raise http_error(status.HTTP_404_NOT_FOUND, str(error)) from error

    @app.post(
        f"{API_PREFIX}/jobs/{{job_id}}/checkpoint",
        response_model=JobResponse,
    )
    async def trigger_job_checkpoint(job_id: str) -> JobResponse:
        try:
            return await run_in_threadpool(run_manager.trigger_checkpoint, job_id)
        except KeyError as error:
            raise http_error(status.HTTP_404_NOT_FOUND, str(error)) from error
        except (RuntimeError, ValueError) as error:
            raise http_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)
            ) from error

    @app.post(
        f"{API_PREFIX}/jobs/{{job_id}}/shutdown",
        response_model=JobResponse,
    )
    async def shutdown_job(job_id: str) -> JobResponse:
        try:
            return await run_in_threadpool(run_manager.shutdown_job, job_id)
        except KeyError as error:
            raise http_error(status.HTTP_404_NOT_FOUND, str(error)) from error
        except RuntimeError as error:
            raise http_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)
            ) from error

    @app.post(
        f"{API_PREFIX}/jobs/{{job_id}}/cancel",
        response_model=JobResponse,
    )
    async def cancel_job(job_id: str) -> JobResponse:
        try:
            return await run_in_threadpool(run_manager.cancel_job, job_id)
        except KeyError as error:
            raise http_error(status.HTTP_404_NOT_FOUND, str(error)) from error
        except RuntimeError as error:
            raise http_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)
            ) from error

    @app.get(
        f"{API_PREFIX}/jobs/{{job_id}}/events",
    )
    async def get_job_events(
        job_id: str,
        last_event_id: int | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        try:
            await run_in_threadpool(run_manager.get_job, job_id)
        except KeyError as error:
            raise http_error(status.HTTP_404_NOT_FOUND, str(error)) from error

        async def stream() -> AsyncIterator[str]:
            after_sequence = last_event_id if last_event_id is not None else -1
            terminal = {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }
            yield "retry: 500\n\n"
            while True:
                events, job_status = await run_in_threadpool(
                    run_manager.wait_for_events,
                    job_id,
                    after_sequence=after_sequence,
                    timeout=10.0,
                )
                if not events:
                    if job_status in terminal:
                        return
                    yield ": keep-alive\n\n"
                    continue
                for event in events:
                    payload = json.dumps(
                        event.model_dump(mode="json", exclude_none=True),
                        separators=(",", ":"),
                    )
                    yield (
                        f"id: {event.sequence}\nevent: {event.type}\n"
                        f"data: {payload}\n\n"
                    )
                    after_sequence = event.sequence
                if job_status in terminal:
                    return

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
