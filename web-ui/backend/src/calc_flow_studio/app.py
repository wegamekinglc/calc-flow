from __future__ import annotations

import ipaddress
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol
from urllib.parse import quote

from calc_flow import (
    CalcFlowError,
    FileProjectStore,
    ProjectDocument,
    Runtime,
    project_json_schema,
)
from calc_flow.store import (
    export_project_json,
    export_project_yaml,
    import_project_json,
    import_project_yaml,
)
from fastapi import FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
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
    RunManager,
    RunManagerError,
)

API_PREFIX = "/api/v3"
MAX_PROJECT_IMPORT_BYTES = 10 * 1024 * 1024
_VALIDATION_REPORT_ADAPTER = TypeAdapter(ValidationReport)


class ProjectStoreProtocol(Protocol):
    async def create(self, project: ProjectDocument) -> None: ...

    async def put(self, project: ProjectDocument) -> None: ...

    async def get(self, project_id: str) -> ProjectDocument: ...

    async def list(self) -> list[ProjectDocument]: ...

    async def delete(self, project_id: str) -> None: ...


class RuntimeProtocol(Protocol):
    def catalog(self) -> list[dict[str, object]]: ...

    def validation_report(self, project_json: str) -> dict[str, object]: ...

    def compile_project(self, project_json: str) -> object: ...


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


def _http_error(code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=code, detail=detail)


def _native_error(error: Exception, *, operation: str) -> HTTPException:
    message = str(error)
    if operation in {"get", "delete"} and (
        isinstance(error, KeyError) or "not found" in message
    ):
        return _http_error(status.HTTP_404_NOT_FOUND, message)
    if operation == "create" and "already exists" in message:
        return _http_error(status.HTTP_409_CONFLICT, message)
    return _http_error(status.HTTP_422_UNPROCESSABLE_CONTENT, message)


async def _bounded_request_body(request: Request) -> bytes:
    declared_lengths = request.headers.getlist("content-length")
    if len(declared_lengths) > 1:
        raise _http_error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            "project import must contain at most one Content-Length header",
        )
    if declared_lengths:
        try:
            length = int(declared_lengths[0])
        except ValueError as error:
            raise _http_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                "project import Content-Length must be an integer",
            ) from error
        if length < 0 or length > MAX_PROJECT_IMPORT_BYTES:
            raise _http_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"project import exceeds the {MAX_PROJECT_IMPORT_BYTES} byte limit",
            )

    content = bytearray()
    async for chunk in request.stream():
        if len(content) + len(chunk) > MAX_PROJECT_IMPORT_BYTES:
            raise _http_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT,
                f"project import exceeds the {MAX_PROJECT_IMPORT_BYTES} byte limit",
            )
        content.extend(chunk)
    return bytes(content)


def _default_frontend_directory() -> Path | None:
    static = Path(__file__).with_name("static")
    return static if (static / "index.html").is_file() else None


def create_app(
    *,
    project_directory: str | Path = ".calc-flow-projects",
    checkpoint_directory: str | Path = ".calc-flow-checkpoints",
    project_store: ProjectStoreProtocol | None = None,
    runtime: RuntimeProtocol | None = None,
    run_manager: RunManagerProtocol | None = None,
    frontend_directory: str | Path | None = None,
) -> FastAPI:
    """Create the local-only v3 API without opening a network listener."""
    projects = project_store or FileProjectStore(project_directory)
    selected_runtime = runtime or Runtime()
    selected_run_manager = (
        run_manager
        if run_manager is not None
        else RunManager(
            runtime=selected_runtime if isinstance(selected_runtime, Runtime) else None,
            checkpoint_directory=checkpoint_directory,
        )
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await run_in_threadpool(selected_run_manager.shutdown)

    app = FastAPI(title="Calc Flow API", version="4.0.0", lifespan=lifespan)
    app.state.project_store = projects
    app.state.runtime = selected_runtime
    app.state.run_manager = selected_run_manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "Last-Event-ID"],
    )

    async def stored_project(project_id: str) -> ProjectDocument:
        try:
            return await projects.get(project_id)
        except (CalcFlowError, KeyError) as error:
            raise _native_error(error, operation="get") from error

    async def runtime_validation_report(
        project: ProjectDocument,
    ) -> ValidationReport:
        try:
            report = await run_in_threadpool(
                selected_runtime.validation_report, project.canonical_json()
            )
        except CalcFlowError as error:
            raise _native_error(error, operation="validate") from error
        normalized = dict(report) if type(report) is dict else report
        if isinstance(normalized, dict) and "kind" not in normalized:
            valid = normalized.get("valid")
            if type(valid) is bool:
                normalized["kind"] = "valid" if valid else "invalid"
        try:
            return _VALIDATION_REPORT_ADAPTER.validate_python(normalized)
        except ValidationError as error:
            first = error.errors(
                include_input=False,
                include_url=False,
            )[0]
            location = ".".join(str(part) for part in first["loc"]) or "<root>"
            raise _http_error(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "runtime validation report violates the v1 contract at "
                f"{location}: {first['msg']}",
            ) from error

    async def validate_for_storage(project: ProjectDocument) -> None:
        report = await runtime_validation_report(project)
        if report.valid is True:
            return
        details = "; ".join(
            issue.message or "invalid project" for issue in report.issues
        )
        raise _http_error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            details or "project validation failed",
        )

    @app.get(f"{API_PREFIX}/catalog")
    def get_catalog() -> list[dict[str, object]]:
        return selected_runtime.catalog()

    @app.get(
        f"{API_PREFIX}/capabilities",
        response_model=CapabilitiesResponse,
    )
    def get_capabilities() -> CapabilitiesResponse:
        try:
            return selected_run_manager.capabilities()
        except CapabilitySnapshotError as error:
            raise _http_error(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                str(error),
            ) from error
        except RunManagerError as error:
            raise _http_error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "runtime capability snapshot is unavailable for this session",
            ) from error

    @app.get(f"{API_PREFIX}/schema/project")
    def get_project_schema() -> dict[str, object]:
        schema = json.loads(project_json_schema())
        assert isinstance(schema, dict)
        return schema

    @app.get(f"{API_PREFIX}/projects", response_model=tuple[ProjectSummary, ...])
    async def list_projects() -> tuple[ProjectSummary, ...]:
        try:
            stored = await projects.list()
        except CalcFlowError as error:
            raise _native_error(error, operation="list") from error
        summaries = (_project_summary(project) for project in stored)
        return tuple(sorted(summaries, key=lambda item: item.id))

    @app.post(
        f"{API_PREFIX}/projects",
        response_model=ProjectDocument,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_project(request: ProjectCreateRequest) -> ProjectDocument:
        project = request.to_project()
        await validate_for_storage(project)
        try:
            await projects.create(project)
        except CalcFlowError as error:
            raise _native_error(error, operation="create") from error
        return project

    @app.post(
        f"{API_PREFIX}/projects/import",
        response_model=ProjectDocument,
        status_code=status.HTTP_201_CREATED,
    )
    async def import_project(
        request: Request,
        format: str = Query(pattern="^(json|yaml)$"),
        replace: bool = False,
    ) -> ProjectDocument:
        content = await _bounded_request_body(request)
        parser = import_project_json if format == "json" else import_project_yaml
        try:
            project = await run_in_threadpool(parser, content)
            await validate_for_storage(project)
            if replace:
                await projects.put(project)
            else:
                await projects.create(project)
        except CalcFlowError as error:
            operation = "put" if replace else "create"
            raise _native_error(error, operation=operation) from error
        return project

    @app.get(f"{API_PREFIX}/projects/{{project_id}}", response_model=ProjectDocument)
    async def get_project(project_id: str) -> ProjectDocument:
        return await stored_project(project_id)

    @app.put(f"{API_PREFIX}/projects/{{project_id}}", response_model=ProjectDocument)
    async def put_project(
        project_id: str, request: ProjectCreateRequest
    ) -> ProjectDocument:
        project = request.to_project()
        if project.root["id"] != project_id:
            raise _http_error(
                status.HTTP_409_CONFLICT,
                "path project ID does not match the document",
            )
        await validate_for_storage(project)
        try:
            await projects.put(project)
        except CalcFlowError as error:
            raise _native_error(error, operation="put") from error
        return project

    @app.delete(
        f"{API_PREFIX}/projects/{{project_id}}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_project(project_id: str) -> Response:
        try:
            await projects.delete(project_id)
        except (CalcFlowError, KeyError) as error:
            raise _native_error(error, operation="delete") from error
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get(f"{API_PREFIX}/projects/{{project_id}}/export")
    async def export_project(
        project_id: str,
        format: str = Query(default="json", pattern="^(json|yaml)$"),
    ) -> PlainTextResponse:
        project = await stored_project(project_id)
        serializer = export_project_json if format == "json" else export_project_yaml
        try:
            document = await run_in_threadpool(serializer, project)
        except CalcFlowError as error:
            raise _native_error(error, operation="export") from error
        media_type = "application/json" if format == "json" else "application/yaml"
        filename = quote(f"{project_id}.{format}", safe="")
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
        return await runtime_validation_report(project)

    @app.post(
        f"{API_PREFIX}/jobs",
        response_model=JobResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_job(request: JobCreateRequest) -> JobResponse:
        project = await stored_project(request.project_id)
        try:
            return await run_in_threadpool(selected_run_manager.submit_job, project)
        except KeyError as error:
            raise _http_error(status.HTTP_404_NOT_FOUND, str(error)) from error
        except (RuntimeError, ValueError) as error:
            raise _http_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)
            ) from error

    @app.get(f"{API_PREFIX}/jobs", response_model=tuple[JobResponse, ...])
    async def list_jobs() -> tuple[JobResponse, ...]:
        return await run_in_threadpool(selected_run_manager.list_jobs)

    @app.get(f"{API_PREFIX}/jobs/{{job_id}}", response_model=JobResponse)
    async def get_job(job_id: str) -> JobResponse:
        try:
            return await run_in_threadpool(selected_run_manager.get_job, job_id)
        except KeyError as error:
            raise _http_error(status.HTTP_404_NOT_FOUND, str(error)) from error

    @app.post(
        f"{API_PREFIX}/jobs/{{job_id}}/checkpoint",
        response_model=JobResponse,
    )
    async def trigger_job_checkpoint(job_id: str) -> JobResponse:
        try:
            return await run_in_threadpool(
                selected_run_manager.trigger_checkpoint, job_id
            )
        except KeyError as error:
            raise _http_error(status.HTTP_404_NOT_FOUND, str(error)) from error
        except (RuntimeError, ValueError) as error:
            raise _http_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)
            ) from error

    @app.post(
        f"{API_PREFIX}/jobs/{{job_id}}/shutdown",
        response_model=JobResponse,
    )
    async def shutdown_job(job_id: str) -> JobResponse:
        try:
            return await run_in_threadpool(selected_run_manager.shutdown_job, job_id)
        except KeyError as error:
            raise _http_error(status.HTTP_404_NOT_FOUND, str(error)) from error
        except RuntimeError as error:
            raise _http_error(
                status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)
            ) from error

    @app.post(
        f"{API_PREFIX}/jobs/{{job_id}}/cancel",
        response_model=JobResponse,
    )
    async def cancel_job(job_id: str) -> JobResponse:
        try:
            return await run_in_threadpool(selected_run_manager.cancel_job, job_id)
        except KeyError as error:
            raise _http_error(status.HTTP_404_NOT_FOUND, str(error)) from error
        except RuntimeError as error:
            raise _http_error(
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
            await run_in_threadpool(selected_run_manager.get_job, job_id)
        except KeyError as error:
            raise _http_error(status.HTTP_404_NOT_FOUND, str(error)) from error

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
                    selected_run_manager.wait_for_events,
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

    @app.get(
        f"{API_PREFIX}/resource-limits",
        response_model=ResourceLimits,
    )
    async def get_resource_limits() -> ResourceLimits:
        return selected_run_manager.resource_limits()

    frontend = (
        Path(frontend_directory)
        if frontend_directory is not None
        else _default_frontend_directory()
    )
    if frontend is not None:
        assets = frontend / "assets"
        index = frontend / "index.html"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")
        if index.is_file():

            @app.get("/", include_in_schema=False)
            def frontend_index() -> FileResponse:
                return FileResponse(index)

    return app


def validate_bind_host(host: str) -> str:
    if host == "localhost":
        return host
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        message = "Calc Flow web server host must be a loopback IP or localhost"
        raise ValueError(message) from error
    if not address.is_loopback:
        message = "Calc Flow web server may bind only to a loopback address"
        raise ValueError(message)
    return host


def serve(
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    project_directory: str | Path = ".calc-flow-projects",
    checkpoint_directory: str | Path = ".calc-flow-checkpoints",
) -> None:
    """Run the unauthenticated v3 service on a loopback interface only."""
    import uvicorn

    validate_bind_host(host)
    uvicorn.run(
        create_app(
            project_directory=project_directory,
            checkpoint_directory=checkpoint_directory,
        ),
        host=host,
        port=port,
    )
