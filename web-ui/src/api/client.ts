import type {
  CatalogResponse,
  CheckpointSummary,
  ProjectConfig,
  ProjectSummary,
  RunRequest,
  RunResponse,
  ValidationReport,
} from '../types';

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (body.detail) message = String(body.detail);
    } catch {
      // Keep the HTTP status when a proxy returns a non-JSON error page.
    }
    throw new ApiError(message, response.status);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  catalog: () => request<CatalogResponse>('/api/v1/catalog'),
  projects: () => request<ProjectSummary[]>('/api/v1/projects'),
  project: (id: string) => request<ProjectConfig>(`/api/v1/projects/${id}`),
  saveProject: (project: ProjectConfig) =>
    request<ProjectConfig>(`/api/v1/projects/${project.id}`, {
      method: 'PUT',
      body: JSON.stringify(project),
    }),
  deleteProject: (id: string) =>
    request<void>(`/api/v1/projects/${id}`, { method: 'DELETE' }),
  validateProject: (id: string) =>
    request<ValidationReport>(`/api/v1/projects/${id}/validate`, {
      method: 'POST',
    }),
  checkpoint: (id: string) =>
    request<CheckpointSummary>(`/api/v1/projects/${id}/checkpoint`),
  resetCheckpoint: (id: string) =>
    request<CheckpointSummary>(`/api/v1/projects/${id}/checkpoint`, {
      method: 'DELETE',
    }),
  runProject: (id: string, run: RunRequest) =>
    request<RunResponse>(`/api/v1/projects/${id}/runs`, {
      method: 'POST',
      body: JSON.stringify(run),
    }),
  run: (id: string) => request<RunResponse>(`/api/v1/runs/${id}`),
  cancelRun: (id: string) =>
    request<RunResponse>(`/api/v1/runs/${id}`, { method: 'DELETE' }),
};
