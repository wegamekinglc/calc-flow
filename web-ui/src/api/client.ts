import type {
  CatalogResponse,
  CheckpointSummary,
  ProjectConfig,
  ProjectCreateRequest,
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

const detailMessage = (detail: unknown): string | null => {
  if (typeof detail === 'string') return detail;
  if (!Array.isArray(detail)) return null;
  const messages = detail.flatMap((item) => {
    if (!item || typeof item !== 'object') return [];
    const error = item as { loc?: unknown; msg?: unknown };
    if (typeof error.msg !== 'string') return [];
    const location = Array.isArray(error.loc)
      ? error.loc.filter((part) => part !== 'body').map(String).join('.')
      : '';
    return [`${location ? `${location}: ` : ''}${error.msg}`];
  });
  return messages.length ? messages.join('; ') : null;
};

async function response(path: string, init?: RequestInit): Promise<Response> {
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
      message = detailMessage(body.detail) ?? message;
    } catch {
      // Keep the HTTP status when a proxy returns a non-JSON error page.
    }
    throw new ApiError(message, response.status);
  }
  return response;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const result = await response(path, init);
  if (result.status === 204) return undefined as T;
  return (await result.json()) as T;
}

async function requestText(
  path: string,
  init?: RequestInit,
): Promise<{ document: string; filename: string | null }> {
  const result = await response(path, init);
  const disposition = result.headers.get('Content-Disposition');
  const filename = disposition?.match(/filename="([^"]+)"/)?.[1] ?? null;
  return { document: await result.text(), filename };
}

export const api = {
  catalog: () => request<CatalogResponse>('/api/v1/catalog'),
  projects: () => request<ProjectSummary[]>('/api/v1/projects'),
  createProject: (project: ProjectConfig | ProjectCreateRequest) => {
    const draft = 'id' in project
      ? (({ id: _id, ...content }) => content)(project)
      : project;
    return request<ProjectConfig>('/api/v1/projects', {
      method: 'POST',
      body: JSON.stringify(draft),
    });
  },
  project: (id: string) => request<ProjectConfig>(`/api/v1/projects/${id}`),
  saveProject: (project: ProjectConfig) =>
    request<ProjectConfig>(`/api/v1/projects/${project.id}`, {
      method: 'PUT',
      body: JSON.stringify(project),
    }),
  deleteProject: (id: string) =>
    request<void>(`/api/v1/projects/${id}`, { method: 'DELETE' }),
  importProject: (document: string, format: 'json' | 'yaml', replace = false) =>
    request<ProjectConfig>(
      `/api/v1/projects/import?format=${format}&replace=${String(replace)}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': format === 'json' ? 'application/json' : 'application/yaml',
        },
        body: document,
      },
    ),
  exportProject: (id: string, format: 'json' | 'yaml') =>
    requestText(`/api/v1/projects/${id}/export?format=${format}`),
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
