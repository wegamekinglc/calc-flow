import type {
  CapabilitiesResponse,
  CatalogResponse,
  JobResponse,
  ProjectCreateRequest,
  ProjectDocument,
  ProjectSummary,
  ValidationReport,
} from '../types';
import {
  ApiContractError,
  decodeCapabilitiesResponse,
  decodeJobResponse,
  decodeValidationReport,
} from './decoders';

export { ApiContractError };

const API_PREFIX = '/api/v3';

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

type Decoder<T> = (value: unknown) => T;

async function request<T>(
  path: string,
  decoder: Decoder<T> | null,
  init?: RequestInit,
): Promise<T> {
  const result = await response(path, init);
  if (result.status === 204) return undefined as T;
  const value: unknown = await result.json();
  return decoder ? decoder(value) : value as T;
}

async function requestText(
  path: string,
  init?: RequestInit,
): Promise<{ document: string; filename: string | null }> {
  const result = await response(path, init);
  const disposition = result.headers.get('Content-Disposition');
  const extended = disposition?.match(/filename\*\s*=\s*([^;]+)/i)?.[1]
    .trim()
    .replace(/^"|"$/g, '');
  let filename: string | null = null;
  if (extended) {
    const encoded = extended.match(/^UTF-8''(.+)$/i)?.[1];
    if (encoded) {
      try {
        filename = decodeURIComponent(encoded);
      } catch {
        // Fall through to the quoted filename for malformed percent encoding.
      }
    }
  }
  filename ??= disposition?.match(/filename\s*=\s*"([^"]+)"/i)?.[1] ?? null;
  return { document: await result.text(), filename };
}

export const api = {
  catalog: () => request<CatalogResponse>(`${API_PREFIX}/catalog`, null),
  capabilities: () => request<CapabilitiesResponse>(
    `${API_PREFIX}/capabilities`,
    decodeCapabilitiesResponse,
  ),
  projects: () => request<ProjectSummary[]>(`${API_PREFIX}/projects`, null),
  createProject: (project: ProjectCreateRequest) =>
    request<ProjectDocument>(`${API_PREFIX}/projects`, null, {
      method: 'POST',
      body: JSON.stringify(project),
    }),
  project: (id: string) => request<ProjectDocument>(
    `${API_PREFIX}/projects/${id}`,
    null,
  ),
  saveProject: (project: ProjectDocument) =>
    request<ProjectDocument>(`${API_PREFIX}/projects/${project.id}`, null, {
      method: 'PUT',
      body: JSON.stringify(project),
    }),
  deleteProject: (id: string) =>
    request<void>(`${API_PREFIX}/projects/${id}`, null, { method: 'DELETE' }),
  importProject: (document: string, format: 'json' | 'yaml', replace = false) =>
    request<ProjectDocument>(
      `${API_PREFIX}/projects/import?format=${format}&replace=${String(replace)}`,
      null,
      {
        method: 'POST',
        headers: {
          'Content-Type': format === 'json' ? 'application/json' : 'application/yaml',
        },
        body: document,
      },
    ),
  exportProject: (id: string, format: 'json' | 'yaml') =>
    requestText(`${API_PREFIX}/projects/${id}/export?format=${format}`),
  validateProject: (id: string) =>
    request<ValidationReport>(
      `${API_PREFIX}/projects/${id}/validate`,
      decodeValidationReport,
      { method: 'POST' },
    ),
  jobs: () => request<JobResponse[]>(`${API_PREFIX}/jobs`, (value) => {
    if (!Array.isArray(value)) throw new ApiContractError('jobs: expected an array');
    return value.map(decodeJobResponse);
  }),
  startJob: (projectId: string) =>
    request<JobResponse>(`${API_PREFIX}/jobs`, decodeJobResponse, {
      method: 'POST',
      body: JSON.stringify({ project_id: projectId }),
    }),
  job: (id: string) => request<JobResponse>(
    `${API_PREFIX}/jobs/${id}`,
    decodeJobResponse,
  ),
  checkpointJob: (id: string) =>
    request<JobResponse>(
      `${API_PREFIX}/jobs/${id}/checkpoint`,
      decodeJobResponse,
      { method: 'POST' },
    ),
  shutdownJob: (id: string) =>
    request<JobResponse>(
      `${API_PREFIX}/jobs/${id}/shutdown`,
      decodeJobResponse,
      { method: 'POST' },
    ),
  cancelJob: (id: string) =>
    request<JobResponse>(
      `${API_PREFIX}/jobs/${id}/cancel`,
      decodeJobResponse,
      { method: 'POST' },
    ),
};
