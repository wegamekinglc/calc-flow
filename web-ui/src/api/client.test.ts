import { afterEach, describe, expect, it, vi } from 'vitest';

import { blankProject } from '../types';
import { api, ApiContractError, ApiError } from './client';

const job = (status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled') => ({
  id: 'job-1',
  project_id: 'project-1',
  status,
  created_at: '2026-01-01T00:00:00Z',
  started_at: status === 'pending' ? null : '2026-01-01T00:00:01Z',
  finished_at: status === 'completed' || status === 'failed' || status === 'cancelled'
    ? '2026-01-01T00:00:02Z'
    : null,
  error_code: status === 'failed' ? 'worker_failed' : null,
  reason_code: null,
  error: status === 'failed' ? 'worker exited' : null,
});

const capabilities = {
  schemaVersion: 3,
  runtime: {
    scope: { kind: 'runtimeSession', sessionId: 'session', revision: 0 },
    packageVersion: '4.0.0',
    projectFormatVersions: [3],
    batchKinds: ['array', 'table'],
    portableArrowTypes: ['int64'],
    operators: [{
      kind: 'expression',
      version: '1',
      inputPorts: [{ name: 'input', kind: 'table', required: true }],
      outputPorts: [{ name: 'output', kind: 'table', required: true }],
      modes: ['batch', 'stream'],
      finality: 'per_row_final',
      requiresDatafusion: true,
      stateful: false,
      microbatchInvariant: true,
      requiresWatermark: false,
      checkpointSupport: 'stateless',
      stateVersion: null,
      deterministic: true,
      replaySafe: true,
      stateLayouts: [],
    }],
    udfs: [],
    providers: [],
    connectors: [],
  },
  preview: {
    inputBatchKinds: ['table'],
    requestInputFormats: ['arrow_ipc', 'columns', 'records'],
    projectInputFormats: ['arrow_ipc', 'csv', 'inline_json', 'json'],
    workerRegistrations: [],
    limits: {
      maxInputBytes: { default: 10, minimum: 1, maximum: 10 },
      maxRows: { default: 10, minimum: 1, maximum: 10 },
      timeoutSeconds: { default: 30, minimum: 1, maximum: 300 },
      memoryLimitMb: { default: 512, minimum: 64, maximum: 4096 },
      outputRows: { default: 1000, minimum: 1, maximum: 10_000 },
    },
  },
};

afterEach(() => vi.unstubAllGlobals());

describe('API client', () => {
  it('decodes the closed capabilities document at the raw HTTP boundary', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(capabilities)))
      .mockResolvedValueOnce(new Response(JSON.stringify({ schemaVersion: 4 })));
    vi.stubGlobal('fetch', fetchMock);

    await expect(api.capabilities()).resolves.toEqual(capabilities);
    await expect(api.capabilities()).rejects.toEqual(
      new ApiContractError('capabilities schema version 4 is unsupported; expected 3'),
    );
  });

  it('loads the bare UDF catalog', async () => {
    const entries = [{
      provider: 'server',
      name: 'double_value',
      version: '1',
      kind: 'data_fusion_scalar',
      signature: { input_types: ['int64'], return_type: 'int64' },
      volatility: 'immutable',
    }];
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(entries), {
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(api.catalog()).resolves.toEqual(entries);
    expect(fetchMock).toHaveBeenCalledWith('/api/v3/catalog', expect.any(Object));
  });

  it('surfaces API detail messages and structured validation errors', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'invalid graph' }), {
        status: 422,
        statusText: 'Unprocessable Content',
      }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        detail: [{ loc: ['body', 'name'], msg: 'Field required' }],
      }), {
        status: 422,
        statusText: 'Unprocessable Content',
      }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        detail: {
          kind: 'invalid',
          valid: false,
          fingerprint: null,
          issues: [{
            path: 'graph.nodes[0].operator.spec.bounds.before_micros',
            code: 'invalid_time_bound',
            message: 'before_micros must be an integer microsecond count in 0..=9007199254740991',
          }],
        },
      }), {
        status: 422,
        statusText: 'Unprocessable Content',
      }));
    vi.stubGlobal('fetch', fetchMock);

    await expect(api.validateProject('bad')).rejects.toEqual(
      new ApiError('invalid graph', 422),
    );
    await expect(api.createProject(blankProject())).rejects.toEqual(
      new ApiError('name: Field required', 422),
    );
    await expect(api.createProject(blankProject())).rejects.toEqual(
      new ApiError(
        'graph.nodes[0].operator.spec.bounds.before_micros: before_micros must be an integer microsecond count in 0..=9007199254740991',
        422,
      ),
    );
  });

  it('rejects invalid job state before application code sees it', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(JSON.stringify({
        ...job('running'),
        status: 'timed_out',
      }))),
    );

    await expect(api.job('job-1')).rejects.toBeInstanceOf(ApiContractError);
  });

  it('uses only the exact continuous job lifecycle routes', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(job('pending'))))
      .mockResolvedValueOnce(new Response(JSON.stringify(job('running'))))
      .mockResolvedValueOnce(new Response(JSON.stringify(job('running'))))
      .mockResolvedValueOnce(new Response(JSON.stringify(job('running'))))
      .mockResolvedValueOnce(new Response(JSON.stringify(job('cancelled'))));
    vi.stubGlobal('fetch', fetchMock);

    await api.startJob('project-1');
    await api.job('job-1');
    await api.checkpointJob('job-1');
    await api.shutdownJob('job-1');
    await api.cancelJob('job-1');

    expect(fetchMock.mock.calls.map(([path]) => path)).toEqual([
      '/api/v3/jobs',
      '/api/v3/jobs/job-1',
      '/api/v3/jobs/job-1/checkpoint',
      '/api/v3/jobs/job-1/shutdown',
      '/api/v3/jobs/job-1/cancel',
    ]);
    expect(fetchMock.mock.calls.map(([, init]) => (init as RequestInit).method)).toEqual([
      'POST',
      undefined,
      'POST',
      'POST',
      'POST',
    ]);
    expect(JSON.parse(String((fetchMock.mock.calls[0]![1] as RequestInit).body))).toEqual({
      project_id: 'project-1',
    });
  });

  it('creates a v3 project with the full client-owned document', async () => {
    const created = blankProject();
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(created), { status: 201 }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await api.createProject(created);

    const init = fetchMock.mock.calls[0]![1] as RequestInit;
    expect(fetchMock.mock.calls[0]![0]).toBe('/api/v3/projects');
    expect(init.method).toBe('POST');
    expect(JSON.parse(String(init.body))).toEqual(created);
  });

  it('imports raw YAML and decodes the RFC 5987 export filename', async () => {
    const project = { ...blankProject(), id: 'project_imported' };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(project), { status: 201 }))
      .mockResolvedValueOnce(new Response('{"id":"imported"}\n', {
        headers: {
          'Content-Disposition': "attachment; filename*=UTF-8''project_%E2%9C%93.json",
        },
      }));
    vi.stubGlobal('fetch', fetchMock);

    await api.importProject('name: Imported\n', 'yaml', true);
    await expect(api.exportProject('imported', 'json')).resolves.toEqual({
      document: '{"id":"imported"}\n',
      filename: 'project_✓.json',
    });
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/v3/projects/import?format=yaml&replace=true',
      expect.objectContaining({ method: 'POST', body: 'name: Imported\n' }),
    );
  });
});
