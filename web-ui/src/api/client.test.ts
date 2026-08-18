import { afterEach, describe, expect, it, vi } from 'vitest';

import { api, ApiContractError, ApiError } from './client';
import { blankProject } from '../types';

afterEach(() => vi.unstubAllGlobals());

describe('API client', () => {
  it('decodes the closed v1 capabilities document at the raw HTTP boundary', async () => {
    const capabilities = {
      schemaVersion: 1,
      runtime: {
        scope: { kind: 'runtimeSession', sessionId: 'session', revision: 0 },
        packageVersion: '2.0.0',
        projectFormatVersions: [2],
        batchKinds: ['array', 'table'],
        portableArrowTypes: ['int64'],
        operators: [],
        udfs: [],
        providers: [],
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
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(capabilities)))
      .mockResolvedValueOnce(new Response(JSON.stringify({ schemaVersion: 2 })))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        ...capabilities,
        optionalFutureField: true,
      })));
    vi.stubGlobal('fetch', fetchMock);

    await expect(api.capabilities()).resolves.toEqual(capabilities);
    await expect(api.capabilities()).rejects.toEqual(
      new ApiContractError('capabilities schema version 2 is unsupported; expected 1'),
    );
    await expect(api.capabilities()).rejects.toBeInstanceOf(ApiContractError);
  });

  it('loads the bare v2 UDF catalog', async () => {
    const entries = [
      {
        provider: 'server',
        name: 'double_value',
        version: '1',
        kind: 'data_fusion_scalar',
        signature: { input_types: ['int64'], return_type: 'int64' },
        volatility: 'immutable',
      },
    ];
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(entries), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const catalog = await api.catalog();

    expect(catalog).toEqual(entries);
    expect(fetchMock).toHaveBeenCalledWith('/api/v3/catalog', expect.any(Object));
  });

  it('surfaces API detail messages', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: 'invalid graph' }), {
          status: 422,
          statusText: 'Unprocessable Content',
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    await expect(api.validateProject('bad')).rejects.toEqual(
      new ApiError('invalid graph', 422),
    );
  });

  it('rejects an old validation response before application code sees it', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({
          valid: true,
          issues: [],
          fingerprint: 'old-server',
        })),
      ),
    );

    await expect(api.validateProject('project')).rejects.toBeInstanceOf(
      ApiContractError,
    );
  });

  it('rejects unknown run states and output kinds from raw responses', async () => {
    const base = {
      id: 'run',
      project_id: 'project',
      created_at: '2026-01-01T00:00:00Z',
      started_at: '2026-01-01T00:00:00Z',
      finished_at: '2026-01-01T00:00:01Z',
      error: null,
      result: null,
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ ...base, status: 'unknown' })),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({
          ...base,
          status: 'completed',
          result: {
            outputs: {
              output: {
                kind: 'tensor',
                total_rows: 1,
                truncated: false,
                data: [1],
                metadata: {},
              },
            },
            node_timings: {},
            datafusion_metrics: [],
            metadata: {},
          },
        })),
      );
    vi.stubGlobal('fetch', fetchMock);

    await expect(api.run('run')).rejects.toBeInstanceOf(ApiContractError);
    await expect(api.run('run')).rejects.toEqual(
      new ApiContractError(
        "run result output 'output' has unsupported kind 'tensor'; "
        + "expected 'table' or 'array'",
      ),
    );
  });

  it('decodes table and array boundary outputs before returning a run', async () => {
    const run = {
      id: 'run',
      project_id: 'project',
      status: 'completed',
      created_at: '2026-01-01T00:00:00Z',
      started_at: '2026-01-01T00:00:00Z',
      finished_at: '2026-01-01T00:00:01Z',
      error: null,
      result: {
        outputs: {
          空表: {
            kind: 'table',
            total_rows: 0,
            truncated: false,
            schema: [{ name: '值', type: 'int64', nullable: true }],
            rows: [],
            metadata: { source: '表' },
          },
          精确数组: {
            kind: 'array',
            backend: 'numpy',
            total_rows: 2,
            truncated: false,
            data: [null, 2],
            metadata: { source: '数组' },
          },
        },
        node_timings: {},
        datafusion_metrics: [],
        metadata: {},
      },
    };
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(JSON.stringify(run))),
    );

    await expect(api.run('run')).resolves.toEqual(run);
  });

  it('formats structured FastAPI validation errors', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            detail: [{ loc: ['body', 'name'], msg: 'Field required' }],
          }),
          { status: 422, statusText: 'Unprocessable Content' },
        ),
      ),
    );

    await expect(api.createProject(blankProject())).rejects.toEqual(
      new ApiError('name: Field required', 422),
    );
  });

  it('creates a v2 project with the full client-owned document', async () => {
    const created = blankProject();
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(created), {
        status: 201,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await api.createProject(created);

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(fetchMock.mock.calls[0][0]).toBe('/api/v3/projects');
    expect(init.method).toBe('POST');
    expect(JSON.parse(String(init.body))).toEqual(created);
  });

  it('imports raw YAML and decodes the RFC 5987 export filename', async () => {
    const project = { ...blankProject(), id: 'project_imported' };
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(JSON.stringify(project), {
          status: 201,
          headers: { 'Content-Type': 'application/json' },
        }),
      )
      .mockResolvedValueOnce(
        new Response('{"id":"imported"}\n', {
          headers: {
            'Content-Type': 'application/json',
            'Content-Disposition':
              "attachment; filename*=UTF-8''project_%E2%9C%93.json",
          },
        }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await api.importProject('name: Imported\n', 'yaml', true);
    const exported = await api.exportProject('imported', 'json');

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/v3/projects/import?format=yaml&replace=true',
      expect.objectContaining({
        method: 'POST',
        body: 'name: Imported\n',
        headers: expect.objectContaining({ 'Content-Type': 'application/yaml' }),
      }),
    );
    expect(exported).toEqual({
      document: '{"id":"imported"}\n',
      filename: 'project_✓.json',
    });
  });

  it('falls back safely when an extended export filename is malformed', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response('{}\n', {
          headers: {
            'Content-Disposition':
              "attachment; filename*=UTF-8''bad%ZZ.json; filename=\"safe.json\"",
          },
        }),
      ),
    );

    await expect(api.exportProject('project_safe', 'json')).resolves.toEqual({
      document: '{}\n',
      filename: 'safe.json',
    });
  });

  it('resets a project checkpoint with the generated contract', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          pipeline_name: 'Main',
          exists: false,
          state_nodes: [],
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const checkpoint = await api.resetCheckpoint('project');

    expect(checkpoint.exists).toBe(false);
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v3/projects/project/checkpoint',
      expect.objectContaining({ method: 'DELETE' }),
    );
  });
});
