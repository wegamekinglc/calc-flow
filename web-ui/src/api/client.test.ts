import { afterEach, describe, expect, it, vi } from 'vitest';

import { api, ApiError } from './client';
import { blankProject } from '../types';

afterEach(() => vi.unstubAllGlobals());

describe('API client', () => {
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
    expect(fetchMock).toHaveBeenCalledWith('/api/v2/catalog', expect.any(Object));
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
    expect(fetchMock.mock.calls[0][0]).toBe('/api/v2/projects');
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
      '/api/v2/projects/import?format=yaml&replace=true',
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
      '/api/v2/projects/project/checkpoint',
      expect.objectContaining({ method: 'DELETE' }),
    );
  });
});
