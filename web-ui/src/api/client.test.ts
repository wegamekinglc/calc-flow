import { afterEach, describe, expect, it, vi } from 'vitest';

import { api, ApiError } from './client';
import { blankProject } from '../types';

afterEach(() => vi.unstubAllGlobals());

describe('API client', () => {
  it('loads the generated catalog contract', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          config_format_version: '1',
          operators: [],
          udfs: [],
          arrow_types: ['int64'],
          limits: {},
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const catalog = await api.catalog();

    expect(catalog.config_format_version).toBe('1');
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/catalog', expect.any(Object));
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

  it('creates a project without sending a client ID', async () => {
    const created = { ...blankProject(), id: 'project_generated' };
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify(created), {
        status: 201,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await api.createProject(blankProject());

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(fetchMock.mock.calls[0][0]).toBe('/api/v1/projects');
    expect(init.method).toBe('POST');
    expect(JSON.parse(String(init.body))).not.toHaveProperty('id');
  });

  it('imports raw YAML and exports text with the server filename', async () => {
    const project = { ...blankProject(), id: 'imported' };
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
            'Content-Disposition': 'attachment; filename="imported.json"',
          },
        }),
      );
    vi.stubGlobal('fetch', fetchMock);

    await api.importProject('name: Imported\n', 'yaml', true);
    const exported = await api.exportProject('imported', 'json');

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/api/v1/projects/import?format=yaml&replace=true',
      expect.objectContaining({
        method: 'POST',
        body: 'name: Imported\n',
        headers: expect.objectContaining({ 'Content-Type': 'application/yaml' }),
      }),
    );
    expect(exported).toEqual({
      document: '{"id":"imported"}\n',
      filename: 'imported.json',
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
      '/api/v1/projects/project/checkpoint',
      expect.objectContaining({ method: 'DELETE' }),
    );
  });
});
