import { afterEach, describe, expect, it, vi } from 'vitest';

import { api, ApiError } from './client';

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
