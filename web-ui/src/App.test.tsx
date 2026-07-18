import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import App, { ARROW_TYPES, connectProject, flowNodeData } from './App';
import { blankProject } from './types';

const response = (body: unknown, status = 200) =>
  Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { 'Content-Type': 'application/json' },
    }),
  );

const catalog = [
  {
    provider: 'server',
    kind: 'data_fusion_scalar',
    name: 'double_value',
    version: '1',
    signature: { input_types: ['int64'], return_type: 'int64' },
    volatility: 'immutable',
  },
];

afterEach(() => vi.unstubAllGlobals());

describe('Calc Flow Studio', () => {
  it('offers exactly the Arrow types accepted by the Rust runtime', () => {
    expect(ARROW_TYPES).toEqual([
      'bool',
      'date32',
      'date64',
      'float32',
      'float64',
      'int8',
      'int16',
      'int32',
      'int64',
      'large_string',
      'string',
      'time32[s]',
      'time64[us]',
      'timestamp[ms]',
      'timestamp[us]',
      'uint8',
      'uint16',
      'uint32',
      'uint64',
    ]);
  });

  it('maps external graph handles from configured ports without defaults', () => {
    const base = blankProject().pipeline.nodes[0];
    const source = {
      ...base,
      input_ports: [],
      output_ports: [{ name: 'rows', kind: 'table' as const, required: true, schema: [] }],
      operator: {
        kind: 'external' as const,
        provider: 'trusted',
        name: 'source',
        version: '1',
        options: {},
      },
    };
    const sink = {
      ...source,
      id: 'sink',
      input_ports: [{ name: 'rows', kind: 'table' as const, required: true, schema: [] }],
      output_ports: [],
      operator: { ...source.operator, name: 'sink' },
    };

    expect(flowNodeData(source)).toMatchObject({
      inputPorts: [],
      outputPorts: ['rows'],
    });
    expect(flowNodeData(sink)).toMatchObject({
      inputPorts: ['rows'],
      outputPorts: [],
    });
  });

  it('suppresses duplicate graph connections without mutating the project', () => {
    const project = blankProject();
    const connection = {
      source: 'source',
      target: 'calculate',
      sourceHandle: 'output',
      targetHandle: 'input',
    };

    const connected = connectProject(project, connection);
    const reconnected = connectProject(connected, connection);

    expect(project.pipeline.edges).toEqual([]);
    expect(reconnected.pipeline.edges).toEqual([
      {
        source_node: 'source',
        target_node: 'calculate',
        source_port: 'output',
        target_port: 'input',
      },
    ]);
  });

  it('loads the catalog and adds a DataFusion SQL node', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL) => {
        const path = String(input);
        if (path.endsWith('/catalog')) return response(catalog);
        if (path.endsWith('/projects')) return response([]);
        throw new Error(`Unexpected request ${path}`);
      }),
    );

    render(<App />);

    expect(await screen.findByText('Build the flow')).toBeInTheDocument();
    expect(screen.queryByText(/pandas/i)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Array expression/i })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /DataFusion SQL/i }));

    expect(screen.getByText('sql')).toBeInTheDocument();
    expect(screen.getByLabelText('DataFusion SQL')).toHaveValue('SELECT * FROM input');
  });

  it('creates an unsaved draft before validating it', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith('/catalog')) return response(catalog);
      if (path.endsWith('/projects') && !init?.method) return response([]);
      if (path.endsWith('/projects') && init?.method === 'POST') {
        return response(
          JSON.parse(String(init.body)),
          201,
        );
      }
      if (path.includes('/projects/project_') && path.endsWith('/validate')) {
        return response({
          valid: true,
          issues: [],
          fingerprint: 'abc',
        });
      }
      throw new Error(`Unexpected request ${path}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);
    await screen.findByText('Build the flow');

    fireEvent.click(screen.getByRole('button', { name: 'Validate' }));

    expect(await screen.findByText('Graph is valid')).toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(5));
    const createCall = fetchMock.mock.calls.find(
      ([path, init]) => String(path).endsWith('/projects') && init?.method === 'POST',
    );
    const created = JSON.parse(String(createCall?.[1]?.body));
    expect(created.format_version).toBe(2);
    expect(created.id).toMatch(/^project_[0-9a-f]{32}$/);
    expect(created.pipeline.nodes[0].operator.kind).toBe('expression');
  });

  it('submits parsed records with the v2 preview contract', async () => {
    class FakeEventSource {
      static readonly instances: FakeEventSource[] = [];

      readonly close = vi.fn();

      constructor(readonly url: string) {
        FakeEventSource.instances.push(this);
      }

      addEventListener() {}
      removeEventListener() {}
    }

    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith('/catalog')) return response(catalog);
      if (path.endsWith('/projects') && !init?.method) return response([]);
      if (path.endsWith('/projects') && init?.method === 'POST') {
        return response(JSON.parse(String(init.body)), 201);
      }
      if (path.includes('/projects/project_') && path.endsWith('/runs')) {
        return response(
          {
            id: 'run_1',
            project_id: path.split('/').at(-2),
            status: 'pending',
            created_at: '2026-01-01T00:00:00Z',
          },
          202,
        );
      }
      throw new Error(`Unexpected request ${path}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('EventSource', FakeEventSource);
    const { unmount } = render(<App />);
    await screen.findByText('Build the flow');

    fireEvent.click(screen.getByRole('button', { name: /Run preview/ }));

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([path]) => String(path).includes('/api/v2/projects/project_') && String(path).endsWith('/runs'),
        ),
      ).toBe(true),
    );
    const runCall = fetchMock.mock.calls.find(
      ([path]) => String(path).includes('/api/v2/projects/project_') && String(path).endsWith('/runs'),
    );
    expect(JSON.parse(String(runCall?.[1]?.body))).toEqual({
      inputs: {
        input: {
          format: 'records',
          data: [
            { a: 1, b: 2 },
            { a: 3, b: 4 },
          ],
          source_id: 'browser-preview',
        },
      },
    });

    await waitFor(() =>
      expect(FakeEventSource.instances.map(({ url }) => url)).toEqual([
        '/api/v2/runs/run_1/events',
      ]),
    );
    const [source] = FakeEventSource.instances;

    unmount();

    expect(source.close).toHaveBeenCalledOnce();
  });
});
