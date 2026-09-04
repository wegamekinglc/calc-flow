import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import App, { ARROW_TYPES, connectProject, flowNodeData } from './App';
import { at, blankProject } from './types';

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

const capabilities = {
  schemaVersion: 3,
  runtime: {
    scope: { kind: 'runtimeSession', sessionId: 'session', revision: 0 },
    packageVersion: '4.0.0',
    projectFormatVersions: [3],
    batchKinds: ['table', 'array'],
    portableArrowTypes: ['int64'],
    operators: [],
    udfs: [],
    providers: [],
    connectors: [
      {
        provider: 'builtin',
        name: 'file',
        version: '1',
        kind: 'both',
        capabilities: {
          delivery: 'exactly_once',
          replay: 'replayable_exact',
          watermark: 'generated_only',
          transaction: 'pre_commit_commit',
          snapshot: true,
          polling: true,
          cdc: false,
          lookup: false,
        },
        formats: ['json'],
        optionsSchema: {},
      },
    ],
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

const delayedTextFile = (name: string) => {
  let resolve!: (value: string) => void;
  const file = {
    name,
    text: vi.fn(() => new Promise<string>((complete) => {
      resolve = complete;
    })),
  } as unknown as File;
  return { file, resolve: (value: string) => resolve(value) };
};

const commitDataSourceText = (sourceLabel: string, dataText: string) => {
  fireEvent.click(
    screen.getByRole('button', { name: `Edit data source ${sourceLabel}` }),
  );
  fireEvent.change(
    screen.getByRole('textbox', {
      name: `Data source data for ${sourceLabel}`,
    }),
    { target: { value: dataText } },
  );
  fireEvent.click(screen.getByRole('button', { name: 'Confirm' }));
};

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

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

  it('restores, adjusts, and persists workspace panel widths', async () => {
    localStorage.setItem('calc-flow-studio:panel-layout:v1', JSON.stringify({
      version: 1,
      toolbox: 300,
      inspector: 410,
      metrics: 360,
    }));
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL) => {
        const path = String(input);
        if (path.endsWith('/catalog')) return response(catalog);
      if (path.endsWith('/capabilities')) return response(capabilities);
        if (path.endsWith('/projects')) return response([]);
        throw new Error(`Unexpected request ${path}`);
      }),
    );
    const { container } = render(<App />);

    await screen.findByText('Build the flow');
    const shell = container.querySelector<HTMLElement>('.studio-shell');
    expect(shell?.style.getPropertyValue('--toolbox-width')).toBe('300px');
    expect(shell?.style.getPropertyValue('--inspector-width')).toBe('410px');

    fireEvent.keyDown(screen.getByRole('separator', { name: 'Resize Toolbox' }), {
      key: 'ArrowRight',
    });
    expect(shell?.style.getPropertyValue('--toolbox-width')).toBe('316px');
    await waitFor(() => expect(JSON.parse(
      localStorage.getItem('calc-flow-studio:panel-layout:v1') ?? '{}',
    )).toMatchObject({ version: 1, toolbox: 316, inspector: 410 }));
  });

  it('uses default workspace widths when saved layout JSON is corrupt', async () => {
    localStorage.setItem('calc-flow-studio:panel-layout:v1', '{bad json');
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL) => {
        const path = String(input);
        if (path.endsWith('/catalog')) return response(catalog);
      if (path.endsWith('/capabilities')) return response(capabilities);
        if (path.endsWith('/projects')) return response([]);
        throw new Error(`Unexpected request ${path}`);
      }),
    );
    const { container } = render(<App />);

    await screen.findByText('Build the flow');
    const shell = container.querySelector<HTMLElement>('.studio-shell');
    expect(shell?.style.getPropertyValue('--toolbox-width')).toBe('235px');
    expect(shell?.style.getPropertyValue('--inspector-width')).toBe('335px');
  });

  it('maps external graph handles from configured ports without defaults', () => {
    const base = at(blankProject().graph.nodes);
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

    expect(project.graph.edges).toEqual([]);
    expect(reconnected.graph.edges).toEqual([
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
      if (path.endsWith('/capabilities')) return response(capabilities);
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

  it('offers the bounded Join factory only in stream mode with explicit limits', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL) => {
        const path = String(input);
        if (path.endsWith('/catalog')) return response(catalog);
        if (path.endsWith('/capabilities')) return response(capabilities);
        if (path.endsWith('/projects')) return response([]);
        throw new Error(`Unexpected request ${path}`);
      }),
    );

    render(<App />);

    await screen.findByText('Build the flow');
    expect(screen.queryByRole('button', { name: /Stream Join/i })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Stream' }));
    fireEvent.click(screen.getByRole('button', { name: /Stream Join/i }));

    expect(screen.getByLabelText('before micros')).toHaveValue(0);
    expect(screen.getByLabelText('max state rows per side')).toHaveValue(100_000);
    expect(screen.getByLabelText('max state bytes per side')).toHaveValue(134_217_728);
    expect(screen.getByLabelText('max matches per input batch')).toHaveValue(1_000_000);
  });

  it('persists a SQL alias rename across its schema port and incoming edge', async () => {
    const base = blankProject();
    const expression = at(base.graph.nodes);
    const loadedProject = {
      ...base,
      id: 'alias_project',
      name: 'Alias project',
      graph: {
        ...base.graph,
        nodes: [
          {
            id: 'join',
            operator: {
              kind: 'sql' as const,
              query: 'SELECT * FROM left JOIN right USING (id)',
              aliases: ['left', 'right'],
              udfs: [],
            },
            input_ports: [
              { name: 'left', kind: 'table' as const, required: true, schema: [] },
              {
                name: 'right',
                kind: 'table' as const,
                required: true,
                schema: [{ name: 'id', data_type: 'int64', nullable: false }],
              },
            ],
            output_ports: [],
            position: { x: 400, y: 100 },
          },
          { ...expression, id: 'left_branch', position: { x: 80, y: 40 } },
          { ...expression, id: 'right_branch', position: { x: 80, y: 220 } },
        ],
        edges: [
          {
            source_node: 'left_branch',
            source_port: 'output',
            target_node: 'join',
            target_port: 'left',
          },
          {
            source_node: 'right_branch',
            source_port: 'output',
            target_node: 'join',
            target_port: 'right',
          },
        ],
      },
    };
    const summaries = [{
      id: loadedProject.id,
      name: loadedProject.name,
      description: loadedProject.description,
      node_count: loadedProject.graph.nodes.length,
    }];
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith('/catalog')) return response(catalog);
      if (path.endsWith('/capabilities')) return response(capabilities);
      if (path.endsWith('/projects') && !init?.method) return response(summaries);
      if (path.endsWith('/projects/alias_project') && !init?.method) {
        return response(loadedProject);
      }
      if (path.endsWith('/projects/alias_project') && init?.method === 'PUT') {
        return response(JSON.parse(String(init.body)));
      }
      throw new Error(`Unexpected request ${path}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);

    const right = await screen.findByLabelText('Input alias 2');
    fireEvent.change(right, { target: { value: 'rhs' } });
    fireEvent.keyDown(right, { key: 'Enter' });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() => expect(
      fetchMock.mock.calls.some(([, init]) => init?.method === 'PUT'),
    ).toBe(true));
    const saveCall = fetchMock.mock.calls.find(([, init]) => init?.method === 'PUT');
    const saved = JSON.parse(String(saveCall?.[1]?.body));
    expect(saved.graph.nodes[0].operator.aliases).toEqual(['left', 'rhs']);
    expect(saved.graph.nodes[0].input_ports[1]).toEqual({
      name: 'rhs',
      kind: 'table',
      required: true,
      schema: [{ name: 'id', data_type: 'int64', nullable: false }],
    });
    expect(saved.graph.edges[1].target_port).toBe('rhs');
    expect(at(loadedProject.graph.nodes).operator).toMatchObject({
      aliases: ['left', 'right'],
    });
    expect(at(loadedProject.graph.edges, 1).target_port).toBe('right');
  });

  it('creates an unsaved draft before validating it', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith('/catalog')) return response(catalog);
      if (path.endsWith('/capabilities')) return response(capabilities);
      if (path.endsWith('/projects') && !init?.method) return response([]);
      if (path.endsWith('/projects') && init?.method === 'POST') {
        return response(
          JSON.parse(String(init.body)),
          201,
        );
      }
      if (path.includes('/projects/project_') && path.endsWith('/validate')) {
        return response({
          kind: 'valid',
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
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(6));
    const createCall = fetchMock.mock.calls.find(
      ([path, init]) => String(path).endsWith('/projects') && init?.method === 'POST',
    );
    const created = JSON.parse(String(createCall?.[1]?.body));
    expect(created.format_version).toBe(3);
    expect(created.id).toMatch(/^project_[0-9a-f]{32}$/);
    expect(created.graph.nodes[0].operator.kind).toBe('expression');
  });

  it('persists every loaded batch source with the saved-source contract', async () => {
    const loadedProject = {
      ...blankProject(),
      id: 'two_source',
      name: 'Two source flow',
      data_sources: [
        {
          id: 'left',
          input: 'left_source',
          format: 'inline_json' as const,
          data: [{ id: 1, value: 2 }],
        },
        {
          id: 'right',
          input: 'right_source',
          format: 'csv' as const,
          data: 'id,adjustment\n1,10\n',
        },
      ],
    };
    const summaries = [
      {
        id: loadedProject.id,
        name: loadedProject.name,
        description: loadedProject.description,
        node_count: loadedProject.graph.nodes.length,
      },
    ];
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith('/catalog')) return response(catalog);
      if (path.endsWith('/capabilities')) return response(capabilities);
      if (path.endsWith('/projects') && !init?.method) return response(summaries);
      if (path.endsWith('/projects/two_source') && !init?.method) {
        return response(loadedProject);
      }
      if (path.endsWith('/projects/two_source') && init?.method === 'PUT') {
        return response(JSON.parse(String(init.body)));
      }
      throw new Error(`Unexpected request ${path}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);

    await waitFor(() =>
      expect(screen.getByLabelText('Source ID 1')).toHaveValue('left'),
    );
    expect(screen.getByLabelText('Source ID 2')).toHaveValue('right');
    commitDataSourceText('left', '[{"id":1,"value":4}]');

    fireEvent.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(
          ([path, init]) =>
            String(path).endsWith('/projects/two_source') && init?.method === 'PUT',
        ),
      ).toBe(true),
    );
    const saveCall = fetchMock.mock.calls.find(
      ([path, init]) =>
        String(path).endsWith('/projects/two_source') && init?.method === 'PUT',
    );
    expect(JSON.parse(String(saveCall?.[1]?.body)).data_sources).toEqual([
      {
        id: 'left',
        input: 'left_source',
        format: 'inline_json',
        data: [{ id: 1, value: 4 }],
      },
      {
        id: 'right',
        input: 'right_source',
        format: 'csv',
        data: 'id,adjustment\n1,10\n',
      },
    ]);
  });

  it('starts a continuous job and rejects stale contract state', async () => {
    class FakeEventSource {
      static readonly instances: FakeEventSource[] = [];

      readonly close = vi.fn();
      private readonly listeners = new Map<string, Set<() => void>>();

      constructor(readonly url: string) {
        FakeEventSource.instances.push(this);
      }

      addEventListener(type: string, listener: () => void) {
        const listeners = this.listeners.get(type) ?? new Set();
        listeners.add(listener);
        this.listeners.set(type, listeners);
      }

      removeEventListener(type: string, listener: () => void) {
        this.listeners.get(type)?.delete(listener);
      }

      emit(type: string) {
        this.listeners.get(type)?.forEach((listener) => listener());
      }
    }

    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith('/catalog')) return response(catalog);
      if (path.endsWith('/capabilities')) return response(capabilities);
      if (path.endsWith('/projects') && !init?.method) return response([]);
      if (path.endsWith('/projects') && init?.method === 'POST') {
        return response(JSON.parse(String(init.body)), 201);
      }
      if (path.endsWith('/jobs') && init?.method === 'POST') {
        const projectId = JSON.parse(String(init.body)).project_id;
        return response(
          {
            id: 'job_contract_error',
            project_id: projectId,
            status: 'pending',
            created_at: '2026-01-01T00:00:00Z',
            started_at: null,
            finished_at: null,
            error_code: null,
            reason_code: null,
            error: null,
          },
          202,
        );
      }
      if (path.endsWith('/jobs/job_contract_error')) {
        return response({
          id: 'job_contract_error',
          project_id: 'project_invalid',
          status: 'timed_out',
          created_at: '2026-01-01T00:00:00Z',
          started_at: null,
          finished_at: null,
          error_code: null,
          reason_code: null,
          error: null,
        });
      }
      throw new Error(`Unexpected request ${path}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('EventSource', FakeEventSource);
    const { container } = render(<App />);

    await screen.findByText('Build the flow');
    fireEvent.click(screen.getByRole('button', { name: 'Stream' }));
    fireEvent.click(screen.getByRole('button', { name: /Start job/ }));
    await waitFor(() => expect(FakeEventSource.instances).toHaveLength(1));
    expect(container.querySelector('.status-pill')).toHaveTextContent('pending');

    act(() => {
      at(FakeEventSource.instances).emit('state');
    });

    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(
      "job.status: expected 'pending' or 'running' or 'completed' or 'failed' or 'cancelled'",
    ));
    expect(container.querySelector('.status-pill')).not.toBeInTheDocument();
    expect(at(FakeEventSource.instances).close).toHaveBeenCalledOnce();
  });

  it('blocks every persistence action when a source draft is invalid', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith('/catalog')) return response(catalog);
      if (path.endsWith('/capabilities')) return response(capabilities);
      if (path.endsWith('/projects') && !init?.method) return response([]);
      throw new Error(`Unexpected request ${path}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);

    const edit = await screen.findByRole('button', {
      name: 'Edit data source sample',
    });
    const invalidFile = {
      name: 'invalid.json',
      text: vi.fn().mockResolvedValue('[{'),
    } as unknown as File;
    fireEvent.change(screen.getByLabelText('Load file 1'), {
      target: { files: [invalidFile] },
    });
    await waitFor(() =>
      expect(screen.getByLabelText('Data 1 preview')).toHaveTextContent('[{'),
    );

    const actions = [
      screen.getByRole('button', { name: 'Save' }),
      screen.getByRole('button', { name: 'Validate' }),
    ];
    for (const action of actions) {
      fireEvent.click(action);
      expect(await screen.findByText('Data source sample contains invalid inline JSON'))
        .toBeInTheDocument();
      await waitFor(() => expect(action).toBeEnabled());
    }

    expect(edit).toHaveAttribute('aria-invalid', 'true');
    expect(
      fetchMock.mock.calls.filter(([path, init]) =>
        Boolean(init?.method)
        || /\/validate$|\/jobs$/.test(String(path)),
      ),
    ).toEqual([]);

    fireEvent.change(screen.getByLabelText('Format 1'), {
      target: { value: 'csv' },
    });
    expect(edit).toHaveAttribute('aria-invalid', 'false');
  });

  it('replaces committed source drafts when switching projects', async () => {
    const first = {
      ...blankProject(),
      id: 'first',
      name: 'First flow',
      data_sources: [
        {
          id: 'left',
          input: 'left_source',
          format: 'inline_json' as const,
          data: [{ value: 1 }],
        },
      ],
    };
    const second = {
      ...blankProject(),
      id: 'second',
      name: 'Second flow',
      data_sources: [
        {
          id: 'right',
          input: 'right_source',
          format: 'csv' as const,
          data: 'value\n2\n',
        },
      ],
    };
    const summaries = [first, second].map((item) => ({
      id: item.id,
      name: item.name,
      description: item.description,
      node_count: item.graph.nodes.length,
    }));
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
        const path = String(input);
        if (path.endsWith('/catalog')) return response(catalog);
      if (path.endsWith('/capabilities')) return response(capabilities);
        if (path.endsWith('/projects') && !init?.method) return response(summaries);
        if (path.endsWith('/projects/first')) return response(first);
        if (path.endsWith('/projects/second')) return response(second);
        throw new Error(`Unexpected request ${path}`);
      }),
    );
    render(<App />);

    await waitFor(() =>
      expect(screen.getByLabelText('Source ID 1')).toHaveValue('left'),
    );
    fireEvent.change(screen.getByLabelText('Source ID 1'), {
      target: { value: 'edited-left' },
    });
    commitDataSourceText('edited-left', '[{"value":99}]');
    expect(screen.getByLabelText('Data 1 preview'))
      .toHaveTextContent('[{"value":99}]');

    fireEvent.change(screen.getByLabelText('Project'), {
      target: { value: 'second' },
    });

    await waitFor(() => {
      expect(screen.getByLabelText('Source ID 1')).toHaveValue('right');
      expect(screen.getByLabelText('Data 1 preview')).toHaveTextContent('value 2');
    });
  });

  it('isolates a delayed file read from persistence and project replacement', async () => {
    const first = {
      ...blankProject(),
      id: 'first',
      name: 'First flow',
      data_sources: [
        {
          id: 'left',
          input: 'left_source',
          format: 'inline_json' as const,
          data: [{ value: 1 }],
        },
      ],
    };
    const second = {
      ...blankProject(),
      id: 'second',
      name: 'Second flow',
      data_sources: [
        {
          id: 'right',
          input: 'right_source',
          format: 'csv' as const,
          data: 'value\n2\n',
        },
      ],
    };
    const summaries = [first, second].map((item) => ({
      id: item.id,
      name: item.name,
      description: item.description,
      node_count: item.graph.nodes.length,
    }));
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith('/catalog')) return response(catalog);
      if (path.endsWith('/capabilities')) return response(capabilities);
      if (path.endsWith('/projects') && !init?.method) return response(summaries);
      if (path.endsWith('/projects/first') && !init?.method) return response(first);
      if (path.endsWith('/projects/second') && !init?.method) return response(second);
      if (path.endsWith('/projects/first') && init?.method === 'PUT') {
        return response(JSON.parse(String(init.body)));
      }
      if (path.endsWith('/projects/second') && init?.method === 'PUT') {
        return response(JSON.parse(String(init.body)));
      }
      throw new Error(`Unexpected request ${path}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);

    await waitFor(() =>
      expect(screen.getByLabelText('Source ID 1')).toHaveValue('left'),
    );
    const delayed = delayedTextFile('left.json');
    const fileInput = screen.getByLabelText('Load file 1');
    const save = screen.getByRole('button', { name: 'Save' });
    Object.defineProperty(fileInput, 'files', {
      configurable: true,
      value: [delayed.file],
    });
    fireEvent.change(fileInput);
    expect(save).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Edit data source left' }))
      .toBeDisabled();
    fireEvent.click(save);

    expect(
      fetchMock.mock.calls.some(([, init]) => init?.method === 'PUT'),
    ).toBe(false);

    fireEvent.change(screen.getByLabelText('Project'), {
      target: { value: 'second' },
    });
    await waitFor(() => {
      expect(screen.getByLabelText('Source ID 1')).toHaveValue('right');
      expect(screen.getByLabelText('Data 1 preview')).toHaveTextContent('value 2');
      expect(screen.getByRole('button', { name: 'Edit data source right' }))
        .toBeEnabled();
    });

    await act(async () => {
      delayed.resolve('[{"value":99}]');
      await Promise.resolve();
    });
    await waitFor(() => expect(save).toBeEnabled());
    expect(screen.getByLabelText('Data 1 preview')).toHaveTextContent('value 2');

    fireEvent.click(save);
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.filter(([, init]) => init?.method === 'PUT'),
      ).toHaveLength(1),
    );
    const saveCall = fetchMock.mock.calls.find(
      ([path, init]) =>
        String(path).endsWith('/projects/second') && init?.method === 'PUT',
    );
    expect(JSON.parse(String(saveCall?.[1]?.body)).data_sources).toEqual(
      second.data_sources,
    );
  });

  it('keeps the newest same-source file when reads resolve in reverse', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith('/catalog')) return response(catalog);
      if (path.endsWith('/capabilities')) return response(capabilities);
      if (path.endsWith('/projects') && !init?.method) return response([]);
      if (path.endsWith('/projects') && init?.method === 'POST') {
        return response(JSON.parse(String(init.body)), 201);
      }
      throw new Error(`Unexpected request ${path}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);

    const edit = await screen.findByRole('button', {
      name: 'Edit data source sample',
    });
    const first = delayedTextFile('first.json');
    const second = delayedTextFile('second.json');
    const fileInput = screen.getByLabelText('Load file 1');
    const save = screen.getByRole('button', { name: 'Save' });

    fireEvent.change(fileInput, { target: { files: [first.file] } });
    fireEvent.change(fileInput, { target: { files: [second.file] } });
    expect(save).toBeDisabled();

    await act(async () => {
      second.resolve('[{"value":2}]');
      await Promise.resolve();
    });
    expect(save).toBeDisabled();
    expect(edit).toBeDisabled();
    expect(screen.getByLabelText('Data 1 preview'))
      .toHaveTextContent('[{"value":2}]');

    await act(async () => {
      first.resolve('[{"value":1}]');
      await Promise.resolve();
    });
    await waitFor(() => expect(save).toBeEnabled());
    expect(edit).toBeEnabled();
    expect(screen.getByLabelText('Data 1 preview'))
      .toHaveTextContent('[{"value":2}]');

    fireEvent.click(save);
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST'),
      ).toHaveLength(1),
    );
    const createCall = fetchMock.mock.calls.find(
      ([path, init]) =>
        String(path).endsWith('/projects') && init?.method === 'POST',
    );
    expect(JSON.parse(String(createCall?.[1]?.body)).data_sources[0].data)
      .toEqual([{ value: 2 }]);
  });

  it('rejects a pending file result after its source format changes', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn((input: RequestInfo | URL) => {
        const path = String(input);
        if (path.endsWith('/catalog')) return response(catalog);
      if (path.endsWith('/capabilities')) return response(capabilities);
        if (path.endsWith('/projects')) return response([]);
        throw new Error(`Unexpected request ${path}`);
      }),
    );
    render(<App />);

    const edit = await screen.findByRole('button', {
      name: 'Edit data source sample',
    });
    const preview = screen.getByLabelText('Data 1 preview');
    const committedBefore = preview.textContent;
    const delayed = delayedTextFile('older.json');

    fireEvent.change(screen.getByLabelText('Load file 1'), {
      target: { files: [delayed.file] },
    });
    expect(edit).toBeDisabled();

    fireEvent.change(screen.getByLabelText('Format 1'), {
      target: { value: 'csv' },
    });
    expect(screen.getByLabelText('Format 1')).toHaveValue('csv');
    fireEvent.change(screen.getByLabelText('Format 1'), {
      target: { value: 'inline_json' },
    });
    expect(screen.getByLabelText('Format 1')).toHaveValue('inline_json');

    await act(async () => {
      delayed.resolve('value\n99\n');
      await Promise.resolve();
    });

    await waitFor(() => expect(edit).toBeEnabled());
    expect(preview.textContent).toBe(committedBefore);
  });

  it('excludes a pending file read before a later confirmed manual edit', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input);
      if (path.endsWith('/catalog')) return response(catalog);
      if (path.endsWith('/capabilities')) return response(capabilities);
      if (path.endsWith('/projects') && !init?.method) return response([]);
      if (path.endsWith('/projects') && init?.method === 'POST') {
        return response(JSON.parse(String(init.body)), 201);
      }
      throw new Error(`Unexpected request ${path}`);
    });
    vi.stubGlobal('fetch', fetchMock);
    render(<App />);

    const edit = await screen.findByRole('button', {
      name: 'Edit data source sample',
    });
    const delayed = delayedTextFile('older.json');
    const fileInput = screen.getByLabelText('Load file 1');
    const save = screen.getByRole('button', { name: 'Save' });

    fireEvent.change(fileInput, { target: { files: [delayed.file] } });
    expect(save).toBeDisabled();
    expect(edit).toBeDisabled();

    await act(async () => {
      delayed.resolve('[{"value":1}]');
      await Promise.resolve();
    });
    await waitFor(() => expect(save).toBeEnabled());
    expect(edit).toBeEnabled();
    expect(screen.getByLabelText('Data 1 preview'))
      .toHaveTextContent('[{"value":1}]');

    commitDataSourceText('sample', '[{"value":7}]');
    expect(screen.getByLabelText('Data 1 preview'))
      .toHaveTextContent('[{"value":7}]');

    fireEvent.click(save);
    await waitFor(() =>
      expect(
        fetchMock.mock.calls.filter(([, init]) => init?.method === 'POST'),
      ).toHaveLength(1),
    );
    const createCall = fetchMock.mock.calls.find(
      ([path, init]) =>
        String(path).endsWith('/projects') && init?.method === 'POST',
    );
    expect(JSON.parse(String(createCall?.[1]?.body)).data_sources[0].data)
      .toEqual([{ value: 7 }]);
  });
});
