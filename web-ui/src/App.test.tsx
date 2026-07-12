import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import App from './App';

const response = (body: unknown, status = 200) =>
  Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { 'Content-Type': 'application/json' },
    }),
  );

const catalog = {
  config_format_version: '1',
  operators: [
    { kind: 'expression', label: 'DataFusion expression', backend_selector: false },
    { kind: 'sql', label: 'DataFusion SQL', backend_selector: false },
  ],
  udfs: [
    {
      kind: 'datafusion_scalar',
      name: 'double_value',
      version: '1',
      description: 'Double a value',
    },
  ],
  arrow_types: ['float64', 'int64', 'string'],
  limits: { max_rows: 100000 },
};

afterEach(() => vi.unstubAllGlobals());

describe('Calc Flow Studio', () => {
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
          { ...JSON.parse(String(init.body)), id: 'project_generated' },
          201,
        );
      }
      if (path.endsWith('/projects/project_generated/validate')) {
        return response({
          valid: true,
          errors: [],
          warnings: [],
          fingerprint: 'abc',
          graph_inputs: ['input'],
          graph_outputs: ['output'],
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
    expect(JSON.parse(String(createCall?.[1]?.body))).not.toHaveProperty('id');
  });
});
