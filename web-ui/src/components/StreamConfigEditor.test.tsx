import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { at, blankProject, type ConnectorCapability } from '../types';
import { StreamConfigEditor } from './StreamConfigEditor';

const connectors: ConnectorCapability[] = [
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
];

describe('StreamConfigEditor', () => {
  it('switches atomically from batch inputs to connector bindings', () => {
    const onChange = vi.fn();
    render(
      <StreamConfigEditor
        project={blankProject()}
        connectors={connectors}
        onChange={onChange}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Stream' }));

    expect(onChange).toHaveBeenCalledOnce();
    expect(at(onChange.mock.calls)[0]).toMatchObject({
      runtime: {
        mode: 'stream',
        options: { checkpoint_interval_ms: 30_000 },
      },
      data_sources: [],
      sources: [{
        binding: 'input',
        connector: { provider: 'builtin', name: 'file', version: '1' },
      }],
      sinks: [{
        binding: 'output',
        connector: { provider: 'builtin', name: 'file', version: '1' },
        delivery: 'at_least_once',
      }],
    });
  });

  it('edits stream limits and rejects malformed option JSON locally', () => {
    const batchProject = blankProject();
    const streamProject = {
      ...batchProject,
      runtime: {
        mode: 'stream' as const,
        options: {
          checkpoint_interval_ms: 30_000,
          max_batch_rows: 10_000,
          max_batch_bytes: 64 * 1024 * 1024,
        },
      },
      data_sources: [],
      sources: [{
        binding: 'input',
        connector: { provider: 'builtin', name: 'file', version: '1' },
        format: null,
        options: { path: 'input.json', format: 'json' },
        secrets: {},
        watermark: { policy: 'disabled' as const },
        schema: [],
      }],
      sinks: [],
    };
    const onChange = vi.fn();
    render(
      <StreamConfigEditor
        project={streamProject}
        connectors={connectors}
        onChange={onChange}
      />,
    );

    fireEvent.change(screen.getByLabelText('Checkpoint interval (ms)'), {
      target: { value: '5000' },
    });
    expect(at(onChange.mock.calls)[0].runtime.options.checkpoint_interval_ms).toBe(5000);

    fireEvent.change(screen.getByLabelText('Options'), {
      target: { value: '{bad' },
    });
    expect(screen.getByText(/Expected property name|JSON/)).toBeInTheDocument();
    expect(onChange).toHaveBeenCalledOnce();
  });

  it('offers best-effort as an explicit sink delivery request', () => {
    const project = {
      ...blankProject(),
      runtime: {
        mode: 'stream' as const,
        options: {
          checkpoint_interval_ms: 30_000,
          max_batch_rows: 10_000,
          max_batch_bytes: 64 * 1024 * 1024,
        },
      },
      data_sources: [],
      sources: [],
      sinks: [{
        binding: 'output',
        connector: { provider: 'builtin', name: 'file', version: '1' },
        delivery: 'at_least_once' as const,
        format: null,
        options: {},
        secrets: {},
      }],
    };

    render(
      <StreamConfigEditor
        project={project}
        connectors={connectors}
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByRole('option', { name: 'Best effort' })).toHaveValue('best_effort');
  });
});
