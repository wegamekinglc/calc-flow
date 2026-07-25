import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { ResultsPanel } from './ResultsPanel';
import type { RunResponse } from '../types';

describe('ResultsPanel', () => {
  it('shows output rows, node timing, and DataFusion plans', () => {
    const run: RunResponse = {
      id: 'run',
      project_id: 'demo',
      status: 'completed',
      created_at: '2026-01-01T00:00:00Z',
      started_at: '2026-01-01T00:00:00Z',
      finished_at: '2026-01-01T00:00:01Z',
      error: null,
      result: {
        outputs: {
          output: {
            kind: 'table',
            total_rows: 1,
            truncated: false,
            schema: [{ name: 'total', type: 'int64', nullable: true }],
            rows: [{ total: 3 }],
            metadata: {},
          },
        },
        node_timings: {
          calculate: { duration_ns: 2_000_000, input_rows: { input: 1 }, output_rows: { output: 1 } },
        },
        datafusion_metrics: [
          {
            query_id: 1,
            node_id: 'calculate',
            planning_ns: 1_000_000,
            execution_ns: 2_000_000,
            output_rows: 1,
            logical_plan: 'Projection: total',
            physical_plan: 'ProjectionExec',
          },
        ],
        metadata: {},
      },
    };

    render(
      <ResultsPanel
        validation={null}
        run={run}
        metricsWidth={330}
        onMetricsWidthChange={vi.fn()}
        onMetricsWidthReset={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByText('output')).toBeInTheDocument();
    expect(screen.getByText('total')).toBeInTheDocument();
    expect(screen.getByText('3')).toBeInTheDocument();
    expect(screen.getByText('calculate')).toBeInTheDocument();
    expect(screen.getByText('Logical plan')).toBeInTheDocument();
  });

  it('renders exact v2 validation issues', () => {
    render(
      <ResultsPanel
        validation={{
          kind: 'invalid',
          valid: false,
          issues: [
            { path: 'pipeline.nodes[0]', code: 'invalid_expression', message: 'bad expression' },
          ],
          fingerprint: null,
        }}
        run={null}
        metricsWidth={330}
        onMetricsWidthChange={vi.fn()}
        onMetricsWidthReset={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByText('Graph needs attention')).toBeInTheDocument();
    expect(screen.getByText('bad expression')).toBeInTheDocument();
  });

  it('renders zero-length and Unicode array outputs from the generated union', () => {
    const run: RunResponse = {
      id: 'run',
      project_id: 'demo',
      status: 'completed',
      created_at: '2026-01-01T00:00:00Z',
      started_at: '2026-01-01T00:00:00Z',
      finished_at: '2026-01-01T00:00:01Z',
      error: null,
      result: {
        outputs: {
          空数组: {
            kind: 'array',
            backend: 'numpy',
            total_rows: 0,
            truncated: false,
            data: [],
            metadata: { source: '数组' },
          },
        },
        node_timings: {},
        datafusion_metrics: [],
        metadata: {},
      },
    };

    render(
      <ResultsPanel
        validation={null}
        run={run}
        metricsWidth={330}
        onMetricsWidthChange={vi.fn()}
        onMetricsWidthReset={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByText('空数组')).toBeInTheDocument();
    expect(screen.getByText('0 rows')).toBeInTheDocument();
    expect(screen.getByText('[]')).toBeInTheDocument();
  });

  it('resizes the metrics panel with an end-growing separator', () => {
    const onMetricsWidthChange = vi.fn();
    const run: RunResponse = {
      id: 'run',
      project_id: 'demo',
      status: 'completed',
      created_at: '2026-01-01T00:00:00Z',
      started_at: '2026-01-01T00:00:00Z',
      finished_at: '2026-01-01T00:00:01Z',
      error: null,
      result: {
        outputs: {
          output: {
            kind: 'table',
            total_rows: 1,
            truncated: false,
            schema: [{ name: 'total', type: 'int64', nullable: true }],
            rows: [{ total: 3 }],
            metadata: {},
          },
        },
        node_timings: {},
        datafusion_metrics: [],
        metadata: {},
      },
    };
    const { container } = render(
      <ResultsPanel
        validation={null}
        run={run}
        metricsWidth={300}
        onMetricsWidthChange={onMetricsWidthChange}
        onMetricsWidthReset={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(container.querySelector<HTMLElement>('.result-grid')?.style
      .getPropertyValue('--metrics-width')).toBe('300px');
    fireEvent.keyDown(screen.getByRole('separator', { name: 'Resize Metrics' }), {
      key: 'ArrowLeft',
    });
    expect(onMetricsWidthChange).toHaveBeenLastCalledWith(316);
  });
});
