import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import type { JobEvent, JobResponse } from '../types';
import { ResultsPanel } from './ResultsPanel';

const runningJob: JobResponse = {
  id: 'job-1',
  project_id: 'demo',
  status: 'running',
  created_at: '2026-01-01T00:00:00Z',
  started_at: '2026-01-01T00:00:01Z',
  finished_at: null,
  error_code: null,
  error: null,
};

const progress: JobEvent = {
  sequence: 8,
  timestamp: '2026-01-01T00:00:08Z',
  type: 'progress',
  message: 'running',
  state: 'running',
  epoch: 7,
  watermark: '2026-01-01T00:00:07Z',
  throughput_rows: 42,
  queue_envelopes: 3,
  queue_rows: 12,
  queue_bytes: 2048,
  backpressure_events: 2,
  late_rows: 1,
};

describe('ResultsPanel', () => {
  it('renders continuous progress and all lifecycle controls', () => {
    const onCheckpoint = vi.fn();
    const onShutdown = vi.fn();
    const onCancel = vi.fn();
    render(
      <ResultsPanel
        validation={null}
        job={runningJob}
        progress={progress}
        busy={false}
        onCheckpoint={onCheckpoint}
        onShutdown={onShutdown}
        onCancel={onCancel}
      />,
    );

    expect(screen.getByText('Job observatory')).toBeInTheDocument();
    expect(screen.getByText('7')).toBeInTheDocument();
    expect(screen.getByText('42 rows')).toBeInTheDocument();
    expect(screen.getByText('3 envelopes')).toBeInTheDocument();
    expect(screen.getByText('12 rows · 2,048 bytes')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Checkpoint' }));
    fireEvent.click(screen.getByRole('button', { name: 'Graceful stop' }));
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onCheckpoint).toHaveBeenCalledOnce();
    expect(onShutdown).toHaveBeenCalledOnce();
    expect(onCancel).toHaveBeenCalledOnce();
  });

  it('renders exact v3 validation issues', () => {
    render(
      <ResultsPanel
        validation={{
          kind: 'invalid',
          valid: false,
          issues: [{
            path: 'graph.nodes[0]',
            code: 'invalid_expression',
            message: 'bad expression',
          }],
          fingerprint: null,
        }}
        job={null}
        progress={null}
        busy={false}
        onCheckpoint={vi.fn()}
        onShutdown={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByText('Graph needs attention')).toBeInTheDocument();
    expect(screen.getByText('bad expression')).toBeInTheDocument();
  });

  it('shows typed worker failures and disables terminal controls', () => {
    render(
      <ResultsPanel
        validation={null}
        job={{
          ...runningJob,
          status: 'failed',
          finished_at: '2026-01-01T00:00:02Z',
          error_code: 'worker_failed',
          error: 'worker exited unexpectedly',
        }}
        progress={progress}
        busy={false}
        onCheckpoint={vi.fn()}
        onShutdown={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByText('worker_failed')).toBeInTheDocument();
    expect(screen.getByText('worker exited unexpectedly')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Checkpoint' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Graceful stop' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
  });
});
