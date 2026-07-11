import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { CheckpointControl } from './CheckpointControl';

describe('CheckpointControl', () => {
  it('shows compatible recovery metadata and resets it', () => {
    const reset = vi.fn();
    render(
      <CheckpointControl
        checkpoint={{
          pipeline_name: 'Main',
          exists: true,
          compatible: true,
          pipeline_fingerprint: 'abc',
          sequence: 4,
          source_cursor: { offset: 12 },
          created_at: '2026-01-01T00:00:00Z',
          state_nodes: ['counter'],
        }}
        busy={false}
        onInspect={vi.fn()}
        onReset={reset}
      />,
    );

    expect(screen.getByText('Compatible recovery point')).toBeInTheDocument();
    expect(screen.getByText('Cursor {"offset":12}')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Reset' }));
    expect(reset).toHaveBeenCalledOnce();
  });

  it('labels stale checkpoints and explains preview behavior', () => {
    render(
      <CheckpointControl
        checkpoint={{
          pipeline_name: 'Main',
          exists: true,
          compatible: false,
          pipeline_fingerprint: 'old',
          sequence: 0,
          source_cursor: null,
          created_at: null,
          state_nodes: [],
        }}
        busy={false}
        onInspect={vi.fn()}
        onReset={vi.fn()}
      />,
    );

    expect(screen.getByText('Stale fingerprint')).toBeInTheDocument();
    expect(screen.getByText(/Preview runs do not create checkpoints/)).toBeInTheDocument();
  });
});
