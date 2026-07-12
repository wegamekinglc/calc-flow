import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { RunResponse } from '../types';
import { useRunEvents } from './useRunEvents';

class FakeEventSource {
  static instances: FakeEventSource[] = [];

  readonly close = vi.fn();
  onerror: (() => void) | null = null;
  onopen: (() => void) | null = null;
  private readonly listeners = new Map<string, Set<() => void>>();

  constructor(readonly url: string) {
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: () => void) {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
  }

  emit(type: string) {
    this.listeners.get(type)?.forEach((listener) => listener());
  }
}

const run = (status: RunResponse['status']): RunResponse => ({
  id: 'run-1',
  project_id: 'project-1',
  status,
  created_at: '2026-01-01T00:00:00Z',
  started_at: null,
  finished_at: status === 'completed' ? '2026-01-01T00:00:01Z' : null,
  error: null,
  result: status === 'completed' ? { outputs: {} } : null,
});

afterEach(() => {
  FakeEventSource.instances = [];
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('useRunEvents', () => {
  it('refreshes authoritative state and closes on a terminal event', async () => {
    const onUpdate = vi.fn();
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(run('completed')), {
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    renderHook(() => useRunEvents('run-1', onUpdate));
    const source = FakeEventSource.instances[0];
    act(() => source.emit('completed'));

    await waitFor(() => expect(onUpdate).toHaveBeenCalledWith(run('completed')));
    expect(source.url).toBe('/api/v1/runs/run-1/events');
    expect(source.close).toHaveBeenCalledOnce();
  });

  it('falls back to polling after two consecutive stream errors', async () => {
    const onUpdate = vi.fn();
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(run('completed')), {
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    renderHook(() => useRunEvents('run-1', onUpdate));
    const source = FakeEventSource.instances[0];
    act(() => {
      source.onerror?.();
      source.onerror?.();
    });

    await waitFor(() => expect(onUpdate).toHaveBeenCalledWith(run('completed')));
    expect(source.close).toHaveBeenCalledOnce();
  });
});
