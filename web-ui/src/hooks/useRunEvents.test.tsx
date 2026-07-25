import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { RunResponse } from '../types';
import { useRunEvents } from './useRunEvents';

const eventTypes = [
  'created',
  'running',
  'completed',
  'failed',
  'cancelled',
  'timed_out',
];

class FakeEventSource {
  static instances: FakeEventSource[] = [];

  readonly close = vi.fn();
  readonly removeEventListener = vi.fn(
    (type: string, listener: () => void) => this.listeners.get(type)?.delete(listener),
  );
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

const run = (): RunResponse => ({
  id: 'run-1',
  project_id: 'project-1',
  status: 'completed',
  created_at: '2026-01-01T00:00:00Z',
  started_at: '2026-01-01T00:00:00Z',
  finished_at: '2026-01-01T00:00:01Z',
  error: null,
  result: {
    outputs: {},
    node_timings: {},
    datafusion_metrics: [],
    metadata: {},
  },
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
        new Response(JSON.stringify(run()), {
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    renderHook(() => useRunEvents('run-1', onUpdate));
    const source = FakeEventSource.instances[0];
    act(() => source.emit('completed'));

    await waitFor(() => expect(onUpdate).toHaveBeenCalledWith(run()));
    expect(source.url).toBe('/api/v2/runs/run-1/events');
    expect(source.close).toHaveBeenCalledOnce();
  });

  it('falls back to polling after two consecutive stream errors', async () => {
    const onUpdate = vi.fn();
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(run()), {
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    renderHook(() => useRunEvents('run-1', onUpdate));
    const source = FakeEventSource.instances[0];
    act(() => {
      source.emit('error');
      source.emit('error');
    });

    await waitFor(() => expect(onUpdate).toHaveBeenCalledWith(run()));
    expect(source.close).toHaveBeenCalledOnce();
  });

  it('removes every stream listener when the owner unmounts', () => {
    vi.stubGlobal('EventSource', FakeEventSource);
    const { unmount } = renderHook(() => useRunEvents('run-1', vi.fn()));
    const source = FakeEventSource.instances[0];

    unmount();

    expect(source.removeEventListener).toHaveBeenCalledTimes(eventTypes.length + 2);
    expect(source.removeEventListener).toHaveBeenCalledWith('open', expect.any(Function));
    expect(source.removeEventListener).toHaveBeenCalledWith('error', expect.any(Function));
  });
});
