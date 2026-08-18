import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiContractError } from '../api/client';
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

const runningRun = (): RunResponse => ({
  id: 'run-1',
  project_id: 'project-1',
  status: 'running',
  created_at: '2026-01-01T00:00:00Z',
  started_at: '2026-01-01T00:00:00Z',
  finished_at: null,
  error: null,
  result: null,
});

afterEach(() => {
  FakeEventSource.instances = [];
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('useRunEvents', () => {
  it('refreshes authoritative state and closes on a terminal event', async () => {
    const onUpdate = vi.fn();
    const onError = vi.fn();
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(run()), {
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    renderHook(() => useRunEvents('run-1', onUpdate, onError));
    const source = FakeEventSource.instances[0];
    act(() => source.emit('completed'));

    await waitFor(() => expect(onUpdate).toHaveBeenCalledWith(run()));
    expect(onError).not.toHaveBeenCalled();
    expect(source.url).toBe('/api/v3/runs/run-1/events');
    expect(source.close).toHaveBeenCalledOnce();
  });

  it('surfaces contract failures and stops tracking stale run state', async () => {
    const onUpdate = vi.fn();
    const onError = vi.fn();
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ ...run(), status: 'unknown' }), {
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    renderHook(() => useRunEvents('run-1', onUpdate, onError));
    const source = FakeEventSource.instances[0];
    act(() => source.emit('running'));

    await waitFor(() => expect(onError).toHaveBeenCalledOnce());
    expect(onError.mock.calls[0][0]).toBeInstanceOf(ApiContractError);
    expect(onUpdate).not.toHaveBeenCalled();
    expect(source.close).toHaveBeenCalledOnce();
  });

  it('makes contract failures terminal for overlapping refreshes', async () => {
    vi.useFakeTimers();
    const onUpdate = vi.fn();
    const onError = vi.fn();
    const requests: Array<(response: Response) => void> = [];
    const holdRequest = () => new Promise<Response>((resolve) => {
      requests.push(resolve);
    });
    const fetchMock = vi.fn()
      .mockImplementationOnce(holdRequest)
      .mockImplementationOnce(holdRequest)
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValue(
        new Response(JSON.stringify(run()), {
          headers: { 'Content-Type': 'application/json' },
        }),
      );
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal('fetch', fetchMock);

    renderHook(() => useRunEvents('run-1', onUpdate, onError));
    const source = FakeEventSource.instances[0];
    await act(async () => {
      source.emit('running');
      source.emit('completed');
      source.emit('error');
      source.emit('error');
      await Promise.resolve();
    });
    expect(fetchMock).toHaveBeenCalledTimes(3);

    await act(async () => {
      requests[0](
        new Response(JSON.stringify({ ...run(), status: 'unknown' }), {
          headers: { 'Content-Type': 'application/json' },
        }),
      );
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(onError).toHaveBeenCalledOnce();

    await act(async () => {
      requests[1](
        new Response(JSON.stringify(run()), {
          headers: { 'Content-Type': 'application/json' },
        }),
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(onUpdate).not.toHaveBeenCalled();
    expect(source.close).toHaveBeenCalledOnce();
    await vi.advanceTimersByTimeAsync(1_000);
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it('does not let an older refresh overwrite a newer terminal response', async () => {
    const onUpdate = vi.fn();
    const onError = vi.fn();
    const requests: Array<(response: Response) => void> = [];
    const holdRequest = () => new Promise<Response>((resolve) => {
      requests.push(resolve);
    });
    const fetchMock = vi.fn()
      .mockImplementationOnce(holdRequest)
      .mockImplementationOnce(holdRequest);
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal('fetch', fetchMock);

    renderHook(() => useRunEvents('run-1', onUpdate, onError));
    const source = FakeEventSource.instances[0];
    await act(async () => {
      source.emit('running');
      source.emit('completed');
      await Promise.resolve();
    });
    expect(fetchMock).toHaveBeenCalledTimes(2);

    await act(async () => {
      requests[1](
        new Response(JSON.stringify(run()), {
          headers: { 'Content-Type': 'application/json' },
        }),
      );
      await Promise.resolve();
      await Promise.resolve();
    });
    expect(onUpdate).toHaveBeenCalledWith(run());

    await act(async () => {
      requests[0](
        new Response(JSON.stringify(runningRun()), {
          headers: { 'Content-Type': 'application/json' },
        }),
      );
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(onUpdate.mock.calls.map(([current]) => current.status)).toEqual([
      'completed',
    ]);
    expect(onError).not.toHaveBeenCalled();
    expect(source.close).toHaveBeenCalledOnce();
  });

  it('falls back to polling after two consecutive stream errors', async () => {
    const onUpdate = vi.fn();
    const onError = vi.fn();
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify(run()), {
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    );

    renderHook(() => useRunEvents('run-1', onUpdate, onError));
    const source = FakeEventSource.instances[0];
    act(() => {
      source.emit('error');
      source.emit('error');
    });

    await waitFor(() => expect(onUpdate).toHaveBeenCalledWith(run()));
    expect(onError).not.toHaveBeenCalled();
    expect(source.close).toHaveBeenCalledOnce();
  });

  it('silently retries a transient network failure while polling', async () => {
    vi.useFakeTimers();
    const onUpdate = vi.fn();
    const onError = vi.fn();
    const fetchMock = vi.fn()
      .mockRejectedValueOnce(new TypeError('Failed to fetch'))
      .mockResolvedValue(
        new Response(JSON.stringify(run()), {
          headers: { 'Content-Type': 'application/json' },
        }),
      );
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal('fetch', fetchMock);

    renderHook(() => useRunEvents('run-1', onUpdate, onError));
    const source = FakeEventSource.instances[0];
    await act(async () => {
      source.emit('error');
      source.emit('error');
      await Promise.resolve();
    });

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(onError).not.toHaveBeenCalled();

    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });

    expect(onUpdate).toHaveBeenCalledWith(run());
    expect(onError).not.toHaveBeenCalled();
  });

  it('removes every stream listener when the owner unmounts', () => {
    vi.stubGlobal('EventSource', FakeEventSource);
    const { unmount } = renderHook(() => useRunEvents('run-1', vi.fn(), vi.fn()));
    const source = FakeEventSource.instances[0];

    unmount();

    expect(source.removeEventListener).toHaveBeenCalledTimes(eventTypes.length + 2);
    expect(source.removeEventListener).toHaveBeenCalledWith('open', expect.any(Function));
    expect(source.removeEventListener).toHaveBeenCalledWith('error', expect.any(Function));
  });
});
