import { act, renderHook, waitFor } from '@testing-library/react';
import type { Dispatch } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiContractError } from '../api/client';
import type { JobResponse } from '../types';
import { useJobEvents } from './useJobEvents';

const eventTypes = ['state', 'progress', 'checkpoint', 'terminal'];

class FakeEventSource {
  static instances: FakeEventSource[] = [];

  readonly close = vi.fn();
  readonly removeEventListener = vi.fn((type: string, listener: EventListener) => {
    this.listeners.get(type)?.delete(listener);
  });
  readonly url: string;
  private readonly listeners = new Map<string, Set<EventListener>>();

  constructor(url: string) {
    this.url = url;
    FakeEventSource.instances.push(this);
  }

  addEventListener(type: string, listener: EventListener) {
    const listeners = this.listeners.get(type) ?? new Set();
    listeners.add(listener);
    this.listeners.set(type, listeners);
  }

  emit(type: string, data = '') {
    const event = data
      ? new MessageEvent(type, { data })
      : new Event(type);
    this.listeners.get(type)?.forEach((listener) => {
      listener(event);
    });
  }
}

const job = (status: JobResponse['status']): JobResponse => ({
  id: 'job-1',
  project_id: 'project-1',
  status,
  created_at: '2026-01-01T00:00:00Z',
  started_at: status === 'pending' ? null : '2026-01-01T00:00:01Z',
  finished_at: status === 'completed' || status === 'failed' || status === 'cancelled'
    ? '2026-01-01T00:00:02Z'
    : null,
  error_code: status === 'failed' ? 'worker_failed' : null,
  error: status === 'failed' ? 'worker exited' : null,
} as JobResponse);

afterEach(() => {
  FakeEventSource.instances = [];
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe('useJobEvents', () => {
  it('forwards progress payloads and refreshes authoritative job state', async () => {
    const onUpdate = vi.fn();
    const onEvent = vi.fn();
    const onError = vi.fn();
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify(job('running'))),
    ));

    renderHook(() => {
      useJobEvents('job-1', onUpdate, onEvent, onError);
    });
    const source = FakeEventSource.instances[0];
    const progress = {
      sequence: 2,
      timestamp: '2026-01-01T00:00:02Z',
      type: 'progress',
      message: 'running',
      epoch: 7,
      throughput_rows: 42,
    };
    act(() => {
      source.emit('progress', JSON.stringify(progress));
    });

    await waitFor(() => expect(onUpdate).toHaveBeenCalledWith(job('running')));
    expect(onEvent).toHaveBeenCalledWith(progress);
    expect(onError).not.toHaveBeenCalled();
    expect(source.url).toBe('/api/v3/jobs/job-1/events');
  });

  it('closes the stream after authoritative terminal state', async () => {
    const onUpdate = vi.fn();
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify(job('completed'))),
    ));

    renderHook(() => {
      useJobEvents('job-1', onUpdate, vi.fn(), vi.fn());
    });
    const source = FakeEventSource.instances[0];
    act(() => {
      source.emit('terminal');
    });

    await waitFor(() => expect(onUpdate).toHaveBeenCalledWith(job('completed')));
    expect(source.close).toHaveBeenCalledOnce();
  });

  it('surfaces contract failures and stops tracking stale state', async () => {
    const onUpdate = vi.fn();
    const onError = vi.fn();
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ ...job('running'), status: 'timed_out' })),
    ));

    renderHook(() => {
      useJobEvents('job-1', onUpdate, vi.fn(), onError);
    });
    const source = FakeEventSource.instances[0];
    act(() => {
      source.emit('state');
    });

    await waitFor(() => expect(onError).toHaveBeenCalledOnce());
    expect(onError.mock.calls[0][0]).toBeInstanceOf(ApiContractError);
    expect(onUpdate).not.toHaveBeenCalled();
    expect(source.close).toHaveBeenCalledOnce();
  });

  it('does not let an older refresh overwrite a newer terminal response', async () => {
    const requests: Dispatch<Response>[] = [];
    const fetchMock = vi.fn().mockImplementation(
      () => new Promise<Response>((resolve) => {
        requests.push(resolve);
      }),
    );
    const onUpdate = vi.fn();
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal('fetch', fetchMock);

    renderHook(() => {
      useJobEvents('job-1', onUpdate, vi.fn(), vi.fn());
    });
    const source = FakeEventSource.instances[0];
    act(() => {
      source.emit('progress');
      source.emit('terminal');
    });
    await waitFor(() => expect(requests).toHaveLength(2));

    await act(async () => {
      requests[1](new Response(JSON.stringify(job('completed'))));
      await Promise.resolve();
    });
    await act(async () => {
      requests[0](new Response(JSON.stringify(job('running'))));
      await Promise.resolve();
    });

    expect(onUpdate.mock.calls.map(([current]) => current.status)).toEqual(['completed']);
  });

  it('falls back to polling after two consecutive stream errors', async () => {
    const onUpdate = vi.fn();
    vi.stubGlobal('EventSource', FakeEventSource);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(
      new Response(JSON.stringify(job('completed'))),
    ));

    renderHook(() => {
      useJobEvents('job-1', onUpdate, vi.fn(), vi.fn());
    });
    const source = FakeEventSource.instances[0];
    act(() => {
      source.emit('error');
      source.emit('error');
    });

    await waitFor(() => expect(onUpdate).toHaveBeenCalledWith(job('completed')));
    expect(source.close).toHaveBeenCalledOnce();
  });

  it('removes every stream listener when the owner unmounts', () => {
    vi.stubGlobal('EventSource', FakeEventSource);
    const { unmount } = renderHook(
      () => {
        useJobEvents('job-1', vi.fn(), vi.fn(), vi.fn());
      },
    );
    const source = FakeEventSource.instances[0];

    unmount();

    expect(source.removeEventListener).toHaveBeenCalledTimes(eventTypes.length + 2);
    expect(source.removeEventListener).toHaveBeenCalledWith('open', expect.any(Function));
    expect(source.removeEventListener).toHaveBeenCalledWith('error', expect.any(Function));
  });
});
