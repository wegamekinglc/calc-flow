import { useEffect } from 'react';

import { api } from '../api/client';
import type { RunResponse } from '../types';

const terminalStatuses = new Set<RunResponse['status']>([
  'completed',
  'failed',
  'cancelled',
  'timed_out',
]);

const eventTypes = [
  'created',
  'running',
  'completed',
  'failed',
  'cancelled',
  'timed_out',
];

export function useRunEvents(
  runId: string | null,
  onUpdate: (run: RunResponse) => void,
): void {
  useEffect(() => {
    if (!runId) return;

    let active = true;
    let closed = false;
    let consecutiveErrors = 0;
    let pollTimer: number | undefined;
    const source = new EventSource(`/api/v2/runs/${runId}/events`);

    const closeSource = () => {
      if (closed) return;
      closed = true;
      source.close();
    };

    const refresh = async (): Promise<boolean> => {
      try {
        const current = await api.run(runId);
        if (!active) return true;
        onUpdate(current);
        if (terminalStatuses.has(current.status)) {
          closeSource();
          return true;
        }
      } catch {
        // A later event or polling attempt can recover a transient request failure.
      }
      return false;
    };

    const poll = async () => {
      if (!active) return;
      if (await refresh()) return;
      pollTimer = window.setTimeout(() => void poll(), 500);
    };

    const refreshFromEvent = () => void refresh();
    const handleOpen = () => {
      consecutiveErrors = 0;
    };
    const handleError = () => {
      consecutiveErrors += 1;
      if (consecutiveErrors < 2) return;
      closeSource();
      void poll();
    };
    eventTypes.forEach((type) => source.addEventListener(type, refreshFromEvent));
    source.addEventListener('open', handleOpen);
    source.addEventListener('error', handleError);

    return () => {
      active = false;
      closeSource();
      if (pollTimer !== undefined) window.clearTimeout(pollTimer);
      eventTypes.forEach((type) => source.removeEventListener(type, refreshFromEvent));
      source.removeEventListener('open', handleOpen);
      source.removeEventListener('error', handleError);
    };
  }, [onUpdate, runId]);
}
