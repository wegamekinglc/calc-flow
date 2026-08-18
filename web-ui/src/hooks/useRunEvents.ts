import { useEffect } from 'react';

import { api, ApiContractError } from '../api/client';
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
  onError: (error: ApiContractError) => void,
): void {
  useEffect(() => {
    if (!runId) return;

    let active = true;
    let closed = false;
    let consecutiveErrors = 0;
    let pollTimer: number | undefined;
    const source = new EventSource(`/api/v3/runs/${runId}/events`);

    const closeSource = () => {
      if (closed) return;
      closed = true;
      source.close();
    };

    const stop = () => {
      active = false;
      closeSource();
      if (pollTimer !== undefined) {
        window.clearTimeout(pollTimer);
        pollTimer = undefined;
      }
    };

    const refresh = async (): Promise<boolean> => {
      if (!active) return true;
      try {
        const current = await api.run(runId);
        if (!active) return true;
        onUpdate(current);
        if (terminalStatuses.has(current.status)) {
          stop();
          return true;
        }
      } catch (error) {
        if (!active) return true;
        if (error instanceof ApiContractError) {
          stop();
          onError(error);
          return true;
        }
        // A later event or polling attempt can recover a transient request failure.
      }
      return false;
    };

    const poll = async () => {
      if (!active) return;
      if (await refresh()) return;
      if (!active) return;
      pollTimer = window.setTimeout(() => {
        pollTimer = undefined;
        void poll();
      }, 500);
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
      stop();
      eventTypes.forEach((type) => source.removeEventListener(type, refreshFromEvent));
      source.removeEventListener('open', handleOpen);
      source.removeEventListener('error', handleError);
    };
  }, [onError, onUpdate, runId]);
}
