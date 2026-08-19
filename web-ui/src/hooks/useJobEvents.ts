import { useEffect } from 'react';

import { api, ApiContractError } from '../api/client';
import type { JobEvent, JobResponse } from '../types';

const terminalStatuses = new Set<JobResponse['status']>([
  'completed',
  'failed',
  'cancelled',
]);

const eventTypes = ['state', 'progress', 'checkpoint', 'terminal'];

export function useJobEvents(
  jobId: string | null,
  onUpdate: (job: JobResponse) => void,
  onEvent: (event: JobEvent) => void,
  onError: (error: ApiContractError) => void,
): void {
  useEffect(() => {
    if (!jobId) return;

    let active = true;
    let closed = false;
    let consecutiveErrors = 0;
    let refreshRevision = 0;
    let pollTimer: number | undefined;
    const source = new EventSource(`/api/v3/jobs/${jobId}/events`);

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
      const revision = ++refreshRevision;
      try {
        const current = await api.job(jobId);
        if (!active || revision !== refreshRevision) return !active;
        onUpdate(current);
        if (terminalStatuses.has(current.status)) {
          stop();
          return true;
        }
      } catch (error) {
        if (!active || revision !== refreshRevision) return !active;
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

    const refreshFromEvent = (raw: Event) => {
      if (raw instanceof MessageEvent) {
        try {
          const event = JSON.parse(raw.data) as JobEvent;
          if (event && typeof event === 'object' && typeof event.type === 'string') {
            onEvent(event);
          }
        } catch {
          // The status refresh still recovers from a malformed optional summary.
        }
      }
      void refresh();
    };
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
  }, [jobId, onError, onEvent, onUpdate]);
}
