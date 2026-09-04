import { useEffect } from 'react';
import type { Dispatch } from 'react';

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
  onUpdate: Dispatch<JobResponse>,
  onEvent: Dispatch<JobEvent>,
  onError: Dispatch<ApiContractError>,
): void {
  useEffect(() => {
    if (!jobId) return;

    let active = true;
    // Read the cancellation flag through a helper: control-flow narrowing must
    // not carry across the awaited job request, where stop() may clear it.
    const isLive = (): boolean => active;
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
      if (!isLive()) return true;
      const revision = ++refreshRevision;
      try {
        const current = await api.job(jobId);
        if (!isLive()) return true;
        if (revision !== refreshRevision) return false;
        onUpdate(current);
        if (terminalStatuses.has(current.status)) {
          stop();
          return true;
        }
      } catch (error) {
        if (!isLive()) return true;
        if (revision !== refreshRevision) return false;
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
      if (!isLive()) return;
      if (await refresh()) return;
      if (!isLive()) return;
      pollTimer = window.setTimeout(() => {
        pollTimer = undefined;
        void poll();
      }, 500);
    };

    const refreshFromEvent = (raw: Event) => {
      if (raw instanceof MessageEvent) {
        try {
          const event: unknown = JSON.parse(raw.data);
          if (
            event !== null
            && typeof event === 'object'
            && 'type' in event
            && typeof event.type === 'string'
          ) {
            onEvent(event as JobEvent);
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
