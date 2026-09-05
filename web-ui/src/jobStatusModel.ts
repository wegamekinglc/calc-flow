import type { JobResponse } from './types';

export type JobStatus = JobResponse['status'];

/** Job statuses the Studio treats as terminal for event streaming and polling. */
export const TERMINAL_JOB_STATUSES: readonly JobStatus[] = [
  'completed',
  'failed',
  'cancelled',
] as const;

const TERMINAL_JOB_STATUS_SET: ReadonlySet<JobStatus> = new Set(TERMINAL_JOB_STATUSES);

export const isTerminalJobStatus = (status: JobStatus): boolean =>
  TERMINAL_JOB_STATUS_SET.has(status);

/** True while the job can still accept checkpoint, shutdown, or cancel actions. */
export const isJobActive = (status: JobStatus): boolean =>
  status === 'pending' || status === 'running';
