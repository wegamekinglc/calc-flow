import type { JobEvent, JobResponse, ValidationReport } from '../types';

interface ResultsPanelProps {
  validation: ValidationReport | null;
  job: JobResponse | null;
  progress: JobEvent | null;
  busy: boolean;
  onCheckpoint: () => void;
  onShutdown: () => void;
  onCancel: () => void;
}

const number = (value: number | undefined): string =>
  (value ?? 0).toLocaleString();

export function ResultsPanel({
  validation,
  job,
  progress,
  busy,
  onCheckpoint,
  onShutdown,
  onCancel,
}: ResultsPanelProps) {
  const active = job?.status === 'pending' || job?.status === 'running';

  return (
    <section className="results panel job-observatory">
      <div className="panel-heading results-heading">
        <div>
          <span className="eyebrow">Continuous runtime</span>
          <h2>Job observatory</h2>
        </div>
        {job && <span className={`status-pill ${job.status}`}>{job.status}</span>}
        <div className="job-actions">
          <button
            className="ghost-button"
            type="button"
            disabled={busy || !active}
            onClick={onCheckpoint}
          >
            Checkpoint
          </button>
          <button
            className="ghost-button"
            type="button"
            disabled={busy || !active}
            onClick={onShutdown}
          >
            Graceful stop
          </button>
          <button
            className="text-button"
            type="button"
            disabled={busy || !active}
            onClick={onCancel}
          >
            Cancel
          </button>
        </div>
      </div>

      {validation && (
        <div className={`validation-banner ${validation.valid ? 'valid' : 'invalid'}`}>
          <strong>{validation.valid ? 'Graph is valid' : 'Graph needs attention'}</strong>
          <span>
            {validation.valid
              ? validation.fingerprint
                ? `Fingerprint ${validation.fingerprint}`
                : 'No validation issues'
              : validation.issues.map((issue) => issue.message).join(' · ')}
          </span>
        </div>
      )}

      {job?.status === 'failed' && (
        <div className="validation-banner invalid">
          <strong>{job.error_code}</strong>
          <span>{job.error}</span>
        </div>
      )}

      {!job && (
        <div className="empty-state">
          <div className="empty-orbit" />
          <p>Save a stream project, then start a persistent job to observe it.</p>
        </div>
      )}

      {job && (
        <div className="job-metrics" aria-label="Continuous job metrics">
          <article className="metric-card">
            <span className="eyebrow">Job state</span>
            <strong>{progress?.state ?? job.status}</strong>
            <small>{job.id}</small>
          </article>
          <article className="metric-card">
            <span className="eyebrow">Epoch</span>
            <strong>{progress?.epoch ?? '—'}</strong>
            <small>last observed checkpoint</small>
          </article>
          <article className="metric-card">
            <span className="eyebrow">Watermark</span>
            <strong>{progress?.watermark ? new Date(progress.watermark).toLocaleString() : '—'}</strong>
            <small>aggregate event time</small>
          </article>
          <article className="metric-card">
            <span className="eyebrow">Throughput</span>
            <strong>{number(progress?.throughput_rows)} rows</strong>
            <small>cumulative source rows</small>
          </article>
          <article className="metric-card">
            <span className="eyebrow">Queue</span>
            <strong>{number(progress?.queue_envelopes)} envelopes</strong>
            <small>{number(progress?.queue_rows)} rows · {number(progress?.queue_bytes)} bytes</small>
          </article>
          <article className="metric-card">
            <span className="eyebrow">Backpressure</span>
            <strong>{number(progress?.backpressure_events)}</strong>
            <small>blocked sends</small>
          </article>
          <article className="metric-card">
            <span className="eyebrow">Late rows</span>
            <strong>{number(progress?.late_rows)}</strong>
            <small>event-time rejections</small>
          </article>
        </div>
      )}
    </section>
  );
}
