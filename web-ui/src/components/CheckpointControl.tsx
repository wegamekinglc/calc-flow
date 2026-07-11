import type { CheckpointSummary } from '../types';

interface CheckpointControlProps {
  checkpoint: CheckpointSummary | null;
  busy: boolean;
  onInspect: () => void;
  onReset: () => void;
}

const checkpointStatus = (checkpoint: CheckpointSummary | null): string => {
  if (!checkpoint) return 'Not inspected';
  if (!checkpoint.exists) return 'No stored checkpoint';
  return checkpoint.compatible ? 'Compatible recovery point' : 'Stale fingerprint';
};

export function CheckpointControl({
  checkpoint,
  busy,
  onInspect,
  onReset,
}: CheckpointControlProps) {
  return (
    <section className="checkpoint-control">
      <span className="eyebrow">Runner recovery</span>
      <div className="checkpoint-status">
        <strong>{checkpointStatus(checkpoint)}</strong>
        {checkpoint?.exists && (
          <>
            <span>Sequence {checkpoint.sequence?.toLocaleString() ?? 'unknown'}</span>
            <span>Cursor {JSON.stringify(checkpoint.source_cursor)}</span>
            <span>{checkpoint.state_nodes.length} stateful node{checkpoint.state_nodes.length === 1 ? '' : 's'}</span>
            {checkpoint.created_at && <span>Saved {new Date(checkpoint.created_at).toLocaleString()}</span>}
          </>
        )}
      </div>
      {checkpoint?.exists && checkpoint.compatible === false && (
        <p className="checkpoint-warning">The saved fingerprint cannot be restored into the current graph.</p>
      )}
      <p>Micro-batch and streaming runners use this store. Preview runs do not create checkpoints.</p>
      <div className="checkpoint-actions">
        <button className="ghost-button" type="button" disabled={busy} onClick={onInspect}>Inspect</button>
        <button className="text-button" type="button" disabled={busy || !checkpoint?.exists} onClick={onReset}>Reset</button>
      </div>
    </section>
  );
}
