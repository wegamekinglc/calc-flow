interface ProjectActionsProps {
  persisted: boolean;
  busy: boolean;
  onNew: () => void;
  onDelete: () => void;
  onImport: (file: File) => void;
  onExport: (format: 'json' | 'yaml') => void;
}

export function ProjectActions({
  persisted,
  busy,
  onNew,
  onDelete,
  onImport,
  onExport,
}: ProjectActionsProps) {
  return (
    <div className="project-actions">
      <button className="ghost-button" type="button" disabled={busy} onClick={onNew}>
        New
      </button>
      <label className="ghost-button file-button">
        Import
        <input
          aria-label="Import project"
          type="file"
          accept=".json,.yaml,.yml,application/json,application/yaml"
          disabled={busy}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onImport(file);
            event.target.value = '';
          }}
        />
      </label>
      <button
        className="ghost-button"
        type="button"
        disabled={busy || !persisted}
        onClick={() => onExport('json')}
      >
        Export JSON
      </button>
      <button
        className="ghost-button"
        type="button"
        disabled={busy || !persisted}
        onClick={() => onExport('yaml')}
      >
        Export YAML
      </button>
      <button
        className="text-button"
        type="button"
        disabled={busy || !persisted}
        onClick={onDelete}
      >
        Delete
      </button>
    </div>
  );
}
