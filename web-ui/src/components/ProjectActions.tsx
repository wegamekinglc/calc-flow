import { useState } from 'react';

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
  const [importKey, setImportKey] = useState(0);

  return (
    <div className="project-actions">
      <button
        className="ghost-button topbar-control"
        type="button"
        disabled={busy}
        onClick={onNew}
      >
        New
      </button>
      <label className="ghost-button file-button topbar-control">
        Import
        <input
          key={importKey}
          aria-label="Import project"
          type="file"
          accept=".json,.yaml,.yml,application/json,application/yaml"
          disabled={busy}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (!file) return;
            onImport(file);
            setImportKey((current) => current + 1);
          }}
        />
      </label>
      <button
        className="ghost-button topbar-control"
        type="button"
        disabled={busy || !persisted}
        onClick={() => onExport('json')}
      >
        Export JSON
      </button>
      <button
        className="ghost-button topbar-control"
        type="button"
        disabled={busy || !persisted}
        onClick={() => onExport('yaml')}
      >
        Export YAML
      </button>
      <button
        className="text-button topbar-control"
        type="button"
        disabled={busy || !persisted}
        onClick={onDelete}
      >
        Delete
      </button>
    </div>
  );
}
