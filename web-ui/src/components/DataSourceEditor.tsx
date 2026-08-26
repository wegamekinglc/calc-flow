import { useEffect, useRef, useState, type MouseEvent } from 'react';

import type { DataSourceSpec } from '../types';
import { DataSourceDialog } from './DataSourceDialog';
import {
  DATA_SOURCE_FORMATS,
  type DataSourceDraft,
  type DataSourceFormat,
} from './dataSourceEditorModel';

export interface DataSourceEditorProps {
  readonly sources: readonly DataSourceSpec[];
  readonly drafts: readonly DataSourceDraft[];
  readonly busy: boolean;
  readonly pendingSourceKeys: ReadonlySet<string>;
  readonly onAdd: () => void;
  readonly onRemove: (index: number) => void;
  readonly onFieldChange: (
    index: number,
    field: 'id' | 'input' | 'format',
    value: string,
  ) => void;
  readonly onDataChange: (index: number, value: string) => void;
  readonly onLoadFile: (index: number, file: File) => void;
}

const SOURCE_FORMAT_LABELS: Record<DataSourceFormat, string> = {
  inline_json: 'Inline JSON',
  json: 'JSON / JSONL',
  csv: 'CSV',
  arrow_ipc: 'Arrow IPC',
};

const SOURCE_FILE_ACCEPT: Record<DataSourceFormat, string> = {
  inline_json: '.json,application/json',
  json: '.json,.jsonl,.ndjson,application/json,application/x-ndjson',
  csv: '.csv,text/csv',
  arrow_ipc: '.arrow,.ipc,application/vnd.apache.arrow.file,application/vnd.apache.arrow.stream',
};

interface ActiveEditor {
  readonly key: string;
  readonly format: DataSourceFormat;
  readonly initialText: string;
  readonly sourceLabel: string;
}

const DATA_PREVIEW_LIMIT = 240;

const dataPreview = (dataText: string): string => {
  const normalized = dataText.trim();
  if (!normalized) return 'No data';
  return normalized.length > DATA_PREVIEW_LIMIT
    ? `${normalized.slice(0, DATA_PREVIEW_LIMIT)}…`
    : normalized;
};

export function DataSourceEditor({
  sources,
  drafts,
  busy,
  pendingSourceKeys,
  onAdd,
  onRemove,
  onFieldChange,
  onDataChange,
  onLoadFile,
}: DataSourceEditorProps) {
  const [activeEditor, setActiveEditor] = useState<ActiveEditor | null>(null);
  const restoreFocusRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (activeEditor && !drafts.some((draft) => draft.key === activeEditor.key)) {
      setActiveEditor(null);
    }
  }, [activeEditor, drafts]);

  useEffect(() => {
    if (activeEditor) return;
    const opener = restoreFocusRef.current;
    restoreFocusRef.current = null;
    if (opener?.isConnected) opener.focus();
  }, [activeEditor]);

  const dismissEditor = () => {
    setActiveEditor(null);
  };

  const openEditor = (
    index: number,
    event: MouseEvent<HTMLButtonElement>,
  ) => {
    const source = sources[index];
    const draft = drafts[index];
    if (!source || !draft) return;
    restoreFocusRef.current = event.currentTarget;
    setActiveEditor({
      key: draft.key,
      format: source.format as DataSourceFormat,
      initialText: draft.dataText,
      sourceLabel: source.id || String(index + 1),
    });
  };

  const commitEditor = (dataText: string) => {
    if (!activeEditor) return;
    const currentIndex = drafts.findIndex(
      (draft) => draft.key === activeEditor.key,
    );
    if (currentIndex >= 0) onDataChange(currentIndex, dataText);
    setActiveEditor(null);
  };

  return (
    <section className="data-source-editor" aria-labelledby="data-source-heading">
      <div className="data-source-heading">
        <span className="eyebrow" id="data-source-heading">Data sources</span>
        <button className="text-button" type="button" disabled={busy} onClick={onAdd}>
          Add data source
        </button>
      </div>
      {!sources.length && (
        <p className="data-source-empty">
          Add one data source for every external graph input.
        </p>
      )}
      <div className="data-source-list">
        {sources.map((source, index) => {
          const number = index + 1;
          const draft = drafts[index];
          const errorId = `data-source-error-${draft?.key ?? number}`;
          const format = source.format as DataSourceFormat;
          const editorPending = draft
            ? pendingSourceKeys.has(draft.key)
            : false;
          const dialogOwnsSource = activeEditor?.key === draft?.key;
          const sourceLabel = source.id || String(number);
          return (
            <article className="data-source-card" key={draft?.key ?? `${source.id}-${number}`}>
              <header>
                <strong>{source.id || `Source ${number}`}</strong>
                <button
                  className="icon-button"
                  type="button"
                  disabled={busy}
                  aria-label={`Remove source ${number}`}
                  onClick={() => onRemove(index)}
                >
                  Remove
                </button>
              </header>
              <label>
                Source ID {number}
                <input
                  disabled={busy}
                  value={source.id}
                  onChange={(event) => onFieldChange(index, 'id', event.target.value)}
                />
              </label>
              <label>
                Graph input {number}
                <input
                  disabled={busy}
                  value={source.input}
                  onChange={(event) => onFieldChange(index, 'input', event.target.value)}
                />
              </label>
              <label>
                Format {number}
                <select
                  disabled={busy}
                  value={source.format}
                  onChange={(event) => onFieldChange(index, 'format', event.target.value)}
                >
                  {DATA_SOURCE_FORMATS.map((candidate) => (
                    <option value={candidate} key={candidate}>
                      {SOURCE_FORMAT_LABELS[candidate]}
                    </option>
                  ))}
                </select>
              </label>
              <div className="data-source-data">
                <div
                  className="data-source-preview"
                  aria-label={`Data ${number} preview`}
                >
                  <span>Data {number}</span>
                  <pre>{dataPreview(draft?.dataText ?? '')}</pre>
                </div>
                <button
                  className="ghost-button data-source-edit-button"
                  type="button"
                  disabled={busy || editorPending || !draft}
                  aria-label={`Edit data source ${sourceLabel}`}
                  aria-invalid={Boolean(draft?.error)}
                  aria-describedby={draft?.error ? errorId : undefined}
                  onClick={(event) => openEditor(index, event)}
                >
                  Edit data
                </button>
              </div>
              {draft?.error && (
                <p className="data-source-error" id={errorId}>{draft.error}</p>
              )}
              <label className="file-button">
                Load file
                <input
                  type="file"
                  disabled={busy || dialogOwnsSource}
                  aria-label={`Load file ${number}`}
                  accept={SOURCE_FILE_ACCEPT[format]}
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) onLoadFile(index, file);
                    event.target.value = '';
                  }}
                />
              </label>
            </article>
          );
        })}
      </div>
      {activeEditor && (
        <DataSourceDialog
          key={activeEditor.key}
          format={activeEditor.format}
          initialText={activeEditor.initialText}
          sourceLabel={activeEditor.sourceLabel}
          onConfirm={commitEditor}
          onDismiss={dismissEditor}
        />
      )}
    </section>
  );
}
