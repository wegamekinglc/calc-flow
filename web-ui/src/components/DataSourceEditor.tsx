import type { DataSourceSpec } from '../types';
import {
  DATA_SOURCE_FORMATS,
  type DataSourceDraft,
  type DataSourceFormat,
} from './dataSourceEditor';

export interface DataSourceEditorProps {
  readonly sources: readonly DataSourceSpec[];
  readonly drafts: readonly DataSourceDraft[];
  readonly busy: boolean;
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

export function DataSourceEditor({
  sources,
  drafts,
  busy,
  onAdd,
  onRemove,
  onFieldChange,
  onDataChange,
  onLoadFile,
}: DataSourceEditorProps) {
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
              <label>
                Data {number}
                <textarea
                  rows={7}
                  disabled={busy}
                  value={draft?.dataText ?? ''}
                  aria-invalid={Boolean(draft?.error)}
                  aria-describedby={draft?.error ? errorId : undefined}
                  onChange={(event) => onDataChange(index, event.target.value)}
                />
              </label>
              {draft?.error && (
                <p className="data-source-error" id={errorId}>{draft.error}</p>
              )}
              <label className="file-button">
                Load file
                <input
                  type="file"
                  disabled={busy}
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
    </section>
  );
}
