import { useEffect, useState } from 'react';

import { validateInputAlias } from './inputAliasEditorModel';

interface InputAliasEditorProps {
  aliases: readonly string[];
  onAdd: () => void;
  onRename: (alias: string, nextAlias: string) => void;
  onRemove: (alias: string) => void;
}

interface InputAliasRowProps {
  alias: string;
  aliases: readonly string[];
  index: number;
  onRename: (alias: string, nextAlias: string) => void;
  onRemove: (alias: string) => void;
}

function InputAliasRow({
  alias,
  aliases,
  index,
  onRename,
  onRemove,
}: InputAliasRowProps) {
  const [draft, setDraft] = useState(alias);
  const [error, setError] = useState<string | null>(null);
  const errorId = `input-alias-${index + 1}-error`;

  useEffect(() => {
    setDraft(alias);
    setError(null);
  }, [alias]);

  const commit = () => {
    const nextError = validateInputAlias(draft, alias, aliases);
    setError(nextError);
    if (nextError !== null) return;
    const nextAlias = draft.trim();
    if (nextAlias !== alias) onRename(alias, nextAlias);
  };

  return (
    <div className="input-alias-row">
      <label>
        Input alias {index + 1}
        <input
          value={draft}
          aria-invalid={error !== null}
          aria-describedby={error ? errorId : undefined}
          onChange={(event) => {
            setDraft(event.target.value);
            setError(null);
          }}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              commit();
            } else if (event.key === 'Escape') {
              event.preventDefault();
              setDraft(alias);
              setError(null);
            }
          }}
        />
      </label>
      <button
        className="icon-button"
        type="button"
        aria-label={`Remove input alias ${index + 1}`}
        onPointerDown={(event) => event.preventDefault()}
        onClick={() => onRemove(alias)}
      >
        ×
      </button>
      {error && <p className="input-alias-error" id={errorId}>{error}</p>}
    </div>
  );
}

export function InputAliasEditor({
  aliases,
  onAdd,
  onRename,
  onRemove,
}: InputAliasEditorProps) {
  return (
    <section className="input-alias-editor" aria-label="Input aliases">
      {aliases.map((alias, index) => (
        <InputAliasRow
          alias={alias}
          aliases={aliases}
          index={index}
          key={alias}
          onRename={onRename}
          onRemove={onRemove}
        />
      ))}
      <button className="text-button" type="button" onClick={onAdd}>
        Add input alias
      </button>
    </section>
  );
}
