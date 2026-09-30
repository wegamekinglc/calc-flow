import type { ArrowFieldConfig } from '../types';
import { useEditableRows } from './useEditableRows';

interface SchemaEditorProps {
  fields: ArrowFieldConfig[];
  arrowTypes: readonly string[];
  onChange: (fields: ArrowFieldConfig[]) => void;
}

export function SchemaEditor({ fields, arrowTypes, onChange }: SchemaEditorProps) {
  const { rows, update, append, remove } = useEditableRows(fields, onChange);

  return (
    <div className="schema-editor">
      {rows.map(({ key, value: field }, index) => (
        <div className="schema-row" key={key}>
          <input
            aria-label="Field name"
            value={field.name}
            onChange={(event) => update(index, { name: event.target.value })}
          />
          <select
            aria-label="Field type"
            value={field.data_type}
            onChange={(event) => update(index, { data_type: event.target.value })}
          >
            {arrowTypes.map((type) => (
              <option key={type}>{type}</option>
            ))}
          </select>
          <label className="nullable-toggle">
            <input
              type="checkbox"
              checked={field.nullable}
              onChange={(event) => update(index, { nullable: event.target.checked })}
            />
            null
          </label>
          <button
            className="icon-button"
            type="button"
            aria-label={`Remove ${field.name}`}
            onClick={() => remove(index)}
          >
            ×
          </button>
        </div>
      ))}
      <button
        className="text-button"
        type="button"
        onClick={() =>
          append({ name: `field_${fields.length + 1}`, data_type: 'float64', nullable: true })
        }
      >
        + field
      </button>
    </div>
  );
}
