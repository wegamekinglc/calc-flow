import { SchemaEditor } from './SchemaEditor';
import type { ArrowFieldConfig, JSONValue, NodeConfig, PortConfig } from '../types';

interface NodeInspectorProps {
  node: NodeConfig;
  arrowTypes: string[];
  udfs: Record<string, JSONValue>[];
  onChange: (node: NodeConfig) => void;
  onDelete: () => void;
}

const value = (entry: Record<string, JSONValue>, key: string): string => String(entry[key] ?? '');

export function NodeInspector({ node, arrowTypes, udfs, onChange, onDelete }: NodeInspectorProps) {
  const patch = (change: Partial<NodeConfig>) => onChange({ ...node, ...change });
  const isTable = node.kind !== 'array_expression';
  const inputNames = node.kind === 'sql' ? node.inputs : ['input'];
  const matchingUdfs = udfs.filter((entry) =>
    node.kind === 'array_expression' ? entry.kind === 'array' : entry.kind === 'datafusion_scalar',
  );

  const schema = (direction: 'input' | 'output', name: string): ArrowFieldConfig[] => {
    const ports = direction === 'input' ? node.input_ports : node.output_ports;
    return ports.find((port) => port.name === name)?.schema ?? [];
  };

  const updateSchema = (
    direction: 'input' | 'output',
    name: string,
    fields: ArrowFieldConfig[],
  ) => {
    const names = direction === 'input' ? inputNames : ['output'];
    const current = direction === 'input' ? node.input_ports : node.output_ports;
    const ports: PortConfig[] = names.map((portName) => {
      const existing = current.find((port) => port.name === portName);
      return {
        name: portName,
        kind: isTable ? 'table' : 'array',
        required: true,
        schema: portName === name ? (fields.length ? fields : null) : (existing?.schema ?? null),
      };
    });
    patch(direction === 'input' ? { input_ports: ports } : { output_ports: ports });
  };

  const toggleUdf = (name: string, version: string, checked: boolean) => {
    const others = node.udfs.filter((reference) => reference.name !== name);
    patch({ udfs: checked ? [...others, { name, version }] : others });
  };

  return (
    <aside className="inspector panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Node inspector</span>
          <h2>{node.id}</h2>
        </div>
        <span className={`kind-chip ${node.kind}`}>{node.kind.replace('_', ' ')}</span>
      </div>

      <label>
        Node ID
        <input value={node.id} disabled />
      </label>

      {node.kind === 'expression' && (
        <>
          <label>
            DataFusion expression
            <textarea
              rows={5}
              value={node.expression ?? ''}
              onChange={(event) => patch({ expression: event.target.value, select: [] })}
            />
          </label>
          <label>
            Filter expression
            <input
              placeholder="amount > 0"
              value={node.filter_expression ?? ''}
              onChange={(event) => patch({ filter_expression: event.target.value || null })}
            />
          </label>
        </>
      )}

      {node.kind === 'sql' && (
        <>
          <label>
            Input aliases
            <input
              value={node.inputs.join(', ')}
              onChange={(event) =>
                patch({
                  inputs: event.target.value
                    .split(',')
                    .map((item) => item.trim())
                    .filter(Boolean),
                  input_ports: [],
                })
              }
            />
          </label>
          <label>
            DataFusion SQL
            <textarea
              rows={9}
              value={node.query ?? ''}
              onChange={(event) => patch({ query: event.target.value })}
            />
          </label>
        </>
      )}

      {node.kind === 'array_expression' && (
        <>
          <label>
            Array backend
            <select
              value={node.backend ?? 'numpy'}
              onChange={(event) => patch({ backend: event.target.value as 'numpy' | 'jax' })}
            >
              <option value="numpy">NumPy</option>
              <option value="jax">JAX</option>
            </select>
          </label>
          <label>
            Restricted expression
            <textarea
              rows={6}
              value={node.expression ?? ''}
              onChange={(event) => patch({ expression: event.target.value })}
            />
          </label>
        </>
      )}

      <section className="inspector-section">
        <h3>Registered UDFs</h3>
        {matchingUdfs.length === 0 && <p className="muted">No compatible UDFs installed.</p>}
        {matchingUdfs.map((entry) => {
          const name = value(entry, 'name');
          const version = value(entry, 'version');
          const checked = node.udfs.some((reference) => reference.name === name && reference.version === version);
          return (
            <label className="udf-option" key={`${name}-${version}`}>
              <input
                type="checkbox"
                checked={checked}
                onChange={(event) => toggleUdf(name, version, event.target.checked)}
              />
              <span>
                <strong>{name}</strong>
                <small>v{version} · {value(entry, 'description')}</small>
              </span>
            </label>
          );
        })}
      </section>

      {isTable && (
        <section className="inspector-section">
          <h3>Declared Arrow schemas</h3>
          {inputNames.map((name) => (
            <div className="port-schema" key={name}>
              <span className="port-label">in · {name}</span>
              <SchemaEditor
                fields={schema('input', name)}
                arrowTypes={arrowTypes}
                onChange={(fields) => updateSchema('input', name, fields)}
              />
            </div>
          ))}
          <div className="port-schema">
            <span className="port-label">out · output</span>
            <SchemaEditor
              fields={schema('output', 'output')}
              arrowTypes={arrowTypes}
              onChange={(fields) => updateSchema('output', 'output', fields)}
            />
          </div>
        </section>
      )}

      <button className="danger-button" type="button" onClick={onDelete}>
        Delete node
      </button>
    </aside>
  );
}
