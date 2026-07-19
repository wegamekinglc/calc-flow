import { SchemaEditor } from './SchemaEditor';
import { InputAliasEditor } from './InputAliasEditor';
import type { SqlInputAliasEdit } from './inputAliasEditor';
import type {
  ArrowFieldConfig,
  NodeConfig,
  OperatorSpec,
  PortConfig,
  UdfCatalogEntry,
  UdfReference,
} from '../types';

interface NodeInspectorProps {
  node: NodeConfig;
  arrowTypes: readonly string[];
  udfs: UdfCatalogEntry[];
  onChange: (node: NodeConfig) => void;
  onSqlAliasEdit: (edit: SqlInputAliasEdit) => void;
  onDelete: () => void;
}

type ExpressionOperator = Extract<OperatorSpec, { kind: 'expression' }>;
type SqlOperator = Extract<OperatorSpec, { kind: 'sql' }>;

export function NodeInspector({
  node,
  arrowTypes,
  udfs,
  onChange,
  onSqlAliasEdit,
  onDelete,
}: NodeInspectorProps) {
  const patchNode = (change: Partial<NodeConfig>) => onChange({ ...node, ...change });
  const patchExpression = (change: Partial<ExpressionOperator>) => {
    if (node.operator.kind !== 'expression') return;
    patchNode({ operator: { ...node.operator, ...change } });
  };
  const patchSql = (change: Partial<SqlOperator>) => {
    if (node.operator.kind !== 'sql') return;
    patchNode({ operator: { ...node.operator, ...change } });
  };
  const declaredInputs = node.input_ports.map((port) => port.name);
  const inputNames = declaredInputs.length
    ? declaredInputs
    : node.operator.kind === 'sql'
      ? node.operator.aliases
      : node.operator.kind === 'expression'
        ? ['input']
        : [];
  const outputNames = node.output_ports.length
    ? node.output_ports.map((port) => port.name)
    : node.operator.kind === 'external'
      ? []
      : ['output'];
  const isTable =
    node.operator.kind !== 'external'
    || ![...node.input_ports, ...node.output_ports].some((port) => port.kind === 'array');
  const matchingUdfs =
    node.operator.kind === 'expression' || node.operator.kind === 'sql'
      ? udfs.filter((entry) => entry.kind === 'data_fusion_scalar')
      : [];

  const schema = (direction: 'input' | 'output', name: string): ArrowFieldConfig[] => {
    const ports = direction === 'input' ? node.input_ports : node.output_ports;
    return ports.find((port) => port.name === name)?.schema ?? [];
  };

  const updateSchema = (
    direction: 'input' | 'output',
    name: string,
    fields: ArrowFieldConfig[],
  ) => {
    const names = direction === 'input' ? inputNames : outputNames;
    const current = direction === 'input' ? node.input_ports : node.output_ports;
    const ports: PortConfig[] = names.map((portName) => {
      const existing = current.find((port) => port.name === portName);
      return {
        name: portName,
        kind: existing?.kind ?? (isTable ? 'table' : 'array'),
        required: existing?.required ?? true,
        schema: portName === name ? fields : (existing?.schema ?? []),
      };
    });
    patchNode(direction === 'input' ? { input_ports: ports } : { output_ports: ports });
  };

  const toggleUdf = (entry: UdfCatalogEntry, checked: boolean) => {
    if (node.operator.kind !== 'expression' && node.operator.kind !== 'sql') return;
    const reference: UdfReference = {
      provider: entry.provider,
      name: entry.name,
      version: entry.version,
      kind: entry.kind,
    };
    const others = node.operator.udfs.filter(
      (current) =>
        current.provider !== reference.provider
        || current.name !== reference.name
        || current.version !== reference.version
        || current.kind !== reference.kind,
    );
    const udfs = checked ? [...others, reference] : others;
    if (node.operator.kind === 'expression') patchExpression({ udfs });
    else patchSql({ udfs });
  };

  return (
    <aside className="inspector panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Node inspector</span>
          <h2>{node.id}</h2>
        </div>
        <span className={`kind-chip ${node.operator.kind}`}>
          {node.operator.kind.replace('_', ' ')}
        </span>
      </div>

      <label>
        Node ID
        <input value={node.id} disabled />
      </label>

      {node.operator.kind === 'expression' && (
        <>
          <label>
            DataFusion expression
            <textarea
              rows={5}
              value={node.operator.expression}
              onChange={(event) => patchExpression({
                expression: event.target.value,
                select: [],
              })}
            />
          </label>
          <label>
            Filter expression
            <input
              placeholder="amount > 0"
              value={node.operator.filter ?? ''}
              onChange={(event) => patchExpression({ filter: event.target.value || null })}
            />
          </label>
        </>
      )}

      {node.operator.kind === 'sql' && (
        <>
          <InputAliasEditor
            aliases={node.operator.aliases}
            onAdd={() => onSqlAliasEdit({ type: 'add' })}
            onRename={(alias, nextAlias) => onSqlAliasEdit({
              type: 'rename',
              alias,
              nextAlias,
            })}
            onRemove={(alias) => onSqlAliasEdit({ type: 'remove', alias })}
          />
          <label>
            DataFusion SQL
            <textarea
              rows={9}
              value={node.operator.query}
              onChange={(event) => patchSql({ query: event.target.value })}
            />
          </label>
        </>
      )}

      {node.operator.kind === 'external' && (
        <section className="inspector-section">
          <h3>External provider</h3>
          <p className="muted">
            {node.operator.provider} · {node.operator.name} · v{node.operator.version}
          </p>
        </section>
      )}

      <section className="inspector-section">
        <h3>Registered UDFs</h3>
        {matchingUdfs.length === 0 && <p className="muted">No compatible UDFs installed.</p>}
        {matchingUdfs.map((entry) => {
          const operatorUdfs =
            node.operator.kind === 'expression' || node.operator.kind === 'sql'
              ? node.operator.udfs
              : [];
          const checked = operatorUdfs.some(
            (reference) =>
              reference.provider === entry.provider
              && reference.name === entry.name
              && reference.version === entry.version
              && reference.kind === entry.kind,
          );
          return (
            <label
              className="udf-option"
              key={`${entry.provider}-${entry.name}-${entry.version}-${entry.kind}`}
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={(event) => toggleUdf(entry, event.target.checked)}
              />
              <span>
                <strong>{entry.name}</strong>
                <small>
                  {entry.provider} · v{entry.version} · {entry.signature.return_type}
                </small>
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
          {outputNames.map((name) => (
            <div className="port-schema" key={name}>
              <span className="port-label">out · {name}</span>
              <SchemaEditor
                fields={schema('output', name)}
                arrowTypes={arrowTypes}
                onChange={(fields) => updateSchema('output', name, fields)}
              />
            </div>
          ))}
        </section>
      )}

      <button className="danger-button" type="button" onClick={onDelete}>
        Delete node
      </button>
    </aside>
  );
}
