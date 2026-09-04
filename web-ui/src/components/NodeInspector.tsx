import { SchemaEditor } from './SchemaEditor';
import { InputAliasEditor } from './InputAliasEditor';
import type { SqlInputAliasEdit } from './inputAliasEditorModel';
import type {
  ArrowFieldConfig,
  NodeConfig,
  OperatorSpec,
  PortConfig,
  UdfCatalogEntry,
  UdfReference,
} from '../types';
import type { LoweredNodeInspection } from './projectInspectionModel';
import { derivedInputNames, derivedOutputNames } from '../portNamesModel';

interface NodeInspectorProps {
  node: NodeConfig;
  inspection?: LoweredNodeInspection;
  arrowTypes: readonly string[];
  udfs: UdfCatalogEntry[];
  onChange: (node: NodeConfig) => void;
  onSqlAliasEdit: (edit: SqlInputAliasEdit) => void;
  onDelete: () => void;
}

type ExpressionOperator = Extract<OperatorSpec, { kind: 'expression' }>;
type SqlOperator = Extract<OperatorSpec, { kind: 'sql' }>;
type StreamJoinOperator = Extract<OperatorSpec, { kind: 'stream_join' }>;
type StreamJoinBounds = StreamJoinOperator['spec']['bounds'];
type StreamJoinLimits = StreamJoinOperator['spec']['limits'];

/** Parses one integer input, truncating fractions and clamping to [min, MAX_SAFE_INTEGER]. */
const parseBoundedInteger = (raw: string, min: number): number => {
  const parsed = Math.trunc(Number(raw));
  if (!Number.isFinite(parsed)) return min;
  return Math.min(Math.max(parsed, min), Number.MAX_SAFE_INTEGER);
};

const boundMicros = (bounds: StreamJoinBounds, field: keyof StreamJoinBounds): number =>
  (field === 'before_micros' ? bounds.before_micros : bounds.after_micros);

const withBound = (
  bounds: StreamJoinBounds,
  field: keyof StreamJoinBounds,
  raw: string,
): StreamJoinBounds => (field === 'before_micros'
  ? { ...bounds, before_micros: parseBoundedInteger(raw, 0) }
  : { ...bounds, after_micros: parseBoundedInteger(raw, 0) });

const limitValue = (limits: StreamJoinLimits, field: keyof StreamJoinLimits): number => {
  switch (field) {
    case 'max_state_rows_per_side':
      return limits.max_state_rows_per_side;
    case 'max_state_bytes_per_side':
      return limits.max_state_bytes_per_side;
    default:
      return limits.max_matches_per_input_batch;
  }
};

const withLimit = (
  limits: StreamJoinLimits,
  field: keyof StreamJoinLimits,
  raw: string,
): StreamJoinLimits => {
  const value = parseBoundedInteger(raw, 1);
  switch (field) {
    case 'max_state_rows_per_side':
      return { ...limits, max_state_rows_per_side: value };
    case 'max_state_bytes_per_side':
      return { ...limits, max_state_bytes_per_side: value };
    default:
      return { ...limits, max_matches_per_input_batch: value };
  }
};

export function NodeInspector({
  node,
  inspection,
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
  const patchStreamJoin = (change: Partial<StreamJoinOperator['spec']>) => {
    if (node.operator.kind !== 'stream_join') return;
    patchNode({ operator: { ...node.operator, spec: { ...node.operator.spec, ...change } } });
  };
  const inputNames = derivedInputNames(node);
  const outputNames = derivedOutputNames(node);
  const isTable =
    node.operator.kind !== 'external'
    || ![...node.input_ports, ...node.output_ports].some((port) => port.kind === 'array');
  const matchingUdfs =
    node.operator.kind === 'expression' || node.operator.kind === 'sql'
      ? udfs
      : [];
  const streamJoin = node.operator.kind === 'stream_join' ? node.operator : null;

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

      {streamJoin && (
        <section className="inspector-section">
          <h3>Bounded event-time Join</h3>
          <label>
            Left keys
            <input
              placeholder="account_id, region"
              value={streamJoin.spec.left_keys.join(', ')}
              onChange={(event) => {
                patchStreamJoin({
                  left_keys: event.target.value.split(',').map((key) => key.trim()).filter(Boolean),
                });
              }}
            />
          </label>
          <label>
            Right keys
            <input
              placeholder="account_id, region"
              value={streamJoin.spec.right_keys.join(', ')}
              onChange={(event) => {
                patchStreamJoin({
                  right_keys: event.target.value.split(',').map((key) => key.trim()).filter(Boolean),
                });
              }}
            />
          </label>
          <label>
            Left event-time column
            <input
              value={streamJoin.spec.left_event_time}
              onChange={(event) => {
                patchStreamJoin({ left_event_time: event.target.value });
              }}
            />
          </label>
          <label>
            Right event-time column
            <input
              value={streamJoin.spec.right_event_time}
              onChange={(event) => {
                patchStreamJoin({ right_event_time: event.target.value });
              }}
            />
          </label>
          {(['before_micros', 'after_micros'] as const).map((field) => (
            <label key={field}>
              {field.replace('_', ' ')}
              <input
                type="number"
                min={0}
                max={Number.MAX_SAFE_INTEGER}
                value={boundMicros(streamJoin.spec.bounds, field)}
                onChange={(event) => {
                  patchStreamJoin({
                    bounds: withBound(streamJoin.spec.bounds, field, event.target.value),
                  });
                }}
              />
            </label>
          ))}
          {([
            'max_state_rows_per_side',
            'max_state_bytes_per_side',
            'max_matches_per_input_batch',
          ] as const).map((field) => (
            <label key={field}>
              {field.replaceAll('_', ' ')}
              <input
                type="number"
                min={1}
                max={Number.MAX_SAFE_INTEGER}
                value={limitValue(streamJoin.spec.limits, field)}
                onChange={(event) => {
                  patchStreamJoin({
                    limits: withLimit(streamJoin.spec.limits, field, event.target.value),
                  });
                }}
              />
            </label>
          ))}
          <label>
            Left prefix
            <input
              value={streamJoin.spec.left_prefix}
              onChange={(event) => {
                patchStreamJoin({ left_prefix: event.target.value });
              }}
            />
          </label>
          <label>
            Right prefix
            <input
              value={streamJoin.spec.right_prefix}
              onChange={(event) => {
                patchStreamJoin({ right_prefix: event.target.value });
              }}
            />
          </label>
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

      {inspection && (
        <section
          aria-label="Lowered project inspection"
          className="inspector-section project-inspection"
        >
          <div className="inspection-heading">
            <h3>Lowered project inspection</h3>
            <span>{inspection.contract}</span>
          </div>
          <dl className="inspection-facts">
            <div>
              <dt>Lowered node</dt>
              <dd>{inspection.nodeId} · {inspection.nodeKind}</dd>
            </div>
            <div>
              <dt>State estimate</dt>
              <dd>{inspection.state}</dd>
            </div>
            <div>
              <dt>Watermark</dt>
              <dd>{inspection.watermark}</dd>
            </div>
            <div>
              <dt>Provider identity</dt>
              <dd>{inspection.providerIdentity}</dd>
            </div>
          </dl>
          <h4>Source expressions</h4>
          {inspection.sourceExpressions.length ? (
            <ul className="inspection-list source-expression-list">
              {inspection.sourceExpressions.map((expression, index) => (
                <li key={`${index}-${expression}`}><code>{expression}</code></li>
              ))}
            </ul>
          ) : <p className="inspection-empty">none recorded by this node</p>}
          <h4>Static inputs</h4>
          {inspection.staticInputs.length ? (
            <ul className="inspection-list">
              {inspection.staticInputs.map((input) => <li key={input}>{input}</li>)}
            </ul>
          ) : <p className="muted">none declared</p>}
          <h4>Copy boundaries</h4>
          {inspection.copyBoundaries.length ? (
            <ol className="inspection-list">
              {inspection.copyBoundaries.map((boundary) => (
                <li key={boundary}>{boundary}</li>
              ))}
            </ol>
          ) : <p className="inspection-empty">none</p>}
        </section>
      )}

      <button className="danger-button" type="button" onClick={onDelete}>
        Delete node
      </button>
    </aside>
  );
}
