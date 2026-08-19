import { useEffect, useState } from 'react';

import type {
  ConnectorCapability,
  EditableProject,
  ProjectSinkBinding,
  ProjectSourceBinding,
} from '../types';

interface StreamConfigEditorProps {
  project: EditableProject;
  connectors: readonly ConnectorCapability[];
  onChange: (project: EditableProject) => void;
}

interface JsonEditorProps {
  label: string;
  value: object;
  onChange: (value: Record<string, unknown>) => void;
}

const connectorKey = (connector: ConnectorCapability): string =>
  `${connector.provider}:${connector.name}:${connector.version}`;

const connectorRef = (connector: ConnectorCapability) => ({
  provider: connector.provider,
  name: connector.name,
  version: connector.version,
});

function JsonEditor({ label, value, onChange }: JsonEditorProps) {
  const canonical = JSON.stringify(value, null, 2);
  const [text, setText] = useState(canonical);
  const [error, setError] = useState('');

  useEffect(() => setText(canonical), [canonical]);

  return (
    <label>
      {label}
      <textarea
        value={text}
        rows={4}
        onChange={(event) => {
          const next = event.target.value;
          setText(next);
          try {
            const parsed: unknown = JSON.parse(next);
            if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
              throw new Error('expected a JSON object');
            }
            setError('');
            onChange(parsed as Record<string, unknown>);
          } catch (parseError) {
            setError((parseError as Error).message);
          }
        }}
      />
      {error && <small className="field-error">{error}</small>}
    </label>
  );
}

const defaultSource = (
  connector: ConnectorCapability | undefined,
): ProjectSourceBinding | null => connector ? ({
  binding: 'input',
  connector: connectorRef(connector),
  format: null,
  options: connector.name === 'file'
    ? { path: '.calc-flow-input/input.json', format: 'json' }
    : {},
  secrets: {},
  watermark: { policy: 'disabled' },
  schema: [],
}) : null;

const defaultSink = (
  connector: ConnectorCapability | undefined,
): ProjectSinkBinding | null => connector ? ({
  binding: 'output',
  connector: connectorRef(connector),
  delivery: 'at_least_once',
  format: null,
  options: connector.name === 'file'
    ? { path: '.calc-flow-output', output: 'results' }
    : {},
  secrets: {},
}) : null;

export function StreamConfigEditor({
  project,
  connectors,
  onChange,
}: StreamConfigEditorProps) {
  const sources = connectors.filter(({ kind }) => kind === 'source' || kind === 'both');
  const sinks = connectors.filter(({ kind }) => kind === 'sink' || kind === 'both');
  const streamOptions = project.runtime.mode === 'stream'
    ? project.runtime.options
    : null;
  const streamMode = streamOptions !== null;

  const setMode = (mode: 'batch' | 'stream') => {
    if (mode === 'stream') {
      const source = defaultSource(sources[0]);
      const sink = defaultSink(sinks[0]);
      onChange({
        ...project,
        runtime: {
          mode: 'stream',
          options: {
            checkpoint_interval_ms: 30_000,
            max_batch_rows: 10_000,
            max_batch_bytes: 64 * 1024 * 1024,
          },
        },
        data_sources: [],
        sources: source ? [source] : [],
        sinks: sink ? [sink] : [],
      });
      return;
    }
    onChange({
      ...project,
      runtime: {
        mode: 'batch',
        options: {
          max_input_bytes: 10 * 1024 * 1024,
          max_rows: 100_000,
          timeout_seconds: 30,
          memory_limit_mb: 512,
          output_rows: 1000,
        },
      },
      data_sources: project.data_sources.length ? project.data_sources : [{
        id: 'sample',
        input: 'input',
        format: 'inline_json',
        data: [{ a: 1, b: 2 }],
      }],
      sources: [],
      sinks: [],
    });
  };

  const updateSource = (index: number, update: Partial<ProjectSourceBinding>) => {
    onChange({
      ...project,
      sources: project.sources.map((source, current) =>
        current === index ? { ...source, ...update } : source),
    });
  };

  const updateSink = (index: number, update: Partial<ProjectSinkBinding>) => {
    onChange({
      ...project,
      sinks: project.sinks.map((sink, current) =>
        current === index ? { ...sink, ...update } : sink),
    });
  };

  return (
    <section className="stream-config">
      <span className="eyebrow">Runtime mode</span>
      <div className="segmented-control" role="group" aria-label="Runtime mode">
        <button
          type="button"
          className={!streamMode ? 'active' : ''}
          onClick={() => setMode('batch')}
        >
          Batch
        </button>
        <button
          type="button"
          className={streamMode ? 'active' : ''}
          onClick={() => setMode('stream')}
        >
          Stream
        </button>
      </div>

      {streamOptions && (
        <>
          <label>
            Checkpoint interval (ms)
            <input
              type="number"
              min={1}
              value={streamOptions.checkpoint_interval_ms}
              onChange={(event) => onChange({
                ...project,
                runtime: {
                  mode: 'stream',
                  options: {
                    ...streamOptions,
                    checkpoint_interval_ms: Number(event.target.value),
                  },
                },
              })}
            />
          </label>
          <label>
            State root
            <input
              value={project.state.root}
              onChange={(event) => onChange({
                ...project,
                state: { ...project.state, root: event.target.value },
              })}
            />
          </label>

          <div className="binding-heading">
            <strong>Sources</strong>
            <button
              type="button"
              className="text-button"
              disabled={!sources.length}
              onClick={() => {
                const source = defaultSource(sources[0]);
                if (source) onChange({ ...project, sources: [...project.sources, source] });
              }}
            >
              Add
            </button>
          </div>
          {project.sources.map((source, index) => (
            <article className="binding-card" key={`source-${index}`}>
              <label>
                Graph input
                <input
                  value={source.binding}
                  onChange={(event) => updateSource(index, { binding: event.target.value })}
                />
              </label>
              <label>
                Connector
                <select
                  value={`${source.connector.provider}:${source.connector.name}:${source.connector.version}`}
                  onChange={(event) => {
                    const selected = sources.find((item) => connectorKey(item) === event.target.value);
                    if (selected) updateSource(index, { connector: connectorRef(selected) });
                  }}
                >
                  {sources.map((connector) => (
                    <option key={connectorKey(connector)} value={connectorKey(connector)}>
                      {connector.name}@{connector.version}
                    </option>
                  ))}
                </select>
              </label>
              <JsonEditor
                label="Options"
                value={source.options}
                onChange={(options) => updateSource(index, { options })}
              />
              <JsonEditor
                label="Secret references"
                value={source.secrets}
                onChange={(secrets) => updateSource(index, {
                  secrets: secrets as ProjectSourceBinding['secrets'],
                })}
              />
              <button
                type="button"
                className="text-button danger"
                onClick={() => onChange({
                  ...project,
                  sources: project.sources.filter((_, current) => current !== index),
                })}
              >
                Remove source
              </button>
            </article>
          ))}

          <div className="binding-heading">
            <strong>Sinks</strong>
            <button
              type="button"
              className="text-button"
              disabled={!sinks.length}
              onClick={() => {
                const sink = defaultSink(sinks[0]);
                if (sink) onChange({ ...project, sinks: [...project.sinks, sink] });
              }}
            >
              Add
            </button>
          </div>
          {project.sinks.map((sink, index) => (
            <article className="binding-card" key={`sink-${index}`}>
              <label>
                Graph output
                <input
                  value={sink.binding}
                  onChange={(event) => updateSink(index, { binding: event.target.value })}
                />
              </label>
              <label>
                Connector
                <select
                  value={`${sink.connector.provider}:${sink.connector.name}:${sink.connector.version}`}
                  onChange={(event) => {
                    const selected = sinks.find((item) => connectorKey(item) === event.target.value);
                    if (selected) updateSink(index, { connector: connectorRef(selected) });
                  }}
                >
                  {sinks.map((connector) => (
                    <option key={connectorKey(connector)} value={connectorKey(connector)}>
                      {connector.name}@{connector.version}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Delivery
                <select
                  value={sink.delivery}
                  onChange={(event) => updateSink(index, {
                    delivery: event.target.value as ProjectSinkBinding['delivery'],
                  })}
                >
                  <option value="best_effort">Best effort</option>
                  <option value="at_least_once">At least once</option>
                  <option value="exactly_once">Exactly once</option>
                </select>
              </label>
              <JsonEditor
                label="Options"
                value={sink.options}
                onChange={(options) => updateSink(index, { options })}
              />
              <JsonEditor
                label="Secret references"
                value={sink.secrets}
                onChange={(secrets) => updateSink(index, {
                  secrets: secrets as ProjectSinkBinding['secrets'],
                })}
              />
              <button
                type="button"
                className="text-button danger"
                onClick={() => onChange({
                  ...project,
                  sinks: project.sinks.filter((_, current) => current !== index),
                })}
              >
                Remove sink
              </button>
            </article>
          ))}
        </>
      )}
    </section>
  );
}
