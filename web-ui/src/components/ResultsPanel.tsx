import { useEffect, type CSSProperties } from 'react';

import { PanelResizeHandle } from './PanelResizeHandle';
import {
  PANEL_LIMITS,
  maxMetricsWidth,
  useElementWidth,
} from './panelLayout';
import type { RunResponse, RunResultPreview, ValidationReport } from '../types';

interface ResultsPanelProps {
  validation: ValidationReport | null;
  run: RunResponse | null;
  metricsWidth: number;
  onMetricsWidthChange: (width: number) => void;
  onMetricsWidthReset: () => void;
  onCancel: () => void;
}

const milliseconds = (nanoseconds: number): string => `${(nanoseconds / 1_000_000).toFixed(2)} ms`;

export function ResultsPanel({
  validation,
  run,
  metricsWidth,
  onMetricsWidthChange,
  onMetricsWidthReset,
  onCancel,
}: ResultsPanelProps) {
  const result = run?.result as unknown as RunResultPreview | null | undefined;
  const { ref: resultGridRef, width: resultGridWidth } = useElementWidth<HTMLDivElement>();
  const metricsMaximum = resultGridWidth > 0
    ? maxMetricsWidth(resultGridWidth)
    : Math.max(PANEL_LIMITS.inspector.max, metricsWidth);
  const safeMetricsWidth = Math.min(
    metricsMaximum,
    Math.max(PANEL_LIMITS.metrics.min, metricsWidth),
  );
  useEffect(() => {
    if (resultGridWidth > 0 && safeMetricsWidth !== metricsWidth) {
      onMetricsWidthChange(safeMetricsWidth);
    }
  }, [metricsWidth, onMetricsWidthChange, resultGridWidth, safeMetricsWidth]);
  const resultGridStyle = {
    '--metrics-width': `${safeMetricsWidth}px`,
  } as CSSProperties;

  return (
    <section className="results panel">
      <div className="panel-heading results-heading">
        <div>
          <span className="eyebrow">Validation & preview</span>
          <h2>Run observatory</h2>
        </div>
        {run && <span className={`status-pill ${run.status}`}>{run.status}</span>}
        {run && ['pending', 'running'].includes(run.status) && (
          <button className="text-button" type="button" onClick={onCancel}>Cancel</button>
        )}
      </div>

      {validation && (
        <div className={`validation-banner ${validation.valid ? 'valid' : 'invalid'}`}>
          <strong>{validation.valid ? 'Graph is valid' : 'Graph needs attention'}</strong>
          <span>
            {validation.valid
              ? validation.fingerprint
                ? `Fingerprint ${validation.fingerprint}`
                : 'No validation issues'
              : validation.issues.map((issue) => issue.message).join(' · ')}
          </span>
        </div>
      )}

      {run?.error && <div className="validation-banner invalid">{run.error}</div>}

      {!result && !run?.error && (
        <div className="empty-state">
          <div className="empty-orbit" />
          <p>Validate or run the graph to inspect rows, plans, and node timings.</p>
        </div>
      )}

      {result && (
        <div className="result-grid" ref={resultGridRef} style={resultGridStyle}>
          <div className="output-stack">
            {Object.entries(result.outputs).map(([name, output]) => (
              <article className="output-card" key={name}>
                <header>
                  <div>
                    <span className="eyebrow">Graph output</span>
                    <h3>{name}</h3>
                  </div>
                  <span>{output.total_rows.toLocaleString()} rows{output.truncated ? ' · preview' : ''}</span>
                </header>
                {output.kind === 'table' && output.rows && (
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>{(output.schema ?? []).map((field) => <th key={field.name}>{field.name}<small>{field.type}</small></th>)}</tr>
                      </thead>
                      <tbody>
                        {output.rows.map((row, index) => (
                          <tr key={index}>{(output.schema ?? []).map((field) => <td key={field.name}>{String(row[field.name] ?? 'null')}</td>)}</tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
                {output.kind === 'array' && <pre>{JSON.stringify(output.data, null, 2)}</pre>}
              </article>
            ))}
          </div>

          <PanelResizeHandle
            label="Resize Metrics"
            value={safeMetricsWidth}
            min={PANEL_LIMITS.metrics.min}
            max={metricsMaximum}
            grow="end"
            onChange={onMetricsWidthChange}
            onReset={onMetricsWidthReset}
          />

          <aside className="metrics-stack">
            <article className="metric-card">
              <span className="eyebrow">Node timings</span>
              {Object.entries(result.node_timings).map(([node, timing]) => (
                <div className="metric-row" key={node}>
                  <strong>{node}</strong>
                  <span>{milliseconds(timing.duration_ns)}</span>
                </div>
              ))}
            </article>
            {result.datafusion_metrics.map((metric, index) => (
              <article className="plan-card" key={`${metric.query_id}-${metric.node_id}-${index}`}>
                <span className="eyebrow">DataFusion · {metric.node_id ?? `query ${metric.query_id}`}</span>
                <div className="metric-row"><span>Planning</span><strong>{milliseconds(metric.planning_ns)}</strong></div>
                <div className="metric-row"><span>Execution</span><strong>{milliseconds(metric.execution_ns)}</strong></div>
                <details>
                  <summary>Logical plan</summary>
                  <pre>{metric.logical_plan}</pre>
                </details>
                <details>
                  <summary>Physical plan</summary>
                  <pre>{metric.physical_plan}</pre>
                </details>
              </article>
            ))}
          </aside>
        </div>
      )}
    </section>
  );
}
