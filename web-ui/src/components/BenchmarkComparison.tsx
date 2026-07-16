import { useMemo, useRef, useState } from 'react';

import {
  compareBenchmarkReports,
  parseBenchmarkReport,
  type BenchmarkReport,
} from './benchmarkComparison';

const duration = (seconds: number): string => {
  if (seconds >= 1) return `${seconds.toFixed(3)} s`;
  if (seconds >= 0.001) return `${(seconds * 1_000).toFixed(3)} ms`;
  return `${(seconds * 1_000_000).toFixed(2)} µs`;
};

const reportLabel = (report: BenchmarkReport | null): string => {
  if (!report) return 'No report loaded';
  const commit = report.commit_info?.id;
  return `${report.benchmarks.length} cases${typeof commit === 'string' ? ` · ${commit.slice(0, 8)}` : ''}`;
};

const issueValue = (value: unknown): string => {
  if (value === undefined) return 'missing';
  if (typeof value === 'string') return value;
  return JSON.stringify(value);
};

export function BenchmarkComparison() {
  const [baseline, setBaseline] = useState<BenchmarkReport | null>(null);
  const [current, setCurrent] = useState<BenchmarkReport | null>(null);
  const [error, setError] = useState('');
  const loadGeneration = useRef({ baseline: 0, current: 0 });
  const result = useMemo(
    () => (baseline && current ? compareBenchmarkReports(baseline, current) : null),
    [baseline, current],
  );

  const load = async (file: File | undefined, target: 'baseline' | 'current') => {
    if (!file) return;
    const generation = loadGeneration.current[target] + 1;
    loadGeneration.current[target] = generation;
    if (target === 'baseline') setBaseline(null);
    else setCurrent(null);
    setError('');
    try {
      const contents = await file.text();
      if (loadGeneration.current[target] !== generation) return;
      const report = parseBenchmarkReport(JSON.parse(contents) as unknown);
      if (target === 'baseline') setBaseline(report);
      else setCurrent(report);
    } catch (caught) {
      if (loadGeneration.current[target] !== generation) return;
      setError((caught as Error).message);
    }
  };

  return (
    <section className="benchmark-panel panel">
      <div className="panel-heading benchmark-heading">
        <div>
          <span className="eyebrow">Performance artifacts</span>
          <h2>Benchmark comparison</h2>
          <p>Compare compatible pytest-benchmark JSON reports. Cases above 5% CoV stay informational.</p>
        </div>
        <div className="benchmark-loaders">
          <label className="benchmark-file">
            Baseline
            <span>{reportLabel(baseline)}</span>
            <input aria-label="Baseline benchmark report" type="file" accept="application/json,.json" onChange={(event) => void load(event.target.files?.[0], 'baseline')} />
          </label>
          <label className="benchmark-file">
            Current
            <span>{reportLabel(current)}</span>
            <input aria-label="Current benchmark report" type="file" accept="application/json,.json" onChange={(event) => void load(event.target.files?.[0], 'current')} />
          </label>
        </div>
      </div>

      {error && <div role="alert" className="validation-banner invalid">{error}</div>}
      {result?.status === 'incompatible' && (
        <div role="alert" className="validation-banner invalid">
          <strong>Incompatible benchmark reports</strong>
          <ul>
            {result.issues.map((issue, index) => (
              <li key={`${issue.code}-${issue.field}-${index}`}>
                <code>{issue.code}</code>{' '}
                <span>{issue.field}</span>{' '}
                {issueValue(issue.baseline)} → {issueValue(issue.current)}
              </li>
            ))}
          </ul>
        </div>
      )}
      {result?.status === 'unverified' && result.issues.length > 0 && (
        <div className="validation-banner">
          <strong>Unverified</strong>
          <span>No performance classification was made because contract-v2 metadata is missing.</span>
        </div>
      )}
      {result?.status === 'unverified' && result.issues.length === 0 && (
        <div className="validation-banner">The reports do not contain matching contract-v2 benchmark work.</div>
      )}
      {!baseline || !current ? (
        <div className="benchmark-empty">Load a baseline and current report produced by the same benchmark scale and runner class.</div>
      ) : result?.status === 'compatible' && result.rows.length > 0 && (
        <div className="table-wrap benchmark-table">
          <table>
            <thead>
              <tr><th>Scenario</th><th>Baseline</th><th>Current</th><th>Change</th><th>CoV</th><th>Assessment</th></tr>
            </thead>
            <tbody>
              {result.rows.map((row) => (
                <tr key={row.key}>
                  <td>{row.scenario}{row.backend && <small>{row.backend}</small>}</td>
                  <td>{duration(row.baselineMean)}</td>
                  <td>{duration(row.currentMean)}</td>
                  <td className={row.deltaPercent > 0 ? 'negative-delta' : 'positive-delta'}>{row.deltaPercent > 0 ? '+' : ''}{row.deltaPercent.toFixed(1)}%</td>
                  <td>{row.baselineCovPercent.toFixed(1)}% → {row.currentCovPercent.toFixed(1)}%</td>
                  <td><span className={`comparison-status ${row.status}`}>{row.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
