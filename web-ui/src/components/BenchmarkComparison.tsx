import { useMemo, useState } from 'react';

interface BenchmarkStats {
  mean: number;
  stddev: number;
  rounds: number;
}

interface BenchmarkEntry {
  name: string;
  fullname: string;
  group?: string;
  stats: BenchmarkStats;
  extra_info?: Record<string, unknown>;
}

export interface BenchmarkReport {
  benchmarks: BenchmarkEntry[];
  machine_info?: Record<string, unknown>;
  commit_info?: Record<string, unknown>;
}

export interface BenchmarkComparisonRow {
  key: string;
  scenario: string;
  backend: string | null;
  scale: string | null;
  baselineMean: number;
  currentMean: number;
  deltaPercent: number;
  baselineCovPercent: number;
  currentCovPercent: number;
  status: 'regression' | 'improvement' | 'stable' | 'noisy';
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const finiteNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

const parseEntry = (value: unknown): BenchmarkEntry => {
  if (!isRecord(value) || typeof value.name !== 'string' || typeof value.fullname !== 'string') {
    throw new Error('Benchmark entries require name and fullname strings');
  }
  if (!isRecord(value.stats)) throw new Error(`Benchmark ${value.name} has no stats object`);
  const { mean, stddev, rounds } = value.stats;
  if (
    !finiteNumber(mean)
    || mean <= 0
    || !finiteNumber(stddev)
    || stddev < 0
    || !finiteNumber(rounds)
    || !Number.isInteger(rounds)
    || rounds <= 0
  ) {
    throw new Error(`Benchmark ${value.name} contains invalid statistics`);
  }
  return {
    name: value.name,
    fullname: value.fullname,
    group: typeof value.group === 'string' ? value.group : undefined,
    stats: { mean, stddev, rounds },
    extra_info: isRecord(value.extra_info) ? value.extra_info : undefined,
  };
};

export const parseBenchmarkReport = (value: unknown): BenchmarkReport => {
  if (!isRecord(value) || !Array.isArray(value.benchmarks)) {
    throw new Error('Expected a pytest-benchmark JSON report');
  }
  if (!value.benchmarks.length) throw new Error('Benchmark report contains no cases');
  return {
    benchmarks: value.benchmarks.map(parseEntry),
    machine_info: isRecord(value.machine_info) ? value.machine_info : undefined,
    commit_info: isRecord(value.commit_info) ? value.commit_info : undefined,
  };
};

const textExtra = (entry: BenchmarkEntry, name: string): string | null => {
  const value = entry.extra_info?.[name];
  return typeof value === 'string' ? value : null;
};

const entryKey = (entry: BenchmarkEntry): string => {
  const scenario = textExtra(entry, 'scenario') ?? entry.fullname;
  const backend = textExtra(entry, 'backend') ?? '';
  return `${scenario}\u0000${backend}`;
};

const coefficientOfVariation = (entry: BenchmarkEntry): number =>
  (entry.stats.stddev / entry.stats.mean) * 100;

export const compareBenchmarkReports = (
  baseline: BenchmarkReport,
  current: BenchmarkReport,
): BenchmarkComparisonRow[] => {
  const baselineByKey = new Map(baseline.benchmarks.map((entry) => [entryKey(entry), entry]));
  return current.benchmarks.flatMap((entry) => {
    const key = entryKey(entry);
    const reference = baselineByKey.get(key);
    if (!reference) return [];
    const baselineScale = textExtra(reference, 'scale');
    const currentScale = textExtra(entry, 'scale');
    if (baselineScale && currentScale && baselineScale !== currentScale) return [];
    const baselineCovPercent = coefficientOfVariation(reference);
    const currentCovPercent = coefficientOfVariation(entry);
    const deltaPercent = ((entry.stats.mean / reference.stats.mean) - 1) * 100;
    let status: BenchmarkComparisonRow['status'] = 'stable';
    if (baselineCovPercent > 5 || currentCovPercent > 5) status = 'noisy';
    else if (deltaPercent > 10) status = 'regression';
    else if (deltaPercent < -10) status = 'improvement';
    return [{
      key,
      scenario: textExtra(entry, 'scenario') ?? entry.name,
      backend: textExtra(entry, 'backend'),
      scale: textExtra(entry, 'scale'),
      baselineMean: reference.stats.mean,
      currentMean: entry.stats.mean,
      deltaPercent,
      baselineCovPercent,
      currentCovPercent,
      status,
    }];
  }).sort((left, right) => right.deltaPercent - left.deltaPercent);
};

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

export function BenchmarkComparison() {
  const [baseline, setBaseline] = useState<BenchmarkReport | null>(null);
  const [current, setCurrent] = useState<BenchmarkReport | null>(null);
  const [error, setError] = useState('');
  const rows = useMemo(
    () => (baseline && current ? compareBenchmarkReports(baseline, current) : []),
    [baseline, current],
  );

  const load = async (file: File | undefined, target: 'baseline' | 'current') => {
    if (!file) return;
    try {
      const report = parseBenchmarkReport(JSON.parse(await file.text()) as unknown);
      if (target === 'baseline') setBaseline(report);
      else setCurrent(report);
      setError('');
    } catch (caught) {
      setError((caught as Error).message);
    }
  };

  const scales = new Set(
    [baseline, current]
      .flatMap((report) => report?.benchmarks ?? [])
      .map((entry) => textExtra(entry, 'scale'))
      .filter((scale): scale is string => scale !== null),
  );

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

      {error && <div className="validation-banner invalid">{error}</div>}
      {scales.size > 1 && (
        <div className="validation-banner invalid">Reports contain different dataset scales and should not be compared.</div>
      )}
      {baseline && current && scales.size <= 1 && !rows.length && (
        <div className="validation-banner invalid">The reports do not contain matching scenarios.</div>
      )}
      {!baseline || !current ? (
        <div className="benchmark-empty">Load a baseline and current report produced by the same benchmark scale and runner class.</div>
      ) : scales.size <= 1 && rows.length > 0 && (
        <div className="table-wrap benchmark-table">
          <table>
            <thead>
              <tr><th>Scenario</th><th>Baseline</th><th>Current</th><th>Change</th><th>CoV</th><th>Assessment</th></tr>
            </thead>
            <tbody>
              {rows.map((row) => (
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
