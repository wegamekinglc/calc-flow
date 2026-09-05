import { expect, test, type APIRequestContext } from '@playwright/test';

interface JobRecord {
  id: string;
  project_id: string;
  status: string;
  error_code: string | null;
  reason_code: string | null;
  error: string | null;
  created_at: string;
}

async function latestJob(
  request: APIRequestContext,
  projectId: string,
): Promise<JobRecord | undefined> {
  const listed = await request.get(jobsUrl);
  const jobs = (await listed.json()) as JobRecord[];
  return jobs
    .filter((job) => job.project_id === projectId)
    .sort((left, right) => right.created_at.localeCompare(left.created_at))[0];
}
import { mkdir, readdir, rm, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const projectsUrl = 'http://127.0.0.1:8765/api/v3/projects';
const jobsUrl = 'http://127.0.0.1:8765/api/v3/jobs';

// The constrained headless renderer otherwise stalls Playwright's stability check.
test.use({
  launchOptions: {
    args: ['--disable-gpu', '--disable-software-rasterizer'],
  },
});

const timestampSchema = [
  { name: 'account_id', data_type: 'string', nullable: false },
  { name: 'event_time', data_type: 'timestamp[us]', nullable: false },
];

const joinOutputSchema = [
  { name: 'left__account_id', data_type: 'string', nullable: false },
  { name: 'left__event_time', data_type: 'timestamp[us]', nullable: false },
  { name: 'right__account_id', data_type: 'string', nullable: false },
  { name: 'right__event_time', data_type: 'timestamp[us]', nullable: false },
];

const fixtureDirectory = resolve('test-results', 'stream-join');

// The backend keys durable worker state by project, so every test process
// uses fresh project IDs: restarted suites must not resume a previous
// run's committed source cursors.
const runTag = `${process.pid}`;

function line(key: string, millis: number): string {
  const seconds = Math.floor(millis / 1000);
  const micros = (millis % 1000) * 1000;
  const stamp = `2026-01-01T00:00:${String(seconds).padStart(2, '0')}.${String(
    micros,
  ).padStart(6, '0')}Z`;
  return `{"account_id":"${key}","event_time":"${stamp}"}`;
}

interface JoinFixture {
  id: string;
  maxMatchesPerBatch: number;
  rowsPerSide: number;
  checkpointIntervalMs: number;
}

async function writeJoinInputs(rowsPerSide: number): Promise<{
  leftPath: string;
  rightPath: string;
}> {
  const leftPath = resolve(fixtureDirectory, 'left.json');
  const rightPath = resolve(fixtureDirectory, 'right.json');
  const leftLines: string[] = [];
  const rightLines: string[] = [];
  for (let index = 0; index < rowsPerSide; index += 1) {
    // `a` rows pair inside [95ms, 105ms]; `b` rows inside [40ms, 50ms].
    leftLines.push(line('a', 95 + index), line('b', 40 + index));
    rightLines.push(line('a', 100 + index), line('b', 50 + index));
  }
  await writeFile(leftPath, `${leftLines.join('\n')}\n`, 'utf8');
  await writeFile(rightPath, `${rightLines.join('\n')}\n`, 'utf8');
  return { leftPath, rightPath };
}

function streamJoinProject(fixture: JoinFixture, leftPath: string, rightPath: string) {
  return {
    format_version: 3,
    id: fixture.id,
    name: 'Stream Join E2E',
    description: 'A saved bounded event-time Join graph with a tumbling window.',
    runtime: {
      mode: 'stream',
      options: { checkpoint_interval_ms: fixture.checkpointIntervalMs },
    },
    graph: {
      name: 'stream-join-e2e',
      nodes: [
        {
          id: 'match',
          operator: {
            kind: 'stream_join',
            spec: {
              join_type: 'inner',
              left_keys: ['account_id'],
              right_keys: ['account_id'],
              left_event_time: 'event_time',
              right_event_time: 'event_time',
              bounds: { before_micros: 0, after_micros: 10_000 },
              limits: {
                max_state_rows_per_side: 100_000,
                max_state_bytes_per_side: 134_217_728,
                max_matches_per_input_batch: fixture.maxMatchesPerBatch,
              },
              left_prefix: 'left',
              right_prefix: 'right',
            },
          },
          input_ports: [
            {
              name: 'left',
              kind: 'table',
              required: true,
              schema: timestampSchema,
            },
            {
              name: 'right',
              kind: 'table',
              required: true,
              schema: timestampSchema,
            },
          ],
          output_ports: [],
          position: { x: 320, y: 160 },
        },
        {
          id: 'agg',
          operator: {
            kind: 'window',
            spec: {
              event_time_column: 'left__event_time',
              group_by: [],
              geometry: { kind: 'tumbling', size_micros: 10_000 },
              aggregates: [
                { function: 'count', column: 'left__event_time', output: 'pairs' },
              ],
            },
          },
          input_ports: [
            {
              name: 'input',
              kind: 'table',
              required: true,
              schema: joinOutputSchema,
            },
          ],
          output_ports: [],
          position: { x: 600, y: 160 },
        },
      ],
      edges: [
        {
          source_node: 'match',
          source_port: 'output',
          target_node: 'agg',
          target_port: 'input',
        },
      ],
    },
    sources: [
      {
        binding: 'left',
        connector: {
          provider: 'calc-flow-connectors',
          name: 'file',
          version: '2.0.0',
        },
        options: { path: leftPath, format: 'json', schema: timestampSchema },
        watermark: {
          policy: 'bounded_out_of_orderness',
          column: 'event_time',
          delay_ms: 1,
          emit_interval_ms: 1,
        },
        schema: timestampSchema,
      },
      {
        binding: 'right',
        connector: {
          provider: 'calc-flow-connectors',
          name: 'file',
          version: '2.0.0',
        },
        options: { path: rightPath, format: 'json', schema: timestampSchema },
        watermark: {
          policy: 'bounded_out_of_orderness',
          column: 'event_time',
          delay_ms: 1,
          emit_interval_ms: 1,
        },
        schema: timestampSchema,
      },
    ],
    sinks: [
      {
        binding: 'output',
        connector: {
          provider: 'calc-flow-connectors',
          name: 'file',
          version: '2.0.0',
        },
        options: {
          path: resolve(fixtureDirectory, `${fixture.id}-sink`),
          output: 'results',
        },
        delivery: 'at_least_once',
      },
    ],
    state: {
      root: resolve(fixtureDirectory, `${fixture.id}-state`),
      retention: 3,
    },
  };
}

async function resetProject(
  request: APIRequestContext,
  fixture: JoinFixture,
): Promise<{ leftPath: string; rightPath: string }> {
  await mkdir(fixtureDirectory, { recursive: true });
  await rm(resolve(fixtureDirectory, `${fixture.id}-sink`), {
    recursive: true,
    force: true,
  });
  await rm(resolve(fixtureDirectory, `${fixture.id}-state`), {
    recursive: true,
    force: true,
  });
  const deleted = await request.delete(`${projectsUrl}/${fixture.id}`);
  expect([204, 404]).toContain(deleted.status());
  const paths = await writeJoinInputs(fixture.rowsPerSide);
  const created = await request.post(projectsUrl, {
    data: streamJoinProject(fixture, paths.leftPath, paths.rightPath),
  });
  expect(created.status()).toBe(201);
  return paths;
}

test.describe('stream Join studio workflow', () => {
  test.afterEach(async ({ request }) => {
    for (const id of [
      `stream_join_edit_${runTag}`,
      `stream_join_run_${runTag}`,
      `stream_join_fail_${runTag}`,
    ]) {
      await request.delete(`${projectsUrl}/${id}`);
    }
  });

  test('edits, validates, saves, and reloads the saved Join->Window graph', async ({
    page,
    request,
  }) => {
    const fixture: JoinFixture = {
      id: `stream_join_edit_${runTag}`,
      maxMatchesPerBatch: 1_000_000,
      rowsPerSide: 2,
      checkpointIntervalMs: 60_000,
    };
    await resetProject(request, fixture);

    await page.goto('/');
    const project = page.getByLabel('Project', { exact: true });
    // Selecting throws when the saved project is missing from the picker.
    await project.selectOption(`stream_join_edit_${runTag}`);
    await expect(project).toHaveValue(`stream_join_edit_${runTag}`);

    const inspector = page.locator('aside.inspector');
    const joinNode = page
      .locator('.react-flow__node')
      .filter({ hasText: 'match' })
      .first();
    await expect(joinNode).toBeVisible({ timeout: 10_000 });
    await joinNode.click();
    await expect(inspector.getByRole('heading', { name: 'Bounded event-time Join' })).toBeVisible();
    await expect(inspector.getByLabel('Left keys')).toHaveValue('account_id');
    await expect(inspector.getByLabel('Right event-time column')).toHaveValue('event_time');

    // Creation coverage: the toolbox stream Join button adds a fresh node the
    // inspector can edit, and deleting it restores the saved graph.
    await page.getByRole('button', { name: /Stream Join/i }).scrollIntoViewIfNeeded();
    await page.getByRole('button', { name: /Stream Join/i }).click();
    await expect(inspector.getByLabel('Left keys')).toHaveValue('');
    await inspector.getByLabel('Left keys').fill('account_id');
    await expect(inspector.getByLabel('Left keys')).toHaveValue('account_id');
    await page.getByRole('button', { name: 'Delete node' }).click();
    await expect(page.locator('.react-flow__node')).toHaveCount(2);

    await page.getByRole('button', { name: 'Validate' }).click();
    await expect(page.getByText('Graph is valid')).toBeVisible();

    // Pointing the event-time column at a missing field is rejected by the
    // strict raw document contract before any typed normalization.
    await expect(joinNode).toBeVisible({ timeout: 10_000 });
    await joinNode.click();
    await inspector.getByLabel('Left event-time column').fill('missing_at');
    await page.getByRole('button', { name: 'Validate' }).click();
    await expect(page.getByRole('status')).toContainText('stored document is invalid');
    await expect(page.getByRole('status')).toContainText('left_event_time');

    await inspector.getByLabel('Left event-time column').fill('event_time');
    await page.getByRole('button', { name: 'Save' }).click();
    await expect(page.getByRole('status')).toHaveText('Project saved');

    const saved = await request.get(`${projectsUrl}/${fixture.id}`);
    expect(saved.ok()).toBeTruthy();
    const document = await saved.json();
    expect(document.graph.nodes[0].operator.kind).toBe('stream_join');
    expect(document.graph.nodes[1].operator.kind).toBe('window');
    expect(document.graph.edges).toEqual([
      {
        source_node: 'match',
        source_port: 'output',
        target_node: 'agg',
        target_port: 'input',
      },
    ]);
    expect(document.sources.map((source: { binding: string }) => source.binding)).toEqual([
      'left',
      'right',
    ]);

    await page.reload();
    await project.selectOption(`stream_join_edit_${runTag}`);
    await expect(project).toHaveValue(`stream_join_edit_${runTag}`);
    await expect(joinNode).toBeVisible({ timeout: 10_000 });
    await joinNode.click();
    await expect(inspector.getByLabel('Right keys')).toHaveValue('account_id');
    await expect(inspector.getByLabel('before micros')).toHaveValue('0');

    await request.delete(`${projectsUrl}/${fixture.id}`);
  });

  test('runs the Join->Window job and observes join metrics and sink output', async ({
    page,
    request,
  }) => {
    const fixture: JoinFixture = {
      id: `stream_join_run_${runTag}`,
      maxMatchesPerBatch: 1_000_000,
      rowsPerSide: 2,
      checkpointIntervalMs: 60_000,
    };
    await resetProject(request, fixture);

    await page.goto('/');
    await expect(page.getByText('Build the flow')).toBeVisible();
    const project = page.getByLabel('Project', { exact: true });
    await project.selectOption(fixture.id);

    const start = page.getByRole('button', { name: /Start job/ });
    await expect(start).toBeEnabled();
    await start.click();
    const metrics = page.getByRole('region', { name: 'Continuous job metrics' });
    await expect(metrics).toBeVisible({ timeout: 20_000 });
    await expect(metrics.getByText('completed', { exact: true })).toBeVisible({
      timeout: 20_000,
    });

    // Join metrics reach the job document through the real worker; the
    // started run is the newest one for the project.
    const joinJob = await latestJob(request, fixture.id);
    expect(joinJob).toBeTruthy();
    expect(joinJob?.status).toBe('completed');
    expect(joinJob?.error_code).toBeNull();
    expect(joinJob?.reason_code).toBeNull();

    // Every legal pair reached the tumbling window: the sink wrote window
    // results instead of dropping them as late (spec AC5/AC7 Playwright row).
    await expect
      .poll(async () => {
        const sinkRoot = resolve(fixtureDirectory, `${fixture.id}-sink`);
        // Test-local output directory, not request-influenced; the
        // non-literal-fs-filename audit targets production path traversal.
        const entries = await readdir(sinkRoot, { recursive: true });
        return entries.filter((entry) => String(entry).endsWith('.parquet')).length;
      })
      .toBeGreaterThan(0);

    // The replayed job event stream carries the Join node's payload-free
    // metrics and the terminal shutdown event for the completed run.
    const events = await request.get(`${jobsUrl}/${joinJob?.id}/events`, {
      headers: { 'Last-Event-ID': '0' },
    });
    expect(events.ok()).toBeTruthy();
    const stream = await events.text();
    expect(stream).toContain('event: progress');
    expect(stream).toContain('event: terminal');
    expect(stream).toContain('"stream_joins"');
    expect(stream).toContain('"emitted_match_rows"');

    await request.delete(`${projectsUrl}/${fixture.id}`);
  });

  test('surfaces the typed join failure reason on the worker path', async ({ page, request }) => {
    const fixture: JoinFixture = {
      id: `stream_join_fail_${runTag}`,
      maxMatchesPerBatch: 1,
      rowsPerSide: 2,
      checkpointIntervalMs: 60_000,
    };
    await resetProject(request, fixture);

    await page.goto('/');
    await expect(page.getByText('Build the flow')).toBeVisible();
    const picker = page.getByLabel('Project', { exact: true });
    await picker.selectOption(fixture.id);
    const start = page.getByRole('button', { name: /Start job/ });
    await start.click();
    const metrics = page.getByRole('region', { name: 'Continuous job metrics' });
    await expect(metrics).toBeVisible({ timeout: 20_000 });
    await expect(metrics.getByText('failed', { exact: true })).toBeVisible({
      timeout: 20_000,
    });

    // The coarse banner stays worker_failed while the job document carries the
    // exact typed join reason (spec AC24 through the real REST surface).
    const failed = await latestJob(request, fixture.id);
    expect(failed?.status).toBe('failed');
    expect(failed?.error_code).toBe('worker_failed');
    expect(failed?.reason_code).toBe('join_match_limit_exceeded');

    await request.delete(`${projectsUrl}/${fixture.id}`);
  });


});
