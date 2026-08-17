import {
  expect,
  test,
  type APIRequestContext,
  type Locator,
  type Page,
} from '@playwright/test';
import { readFile } from 'node:fs/promises';

const projectsUrl = 'http://127.0.0.1:8765/api/v3/projects';
const twoSourceProjectUrl = `${projectsUrl}/two_source_e2e`;

// The constrained headless renderer otherwise stalls Playwright's stability check.
test.use({
  launchOptions: {
    args: ['--disable-gpu', '--disable-software-rasterizer'],
  },
});

const twoSourceProject = {
  format_version: 2,
  id: 'two_source_e2e',
  name: 'Two source E2E',
  description: 'Two independent saved sources join downstream.',
  pipeline: {
    name: 'Two source pipeline',
    nodes: [
      {
        id: 'left_branch',
        operator: {
          kind: 'sql',
          query: 'SELECT id, value * 2 AS left_value FROM left_source',
          aliases: ['left_source'],
          udfs: [],
        },
        input_ports: [],
        output_ports: [],
        position: { x: 80, y: 80 },
      },
      {
        id: 'right_branch',
        operator: {
          kind: 'sql',
          query: 'SELECT id, adjustment AS right_value FROM right_source',
          aliases: ['right_source'],
          udfs: [],
        },
        input_ports: [],
        output_ports: [],
        position: { x: 80, y: 280 },
      },
      {
        id: 'join_result',
        operator: {
          kind: 'sql',
          query: 'SELECT l.id, l.left_value, r.right_value, l.left_value + r.right_value AS total FROM left l JOIN right r ON l.id = r.id ORDER BY l.id',
          aliases: ['left', 'right'],
          udfs: [],
        },
        input_ports: [],
        output_ports: [],
        position: { x: 480, y: 180 },
      },
    ],
    edges: [
      {
        source_node: 'left_branch',
        source_port: 'output',
        target_node: 'join_result',
        target_port: 'left',
      },
      {
        source_node: 'right_branch',
        source_port: 'output',
        target_node: 'join_result',
        target_port: 'right',
      },
    ],
    datafusion: { batch_size: 8192, target_partitions: 1 },
  },
  data_sources: [
    {
      id: 'left',
      input: 'left_source',
      format: 'inline_json',
      data: [{ id: 1, value: 3 }, { id: 2, value: 5 }],
    },
    {
      id: 'right',
      input: 'right_source',
      format: 'inline_json',
      data: [{ id: 1, adjustment: 10 }, { id: 2, adjustment: 20 }],
    },
  ],
  run_options: {
    max_input_bytes: 10485760,
    max_rows: 100000,
    timeout_seconds: 30,
    memory_limit_mb: 512,
    output_rows: 1000,
  },
};

async function deleteTwoSourceProject(request: APIRequestContext): Promise<number> {
  const response = await request.delete(twoSourceProjectUrl);
  expect([204, 404]).toContain(response.status());
  return response.status();
}

async function panelWidth(page: Page, selector: string): Promise<number> {
  return page.locator(selector).evaluate((element) => element.getBoundingClientRect().width);
}

async function dragSeparator(page: Page, label: string, deltaX: number): Promise<void> {
  const separator = page.getByRole('separator', { name: label });
  await separator.scrollIntoViewIfNeeded();
  const box = await separator.boundingBox();
  expect(box).not.toBeNull();
  const centerX = box!.x + box!.width / 2;
  const centerY = box!.y + box!.height / 2;
  await page.mouse.move(centerX, centerY);
  await page.mouse.down();
  await page.mouse.move(centerX + deltaX, centerY, { steps: 5 });
  await page.mouse.up();
}

interface MeasuredControl {
  readonly label: string;
  readonly locator: Locator;
}

interface ElementBox {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

const toolbarControls = (page: Page): MeasuredControl[] => [
  { label: 'New', locator: page.getByRole('button', { name: 'New', exact: true }) },
  { label: 'Import', locator: page.getByLabel('Import project').locator('..') },
  {
    label: 'Export JSON',
    locator: page.getByRole('button', { name: 'Export JSON', exact: true }),
  },
  {
    label: 'Export YAML',
    locator: page.getByRole('button', { name: 'Export YAML', exact: true }),
  },
  {
    label: 'Delete',
    locator: page.getByRole('button', { name: 'Delete', exact: true }),
  },
  { label: 'Save', locator: page.getByRole('button', { name: 'Save', exact: true }) },
  {
    label: 'Validate',
    locator: page.getByRole('button', { name: 'Validate', exact: true }),
  },
  {
    label: 'Run preview',
    locator: page.getByRole('button', { name: /Run preview/ }),
  },
];

async function measuredBox(locator: Locator, label: string): Promise<ElementBox> {
  await expect(locator, `${label} should be visible`).toBeVisible();
  const box = await locator.boundingBox();
  expect(box, `${label} should have a layout box`).not.toBeNull();
  return box!;
}

const boxesOverlap = (left: ElementBox, right: ElementBox): boolean => {
  const epsilon = 0.5;
  return left.x < right.x + right.width - epsilon
    && left.x + left.width > right.x + epsilon
    && left.y < right.y + right.height - epsilon
    && left.y + left.height > right.y + epsilon;
};

async function expectToolbarInsideViewport(page: Page): Promise<void> {
  const viewport = page.viewportSize();
  expect(viewport).not.toBeNull();
  const controls = toolbarControls(page);
  const boxes = await Promise.all(
    controls.map(async ({ label, locator }) => ({
      label,
      box: await measuredBox(locator, label),
    })),
  );

  for (const { label, box } of boxes) {
    expect(box.x, `${label} left edge`).toBeGreaterThanOrEqual(0);
    expect(box.y, `${label} top edge`).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width, `${label} right edge`)
      .toBeLessThanOrEqual(viewport!.width);
    expect(box.y + box.height, `${label} bottom edge`)
      .toBeLessThanOrEqual(viewport!.height);
    expect(box.height, `${label} explicit height`).toBeCloseTo(36, 0);
  }

  for (let leftIndex = 0; leftIndex < boxes.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < boxes.length; rightIndex += 1) {
      const left = boxes[leftIndex];
      const right = boxes[rightIndex];
      expect(
        boxesOverlap(left.box, right.box),
        `${left.label} must not overlap ${right.label}`,
      ).toBe(false);
      const sameRow = Math.abs(left.box.y - right.box.y) < 1;
      if (sameRow) {
        expect(
          left.box.y + left.box.height,
          `${left.label} and ${right.label} row bottom`,
        ).toBeCloseTo(right.box.y + right.box.height, 0);
      }
    }
  }
}

async function expectDialogInsideViewport(
  page: Page,
  includeError: boolean,
): Promise<void> {
  const viewport = page.viewportSize();
  expect(viewport).not.toBeNull();
  const dialog = page.getByRole('dialog');
  const measured = [
    { label: 'dialog', locator: dialog },
    { label: 'dialog heading', locator: dialog.getByRole('heading') },
    {
      label: 'dialog editor',
      locator: dialog.getByRole('textbox', { name: /Data source data/ }),
    },
    { label: 'dialog actions', locator: dialog.locator('.data-source-dialog-actions') },
    { label: 'Cancel', locator: dialog.getByRole('button', { name: 'Cancel' }) },
    { label: 'Confirm', locator: dialog.getByRole('button', { name: 'Confirm' }) },
  ];
  if (includeError) {
    measured.push({
      label: 'validation error',
      locator: dialog.getByText('Invalid inline JSON'),
    });
  }
  for (const { label, locator } of measured) {
    const box = await measuredBox(locator, label);
    expect(box.x, `${label} left edge`).toBeGreaterThanOrEqual(0);
    expect(box.y, `${label} top edge`).toBeGreaterThanOrEqual(0);
    expect(box.x + box.width, `${label} right edge`)
      .toBeLessThanOrEqual(viewport!.width);
    expect(box.y + box.height, `${label} bottom edge`)
      .toBeLessThanOrEqual(viewport!.height);
  }
  const dimensions = await dialog.evaluate((element) => ({
    scrollWidth: element.scrollWidth,
    clientWidth: element.clientWidth,
  }));
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth);
  const editorBox = await measuredBox(
    dialog.getByRole('textbox', { name: /Data source data/ }),
    'dialog editor',
  );
  expect(editorBox.height).toBeGreaterThanOrEqual(180);
}

async function expectFocusContained(page: Page): Promise<void> {
  await expect.poll(() => page.evaluate(() => {
    const dialog = document.querySelector('dialog[open]');
    return Boolean(dialog?.contains(document.activeElement));
  })).toBe(true);
}

test.describe('Data Source dialog and toolbar layout', () => {
  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 390, height: 844 },
  ]) {
    test(`keeps controls, modal drafts, and focus safe at ${viewport.width}x${viewport.height}`, async ({
      page,
    }) => {
      await page.setViewportSize(viewport);
      await page.goto('/');
      await expect(page.getByText('Build the flow')).toBeVisible();
      await expectToolbarInsideViewport(page);

      const opener = page.getByRole('button', { name: 'Edit data source sample' });
      const preview = page.getByLabel('Data 1 preview');
      await opener.scrollIntoViewIfNeeded();
      const committedBefore = await preview.textContent();

      await opener.focus();
      await opener.click();
      const dialog = page.getByRole('dialog', { name: 'Edit data source sample' });
      const editor = dialog.getByRole('textbox', {
        name: 'Data source data for sample',
      });
      await expect(editor).toBeFocused();
      await expectDialogInsideViewport(page, false);

      for (let cycle = 0; cycle < 10; cycle += 1) {
        await page.keyboard.press('Tab');
        await expectFocusContained(page);
      }
      for (let cycle = 0; cycle < 10; cycle += 1) {
        await page.keyboard.press('Shift+Tab');
        await expectFocusContained(page);
      }

      const discard = async (
        text: string,
        close: () => Promise<void>,
      ): Promise<void> => {
        await editor.fill(text);
        await close();
        await expect(dialog).toBeHidden();
        await expect(preview).toHaveText(committedBefore ?? '');
        await expect(opener).toBeFocused();
      };

      await discard('escape draft', async () => page.keyboard.press('Escape'));

      await opener.click();
      await discard('cancel draft', async () => {
        await dialog.getByRole('button', { name: 'Cancel' }).click();
      });

      await opener.click();
      await discard('close draft', async () => {
        await dialog.getByRole('button', { name: 'Close data source editor' }).click();
      });

      await opener.click();
      await editor.fill('backdrop draft');
      await page.mouse.click(2, 2);
      await expect(dialog).toBeHidden();
      await expect(preview).toHaveText(committedBefore ?? '');
      await expect(opener).toBeFocused();

      const validText = '[{"a":9,"b":1}]';
      await opener.click();
      await editor.fill(validText);
      await dialog.getByRole('button', { name: 'Confirm' }).click();
      await expect(dialog).toBeHidden();
      await expect(preview).toContainText(validText);
      await expect(opener).toBeFocused();

      await opener.click();
      await editor.fill('[{');
      await dialog.getByRole('button', { name: 'Confirm' }).click();
      await expect(dialog).toBeVisible();
      await expect(editor).toHaveValue('[{');
      await expect(editor).toHaveAttribute('aria-invalid', 'true');
      await expect(editor).toHaveAccessibleDescription('Invalid inline JSON');
      await expect(preview).toContainText(validText);
      await expectDialogInsideViewport(page, true);
      await dialog.getByRole('button', { name: 'Cancel' }).click();
      await expect(opener).toBeFocused();
    });
  }
});

test('builds and runs a persisted DataFusion UDF graph without browser code', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('Build the flow')).toBeVisible();
  await expect(page.getByText('double_value', { exact: true })).toBeVisible();

  await page.getByLabel('DataFusion expression').fill('doubled = double_value(a)');
  await page.locator('.udf-option input[type="checkbox"]').check({ force: true });
  await page.getByRole('button', { name: 'Validate' }).click({ force: true });
  await expect(page.getByText('Graph is valid')).toBeVisible();

  await page.getByRole('button', { name: /Run preview/ }).click({ force: true });
  await expect(page.getByText('completed')).toBeVisible({ timeout: 20_000 });
  await expect(page.getByRole('columnheader', { name: /doubled/ })).toBeVisible();
  await page.getByText('Physical plan').click({ force: true });
  await expect(page.getByText('ProjectionExec')).toBeVisible();
  await page.getByRole('button', { name: 'Inspect' }).click({ force: true });
  await expect(page.getByText('No stored checkpoint')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Benchmark comparison' })).toBeVisible();

  const projects = await page.request.get('http://127.0.0.1:8765/api/v3/projects');
  expect(projects.ok()).toBeTruthy();
  const [summary] = await projects.json();
  expect(summary.id).toMatch(/^project_[0-9a-f]{32}$/);
  const project = await page.request.get(
    `http://127.0.0.1:8765/api/v3/projects/${summary.id}`,
  );
  expect(project.ok()).toBeTruthy();
  const document = await project.json();
  expect(document.format_version).toBe(2);
  expect(document.pipeline.nodes[0].operator.udfs).toEqual([
    {
      provider: 'python',
      name: 'double_value',
      version: '1',
      kind: 'data_fusion_scalar',
    },
  ]);
  expect(JSON.stringify(document)).not.toContain('def double_value');

  const validation = await page.request.post(
    `http://127.0.0.1:8765/api/v3/projects/${summary.id}/validate`,
  );
  expect(validation.ok()).toBeTruthy();
  expect(await validation.json()).toMatchObject({ valid: true, issues: [] });

  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Export JSON' }).click({ force: true });
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toBe(`${summary.id}.json`);
  const downloadPath = await download.path();
  expect(downloadPath).not.toBeNull();

  page.once('dialog', (dialog) => dialog.accept());
  await page.getByLabel('Import project').setInputFiles({
    name: download.suggestedFilename(),
    mimeType: 'application/json',
    buffer: await readFile(downloadPath!),
  });
  await expect(page.getByText('Project imported')).toBeVisible();

  page.once('dialog', (dialog) => dialog.accept());
  await page.getByRole('button', { name: 'Delete', exact: true }).click({ force: true });
  await expect(page.getByText('Project deleted')).toBeVisible();
  const remaining = await page.request.get('http://127.0.0.1:8765/api/v3/projects');
  expect(await remaining.json()).toEqual([]);
});

test.describe('persisted two-source SQL join', () => {
  test.beforeEach(async ({ request }) => {
    await deleteTwoSourceProject(request);
  });

  test.afterEach(async ({ request }) => {
    await deleteTwoSourceProject(request);
  });

  test('edits and runs through two saved sources', async ({ page, request }) => {
    const created = await request.post(projectsUrl, { data: twoSourceProject });
    expect(created.status()).toBe(201);

    await page.goto('/');

    const project = page.getByLabel('Project', { exact: true });
    await expect(project).toBeVisible();
    await expect(project).toBeEnabled();
    await project.selectOption('two_source_e2e');
    await expect(project).toHaveValue('two_source_e2e');
    await expect(page.getByLabel('Project name')).toHaveValue('Two source E2E');

    const sources = page.getByRole('region', { name: 'Data sources' });
    await expect(sources.getByRole('article')).toHaveCount(2);
    await expect(page.getByLabel('Graph input 1')).toHaveValue('left_source');
    await expect(page.getByLabel('Graph input 2')).toHaveValue('right_source');

    const toolboxBefore = await panelWidth(page, '.toolbox');
    await dragSeparator(page, 'Resize Toolbox', 40);
    const toolboxWidth = await panelWidth(page, '.toolbox');
    expect(toolboxWidth).toBeGreaterThan(toolboxBefore + 30);

    const inspectorBefore = await panelWidth(page, '.inspector');
    await dragSeparator(page, 'Resize Inspector', -32);
    const inspectorWidth = await panelWidth(page, '.inspector');
    expect(inspectorWidth).toBeGreaterThan(inspectorBefore + 22);

    const addSql = page.getByRole('button', { name: /DataFusion SQL/i });
    await addSql.scrollIntoViewIfNeeded();
    await addSql.click();
    const firstAlias = page.getByRole('textbox', { name: 'Input alias 1', exact: true });
    await expect(firstAlias).toHaveValue('input');
    await page.getByRole('button', { name: 'Add input alias' }).click();
    const secondAlias = page.getByRole('textbox', { name: 'Input alias 2', exact: true });
    await expect(secondAlias).toHaveValue('input_2');
    await secondAlias.fill('right');
    await secondAlias.press('Enter');
    await expect(secondAlias).toHaveValue('right');
    await page.getByRole('button', { name: 'Delete node' }).click();

    await page.locator('.react-flow__node').filter({ hasText: 'join_result' }).click();
    await expect(page.getByRole('textbox', { name: 'Input alias 1', exact: true }))
      .toHaveValue('left');
    await expect(page.getByRole('textbox', { name: 'Input alias 2', exact: true }))
      .toHaveValue('right');

    const addSource = sources.getByRole('button', { name: 'Add data source' });
    await expect(addSource).toBeVisible();
    await expect(addSource).toBeEnabled();
    await addSource.scrollIntoViewIfNeeded();
    await addSource.click();
    await expect(sources.getByRole('article')).toHaveCount(3);
    const removeSource = sources.getByRole('button', { name: 'Remove source 3' });
    await expect(removeSource).toBeVisible();
    await expect(removeSource).toBeEnabled();
    await removeSource.scrollIntoViewIfNeeded();
    await removeSource.click();
    await expect(sources.getByRole('article')).toHaveCount(2);

    await page.getByRole('button', { name: 'Edit data source left' }).click();
    await page.getByRole('textbox', { name: 'Data source data for left' })
      .fill('[{"id":1,"value":4},{"id":2,"value":5}]');
    await page.getByRole('button', { name: 'Confirm' }).click();
    await expect(page.getByLabel('Data 1 preview'))
      .toContainText('[{"id":1,"value":4},{"id":2,"value":5}]');
    const save = page.getByRole('button', { name: 'Save' });
    await expect(save).toBeVisible();
    await expect(save).toBeEnabled();
    await save.click();
    await expect(page.getByRole('status')).toHaveText('Project saved');

    const validate = page.getByRole('button', { name: 'Validate' });
    await expect(validate).toBeVisible();
    await expect(validate).toBeEnabled();
    await validate.click();
    await expect(page.getByText('Graph is valid')).toBeVisible();

    const run = page.getByRole('button', { name: /Run preview/ });
    await expect(run).toBeVisible();
    await expect(run).toBeEnabled();
    await run.click();
    await expect(page.getByText('completed', { exact: true })).toBeVisible({ timeout: 20_000 });
    await expect(page.getByRole('columnheader', { name: /total/ })).toBeVisible();
    await expect(page.getByRole('cell', { name: '18', exact: true })).toBeVisible();

    const metricsBefore = await panelWidth(page, '.metrics-stack');
    await dragSeparator(page, 'Resize Metrics', -40);
    const metricsWidth = await panelWidth(page, '.metrics-stack');
    expect(metricsWidth).toBeGreaterThan(metricsBefore + 30);

    const saved = await request.get(twoSourceProjectUrl);
    expect(saved.ok()).toBeTruthy();
    const document = await saved.json();
    expect(document.data_sources.map((source: { input: string }) => source.input)).toEqual([
      'left_source',
      'right_source',
    ]);

    await page.reload();
    await expect(page.getByLabel('Project', { exact: true })).toHaveValue('two_source_e2e');
    await expect.poll(() => panelWidth(page, '.toolbox')).toBeCloseTo(toolboxWidth, 0);
    await expect.poll(() => panelWidth(page, '.inspector')).toBeCloseTo(inspectorWidth, 0);
    const storedLayout = await page.evaluate(() => JSON.parse(
      localStorage.getItem('calc-flow-studio:panel-layout:v1') ?? '{}',
    ));
    expect(storedLayout).toMatchObject({
      version: 1,
      toolbox: toolboxWidth,
      inspector: inspectorWidth,
      metrics: metricsWidth,
    });

    await page.getByRole('button', { name: /Run preview/ }).click();
    await expect(page.getByText('completed', { exact: true })).toBeVisible({ timeout: 20_000 });
    await expect.poll(() => panelWidth(page, '.metrics-stack')).toBeCloseTo(metricsWidth, 0);

    expect(await deleteTwoSourceProject(request)).toBe(204);
  });
});
