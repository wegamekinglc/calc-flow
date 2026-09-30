import { fireEvent, render, screen, within } from '@testing-library/react';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';

import {
  at,
  blankProject,
  type ArrowFieldConfig,
  type ConnectorCapability,
  type EditableProject,
  type ProjectSinkBinding,
  type ProjectSourceBinding,
} from '../types';
import { SchemaEditor } from './SchemaEditor';
import { StreamConfigEditor } from './StreamConfigEditor';

const connectors: ConnectorCapability[] = [{
  provider: 'builtin', name: 'file', version: '1', kind: 'both',
  capabilities: {
    delivery: 'exactly_once', replay: 'replayable_exact', watermark: 'generated_only',
    transaction: 'pre_commit_commit', snapshot: true, polling: true, cdc: false, lookup: false,
  },
  formats: ['json'], optionsSchema: {},
}];

function SchemaHarness({ initialFields = [
  { name: 'price', data_type: 'float64', nullable: true },
] }: { initialFields?: ArrowFieldConfig[] }) {
  const [fields, setFields] = useState(initialFields);
  return <SchemaEditor fields={fields} arrowTypes={['float64', 'int64']} onChange={setFields} />;
}

const streamProject = (): EditableProject => ({
  ...blankProject(),
  runtime: {
    mode: 'stream',
    options: {
      checkpoint_interval_ms: 30_000,
      max_batch_rows: 10_000,
      max_batch_bytes: 64 * 1024 * 1024,
    },
  },
  data_sources: [],
  sources: [{
    binding: 'input',
    connector: { provider: 'builtin', name: 'file', version: '1' },
    format: null,
    options: { path: 'input.json', format: 'json' },
    secrets: {},
    watermark: { policy: 'disabled' },
    schema: [],
  }],
  sinks: [{
    binding: 'output',
    connector: { provider: 'builtin', name: 'file', version: '1' },
    delivery: 'at_least_once',
    format: null,
    options: { path: 'results' },
    secrets: {},
  }],
});

function StreamHarness({ initialProject = streamProject() }: { initialProject?: EditableProject }) {
  const [project, setProject] = useState(initialProject);
  return <>
    <StreamConfigEditor project={project} connectors={connectors} onChange={setProject} />
    <output data-testid="project">{JSON.stringify(project)}</output>
  </>;
}

const sourceOptions = () => {
  const card = screen.getByLabelText('Graph input').closest<HTMLElement>('article');
  if (!card) throw new Error('Expected a source binding card');
  return within(card).getByLabelText(/^Options/);
};

describe('editable row identity', () => {
  it('keeps focus after editing the schema field name', () => {
    render(<SchemaHarness />);
    const input = screen.getByLabelText('Field name');
    input.focus();
    for (const value of ['price_x', 'price_xy', 'price_xyz']) {
      fireEvent.change(input, { target: { value } });
      expect(screen.getByLabelText('Field name')).toHaveValue(value);
      expect(screen.getByLabelText('Field name')).toHaveFocus();
    }
  });

  it.each(['Graph input', 'Graph output'])('keeps focus after editing %s', (label) => {
    render(<StreamHarness />);
    const input = screen.getByLabelText(label);
    input.focus();
    for (const value of ['renamed', 'renamed_x', 'renamed_xy']) {
      fireEvent.change(input, { target: { value } });
      expect(screen.getByLabelText(label)).toHaveValue(value);
      expect(screen.getByLabelText(label)).toHaveFocus();
    }
    const project = JSON.parse(screen.getByTestId('project').textContent) as EditableProject;
    const original = streamProject();
    expect(label === 'Graph input' ? project.sources : project.sinks).toEqual(
      label === 'Graph input'
        ? [{ ...at(original.sources), binding: 'renamed_xy' }]
        : [{ ...at(original.sinks), binding: 'renamed_xy' }],
    );
  });

  it('preserves an invalid options draft while renaming its source binding', () => {
    render(<StreamHarness />);
    const options = sourceOptions();
    fireEvent.change(options, { target: { value: '{unfinished' } });
    expect(options).toHaveValue('{unfinished');
    fireEvent.change(screen.getByLabelText('Graph input'), {
      target: { value: 'renamed' },
    });

    expect(sourceOptions()).toHaveValue('{unfinished');
  });

  it('control: keeps focus when editing a stream numeric setting', () => {
    render(<StreamHarness />);
    const input = screen.getByLabelText('Checkpoint interval (ms)');
    input.focus();
    fireEvent.change(input, { target: { value: '5000' } });

    expect(screen.getByLabelText('Checkpoint interval (ms)')).toHaveValue(5000);
    expect(screen.getByLabelText('Checkpoint interval (ms)')).toHaveFocus();
  });

  it('keeps a schema row mounted after adding and removing an adjacent field', () => {
    render(<SchemaHarness initialFields={[
      { name: 'first', data_type: 'float64', nullable: true },
      { name: 'second', data_type: 'float64', nullable: true },
    ]} />);
    const input = at(screen.getAllByLabelText('Field name'), 1);
    input.focus();
    fireEvent.click(screen.getByRole('button', { name: '+ field' }));
    fireEvent.click(screen.getByRole('button', { name: 'Remove first' }));
    expect(at(screen.getAllByLabelText('Field name'))).toBe(input);
    fireEvent.change(input, { target: { value: 'second_renamed' } });
    expect(input).toHaveFocus();
    expect(input).toHaveValue('second_renamed');
  });

  it.each([
    { kind: 'source', field: 'sources' as const, label: 'Graph input', addIndex: 0 },
    { kind: 'sink', field: 'sinks' as const, label: 'Graph output', addIndex: 1 },
  ])('preserves the second $kind draft after adding and removing the first row', ({ kind, field, label, addIndex }) => {
    const initialProject = streamProject();
    const original = at<ProjectSourceBinding | ProjectSinkBinding>(initialProject[field]);
    const initial = {
      ...initialProject,
      [field]: [...initialProject[field], { ...original, binding: 'second' }],
    };
    const { container } = render(<StreamHarness initialProject={initial} />);
    const input = at(screen.getAllByLabelText(label), 1);
    const card = input.closest<HTMLElement>('article');
    if (!card) throw new Error('Expected a binding card');
    fireEvent.change(within(card).getByLabelText(/^Options/), {
      target: { value: '{second_draft' },
    });
    const section = container.querySelector<HTMLElement>('.stream-config');
    if (!section) throw new Error('Expected stream settings');
    fireEvent.click(at(screen.getAllByRole('button', { name: `Remove ${kind}` })));
    expect(at(screen.getAllByLabelText(label))).toBe(input);
    fireEvent.click(at(within(section).getAllByRole('button', { name: 'Add' }), addIndex));
    input.focus();
    fireEvent.change(input, { target: { value: 'second_renamed' } });
    expect(input).toHaveFocus();
    expect(within(card).getByLabelText(/^Options/)).toHaveValue('{second_draft');
    const newInput = at(screen.getAllByLabelText(label), 1);
    const newCard = newInput.closest('article');
    if (!newCard) throw new Error('Expected the newly added binding card');
    expect(within(newCard).getByLabelText('Options')).not.toHaveValue('{second_draft');
    const project = JSON.parse(screen.getByTestId('project').textContent) as EditableProject;
    expect(at<ProjectSourceBinding | ProjectSinkBinding>(project[field])).toEqual({
      ...original,
      binding: 'second_renamed',
    });
  });

  it('discards local source drafts when switching projects with the same binding', () => {
    const project = streamProject();
    const onChange = vi.fn();
    const { rerender } = render(<StreamConfigEditor project={project} connectors={connectors} onChange={onChange} />);
    fireEvent.change(sourceOptions(), {
      target: { value: '{other_project_draft' },
    });
    rerender(<StreamConfigEditor project={{ ...project, id: 'other-project' }} connectors={connectors} onChange={onChange} />);
    expect(sourceOptions()).toHaveValue(JSON.stringify(at(project.sources).options, null, 2));
  });

  it('uses replacement source data from external props', () => {
    const project = streamProject();
    const onChange = vi.fn();
    const { rerender } = render(<StreamConfigEditor project={project} connectors={connectors} onChange={onChange} />);
    fireEvent.change(sourceOptions(), {
      target: { value: '{local_draft' },
    });
    const replacement = { ...at(project.sources), options: { path: 'reloaded.json' } };
    rerender(<StreamConfigEditor project={{ ...project, sources: [replacement] }} connectors={connectors} onChange={onChange} />);
    expect(sourceOptions()).toHaveValue(JSON.stringify(replacement.options, null, 2));
  });
});
