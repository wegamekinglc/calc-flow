import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { ProjectActions } from './ProjectActions';

describe('ProjectActions', () => {
  it('dispatches lifecycle actions and passes the selected import file', () => {
    const onNew = vi.fn();
    const onDelete = vi.fn();
    const onExport = vi.fn();
    const onImport = vi.fn();
    render(
      <ProjectActions
        persisted
        busy={false}
        onNew={onNew}
        onDelete={onDelete}
        onExport={onExport}
        onImport={onImport}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'New' }));
    fireEvent.click(screen.getByRole('button', { name: 'Export JSON' }));
    fireEvent.click(screen.getByRole('button', { name: 'Export YAML' }));
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    const file = new File(['name: Imported\n'], 'project.yaml', {
      type: 'application/yaml',
    });
    fireEvent.change(screen.getByLabelText('Import project'), {
      target: { files: [file] },
    });

    expect(onNew).toHaveBeenCalledOnce();
    expect(onExport).toHaveBeenNthCalledWith(1, 'json');
    expect(onExport).toHaveBeenNthCalledWith(2, 'yaml');
    expect(onDelete).toHaveBeenCalledOnce();
    expect(onImport).toHaveBeenCalledWith(file);
  });

  it('disables persisted actions for an unsaved draft', () => {
    render(
      <ProjectActions
        persisted={false}
        busy={false}
        onNew={vi.fn()}
        onDelete={vi.fn()}
        onExport={vi.fn()}
        onImport={vi.fn()}
      />,
    );

    expect(screen.getByRole('button', { name: 'Delete' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Export JSON' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Export YAML' })).toBeDisabled();
  });

  it('replaces the file input after importing without mutating the event target', () => {
    const onImport = vi.fn();
    render(
      <ProjectActions
        persisted
        busy={false}
        onNew={vi.fn()}
        onDelete={vi.fn()}
        onExport={vi.fn()}
        onImport={onImport}
      />,
    );
    const input = screen.getByLabelText('Import project');
    const file = new File(['{}'], 'project.json', { type: 'application/json' });

    fireEvent.change(input, { target: { files: [file] } });

    expect(onImport).toHaveBeenCalledWith(file);
    expect(screen.getByLabelText('Import project')).not.toBe(input);
  });
});
