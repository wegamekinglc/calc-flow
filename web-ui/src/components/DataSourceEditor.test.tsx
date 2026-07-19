import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import {
  DataSourceEditor,
  type DataSourceEditorProps,
} from './DataSourceEditor';

describe('DataSourceEditor', () => {
  it('dispatches changes to the addressed source card', () => {
    const props = {
      sources: [
        { id: 'left', input: 'left_source', format: 'inline_json', data: [] },
        { id: 'right', input: 'right_source', format: 'csv', data: '' },
      ],
      drafts: [
        { key: 'left-key', dataText: '[]', error: null },
        { key: 'right-key', dataText: 'value\n2\n', error: 'CSV problem' },
      ],
      busy: false,
      onAdd: vi.fn(),
      onRemove: vi.fn(),
      onFieldChange: vi.fn(),
      onDataChange: vi.fn(),
      onLoadFile: vi.fn(),
    } satisfies DataSourceEditorProps;

    render(<DataSourceEditor {...props} />);
    expect(screen.getByLabelText('Source ID 1')).toHaveValue('left');
    expect(screen.getByLabelText('Graph input 2')).toHaveValue('right_source');
    expect(screen.getByLabelText('Data 2')).toHaveAttribute('aria-invalid', 'true');
    fireEvent.change(screen.getByLabelText('Graph input 2'), {
      target: { value: 'prices' },
    });
    expect(props.onFieldChange).toHaveBeenCalledWith(1, 'input', 'prices');
    fireEvent.click(screen.getByRole('button', { name: 'Remove source 1' }));
    expect(props.onRemove).toHaveBeenCalledWith(0);
    fireEvent.click(screen.getByRole('button', { name: 'Add data source' }));
    expect(props.onAdd).toHaveBeenCalledOnce();
    const file = new File(['value\n3\n'], 'right.csv', { type: 'text/csv' });
    fireEvent.change(screen.getByLabelText('Load file 2'), {
      target: { files: [file] },
    });
    expect(props.onLoadFile).toHaveBeenCalledWith(1, file);
  });

  it('explains how to configure an empty source list', () => {
    render(
      <DataSourceEditor
        sources={[]}
        drafts={[]}
        busy={false}
        onAdd={vi.fn()}
        onRemove={vi.fn()}
        onFieldChange={vi.fn()}
        onDataChange={vi.fn()}
        onLoadFile={vi.fn()}
      />,
    );

    expect(screen.getByText('Add one data source for every external graph input.'))
      .toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Add data source' })).toBeEnabled();
  });
});
