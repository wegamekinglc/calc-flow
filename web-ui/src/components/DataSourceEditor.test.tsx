import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { at } from '../types';
import {
  DataSourceEditor,
  type DataSourceEditorProps,
} from './DataSourceEditor';

describe('DataSourceEditor', () => {
  it('dispatches fields, removal, addition, and files to the addressed source card', () => {
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
      pendingSourceKeys: new Set<string>(),
      onAdd: vi.fn(),
      onRemove: vi.fn(),
      onFieldChange: vi.fn(),
      onDataChange: vi.fn(),
      onLoadFile: vi.fn(),
    } satisfies DataSourceEditorProps;

    render(<DataSourceEditor {...props} />);
    expect(screen.getByLabelText('Source ID 1')).toHaveValue('left');
    expect(screen.getByLabelText('Graph input 2')).toHaveValue('right_source');
    expect(screen.getByLabelText('Data 2 preview')).toHaveTextContent('value 2');
    expect(screen.getByRole('button', { name: 'Edit data source right' }))
      .toHaveAttribute('aria-invalid', 'true');
    expect(screen.getByRole('button', { name: 'Edit data source right' }))
      .toHaveAccessibleDescription('CSV problem');
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

  it('keeps modal typing local and commits once to the current stable-key source', async () => {
    const onDataChange = vi.fn();
    const initialProps = {
      sources: [
        { id: 'left', input: 'left_source', format: 'inline_json', data: [] },
        { id: 'right', input: 'right_source', format: 'csv', data: '' },
      ],
      drafts: [
        { key: 'left-key', dataText: '[]', error: null },
        { key: 'right-key', dataText: 'value\n2\n', error: null },
      ],
      busy: false,
      pendingSourceKeys: new Set<string>(),
      onAdd: vi.fn(),
      onRemove: vi.fn(),
      onFieldChange: vi.fn(),
      onDataChange,
      onLoadFile: vi.fn(),
    } satisfies DataSourceEditorProps;
    const view = render(<DataSourceEditor {...initialProps} />);
    const opener = screen.getByRole('button', { name: 'Edit data source left' });

    opener.focus();
    fireEvent.click(opener);
    const editor = screen.getByRole('textbox', {
      name: 'Data source data for left',
    });
    expect(editor).toHaveValue('[]');
    fireEvent.change(editor, { target: { value: '[{"value":7}]' } });

    expect(onDataChange).not.toHaveBeenCalled();
    expect(screen.getByLabelText('Data 1 preview')).toHaveTextContent('[]');

    fireEvent.click(screen.getByRole('button', { name: 'Confirm' }));

    expect(onDataChange).toHaveBeenCalledOnce();
    expect(onDataChange).toHaveBeenCalledWith(0, '[{"value":7}]');
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    await waitFor(() => expect(opener).toHaveFocus());

    view.rerender(
      <DataSourceEditor
        {...initialProps}
        drafts={[
          { key: 'left-key', dataText: '[{"value":7}]', error: null },
          at(initialProps.drafts, 1),
        ]}
      />,
    );
    expect(screen.getByLabelText('Data 1 preview'))
      .toHaveTextContent('[{"value":7}]');
  });

  it.each([
    {
      path: 'Cancel',
      dismiss: () => {
        fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
      },
    },
    {
      path: 'close button',
      dismiss: () => {
        fireEvent.click(screen.getByRole('button', { name: 'Close data source editor' }));
      },
    },
    {
      path: 'Escape',
      dismiss: () => {
        fireEvent(
          screen.getByRole('dialog'),
          new Event('cancel', { cancelable: true }),
        );
      },
    },
    {
      path: 'backdrop',
      dismiss: () => {
        const dialog = screen.getByRole('dialog');
        vi.spyOn(dialog, 'getBoundingClientRect').mockReturnValue({
          x: 100,
          y: 100,
          left: 100,
          top: 100,
          right: 700,
          bottom: 700,
          width: 600,
          height: 600,
          toJSON: () => ({}),
        });
        fireEvent.click(dialog, { clientX: 50, clientY: 50 });
      },
    },
  ])(
    'discards text and validation through $path, then reloads committed text',
    async ({ dismiss }) => {
      const onDataChange = vi.fn();
      render(
        <DataSourceEditor
          sources={[
            { id: 'left', input: 'left_source', format: 'inline_json', data: [] },
          ]}
          drafts={[{ key: 'left-key', dataText: '[]', error: null }]}
          busy={false}
          pendingSourceKeys={new Set()}
          onAdd={vi.fn()}
          onRemove={vi.fn()}
          onFieldChange={vi.fn()}
          onDataChange={onDataChange}
          onLoadFile={vi.fn()}
        />,
      );
      const opener = screen.getByRole('button', { name: 'Edit data source left' });
      opener.focus();
      fireEvent.click(opener);
      const editor = screen.getByRole('textbox', {
        name: 'Data source data for left',
      });
      fireEvent.change(editor, { target: { value: '[{' } });
      fireEvent.click(screen.getByRole('button', { name: 'Confirm' }));
      expect(screen.getByText('Invalid inline JSON')).toBeInTheDocument();

      dismiss();

      expect(onDataChange).not.toHaveBeenCalled();
      expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
      await waitFor(() => expect(opener).toHaveFocus());

      fireEvent.click(opener);
      expect(screen.getByRole('textbox', {
        name: 'Data source data for left',
      })).toHaveValue('[]');
      expect(screen.queryByText('Invalid inline JSON')).not.toBeInTheDocument();
    },
  );

  it('excludes only the matching source from a pending file-read owner', () => {
    const props = {
      sources: [
        { id: 'left', input: 'left_source', format: 'inline_json', data: [] },
        { id: 'right', input: 'right_source', format: 'csv', data: '' },
      ],
      drafts: [
        { key: 'left-key', dataText: '[]', error: null },
        { key: 'right-key', dataText: 'value\n2\n', error: null },
      ],
      busy: false,
      pendingSourceKeys: new Set(['left-key']),
      onAdd: vi.fn(),
      onRemove: vi.fn(),
      onFieldChange: vi.fn(),
      onDataChange: vi.fn(),
      onLoadFile: vi.fn(),
    } satisfies DataSourceEditorProps;
    const view = render(<DataSourceEditor {...props} />);
    const leftOpener = screen.getByRole('button', {
      name: 'Edit data source left',
    });
    const rightOpener = screen.getByRole('button', {
      name: 'Edit data source right',
    });

    expect(leftOpener).toBeDisabled();
    expect(rightOpener).toBeEnabled();
    expect(screen.getByLabelText('Load file 1')).toBeEnabled();
    expect(screen.getByLabelText('Load file 2')).toBeEnabled();

    view.rerender(
      <DataSourceEditor {...props} pendingSourceKeys={new Set()} />,
    );
    fireEvent.click(rightOpener);

    expect(screen.getByLabelText('Load file 1')).toBeEnabled();
    expect(screen.getByLabelText('Load file 2')).toBeDisabled();
    expect(screen.getByRole('textbox', { name: 'Data source data for right' }))
      .toBeEnabled();
  });

  it('bounds the committed card preview without hiding its editor opener', () => {
    const committedText = 'x'.repeat(300);
    render(
      <DataSourceEditor
        sources={[
          { id: 'left', input: 'left_source', format: 'csv', data: committedText },
        ]}
        drafts={[{ key: 'left-key', dataText: committedText, error: null }]}
        busy={false}
        pendingSourceKeys={new Set()}
        onAdd={vi.fn()}
        onRemove={vi.fn()}
        onFieldChange={vi.fn()}
        onDataChange={vi.fn()}
        onLoadFile={vi.fn()}
      />,
    );

    const preview = screen.getByLabelText('Data 1 preview').querySelector('pre');
    expect(preview).toHaveTextContent(`${'x'.repeat(240)}…`);
    expect(preview).not.toHaveTextContent('x'.repeat(241));
    expect(screen.getByRole('button', { name: 'Edit data source left' }))
      .toBeEnabled();
  });

  it('explains how to configure an empty source list', () => {
    render(
      <DataSourceEditor
        sources={[]}
        drafts={[]}
        busy={false}
        pendingSourceKeys={new Set()}
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
