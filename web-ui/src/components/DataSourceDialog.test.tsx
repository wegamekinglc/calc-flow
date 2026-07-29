import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { DataSourceDialog, type DataSourceDialogProps } from './DataSourceDialog';

const renderDialog = (
  overrides: Partial<DataSourceDialogProps> = {},
) => {
  const props = {
    format: 'inline_json',
    initialText: '[]',
    sourceLabel: 'sample',
    onConfirm: vi.fn(),
    onDismiss: vi.fn(),
    ...overrides,
  } satisfies DataSourceDialogProps;
  render(<DataSourceDialog {...props} />);
  return props;
};

describe('DataSourceDialog', () => {
  it('focuses the local editor and confirms valid inline JSON once', async () => {
    const props = renderDialog();
    const editor = screen.getByRole('textbox', {
      name: 'Data source data for sample',
    });

    await waitFor(() => expect(editor).toHaveFocus());
    fireEvent.change(editor, { target: { value: '[{"value":2}]' } });

    expect(props.onConfirm).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: 'Confirm' }));

    expect(props.onConfirm).toHaveBeenCalledOnce();
    expect(props.onConfirm).toHaveBeenCalledWith('[{"value":2}]');
    expect(props.onDismiss).not.toHaveBeenCalled();
  });

  it('keeps invalid inline JSON local and associates the error with the editor', () => {
    const props = renderDialog();
    const editor = screen.getByRole('textbox', {
      name: 'Data source data for sample',
    });
    fireEvent.change(editor, { target: { value: '[{' } });
    fireEvent.click(screen.getByRole('button', { name: 'Confirm' }));

    expect(props.onConfirm).not.toHaveBeenCalled();
    expect(props.onDismiss).not.toHaveBeenCalled();
    expect(editor).toHaveValue('[{');
    expect(editor).toHaveAttribute('aria-invalid', 'true');
    expect(editor).toHaveAccessibleDescription('Invalid inline JSON');
    expect(screen.getByRole('dialog', { name: 'Edit data source sample' }))
      .toBeInTheDocument();
  });

  it.each(['json', 'csv', 'arrow_ipc'] as const)(
    'keeps %s text opaque on confirmation',
    (format) => {
      const props = renderDialog({ format });
      const editor = screen.getByRole('textbox', {
        name: 'Data source data for sample',
      });
      fireEvent.change(editor, { target: { value: 'not parsed {{' } });
      fireEvent.click(screen.getByRole('button', { name: 'Confirm' }));

      expect(props.onConfirm).toHaveBeenCalledWith('not parsed {{');
      expect(screen.queryByText('Invalid inline JSON')).not.toBeInTheDocument();
    },
  );

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
        const dialog = screen.getByRole('dialog');
        fireEvent(dialog, new Event('cancel', { cancelable: true }));
      },
    },
  ])('routes $path through the shared discard callback', ({ dismiss }) => {
    const props = renderDialog();
    const editor = screen.getByRole('textbox');
    fireEvent.change(editor, { target: { value: 'discard me' } });

    dismiss();

    expect(props.onDismiss).toHaveBeenCalledOnce();
    expect(props.onConfirm).not.toHaveBeenCalled();
  });

  it('dismisses only a real backdrop coordinate', () => {
    const props = renderDialog();
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

    fireEvent.click(dialog, { clientX: 200, clientY: 200 });
    expect(props.onDismiss).not.toHaveBeenCalled();

    fireEvent.click(dialog, { clientX: 50, clientY: 50 });
    expect(props.onDismiss).toHaveBeenCalledOnce();
    expect(props.onConfirm).not.toHaveBeenCalled();
  });

  it('contains forward and reverse sequential focus inside the dialog', () => {
    renderDialog();
    const close = screen.getByRole('button', { name: 'Close data source editor' });
    const confirm = screen.getByRole('button', { name: 'Confirm' });

    confirm.focus();
    fireEvent.keyDown(confirm, { key: 'Tab' });
    expect(close).toHaveFocus();

    fireEvent.keyDown(close, { key: 'Tab', shiftKey: true });
    expect(confirm).toHaveFocus();
  });
});
