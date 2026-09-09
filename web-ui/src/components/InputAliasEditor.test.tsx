import { fireEvent as fireDOMEvent, queryByLabelText, waitFor } from '@testing-library/dom';
import { fireEvent, render, screen } from '@testing-library/react';
import { createRoot } from 'react-dom/client';
import { describe, expect, it, vi } from 'vitest';

import { InputAliasEditor } from './InputAliasEditor';

describe('InputAliasEditor', () => {
  it('keeps an edit made as soon as the alias input appears', async () => {
    const environment = globalThis as typeof globalThis & {
      IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
    };
    const previousActEnvironment = environment.IS_REACT_ACT_ENVIRONMENT;
    // Preserve browser timing: RTL render flushes effects before the first edit.
    environment.IS_REACT_ACT_ENVIRONMENT = false;
    const container = document.createElement('div');
    document.body.append(container);
    const root = createRoot(container);
    let observer: MutationObserver | undefined;
    try {
      const edited = new Promise<void>((resolve) => {
        observer = new MutationObserver(() => {
          const input = queryByLabelText<HTMLInputElement>(container, 'Input alias 2');
          if (input === null) return;
          observer?.disconnect();
          fireDOMEvent.change(input, { target: { value: 'rhs' } });
          resolve();
        });
        observer.observe(container, { childList: true, subtree: true });
      });
      root.render(
        <InputAliasEditor
          aliases={['left', 'right']}
          onAdd={vi.fn()}
          onRename={vi.fn()}
          onRemove={vi.fn()}
        />,
      );
      await edited;
      await waitFor(() => expect(screen.getByLabelText('Input alias 2')).toHaveValue('rhs'));
    } finally {
      observer?.disconnect();
      root.unmount();
      container.remove();
      environment.IS_REACT_ACT_ENVIRONMENT = previousActEnvironment;
    }
  });

  it('adds and commits a second alias as an independent row', () => {
    const onAdd = vi.fn();
    const onRename = vi.fn();
    const onRemove = vi.fn();
    const { rerender } = render(
      <InputAliasEditor
        aliases={['left']}
        onAdd={onAdd}
        onRename={onRename}
        onRemove={onRemove}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Add input alias' }));
    expect(onAdd).toHaveBeenCalledOnce();

    rerender(
      <InputAliasEditor
        aliases={['left', 'input']}
        onAdd={onAdd}
        onRename={onRename}
        onRemove={onRemove}
      />,
    );
    const second = screen.getByLabelText('Input alias 2');
    fireEvent.change(second, { target: { value: 'right' } });
    fireEvent.keyDown(second, { key: 'Enter' });

    expect(onRename).toHaveBeenCalledWith('input', 'right');
  });

  it('keeps invalid drafts local and restores the saved alias on Escape', () => {
    const onRename = vi.fn();
    render(
      <InputAliasEditor
        aliases={['left', 'right']}
        onAdd={vi.fn()}
        onRename={onRename}
        onRemove={vi.fn()}
      />,
    );

    const second = screen.getByLabelText('Input alias 2');
    fireEvent.change(second, { target: { value: 'left' } });
    fireEvent.blur(second);

    expect(screen.getByText('Input aliases must be unique')).toBeInTheDocument();
    expect(second).toHaveAttribute('aria-invalid', 'true');
    expect(onRename).not.toHaveBeenCalled();

    fireEvent.keyDown(second, { key: 'Escape' });
    expect(second).toHaveValue('right');
    expect(screen.queryByText('Input aliases must be unique')).not.toBeInTheDocument();
  });

  it('rejects an empty alias without replacing the saved value', () => {
    const onRename = vi.fn();
    render(
      <InputAliasEditor
        aliases={['input']}
        onAdd={vi.fn()}
        onRename={onRename}
        onRemove={vi.fn()}
      />,
    );

    const alias = screen.getByLabelText('Input alias 1');
    fireEvent.change(alias, { target: { value: '   ' } });
    fireEvent.blur(alias);

    expect(screen.getByText('Input alias is required')).toBeInTheDocument();
    expect(onRename).not.toHaveBeenCalled();
  });

  it('removes the selected saved row without committing a dirty draft', () => {
    const onRename = vi.fn();
    const onRemove = vi.fn();
    render(
      <InputAliasEditor
        aliases={['left', 'right']}
        onAdd={vi.fn()}
        onRename={onRename}
        onRemove={onRemove}
      />,
    );

    fireEvent.change(screen.getByLabelText('Input alias 2'), {
      target: { value: 'temporary' },
    });
    const remove = screen.getByRole('button', { name: 'Remove input alias 2' });
    fireEvent.pointerDown(remove);
    fireEvent.click(remove);

    expect(onRemove).toHaveBeenCalledWith('right');
    expect(onRename).not.toHaveBeenCalled();
  });
});
