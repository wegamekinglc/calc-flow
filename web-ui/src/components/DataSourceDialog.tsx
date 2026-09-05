import {
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type MouseEvent,
} from 'react';

import type { DataSourceFormat } from './dataSourceEditorModel';

export interface DataSourceDialogProps {
  readonly format: DataSourceFormat;
  readonly initialText: string;
  readonly sourceLabel: string;
  readonly onConfirm: (dataText: string) => void;
  readonly onDismiss: () => void;
}

const FOCUSABLE_SELECTOR = [
  'button:not([disabled])',
  'textarea:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(',');

const focusableElements = (dialog: HTMLDialogElement): HTMLElement[] =>
  Array.from(dialog.querySelectorAll<HTMLElement>(FOCUSABLE_SELECTOR))
    .filter((element) => element.getAttribute('aria-hidden') !== 'true');

export function DataSourceDialog({
  format,
  initialText,
  sourceLabel,
  onConfirm,
  onDismiss,
}: DataSourceDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const editorRef = useRef<HTMLTextAreaElement>(null);
  const headingId = useId();
  const errorId = useId();
  const [dataText, setDataText] = useState(initialText);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (!dialog.open) {
      if (typeof dialog.showModal === 'function') {
        dialog.showModal();
      } else {
        dialog.setAttribute('open', '');
      }
    }
    editorRef.current?.focus();
    return () => {
      if (!dialog.open) return;
      if (typeof dialog.close === 'function') {
        dialog.close();
      } else {
        dialog.removeAttribute('open');
      }
    };
  }, []);

  const closeDialog = () => {
    const dialog = dialogRef.current;
    if (dialog?.open) {
      if (typeof dialog.close === 'function') {
        dialog.close();
      } else {
        dialog.removeAttribute('open');
      }
    }
  };

  const discard = () => {
    closeDialog();
    onDismiss();
  };

  const confirm = () => {
    if (format === 'inline_json') {
      try {
        JSON.parse(dataText);
      } catch {
        setError('Invalid inline JSON');
        return;
      }
    }
    closeDialog();
    onConfirm(dataText);
  };

  const handleBackdropClick = (event: MouseEvent<HTMLDialogElement>) => {
    if (event.target !== event.currentTarget) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const outside = event.clientX < bounds.left
      || event.clientX > bounds.right
      || event.clientY < bounds.top
      || event.clientY > bounds.bottom;
    if (outside) discard();
  };

  const containFocus = (event: KeyboardEvent<HTMLDialogElement>) => {
    if (event.key !== 'Tab') return;
    const dialog = dialogRef.current;
    if (!dialog) return;
    const focusable = focusableElements(dialog);
    if (!focusable.length) {
      event.preventDefault();
      return;
    }
    const first = focusable.at(0);
    const last = focusable.at(-1);
    if (first === undefined || last === undefined) {
      event.preventDefault();
      return;
    }
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    } else if (!dialog.contains(document.activeElement)) {
      event.preventDefault();
      (event.shiftKey ? last : first).focus();
    }
  };

  return (
    <dialog
      ref={dialogRef}
      className="data-source-dialog"
      aria-labelledby={headingId}
      aria-modal="true"
      onCancel={(event) => {
        event.preventDefault();
        discard();
      }}
      onClick={handleBackdropClick}
      onKeyDown={containFocus}
    >
      <div className="data-source-dialog-surface">
        <header className="data-source-dialog-header">
          <div>
            <span className="eyebrow">Data source</span>
            <h2 id={headingId}>Edit data source {sourceLabel}</h2>
          </div>
          <button
            className="icon-button"
            type="button"
            aria-label="Close data source editor"
            onClick={discard}
          >
            ×
          </button>
        </header>
        <div className="data-source-dialog-body">
          <label>
            Data
            <textarea
              ref={editorRef}
              className="data-source-dialog-editor"
              aria-label={`Data source data for ${sourceLabel}`}
              aria-invalid={Boolean(error)}
              aria-describedby={error ? errorId : undefined}
              value={dataText}
              onChange={(event) => {
                setDataText(event.target.value);
                setError(null);
              }}
            />
          </label>
          {error && (
            <p className="data-source-dialog-error" id={errorId}>
              {error}
            </p>
          )}
        </div>
        <footer className="data-source-dialog-actions">
          <button className="ghost-button" type="button" onClick={discard}>
            Cancel
          </button>
          <button className="run-button" type="button" onClick={confirm}>
            Confirm
          </button>
        </footer>
      </div>
    </dialog>
  );
}
