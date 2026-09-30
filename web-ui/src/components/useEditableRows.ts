import { useState } from 'react';

interface EditableRow<T> {
  readonly key: string;
  readonly value: T;
}

const rowFor = <T,>(value: T): EditableRow<T> => ({ key: crypto.randomUUID(), value });

const reconcileRows = <T,>(
  items: readonly T[],
  previous: readonly EditableRow<T>[],
): EditableRow<T>[] => {
  const used = new Set<string>();
  return items.map((value) => {
    const row = previous.find((candidate) => candidate.value === value && !used.has(candidate.key))
      ?? rowFor(value);
    used.add(row.key);
    return row;
  });
};

// Keep editable row identity in UI state, separate from project values.
export function useEditableRows<T extends object>(
  items: readonly T[],
  onChange: (items: T[]) => void,
  scope?: string,
) {
  const [previous, setPrevious] = useState(() => ({ items, scope, rows: items.map(rowFor) }));
  let rows = previous.rows;
  if (previous.items !== items || previous.scope !== scope) {
    rows = reconcileRows(items, previous.scope === scope ? previous.rows : []);
    setPrevious({ items, scope, rows });
  }

  const commit = (nextRows: EditableRow<T>[]) => {
    const nextItems = nextRows.map((row) => row.value);
    setPrevious({ items: nextItems, scope, rows: nextRows });
    onChange(nextItems);
  };

  return {
    rows,
    update: (index: number, change: Partial<T>) => commit(rows.map((row, current) =>
      current === index ? { ...row, value: { ...row.value, ...change } } : row)),
    append: (value: T) => commit([...rows, rowFor(value)]),
    remove: (index: number) => commit(rows.filter((_, current) => current !== index)),
  };
}
