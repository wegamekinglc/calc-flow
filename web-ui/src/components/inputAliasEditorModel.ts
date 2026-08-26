import type { EditableProject, NodeConfig, PortConfig } from '../types';

export type SqlInputAliasEdit =
  | { type: 'add' }
  | { type: 'rename'; alias: string; nextAlias: string }
  | { type: 'remove'; alias: string };

export const nextInputAlias = (aliases: readonly string[]): string => {
  if (!aliases.includes('input')) return 'input';
  let index = 2;
  while (aliases.includes(`input_${index}`)) index += 1;
  return `input_${index}`;
};

export const validateInputAlias = (
  draft: string,
  current: string,
  aliases: readonly string[],
): string | null => {
  const alias = draft.trim();
  if (!alias) return 'Input alias is required';
  if (alias !== current && aliases.includes(alias)) {
    return 'Input aliases must be unique';
  }
  return null;
};

const addExplicitPort = (
  ports: readonly PortConfig[],
  alias: string,
): PortConfig[] => ports.length === 0
  ? []
  : [
      ...ports,
      { name: alias, kind: 'table', required: true, schema: [] },
    ];

export const editSqlInputAliases = (
  project: EditableProject,
  nodeId: string,
  edit: SqlInputAliasEdit,
): EditableProject => {
  const current = project.graph.nodes.find((node) => node.id === nodeId);
  if (!current || current.operator.kind !== 'sql') return project;

  const aliases = current.operator.aliases;
  let nextAliases: string[];
  let nextPorts: PortConfig[];

  if (edit.type === 'add') {
    const alias = nextInputAlias(aliases);
    nextAliases = [...aliases, alias];
    nextPorts = addExplicitPort(current.input_ports, alias);
  } else if (edit.type === 'rename') {
    const alias = edit.nextAlias.trim();
    if (
      !aliases.includes(edit.alias)
      || alias === edit.alias
      || validateInputAlias(alias, edit.alias, aliases) !== null
    ) return project;
    nextAliases = aliases.map((currentAlias) =>
      currentAlias === edit.alias ? alias : currentAlias,
    );
    nextPorts = current.input_ports.map((port) =>
      port.name === edit.alias ? { ...port, name: alias } : port,
    );
  } else {
    if (!aliases.includes(edit.alias)) return project;
    nextAliases = aliases.filter((alias) => alias !== edit.alias);
    nextPorts = current.input_ports.filter((port) => port.name !== edit.alias);
  }

  const node = {
    ...current,
    operator: { ...current.operator, aliases: nextAliases },
    input_ports: nextPorts,
  } satisfies NodeConfig;

  const edges = edit.type === 'remove'
    ? project.graph.edges.filter(
        (edge) => edge.target_node !== nodeId || edge.target_port !== edit.alias,
      )
    : edit.type === 'rename'
      ? project.graph.edges.map((edge) =>
          edge.target_node === nodeId && edge.target_port === edit.alias
            ? { ...edge, target_port: edit.nextAlias.trim() }
            : edge,
        )
      : project.graph.edges;

  return {
    ...project,
    graph: {
      ...project.graph,
      nodes: project.graph.nodes.map((candidate) =>
        candidate.id === nodeId ? node : candidate,
      ),
      edges,
    },
  };
};
