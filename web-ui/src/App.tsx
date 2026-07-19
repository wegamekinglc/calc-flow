import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type NodeChange,
  type NodeTypes,
} from '@xyflow/react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import { api } from './api/client';
import {
  CalculationNode,
  type CalculationFlowNode,
  type FlowNodeData,
} from './components/CalculationNode';
import { BenchmarkComparison } from './components/BenchmarkComparison';
import { CheckpointControl } from './components/CheckpointControl';
import { DataSourceEditor } from './components/DataSourceEditor';
import { NodeInspector } from './components/NodeInspector';
import { ProjectActions } from './components/ProjectActions';
import { ResultsPanel } from './components/ResultsPanel';
import {
  createDataSourceDrafts,
  materializeDataSources,
  nextDataSource,
  type DataSourceDraft,
  type DataSourceFormat,
} from './components/dataSourceEditor';
import { useRunEvents } from './hooks/useRunEvents';
import {
  blankProject,
  type CatalogResponse,
  type CheckpointSummary,
  type EditableProject,
  type NodeConfig,
  type ProjectDocument,
  type ProjectSummary,
  type RunResponse,
  type ValidationReport,
} from './types';

type FlowNode = CalculationFlowNode;

const nodeTypes: NodeTypes = { calculation: CalculationNode };
export const ARROW_TYPES = [
  'bool',
  'date32',
  'date64',
  'float32',
  'float64',
  'int8',
  'int16',
  'int32',
  'int64',
  'large_string',
  'string',
  'time32[s]',
  'time64[us]',
  'timestamp[ms]',
  'timestamp[us]',
  'uint8',
  'uint16',
  'uint32',
  'uint64',
] as const;
type EditableNodeKind = Extract<NodeConfig['operator']['kind'], 'expression' | 'sql'>;
type ProjectUpdate = EditableProject | ((current: EditableProject) => EditableProject);

const nodeColor = (node: FlowNode) => {
  if (node.data.kind === 'sql') return '#ef9456';
  if (node.data.kind === 'external') return '#a994ff';
  return '#56d5b2';
};

const nextId = (
  kind: EditableNodeKind,
  nodes: readonly NodeConfig[],
  index = nodes.length + 1,
): string => {
  const candidate = `${kind}_${index}`;
  return nodes.some((node) => node.id === candidate)
    ? nextId(kind, nodes, index + 1)
    : candidate;
};

const makeNode = (
  kind: EditableNodeKind,
  nodes: readonly NodeConfig[],
): NodeConfig => {
  const id = nextId(kind, nodes);
  const shared = {
    id,
    input_ports: [],
    output_ports: [],
    position: { x: 120 + nodes.length * 32, y: 100 + nodes.length * 36 },
  };
  if (kind === 'sql') {
    return {
      ...shared,
      operator: {
        kind: 'sql',
        query: 'SELECT * FROM input',
        aliases: ['input'],
        udfs: [],
      },
    };
  }
  return {
    ...shared,
    operator: {
      kind: 'expression',
      expression: 'result = value + 1',
      select: [],
      filter: null,
      udfs: [],
    },
  };
};

export const flowNodeData = (node: NodeConfig): FlowNodeData => ({
  label: node.id,
  kind: node.operator.kind,
  inputPorts: node.input_ports.length
    ? node.input_ports.map((port) => port.name)
    : node.operator.kind === 'sql'
      ? node.operator.aliases
      : node.operator.kind === 'expression'
        ? ['input']
        : [],
  outputPorts: node.output_ports.length
    ? node.output_ports.map((port) => port.name)
    : node.operator.kind === 'external'
      ? []
      : ['output'],
});

export const connectProject = (
  project: EditableProject,
  connection: Connection,
): EditableProject => {
  if (!connection.source || !connection.target) return project;
  const edge = {
    source_node: connection.source,
    target_node: connection.target,
    source_port: connection.sourceHandle ?? 'output',
    target_port: connection.targetHandle ?? 'input',
  };
  const duplicate = project.pipeline.edges.some(
    (current) =>
      current.source_node === edge.source_node
      && current.target_node === edge.target_node
      && current.source_port === edge.source_port
      && current.target_port === edge.target_port,
  );
  if (duplicate) return project;
  return {
    ...project,
    pipeline: {
      ...project.pipeline,
      edges: [...project.pipeline.edges, edge],
    },
  };
};

const fileToBase64 = async (file: File): Promise<string> => {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = '';
  for (let offset = 0; offset < bytes.length; offset += 8192) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
  }
  return btoa(binary);
};

export default function App() {
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [project, setProject] = useState<EditableProject>(() => blankProject());
  const [sourceDrafts, setSourceDrafts] = useState<DataSourceDraft[]>(() =>
    createDataSourceDrafts(project.data_sources),
  );
  const [persisted, setPersisted] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string>('calculate');
  const [validation, setValidation] = useState<ValidationReport | null>(null);
  const [run, setRun] = useState<RunResponse | null>(null);
  const [checkpoint, setCheckpoint] = useState<CheckpointSummary | null>(null);
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);

  const refreshProjects = useCallback(async () => {
    const items = await api.projects();
    setProjects(items);
    return items;
  }, []);

  const replaceEditableProject = useCallback(
    (next: EditableProject, isPersisted: boolean) => {
      setProject(next);
      setSourceDrafts(createDataSourceDrafts(next.data_sources));
      setPersisted(isPersisted);
      setSelectedNodeId(next.pipeline.nodes[0]?.id ?? '');
      setValidation(null);
      setRun(null);
      setCheckpoint(null);
    },
    [],
  );

  useEffect(() => {
    const controller = new AbortController();
    const initialize = async () => {
      try {
        const [loadedCatalog, loadedProjects] = await Promise.all([
          api.catalog(),
          api.projects(),
        ]);
        if (controller.signal.aborted) return;
        setCatalog(loadedCatalog);
        setProjects(loadedProjects);
        if (loadedProjects.length) {
          const loaded = await api.project(loadedProjects[0].id);
          if (controller.signal.aborted) return;
          replaceEditableProject(loaded, true);
        }
      } catch (error) {
        if (!controller.signal.aborted) setMessage((error as Error).message);
      }
    };
    void initialize();
    return () => controller.abort();
  }, [replaceEditableProject]);

  const updateProject = useCallback((update: ProjectUpdate) => {
    setProject((current) =>
      typeof update === 'function' ? update(current) : update,
    );
    setValidation(null);
    setRun(null);
    setCheckpoint(null);
  }, []);

  useRunEvents(run?.id ?? null, setRun);

  const flowNodes = useMemo<FlowNode[]>(
    () =>
      project.pipeline.nodes.map((node) => ({
        id: node.id,
        type: 'calculation',
        position: node.position ?? { x: 0, y: 0 },
        data: flowNodeData(node),
        className: `flow-node ${node.operator.kind}`,
        selected: selectedNodeId === node.id,
      })),
    [project.pipeline.nodes, selectedNodeId],
  );

  const flowEdges = useMemo<Edge[]>(
    () =>
      project.pipeline.edges.map((edge, index) => ({
        id: `${edge.source_node}:${edge.source_port}-${edge.target_node}:${edge.target_port}-${index}`,
        source: edge.source_node,
        target: edge.target_node,
        sourceHandle: edge.source_port,
        targetHandle: edge.target_port,
        data: { sourcePort: edge.source_port, targetPort: edge.target_port },
        animated: true,
      })),
    [project.pipeline.edges],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange<FlowNode>[]) => {
      const positionChanges = changes.filter((change) => change.type === 'position');
      if (!positionChanges.length) return;
      const changed = applyNodeChanges(positionChanges, flowNodes);
      const positions = new Map(changed.map((node) => [node.id, node.position]));
      updateProject((current) => ({
        ...current,
        pipeline: {
          ...current.pipeline,
          nodes: current.pipeline.nodes.map((node) => ({
            ...node,
            position: positions.get(node.id) ?? node.position,
          })),
        },
      }));
    },
    [flowNodes, updateProject],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange<Edge>[]) => {
      const structuralChanges = changes.filter((change) => change.type === 'remove');
      if (!structuralChanges.length) return;
      const changed = applyEdgeChanges(structuralChanges, flowEdges);
      updateProject((current) => ({
        ...current,
        pipeline: {
          ...current.pipeline,
          edges: changed.map((edge) => ({
            source_node: edge.source,
            target_node: edge.target,
            source_port: String(edge.data?.sourcePort ?? edge.sourceHandle ?? 'output'),
            target_port: String(edge.data?.targetPort ?? edge.targetHandle ?? 'input'),
          })),
        },
      }));
    },
    [flowEdges, updateProject],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      updateProject((current) => connectProject(current, connection));
    },
    [updateProject],
  );

  const selectedNode = project.pipeline.nodes.find((node) => node.id === selectedNodeId) ?? null;

  const persistProject = async (
    nextProject: EditableProject,
  ): Promise<ProjectDocument> => {
    const saved = persisted
      ? await api.saveProject(nextProject)
      : await api.createProject(nextProject);
    setProject(saved);
    setPersisted(true);
    await refreshProjects();
    return saved;
  };

  const prepareProject = (): EditableProject | null => {
    const materialized = materializeDataSources(
      project.data_sources,
      sourceDrafts,
    );
    setSourceDrafts(materialized.drafts);
    if (!materialized.ok) {
      setMessage(materialized.message);
      return null;
    }
    const prepared = { ...project, data_sources: materialized.sources };
    setProject(prepared);
    return prepared;
  };

  const save = async () => {
    setBusy(true);
    try {
      const prepared = prepareProject();
      if (!prepared) return;
      await persistProject(prepared);
      setMessage('Project saved');
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const validate = async () => {
    setBusy(true);
    try {
      const prepared = prepareProject();
      if (!prepared) return;
      const saved = await persistProject(prepared);
      const report = await api.validateProject(saved.id);
      setValidation(report);
      setMessage(report.valid ? 'Graph compiled successfully' : 'Validation failed');
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const execute = async () => {
    setBusy(true);
    setMessage('');
    try {
      const prepared = prepareProject();
      if (!prepared) return;
      const saved = await persistProject(prepared);
      const submitted = await api.runProject(saved.id, {});
      setRun(submitted);
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const inspectCheckpoint = async () => {
    setBusy(true);
    try {
      const prepared = prepareProject();
      if (!prepared) return;
      const saved = await persistProject(prepared);
      setCheckpoint(await api.checkpoint(saved.id));
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const resetCheckpoint = async () => {
    if (!persisted) return;
    setBusy(true);
    try {
      setCheckpoint(await api.resetCheckpoint(project.id));
      setMessage('Runner checkpoint reset');
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const addDataSource = () => {
    const source = nextDataSource(project.data_sources);
    updateProject((current) => ({
      ...current,
      data_sources: [...current.data_sources, source],
    }));
    setSourceDrafts((current) => [
      ...current,
      ...createDataSourceDrafts([source]),
    ]);
  };

  const removeDataSource = (index: number) => {
    updateProject((current) => ({
      ...current,
      data_sources: current.data_sources.filter(
        (_, currentIndex) => currentIndex !== index,
      ),
    }));
    setSourceDrafts((current) =>
      current.filter((_, currentIndex) => currentIndex !== index),
    );
  };

  const updateDataSourceField = (
    index: number,
    field: 'id' | 'input' | 'format',
    value: string,
  ) => updateProject((current) => ({
    ...current,
    data_sources: current.data_sources.map((source, currentIndex) =>
      currentIndex === index
        ? {
            ...source,
            [field]: field === 'format' ? value as DataSourceFormat : value,
          }
        : source,
    ),
  }));

  const updateDataSourceData = (index: number, dataText: string) => {
    setSourceDrafts((current) =>
      current.map((draft, currentIndex) =>
        currentIndex === index ? { ...draft, dataText, error: null } : draft,
      ),
    );
    setValidation(null);
    setRun(null);
  };

  const loadDataSourceFile = async (index: number, file: File) => {
    try {
      const format = project.data_sources[index]?.format;
      const dataText = format === 'arrow_ipc'
        ? await fileToBase64(file)
        : await file.text();
      updateDataSourceData(index, dataText);
    } catch (error) {
      setMessage((error as Error).message);
    }
  };

  const cancelRun = async () => {
    if (!run) return;
    try {
      setRun(await api.cancelRun(run.id));
    } catch (error) {
      setMessage((error as Error).message);
    }
  };

  const newProject = () => {
    const fresh = blankProject();
    replaceEditableProject(fresh, false);
    setMessage('');
  };

  const importProject = async (file: File) => {
    const extension = file.name.toLowerCase().split('.').pop();
    const format = extension === 'yaml' || extension === 'yml'
      ? 'yaml'
      : extension === 'json'
        ? 'json'
        : null;
    if (!format) {
      setMessage('Project import requires a .json, .yaml, or .yml file');
      return;
    }
    setBusy(true);
    try {
      const document = await file.text();
      let imported: ProjectDocument;
      try {
        imported = await api.importProject(document, format);
      } catch (error) {
        if (
          !(error instanceof Error)
          || !('status' in error)
          || error.status !== 409
          || !window.confirm('Replace the existing project with this import?')
        ) {
          throw error;
        }
        imported = await api.importProject(document, format, true);
      }
      replaceEditableProject(imported, true);
      await refreshProjects();
      setMessage('Project imported');
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const exportProject = async (format: 'json' | 'yaml') => {
    if (!persisted) return;
    setBusy(true);
    try {
      const exported = await api.exportProject(project.id, format);
      const url = URL.createObjectURL(
        new Blob([exported.document], {
          type: format === 'json' ? 'application/json' : 'application/yaml',
        }),
      );
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = exported.filename ?? `${project.id}.${format}`;
      anchor.click();
      URL.revokeObjectURL(url);
      setMessage(`Project exported as ${format.toUpperCase()}`);
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const deleteProject = async () => {
    if (!persisted || !window.confirm(`Delete ${project.name}?`)) return;
    setBusy(true);
    try {
      await api.deleteProject(project.id);
      const remaining = await refreshProjects();
      if (remaining.length) await loadProject(remaining[0].id);
      else newProject();
      setMessage('Project deleted');
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const addNode = (kind: EditableNodeKind) => {
    const node = makeNode(kind, project.pipeline.nodes);
    updateProject((current) => ({
      ...current,
      pipeline: { ...current.pipeline, nodes: [...current.pipeline.nodes, node] },
    }));
    setSelectedNodeId(node.id);
  };

  const updateNode = (node: NodeConfig) => {
    updateProject((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        nodes: current.pipeline.nodes.map((item) => (item.id === node.id ? node : item)),
      },
    }));
  };

  const deleteSelectedNode = () => {
    if (!selectedNode) return;
    const nodes = project.pipeline.nodes.filter((node) => node.id !== selectedNode.id);
    updateProject((current) => ({
      ...current,
      pipeline: {
        ...current.pipeline,
        nodes: current.pipeline.nodes.filter((node) => node.id !== selectedNode.id),
        edges: current.pipeline.edges.filter(
          (edge) => edge.source_node !== selectedNode.id && edge.target_node !== selectedNode.id,
        ),
      },
    }));
    setSelectedNodeId(nodes[0]?.id ?? '');
  };

  const loadProject = async (id: string) => {
    if (!id) {
      newProject();
      return;
    }
    const loaded = await api.project(id);
    replaceEditableProject(loaded, true);
  };

  return (
    <main className="studio-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark"><span /><span /><span /></div>
          <div><strong>Calc Flow</strong><small>DataFusion studio</small></div>
        </div>
        <div className="project-switcher">
          <select aria-label="Project" value={persisted && projects.some((item) => item.id === project.id) ? project.id : ''} onChange={(event) => void loadProject(event.target.value)}>
            <option value="">New project</option>
            {projects.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}
          </select>
          <input aria-label="Project name" value={project.name} onChange={(event) => {
            const name = event.target.value;
            updateProject((current) => ({ ...current, name }));
          }} />
        </div>
        <div className="topbar-actions">
          <ProjectActions
            persisted={persisted}
            busy={busy}
            onNew={newProject}
            onImport={(file) => void importProject(file)}
            onExport={(format) => void exportProject(format)}
            onDelete={() => void deleteProject()}
          />
          <button className="ghost-button" type="button" disabled={busy} onClick={() => void save()}>Save</button>
          <button className="ghost-button" type="button" disabled={busy} onClick={() => void validate()}>Validate</button>
          <button className="run-button" type="button" disabled={busy} onClick={() => void execute()}><span>▶</span> Run preview</button>
        </div>
      </header>

      {message && <div className="toast" role="status" onClick={() => setMessage('')}>{message}</div>}

      <section className="workspace">
        <aside className="toolbox panel">
          <span className="eyebrow">Node catalog</span>
          <h2>Build the flow</h2>
          <p>Drag connections between typed calculation nodes.</p>
          <button className="node-tool expression" type="button" onClick={() => addNode('expression')}><span>ƒx</span><div><strong>Expression</strong><small>Project · filter · calculate</small></div></button>
          <button className="node-tool sql" type="button" onClick={() => addNode('sql')}><span>SQL</span><div><strong>DataFusion SQL</strong><small>Join · aggregate · window</small></div></button>

          <DataSourceEditor
            sources={project.data_sources}
            drafts={sourceDrafts}
            busy={busy}
            onAdd={addDataSource}
            onRemove={removeDataSource}
            onFieldChange={updateDataSourceField}
            onDataChange={updateDataSourceData}
            onLoadFile={(index, file) => void loadDataSourceFile(index, file)}
          />
          <CheckpointControl
            checkpoint={checkpoint}
            busy={busy}
            onInspect={() => void inspectCheckpoint()}
            onReset={() => void resetCheckpoint()}
          />
        </aside>

        <section className="canvas-panel">
          <div className="canvas-meta"><span>{project.pipeline.nodes.length} nodes</span><span>{project.pipeline.edges.length} edges</span><span>DataFusion · {project.pipeline.datafusion.target_partitions} partition</span></div>
          <ReactFlow<FlowNode, Edge>
            nodes={flowNodes}
            edges={flowEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={(_, node) => setSelectedNodeId(node.id)}
            nodeTypes={nodeTypes}
            fitView
            minZoom={0.3}
            maxZoom={1.8}
          >
            <Background variant={BackgroundVariant.Dots} gap={24} size={1.5} color="#2c4942" />
            <MiniMap nodeColor={nodeColor} maskColor="rgba(7, 20, 18, 0.72)" />
            <Controls />
          </ReactFlow>
        </section>

        {selectedNode ? (
          <NodeInspector
            node={selectedNode}
            arrowTypes={ARROW_TYPES}
            udfs={catalog ?? []}
            onChange={updateNode}
            onDelete={deleteSelectedNode}
          />
        ) : (
          <aside className="inspector panel"><div className="empty-state"><p>Select a node to edit its calculation.</p></div></aside>
        )}
      </section>

      <ResultsPanel
        validation={validation}
        run={run}
        onCancel={() => void cancelRun()}
      />
      <BenchmarkComparison />
    </main>
  );
}
