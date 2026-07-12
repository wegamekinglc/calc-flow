import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  addEdge,
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
} from './components/CalculationNode';
import { BenchmarkComparison } from './components/BenchmarkComparison';
import { CheckpointControl } from './components/CheckpointControl';
import { NodeInspector } from './components/NodeInspector';
import { ProjectActions } from './components/ProjectActions';
import { ResultsPanel } from './components/ResultsPanel';
import { useRunEvents } from './hooks/useRunEvents';
import {
  blankProject,
  type CatalogResponse,
  type CheckpointSummary,
  type EditableProject,
  type JSONValue,
  type NodeConfig,
  type ProjectConfig,
  type ProjectSummary,
  type RunResponse,
  type ValidationReport,
} from './types';

type FlowNode = CalculationFlowNode;

const nodeTypes: NodeTypes = { calculation: CalculationNode };

const nodeColor = (node: FlowNode) => {
  if (node.data.kind === 'sql') return '#ef9456';
  if (node.data.kind === 'array_expression') return '#a994ff';
  return '#56d5b2';
};

const nextId = (kind: NodeConfig['kind'], nodes: NodeConfig[]) => {
  const prefix = kind === 'array_expression' ? 'array' : kind;
  let index = nodes.length + 1;
  while (nodes.some((node) => node.id === `${prefix}_${index}`)) index += 1;
  return `${prefix}_${index}`;
};

const makeNode = (kind: NodeConfig['kind'], nodes: NodeConfig[]): NodeConfig => {
  const id = nextId(kind, nodes);
  const shared = {
    id,
    kind,
    select: [],
    filter_expression: null,
    udfs: [],
    input_ports: [],
    output_ports: [],
    position: { x: 120 + nodes.length * 32, y: 100 + nodes.length * 36 },
  };
  if (kind === 'sql') {
    return {
      ...shared,
      expression: null,
      query: 'SELECT * FROM input',
      inputs: ['input'],
      backend: null,
    };
  }
  if (kind === 'array_expression') {
    return {
      ...shared,
      expression: 'x * 2',
      query: null,
      inputs: [],
      backend: 'numpy',
    };
  }
  return {
    ...shared,
    expression: 'result = value + 1',
    query: null,
    inputs: [],
    backend: null,
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
  const [selectedNodeId, setSelectedNodeId] = useState<string>('calculate');
  const [validation, setValidation] = useState<ValidationReport | null>(null);
  const [run, setRun] = useState<RunResponse | null>(null);
  const [checkpoint, setCheckpoint] = useState<CheckpointSummary | null>(null);
  const [sampleFormat, setSampleFormat] = useState<'inline_json' | 'json' | 'csv' | 'arrow_ipc'>('inline_json');
  const [sampleInputName, setSampleInputName] = useState('input');
  const [sampleData, setSampleData] = useState('[{"a": 1, "b": 2}, {"a": 3, "b": 4}]');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);

  const refreshProjects = useCallback(async () => {
    const items = await api.projects();
    setProjects(items);
    return items;
  }, []);

  useEffect(() => {
    Promise.all([api.catalog(), refreshProjects()])
      .then(async ([loadedCatalog, loadedProjects]) => {
        setCatalog(loadedCatalog);
        if (loadedProjects.length) {
          const loaded = await api.project(loadedProjects[0].id);
          setProject(loaded);
          setSelectedNodeId(loaded.pipeline.nodes[0]?.id ?? '');
        }
      })
      .catch((error: Error) => setMessage(error.message));
  }, [refreshProjects]);

  const updateProject = useCallback((next: EditableProject) => {
    setProject(next);
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
        position: node.position,
        data: {
          label: node.id,
          kind: node.kind,
          inputPorts: node.kind === 'sql' ? node.inputs : ['input'],
          outputPorts: ['output'],
        },
        className: `flow-node ${node.kind}`,
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
      updateProject({
        ...project,
        pipeline: {
          ...project.pipeline,
          nodes: project.pipeline.nodes.map((node) => ({
            ...node,
            position: positions.get(node.id) ?? node.position,
          })),
        },
      });
    },
    [flowNodes, project, updateProject],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange<Edge>[]) => {
      const structuralChanges = changes.filter((change) => change.type === 'remove');
      if (!structuralChanges.length) return;
      const changed = applyEdgeChanges(structuralChanges, flowEdges);
      updateProject({
        ...project,
        pipeline: {
          ...project.pipeline,
          edges: changed.map((edge) => ({
            source_node: edge.source,
            target_node: edge.target,
            source_port: String(edge.data?.sourcePort ?? edge.sourceHandle ?? 'output'),
            target_port: String(edge.data?.targetPort ?? edge.targetHandle ?? 'input'),
          })),
        },
      });
    },
    [flowEdges, project, updateProject],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      const changed = addEdge(connection, flowEdges);
      const edge = changed.at(-1);
      if (!edge) return;
      updateProject({
        ...project,
        pipeline: {
          ...project.pipeline,
          edges: [
            ...project.pipeline.edges,
            {
              source_node: connection.source,
              target_node: connection.target,
              source_port: connection.sourceHandle ?? 'output',
              target_port: connection.targetHandle ?? 'input',
            },
          ],
        },
      });
    },
    [flowEdges, project, updateProject],
  );

  const selectedNode = project.pipeline.nodes.find((node) => node.id === selectedNodeId) ?? null;

  const persistProject = async (): Promise<ProjectConfig> => {
    const saved = 'id' in project
      ? await api.saveProject(project)
      : await api.createProject(project);
    setProject(saved);
    await refreshProjects();
    return saved;
  };

  const save = async () => {
    setBusy(true);
    try {
      await persistProject();
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
      const saved = await persistProject();
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
      const saved = await persistProject();
      let data: JSONValue = sampleData;
      if (sampleFormat === 'inline_json') data = JSON.parse(sampleData) as JSONValue;
      const submitted = await api.runProject(saved.id, {
        inputs: {
          [sampleInputName]: { format: sampleFormat, data, source_id: 'browser-preview' },
        },
      });
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
      const saved = await persistProject();
      setCheckpoint(await api.checkpoint(saved.id));
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const resetCheckpoint = async () => {
    if (!('id' in project)) return;
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

  const newProject = () => {
    const fresh = blankProject();
    setProject(fresh);
    setSelectedNodeId(fresh.pipeline.nodes[0].id);
    setValidation(null);
    setRun(null);
    setCheckpoint(null);
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
      let imported: ProjectConfig;
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
      setProject(imported);
      setSelectedNodeId(imported.pipeline.nodes[0]?.id ?? '');
      setValidation(null);
      setRun(null);
      setCheckpoint(null);
      await refreshProjects();
      setMessage('Project imported');
    } catch (error) {
      setMessage((error as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const exportProject = async (format: 'json' | 'yaml') => {
    if (!('id' in project)) return;
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
    if (!('id' in project) || !window.confirm(`Delete ${project.name}?`)) return;
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

  const addNode = (kind: NodeConfig['kind']) => {
    const node = makeNode(kind, project.pipeline.nodes);
    updateProject({
      ...project,
      pipeline: { ...project.pipeline, nodes: [...project.pipeline.nodes, node] },
    });
    setSelectedNodeId(node.id);
  };

  const updateNode = (node: NodeConfig) => {
    updateProject({
      ...project,
      pipeline: {
        ...project.pipeline,
        nodes: project.pipeline.nodes.map((item) => (item.id === node.id ? node : item)),
      },
    });
  };

  const deleteSelectedNode = () => {
    if (!selectedNode) return;
    const nodes = project.pipeline.nodes.filter((node) => node.id !== selectedNode.id);
    updateProject({
      ...project,
      pipeline: {
        ...project.pipeline,
        nodes,
        edges: project.pipeline.edges.filter(
          (edge) => edge.source_node !== selectedNode.id && edge.target_node !== selectedNode.id,
        ),
      },
    });
    setSelectedNodeId(nodes[0]?.id ?? '');
  };

  const loadProject = async (id: string) => {
    if (!id) {
      newProject();
      return;
    }
    const loaded = await api.project(id);
    setProject(loaded);
    setSelectedNodeId(loaded.pipeline.nodes[0]?.id ?? '');
    setValidation(null);
    setRun(null);
    setCheckpoint(null);
  };

  return (
    <main className="studio-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark"><span /><span /><span /></div>
          <div><strong>Calc Flow</strong><small>DataFusion studio</small></div>
        </div>
        <div className="project-switcher">
          <select aria-label="Project" value={'id' in project && projects.some((item) => item.id === project.id) ? project.id : ''} onChange={(event) => void loadProject(event.target.value)}>
            <option value="">New project</option>
            {projects.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}
          </select>
          <input aria-label="Project name" value={project.name} onChange={(event) => updateProject({ ...project, name: event.target.value })} />
        </div>
        <div className="topbar-actions">
          <ProjectActions
            persisted={'id' in project}
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
          <button className="node-tool array" type="button" onClick={() => addNode('array_expression')}><span>[ ]</span><div><strong>Array expression</strong><small>NumPy or JAX</small></div></button>

          <div className="sample-editor">
            <span className="eyebrow">Preview input</span>
            <label>Graph input<input value={sampleInputName} onChange={(event) => setSampleInputName(event.target.value)} /></label>
            <label>Format<select value={sampleFormat} onChange={(event) => setSampleFormat(event.target.value as typeof sampleFormat)}><option value="inline_json">Inline JSON</option><option value="json">JSON / NDJSON</option><option value="csv">CSV</option><option value="arrow_ipc">Arrow IPC</option></select></label>
            <textarea aria-label="Sample data" rows={8} value={sampleData} onChange={(event) => setSampleData(event.target.value)} />
            <label className="file-button">Load file<input type="file" accept=".json,.jsonl,.csv,.arrow,.ipc" onChange={(event) => {
              const file = event.target.files?.[0];
              if (!file) return;
              if (sampleFormat === 'arrow_ipc') void fileToBase64(file).then(setSampleData);
              else void file.text().then(setSampleData);
            }} /></label>
          </div>
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
            arrowTypes={catalog?.arrow_types ?? ['float64', 'int64', 'string']}
            udfs={(catalog?.udfs ?? []) as Record<string, JSONValue>[]}
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
        onCancel={() => run && void api.cancelRun(run.id).then(setRun)}
      />
      <BenchmarkComparison />
    </main>
  );
}
