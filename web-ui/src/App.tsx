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
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
} from 'react';

import { api } from './api/client';
import {
  CalculationNode,
  type CalculationFlowNode,
  type FlowNodeData,
} from './components/CalculationNode';
import { BenchmarkComparison } from './components/BenchmarkComparison';
import { DataSourceEditor } from './components/DataSourceEditor';
import { NodeInspector } from './components/NodeInspector';
import { inspectLoweredNode } from './components/projectInspectionModel';
import { PanelResizeHandle } from './components/PanelResizeHandle';
import { ProjectActions } from './components/ProjectActions';
import { ResultsPanel } from './components/ResultsPanel';
import { StreamConfigEditor } from './components/StreamConfigEditor';
import {
  createDataSourceDrafts,
  materializeDataSources,
  nextDataSource,
  type DataSourceDraft,
  type DataSourceFormat,
} from './components/dataSourceEditorModel';
import { editSqlInputAliases } from './components/inputAliasEditorModel';
import { derivedInputNames, derivedOutputNames } from './portNamesModel';
import { isJobActive } from './jobStatusModel';
import {
  PANEL_LIMITS,
  PANEL_RESIZE_HANDLE_WIDTH,
  clampWorkspaceLayout,
  useElementWidth,
  usePanelLayout,
} from './components/panelLayout';
import { useBusyAction } from './hooks/useBusyAction';
import { useJobEvents } from './hooks/useJobEvents';
import {
  blankProject,
  type CatalogResponse,
  type CapabilitiesResponse,
  type EditableProject,
  type JobEvent,
  type JobResponse,
  type NodeConfig,
  type ProjectDocument,
  type ProjectSummary,
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
type EditableNodeKind = Extract<
  NodeConfig['operator']['kind'],
  'expression' | 'sql' | 'stream_join'
>;
type ProjectUpdate = EditableProject | ((current: EditableProject) => EditableProject);
type SourceDraftUpdate =
  | DataSourceDraft[]
  | ((current: DataSourceDraft[]) => DataSourceDraft[]);

const nodeColor = (node: FlowNode) => {
  if (node.data.kind === 'sql') return '#ef9456';
  if (node.data.kind === 'stream_join') return '#56a8d5';
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
  if (kind === 'stream_join') {
    return {
      ...shared,
      input_ports: [
        { name: 'left', kind: 'table', required: true, schema: [] },
        { name: 'right', kind: 'table', required: true, schema: [] },
      ],
      operator: {
        kind: 'stream_join',
        spec: {
          join_type: 'inner',
          left_keys: [],
          right_keys: [],
          left_event_time: '',
          right_event_time: '',
          bounds: { before_micros: 0, after_micros: 0 },
          limits: {
            max_state_rows_per_side: 100_000,
            max_state_bytes_per_side: 134_217_728,
            max_matches_per_input_batch: 1_000_000,
          },
          left_prefix: 'left',
          right_prefix: 'right',
        },
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
  inputPorts: derivedInputNames(node),
  outputPorts: derivedOutputNames(node),
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
  const duplicate = project.graph.edges.some(
    (current) =>
      current.source_node === edge.source_node
      && current.target_node === edge.target_node
      && current.source_port === edge.source_port
      && current.target_port === edge.target_port,
  );
  if (duplicate) return project;
  return {
    ...project,
    graph: {
      ...project.graph,
      edges: [...project.graph.edges, edge],
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
  const { layout, setPanelWidth, resetPanelWidth } = usePanelLayout();
  const { ref: workspaceRef, width: workspaceWidth } = useElementWidth<HTMLElement>();
  const [catalog, setCatalog] = useState<CatalogResponse | null>(null);
  const [capabilities, setCapabilities] = useState<CapabilitiesResponse | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [project, setProject] = useState<EditableProject>(() => blankProject());
  const projectRef = useRef(project);
  const [sourceDrafts, setSourceDrafts] = useState<DataSourceDraft[]>(() =>
    createDataSourceDrafts(project.data_sources),
  );
  const sourceDraftsRef = useRef(sourceDrafts);
  const [persisted, setPersisted] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string>('calculate');
  const [validation, setValidation] = useState<ValidationReport | null>(null);
  const [job, setJob] = useState<JobResponse | null>(null);
  const [progress, setProgress] = useState<JobEvent | null>(null);
  const [message, setMessage] = useState('');
  const { busy, run } = useBusyAction(setMessage);

  const [pendingFileReads, setPendingFileReads] = useState(0);
  const [pendingFileReadKeys, setPendingFileReadKeys] = useState<
    ReadonlySet<string>
  >(() => new Set());
  const pendingFileReadsRef = useRef(0);
  const pendingFileReadKeyCountsRef = useRef(new Map<string, number>());
  const fileReadTokensRef = useRef(new Map<string, symbol>());

  const updateSourceDrafts = useCallback((update: SourceDraftUpdate) => {
    const next = typeof update === 'function'
      ? update(sourceDraftsRef.current)
      : update;
    sourceDraftsRef.current = next;
    setSourceDrafts(next);
  }, []);

  const refreshProjects = useCallback(async () => {
    const items = await api.projects();
    setProjects(items);
    return items;
  }, []);

  const replaceEditableProject = useCallback(
    (next: EditableProject, isPersisted: boolean) => {
      const drafts = createDataSourceDrafts(next.data_sources);
      projectRef.current = next;
      sourceDraftsRef.current = drafts;
      fileReadTokensRef.current.clear();
      setProject(next);
      setSourceDrafts(drafts);
      setPersisted(isPersisted);
      setSelectedNodeId(next.graph.nodes[0]?.id ?? '');
      setValidation(null);
      setJob(null);
      setProgress(null);
    },
    [],
  );

  useEffect(() => {
    const controller = new AbortController();
    // Read the flag through a helper: narrowing of signal.aborted must not
    // carry across the awaits, where cleanup may abort the controller.
    const aborted = (): boolean => controller.signal.aborted;
    const initialize = async () => {
      try {
        const [loadedCatalog, loadedCapabilities, loadedProjects] = await Promise.all([
          api.catalog(),
          api.capabilities(),
          api.projects(),
        ]);
        if (aborted()) return;
        setCatalog(loadedCatalog);
        setCapabilities(loadedCapabilities);
        setProjects(loadedProjects);
        const [firstProject] = loadedProjects;
        if (firstProject) {
          const loaded = await api.project(firstProject.id);
          if (aborted()) return;
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
    const next = typeof update === 'function'
      ? update(projectRef.current)
      : update;
    projectRef.current = next;
    setProject(next);
    setValidation(null);
  }, []);

  const handleJobUpdate = useCallback((next: JobResponse) => {
    setJob(next);
  }, []);

  const handleJobEvent = useCallback((event: JobEvent) => {
    setProgress((current) => ({ ...current, ...event }));
  }, []);

  const handleJobError = useCallback((error: Error) => {
    setJob(null);
    setProgress(null);
    setMessage(error.message);
  }, []);

  const observedJobId = job !== null && isJobActive(job.status)
    ? job.id
    : null;
  useJobEvents(observedJobId, handleJobUpdate, handleJobEvent, handleJobError);

  const flowNodes = useMemo<FlowNode[]>(
    () =>
      project.graph.nodes.map((node) => ({
        id: node.id,
        type: 'calculation',
        position: node.position ?? { x: 0, y: 0 },
        data: flowNodeData(node),
        className: `flow-node ${node.operator.kind}`,
        selected: selectedNodeId === node.id,
      })),
    [project.graph.nodes, selectedNodeId],
  );

  const flowEdges = useMemo<Edge[]>(
    () =>
      project.graph.edges.map((edge, index) => ({
        id: `${edge.source_node}:${edge.source_port}-${edge.target_node}:${edge.target_port}-${index}`,
        source: edge.source_node,
        target: edge.target_node,
        sourceHandle: edge.source_port,
        targetHandle: edge.target_port,
        data: { sourcePort: edge.source_port, targetPort: edge.target_port },
        animated: true,
      })),
    [project.graph.edges],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange<FlowNode>[]) => {
      const positionChanges = changes.filter((change) => change.type === 'position');
      if (!positionChanges.length) return;
      const changed = applyNodeChanges(positionChanges, flowNodes);
      const positions = new Map(changed.map((node) => [node.id, node.position]));
      updateProject((current) => ({
        ...current,
        graph: {
          ...current.graph,
          nodes: current.graph.nodes.map((node) => ({
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
        graph: {
          ...current.graph,
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

  const selectedNode = project.graph.nodes.find((node) => node.id === selectedNodeId) ?? null;

  const persistProject = async (
    nextProject: EditableProject,
  ): Promise<ProjectDocument> => {
    const saved = persisted
      ? await api.saveProject(nextProject)
      : await api.createProject(nextProject);
    projectRef.current = saved;
    setProject(saved);
    setPersisted(true);
    await refreshProjects();
    return saved;
  };

  const prepareProject = (): EditableProject | null => {
    if (projectRef.current.runtime.mode === 'stream') {
      const prepared = { ...projectRef.current, data_sources: [] };
      projectRef.current = prepared;
      setProject(prepared);
      return prepared;
    }
    if (pendingFileReadsRef.current > 0) {
      setMessage('Data source files are still loading');
      return null;
    }
    const materialized = materializeDataSources(
      projectRef.current.data_sources,
      sourceDraftsRef.current,
    );
    updateSourceDrafts(materialized.drafts);
    if (!materialized.ok) {
      setMessage(materialized.message);
      return null;
    }
    const prepared = {
      ...projectRef.current,
      data_sources: materialized.sources,
    };
    projectRef.current = prepared;
    setProject(prepared);
    return prepared;
  };

  const save = () => run(async () => {
    const prepared = prepareProject();
    if (!prepared) return;
    await persistProject(prepared);
    setMessage('Project saved');
  });

  const validate = () => run(async () => {
    const prepared = prepareProject();
    if (!prepared) return;
    const saved = await persistProject(prepared);
    const report = await api.validateProject(saved.id);
    setValidation(report);
    setMessage(report.valid ? 'Graph compiled successfully' : 'Validation failed');
  });

  const startJob = () => {
    setMessage('');
    void run(async () => {
      const prepared = prepareProject();
      if (!prepared) return;
      if (prepared.runtime.mode !== 'stream') {
        setMessage('Switch the project to Stream mode before starting a job');
        return;
      }
      const saved = await persistProject(prepared);
      const submitted = await api.startJob(saved.id);
      setProgress(null);
      setJob(submitted);
      setMessage('Continuous job started');
    });
  };

  const checkpointJob = () => {
    if (!job) return;
    void run(async () => {
      setJob(await api.checkpointJob(job.id));
      setMessage('Checkpoint requested');
    });
  };

  const shutdownJob = () => {
    if (!job) return;
    void run(async () => {
      setJob(await api.shutdownJob(job.id));
      setMessage('Graceful shutdown requested');
    });
  };

  const addDataSource = () => {
    const source = nextDataSource(project.data_sources);
    updateProject((current) => ({
      ...current,
      data_sources: [...current.data_sources, source],
    }));
    updateSourceDrafts((current) => [
      ...current,
      ...createDataSourceDrafts([source]),
    ]);
  };

  const removeDataSource = (index: number) => {
    const draftKey = sourceDraftsRef.current[index]?.key;
    if (draftKey) fileReadTokensRef.current.delete(draftKey);
    updateProject((current) => ({
      ...current,
      data_sources: current.data_sources.filter(
        (_, currentIndex) => currentIndex !== index,
      ),
    }));
    updateSourceDrafts((current) =>
      current.filter((_, currentIndex) => currentIndex !== index),
    );
  };

  const updateDataSourceField = (
    index: number,
    field: 'id' | 'input' | 'format',
    value: string,
  ) => {
    const draftKey = sourceDraftsRef.current[index]?.key;
    updateProject((current) => ({
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
    if (field === 'format') {
      if (draftKey) fileReadTokensRef.current.delete(draftKey);
      updateSourceDrafts((current) =>
        current.map((draft, currentIndex) =>
          currentIndex === index ? { ...draft, error: null } : draft,
        ),
      );
    }
  };

  const updateDataSourceData = (
    index: number,
    dataText: string,
    invalidateFileRead = true,
  ) => {
    const draftKey = sourceDraftsRef.current[index]?.key;
    if (invalidateFileRead && draftKey) {
      fileReadTokensRef.current.delete(draftKey);
    }
    updateSourceDrafts((current) =>
      current.map((draft, currentIndex) =>
        currentIndex === index ? { ...draft, dataText, error: null } : draft,
      ),
    );
    setValidation(null);
  };

  const loadDataSourceFile = async (index: number, file: File) => {
    const draftKey = sourceDraftsRef.current[index]?.key;
    const format = projectRef.current.data_sources[index]?.format;
    if (!draftKey || !format) return;
    const fileReadToken = Symbol(draftKey);
    fileReadTokensRef.current.set(draftKey, fileReadToken);

    const currentTargetIndex = (): number => {
      const currentIndex = sourceDraftsRef.current.findIndex(
        (draft) => draft.key === draftKey,
      );
      return currentIndex >= 0
        && projectRef.current.data_sources[currentIndex]?.format === format
        && fileReadTokensRef.current.get(draftKey) === fileReadToken
        ? currentIndex
        : -1;
    };

    pendingFileReadsRef.current += 1;
    setPendingFileReads(pendingFileReadsRef.current);
    const keyCounts = pendingFileReadKeyCountsRef.current;
    keyCounts.set(draftKey, (keyCounts.get(draftKey) ?? 0) + 1);
    setPendingFileReadKeys(new Set(keyCounts.keys()));
    try {
      const dataText = format === 'arrow_ipc'
        ? await fileToBase64(file)
        : await file.text();
      const currentIndex = currentTargetIndex();
      if (currentIndex >= 0) {
        updateDataSourceData(currentIndex, dataText, false);
      }
    } catch (error) {
      if (currentTargetIndex() >= 0) setMessage((error as Error).message);
    } finally {
      if (fileReadTokensRef.current.get(draftKey) === fileReadToken) {
        fileReadTokensRef.current.delete(draftKey);
      }
      pendingFileReadsRef.current = Math.max(
        0,
        pendingFileReadsRef.current - 1,
      );
      setPendingFileReads(pendingFileReadsRef.current);
      const remainingForKey = Math.max(
        0,
        (keyCounts.get(draftKey) ?? 0) - 1,
      );
      if (remainingForKey > 0) {
        keyCounts.set(draftKey, remainingForKey);
      } else {
        keyCounts.delete(draftKey);
      }
      setPendingFileReadKeys(new Set(keyCounts.keys()));
    }
  };

  const cancelJob = () => {
    if (!job) return;
    void run(async () => {
      setJob(await api.cancelJob(job.id));
      setMessage('Job cancelled');
    });
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
    void run(async () => {
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
    });
  };

  const exportProject = (format: 'json' | 'yaml') => {
    if (!persisted) return;
    void run(async () => {
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
    });
  };

  const deleteProject = () => {
    if (!persisted || !window.confirm(`Delete ${project.name}?`)) return;
    void run(async () => {
      await api.deleteProject(project.id);
      const remaining = await refreshProjects();
      const [nextProject] = remaining;
      if (nextProject) await loadProject(nextProject.id);
      else newProject();
      setMessage('Project deleted');
    });
  };

  const addNode = (kind: EditableNodeKind) => {
    const node = makeNode(kind, project.graph.nodes);
    updateProject((current) => ({
      ...current,
      graph: { ...current.graph, nodes: [...current.graph.nodes, node] },
    }));
    setSelectedNodeId(node.id);
  };

  const updateNode = (node: NodeConfig) => {
    updateProject((current) => ({
      ...current,
      graph: {
        ...current.graph,
        nodes: current.graph.nodes.map((item) => (item.id === node.id ? node : item)),
      },
    }));
  };

  const deleteSelectedNode = () => {
    if (!selectedNode) return;
    const nodes = project.graph.nodes.filter((node) => node.id !== selectedNode.id);
    updateProject((current) => ({
      ...current,
      graph: {
        ...current.graph,
        nodes: current.graph.nodes.filter((node) => node.id !== selectedNode.id),
        edges: current.graph.edges.filter(
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

  const persistenceBusy = busy || pendingFileReads > 0;
  const activeJob = job !== null && isJobActive(job.status);
  const workspaceLayout = useMemo(
    () => workspaceWidth > 0
      ? clampWorkspaceLayout(layout, workspaceWidth)
      : layout,
    [layout, workspaceWidth],
  );
  useEffect(() => {
    if (workspaceWidth <= 0) return;
    if (workspaceLayout.toolbox !== layout.toolbox) {
      setPanelWidth('toolbox', workspaceLayout.toolbox);
    }
    if (workspaceLayout.inspector !== layout.inspector) {
      setPanelWidth('inspector', workspaceLayout.inspector);
    }
  }, [layout, setPanelWidth, workspaceLayout, workspaceWidth]);
  const toolboxMaximum = workspaceWidth > 0
    ? Math.min(
        PANEL_LIMITS.toolbox.max,
        Math.max(
          PANEL_LIMITS.toolbox.min,
          workspaceWidth
            - PANEL_LIMITS.canvasMin
            - 2 * PANEL_RESIZE_HANDLE_WIDTH
            - workspaceLayout.inspector,
        ),
      )
    : PANEL_LIMITS.toolbox.max;
  const inspectorMaximum = workspaceWidth > 0
    ? Math.min(
        PANEL_LIMITS.inspector.max,
        Math.max(
          PANEL_LIMITS.inspector.min,
          workspaceWidth
            - PANEL_LIMITS.canvasMin
            - 2 * PANEL_RESIZE_HANDLE_WIDTH
            - workspaceLayout.toolbox,
        ),
      )
    : PANEL_LIMITS.inspector.max;
  const studioStyle = {
    '--toolbox-width': `${workspaceLayout.toolbox}px`,
    '--inspector-width': `${workspaceLayout.inspector}px`,
  } as CSSProperties;

  return (
    <main className="studio-shell" style={studioStyle}>
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
          <button className="ghost-button topbar-control" type="button" disabled={persistenceBusy} onClick={() => void save()}>Save</button>
          <button className="ghost-button topbar-control" type="button" disabled={persistenceBusy} onClick={() => void validate()}>Validate</button>
          <button className="run-button topbar-control" type="button" disabled={persistenceBusy || project.runtime.mode !== 'stream' || activeJob} onClick={() => void startJob()}><span>▶</span> Start job</button>
        </div>
      </header>

      {message && <div className="toast" role="status" onClick={() => setMessage('')}>{message}</div>}

      <section className="workspace" ref={workspaceRef}>
        <aside className="toolbox panel">
          <span className="eyebrow">Node catalog</span>
          <h2>Build the flow</h2>
          <p>Drag connections between typed calculation nodes.</p>
          <button className="node-tool expression" type="button" onClick={() => addNode('expression')}><span>ƒx</span><div><strong>Expression</strong><small>Project · filter · calculate</small></div></button>
          <button className="node-tool sql" type="button" onClick={() => addNode('sql')}><span>SQL</span><div><strong>DataFusion SQL</strong><small>Join · aggregate · window</small></div></button>
          {project.runtime.mode === 'stream' && (
            <button className="node-tool stream_join" type="button" onClick={() => { addNode('stream_join'); }}><span>⋈</span><div><strong>Stream Join</strong><small>Bounded · event time · inner</small></div></button>
          )}

          <StreamConfigEditor
            project={project}
            connectors={capabilities?.runtime.connectors ?? []}
            onChange={updateProject}
          />
          {project.runtime.mode === 'batch' && (
            <DataSourceEditor
              sources={project.data_sources}
              drafts={sourceDrafts}
              busy={busy}
              pendingSourceKeys={pendingFileReadKeys}
              onAdd={addDataSource}
              onRemove={removeDataSource}
              onFieldChange={updateDataSourceField}
              onDataChange={updateDataSourceData}
              onLoadFile={(index, file) => void loadDataSourceFile(index, file)}
            />
          )}
        </aside>

        <PanelResizeHandle
          label="Resize Toolbox"
          value={workspaceLayout.toolbox}
          min={PANEL_LIMITS.toolbox.min}
          max={toolboxMaximum}
          grow="start"
          onChange={(width) => setPanelWidth('toolbox', width)}
          onReset={() => resetPanelWidth('toolbox')}
        />

        <section className="canvas-panel">
          <div className="canvas-meta"><span>{project.graph.nodes.length} nodes</span><span>{project.graph.edges.length} edges</span><span>DataFusion · {project.graph.datafusion.target_partitions} partition</span></div>
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

        <PanelResizeHandle
          label="Resize Inspector"
          value={workspaceLayout.inspector}
          min={PANEL_LIMITS.inspector.min}
          max={inspectorMaximum}
          grow="end"
          onChange={(width) => setPanelWidth('inspector', width)}
          onReset={() => resetPanelWidth('inspector')}
        />

        {selectedNode ? (
          <NodeInspector
            node={selectedNode}
            inspection={inspectLoweredNode(project, selectedNode)}
            arrowTypes={ARROW_TYPES}
            udfs={catalog ?? []}
            onChange={updateNode}
            onSqlAliasEdit={(edit) => {
              const nodeId = selectedNode.id;
              updateProject((current) => editSqlInputAliases(current, nodeId, edit));
            }}
            onDelete={deleteSelectedNode}
          />
        ) : (
          <aside className="inspector panel"><div className="empty-state"><p>Select a node to edit its calculation.</p></div></aside>
        )}
      </section>

      <ResultsPanel
        validation={validation}
        job={job}
        progress={progress}
        busy={busy}
        onCheckpoint={() => void checkpointJob()}
        onShutdown={() => void shutdownJob()}
        onCancel={() => void cancelJob()}
      />
      <BenchmarkComparison />
    </main>
  );
}
