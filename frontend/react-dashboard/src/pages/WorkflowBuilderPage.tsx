import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import Card from '../components/Card';
import { api } from '../api/client';
import type {
  WorkflowDefinition,
  WorkflowEdge,
  WorkflowNode,
  WorkflowNodeType,
  WorkflowSummary,
  WorkflowRunRecord,
  WorkflowRunStep,
  RunnerHardwareSnapshot
} from '../api/types';

const NODE_TYPES: WorkflowNodeType[] = ['signature', 'verifier', 'reward', 'training', 'deployment', 'custom'];
const TYPE_LABELS: Record<WorkflowNodeType, string> = {
  signature: 'Signature',
  verifier: 'Verifier',
  reward: 'Reward',
  training: 'Training',
  deployment: 'Deployment',
  custom: 'Custom',
};

const emptyWorkflow = (): WorkflowDefinition => ({
  name: 'New Workflow',
  description: '',
  tenant: '',
  version: 'v1',
  tags: [],
  nodes: [],
  edges: [],
});

const createNodeId = (type: WorkflowNodeType) => `${type}-${Math.random().toString(36).slice(2, 8)}`;
const createEdgeId = () => `edge-${Math.random().toString(36).slice(2, 8)}`;

const createNode = (type: WorkflowNodeType, nodes: WorkflowNode[]): WorkflowNode => {
  const index = nodes.filter((node) => node.type === type).length + 1;
  const base: WorkflowNode = {
    id: createNodeId(type),
    type,
    name: `${TYPE_LABELS[type]} ${index}`,
    description: '',
    position: { x: (nodes.length % 4) * 220, y: Math.floor(nodes.length / 4) * 160 },
  };

  switch (type) {
    case 'signature':
      return {
        ...base,
        signature: {
          prompt: '',
          runtime: 'python',
          tools: [],
          temperature: 0.1,
        },
      };
    case 'verifier':
      return {
        ...base,
        verifier: {
          command: 'pytest -q',
          weight: 1,
          timeout_sec: 120,
        },
      };
    case 'reward':
      return {
        ...base,
        reward: {
          metric: 'pass_rate',
          target: 0.9,
          aggregator: 'mean',
        },
      };
    case 'training':
      return {
        ...base,
        training: {
          method: 'gepa',
          skill: 'default',
          schedule: 'nightly',
        },
      };
    case 'deployment':
      return {
        ...base,
        deployment: {
          tenant: 'default',
          domain: 'inference',
          channel: 'stable',
          autopromote: true,
        },
      };
    case 'custom':
      return {
        ...base,
        custom: {
          note: 'Custom configuration',
        },
      };
    default:
      return base;
  }
};

const createEdge = (workflow: WorkflowDefinition): WorkflowEdge => {
  const [first, second] = workflow.nodes;
  return {
    id: createEdgeId(),
    source: first?.id ?? '',
    target: (second ?? first)?.id ?? '',
    kind: 'data',
  };
};

const formatTime = (iso?: string) => {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
};

interface NodeCardProps {
  node: WorkflowNode;
  onUpdate: (updates: Partial<WorkflowNode>) => void;
  onRemove: () => void;
}

const NodeCard = ({ node, onUpdate, onRemove }: NodeCardProps) => {
  const commonPositionInputs = (
    <div className="grid gap-3 sm:grid-cols-2">
      <label className="grid gap-1 text-xs">
        <span className="text-slate-400">Position X</span>
        <input
          type="number"
          value={node.position?.x ?? 0}
          onChange={(event) =>
            onUpdate({ position: { x: Number(event.target.value), y: node.position?.y ?? 0 } })
          }
          className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
        />
      </label>
      <label className="grid gap-1 text-xs">
        <span className="text-slate-400">Position Y</span>
        <input
          type="number"
          value={node.position?.y ?? 0}
          onChange={(event) =>
            onUpdate({ position: { x: node.position?.x ?? 0, y: Number(event.target.value) } })
          }
          className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
        />
      </label>
    </div>
  );

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-900/50 p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1">
          <div className="text-xs uppercase tracking-wide text-emerald-300">{TYPE_LABELS[node.type]}</div>
          <input
            type="text"
            value={node.name}
            onChange={(event) => onUpdate({ name: event.target.value })}
            className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
          />
        </div>
        <button
          type="button"
          onClick={onRemove}
          className="rounded border border-slate-700 px-2 py-1 text-xs uppercase tracking-wide text-slate-300 hover:border-red-500 hover:text-red-300"
        >
          Remove
        </button>
      </div>

      <textarea
        value={node.description ?? ''}
        onChange={(event) => onUpdate({ description: event.target.value })}
        rows={2}
        className="mt-3 w-full rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
        placeholder="Description or notes"
      />

      <div className="mt-3">{commonPositionInputs}</div>

      {node.type === 'signature' && (
        <div className="mt-3 grid gap-3">
          <label className="grid gap-1 text-xs">
            <span className="text-slate-400">Prompt</span>
            <textarea
              value={node.signature?.prompt ?? ''}
              onChange={(event) =>
                onUpdate({
                  signature: {
                    prompt: event.target.value,
                    runtime: node.signature?.runtime ?? 'python',
                    tools: node.signature?.tools ?? [],
                    temperature: node.signature?.temperature,
                    max_tokens: node.signature?.max_tokens,
                    execution_domain: node.signature?.execution_domain,
                  },
                })
              }
              rows={3}
              className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
            />
          </label>
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="grid gap-1 text-xs">
              <span className="text-slate-400">Runtime</span>
              <input
                type="text"
                value={node.signature?.runtime ?? ''}
                onChange={(event) =>
                  onUpdate({
                    signature: {
                      prompt: node.signature?.prompt ?? '',
                      runtime: event.target.value,
                      tools: node.signature?.tools ?? [],
                      temperature: node.signature?.temperature,
                      max_tokens: node.signature?.max_tokens,
                      execution_domain: node.signature?.execution_domain,
                    },
                  })
                }
                className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
              />
            </label>
            <label className="grid gap-1 text-xs">
              <span className="text-slate-400">Temperature</span>
              <input
                type="number"
                step="0.01"
                value={node.signature?.temperature ?? 0}
                onChange={(event) =>
                  onUpdate({
                    signature: {
                      prompt: node.signature?.prompt ?? '',
                      runtime: node.signature?.runtime ?? 'python',
                      tools: node.signature?.tools ?? [],
                      temperature: Number(event.target.value),
                      max_tokens: node.signature?.max_tokens,
                      execution_domain: node.signature?.execution_domain,
                    },
                  })
                }
                className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
              />
            </label>
          </div>
          <label className="grid gap-1 text-xs">
            <span className="text-slate-400">Tools</span>
            <input
              type="text"
              value={(node.signature?.tools ?? []).join(', ')}
              onChange={(event) =>
                onUpdate({
                  signature: {
                    prompt: node.signature?.prompt ?? '',
                    runtime: node.signature?.runtime ?? 'python',
                    tools: event.target.value
                      .split(',')
                      .map((item) => item.trim())
                      .filter(Boolean),
                    temperature: node.signature?.temperature,
                    max_tokens: node.signature?.max_tokens,
                    execution_domain: node.signature?.execution_domain,
                  },
                })
              }
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
              placeholder="comma,separated,tools"
            />
          </label>
        </div>
      )}

      {node.type === 'verifier' && (
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <label className="grid gap-1 text-xs sm:col-span-2">
            <span className="text-slate-400">Command</span>
            <input
              type="text"
              value={node.verifier?.command ?? ''}
              onChange={(event) =>
                onUpdate({
                  verifier: {
                    command: event.target.value,
                    weight: node.verifier?.weight ?? 1,
                    penalty: node.verifier?.penalty,
                    timeout_sec: node.verifier?.timeout_sec,
                    success_signal: node.verifier?.success_signal,
                  },
                })
              }
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
            />
          </label>
          <label className="grid gap-1 text-xs">
            <span className="text-slate-400">Weight</span>
            <input
              type="number"
              step="0.1"
              value={node.verifier?.weight ?? 1}
              onChange={(event) =>
                onUpdate({
                  verifier: {
                    command: node.verifier?.command ?? '',
                    weight: Number(event.target.value),
                    penalty: node.verifier?.penalty,
                    timeout_sec: node.verifier?.timeout_sec,
                    success_signal: node.verifier?.success_signal,
                  },
                })
              }
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
            />
          </label>
          <label className="grid gap-1 text-xs">
            <span className="text-slate-400">Penalty</span>
            <input
              type="number"
              step="0.1"
              value={node.verifier?.penalty ?? ''}
              onChange={(event) =>
                onUpdate({
                  verifier: {
                    command: node.verifier?.command ?? '',
                    weight: node.verifier?.weight ?? 1,
                    penalty: event.target.value ? Number(event.target.value) : undefined,
                    timeout_sec: node.verifier?.timeout_sec,
                    success_signal: node.verifier?.success_signal,
                  },
                })
              }
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
            />
          </label>
          <label className="grid gap-1 text-xs">
            <span className="text-slate-400">Timeout (sec)</span>
            <input
              type="number"
              value={node.verifier?.timeout_sec ?? ''}
              onChange={(event) =>
                onUpdate({
                  verifier: {
                    command: node.verifier?.command ?? '',
                    weight: node.verifier?.weight ?? 1,
                    penalty: node.verifier?.penalty,
                    timeout_sec: event.target.value ? Number(event.target.value) : undefined,
                    success_signal: node.verifier?.success_signal,
                  },
                })
              }
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
            />
          </label>
        </div>
      )}

      {node.type === 'reward' && (
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <label className="grid gap-1 text-xs">
            <span className="text-slate-400">Metric</span>
            <input
              type="text"
              value={node.reward?.metric ?? ''}
              onChange={(event) =>
                onUpdate({
                  reward: {
                    metric: event.target.value,
                    target: node.reward?.target,
                    aggregator: node.reward?.aggregator,
                    window: node.reward?.window,
                    scale: node.reward?.scale,
                  },
                })
              }
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
            />
          </label>
          <label className="grid gap-1 text-xs">
            <span className="text-slate-400">Target</span>
            <input
              type="number"
              step="0.01"
              value={node.reward?.target ?? ''}
              onChange={(event) =>
                onUpdate({
                  reward: {
                    metric: node.reward?.metric ?? '',
                    target: event.target.value ? Number(event.target.value) : undefined,
                    aggregator: node.reward?.aggregator,
                    window: node.reward?.window,
                    scale: node.reward?.scale,
                  },
                })
              }
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
            />
          </label>
          <label className="grid gap-1 text-xs">
            <span className="text-slate-400">Aggregator</span>
            <input
              type="text"
              value={node.reward?.aggregator ?? ''}
              onChange={(event) =>
                onUpdate({
                  reward: {
                    metric: node.reward?.metric ?? '',
                    target: node.reward?.target,
                    aggregator: event.target.value,
                    window: node.reward?.window,
                    scale: node.reward?.scale,
                  },
                })
              }
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
            />
          </label>
          <label className="grid gap-1 text-xs">
            <span className="text-slate-400">Window</span>
            <input
              type="number"
              value={node.reward?.window ?? ''}
              onChange={(event) =>
                onUpdate({
                  reward: {
                    metric: node.reward?.metric ?? '',
                    target: node.reward?.target,
                    aggregator: node.reward?.aggregator,
                    window: event.target.value ? Number(event.target.value) : undefined,
                    scale: node.reward?.scale,
                  },
                })
              }
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
            />
          </label>
        </div>
      )}

      {node.type === 'training' && (
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <label className="grid gap-1 text-xs">
            <span className="text-slate-400">Method</span>
            <input
              type="text"
              value={node.training?.method ?? ''}
              onChange={(event) =>
                onUpdate({
                  training: {
                    method: event.target.value,
                    skill: node.training?.skill ?? '',
                    dataset: node.training?.dataset,
                    schedule: node.training?.schedule,
                    max_steps: node.training?.max_steps,
                    slurm_profile: node.training?.slurm_profile,
                    trigger_metric: node.training?.trigger_metric,
                  },
                })
              }
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
            />
          </label>
          <label className="grid gap-1 text-xs">
            <span className="text-slate-400">Skill</span>
            <input
              type="text"
              value={node.training?.skill ?? ''}
              onChange={(event) =>
                onUpdate({
                  training: {
                    method: node.training?.method ?? '',
                    skill: event.target.value,
                    dataset: node.training?.dataset,
                    schedule: node.training?.schedule,
                    max_steps: node.training?.max_steps,
                    slurm_profile: node.training?.slurm_profile,
                    trigger_metric: node.training?.trigger_metric,
                  },
                })
              }
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
            />
          </label>
          <label className="grid gap-1 text-xs sm:col-span-2">
            <span className="text-slate-400">Dataset Path</span>
            <input
              type="text"
              value={node.training?.dataset ?? ''}
              onChange={(event) =>
                onUpdate({
                  training: {
                    method: node.training?.method ?? '',
                    skill: node.training?.skill ?? '',
                    dataset: event.target.value,
                    schedule: node.training?.schedule,
                    max_steps: node.training?.max_steps,
                    slurm_profile: node.training?.slurm_profile,
                    trigger_metric: node.training?.trigger_metric,
                  },
                })
              }
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
            />
          </label>
        </div>
      )}

      {node.type === 'deployment' && (
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <label className="grid gap-1 text-xs">
            <span className="text-slate-400">Tenant</span>
            <input
              type="text"
              value={node.deployment?.tenant ?? ''}
              onChange={(event) =>
                onUpdate({
                  deployment: {
                    tenant: event.target.value,
                    domain: node.deployment?.domain ?? '',
                    channel: node.deployment?.channel ?? '',
                    strategy: node.deployment?.strategy,
                    autopromote: node.deployment?.autopromote,
                  },
                })
              }
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
            />
          </label>
          <label className="grid gap-1 text-xs">
            <span className="text-slate-400">Domain</span>
            <input
              type="text"
              value={node.deployment?.domain ?? ''}
              onChange={(event) =>
                onUpdate({
                  deployment: {
                    tenant: node.deployment?.tenant ?? '',
                    domain: event.target.value,
                    channel: node.deployment?.channel ?? '',
                    strategy: node.deployment?.strategy,
                    autopromote: node.deployment?.autopromote,
                  },
                })
              }
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
            />
          </label>
          <label className="grid gap-1 text-xs">
            <span className="text-slate-400">Channel</span>
            <input
              type="text"
              value={node.deployment?.channel ?? ''}
              onChange={(event) =>
                onUpdate({
                  deployment: {
                    tenant: node.deployment?.tenant ?? '',
                    domain: node.deployment?.domain ?? '',
                    channel: event.target.value,
                    strategy: node.deployment?.strategy,
                    autopromote: node.deployment?.autopromote,
                  },
                })
              }
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-white focus:border-emerald-500 focus:outline-none"
            />
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-300">
            <input
              type="checkbox"
              checked={Boolean(node.deployment?.autopromote)}
              onChange={(event) =>
                onUpdate({
                  deployment: {
                    tenant: node.deployment?.tenant ?? '',
                    domain: node.deployment?.domain ?? '',
                    channel: node.deployment?.channel ?? '',
                    strategy: node.deployment?.strategy,
                    autopromote: event.target.checked,
                  },
                })
              }
              className="h-4 w-4 rounded border border-slate-600 bg-slate-900 accent-emerald-500"
            />
            Autopromote when rewards improve
          </label>
        </div>
      )}

      {node.type === 'custom' && (
        <label className="mt-3 grid gap-1 text-xs">
          <span className="text-slate-400">Custom JSON</span>
          <textarea
            value={JSON.stringify(node.custom ?? {}, null, 2)}
            onChange={(event) => {
              try {
                const parsed = JSON.parse(event.target.value);
                onUpdate({ custom: parsed });
              } catch {
                // ignore parse errors to avoid locking the textarea
              }
            }}
            rows={4}
            className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none font-mono"
          />
        </label>
      )}
    </div>
  );
};

const statusForWorkflow = (updatedAt?: string) => {
  if (!updatedAt) {
    return { label: 'Unknown', className: 'bg-slate-700/60 text-slate-300' };
  }
  const updated = new Date(updatedAt).getTime();
  const now = Date.now();
  const diffMinutes = (now - updated) / 60000;
  if (Number.isNaN(diffMinutes)) {
    return { label: 'Unknown', className: 'bg-slate-700/60 text-slate-300' };
  }
  if (diffMinutes <= 15) {
    return { label: 'Active', className: 'bg-emerald-600/20 text-emerald-300 border border-emerald-500/40' };
  }
  if (diffMinutes <= 180) {
    return { label: 'Warm', className: 'bg-amber-500/10 text-amber-300 border border-amber-400/40' };
  }
  return { label: 'Stale', className: 'bg-slate-700/60 text-slate-400 border border-slate-600/60' };
};

const WorkflowBuilderPage = () => {
  const queryClient = useQueryClient();
  const [selectedId, setSelectedId] = useState<string>('');
  const [draft, setDraft] = useState<WorkflowDefinition>(emptyWorkflow());
  const [tagsField, setTagsField] = useState('');

  const workflowsQuery = useQuery<WorkflowSummary[]>({
    queryKey: ['workflows'],
    queryFn: api.listWorkflows,
    refetchInterval: 15000,
  });

  const workflowDetailQuery = useQuery({
    queryKey: ['workflow', selectedId],
    enabled: Boolean(selectedId),
    queryFn: async () => api.getWorkflow(selectedId),
  });

  const workflowRunsQuery = useQuery<WorkflowRunRecord[]>({
    queryKey: ['workflow-runs', selectedId],
    enabled: Boolean(selectedId),
    queryFn: async () => api.getWorkflowRuns(selectedId, 25),
    refetchInterval: 15000,
  });

  const runnerHardwareQuery = useQuery<RunnerHardwareSnapshot>({
    queryKey: ['runner-hardware'],
    queryFn: api.getRunnerHardware,
    refetchInterval: 15000,
  });

  useEffect(() => {
    if (workflowDetailQuery.data) {
      const wf = workflowDetailQuery.data;
      setDraft({
        ...wf,
        tags: Array.isArray(wf.tags) ? wf.tags : [],
        nodes: Array.isArray(wf.nodes) ? wf.nodes : [],
        edges: Array.isArray(wf.edges) ? wf.edges : [],
      });
      setTagsField((wf.tags || []).join(', '));
    }
  }, [workflowDetailQuery.data]);

  const saveMutation = useMutation({
    mutationFn: (workflow: WorkflowDefinition) => api.saveWorkflow(workflow),
    onSuccess: (saved) => {
      setDraft({
        ...saved,
        tags: Array.isArray(saved.tags) ? saved.tags : [],
        nodes: Array.isArray(saved.nodes) ? saved.nodes : [],
        edges: Array.isArray(saved.edges) ? saved.edges : [],
      });
      setTagsField((saved.tags || []).join(', '));
      if (saved.id) {
        setSelectedId(saved.id);
      }
      queryClient.invalidateQueries({ queryKey: ['workflows'] });
    },
  });

  const workflowSummaries = workflowsQuery.data ?? [];
  const isSavable = useMemo(() => draft.name.trim().length > 0 && draft.nodes.length > 0 && draft.edges.length > 0, [draft]);
  const workflowRuns = workflowRunsQuery.data ?? [];
  const runnerHardware = runnerHardwareQuery.data;
  const selectedWorkflowName = workflowDetailQuery.data?.name ?? draft.name;

  const updateNodePartial = (id: string, updates: Partial<WorkflowNode>) => {
    setDraft((prev) => ({
      ...prev,
      nodes: prev.nodes.map((node) => (node.id === id ? { ...node, ...updates } : node)),
    }));
  };

  const removeNode = (id: string) => {
    setDraft((prev) => ({
      ...prev,
      nodes: prev.nodes.filter((node) => node.id !== id),
      edges: prev.edges.filter((edge) => edge.source !== id && edge.target !== id),
    }));
  };

  const updateEdge = (id: string, updates: Partial<WorkflowEdge>) => {
    setDraft((prev) => ({
      ...prev,
      edges: prev.edges.map((edge) => (edge.id === id ? { ...edge, ...updates } : edge)),
    }));
  };

  const removeEdge = (id: string) => {
    setDraft((prev) => ({
      ...prev,
      edges: prev.edges.filter((edge) => edge.id !== id),
    }));
  };

  const handleSave = () => {
    saveMutation.mutate(draft);
  };

  const resetWorkflow = () => {
    setSelectedId('');
    setDraft(emptyWorkflow());
    setTagsField('');
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-3xl font-semibold text-white">Workflow Composer</h1>
          <p className="text-sm text-slate-400">
            Visually assemble DSPy agent workflows, then materialize them into the Go supervisor and Rust runner.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            onClick={resetWorkflow}
            className="rounded-lg border border-slate-600 px-4 py-2 text-sm font-medium text-slate-200 hover:border-slate-400"
          >
            New Workflow
          </button>
          <button
            type="button"
            disabled={!isSavable || saveMutation.isPending}
            onClick={handleSave}
            className={`rounded-lg px-4 py-2 text-sm font-semibold text-white ${
              isSavable && !saveMutation.isPending ? 'bg-emerald-600 hover:bg-emerald-500' : 'bg-slate-600 cursor-not-allowed'
            }`}
          >
            {saveMutation.isPending ? 'Saving…' : selectedId ? 'Save Changes' : 'Create Workflow'}
          </button>
        </div>
      </div>

      <div className="grid gap-6 xl:grid-cols-[320px_1fr]">
        <Card
          title="Saved Workflows"
          subtitle="Select to load and edit"
          loading={workflowsQuery.isPending}
          error={workflowsQuery.error instanceof Error ? workflowsQuery.error.message : undefined}
        >
          {workflowSummaries.length === 0 ? (
            <div className="text-sm text-slate-400">No workflows yet. Create one to get started.</div>
          ) : (
            <ul className="space-y-2 text-sm">
              {workflowSummaries.map((wf) => (
                <li key={wf.id}>
                  <button
                    type="button"
                    onClick={() => setSelectedId(wf.id)}
                    className={`w-full rounded-lg border px-3 py-2 text-left transition ${
                      wf.id === selectedId
                        ? 'border-emerald-500 bg-emerald-500/10 text-emerald-200'
                        : 'border-slate-700 bg-slate-900/40 text-slate-200 hover:border-slate-500'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{wf.name}</span>
                      <span className="text-xs text-slate-400">{wf.node_count} nodes</span>
                    </div>
                    <div className="mt-1 flex items-center justify-between text-xs text-slate-500">
                      <span>{formatTime(wf.updated_at)}</span>
                      {(() => {
                        const status = statusForWorkflow(wf.updated_at);
                        return (
                          <span className={`rounded-full px-2 py-0.5 text-[10px] uppercase tracking-wide ${status.className}`}>
                            {status.label}
                          </span>
                        );
                      })()}
                    </div>
                    {wf.tags?.length ? (
                      <div className="mt-1 flex flex-wrap gap-1">
                        {wf.tags.map((tag) => (
                          <span key={`${wf.id}-${tag}`} className="rounded bg-slate-700/70 px-2 py-1 text-[10px] uppercase tracking-wide text-slate-300">
                            {tag}
                          </span>
                        ))}
                      </div>
                    ) : null}
                    <div className="mt-1 text-xs text-slate-500">Tenant: {wf.tenant || '—'}</div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Card>

        <div className="space-y-6">
          <Card title="Workflow Metadata" subtitle="Naming, tenancy, tags">
            <div className="grid gap-4 md:grid-cols-2">
              <label className="grid gap-1 text-sm">
                <span className="text-slate-300">Name</span>
                <input
                  type="text"
                  value={draft.name}
                  onChange={(event) => setDraft((prev) => ({ ...prev, name: event.target.value }))}
                  className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                  placeholder="Workflow name"
                />
              </label>
              <label className="grid gap-1 text-sm">
                <span className="text-slate-300">Tenant</span>
                <input
                  type="text"
                  value={draft.tenant ?? ''}
                  onChange={(event) => setDraft((prev) => ({ ...prev, tenant: event.target.value }))}
                  className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                  placeholder="tenant id"
                />
              </label>
              <label className="grid gap-1 text-sm md:col-span-2">
                <span className="text-slate-300">Description</span>
                <textarea
                  value={draft.description ?? ''}
                  onChange={(event) => setDraft((prev) => ({ ...prev, description: event.target.value }))}
                  rows={3}
                  className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                  placeholder="High-level objective or curriculum"
                />
              </label>
              <label className="grid gap-1 text-sm">
                <span className="text-slate-300">Version</span>
                <input
                  type="text"
                  value={draft.version ?? ''}
                  onChange={(event) => setDraft((prev) => ({ ...prev, version: event.target.value }))}
                  className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                  placeholder="v1"
                />
              </label>
              <label className="grid gap-1 text-sm">
                <span className="text-slate-300">Tags</span>
                <input
                  type="text"
                  value={tagsField}
                  onChange={(event) => {
                    const value = event.target.value;
                    setTagsField(value);
                    setDraft((prev) => ({
                      ...prev,
                      tags: value
                        .split(',')
                        .map((tag) => tag.trim())
                        .filter(Boolean),
                    }));
                  }}
                  className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                  placeholder="comma,separated,tags"
                />
              </label>
              <div className="grid gap-1 text-xs text-slate-400 md:col-span-2">
                <div>Workflow ID: {draft.id ?? 'not yet created'}</div>
                <div>Updated: {formatTime(draft.updated_at)}</div>
              </div>
            </div>
          </Card>

          <Card
            title="Nodes"
            subtitle="Configure the building blocks of your workflow graph"
            actions={
              <div className="flex flex-wrap gap-2">
                {NODE_TYPES.map((type) => (
                  <button
                    key={`add-${type}`}
                    type="button"
                    onClick={() => setDraft((prev) => ({ ...prev, nodes: [...prev.nodes, createNode(type, prev.nodes)] }))}
                    className="rounded-lg border border-slate-600 px-3 py-1 text-xs font-medium text-slate-200 hover:border-emerald-500"
                  >
                    + {TYPE_LABELS[type]}
                  </button>
                ))}
              </div>
            }
          >
            {draft.nodes.length === 0 ? (
              <div className="text-sm text-slate-400">Add nodes to begin composing the workflow.</div>
            ) : (
              <div className="space-y-4">
                {draft.nodes.map((node) => (
                  <NodeCard
                    key={node.id}
                    node={node}
                    onUpdate={(updates) => updateNodePartial(node.id, updates)}
                    onRemove={() => removeNode(node.id)}
                  />
                ))}
              </div>
            )}
          </Card>

          <Card title="Edges" subtitle="Define the data or control flow between nodes">
            <div className="space-y-3">
              {draft.edges.map((edge) => (
                <div key={edge.id} className="grid gap-3 rounded-lg border border-slate-700 bg-slate-900/40 p-3 sm:grid-cols-[1fr_1fr_140px_80px] sm:items-end">
                  <label className="grid gap-1 text-xs">
                    <span className="text-slate-400">Source</span>
                    <select
                      value={edge.source}
                      onChange={(event) => updateEdge(edge.id, { source: event.target.value })}
                      className="rounded border border-slate-700 bg-slate-900 px-2 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                    >
                      <option value="">Select source</option>
                      {draft.nodes.map((node) => (
                        <option key={`edge-${edge.id}-src-${node.id}`} value={node.id}>
                          {node.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="grid gap-1 text-xs">
                    <span className="text-slate-400">Target</span>
                    <select
                      value={edge.target}
                      onChange={(event) => updateEdge(edge.id, { target: event.target.value })}
                      className="rounded border border-slate-700 bg-slate-900 px-2 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                    >
                      <option value="">Select target</option>
                      {draft.nodes.map((node) => (
                        <option key={`edge-${edge.id}-dst-${node.id}`} value={node.id}>
                          {node.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="grid gap-1 text-xs">
                    <span className="text-slate-400">Kind</span>
                    <select
                      value={edge.kind}
                      onChange={(event) => updateEdge(edge.id, { kind: event.target.value as WorkflowEdge['kind'] })}
                      className="rounded border border-slate-700 bg-slate-900 px-2 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                    >
                      <option value="data">Data</option>
                      <option value="control">Control</option>
                      <option value="reward">Reward</option>
                    </select>
                  </label>
                  <div className="flex items-center justify-between gap-2">
                    <input
                      type="text"
                      value={edge.label ?? ''}
                      onChange={(event) => updateEdge(edge.id, { label: event.target.value })}
                      placeholder="Label"
                      className="w-full rounded border border-slate-700 bg-slate-900 px-2 py-2 text-sm text-white focus:border-emerald-500 focus:outline-none"
                    />
                    <button
                      type="button"
                      onClick={() => removeEdge(edge.id)}
                      className="rounded border border-slate-700 px-2 py-1 text-xs uppercase tracking-wide text-slate-300 hover:border-red-500 hover:text-red-300"
                    >
                      Remove
                    </button>
                  </div>
                </div>
              ))}

              <button
                type="button"
                disabled={draft.nodes.length < 1}
                onClick={() => setDraft((prev) => ({ ...prev, edges: [...prev.edges, createEdge(prev)] }))}
                className="rounded-lg border border-slate-600 px-3 py-2 text-sm font-medium text-slate-200 hover:border-emerald-500 disabled:cursor-not-allowed disabled:border-slate-800 disabled:text-slate-500"
              >
                + Add Edge
              </button>
            </div>
          </Card>

          <Card title="Workflow JSON" subtitle="Preview the payload sent to the supervisor">
            <pre className="max-h-72 overflow-auto rounded bg-slate-900/80 p-4 text-xs text-emerald-200">
              {JSON.stringify(draft, null, 2)}
            </pre>
          </Card>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card
          title="Runner Hardware"
          subtitle="Auto-detected resources propagated to workflow payloads"
          isLoading={runnerHardwareQuery.isLoading}
        >
          {runnerHardware ? (
            <div className="space-y-3 text-sm">
              <div className="flex flex-wrap justify-between gap-2">
                <span className="text-slate-400">Hostname</span>
                <span className="font-mono text-slate-100">{runnerHardware.hostname}</span>
              </div>
              <div className="flex flex-wrap justify-between gap-2">
                <span className="text-slate-400">Detected</span>
                <span className="font-mono text-slate-100">{formatTime(runnerHardware.detected_at)}</span>
              </div>
              <div className="rounded border border-slate-700 bg-slate-900/60 p-3">
                <div className="text-xs uppercase tracking-wide text-slate-500">CPU</div>
                <div className="text-sm text-white">{runnerHardware.cpu.brand}</div>
                <div className="text-xs text-slate-400">{runnerHardware.cpu.logical_cores} logical / {runnerHardware.cpu.physical_cores} physical cores</div>
              </div>
              <div className="rounded border border-slate-700 bg-slate-900/60 p-3">
                <div className="text-xs uppercase tracking-wide text-slate-500">Memory</div>
                <div className="text-sm text-white">
                  {(runnerHardware.memory.total_bytes / (1024 ** 3)).toFixed(1)} GB total ·
                  {(runnerHardware.memory.available_bytes / (1024 ** 3)).toFixed(1)} GB free
                </div>
              </div>
              <div className="space-y-2">
                <div className="text-xs uppercase tracking-wide text-slate-500">Accelerators</div>
                {runnerHardware.accelerators.length === 0 ? (
                  <div className="text-xs text-slate-500">No GPUs detected</div>
                ) : (
                  runnerHardware.accelerators.map((accel, index) => (
                    <div key={`${accel.vendor}-${accel.model}-${index}`} className="rounded border border-slate-700 bg-slate-900/60 p-3">
                      <div className="text-sm text-white">{accel.model} ({accel.vendor.toUpperCase()})</div>
                      <div className="text-xs text-slate-400">
                        {accel.count}× · {(accel.memory_mb / 1024).toFixed(1)} GB each
                        {accel.compute_capability ? ` · CC ${accel.compute_capability}` : ''}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          ) : (
            <div className="text-sm text-slate-500">Hardware snapshot unavailable.</div>
          )}
        </Card>

        <Card
          title="Recent Workflow Runs"
          subtitle={selectedId ? `Latest executions for ${selectedWorkflowName}` : 'Select a workflow to view recent runs'}
          isLoading={workflowRunsQuery.isLoading}
        >
          {selectedId ? (
            workflowRuns.length === 0 ? (
              <div className="text-sm text-slate-500">No runs recorded yet.</div>
            ) : (
              <div className="space-y-3">
                {workflowRuns.slice(0, 5).map((run) => (
                  <div key={run.id} className="rounded border border-slate-700 bg-slate-900/60 p-3">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="font-semibold text-white">Run {run.id.slice(0, 8)}</div>
                      <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${
                        run.status === 'succeeded'
                          ? 'bg-emerald-900/50 text-emerald-300'
                          : run.status === 'failed'
                          ? 'bg-red-900/40 text-red-300'
                          : 'bg-slate-800 text-slate-300'
                      }`}>
                        {run.status}
                      </span>
                    </div>
                    <div className="mt-1 text-xs text-slate-400">Started {formatTime(run.created_at)}</div>
                    {run.completed_at && (
                      <div className="text-xs text-slate-500">Finished {formatTime(run.completed_at)}</div>
                    )}
                    <div className="mt-2 space-y-1">
                      {run.steps.slice(0, 3).map((step: WorkflowRunStep) => (
                        <div key={`${run.id}-${step.node_id}`} className="flex justify-between text-xs">
                          <span className="text-slate-300">{step.name}</span>
                          <span className={`font-mono ${step.status === 'succeeded' ? 'text-emerald-300' : step.status === 'failed' ? 'text-red-300' : 'text-slate-400'}`}>
                            {step.status}
                          </span>
                        </div>
                      ))}
                      {run.steps.length > 3 && (
                        <div className="text-xs text-slate-500">+{run.steps.length - 3} more steps…</div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )
          ) : (
            <div className="text-sm text-slate-500">Select a workflow to inspect recent runs.</div>
          )}
        </Card>
      </div>
    </div>
  );
};

export default WorkflowBuilderPage;
