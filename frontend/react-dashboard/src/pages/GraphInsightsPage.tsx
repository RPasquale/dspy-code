import { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import Card from '../components/Card';
import { api } from '../api/client';
import styles from './SystemMapPage.module.css';

const GraphInsightsPage = () => {
  const snapshotsQuery = useQuery({ queryKey: ['graph-snapshots'], queryFn: () => api.getGraphSnapshots(20), staleTime: 60_000 });
  const mctsQuery = useQuery({ queryKey: ['graph-mcts-top'], queryFn: () => api.getGraphMctsTop(12), staleTime: 30_000 });
  const [memoryQuery, setMemoryQuery] = useState<string>('graph memory review');
  const [memoryQueryDraft, setMemoryQueryDraft] = useState<string>('graph memory review');
  const [fromTs, setFromTs] = useState<string>('');
  const [toTs, setToTs] = useState<string>('');

  const diffQuery = useQuery({
    queryKey: ['graph-diff', fromTs, toTs],
    queryFn: () => api.getGraphDiff(fromTs, toTs),
    enabled: Boolean(fromTs && toTs && fromTs !== toTs),
    staleTime: 30_000,
  });

  const patternsQuery = useQuery({ queryKey: ['graph-patterns'], queryFn: () => api.getGraphPatterns('cycles', { maxLength: 6 }), staleTime: 60_000 });
  const memoryReportQuery = useQuery({
    queryKey: ['graph-memory-report', memoryQuery],
    queryFn: () => api.getGraphMemoryReport(memoryQuery, 12),
    staleTime: 30_000,
  });

  const snapshots = snapshotsQuery.data?.snapshots || [];
  const diff = diffQuery.data;
  const mctsNodes = mctsQuery.data?.nodes || [];
  const cycles = (patternsQuery.data?.cycles as string[][]) || [];
  const memoryReport = memoryReportQuery.data;
  const rewardBreakdown = useMemo(() => Object.entries(memoryReport?.reward_breakdown ?? {}), [memoryReport?.reward_breakdown]);
  const recommendedPaths = memoryReport?.recommended_paths ?? [];
  const verifierTargets = memoryReport?.verifier_targets ?? [];

  useMemo(() => {
    if (snapshots.length >= 2) {
      const sorted = [...snapshots].sort((a, b) => a.timestamp - b.timestamp);
      setFromTs(String(sorted[sorted.length - 2].timestamp));
      setToTs(String(sorted[sorted.length - 1].timestamp));
    } else if (snapshots.length === 1) {
      setFromTs(String(snapshots[0].timestamp));
      setToTs(String(snapshots[0].timestamp));
    }
  }, [snapshots.length]);

  const applyMemoryQuery = () => {
    const trimmed = memoryQueryDraft.trim();
    setMemoryQuery(trimmed || 'graph memory review');
  };

  return (
    <div className={styles.wrapper}>
      <h1 className="mb-4 text-3xl font-semibold text-white">Graph Insights</h1>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Snapshots" subtitle="Stored graph snapshots from recent syncs" dense>
          {snapshots.length === 0 ? (
            <div className="text-sm text-slate-400">No snapshots captured yet.</div>
          ) : (
            <table className="w-full text-left text-xs text-slate-300">
              <thead>
                <tr className="text-slate-400">
                  <th className="py-2">Timestamp</th>
                  <th className="py-2">Nodes</th>
                  <th className="py-2">Edges</th>
                </tr>
              </thead>
              <tbody>
                {snapshots.map((snap) => (
                  <tr key={snap.timestamp} className="border-t border-slate-800/60">
                    <td className="py-2">{new Date(snap.timestamp * 1000).toLocaleString()}</td>
                    <td className="py-2">{Object.keys(snap.nodes || {}).length}</td>
                    <td className="py-2">{snap.edges?.length ?? 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
        <Card title="Snapshot Diff" subtitle="Compare two snapshots to inspect structural changes" dense>
          {snapshots.length < 2 ? (
            <div className="text-sm text-slate-400">Need at least two snapshots to compute diff.</div>
          ) : (
            <div className="space-y-3 text-sm text-slate-200">
              <div className="flex flex-wrap gap-3">
                <label className="flex items-center gap-2">
                  <span className="text-slate-400">From</span>
                  <select className={styles.input} value={fromTs} onChange={(e)=>setFromTs(e.target.value)}>
                    {snapshots.map((snap) => (
                      <option key={`diff-a-${snap.timestamp}`} value={snap.timestamp}>{new Date(snap.timestamp * 1000).toLocaleString()}</option>
                    ))}
                  </select>
                </label>
                <label className="flex items-center gap-2">
                  <span className="text-slate-400">To</span>
                  <select className={styles.input} value={toTs} onChange={(e)=>setToTs(e.target.value)}>
                    {snapshots.map((snap) => (
                      <option key={`diff-b-${snap.timestamp}`} value={snap.timestamp}>{new Date(snap.timestamp * 1000).toLocaleString()}</option>
                    ))}
                  </select>
                </label>
              </div>
              {diff ? (
                <div className="grid gap-2 text-xs text-slate-300">
                  <div>
                    <strong>Nodes Added</strong>
                    <div className="mt-1 flex flex-wrap gap-2">{diff.nodes_added.length ? diff.nodes_added.map((n) => <span key={`da-${n}`} className="rounded bg-emerald-500/10 px-2 py-1 text-emerald-300">{n}</span>) : <span className="text-slate-500">—</span>}</div>
                  </div>
                  <div>
                    <strong>Nodes Removed</strong>
                    <div className="mt-1 flex flex-wrap gap-2">{diff.nodes_removed.length ? diff.nodes_removed.map((n) => <span key={`dr-${n}`} className="rounded bg-rose-500/10 px-2 py-1 text-rose-300">{n}</span>) : <span className="text-slate-500">—</span>}</div>
                  </div>
                  <div className="grid gap-4 sm:grid-cols-2">
                    <div>
                      <strong>Edges Added</strong>
                      <ul className="mt-1 space-y-1 max-h-28 overflow-auto text-slate-400">{diff.edges_added.length ? diff.edges_added.map((edge) => <li key={`ea-${edge}`}>{edge}</li>) : <li>—</li>}</ul>
                    </div>
                    <div>
                      <strong>Edges Removed</strong>
                      <ul className="mt-1 space-y-1 max-h-28 overflow-auto text-slate-400">{diff.edges_removed.length ? diff.edges_removed.map((edge) => <li key={`er-${edge}`}>{edge}</li>) : <li>—</li>}</ul>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-xs text-slate-400">Select two different timestamps to compute diff.</div>
              )}
            </div>
          )}
        </Card>
      </div>

      <div className="grid gap-4 mt-6 lg:grid-cols-2">
        <Card title="Top MCTS Priorities" subtitle="High-impact nodes according to Monte Carlo refresh" dense>
          {mctsQuery.isLoading ? (
            <div className="text-sm text-slate-400">Loading…</div>
          ) : mctsNodes.length === 0 ? (
            <div className="text-sm text-slate-400">No priorities computed yet. Run the refresh job to populate data.</div>
          ) : (
            <table className="w-full text-left text-xs text-slate-300">
              <thead>
                <tr className="text-slate-400">
                  <th className="py-2">Node</th>
                  <th className="py-2">Priority</th>
                  <th className="py-2">Language</th>
                  <th className="py-2">Owner</th>
                </tr>
              </thead>
              <tbody>
                {mctsNodes.map((node) => (
                  <tr key={`mcts-${node.id}`} className="border-t border-slate-800/60">
                    <td className="py-2">{node.id}</td>
                    <td className="py-2 text-emerald-300">{node.priority.toFixed(3)}</td>
                    <td className="py-2">{node.language || '—'}</td>
                    <td className="py-2">{node.owner || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>
        <Card title="Graph Patterns" subtitle="Cycles provide hints for tangled dependencies" dense>
          {patternsQuery.isLoading ? (
            <div className="text-sm text-slate-400">Loading…</div>
          ) : cycles.length === 0 ? (
            <div className="text-sm text-slate-400">No cycles detected up to length 6.</div>
          ) : (
            <ul className="space-y-2 text-xs text-slate-300 max-h-48 overflow-auto">
              {cycles.slice(0, 20).map((cycle, idx) => (
                <li key={`cycle-${idx}`}>
                  <span className="text-slate-400">Cycle {idx + 1}:</span> {cycle.join(' → ')}
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <div className="grid gap-4 mt-6">
        <Card title="Graph Memory Summary" subtitle="Fused RAG + memory verifier signals" dense>
          <div className="mb-3 flex flex-wrap gap-3 text-xs text-slate-300">
            <label className="flex items-center gap-2">
              <span className="text-slate-400">Focus query</span>
              <input
                className={styles.input}
                value={memoryQueryDraft}
                onChange={(e) => setMemoryQueryDraft(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    applyMemoryQuery();
                  }
                }}
              />
            </label>
            <button
              type="button"
              className="rounded-md border border-slate-700 bg-slate-900/60 px-3 py-1 text-xs text-slate-200 hover:border-slate-500"
              onClick={applyMemoryQuery}
            >
              Refresh
            </button>
            <span className="ml-auto text-slate-500">Reward: <span className="text-emerald-300">{memoryReport ? memoryReport.reward.toFixed(3) : '—'}</span></span>
          </div>
          {memoryReportQuery.isLoading ? (
            <div className="text-sm text-slate-400">Loading…</div>
          ) : !memoryReport ? (
            <div className="text-sm text-slate-400">No memory fusion data yet.</div>
          ) : (
            <div className="grid gap-3 text-xs text-slate-200">
              <div>
                <strong>Summary</strong>
                <p className="mt-1 whitespace-pre-line text-slate-300">{memoryReport.summary || '—'}</p>
              </div>
              <div className="grid gap-2 md:grid-cols-2">
                <div>
                  <strong>Recommended paths</strong>
                  <ul className="mt-1 space-y-1 max-h-28 overflow-auto">
                    {recommendedPaths.length ? recommendedPaths.map((path) => (
                      <li key={`path-${path}`} className="rounded bg-slate-900/60 px-2 py-1 text-slate-300">{path}</li>
                    )) : <li className="text-slate-500">—</li>}
                  </ul>
                </div>
                <div>
                  <strong>Verifier targets</strong>
                  <ul className="mt-1 space-y-1 max-h-28 overflow-auto">
                    {verifierTargets.length ? verifierTargets.map((target) => (
                      <li key={`vt-${target}`} className="rounded bg-slate-900/60 px-2 py-1 text-slate-300">{target}</li>
                    )) : <li className="text-slate-500">—</li>}
                  </ul>
                </div>
              </div>
              <div>
                <strong>Reward breakdown</strong>
                <div className="mt-1 grid gap-1 sm:grid-cols-2">
                  {rewardBreakdown.length ? rewardBreakdown.slice(0, 8).map(([kind, value]) => (
                    <div key={`rb-${kind}`} className="flex items-center justify-between rounded bg-slate-900/60 px-2 py-1">
                      <span className="text-slate-400">{kind}</span>
                      <span className="text-emerald-300">{value.toFixed(3)}</span>
                    </div>
                  )) : <span className="text-slate-500">No verifier scores yet.</span>}
                </div>
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};

export default GraphInsightsPage;
