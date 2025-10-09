import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import Card from '../components/Card';
import { api } from '../api/client';
import styles from './SystemMapPage.module.css';
import type { GraphSnapshot, GraphDiffResponse, GraphMctsNode } from '../api/types';

const SignatureGraphPage = () => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [timeframe, setTimeframe] = useState<'1h'|'24h'|'7d'|'30d'>('24h');
  const [env, setEnv] = useState<string>('');
  const [graph, setGraph] = useState<{ nodes: { id: string; type: 'signature'|'verifier' }[]; edges: { source: string; target: string; avg: number; count: number }[]; edges_sig_sig?: { a: string; b: string; count: number }[] } | null>(null);
  const [minReward, setMinReward] = useState<string>('');
  const [verifier, setVerifier] = useState<string>('');
  const [showSigVer, setShowSigVer] = useState(true);
  const [showSigSig, setShowSigSig] = useState(true);
  const [snapshots, setSnapshots] = useState<GraphSnapshot[]>([]);
  const [snapshotA, setSnapshotA] = useState<string>('');
  const [snapshotB, setSnapshotB] = useState<string>('');
  const [snapshotDiff, setSnapshotDiff] = useState<GraphDiffResponse | null>(null);
  const [mctsTop, setMctsTop] = useState<GraphMctsNode[]>([]);
  const [mixedLanguageNodes, setMixedLanguageNodes] = useState<string[]>([]);

  const load = async () => {
    try {
      const params = new URLSearchParams();
      params.set('timeframe', timeframe);
      if (env) params.set('env', env);
      if (minReward) params.set('min_reward', minReward);
      if (verifier) params.set('verifier', verifier);
      const resp = await fetch(`/api/signature/graph?${params.toString()}`);
      if (resp.ok) setGraph(await resp.json());
    } catch {}
  };
  useEffect(() => { load(); }, [timeframe, env]);

  useEffect(() => {
    (async () => {
      try {
        const res = await api.getGraphSnapshots(10);
        const list = res.snapshots || [];
        setSnapshots(list);
        if (list.length >= 2) {
          setSnapshotA(String(list[list.length - 2].timestamp));
          setSnapshotB(String(list[list.length - 1].timestamp));
        } else if (list.length === 1) {
          const ts = String(list[0].timestamp);
          setSnapshotA(ts);
          setSnapshotB(ts);
        }
      } catch {}
    })();
    (async () => {
      try {
        const res = await api.getGraphMctsTop(8);
        setMctsTop(res.nodes || []);
      } catch {
        setMctsTop([]);
      }
    })();
    (async () => {
      try {
        const res = await api.getGraphPatterns('mixed-language');
        setMixedLanguageNodes((res.nodes as string[]) || []);
      } catch {}
    })();
  }, []);

  useEffect(() => {
    const fetchDiff = async () => {
      if (!snapshotA || !snapshotB || snapshotA === snapshotB) {
        setSnapshotDiff(null);
        return;
      }
      try {
        const diff = await api.getGraphDiff(snapshotA, snapshotB);
        setSnapshotDiff(diff);
      } catch {
        setSnapshotDiff(null);
      }
    };
    fetchDiff();
  }, [snapshotA, snapshotB]);

  useEffect(() => {
    if (!svgRef.current || !graph) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    const width = svgRef.current.clientWidth || 800; const height = 500;
    svg.attr('viewBox', `0 0 ${width} ${height}`);
    const nodes = graph.nodes.map((n) => ({ ...n }));
    const links = (showSigVer ? graph.edges.map((e) => ({ ...e, kind: 'sig-ver' })) : []) as any[];
    const color = (d: any) => (d.type === 'signature' ? '#60a5fa' : '#34d399');
    const radius = (d: any) => (d.type === 'signature' ? 8 : 6);
    const sim = d3.forceSimulation(nodes as any)
      .force('link', d3.forceLink(links as any).id((d: any) => d.id).distance((d: any) => 60 + (10 - Math.min(10, (d.count || 1)))) )
      .force('charge', d3.forceManyBody().strength(-180))
      .force('center', d3.forceCenter(width/2, height/2));

    // add co-occurrence edges between signatures (faint)
    const linksSS = (showSigSig && graph.edges_sig_sig ? graph.edges_sig_sig.map((e) => ({ source: e.a, target: e.b, count: e.count, kind: 'sig-sig' })) : []) as any[];
    const link = svg.append('g').selectAll('line').data([...links, ...linksSS] as any).enter().append('line')
      .attr('stroke', (d: any) => d.kind === 'sig-sig' ? 'rgba(148,163,184,0.25)' : 'rgba(148,163,184,0.6)')
      .attr('stroke-width', (d: any) => d.kind === 'sig-sig' ? 1 : (1 + Math.min(4, (d.count || 1) / 10)))
      .append('title').text((d: any) => d.kind === 'sig-sig' ? `co-occur: ${d.count}` : `avg: ${d.avg?.toFixed?.(3) ?? d.avg} | count: ${d.count}`);

    const node = svg.append('g').selectAll('circle').data(nodes as any).enter().append('circle')
      .attr('r', radius as any)
      .attr('fill', color as any)
      .call(d3.drag<SVGCircleElement, any>()
        .on('start', (event, d: any) => { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag', (event, d: any) => { d.fx = event.x; d.fy = event.y; })
        .on('end', (event, d: any) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      )
      .append('title').text((d: any) => `${d.type}: ${d.id}`);
    const label = svg.append('g').selectAll('text').data(nodes as any).enter().append('text')
      .text((d: any) => d.id)
      .attr('font-size', 10)
      .attr('fill', 'rgba(226,232,240,0.85)');

    sim.on('tick', () => {
      link.attr('x1', (d: any) => d.source.x).attr('y1', (d: any) => d.source.y).attr('x2', (d: any) => d.target.x).attr('y2', (d: any) => d.target.y);
      node.attr('cx', (d: any) => d.x).attr('cy', (d: any) => d.y);
      label.attr('x', (d: any) => (d.x + 8)).attr('y', (d: any) => (d.y + 3));
    });
  }, [graph]);

  return (
    <div className={styles.wrapper}>
      <h1>Signature Graph</h1>
      <div className={styles.controls} style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
        <span className={styles.muted}>Timeframe</span>
        <select className={styles.input} value={timeframe} onChange={(e) => setTimeframe(e.target.value as any)}>
          <option value="1h">1h</option>
          <option value="24h">24h</option>
          <option value="7d">7d</option>
          <option value="30d">30d</option>
        </select>
        <span className={styles.muted}>Env</span>
        <select className={styles.input} value={env} onChange={(e) => setEnv(e.target.value)}>
          <option value="">All</option>
          <option value="development">development</option>
          <option value="testing">testing</option>
          <option value="staging">staging</option>
          <option value="production">production</option>
          <option value="local">local</option>
        </select>
        <button className={styles.button} onClick={load}>Refresh</button>
        <span className={styles.muted}>Min reward</span>
        <input className={styles.input} value={minReward} onChange={(e)=>setMinReward(e.target.value)} placeholder="e.g., 0.5" style={{ width: 100 }} />
        <span className={styles.muted}>Verifier</span>
        <input className={styles.input} value={verifier} onChange={(e)=>setVerifier(e.target.value)} placeholder="filter by verifier" style={{ width: 200 }} />
        <button className={styles.button} onClick={async ()=>{
          const params = new URLSearchParams();
          params.set('timeframe', timeframe);
          if (env) params.set('env', env);
          if (minReward) params.set('min_reward', minReward);
          if (verifier) params.set('verifier', verifier);
          params.set('download', '1');
          window.open(`/api/signature/graph?${params.toString()}`, '_blank');
        }}>Export JSON</button>
      </div>
      <Card title="Graph" subtitle="Signatures (blue), verifiers (green); edge width ~ count; signature co-occurrence edges shown as faint lines" dense>
        <svg ref={svgRef} className={styles.canvas} style={{ width: '100%', height: 520 }} />
      </Card>
      <div className={styles.controls} style={{ display: 'flex', gap: 8, marginTop: 8 }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <input type="checkbox" checked={showSigVer} onChange={(e)=>setShowSigVer(e.target.checked)} />
          <span className={styles.muted}>Show Sig→Verifier</span>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <input type="checkbox" checked={showSigSig} onChange={(e)=>setShowSigSig(e.target.checked)} />
          <span className={styles.muted}>Show Sig↔Sig Co-occurrence</span>
        </label>
      </div>

      <div className="grid gap-4 lg:grid-cols-2 mt-8">
        <Card title="Graph Snapshot Diff" subtitle="Compare two stored snapshots to see structural changes" dense>
          {snapshots.length === 0 ? (
            <div className="text-sm text-slate-400">No snapshots captured yet.</div>
          ) : (
            <div className="space-y-3 text-sm text-slate-200">
              <div className="flex flex-wrap gap-3">
                <label className="flex items-center gap-2">
                  <span className="text-slate-400">From</span>
                  <select className={styles.input} value={snapshotA} onChange={(e)=>setSnapshotA(e.target.value)}>
                    {snapshots.map((snap) => (
                      <option key={snap.timestamp} value={snap.timestamp}>
                        {new Date(snap.timestamp * 1000).toLocaleString()}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="flex items-center gap-2">
                  <span className="text-slate-400">To</span>
                  <select className={styles.input} value={snapshotB} onChange={(e)=>setSnapshotB(e.target.value)}>
                    {snapshots.map((snap) => (
                      <option key={snap.timestamp} value={snap.timestamp}>
                        {new Date(snap.timestamp * 1000).toLocaleString()}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              {snapshotDiff ? (
                <div className="grid gap-2 text-xs sm:text-sm text-slate-300">
                  <div>
                    <strong>Nodes Added</strong>
                    <div className="flex flex-wrap gap-2 mt-1">{snapshotDiff.nodes_added.length ? snapshotDiff.nodes_added.map((n) => <span key={`na-${n}`} className="rounded bg-emerald-500/10 px-2 py-1 text-emerald-300">{n}</span>) : <span className="text-slate-500">—</span>}</div>
                  </div>
                  <div>
                    <strong>Nodes Removed</strong>
                    <div className="flex flex-wrap gap-2 mt-1">{snapshotDiff.nodes_removed.length ? snapshotDiff.nodes_removed.map((n) => <span key={`nr-${n}`} className="rounded bg-rose-500/10 px-2 py-1 text-rose-300">{n}</span>) : <span className="text-slate-500">—</span>}</div>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <strong>Edges Added</strong>
                      <ul className="mt-1 space-y-1 text-slate-400 max-h-32 overflow-auto">
                        {snapshotDiff.edges_added.length ? snapshotDiff.edges_added.map((e) => <li key={`ea-${e}`}>{e}</li>) : <li>—</li>}
                      </ul>
                    </div>
                    <div>
                      <strong>Edges Removed</strong>
                      <ul className="mt-1 space-y-1 text-slate-400 max-h-32 overflow-auto">
                        {snapshotDiff.edges_removed.length ? snapshotDiff.edges_removed.map((e) => <li key={`er-${e}`}>{e}</li>) : <li>—</li>}
                      </ul>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="text-xs text-slate-400">Select two different snapshots to view a diff.</div>
              )}
            </div>
          )}
        </Card>
        <Card title="Top MCTS Priorities" subtitle="Nodes the agent currently considers high-impact" dense>
          {mctsTop.length === 0 ? (
            <div className="text-sm text-slate-400">No priorities computed yet.</div>
          ) : (
            <table className="w-full text-xs sm:text-sm text-left text-slate-300">
              <thead>
                <tr className="text-slate-400">
                  <th className="py-1">Node</th>
                  <th className="py-1">Priority</th>
                  <th className="py-1">Language</th>
                  <th className="py-1">Owner</th>
                </tr>
              </thead>
              <tbody>
                {mctsTop.map((node) => (
                  <tr key={node.id} className="border-t border-slate-800/60">
                    <td className="py-1">{node.id}</td>
                    <td className="py-1 text-emerald-300">{node.priority.toFixed(3)}</td>
                    <td className="py-1">{node.language || '—'}</td>
                    <td className="py-1">{node.owner || '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {mixedLanguageNodes.length > 0 && (
            <div className="mt-3 text-xs text-slate-400">
              <strong>Mixed-language neighbors:</strong>
              <div className="mt-1 flex flex-wrap gap-2">
                {mixedLanguageNodes.map((node) => (
                  <span key={`mixed-${node}`} className="rounded bg-indigo-500/10 px-2 py-1 text-indigo-200">{node}</span>
                ))}
              </div>
            </div>
          )}
        </Card>
      </div>
    </div>
  );
};

export default SignatureGraphPage;
