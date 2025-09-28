import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import Card from '../components/Card';
import { api } from '../api/client';
import styles from './SystemMapPage.module.css';

const SignatureGraphPage = () => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [timeframe, setTimeframe] = useState<'1h'|'24h'|'7d'|'30d'>('24h');
  const [env, setEnv] = useState<string>('');
  const [graph, setGraph] = useState<{ nodes: { id: string; type: 'signature'|'verifier' }[]; edges: { source: string; target: string; avg: number; count: number }[]; edges_sig_sig?: { a: string; b: string; count: number }[] } | null>(null);
  const [minReward, setMinReward] = useState<string>('');
  const [verifier, setVerifier] = useState<string>('');
  const [showSigVer, setShowSigVer] = useState(true);
  const [showSigSig, setShowSigSig] = useState(true);

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
    </div>
  );
};

export default SignatureGraphPage;
