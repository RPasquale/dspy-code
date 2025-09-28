import { FormEvent, useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Bar, Line } from 'react-chartjs-2';
import OptimizationHistory from './OptimizationHistory';
import Card from '../components/Card';
import { api } from '../api/client';
import type { SignatureSummary, OptimizeSignatureRequest } from '../api/types';
import { ensureChartsRegistered } from '../lib/registerCharts';

const SignaturesPage = () => {
  ensureChartsRegistered();
  const queryClient = useQueryClient();
  const [selected, setSelected] = useState<string | null>(null);
  const [optType, setOptType] = useState<string>('gepa');
  const [timeframe, setTimeframe] = useState<'1h'|'24h'|'7d'|'30d'>('24h');
  const [envFilter, setEnvFilter] = useState<string>('');

  const { data: sigList, isFetching } = useQuery({ queryKey: ['signatures'], queryFn: api.getSignatures, refetchInterval: 20000 });
  const { data: detail, refetch: refetchDetail } = useQuery({
    queryKey: ['signature-detail', selected],
    queryFn: () => api.getSignatureDetail(selected || ''),
    enabled: !!selected,
    refetchInterval: 20000
  });
  const { data: schema } = useQuery({
    queryKey: ['signature-schema', selected],
    queryFn: () => api.getSignatureSchema(selected || ''),
    enabled: !!selected,
    refetchInterval: 60000
  });
  const { data: verifiersData } = useQuery({
    queryKey: ['verifiers-for-sigs'],
    queryFn: api.getVerifiers,
    refetchInterval: 30000
  });
  const [verFilter, setVerFilter] = useState<string>('');
  const { data: analytics } = useQuery({
    queryKey: ['signature-analytics', selected, timeframe, envFilter, verFilter],
    queryFn: () => api.getSignatureAnalytics(selected || '', timeframe, envFilter || undefined, verFilter || undefined),
    enabled: !!selected,
    refetchInterval: 20000
  });
  const [feat, setFeat] = useState<any | null>(null);
  const [gepa, setGepa] = useState<any | null>(null);
  const [newSigName, setNewSigName] = useState('');
  const [newSigType, setNewSigType] = useState<'analysis'|'execution'|'coordination'|'verification'|'modification'|'search'>('analysis');
  const [newSigTools, setNewSigTools] = useState('');
  const [newSigDescription, setNewSigDescription] = useState('');

  const updateSig = useMutation({
    mutationFn: api.updateSignature,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['signatures'] });
      refetchDetail();
    }
  });
  const optimize = useMutation({
    mutationFn: (payload: OptimizeSignatureRequest) => api.optimizeSignature(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['signatures'] });
      refetchDetail();
    }
  });

  const createSig = useMutation({
    mutationFn: (payload: { name: string; type?: string; description?: string; tools?: string[]; active?: boolean }) => api.createSignature(payload),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: ['signatures'] });
      if (variables?.name) {
        setSelected(variables.name);
      }
      setNewSigName('');
      setNewSigDescription('');
      setNewSigTools('');
    }
  });

  const deleteSig = useMutation({
    mutationFn: (name: string) => api.deleteSignature(name),
    onSuccess: (_, name) => {
      queryClient.invalidateQueries({ queryKey: ['signatures'] });
      if (selected === name) {
        setSelected(null);
      }
    }
  });

  const handleCreateSignature = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const name = newSigName.trim();
    if (!name || createSig.isPending) return;
    const toolList = newSigTools
      .split(',')
      .map((token) => token.trim())
      .filter(Boolean);
    createSig.mutate({
      name,
      type: newSigType,
      description: newSigDescription.trim() || undefined,
      tools: toolList.length ? toolList : undefined,
      active: true
    });
  };

  const signatures = useMemo(() => (sigList?.signatures ?? []) as SignatureSummary[], [sigList]);

  useEffect(() => {
    if (!selected && signatures.length > 0) setSelected(signatures[0].name);
  }, [selected, signatures]);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-slate-900 mb-2">Signatures</h1>
        <p className="text-slate-600">Manage and monitor DSPy signature performance and optimization</p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <Card title="All Signatures" subtitle={isFetching ? 'Refreshing…' : undefined}>
        <form onSubmit={handleCreateSignature} className="grid grid-cols-1 md:grid-cols-5 gap-3 mb-4">
          <input
            className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-blue-500"
            placeholder="Signature name"
            value={newSigName}
            onChange={(event) => setNewSigName(event.target.value)}
            required
          />
          <select
            className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-blue-500"
            value={newSigType}
            onChange={(event) => setNewSigType(event.target.value as any)}
          >
            {['analysis', 'execution', 'coordination', 'verification', 'modification', 'search'].map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
          <input
            className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-blue-500"
            placeholder="Tools (comma separated)"
            value={newSigTools}
            onChange={(event) => setNewSigTools(event.target.value)}
          />
          <input
            className="rounded-md border border-slate-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:ring-blue-500"
            placeholder="Description"
            value={newSigDescription}
            onChange={(event) => setNewSigDescription(event.target.value)}
          />
          <button
            type="submit"
            disabled={createSig.isPending || !newSigName.trim()}
            className="inline-flex items-center justify-center rounded-md bg-slate-900 px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {createSig.isPending ? 'Adding…' : 'Add Signature'}
          </button>
        </form>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Name</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Type</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Perf</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Success</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Avg RT</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Iter</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider">Active</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-slate-500 uppercase tracking-wider"></th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-slate-200">
                {signatures.map((s) => (
                  <tr key={s.name} className="hover:bg-slate-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-slate-900">{s.name}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{s.type}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{s.performance.toFixed(1)}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{(s.success_rate ?? 0).toFixed(1)}%</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{(s.avg_response_time ?? 0).toFixed(2)}s</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{s.iterations}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">{s.active ? 'Yes' : 'No'}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-slate-500">
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          className="inline-flex items-center px-3 py-1 border border-slate-300 shadow-sm text-sm leading-4 font-medium rounded-md text-slate-700 bg-white hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                          onClick={() => setSelected(s.name)}
                        >
                          View
                        </button>
                        <button
                          type="button"
                          className="inline-flex items-center px-3 py-1 border border-red-200 shadow-sm text-sm leading-4 font-medium rounded-md text-red-700 bg-white hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:cursor-not-allowed disabled:opacity-60"
                          onClick={(event) => { event.stopPropagation(); if (!deleteSig.isPending && window.confirm(`Remove ${s.name}?`)) { deleteSig.mutate(s.name); } }}
                          disabled={deleteSig.isPending && deleteSig.variables === s.name}
                        >
                          {deleteSig.isPending && deleteSig.variables === s.name ? 'Deleting…' : 'Delete'}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
                {signatures.length === 0 && (
                  <tr>
                    <td colSpan={8} className="px-6 py-4 text-center text-sm text-slate-500">No signatures found</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>

        <Card title={selected ? `Details: ${selected}` : 'Details'} subtitle={detail ? `Updated: ${new Date((detail?.metrics?.last_updated || '')).toLocaleString()}` : undefined}>
          {detail ? (
            <div>
              <div className={styles.gridRow}>
                <div className={styles.field}><span>Performance</span><strong>{detail.metrics.performance.toFixed(2)}</strong></div>
                <div className={styles.field}><span>Success</span><strong>{detail.metrics.success_rate.toFixed(1)}%</strong></div>
                <div className={styles.field}><span>Avg RT</span><strong>{detail.metrics.avg_response_time.toFixed(2)}s</strong></div>
                <div className={styles.field}><span>Iterations</span><strong>{detail.metrics.iterations}</strong></div>
                <div className={styles.field}><span>Type</span>
                  <select
                    className={styles.input}
                    value={detail.metrics.type}
                    onChange={(e) => updateSig.mutate({ name: detail.metrics.name, type: e.target.value })}
                  >
                    {['analysis', 'execution', 'coordination', 'verification', 'modification', 'search'].map((t) => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                </div>
                <div className={styles.field}><span>Active</span>
                  <input
                    type="checkbox"
                    checked={detail.metrics.active}
                    onChange={(e) => updateSig.mutate({ name: detail.metrics.name, active: e.target.checked })}
                  />
                </div>
              </div>

              {detail.policy_summary && (
                <div className={styles.gridRow}>
                  <div className={styles.field} style={{ gridColumn: '1 / span 3' }}>
                    <span>Top Tools (24h Δ vs 7d)</span>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 6 }}>
                      {Object.entries(detail.policy_summary.tools || {}).sort((a: any,b: any) => ((b[1]?.last24h?.mean||0) - (a[1]?.last24h?.mean||0))).slice(0, 6).map(([k, v]: any) => (
                        <span key={k} style={{ padding: '2px 8px', borderRadius: 999, background: 'rgba(30,41,59,0.6)', border: '1px solid rgba(148,163,184,0.2)' }}>{k}: {(v?.last24h?.mean ?? 0).toFixed?.(3)} ({(v?.delta ?? 0) >= 0 ? '+' : ''}{(v?.delta ?? 0).toFixed?.(3)})</span>
                      ))}
                    </div>
                  </div>
                  <div className={styles.field} style={{ gridColumn: '4 / span 3' }}>
                    <span>Rule Hits</span>
                    <div style={{ maxHeight: 120, overflow: 'auto', background: 'rgba(2,6,23,0.6)', border: '1px solid rgba(148,163,184,0.2)', borderRadius: 6, padding: 8, marginTop: 6 }}>
                      {(detail.policy_summary.rule_hits || []).slice(0, 10).map((r: any, i: number) => (
                        <div key={i} style={{ marginBottom: 6 }}>
                          <div style={{ color: '#e5e7eb' }}>{r.regex}</div>
                          <div style={{ color: '#9ca3af', fontSize: 12 }}>24h: {r.hits24h} • 7d: {r.hits7d}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              <div style={{ marginTop: 12 }} className={styles.controls}>
                <select className={styles.input} value={optType} onChange={(e) => setOptType(e.target.value)}>
                  <option value="gepa">GEPA</option>
                  <option value="learning_rate">Learning rate</option>
                  <option value="prompt_tuning">Prompt tuning</option>
                </select>
                <button
                  className={styles.button}
                  onClick={() => selected && optimize.mutate({ signature_name: selected, type: optType })}
                  disabled={optimize.isPending}
                >
                  {optimize.isPending ? 'Optimizing…' : 'Optimize'}
                </button>
                <span className={styles.muted} style={{ marginLeft: 8 }}>Analytics window:</span>
                <select className={styles.input} value={timeframe} onChange={(e) => setTimeframe(e.target.value as any)}>
                  <option value="1h">1h</option>
                  <option value="24h">24h</option>
                  <option value="7d">7d</option>
                  <option value="30d">30d</option>
                </select>
                <span className={styles.muted} style={{ marginLeft: 8 }}>Env:</span>
                <select className={styles.input} value={envFilter} onChange={(e) => setEnvFilter(e.target.value)}>
                  <option value="">All</option>
                  <option value="development">development</option>
                  <option value="testing">testing</option>
                  <option value="staging">staging</option>
                  <option value="production">production</option>
                  <option value="local">local</option>
                </select>
                <span className={styles.muted} style={{ marginLeft: 8 }}>Verifier:</span>
                <select className={styles.input} value={verFilter} onChange={(e) => setVerFilter(e.target.value)}>
                  <option value="">All</option>
                  {(analytics?.related_verifiers || []).map((v) => (
                    <option key={v.name} value={v.name}>{v.name}</option>
                  ))}
                </select>
                <button className={styles.button} style={{ marginLeft: 8 }} onClick={async () => {
                  if (!selected) return;
                  try { setFeat(await api.getSignatureFeatureAnalysis(selected, timeframe, envFilter || undefined)); } catch {}
                }}>Compute Feature Direction</button>
                {feat && (
                  <button className={styles.button + ' ' + 'secondary'} style={{ marginLeft: 8 }} onClick={() => {
                    const blob = new Blob([JSON.stringify(feat.direction || [], null, 2)], { type: 'application/json' });
                    const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = `${selected}-direction.json`; a.click(); URL.revokeObjectURL(url);
                  }}>Download Direction</button>
                )}
                <button className={styles.button} style={{ marginLeft: 8 }} onClick={async () => {
                  if (!selected) return; const w = prompt('Window seconds for GEPA effect (default 86400)', '86400') || '86400';
                  try {
                    const resp = await fetch(`/api/signature/gepa-analysis?name=${encodeURIComponent(selected)}&window=${encodeURIComponent(w)}${envFilter?`&env=${encodeURIComponent(envFilter)}`:''}`);
                    setGepa(await resp.json());
                  } catch {}
                }}>Analyze GEPA Effects</button>
              </div>

              {schema && (
                <div className={styles.schemaGrid}>
                  <div>
                    <h3 style={{ margin: '12px 0 6px' }}>Inputs</h3>
                    <table className={styles.table}>
                      <thead><tr><th>Name</th><th>Description</th><th>Default</th></tr></thead>
                      <tbody>
                        {(schema.inputs || []).map((f) => (
                          <tr key={f.name}><td>{f.name}</td><td>{f.desc || '-'}</td><td>{String(f.default ?? '')}</td></tr>
                        ))}
                        {(schema.inputs || []).length === 0 && <tr><td colSpan={3} className={styles.muted}>No inputs detected</td></tr>}
                      </tbody>
                    </table>
                  </div>
                  <div>
                    <h3 style={{ margin: '12px 0 6px' }}>Outputs</h3>
                    <table className={styles.table}>
                      <thead><tr><th>Name</th><th>Description</th><th>Default</th></tr></thead>
                      <tbody>
                        {(schema.outputs || []).map((f) => (
                          <tr key={f.name}><td>{f.name}</td><td>{f.desc || '-'}</td><td>{String(f.default ?? '')}</td></tr>
                        ))}
                        {(schema.outputs || []).length === 0 && <tr><td colSpan={3} className={styles.muted}>No outputs detected</td></tr>}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Optimization History */}
              <OptimizationHistory name={selected} />

              <div className={styles.chart + ' anim-slide-up'}>
                <Line
                  data={{
                    labels: (detail.trend || []).map((d) => new Date(((d as any).timestamp || Date.now()) * 1000).toLocaleTimeString()),
                    datasets: [
                      {
                        label: 'Performance',
                        data: (detail.trend || []).map((d) => Number((d as any).performance_score || 0)),
                        borderColor: '#22c55e',
                        backgroundColor: 'rgba(34,197,94,0.15)',
                        fill: true,
                        tension: 0.35,
                        pointRadius: 0
                      }
                    ]
                  }}
                  options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { display: false }, y: { display: true } } }}
                />
              </div>

              {analytics && (
                <>
                  <div className={styles.chart + ' anim-fade-in'}>
                    <Bar
                      data={{
                        labels: (analytics.related_verifiers || []).map((v) => v.name),
                        datasets: [
                          { label: 'avg verifier score', data: (analytics.related_verifiers || []).map((v) => v.avg_score), backgroundColor: 'rgba(59,130,246,0.4)' },
                          { label: 'count', data: (analytics.related_verifiers || []).map((v) => v.count), backgroundColor: 'rgba(16,185,129,0.35)', yAxisID: 'y1' }
                        ]
                      }}
                      options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } }, scales: { y: { beginAtZero: true }, y1: { beginAtZero: true, position: 'right', grid: { drawOnChartArea: false } } } }}
                    />
                  </div>
                  <div className={styles.controls}>
                    <span className={styles.muted}>Top verifiers:</span>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {(analytics.related_verifiers || []).slice(0, 5).map((v: any) => (
                        <span key={v.name} data-testid={`verifier-chip-${v.name}`} style={{ padding: '2px 8px', borderRadius: 999, background: 'rgba(30,41,59,0.6)', border: '1px solid rgba(148,163,184,0.2)' }}>{v.name}: {(v.avg_score ?? 0).toFixed?.(3) ?? v.avg_score}</span>
                      ))}
                    </div>
                  </div>
                  <div className={styles.chart + ' anim-fade-in'}>
                    <Bar
                      data={{
                        labels: (analytics.reward_summary?.hist?.bins || []).map((_, i) => `${i}`),
                        datasets: [
                          { label: 'reward hist', data: analytics.reward_summary?.hist?.counts || [], backgroundColor: 'rgba(234,179,8,0.5)' }
                        ]
                      }}
                      options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true }, x: { display: false } } }}
                    />
                  </div>
                  <div className={styles.controls}>
                    <span className={styles.muted}>Context keywords:</span>
                    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                      {Object.entries(analytics.context_keywords || {}).map(([k, v]) => (
                        <span key={k} style={{ padding: '2px 8px', borderRadius: 999, background: 'rgba(30,41,59,0.6)', border: '1px solid rgba(148,163,184,0.2)' }}>{k}: {String(v)}</span>
                      ))}
                    </div>
                  </div>
                  {Array.isArray(analytics.top_embeddings) && analytics.top_embeddings.length > 0 && (
                    <div className={styles.chart + ' anim-fade-in'}>
                      <Bar
                        data={() => {
                          const sorted = [...(analytics.top_embeddings || [])].sort((a, b) => (b.avg_reward - a.avg_reward) || (b.count - a.count)).slice(0, 10);
                          return {
                            labels: sorted.map((e) => e.doc_id.slice(0, 8) + '…'),
                            datasets: [
                              { label: 'avg reward', data: sorted.map((e) => e.avg_reward), backgroundColor: 'rgba(34,197,94,0.4)' },
                              { label: 'count', data: sorted.map((e) => e.count), backgroundColor: 'rgba(59,130,246,0.35)', yAxisID: 'y1' }
                            ]
                          };
                        }}
                        options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } }, scales: { y: { beginAtZero: true }, y1: { beginAtZero: true, position: 'right', grid: { drawOnChartArea: false } } } }}
                      />
                    </div>
                  )}
                  {Array.isArray((analytics as any).clusters) && (analytics as any).clusters.length > 0 && (
                    <div className={styles.chart + ' anim-fade-in'}>
                      <Bar
                        data={{
                          labels: (analytics as any).clusters.map((c: any) => `Cluster ${c.id}`),
                          datasets: [
                            { label: 'avg reward', data: (analytics as any).clusters.map((c: any) => c.avg_reward), backgroundColor: 'rgba(234,179,8,0.5)' },
                            { label: 'count', data: (analytics as any).clusters.map((c: any) => c.count), backgroundColor: 'rgba(99,102,241,0.45)', yAxisID: 'y1' }
                          ]
                        }}
                        options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } }, scales: { y: { beginAtZero: true }, y1: { beginAtZero: true, position: 'right', grid: { drawOnChartArea: false } } } }}
                      />
                    </div>
                  )}
                  {analytics.feature_importance && Array.isArray(analytics.feature_importance.top_dims) && analytics.feature_importance.top_dims.length > 0 && (
                    <div className={styles.chart + ' anim-fade-in'}>
                      <Bar
                        data={() => {
                          const td = analytics.feature_importance.top_dims as any[];
                          const labels = td.map((t: any) => `dim ${t.idx}`);
                          const values = td.map((t: any) => t.corr);
                          return {
                            labels,
                            datasets: [
                              { label: 'corr(reward, dim)', data: values, backgroundColor: values.map((v: number) => v >= 0 ? 'rgba(34,197,94,0.5)' : 'rgba(239,68,68,0.5)') }
                            ]
                          };
                        }}
                        options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } }, scales: { y: { beginAtZero: true, suggestedMin: -1.0, suggestedMax: 1.0 } } }}
                      />
                    </div>
                  )}
                  {feat && Array.isArray(feat.top_positive) && (
                    <div className={styles.chart}>
                      <Bar
                        data={() => {
                          const pos = feat.top_positive as any[]; const neg = feat.top_negative as any[];
                          const labels = [...pos.map((t:any)=>`+${t.idx}`), ...neg.map((t:any)=>`-${t.idx}`)];
                          const values = [...pos.map((t:any)=>t.weight), ...neg.map((t:any)=>t.weight)];
                          return { labels, datasets: [{ label: 'regression weight', data: values, backgroundColor: values.map((v:number)=> v>=0?'rgba(59,130,246,0.5)':'rgba(239,68,68,0.5)')}] };
                        }}
                        options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } }, scales: { y: { beginAtZero: false } } }}
                      />
                    </div>
                  )}

                  {gepa && !gepa.error && (
                    <div className={styles.chart}>
                      <Bar
                        data={{
                          labels: ['pre', 'post'],
                          datasets: [
                            { label: 'avg reward', data: [gepa.pre?.avg_reward || 0, gepa.post?.avg_reward || 0], backgroundColor: ['rgba(148,163,184,0.4)', 'rgba(34,197,94,0.5)'] }
                          ]
                        }}
                        options={{ responsive: true, maintainAspectRatio: false, plugins: { legend: { position: 'bottom' } }, scales: { y: { beginAtZero: true } } }}
                      />
                      <div className={styles.controls}>
                        <span className={styles.muted}>Verifier deltas:</span>
                        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                          {(gepa.delta?.verifiers || []).slice(0, 10).map((v: any) => (
                            <span key={v.name} style={{ padding: '2px 8px', borderRadius: 999, background: 'rgba(30,41,59,0.6)', border: '1px solid rgba(148,163,184,0.2)' }}>{v.name}: {v.delta?.toFixed?.(3) ?? v.delta}</span>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          ) : (
            <p className="text-slate-500 text-center py-8">Select a signature to inspect</p>
          )}
        </Card>
      </div>
    </div>
  );
};

export default SignaturesPage;
