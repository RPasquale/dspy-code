import React from 'react'
import { useQuery } from '@tanstack/react-query'
import Card from './Card'
import { api } from '@/api/client'

export default function AgentBrainPanel() {
  const { data, isLoading } = useQuery({ queryKey: ['policy-summary'], queryFn: api.getPolicySummary, refetchInterval: 10000 })
  const tools: Record<string, any> = data?.tools || {}
  const entries = Object.entries(tools).sort((a,b) => ((b[1]?.last24h?.mean||0) - (a[1]?.last24h?.mean||0))).slice(0, 8)
  return (
    <Card title="Agent Brain" subtitle="Policy rules + recent tool rewards">
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
        <div>
          <div style={{ color: '#9ca3af', marginBottom: 6 }}>Top Tools (24h mean, Δ vs 7d)</div>
          {isLoading ? (
            <div className="anim-fade-in" style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr', gap: 6 }}>
              <div className="skeleton skeleton-text" />
              <div className="skeleton skeleton-text" />
              <div className="skeleton skeleton-text" />
              {[...Array(6)].map((_, i) => (
                <React.Fragment key={i}>
                  <div className="skeleton skeleton-text" />
                  <div className="skeleton skeleton-text" />
                  <div className="skeleton skeleton-text" />
                </React.Fragment>
              ))}
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr 1fr', gap: 6 }}>
              <Head text="Tool" /><Head text="μ24h" /><Head text="Δ vs 7d" />
              {entries.map(([k, v]) => (
                <React.Fragment key={k}>
                  <Cell text={k} />
                  <Cell text={fmt(v?.last24h?.mean)} />
                  <Cell text={fmt(v?.delta)} color={(v?.delta||0) >= 0 ? '#34d399' : '#f87171'} />
                </React.Fragment>
              ))}
            </div>
          )}
        </div>
        <div>
          <div style={{ color: '#9ca3af', marginBottom: 6 }}>Rules</div>
          {isLoading ? (
            <div className="skeleton skeleton-block" style={{ maxHeight: 180 }} />
          ) : (
            <div style={{ maxHeight: 180, overflow: 'auto', background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8 }}>
              {(data?.rules || []).length ? (
                (data?.rules || []).map((r: any, i: number) => (
                  <div key={i} style={{ marginBottom: 6 }}>
                    <div style={{ color: '#e5e7eb' }}>{r.regex}</div>
                    {!!(r.prefer_tools?.length) && (<div style={{ color: '#34d399', fontSize: 12 }}>prefer: {r.prefer_tools.join(', ')}</div>)}
                    {!!(r.deny_tools?.length) && (<div style={{ color: '#f87171', fontSize: 12 }}>deny: {r.deny_tools.join(', ')}</div>)}
                  </div>
                ))
              ) : (
                <div style={{ color: '#6b7280' }}>No rules yet.</div>
              )}
            </div>
          )}
        </div>
      </div>
    </Card>
  )
}

function Head({ text }: { text: string }) { return <div style={{ color: '#9ca3af', fontSize: 12 }}>{text}</div> }
function Cell({ text, color }: { text: string; color?: string }) { return <div style={{ color: color || '#e5e7eb', fontSize: 14 }}>{text}</div> }
function fmt(n?: number): string { return (typeof n === 'number' && isFinite(n)) ? n.toFixed(3) : '--' }
