import { useEffect, useState } from 'react'
import { useMutation, useQuery } from '@tanstack/react-query'
import Card from './Card'
import { api } from '@/api/client'

export default function CapacityPanel() {
  const [key, setKey] = useState<string>(() => localStorage.getItem('ADMIN_KEY') || '')
  useEffect(() => { localStorage.setItem('ADMIN_KEY', key) }, [key])
  const status = useQuery({ queryKey: ['capacity-status', key], queryFn: () => api.getCapacityStatus(key), enabled: !!key, refetchInterval: 10000 })
  const cfg = useQuery({ queryKey: ['capacity-config', key], queryFn: () => api.getCapacityConfig(key), enabled: !!key, refetchInterval: 30000 })
  const [storageGB, setStorageGB] = useState(50)
  const [gpuHPD, setGpuHPD] = useState(2)
  const approveStorage = useMutation({ mutationFn: () => api.capacityApprove({ kind: 'storage_budget_increase', value: storageGB }, key), onSuccess: () => status.refetch() })
  const approveGpu = useMutation({ mutationFn: () => api.capacityApprove({ kind: 'gpu_hours_increase', value: gpuHPD }, key), onSuccess: () => status.refetch() })

  const loading = !!key && (status.isLoading || cfg.isLoading)
  return (
    <Card title="Capacity & Budgets" subtitle="Simulated paywall: storage + GPU hours">
      <div style={{ marginBottom: 10 }}>
        <label style={{ color: '#9ca3af', marginRight: 8 }}>Admin Key</label>
        <input type="password" value={key} onChange={(e) => setKey(e.target.value)} className="input" style={{ width: 240 }} placeholder="X-Admin-Key" />
      </div>
      {!key && (
        <div style={{ color: '#9ca3af' }}>
          Enter admin key to modify budgets. You can also use the standalone admin page.
          <div style={{ marginTop: 6 }}><a href="/admin/capacity" style={{ color: '#93c5fd' }}>Open Capacity Admin →</a></div>
        </div>
      )}
      {key && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
          <div>
            <h4 style={{ color: '#e5e7eb', margin: '6px 0' }}>Budgets</h4>
            {loading ? (
              <div className="skeleton skeleton-block" />
            ) : (
              <div style={{ background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8, color: '#9ca3af' }}>
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{JSON.stringify(cfg.data || status.data?.plan || {}, null, 2)}</pre>
              </div>
            )}
            <h4 style={{ color: '#e5e7eb', margin: '10px 0 6px' }}>Proposals</h4>
            {loading ? (
              <div className="skeleton skeleton-block" style={{ minHeight: 80 }} />
            ) : (
              <div style={{ background: '#0b0f17', border: '1px solid #233', borderRadius: 6, padding: 8, color: '#9ca3af', minHeight: 80 }}>
                <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{JSON.stringify(status.data?.proposals || [], null, 2)}</pre>
              </div>
            )}
          </div>
          <div>
            <h4 style={{ color: '#e5e7eb', margin: '6px 0' }}>Adjustments</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <div>
                <label style={{ color: '#9ca3af' }}>Storage (GB)</label>
                <input type="number" min={10} step={10} value={storageGB} onChange={(e) => setStorageGB(parseInt(e.target.value || '50', 10))} className="input" />
                <button onClick={() => approveStorage.mutate()} disabled={approveStorage.isPending} className="btn">{approveStorage.isPending ? 'Applying…' : 'Propose + Approve'}</button>
              </div>
              <div>
                <label style={{ color: '#9ca3af' }}>GPU Hours / day</label>
                <input type="number" min={1} step={1} value={gpuHPD} onChange={(e) => setGpuHPD(parseInt(e.target.value || '2', 10))} className="input" />
                <button onClick={() => approveGpu.mutate()} disabled={approveGpu.isPending} className="btn">{approveGpu.isPending ? 'Applying…' : 'Propose + Approve'}</button>
              </div>
            </div>
            <p style={{ color: '#9ca3af', marginTop: 10 }}>This simulates moving past the paywall when budgets are insufficient.</p>
          </div>
        </div>
      )}
    </Card>
  )
}

const inputStyle: any = {}
const btnStyle: any = {}
