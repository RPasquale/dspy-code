import { useState } from 'react'

export default function AdminSettingsDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [key, setKey] = useState<string>(() => (typeof localStorage !== 'undefined' ? (localStorage.getItem('ADMIN_KEY') || '') : ''))
  const [status, setStatus] = useState<string>('')
  if (!open) return null
  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(2,6,23,0.55)', display: 'flex', alignItems: 'stretch', justifyContent: 'flex-end', zIndex: 50 }} onClick={onClose}>
      <div style={{ width: 360, background: '#0b1220', borderLeft: '1px solid #1f2937', padding: 16 }} onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ margin: 0, color: '#e5e7eb' }}>Admin Settings</h3>
          <button onClick={onClose} style={btnStyle}>Close</button>
        </div>
        <div style={{ marginTop: 16 }}>
          <div style={{ color: '#9ca3af', marginBottom: 6 }}>X-Admin-Key</div>
          <input type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder="Admin key" style={inputStyle} />
          <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
            <button onClick={() => { try { localStorage.setItem('ADMIN_KEY', key); setStatus('Saved'); setTimeout(() => setStatus(''), 1500) } catch { setStatus('Failed') } }} style={btnStyle}>Save</button>
            <button onClick={async () => {
              try {
                const resp = await fetch('/api/capacity/status', { headers: key ? { 'X-Admin-Key': key } : {} })
                if (resp.status === 403) { setStatus('Forbidden'); return }
                const j = await resp.json().catch(() => ({}))
                setStatus(j?.config ? 'OK' : 'Unknown')
              } catch (e) { setStatus('Error') }
            }} style={btnStyle}>Check</button>
            {status && <span style={{ color: '#93c5fd' }}>{status}</span>}
          </div>
          <p style={{ color: '#9ca3af', marginTop: 12 }}>This key is required for dangerous actions like cleanup and budget changes.</p>
        </div>
      </div>
    </div>
  )
}

const inputStyle: any = { width: '100%', padding: 8, borderRadius: 6, border: '1px solid #233', background: '#0b0f17', color: '#e5e7eb' }
const btnStyle: any = { background: '#1f2937', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '6px 10px', cursor: 'pointer' }

