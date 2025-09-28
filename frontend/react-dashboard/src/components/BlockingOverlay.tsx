export default function BlockingOverlay({ reason }: { reason: string }) {
  return (
    <div style={{ position: 'absolute', inset: 0, background: 'rgba(2,6,23,0.75)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10, borderRadius: 8 }}>
      <div style={{ background: 'rgba(15,23,42,0.95)', border: '1px solid rgba(239,68,68,0.4)', boxShadow: '0 8px 24px rgba(0,0,0,0.5)', padding: 16, borderRadius: 10, width: 520, color: '#fecaca' }}>
        <h3 style={{ margin: '0 0 6px 0', color: '#fca5a5' }}>Action Blocked</h3>
        <p style={{ margin: 0 }}>{reason}</p>
        <p style={{ margin: '8px 0 0 0', color: '#9ca3af' }}>Use Cleanup Helpers (Overview) or adjust budgets (Capacity) to proceed.</p>
        <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
          <a href="/dashboard" style={btnStyle}>Open Overview</a>
          <a href="/admin/capacity" style={btnStyle}>Capacity Admin</a>
        </div>
      </div>
    </div>
  )
}

const btnStyle: any = { background: '#1f2937', color: '#e5e7eb', border: '1px solid #374151', borderRadius: 6, padding: '8px 12px', cursor: 'pointer', textDecoration: 'none' }

