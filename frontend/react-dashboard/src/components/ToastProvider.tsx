import React, { createContext, useCallback, useContext, useMemo, useState } from 'react'

type ToastKind = 'ok' | 'warn' | 'err'
type Toast = { id: string; text: string; kind: ToastKind; ttl?: number }

type ToastContextValue = {
  notify: (text: string, kind?: ToastKind, ttlMs?: number) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [items, setItems] = useState<Toast[]>([])

  const notify = useCallback((text: string, kind: ToastKind = 'ok', ttlMs = 3200) => {
    const id = `${Date.now()}_${Math.random().toString(36).slice(2,8)}`
    const t: Toast = { id, text, kind, ttl: ttlMs }
    setItems((prev) => [...prev, t].slice(-6))
    window.setTimeout(() => {
      setItems((prev) => prev.filter((x) => x.id !== id))
    }, ttlMs)
  }, [])

  const value = useMemo(() => ({ notify }), [notify])

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div style={{ position: 'fixed', right: 16, top: 16, zIndex: 60, display: 'flex', flexDirection: 'column', gap: 8 }}>
        {items.map((t) => (
          <div key={t.id} className={`tag anim-fade-in`} style={{ borderColor: 'rgba(148,163,184,0.35)', background: t.kind==='ok' ? 'rgba(16,185,129,0.15)' : (t.kind==='warn' ? 'rgba(245,158,11,0.15)' : 'rgba(239,68,68,0.15)'), color: t.kind==='ok' ? '#86efac' : (t.kind==='warn' ? '#fbbf24' : '#fecaca') }}>
            {t.text}
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error('useToast must be used within ToastProvider')
  return ctx
}

