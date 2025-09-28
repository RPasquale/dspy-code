interface StatusPillProps {
  status: string | undefined;
  text?: string;
  className?: string;
}

const normalizeStatus = (status: string | undefined) => {
  if (!status) return 'unknown';
  const normalized = status.toLowerCase();
  if (['healthy', 'running', 'ready', 'active', 'training', 'ok'].some((token) => normalized.includes(token))) {
    return 'success';
  }
  if (['warn', 'degraded', 'paused', 'standby'].some((token) => normalized.includes(token))) {
    return 'warning';
  }
  if (['error', 'fail', 'unreachable', 'down'].some((token) => normalized.includes(token))) {
    return 'error';
  }
  if (['processing', 'loading', 'pending'].some((token) => normalized.includes(token))) {
    return 'processing';
  }
  return 'unknown';
};

const StatusPill = ({ status, text, className = '' }: StatusPillProps) => {
  const kind = normalizeStatus(status);
  
  const getStatusClasses = () => {
    switch (kind) {
      case 'success':
        return 'status-pill status-success';
      case 'warning':
        return 'status-pill status-warning';
      case 'error':
        return 'status-pill status-error';
      case 'processing':
        return 'status-pill status-processing';
      default:
        return 'status-pill bg-slate-500/20 text-slate-300 border border-slate-500/30';
    }
  };

  return (
    <span className={`${getStatusClasses()} ${className}`}>
      <span className="w-2 h-2 rounded-full bg-current mr-2 animate-pulse" />
      {text ?? status ?? 'Unknown'}
    </span>
  );
};

export default StatusPill;
