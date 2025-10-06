interface StatusPillProps {
  status: string | undefined;
  text?: string;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
}

const normalizeStatus = (status: string | undefined) => {
  if (!status) return 'unknown';
  const normalized = status.toLowerCase();
  if (['healthy', 'running', 'ready', 'active', 'training', 'ok'].some((token) => normalized.includes(token))) {
    return 'online';
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
  return 'offline';
};

const StatusPill = ({ 
  status, 
  text, 
  className = '', 
  size = 'md',
  showIcon = true 
}: StatusPillProps) => {
  const kind = normalizeStatus(status);
  
  const sizeClasses = {
    sm: 'px-2 py-1 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-4 py-2 text-base'
  };

  const getStatusClasses = () => {
    switch (kind) {
      case 'online':
        return 'status-pill status-online';
      case 'warning':
        return 'status-pill status-warning';
      case 'error':
        return 'status-pill status-error';
      case 'processing':
        return 'status-pill status-processing';
      default:
        return 'status-pill status-offline';
    }
  };

  return (
    <span className={`${getStatusClasses()} ${sizeClasses[size]} ${className}`}>
      {showIcon && (
        <span className="w-2 h-2 rounded-full bg-current animate-pulse" />
      )}
      <span>{text ?? status ?? 'Unknown'}</span>
    </span>
  );
};

export default StatusPill;
