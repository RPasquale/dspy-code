import React from 'react';
import { AgentStatus } from '../hooks/useAgentMonitoring';
import StatusPill from './StatusPill';

interface AgentStatusCardProps {
  status: AgentStatus;
  isConnected: boolean;
  className?: string;
}

const AgentStatusCard: React.FC<AgentStatusCardProps> = ({ status, isConnected, className = '' }) => {
  const getStatusColor = (state: string | undefined) => {
    if (!state) return 'unknown';
    switch (state.toLowerCase()) {
      case 'executing':
      case 'training':
        return 'success';
      case 'analyzing':
      case 'planning':
        return 'processing';
      case 'idle':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'unknown';
    }
  };

  const formatLastActivity = (timestamp: number | null) => {
    if (!timestamp) return 'Never';
    
    const now = Date.now() / 1000;
    const diff = now - timestamp;
    
    if (diff < 60) return `${Math.floor(diff)}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return `${Math.floor(diff / 86400)}d ago`;
  };

  return (
    <div className={`glass rounded-xl p-6 card-hover ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gradient">Agent Status</h3>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-success-500 animate-pulse' : 'bg-error-500'}`} />
          <span className={`text-sm font-medium ${isConnected ? 'text-success-300' : 'text-error-300'}`}>
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-400">State</label>
          <StatusPill 
            status={getStatusColor(status.agent_state)} 
            text={status.agent_state} 
          />
        </div>
        
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-400">Learning</label>
          <StatusPill 
            status={status.is_learning ? 'success' : 'warning'} 
            text={status.is_learning ? 'Active' : 'Inactive'} 
          />
        </div>
        
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-400">Recent Actions</label>
          <span className="text-lg font-semibold text-white">{status.recent_actions_count}</span>
        </div>
        
        <div className="space-y-2">
          <label className="text-sm font-medium text-slate-400">Last Activity</label>
          <span className="text-sm text-slate-300">{formatLastActivity(status.last_activity)}</span>
        </div>
      </div>
      
      {status.current_task && (
        <div className="mb-4 p-4 bg-white/5 rounded-lg border border-white/10">
          <label className="text-sm font-medium text-slate-400 mb-2 block">Current Task</label>
          <div className="text-sm text-slate-200 font-mono bg-slate-800/50 p-3 rounded border">
            {status.current_task}
          </div>
        </div>
      )}
      
      {status.workspace_path && (
        <div className="mb-4 p-4 bg-white/5 rounded-lg border border-white/10">
          <label className="text-sm font-medium text-slate-400 mb-2 block">Workspace</label>
          <div className="text-sm text-slate-300 font-mono bg-slate-800/50 p-3 rounded border truncate">
            {status.workspace_path}
          </div>
        </div>
      )}
      
      {status.active_files && status.active_files.length > 0 && (
        <div className="p-4 bg-white/5 rounded-lg border border-white/10">
          <label className="text-sm font-medium text-slate-400 mb-3 block">
            Active Files ({status.active_files.length})
          </label>
          <div className="space-y-2">
            {status.active_files.slice(0, 3).map((file, index) => (
              <div key={index} className="text-sm text-slate-300 font-mono bg-slate-800/50 p-2 rounded border">
                {file}
              </div>
            ))}
            {status.active_files.length > 3 && (
              <div className="text-sm text-slate-400 italic">
                +{status.active_files.length - 3} more files
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentStatusCard;
