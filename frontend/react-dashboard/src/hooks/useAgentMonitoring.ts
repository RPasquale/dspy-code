import { useState, useEffect, useCallback } from 'react';
import { useWebSocket, WebSocketMessage } from './useWebSocket';

export interface AgentAction {
  action_id: string;
  timestamp: number;
  action_type: string;
  reward: number;
  confidence: number;
  execution_time: number;
  result_summary: string;
  environment: string;
}

export interface ActionStatistics {
  total_actions: number;
  avg_reward: number;
  avg_confidence: number;
  avg_execution_time: number;
  high_reward_actions: number;
  action_types: Record<string, number>;
}

export interface AgentStatus {
  agent_state: string;
  current_task: string | null;
  workspace_path: string | null;
  active_files: string[];
  system_health: any;
  recent_actions_count: number;
  is_learning: boolean;
  last_activity: number | null;
}

export interface LearningMetrics {
  training_sessions: number;
  learning_progress: any;
  learning_trends: {
    training_accuracy: { current: number; trend: string };
    validation_accuracy: { current: number; trend: string };
    loss: { current: number; trend: string };
  };
  signature_performance: Record<string, any>;
  retrieval_statistics: any;
  active_signatures: number;
}

export interface SystemMetrics {
  cache_performance: any;
  log_statistics: {
    total_logs: number;
    error_count: number;
    warning_count: number;
    info_count: number;
    recent_errors: string[];
  };
  system_health: any;
  uptime: number;
  memory_usage: string;
  cpu_usage: string;
}

export interface AgentMonitoringData {
  actions: AgentAction[];
  actionStatistics: ActionStatistics;
  agentStatus: AgentStatus;
  learningMetrics: LearningMetrics;
  systemMetrics: SystemMetrics;
  lastUpdate: number;
}

export const useAgentMonitoring = () => {
  const [data, setData] = useState<AgentMonitoringData>({
    actions: [],
    actionStatistics: {
      total_actions: 0,
      avg_reward: 0,
      avg_confidence: 0,
      avg_execution_time: 0,
      high_reward_actions: 0,
      action_types: {}
    },
    agentStatus: {
      agent_state: 'unknown',
      current_task: null,
      workspace_path: null,
      active_files: [],
      system_health: null,
      recent_actions_count: 0,
      is_learning: false,
      last_activity: null
    },
    learningMetrics: {
      training_sessions: 0,
      learning_progress: {},
      learning_trends: {
        training_accuracy: { current: 0, trend: 'stable' },
        validation_accuracy: { current: 0, trend: 'stable' },
        loss: { current: 0, trend: 'stable' }
      },
      signature_performance: {},
      retrieval_statistics: {},
      active_signatures: 0
    },
    systemMetrics: {
      cache_performance: {},
      log_statistics: {
        total_logs: 0,
        error_count: 0,
        warning_count: 0,
        info_count: 0,
        recent_errors: []
      },
      system_health: null,
      uptime: 0,
      memory_usage: '0GB',
      cpu_usage: '0%'
    },
    lastUpdate: 0
  });

  const wsUrl = (() => {
    try {
      const envUrl = (import.meta as any)?.env?.VITE_WS_BASE_URL as string | undefined;
      if (envUrl) return envUrl;
      if (typeof window !== 'undefined') {
        const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
        // Route via nginx to reach upstream WS
        return `${proto}://${window.location.host}/ws/`;
      }
    } catch {}
    return 'ws://localhost:8081/';
  })();
  const { isConnected, lastMessage, sendMessage, error } = useWebSocket(wsUrl);

  const handleMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'actions_update':
        setData(prev => ({
          ...prev,
          actions: message.data.actions || [],
          actionStatistics: message.data.statistics || prev.actionStatistics,
          lastUpdate: message.timestamp
        }));
        break;
      
      case 'status_update':
        setData(prev => ({
          ...prev,
          agentStatus: message.data,
          lastUpdate: message.timestamp
        }));
        break;
      
      case 'learning_update':
        setData(prev => ({
          ...prev,
          learningMetrics: message.data,
          lastUpdate: message.timestamp
        }));
        break;
      
      case 'system_update':
        setData(prev => ({
          ...prev,
          systemMetrics: message.data,
          lastUpdate: message.timestamp
        }));
        break;
      
      default:
        console.log('Unknown message type:', message.type);
    }
  }, []);

  useEffect(() => {
    if (lastMessage) {
      handleMessage(lastMessage);
    }
  }, [lastMessage, handleMessage]);

  const requestData = useCallback((dataType: string) => {
    sendMessage({
      type: 'get_data',
      data_type: dataType
    });
  }, [sendMessage]);

  const subscribe = useCallback((subscriptions: string[]) => {
    sendMessage({
      type: 'subscribe',
      subscriptions
    });
  }, [sendMessage]);

  const ping = useCallback(() => {
    sendMessage({
      type: 'ping'
    });
  }, [sendMessage]);

  return {
    data,
    isConnected,
    error,
    requestData,
    subscribe,
    ping
  };
};
