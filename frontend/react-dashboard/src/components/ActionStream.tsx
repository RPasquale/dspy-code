import React, { useState, useMemo } from 'react';
import { AgentAction, ActionStatistics } from '../hooks/useAgentMonitoring';
import styles from './ActionStream.module.css';

interface ActionStreamProps {
  actions: AgentAction[];
  statistics: ActionStatistics;
}

const ActionStream: React.FC<ActionStreamProps> = ({ actions, statistics }) => {
  const [filter, setFilter] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'timestamp' | 'reward' | 'confidence'>('timestamp');

  const filteredAndSortedActions = useMemo(() => {
    let filtered = actions;
    
    if (filter !== 'all') {
      filtered = actions.filter(action => action.action_type === filter);
    }
    
    return filtered.sort((a, b) => {
      switch (sortBy) {
        case 'timestamp':
          return b.timestamp - a.timestamp;
        case 'reward':
          return b.reward - a.reward;
        case 'confidence':
          return b.confidence - a.confidence;
        default:
          return 0;
      }
    });
  }, [actions, filter, sortBy]);

  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString();
  };

  const getRewardColor = (reward: number) => {
    if (reward >= 0.8) return '#28a745';
    if (reward >= 0.6) return '#ffc107';
    if (reward >= 0.4) return '#fd7e14';
    return '#dc3545';
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return '#28a745';
    if (confidence >= 0.6) return '#ffc107';
    if (confidence >= 0.4) return '#fd7e14';
    return '#dc3545';
  };

  const actionTypes = useMemo(() => {
    const types = new Set(actions.map(action => action.action_type));
    return Array.from(types);
  }, [actions]);

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h3>Action Stream</h3>
        <div className={styles.controls}>
          <select 
            value={filter} 
            onChange={(e) => setFilter(e.target.value)}
            className={styles.filterSelect}
          >
            <option value="all">All Actions</option>
            {actionTypes.map(type => (
              <option key={type} value={type}>{type}</option>
            ))}
          </select>
          <select 
            value={sortBy} 
            onChange={(e) => setSortBy(e.target.value as any)}
            className={styles.sortSelect}
          >
            <option value="timestamp">Sort by Time</option>
            <option value="reward">Sort by Reward</option>
            <option value="confidence">Sort by Confidence</option>
          </select>
        </div>
      </div>

      <div className={styles.statistics}>
        <div className={styles.statItem}>
          <label>Total Actions</label>
          <span className={styles.statValue}>{statistics.total_actions}</span>
        </div>
        <div className={styles.statItem}>
          <label>Avg Reward</label>
          <span className={styles.statValue}>{statistics.avg_reward.toFixed(3)}</span>
        </div>
        <div className={styles.statItem}>
          <label>Avg Confidence</label>
          <span className={styles.statValue}>{statistics.avg_confidence.toFixed(3)}</span>
        </div>
        <div className={styles.statItem}>
          <label>High Reward</label>
          <span className={styles.statValue}>{statistics.high_reward_actions}</span>
        </div>
      </div>

      <div className={styles.actionTypes}>
        <h4>Action Types</h4>
        <div className={styles.typeGrid}>
          {Object.entries(statistics.action_types).map(([type, count]) => (
            <div key={type} className={styles.typeItem}>
              <span className={styles.typeName}>{type}</span>
              <span className={styles.typeCount}>{count}</span>
            </div>
          ))}
        </div>
      </div>

      <div className={styles.actionsList}>
        <h4>Recent Actions ({filteredAndSortedActions.length})</h4>
        <div className={styles.actionsContainer}>
          {filteredAndSortedActions.slice(0, 20).map((action) => (
            <div key={action.action_id} className={styles.actionItem}>
              <div className={styles.actionHeader}>
                <span className={styles.actionType}>{action.action_type}</span>
                <span className={styles.actionTime}>{formatTimestamp(action.timestamp)}</span>
              </div>
              
              <div className={styles.actionMetrics}>
                <div className={styles.metric}>
                  <label>Reward</label>
                  <span 
                    className={styles.metricValue}
                    style={{ color: getRewardColor(action.reward) }}
                  >
                    {action.reward.toFixed(3)}
                  </span>
                </div>
                <div className={styles.metric}>
                  <label>Confidence</label>
                  <span 
                    className={styles.metricValue}
                    style={{ color: getConfidenceColor(action.confidence) }}
                  >
                    {action.confidence.toFixed(3)}
                  </span>
                </div>
                <div className={styles.metric}>
                  <label>Time</label>
                  <span className={styles.metricValue}>{action.execution_time.toFixed(2)}s</span>
                </div>
              </div>
              
              <div className={styles.actionResult}>
                <label>Result</label>
                <div className={styles.resultText}>{action.result_summary}</div>
              </div>
            </div>
          ))}
          
          {filteredAndSortedActions.length === 0 && (
            <div className={styles.emptyState}>
              No actions found for the selected filter.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ActionStream;
