import React, { useState, useEffect } from 'react';
import { useAgentMonitoring } from '../hooks/useAgentMonitoring';
import AgentStatusCard from '../components/AgentStatusCard';
import ActionStream from '../components/ActionStream';
import LearningProgress from '../components/LearningProgress';
import styles from './RealtimeMonitoringPage.module.css';

const RealtimeMonitoringPage: React.FC = () => {
  const { data, isConnected, error, requestData, subscribe } = useAgentMonitoring();
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);

  useEffect(() => {
    // Subscribe to all data types
    subscribe(['actions', 'status', 'learning', 'system']);
  }, [subscribe]);

  useEffect(() => {
    if (data.lastUpdate) {
      setLastUpdate(new Date(data.lastUpdate * 1000));
    }
  }, [data.lastUpdate]);

  const handleRefresh = () => {
    requestData('actions');
    requestData('status');
    requestData('learning');
    requestData('system');
  };

  const formatLastUpdate = () => {
    if (!lastUpdate) return 'Never';
    
    const now = new Date();
    const diff = now.getTime() - lastUpdate.getTime();
    
    if (diff < 1000) return 'Just now';
    if (diff < 60000) return `${Math.floor(diff / 1000)}s ago`;
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    return lastUpdate.toLocaleTimeString();
  };

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <div className={styles.headerContent}>
          <h1>Real-Time Agent Monitoring</h1>
          <p>Watch your DSPy agent work, learn, and improve in real-time</p>
        </div>
        
        <div className={styles.headerControls}>
          <div className={styles.connectionStatus}>
            <div className={`${styles.connectionDot} ${isConnected ? styles.connected : styles.disconnected}`} />
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
            {error && <span className={styles.error}>Error: {error}</span>}
          </div>
          
          <div className={styles.updateInfo}>
            <span>Last update: {formatLastUpdate()}</span>
            <button 
              className={styles.refreshButton}
              onClick={handleRefresh}
              disabled={!isConnected}
            >
              🔄 Refresh
            </button>
          </div>
        </div>
      </div>

      <div className={styles.monitoringGrid}>
        {/* Agent Status - Top Left */}
        <div className={styles.statusSection}>
          <AgentStatusCard 
            status={data.agentStatus} 
            isConnected={isConnected}
          />
        </div>

        {/* Action Stream - Top Right */}
        <div className={styles.actionsSection}>
          <ActionStream 
            actions={data.actions}
            statistics={data.actionStatistics}
          />
        </div>

        {/* Learning Progress - Bottom Left */}
        <div className={styles.learningSection}>
          <LearningProgress metrics={data.learningMetrics} />
        </div>

        {/* System Metrics - Bottom Right */}
        <div className={styles.systemSection}>
          <div className={styles.systemCard}>
            <div className={styles.cardHeader}>
              <h3>System Metrics</h3>
              <div className={styles.healthStatus}>
                <div className={`${styles.healthDot} ${data.systemMetrics.system_health ? styles.healthy : styles.unhealthy}`} />
                <span>{data.systemMetrics.system_health ? 'Healthy' : 'Unknown'}</span>
              </div>
            </div>

            <div className={styles.systemGrid}>
              <div className={styles.systemMetric}>
                <label>Memory Usage</label>
                <span className={styles.systemValue}>{data.systemMetrics.memory_usage}</span>
              </div>
              <div className={styles.systemMetric}>
                <label>CPU Usage</label>
                <span className={styles.systemValue}>{data.systemMetrics.cpu_usage}</span>
              </div>
              <div className={styles.systemMetric}>
                <label>Uptime</label>
                <span className={styles.systemValue}>
                  {Math.floor(data.systemMetrics.uptime / 3600)}h {Math.floor((data.systemMetrics.uptime % 3600) / 60)}m
                </span>
              </div>
              <div className={styles.systemMetric}>
                <label>Total Logs</label>
                <span className={styles.systemValue}>{data.systemMetrics.log_statistics.total_logs}</span>
              </div>
            </div>

            <div className={styles.logStats}>
              <h4>Log Statistics</h4>
              <div className={styles.logGrid}>
                <div className={styles.logStat}>
                  <span className={styles.logCount} style={{ color: '#dc3545' }}>
                    {data.systemMetrics.log_statistics.error_count}
                  </span>
                  <span className={styles.logLabel}>Errors</span>
                </div>
                <div className={styles.logStat}>
                  <span className={styles.logCount} style={{ color: '#ffc107' }}>
                    {data.systemMetrics.log_statistics.warning_count}
                  </span>
                  <span className={styles.logLabel}>Warnings</span>
                </div>
                <div className={styles.logStat}>
                  <span className={styles.logCount} style={{ color: '#28a745' }}>
                    {data.systemMetrics.log_statistics.info_count}
                  </span>
                  <span className={styles.logLabel}>Info</span>
                </div>
              </div>
            </div>

            {data.systemMetrics.log_statistics.recent_errors.length > 0 && (
              <div className={styles.recentErrors}>
                <h4>Recent Errors</h4>
                <div className={styles.errorsList}>
                  {data.systemMetrics.log_statistics.recent_errors.map((error, index) => (
                    <div key={index} className={styles.errorItem}>
                      {error}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Connection Status Banner */}
      {!isConnected && (
        <div className={styles.connectionBanner}>
          <div className={styles.bannerContent}>
            <span>⚠️ WebSocket connection lost. Attempting to reconnect...</span>
            <button onClick={() => window.location.reload()} className={styles.reloadButton}>
              Reload Page
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default RealtimeMonitoringPage;
