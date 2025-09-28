import React from 'react';
import { LearningMetrics } from '../hooks/useAgentMonitoring';

interface LearningProgressProps {
  metrics: LearningMetrics;
  className?: string;
}

const LearningProgress: React.FC<LearningProgressProps> = ({ metrics, className = '' }) => {
  if (!metrics || !metrics.training_sessions) {
    return (
      <div className={`glass rounded-xl p-6 card-hover ${className}`}>
        <div className="text-center text-slate-400">
          <div className="w-8 h-8 bg-slate-600 rounded-full animate-pulse mx-auto mb-4" />
          <p>No learning metrics available</p>
        </div>
      </div>
    );
  }
  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving':
        return '↗️';
      case 'declining':
        return '↘️';
      default:
        return '→';
    }
  };

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'improving':
        return 'text-success-400';
      case 'declining':
        return 'text-error-400';
      default:
        return 'text-slate-400';
    }
  };

  const formatPercentage = (value: number) => {
    return `${(value * 100).toFixed(1)}%`;
  };

  return (
    <div className={`glass rounded-xl p-6 card-hover ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-semibold text-gradient">Learning Progress</h3>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse" />
          <span className="text-sm font-medium text-slate-400">
            {metrics.training_sessions} sessions
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <div className="glass-strong rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-400">Training Accuracy</span>
            <span className={`text-lg ${getTrendColor(metrics.learning_trends.training_accuracy.trend)}`}>
              {getTrendIcon(metrics.learning_trends.training_accuracy.trend)}
            </span>
          </div>
          <div className="text-2xl font-bold text-white mb-1">
            {formatPercentage(metrics.learning_trends.training_accuracy.current)}
          </div>
          <div className="text-xs text-slate-500 capitalize">
            {metrics.learning_trends.training_accuracy.trend}
          </div>
        </div>

        <div className="glass-strong rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-400">Validation Accuracy</span>
            <span className={`text-lg ${getTrendColor(metrics.learning_trends.validation_accuracy.trend)}`}>
              {getTrendIcon(metrics.learning_trends.validation_accuracy.trend)}
            </span>
          </div>
          <div className="text-2xl font-bold text-white mb-1">
            {formatPercentage(metrics.learning_trends.validation_accuracy.current)}
          </div>
          <div className="text-xs text-slate-500 capitalize">
            {metrics.learning_trends.validation_accuracy.trend}
          </div>
        </div>

        <div className="glass-strong rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-400">Loss</span>
            <span className={`text-lg ${getTrendColor(metrics.learning_trends.loss.trend)}`}>
              {getTrendIcon(metrics.learning_trends.loss.trend)}
            </span>
          </div>
          <div className="text-2xl font-bold text-white mb-1">
            {metrics.learning_trends.loss.current.toFixed(4)}
          </div>
          <div className="text-xs text-slate-500 capitalize">
            {metrics.learning_trends.loss.trend}
          </div>
        </div>

        <div className="glass-strong rounded-lg p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-slate-400">Active Signatures</span>
            <div className="w-2 h-2 bg-accent-500 rounded-full animate-pulse" />
          </div>
          <div className="text-2xl font-bold text-white mb-1">
            {metrics.active_signatures}
          </div>
          <div className="text-xs text-slate-500">
            signatures
          </div>
        </div>
      </div>

      <div className="mb-8">
        <h4 className="text-lg font-semibold text-white mb-4">Signature Performance</h4>
        <div className="space-y-4">
          {Object.entries(metrics.signature_performance).map(([name, perf]) => (
            <div key={name} className="glass-strong rounded-lg p-4">
              <div className="flex items-center justify-between mb-4">
                <span className="font-medium text-white font-mono">{name}</span>
                <span className={`status-pill ${perf.active ? 'status-success' : 'status-warning'}`}>
                  {perf.active ? 'Active' : 'Inactive'}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-sm text-slate-400 mb-1">Performance</div>
                  <div className="text-lg font-semibold text-white">
                    {perf.performance_score.toFixed(1)}%
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-slate-400 mb-1">Success Rate</div>
                  <div className="text-lg font-semibold text-white">
                    {perf.success_rate.toFixed(1)}%
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-sm text-slate-400 mb-1">Response Time</div>
                  <div className="text-lg font-semibold text-white">
                    {perf.avg_response_time.toFixed(2)}s
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {metrics.retrieval_statistics && (
        <div>
          <h4 className="text-lg font-semibold text-white mb-4">Retrieval Statistics</h4>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="glass-strong rounded-lg p-4 text-center">
              <div className="text-sm text-slate-400 mb-1">Total Events</div>
              <div className="text-xl font-bold text-white">
                {metrics.retrieval_statistics.total_events || 0}
              </div>
            </div>
            <div className="glass-strong rounded-lg p-4 text-center">
              <div className="text-sm text-slate-400 mb-1">Avg Score</div>
              <div className="text-xl font-bold text-white">
                {metrics.retrieval_statistics.avg_score ? metrics.retrieval_statistics.avg_score.toFixed(3) : '0.000'}
              </div>
            </div>
            <div className="glass-strong rounded-lg p-4 text-center">
              <div className="text-sm text-slate-400 mb-1">Avg Hits/Query</div>
              <div className="text-xl font-bold text-white">
                {metrics.retrieval_statistics.avg_hits_per_query ? metrics.retrieval_statistics.avg_hits_per_query.toFixed(1) : '0.0'}
              </div>
            </div>
            <div className="glass-strong rounded-lg p-4 text-center">
              <div className="text-sm text-slate-400 mb-1">Unique Queries</div>
              <div className="text-xl font-bold text-white">
                {metrics.retrieval_statistics.unique_queries || 0}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default LearningProgress;
