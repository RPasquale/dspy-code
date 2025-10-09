import { useState, useEffect } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import Navigation from './components/Navigation';
import Header from './components/Header';
import AgentPulsePage from './pages/AgentPulsePage';
import PerformanceInsightsPage from './pages/PerformanceInsightsPage';
import DataStreamsPage from './pages/DataStreamsPage';
import SystemOverviewPage from './pages/SystemOverviewPage';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import KeyboardShortcutsModal from './components/KeyboardShortcutsModal';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { useTheme } from './contexts/ThemeContext';
import { usePerformanceOptimization } from './hooks/usePerformanceOptimization';
import GraphInsightsPage from './pages/GraphInsightsPage';
import WorkflowBuilderPage from './pages/WorkflowBuilderPage';

const App = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [shortcutsModalOpen, setShortcutsModalOpen] = useState(false);
  const { currentTheme } = useTheme();
  
  const { startRender, endRender, debounce, metrics } = usePerformanceOptimization({
    enableMetrics: true,
    enableMemoization: true,
    enableDebouncing: true,
    onPerformanceIssue: (metrics) => {
      console.warn('Performance issue detected:', metrics);
    }
  });

  // Keyboard shortcuts
  const shortcuts = [
    {
      key: 'k',
      ctrlKey: true,
      action: () => setShortcutsModalOpen(true),
      description: 'Open keyboard shortcuts',
      category: 'Navigation'
    },
    {
      key: 'Escape',
      action: () => {
        setSidebarOpen(false);
        setShortcutsModalOpen(false);
      },
      description: 'Close modals and sidebar',
      category: 'Navigation'
    },
    {
      key: 'b',
      ctrlKey: true,
      action: () => setSidebarOpen(!sidebarOpen),
      description: 'Toggle sidebar',
      category: 'Navigation'
    },
    {
      key: '1',
      ctrlKey: true,
      action: () => window.location.href = '/agent',
      description: 'Go to Training Progress',
      category: 'Navigation'
    },
    {
      key: '2',
      ctrlKey: true,
      action: () => window.location.href = '/performance',
      description: 'Go to Results & Insights',
      category: 'Navigation'
    },
    {
      key: '3',
      ctrlKey: true,
      action: () => window.location.href = '/streams',
      description: 'Go to Data Sources',
      category: 'Navigation'
    },
    {
      key: '4',
      ctrlKey: true,
      action: () => window.location.href = '/system',
      description: 'Go to Health Status',
      category: 'Navigation'
    },
    {
      key: '5',
      ctrlKey: true,
      action: () => window.location.href = '/analytics',
      description: 'Go to Reports',
      category: 'Navigation'
    },
    {
      key: '6',
      ctrlKey: true,
      action: () => window.location.href = '/workflow',
      description: 'Go to Workflow Composer',
      category: 'Navigation'
    },
    {
      key: 'r',
      ctrlKey: true,
      action: () => window.location.reload(),
      description: 'Refresh page',
      category: 'Actions'
    },
    {
      key: 'e',
      ctrlKey: true,
      action: () => {
        // Trigger export functionality
        const exportButton = document.querySelector('[data-export-button]') as HTMLButtonElement;
        if (exportButton) exportButton.click();
      },
      description: 'Export current data',
      category: 'Actions'
    }
  ];

  useKeyboardShortcuts({ shortcuts });

  // Performance monitoring
  useEffect(() => {
    startRender();
    return () => endRender();
  }, [startRender, endRender]);

  // Apply theme
  useEffect(() => {
    document.documentElement.className = currentTheme.id;
  }, [currentTheme]);

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="flex h-full">
        <Navigation sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
        
        <div className="flex flex-1 flex-col lg:ml-0">
          <Header sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
          
          <main className="flex-1 overflow-y-auto bg-white dark:bg-gray-900">
            <div className="px-4 py-6 sm:px-6 lg:px-8">
              <Routes>
                <Route path="/" element={<Navigate to="/agent" replace />} />
                <Route path="/agent" element={<AgentPulsePage />} />
                <Route path="/performance" element={<PerformanceInsightsPage />} />
                <Route path="/streams" element={<DataStreamsPage />} />
                <Route path="/system" element={<SystemOverviewPage />} />
                <Route path="/analytics" element={<AnalyticsDashboard />} />
                <Route path="/graph" element={<GraphInsightsPage />} />
                <Route path="/workflow" element={<WorkflowBuilderPage />} />
              </Routes>
            </div>
          </main>
        </div>
      </div>

      {/* Keyboard Shortcuts Modal */}
      <KeyboardShortcutsModal
        isOpen={shortcutsModalOpen}
        onClose={() => setShortcutsModalOpen(false)}
        shortcuts={shortcuts}
      />

      {/* Performance Debug (development only) */}
      {process.env.NODE_ENV === 'development' && (
        <div className="fixed bottom-4 right-4 bg-black/80 text-white text-xs p-2 rounded">
          <div>Render: {metrics.renderTime.toFixed(2)}ms</div>
          <div>Memory: {metrics.memoryUsage.toFixed(2)}MB</div>
          <div>Components: {metrics.componentCount}</div>
        </div>
      )}
    </div>
  );
};

export default App;
