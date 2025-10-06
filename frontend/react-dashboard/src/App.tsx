import { useState } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import Navigation from './components/Navigation';
import Header from './components/Header';
import AgentPulsePage from './pages/AgentPulsePage';
import PerformanceInsightsPage from './pages/PerformanceInsightsPage';
import DataStreamsPage from './pages/DataStreamsPage';
import SystemOverviewPage from './pages/SystemOverviewPage';

const App = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

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
              </Routes>
            </div>
          </main>
        </div>
      </div>
    </div>
  );
};

export default App;
