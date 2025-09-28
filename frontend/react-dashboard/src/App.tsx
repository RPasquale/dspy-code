import { NavLink, Route, Routes } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { api } from './api/client';
import OverviewPage from './pages/OverviewPage';
import ExperimentsPage from './pages/ExperimentsPage';
import MonitoringPage from './pages/MonitoringPage';
import RealtimeMonitoringPage from './pages/RealtimeMonitoringPage';
import AdvancedLearningPage from './pages/AdvancedLearningPage';
import SystemMapPage from './pages/SystemMapPage';
import SystemStatsPage from './pages/SystemStatsPage';
import SignatureGraphPage from './pages/SignatureGraphPage';
import BusMetricsPage from './pages/BusMetricsPage';
import MonitorLite from './pages/MonitorLite';
import SignaturesPage from './pages/SignaturesPage';
import VerifiersPage from './pages/VerifiersPage';
import RewardsPage from './pages/RewardsPage';
import ActionsPage from './pages/ActionsPage';
import TrainingPage from './pages/TrainingPage';
import SweepsPage from './pages/SweepsPage';
import CapacityPage from './pages/CapacityPage';
import ProfileSwitcher from './components/ProfileSwitcher';
import SparkAppsPage from './pages/SparkAppsPage';
import EventsPage from './pages/EventsPage';
import EnvQueuePage from './pages/EnvQueuePage';
import StreamsExplorerPage from './pages/StreamsExplorerPage';

const App = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  const { data: busData } = useQuery({
    queryKey: ['bus-metrics-header'],
    queryFn: api.getBusMetrics,
    refetchInterval: 10000
  });
  const { data: dbHealth } = useQuery({ queryKey: ['reddb-health'], queryFn: api.getReddbHealth, refetchInterval: 15000 });
  
  const showBackpressure = (busData?.alerts ?? []).some((a) => a.level === 'warning' && a.message.toLowerCase().includes('backpressure'));
  const backpressureDepth = busData?.thresholds?.backpressure_depth ?? 100;
  const dlqMin = busData?.thresholds?.dlq_min ?? 1;
  const okState = !!busData && !showBackpressure && (busData?.dlq?.total ?? 0) < dlqMin;

  const navigation = [
    {
      name: 'Overview',
      href: '/',
      icon: '🏠',
      current: false
    },
    {
      name: 'Monitoring',
      href: '/monitoring',
      icon: '📊',
      current: false
    },
    {
      name: 'Experiments',
      href: '/experiments',
      icon: '🧪',
      current: false
    },
    {
      name: 'Training',
      href: '/training',
      icon: '🎯',
      current: false
    },
    {
      name: 'Advanced Learning',
      href: '/advanced',
      icon: '🧠',
      current: false
    }
  ];

  const secondaryNav = [
    {
      name: 'Signatures',
      href: '/signatures',
      icon: '📝'
    },
    {
      name: 'Verifiers',
      href: '/verifiers',
      icon: '✅'
    },
    {
      name: 'Rewards',
      href: '/rewards',
      icon: '🏆'
    },
    {
      name: 'Actions',
      href: '/actions',
      icon: '⚡'
    }
  ];

  const systemNav = [
    {
      name: 'System Stats',
      href: '/system-stats',
      icon: '💻'
    },
    {
      name: 'System Map',
      href: '/system',
      icon: '🗺️'
    },
    {
      name: 'Spark Apps',
      href: '/spark',
      icon: '⚡'
    },
    {
      name: 'Events',
      href: '/events',
      icon: '📅'
    },
    {
      name: 'Streams',
      href: '/streams',
      icon: '🌊'
    },
    {
      name: 'Env Queue',
      href: '/env-queue',
      icon: '📋'
    }
  ];

  return (
    <div className="min-h-screen cyber-grid flex">
      {/* Mobile sidebar backdrop */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black bg-opacity-50 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <div className={`fixed inset-y-0 left-0 z-50 w-64 cyber-card border-r border-green-500 transform transition-transform duration-300 ease-in-out lg:translate-x-0 lg:static lg:inset-0 ${
        sidebarOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        <div className="flex flex-col h-full">
          {/* Logo */}
          <div className="flex items-center px-6 py-4 border-b border-slate-200">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">D</span>
            </div>
            <div className="ml-3">
              <h1 className="text-lg font-semibold text-slate-900">DSPy Monitor</h1>
              <p className="text-xs text-slate-500">AI Training Platform</p>
            </div>
          </div>

          {/* Status indicators */}
          <div className="px-6 py-3 border-b border-slate-200">
            <div className="flex items-center space-x-2">
              {showBackpressure && (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-800">
                  ⚠️ Backpressure
                </span>
              )}
              {!showBackpressure && okState && (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  ✅ OK
                </span>
              )}
              {dbHealth && (
                <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                  dbHealth.status === 'ok' 
                    ? 'bg-green-100 text-green-800' 
                    : dbHealth.status === 'warn' 
                    ? 'bg-yellow-100 text-yellow-800' 
                    : 'bg-red-100 text-red-800'
                }`}>
                  {dbHealth.status === 'ok' ? '✅' : dbHealth.status === 'warn' ? '⚠️' : '❌'} DB {dbHealth.status.toUpperCase()}
                </span>
              )}
            </div>
          </div>

          {/* Main navigation */}
          <nav className="flex-1 px-4 py-4 space-y-1">
            <div className="space-y-1">
              {navigation.map((item) => (
                <NavLink
                  key={item.name}
                  to={item.href}
                  end={item.href === '/'}
                  className={({ isActive }) =>
                    `group flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
                      isActive
                        ? 'bg-blue-50 text-blue-700 border border-blue-200'
                        : 'text-slate-700 hover:bg-slate-50 hover:text-slate-900'
                    }`
                  }
                >
                  <span className="mr-3 text-lg">{item.icon}</span>
                  {item.name}
                </NavLink>
              ))}
            </div>

            {/* Secondary navigation */}
            <div className="pt-6">
              <h3 className="px-3 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Analysis
              </h3>
              <div className="mt-2 space-y-1">
                {secondaryNav.map((item) => (
                  <NavLink
                    key={item.name}
                    to={item.href}
                    className={({ isActive }) =>
                      `group flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
                        isActive
                          ? 'bg-blue-50 text-blue-700 border border-blue-200'
                          : 'text-slate-700 hover:bg-slate-50 hover:text-slate-900'
                      }`
                    }
                  >
                    <span className="mr-3 text-lg">{item.icon}</span>
                    {item.name}
                  </NavLink>
                ))}
              </div>
            </div>

            {/* System navigation */}
            <div className="pt-6">
              <h3 className="px-3 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                System
              </h3>
              <div className="mt-2 space-y-1">
                {systemNav.map((item) => (
                  <NavLink
                    key={item.name}
                    to={item.href}
                    className={({ isActive }) =>
                      `group flex items-center px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
                        isActive
                          ? 'bg-blue-50 text-blue-700 border border-blue-200'
                          : 'text-slate-700 hover:bg-slate-50 hover:text-slate-900'
                      }`
                    }
                  >
                    <span className="mr-3 text-lg">{item.icon}</span>
                    {item.name}
                  </NavLink>
                ))}
              </div>
            </div>
          </nav>

          {/* Profile section */}
          <div className="border-t border-slate-200 p-4">
            <ProfileSwitcher />
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 min-h-screen">
        {/* Top bar */}
        <div className="sticky top-0 z-40 bg-white border-b border-slate-200">
          <div className="flex items-center justify-between px-4 py-3">
            <button
              type="button"
              className="lg:hidden p-2 rounded-md text-slate-500 hover:text-slate-900 hover:bg-slate-100"
              onClick={() => setSidebarOpen(true)}
            >
              <span className="sr-only">Open sidebar</span>
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            
            <div className="flex items-center space-x-4">
              <div className="hidden sm:block">
                <h2 className="text-lg font-semibold text-slate-900">DSPy Monitor</h2>
              </div>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
            <Route path="/monitoring" element={<MonitoringPage />} />
            <Route path="/monitor-lite" element={<MonitorLite />} />
            <Route path="/realtime" element={<RealtimeMonitoringPage />} />
            <Route path="/advanced" element={<AdvancedLearningPage />} />
            <Route path="/signatures" element={<SignaturesPage />} />
            <Route path="/verifiers" element={<VerifiersPage />} />
            <Route path="/rewards" element={<RewardsPage />} />
            <Route path="/training" element={<TrainingPage />} />
            <Route path="/sweeps" element={<SweepsPage />} />
            <Route path="/actions" element={<ActionsPage />} />
            <Route path="/system-stats" element={<SystemStatsPage />} />
            <Route path="/system" element={<SystemMapPage />} />
            <Route path="/sig-graph" element={<SignatureGraphPage />} />
            <Route path="/bus" element={<BusMetricsPage />} />
            <Route path="/capacity" element={<CapacityPage />} />
            <Route path="/spark" element={<SparkAppsPage />} />
            <Route path="/events" element={<EventsPage />} />
            <Route path="/streams" element={<StreamsExplorerPage />} />
            <Route path="/env-queue" element={<EnvQueuePage />} />
          </Routes>
        </main>
      </div>
    </div>
  );
};

export default App;