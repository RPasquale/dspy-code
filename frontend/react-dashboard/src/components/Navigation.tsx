import { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import StatusPill from './StatusPill';
import ProfileSwitcher from './ProfileSwitcher';

interface NavigationItem {
  name: string;
  description: string;
  href: string;
  icon: string;
  badge?: string;
  badgeColor?: 'success' | 'warning' | 'error' | 'info';
}

const navigation: NavigationItem[] = [
  {
    name: 'Agent Pulse',
    description: 'Live actions, training, and inference telemetry',
    href: '/agent',
    icon: '⚡'
  },
  {
    name: 'Performance',
    description: 'Rewards, verifiers, and DSPy / RL analytics',
    href: '/performance',
    icon: '📈'
  },
  {
    name: 'Data Streams',
    description: 'Streams, backpressure, and RedDB records',
    href: '/streams',
    icon: '🌊'
  },
  {
    name: 'System',
    description: 'Infrastructure, guardrails, and operations',
    href: '/system',
    icon: '🧩'
  }
];

interface NavigationProps {
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
}

const Navigation = ({ sidebarOpen, setSidebarOpen }: NavigationProps) => {
  const location = useLocation();
  const [searchQuery, setSearchQuery] = useState('');

  const { data: status } = useQuery({
    queryKey: ['status'],
    queryFn: api.getStatus,
    refetchInterval: 12000
  });

  const { data: busMetrics } = useQuery({
    queryKey: ['bus-metrics-header'],
    queryFn: api.getBusMetrics,
    refetchInterval: 20000
  });

  // Close sidebar on route change
  useEffect(() => {
    setSidebarOpen(false);
  }, [location.pathname, setSidebarOpen]);

  // Close sidebar on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setSidebarOpen(false);
      }
    };

    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, [setSidebarOpen]);

  const filteredNavigation = navigation.filter(item =>
    item.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    item.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <>
      {/* Mobile backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 w-80 transform border-r border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 transition-transform duration-300 ease-in-out lg:static lg:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex h-full flex-col">
          {/* Header */}
          <div className="border-b border-gray-200 dark:border-gray-700 p-6">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 text-lg font-bold text-white">
                Δ
              </div>
              <div>
                <div className="text-lg font-semibold text-gray-900 dark:text-white">
                  Agent Console
                </div>
                <div className="text-xs text-gray-500 dark:text-gray-400">
                  Visual intelligence for DSPy
                </div>
              </div>
            </div>
          </div>

          {/* Search */}
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <div className="relative">
              <input
                type="text"
                placeholder="Search pages..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full px-3 py-2 pl-10 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <svg
                className="absolute left-3 top-2.5 h-4 w-4 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 overflow-y-auto p-4">
            <div className="space-y-2">
              {filteredNavigation.map((item) => (
                <NavLink
                  key={item.href}
                  to={item.href}
                  className={({ isActive }) =>
                    `group flex items-center gap-3 rounded-lg px-3 py-3 text-sm transition-all duration-200 ${
                      isActive
                        ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800'
                        : 'text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-white'
                    }`
                  }
                >
                  <span className="text-lg">{item.icon}</span>
                  <div className="flex-1 min-w-0">
                    <div className="font-medium truncate">{item.name}</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 truncate">
                      {item.description}
                    </div>
                  </div>
                  {item.badge && (
                    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                      item.badgeColor === 'success' ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-300' :
                      item.badgeColor === 'warning' ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-300' :
                      item.badgeColor === 'error' ? 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-300' :
                      'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-300'
                    }`}>
                      {item.badge}
                    </span>
                  )}
                </NavLink>
              ))}
            </div>
          </nav>

          {/* Status indicators */}
          <div className="border-t border-gray-200 dark:border-gray-700 p-4 space-y-3">
            <div className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
              System Status
            </div>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600 dark:text-gray-400">Agent</span>
                <StatusPill status={status?.agent?.status} size="sm" />
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600 dark:text-gray-400">Bus</span>
                <StatusPill status={status?.kafka?.status} size="sm" />
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600 dark:text-gray-400">Training</span>
                <StatusPill status={status?.pipeline?.status} size="sm" />
              </div>
            </div>
          </div>

          {/* Profile switcher */}
          <div className="border-t border-gray-200 dark:border-gray-700 p-4">
            <ProfileSwitcher />
          </div>
        </div>
      </aside>
    </>
  );
};

export default Navigation;
