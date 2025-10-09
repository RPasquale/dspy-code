import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import StatusPill from './StatusPill';
import ThemeSwitcher from './ThemeSwitcher';

interface HeaderProps {
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
}

const Header = ({ sidebarOpen, setSidebarOpen }: HeaderProps) => {
  const [notificationsOpen, setNotificationsOpen] = useState(false);

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

  return (
    <header className="sticky top-0 z-30 border-b border-gray-200 dark:border-gray-700 bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm">
      <div className="px-4 py-3 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between">
          {/* Left side */}
          <div className="flex items-center gap-4">
            {/* Mobile menu button */}
            <button
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 lg:hidden"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <svg className="h-6 w-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>

            {/* Breadcrumb */}
            <nav className="hidden md:flex items-center space-x-1 text-sm">
              <span className="text-gray-500 dark:text-gray-400">Dashboard</span>
              <svg className="h-4 w-4 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" />
              </svg>
              <span className="text-gray-900 dark:text-white font-medium">Current Page</span>
            </nav>
          </div>

          {/* Center - Business Status indicators */}
          <div className="hidden lg:flex items-center gap-4">
            <div className="flex items-center gap-2 rounded-full border border-gray-200 dark:border-gray-700 px-3 py-1 text-xs">
              <span className="text-gray-500 dark:text-gray-400">Learning</span>
              <StatusPill status={status?.agent?.status} size="sm" />
              <span className="text-gray-700 dark:text-gray-300">
                {status?.learning_active ? 'Active' : 'Ready'}
              </span>
            </div>
            <div className="flex items-center gap-2 rounded-full border border-gray-200 dark:border-gray-700 px-3 py-1 text-xs">
              <span className="text-gray-500 dark:text-gray-400">Data Quality</span>
              <StatusPill status={status?.kafka?.status} size="sm" />
              <span className="text-gray-700 dark:text-gray-300">
                {busMetrics?.dlq?.total === 0 ? 'Good' : `${busMetrics?.dlq?.total ?? 0} issues`}
              </span>
            </div>
            <div className="flex items-center gap-2 rounded-full border border-gray-200 dark:border-gray-700 px-3 py-1 text-xs">
              <span className="text-gray-500 dark:text-gray-400">System</span>
              <StatusPill status={status?.pipeline?.status} size="sm" />
              <span className="text-gray-700 dark:text-gray-300">
                Operational
              </span>
            </div>
          </div>

          {/* Right side */}
          <div className="flex items-center gap-3">
            <ThemeSwitcher />
            {/* Search */}
            <div className="hidden sm:block">
              <div className="relative">
                <input
                  type="text"
                  placeholder="Search..."
                  className="w-64 px-3 py-2 pl-10 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
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

            {/* Notifications */}
            <div className="relative">
              <button
                className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 relative"
                onClick={() => setNotificationsOpen(!notificationsOpen)}
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-5 5-5-5h5v-5a7.5 7.5 0 1 1 15 0v5z" />
                </svg>
                {/* Notification badge */}
                <span className="absolute -top-1 -right-1 h-3 w-3 bg-red-500 rounded-full"></span>
              </button>

              {/* Notifications dropdown */}
              {notificationsOpen && (
                <div className="absolute right-0 mt-2 w-80 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 z-50">
                  <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                    <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
                      Notifications
                    </h3>
                  </div>
                  <div className="p-2">
                    <div className="text-sm text-gray-500 dark:text-gray-400 text-center py-4">
                      No new notifications
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Settings */}
            <button className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
              <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>

            {/* User menu */}
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-sm font-medium">
                U
              </div>
              <span className="hidden sm:block text-sm font-medium text-gray-900 dark:text-white">
                User
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
