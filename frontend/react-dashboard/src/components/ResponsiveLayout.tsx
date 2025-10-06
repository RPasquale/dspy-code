import { ReactNode, useState, useEffect } from 'react';

interface ResponsiveLayoutProps {
  children: ReactNode;
  sidebar?: ReactNode;
  header?: ReactNode;
  className?: string;
}

const ResponsiveLayout = ({ children, sidebar, header, className = '' }: ResponsiveLayoutProps) => {
  const [isMobile, setIsMobile] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 1024);
    };

    checkMobile();
    window.addEventListener('resize', checkMobile);
    return () => window.removeEventListener('resize', checkMobile);
  }, []);

  return (
    <div className={`min-h-screen bg-gray-50 dark:bg-gray-900 ${className}`}>
      <div className="flex h-full">
        {/* Sidebar */}
        {sidebar && (
          <>
            {/* Mobile backdrop */}
            {isMobile && sidebarOpen && (
              <div
                className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm lg:hidden"
                onClick={() => setSidebarOpen(false)}
              />
            )}

            {/* Sidebar */}
            <aside
              className={`${
                isMobile
                  ? `fixed inset-y-0 left-0 z-50 w-80 transform transition-transform duration-300 ease-in-out ${
                      sidebarOpen ? 'translate-x-0' : '-translate-x-full'
                    }`
                  : 'relative w-80'
              }`}
            >
              {sidebar}
            </aside>
          </>
        )}

        {/* Main content */}
        <div className="flex flex-1 flex-col">
          {/* Header */}
          {header && (
            <header className="sticky top-0 z-30 border-b border-gray-200 dark:border-gray-700 bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm">
              {header}
            </header>
          )}

          {/* Content */}
          <main className="flex-1 overflow-y-auto bg-white dark:bg-gray-900">
            <div className="px-4 py-6 sm:px-6 lg:px-8">
              {children}
            </div>
          </main>
        </div>
      </div>
    </div>
  );
};

export default ResponsiveLayout;
