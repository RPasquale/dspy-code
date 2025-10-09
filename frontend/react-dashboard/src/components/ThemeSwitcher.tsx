import { useState, useMemo } from 'react';
import { useTheme } from '../contexts/ThemeContext';

const ThemeSwitcher = () => {
  const {
    currentTheme,
    themes,
    setTheme,
    createCustomTheme,
    deleteTheme,
    resetToDefault
  } = useTheme();
  const [isOpen, setIsOpen] = useState(false);

  const [defaultThemes, customThemes] = useMemo(() => {
    const defaults = themes.filter(theme => ['default', 'dark', 'cyberpunk'].includes(theme.id));
    const custom = themes.filter(theme => !['default', 'dark', 'cyberpunk'].includes(theme.id));
    return [defaults, custom];
  }, [themes]);

  const handleSelectTheme = (themeId: string) => {
    setTheme(themeId);
    setIsOpen(false);
  };

  const handleCreateCustomTheme = () => {
    const name = typeof window !== 'undefined'
      ? window.prompt('Name your custom theme', `Custom Theme ${customThemes.length + 1}`)
      : null;
    if (!name) return;

    const primary = typeof window !== 'undefined'
      ? window.prompt('Primary color (hex value)', currentTheme.colors.primary)
      : null;
    if (!primary) return;

    const accent = typeof window !== 'undefined'
      ? window.prompt('Secondary color (hex value)', currentTheme.colors.secondary)
      : null;
    if (!accent) return;

    const id = createCustomTheme({
      name,
      colors: {
        ...currentTheme.colors,
        primary: primary || currentTheme.colors.primary,
        secondary: accent || currentTheme.colors.secondary
      },
      fonts: currentTheme.fonts,
      spacing: currentTheme.spacing,
      borderRadius: currentTheme.borderRadius,
      shadows: currentTheme.shadows
    });
    setTheme(id);
    setIsOpen(false);
  };

  const handleDeleteTheme = (themeId: string) => {
    if (typeof window !== 'undefined') {
      const confirmed = window.confirm('Remove this custom theme?');
      if (!confirmed) return;
    }
    deleteTheme(themeId);
  };

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 px-3 py-2 text-sm text-gray-700 dark:text-gray-200 hover:border-blue-500 hover:text-blue-600"
        type="button"
      >
        <span
          className="h-3 w-3 rounded-full"
          style={{ backgroundColor: currentTheme.colors.primary }}
        />
        <span>{currentTheme.name}</span>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-64 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-lg z-50">
          <div className="p-2">
            <div className="px-3 py-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
              Built-in Themes
            </div>
            {defaultThemes.map(theme => (
              <button
                key={theme.id}
                onClick={() => handleSelectTheme(theme.id)}
                className={`flex w-full items-center justify-between rounded-lg px-3 py-2 text-sm transition-colors ${
                  currentTheme.id === theme.id
                    ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-300'
                    : 'text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700'
                }`}
                type="button"
              >
                <div className="flex items-center gap-2">
                  <span
                    className="h-3 w-3 rounded-full"
                    style={{ backgroundColor: theme.colors.primary }}
                  />
                  <span>{theme.name}</span>
                </div>
                {currentTheme.id === theme.id && (
                  <span className="text-xs text-blue-500">Active</span>
                )}
              </button>
            ))}

            {!!customThemes.length && (
              <div className="px-3 py-2 text-xs font-semibold uppercase tracking-wide text-gray-400">
                Custom Themes
              </div>
            )}
            {customThemes.map(theme => (
              <div key={theme.id} className="flex items-center gap-2 px-3 py-2">
                <button
                  onClick={() => handleSelectTheme(theme.id)}
                  className={`flex-1 rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                    currentTheme.id === theme.id
                      ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-600 dark:text-blue-300'
                      : 'text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700'
                  }`}
                  type="button"
                >
                  <div className="flex items-center gap-2">
                    <span
                      className="h-3 w-3 rounded-full"
                      style={{ backgroundColor: theme.colors.primary }}
                    />
                    <span>{theme.name}</span>
                  </div>
                </button>
                <button
                  onClick={() => handleDeleteTheme(theme.id)}
                  className="text-xs text-gray-400 hover:text-red-500"
                  type="button"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
          <div className="border-t border-gray-200 dark:border-gray-700 p-2">
            <button
              onClick={handleCreateCustomTheme}
              className="w-full rounded-lg border border-dashed border-gray-300 dark:border-gray-600 px-3 py-2 text-sm text-gray-600 dark:text-gray-300 hover:border-blue-500 hover:text-blue-600"
              type="button"
            >
              Create Custom Theme
            </button>
            <button
              onClick={() => {
                resetToDefault();
                setIsOpen(false);
              }}
              className="mt-2 w-full rounded-lg px-3 py-2 text-sm text-gray-500 hover:text-blue-600"
              type="button"
            >
              Reset to Default
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ThemeSwitcher;
