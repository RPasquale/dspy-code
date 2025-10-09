import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

export interface Theme {
  id: string;
  name: string;
  colors: {
    primary: string;
    secondary: string;
    success: string;
    warning: string;
    error: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    border: string;
  };
  fonts: {
    primary: string;
    secondary: string;
    mono: string;
  };
  spacing: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
  borderRadius: {
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
  shadows: {
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
}

export interface ThemeContextType {
  currentTheme: Theme;
  themes: Theme[];
  setTheme: (themeId: string) => void;
  createCustomTheme: (theme: Omit<Theme, 'id'>) => string;
  updateTheme: (themeId: string, updates: Partial<Theme>) => void;
  deleteTheme: (themeId: string) => void;
  resetToDefault: () => void;
}

const defaultThemes: Theme[] = [
  {
    id: 'default',
    name: 'Default',
    colors: {
      primary: '#4a9eff',
      secondary: '#64748b',
      success: '#00d4aa',
      warning: '#ffb800',
      error: '#ff4757',
      background: '#ffffff',
      surface: '#f8fafc',
      text: '#1f2937',
      textSecondary: '#6b7280',
      border: '#e5e7eb'
    },
    fonts: {
      primary: 'Inter, system-ui, sans-serif',
      secondary: 'Inter, system-ui, sans-serif',
      mono: 'JetBrains Mono, Fira Code, monospace'
    },
    spacing: {
      xs: '0.25rem',
      sm: '0.5rem',
      md: '1rem',
      lg: '1.5rem',
      xl: '2rem'
    },
    borderRadius: {
      sm: '0.25rem',
      md: '0.5rem',
      lg: '0.75rem',
      xl: '1rem'
    },
    shadows: {
      sm: '0 1px 2px rgba(0, 0, 0, 0.05)',
      md: '0 4px 6px rgba(0, 0, 0, 0.07)',
      lg: '0 10px 15px rgba(0, 0, 0, 0.1)',
      xl: '0 20px 25px rgba(0, 0, 0, 0.1)'
    }
  },
  {
    id: 'dark',
    name: 'Dark',
    colors: {
      primary: '#4a9eff',
      secondary: '#64748b',
      success: '#00d4aa',
      warning: '#ffb800',
      error: '#ff4757',
      background: '#0f172a',
      surface: '#1e293b',
      text: '#f1f5f9',
      textSecondary: '#94a3b8',
      border: '#334155'
    },
    fonts: {
      primary: 'Inter, system-ui, sans-serif',
      secondary: 'Inter, system-ui, sans-serif',
      mono: 'JetBrains Mono, Fira Code, monospace'
    },
    spacing: {
      xs: '0.25rem',
      sm: '0.5rem',
      md: '1rem',
      lg: '1.5rem',
      xl: '2rem'
    },
    borderRadius: {
      sm: '0.25rem',
      md: '0.5rem',
      lg: '0.75rem',
      xl: '1rem'
    },
    shadows: {
      sm: '0 1px 2px rgba(0, 0, 0, 0.3)',
      md: '0 4px 6px rgba(0, 0, 0, 0.4)',
      lg: '0 10px 15px rgba(0, 0, 0, 0.5)',
      xl: '0 20px 25px rgba(0, 0, 0, 0.6)'
    }
  },
  {
    id: 'cyberpunk',
    name: 'Cyberpunk',
    colors: {
      primary: '#00ff00',
      secondary: '#ff00ff',
      success: '#00ff00',
      warning: '#ffff00',
      error: '#ff0040',
      background: '#0a0a0f',
      surface: '#1a1a24',
      text: '#ffffff',
      textSecondary: '#b4b4c7',
      border: '#2a2a35'
    },
    fonts: {
      primary: 'Inter, system-ui, sans-serif',
      secondary: 'Inter, system-ui, sans-serif',
      mono: 'JetBrains Mono, Fira Code, monospace'
    },
    spacing: {
      xs: '0.25rem',
      sm: '0.5rem',
      md: '1rem',
      lg: '1.5rem',
      xl: '2rem'
    },
    borderRadius: {
      sm: '0.25rem',
      md: '0.5rem',
      lg: '0.75rem',
      xl: '1rem'
    },
    shadows: {
      sm: '0 0 10px rgba(0, 255, 0, 0.3)',
      md: '0 0 20px rgba(0, 255, 0, 0.4)',
      lg: '0 0 30px rgba(0, 255, 0, 0.5)',
      xl: '0 0 40px rgba(0, 255, 0, 0.6)'
    }
  }
];

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

interface ThemeProviderProps {
  children: ReactNode;
}

export const ThemeProvider = ({ children }: ThemeProviderProps) => {
  const [themes, setThemes] = useState<Theme[]>(defaultThemes);
  const [currentThemeId, setCurrentThemeId] = useState<string>('default');

  const currentTheme = themes.find(theme => theme.id === currentThemeId) || themes[0];

  // Load theme from localStorage on mount
  useEffect(() => {
    const savedThemeId = localStorage.getItem('theme-id');
    const savedThemes = localStorage.getItem('themes');
    
    if (savedThemeId) {
      setCurrentThemeId(savedThemeId);
    }
    
    if (savedThemes) {
      try {
        const parsedThemes = JSON.parse(savedThemes);
        setThemes([...defaultThemes, ...parsedThemes.filter((t: Theme) => !defaultThemes.some(dt => dt.id === t.id))]);
      } catch (error) {
        console.error('Failed to parse saved themes:', error);
      }
    }
  }, []);

  // Apply theme to CSS variables
  useEffect(() => {
    const root = document.documentElement;
    
    // Apply color variables
    Object.entries(currentTheme.colors).forEach(([key, value]) => {
      root.style.setProperty(`--color-${key}`, value);
    });
    
    // Apply font variables
    Object.entries(currentTheme.fonts).forEach(([key, value]) => {
      root.style.setProperty(`--font-${key}`, value);
    });
    
    // Apply spacing variables
    Object.entries(currentTheme.spacing).forEach(([key, value]) => {
      root.style.setProperty(`--spacing-${key}`, value);
    });
    
    // Apply border radius variables
    Object.entries(currentTheme.borderRadius).forEach(([key, value]) => {
      root.style.setProperty(`--border-radius-${key}`, value);
    });
    
    // Apply shadow variables
    Object.entries(currentTheme.shadows).forEach(([key, value]) => {
      root.style.setProperty(`--shadow-${key}`, value);
    });
    
    // Set dark mode class
    if (currentTheme.id === 'dark' || currentTheme.id === 'cyberpunk') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [currentTheme]);

  const setTheme = (themeId: string) => {
    setCurrentThemeId(themeId);
    localStorage.setItem('theme-id', themeId);
  };

  const createCustomTheme = (theme: Omit<Theme, 'id'>): string => {
    const id = `custom-${Date.now()}`;
    const newTheme: Theme = { ...theme, id };
    
    setThemes(prev => [...prev, newTheme]);
    localStorage.setItem('themes', JSON.stringify(themes.filter(t => !defaultThemes.some(dt => dt.id === t.id))));
    
    return id;
  };

  const updateTheme = (themeId: string, updates: Partial<Theme>) => {
    setThemes(prev => prev.map(theme => 
      theme.id === themeId ? { ...theme, ...updates } : theme
    ));
    
    // Save custom themes to localStorage
    const customThemes = themes.filter(t => !defaultThemes.some(dt => dt.id === t.id));
    localStorage.setItem('themes', JSON.stringify(customThemes));
  };

  const deleteTheme = (themeId: string) => {
    if (defaultThemes.some(theme => theme.id === themeId)) {
      throw new Error('Cannot delete default themes');
    }
    
    setThemes(prev => prev.filter(theme => theme.id !== themeId));
    
    // If current theme is deleted, switch to default
    if (currentThemeId === themeId) {
      setTheme('default');
    }
    
    // Save custom themes to localStorage
    const customThemes = themes.filter(t => !defaultThemes.some(dt => dt.id === t.id));
    localStorage.setItem('themes', JSON.stringify(customThemes));
  };

  const resetToDefault = () => {
    setTheme('default');
    setThemes(defaultThemes);
    localStorage.removeItem('themes');
  };

  return (
    <ThemeContext.Provider
      value={{
        currentTheme,
        themes,
        setTheme,
        createCustomTheme,
        updateTheme,
        deleteTheme,
        resetToDefault
      }}
    >
      {children}
    </ThemeContext.Provider>
  );
};

export default ThemeProvider;
