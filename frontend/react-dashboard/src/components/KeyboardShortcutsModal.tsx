import { useState, useEffect } from 'react';
import { useKeyboardShortcuts, KeyboardShortcut } from '../hooks/useKeyboardShortcuts';

interface KeyboardShortcutsModalProps {
  isOpen: boolean;
  onClose: () => void;
  shortcuts: KeyboardShortcut[];
}

const KeyboardShortcutsModal = ({
  isOpen,
  onClose,
  shortcuts
}: KeyboardShortcutsModalProps) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');

  // Close modal on Escape key
  useKeyboardShortcuts({
    shortcuts: [
      {
        key: 'Escape',
        action: onClose,
        description: 'Close shortcuts modal'
      }
    ],
    enabled: isOpen
  });

  const categories = ['all', ...Array.from(new Set(shortcuts.map(s => s.category).filter(Boolean)))];

  const filteredShortcuts = shortcuts.filter(shortcut => {
    const matchesSearch = shortcut.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         shortcut.key.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || shortcut.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const formatKeyCombo = (shortcut: KeyboardShortcut) => {
    const parts = [];
    
    if (shortcut.ctrlKey) parts.push('Ctrl');
    if (shortcut.shiftKey) parts.push('Shift');
    if (shortcut.altKey) parts.push('Alt');
    if (shortcut.metaKey) parts.push('Cmd');
    
    parts.push(shortcut.key.toUpperCase());
    
    return parts.join(' + ');
  };

  const groupedShortcuts = filteredShortcuts.reduce((acc, shortcut) => {
    const category = shortcut.category || 'General';
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(shortcut);
    return acc;
  }, {} as Record<string, KeyboardShortcut[]>);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-center justify-center p-4">
        {/* Backdrop */}
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm"
          onClick={onClose}
        />

        {/* Modal */}
        <div className="relative w-full max-w-2xl bg-white dark:bg-gray-800 rounded-lg shadow-xl">
          {/* Header */}
          <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
              Keyboard Shortcuts
            </h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Search and Filters */}
          <div className="p-6 border-b border-gray-200 dark:border-gray-700">
            <div className="flex gap-4">
              <div className="flex-1">
                <input
                  type="text"
                  placeholder="Search shortcuts..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="form-input"
                />
              </div>
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="form-select w-48"
              >
                {categories.map(category => (
                  <option key={category} value={category}>
                    {category === 'all' ? 'All Categories' : category}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Shortcuts List */}
          <div className="p-6 max-h-96 overflow-y-auto">
            {Object.keys(groupedShortcuts).length === 0 ? (
              <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                <svg className="w-12 h-12 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <p>No shortcuts found</p>
              </div>
            ) : (
              <div className="space-y-6">
                {Object.entries(groupedShortcuts).map(([category, categoryShortcuts]) => (
                  <div key={category}>
                    <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-3">
                      {category}
                    </h3>
                    <div className="space-y-2">
                      {categoryShortcuts.map((shortcut, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between py-2 px-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700"
                        >
                          <div className="flex-1">
                            <p className="text-sm text-gray-900 dark:text-white">
                              {shortcut.description}
                            </p>
                          </div>
                          <div className="flex items-center gap-1 ml-4">
                            {formatKeyCombo(shortcut).split(' + ').map((key, keyIndex) => (
                              <span key={keyIndex} className="flex items-center gap-1">
                                <kbd className="px-2 py-1 text-xs font-mono bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded border">
                                  {key}
                                </kbd>
                                {keyIndex < formatKeyCombo(shortcut).split(' + ').length - 1 && (
                                  <span className="text-gray-400">+</span>
                                )}
                              </span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between p-6 border-t border-gray-200 dark:border-gray-700">
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Press <kbd className="px-2 py-1 text-xs font-mono bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded border">Esc</kbd> to close
            </p>
            <button
              onClick={onClose}
              className="btn btn-primary"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default KeyboardShortcutsModal;
