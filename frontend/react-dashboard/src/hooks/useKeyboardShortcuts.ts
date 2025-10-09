import { useEffect, useCallback, useRef } from 'react';

export interface KeyboardShortcut {
  key: string;
  ctrlKey?: boolean;
  shiftKey?: boolean;
  altKey?: boolean;
  metaKey?: boolean;
  action: () => void;
  description: string;
  category?: string;
}

export interface UseKeyboardShortcutsOptions {
  shortcuts: KeyboardShortcut[];
  enabled?: boolean;
  preventDefault?: boolean;
  stopPropagation?: boolean;
}

export const useKeyboardShortcuts = ({
  shortcuts,
  enabled = true,
  preventDefault = true,
  stopPropagation = false
}: UseKeyboardShortcutsOptions) => {
  const shortcutsRef = useRef(shortcuts);
  const enabledRef = useRef(enabled);

  // Update refs when props change
  useEffect(() => {
    shortcutsRef.current = shortcuts;
  }, [shortcuts]);

  useEffect(() => {
    enabledRef.current = enabled;
  }, [enabled]);

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    if (!enabledRef.current) return;

    const pressedKey = event.key.toLowerCase();
    const isCtrl = event.ctrlKey;
    const isShift = event.shiftKey;
    const isAlt = event.altKey;
    const isMeta = event.metaKey;

    // Find matching shortcut
    const matchingShortcut = shortcutsRef.current.find(shortcut => {
      const keyMatch = shortcut.key.toLowerCase() === pressedKey;
      const ctrlMatch = (shortcut.ctrlKey || false) === isCtrl;
      const shiftMatch = (shortcut.shiftKey || false) === isShift;
      const altMatch = (shortcut.altKey || false) === isAlt;
      const metaMatch = (shortcut.metaKey || false) === isMeta;

      return keyMatch && ctrlMatch && shiftMatch && altMatch && metaMatch;
    });

    if (matchingShortcut) {
      if (preventDefault) {
        event.preventDefault();
      }
      
      if (stopPropagation) {
        event.stopPropagation();
      }

      matchingShortcut.action();
    }
  }, [preventDefault, stopPropagation]);

  useEffect(() => {
    if (enabled) {
      document.addEventListener('keydown', handleKeyDown);
      return () => document.removeEventListener('keydown', handleKeyDown);
    }
  }, [enabled, handleKeyDown]);

  return {
    shortcuts: shortcutsRef.current
  };
};

export default useKeyboardShortcuts;
