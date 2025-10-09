import { useEffect, useRef, useCallback, useState } from 'react';

export interface PerformanceMetrics {
  renderTime: number;
  memoryUsage: number;
  componentCount: number;
  reRenderCount: number;
}

export interface UsePerformanceOptimizationOptions {
  enableMetrics?: boolean;
  enableVirtualization?: boolean;
  enableMemoization?: boolean;
  enableDebouncing?: boolean;
  debounceDelay?: number;
  onPerformanceIssue?: (metrics: PerformanceMetrics) => void;
}

export const usePerformanceOptimization = (options: UsePerformanceOptimizationOptions = {}) => {
  const {
    enableMetrics = true,
    enableVirtualization = false,
    enableMemoization = true,
    enableDebouncing = true,
    debounceDelay = 300,
    onPerformanceIssue
  } = options;

  const renderStartTime = useRef<number>(0);
  const renderCount = useRef<number>(0);
  const [metrics, setMetrics] = useState<PerformanceMetrics>({
    renderTime: 0,
    memoryUsage: 0,
    componentCount: 0,
    reRenderCount: 0
  });

  // Performance monitoring
  useEffect(() => {
    if (!enableMetrics) return;

    const observer = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      entries.forEach((entry) => {
        if (entry.entryType === 'measure') {
          const renderTime = entry.duration;
          const memoryUsage = (performance as any).memory?.usedJSHeapSize || 0;
          
          const newMetrics = {
            renderTime,
            memoryUsage: memoryUsage / 1024 / 1024, // Convert to MB
            componentCount: document.querySelectorAll('[data-component]').length,
            reRenderCount: renderCount.current
          };

          setMetrics(newMetrics);

          // Check for performance issues
          if (renderTime > 100 || memoryUsage > 50) {
            onPerformanceIssue?.(newMetrics);
          }
        }
      });
    });

    observer.observe({ entryTypes: ['measure'] });

    return () => observer.disconnect();
  }, [enableMetrics, onPerformanceIssue]);

  // Track render performance
  const startRender = useCallback(() => {
    renderStartTime.current = performance.now();
    performance.mark('render-start');
  }, []);

  const endRender = useCallback(() => {
    const renderTime = performance.now() - renderStartTime.current;
    performance.mark('render-end');
    performance.measure('render-duration', 'render-start', 'render-end');
    renderCount.current++;
  }, []);

  // Debounced function wrapper
  const debounce = useCallback(<T extends (...args: any[]) => any>(
    func: T,
    delay: number = debounceDelay
  ): T => {
    if (!enableDebouncing) return func;

    let timeoutId: NodeJS.Timeout;
    return ((...args: Parameters<T>) => {
      clearTimeout(timeoutId);
      timeoutId = setTimeout(() => func(...args), delay);
    }) as T;
  }, [enableDebouncing, debounceDelay]);

  // Throttled function wrapper
  const throttle = useCallback(<T extends (...args: any[]) => any>(
    func: T,
    delay: number = debounceDelay
  ): T => {
    let lastCall = 0;
    return ((...args: Parameters<T>) => {
      const now = Date.now();
      if (now - lastCall >= delay) {
        lastCall = now;
        return func(...args);
      }
    }) as T;
  }, [debounceDelay]);

  // Memoization helper
  const memoize = useCallback(<T extends (...args: any[]) => any>(
    func: T,
    keyGenerator?: (...args: Parameters<T>) => string
  ): T => {
    if (!enableMemoization) return func;

    const cache = new Map<string, ReturnType<T>>();
    
    return ((...args: Parameters<T>) => {
      const key = keyGenerator ? keyGenerator(...args) : JSON.stringify(args);
      
      if (cache.has(key)) {
        return cache.get(key);
      }
      
      const result = func(...args);
      cache.set(key, result);
      
      // Limit cache size
      if (cache.size > 100) {
        const firstKey = cache.keys().next().value;
        cache.delete(firstKey);
      }
      
      return result;
    }) as T;
  }, [enableMemoization]);

  // Virtual scrolling helper
  const useVirtualization = useCallback((
    items: any[],
    itemHeight: number,
    containerHeight: number,
    scrollTop: number
  ) => {
    if (!enableVirtualization) {
      return {
        visibleItems: items,
        startIndex: 0,
        endIndex: items.length - 1,
        totalHeight: items.length * itemHeight
      };
    }

    const startIndex = Math.floor(scrollTop / itemHeight);
    const endIndex = Math.min(
      startIndex + Math.ceil(containerHeight / itemHeight) + 1,
      items.length - 1
    );

    const visibleItems = items.slice(startIndex, endIndex + 1);
    const totalHeight = items.length * itemHeight;

    return {
      visibleItems,
      startIndex,
      endIndex,
      totalHeight
    };
  }, [enableVirtualization]);

  // Intersection Observer for lazy loading
  const useIntersectionObserver = useCallback((
    callback: (entries: IntersectionObserverEntry[]) => void,
    options: IntersectionObserverInit = {}
  ) => {
    const observer = new IntersectionObserver(callback, {
      rootMargin: '50px',
      threshold: 0.1,
      ...options
    });

    return observer;
  }, []);

  // Cleanup function
  const cleanup = useCallback(() => {
    // Clear performance marks
    performance.clearMarks();
    performance.clearMeasures();
    
    // Reset counters
    renderCount.current = 0;
    renderStartTime.current = 0;
  }, []);

  return {
    metrics,
    startRender,
    endRender,
    debounce,
    throttle,
    memoize,
    useVirtualization,
    useIntersectionObserver,
    cleanup
  };
};

export default usePerformanceOptimization;
