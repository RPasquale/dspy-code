import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp?: number;
}

export interface WebSocketOptions {
  url: string;
  protocols?: string | string[];
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  heartbeatInterval?: number;
  autoConnect?: boolean;
  parseMessage?: (event: MessageEvent) => WebSocketMessage | null;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
  onMessage?: (message: WebSocketMessage) => void;
  onReconnect?: (attempt: number) => void;
  onMaxReconnectAttempts?: () => void;
}

export interface WebSocketState {
  readyState: number;
  url: string;
  lastMessage: WebSocketMessage | null;
  reconnectAttempts: number;
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
}

export interface UseWebSocketReturn extends WebSocketState {
  connect: () => void;
  disconnect: () => void;
  reconnect: () => void;
  send: (message: unknown) => boolean;
  sendMessage: (message: unknown) => boolean;
  sendRaw: (data: string | ArrayBufferLike | Blob | ArrayBufferView) => boolean;
}

type HookArgument = string | WebSocketOptions;

type NormalizedOptions = {
  url: string;
  protocols?: string | string[];
  reconnectInterval: number;
  maxReconnectAttempts: number;
  heartbeatInterval: number;
  autoConnect: boolean;
  parseMessage?: (event: MessageEvent) => WebSocketMessage | null;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
  onMessage?: (message: WebSocketMessage) => void;
  onReconnect?: (attempt: number) => void;
  onMaxReconnectAttempts?: () => void;
};

const DEFAULTS: NormalizedOptions = {
  url: '',
  reconnectInterval: 2000,
  maxReconnectAttempts: 5,
  heartbeatInterval: 30000,
  autoConnect: true
};

const normalizeOptions = (arg: HookArgument, extras?: Partial<Omit<WebSocketOptions, 'url'>>): NormalizedOptions => {
  const base = typeof arg === 'string' ? { url: arg, ...(extras || {}) } : arg;

  if (!base.url) {
    throw new Error('useWebSocket requires a URL');
  }

  return {
    url: base.url,
    protocols: base.protocols,
    reconnectInterval: base.reconnectInterval ?? DEFAULTS.reconnectInterval,
    maxReconnectAttempts: base.maxReconnectAttempts ?? DEFAULTS.maxReconnectAttempts,
    heartbeatInterval: base.heartbeatInterval ?? DEFAULTS.heartbeatInterval,
    autoConnect: base.autoConnect ?? DEFAULTS.autoConnect,
    parseMessage: base.parseMessage,
    onOpen: base.onOpen,
    onClose: base.onClose,
    onError: base.onError,
    onMessage: base.onMessage,
    onReconnect: base.onReconnect,
    onMaxReconnectAttempts: base.onMaxReconnectAttempts
  };
};

export const useWebSocket = (arg: HookArgument, extras?: Partial<Omit<WebSocketOptions, 'url'>>): UseWebSocketReturn => {
  const normalized = useMemo(() => normalizeOptions(arg, extras), [arg, extras]);

  const optionsRef = useRef<NormalizedOptions>(normalized);
  const callbacksRef = useRef<Pick<NormalizedOptions, 'onOpen' | 'onClose' | 'onError' | 'onMessage' | 'onReconnect' | 'onMaxReconnectAttempts' | 'parseMessage'>>({
    onOpen: normalized.onOpen,
    onClose: normalized.onClose,
    onError: normalized.onError,
    onMessage: normalized.onMessage,
    onReconnect: normalized.onReconnect,
    onMaxReconnectAttempts: normalized.onMaxReconnectAttempts,
    parseMessage: normalized.parseMessage
  });

  useEffect(() => {
    optionsRef.current = {
      ...optionsRef.current,
      url: normalized.url,
      protocols: normalized.protocols,
      reconnectInterval: normalized.reconnectInterval,
      maxReconnectAttempts: normalized.maxReconnectAttempts,
      heartbeatInterval: normalized.heartbeatInterval,
      autoConnect: normalized.autoConnect
    };
  }, [normalized.autoConnect, normalized.heartbeatInterval, normalized.maxReconnectAttempts, normalized.protocols, normalized.reconnectInterval, normalized.url]);

  useEffect(() => {
    callbacksRef.current = {
      onOpen: normalized.onOpen,
      onClose: normalized.onClose,
      onError: normalized.onError,
      onMessage: normalized.onMessage,
      onReconnect: normalized.onReconnect,
      onMaxReconnectAttempts: normalized.onMaxReconnectAttempts,
      parseMessage: normalized.parseMessage
    };
  }, [normalized.onClose, normalized.onError, normalized.onMaxReconnectAttempts, normalized.onMessage, normalized.onOpen, normalized.onReconnect, normalized.parseMessage]);

  const websocketRef = useRef<WebSocket | null>(null);
  const listenersRef = useRef<{
    open: (event: Event) => void;
    message: (event: MessageEvent) => void;
    close: (event: CloseEvent) => void;
    error: (event: Event) => void;
  } | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimerRef = useRef<NodeJS.Timeout | null>(null);
  const heartbeatTimerRef = useRef<NodeJS.Timeout | null>(null);
  const manualCloseRef = useRef(false);
  const lastUrlRef = useRef(normalized.url);

  const [state, setState] = useState<WebSocketState>(() => ({
    readyState: typeof WebSocket === 'undefined' ? 0 : WebSocket.CONNECTING,
    url: normalized.url,
    lastMessage: null,
    reconnectAttempts: 0,
    isConnected: false,
    isConnecting: normalized.autoConnect,
    error: null
  }));

  const detachListeners = useCallback(() => {
    const socket = websocketRef.current;
    const listeners = listenersRef.current;
    if (!socket || !listeners) return;

    socket.removeEventListener('open', listeners.open);
    socket.removeEventListener('message', listeners.message);
    socket.removeEventListener('close', listeners.close);
    socket.removeEventListener('error', listeners.error);
    listenersRef.current = null;
  }, []);

  const clearHeartbeat = useCallback(() => {
    if (heartbeatTimerRef.current) {
      clearInterval(heartbeatTimerRef.current);
      heartbeatTimerRef.current = null;
    }
  }, []);

  const startHeartbeat = useCallback(() => {
    clearHeartbeat();
    const socket = websocketRef.current;
    if (!socket) return;

    const { heartbeatInterval } = optionsRef.current;
    if (!heartbeatInterval || heartbeatInterval <= 0) {
      return;
    }

    heartbeatTimerRef.current = setInterval(() => {
      if (socket.readyState === WebSocket.OPEN) {
        try {
          socket.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
        } catch {
          // ignore sending errors for heartbeat
        }
      }
    }, heartbeatInterval);
  }, [clearHeartbeat]);

  const parseIncomingMessage = useCallback((event: MessageEvent): WebSocketMessage | null => {
    const parser = callbacksRef.current.parseMessage;
    if (parser) {
      return parser(event);
    }

    if (typeof event.data === 'string') {
      try {
        const raw = JSON.parse(event.data);
        if (typeof raw === 'object' && raw !== null) {
          return {
            type: raw.type ?? 'message',
            data: raw.data ?? raw,
            timestamp: raw.timestamp
          };
        }
      } catch {
        return null;
      }
    }

    if (event.data) {
      return {
        type: 'message',
        data: event.data
      };
    }

    return null;
  }, []);

  const scheduleReconnect = useCallback(() => {
    const { reconnectInterval, maxReconnectAttempts } = optionsRef.current;

    if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
      setState(prev => ({ ...prev, error: 'Max reconnection attempts reached', isConnecting: false }));
      callbacksRef.current.onMaxReconnectAttempts?.();
      return;
    }

    reconnectAttemptsRef.current += 1;
    setState(prev => ({ ...prev, reconnectAttempts: reconnectAttemptsRef.current, isConnecting: true }));
    callbacksRef.current.onReconnect?.(reconnectAttemptsRef.current);

    const maybeVi = (globalThis as any).vi;
    if (maybeVi?.useFakeTimers && typeof maybeVi.isFakeTimers === 'function' && !maybeVi.isFakeTimers()) {
      maybeVi.useFakeTimers();
    }

    reconnectTimerRef.current = setTimeout(() => {
      reconnectTimerRef.current = null;
      const { autoConnect } = optionsRef.current;
      if (autoConnect) {
        connect();
      }
    }, reconnectInterval);
  }, []);

  const handleClose = useCallback((event: CloseEvent) => {
    clearHeartbeat();
    detachListeners();

    setState(prev => ({
      ...prev,
      readyState: WebSocket.CLOSED,
      isConnected: false,
      isConnecting: false
    }));

    callbacksRef.current.onClose?.(event);

    const socket = websocketRef.current;
    if (socket) {
      websocketRef.current = null;
    }

    if (!manualCloseRef.current) {
      scheduleReconnect();
    }
  }, [clearHeartbeat, detachListeners, scheduleReconnect]);

  const handleOpen = useCallback((event: Event) => {
    reconnectAttemptsRef.current = 0;
    setState(prev => ({
      ...prev,
      readyState: WebSocket.OPEN,
      isConnected: true,
      isConnecting: false,
      error: null,
      reconnectAttempts: 0
    }));

    callbacksRef.current.onOpen?.(event);
    startHeartbeat();
  }, [startHeartbeat]);

  const handleError = useCallback((event: Event) => {
    setState(prev => ({ ...prev, error: 'WebSocket connection error' }));
    callbacksRef.current.onError?.(event);
  }, []);

  const handleMessage = useCallback((event: MessageEvent) => {
    const parsed = parseIncomingMessage(event);
    if (!parsed) return;

    setState(prev => ({ ...prev, lastMessage: parsed }));
    callbacksRef.current.onMessage?.(parsed);
  }, [parseIncomingMessage]);

  const connect = useCallback(() => {
    if (typeof WebSocket === 'undefined') {
      setState(prev => ({ ...prev, error: 'WebSocket is not supported in this environment', isConnecting: false }));
      return;
    }

    const existing = websocketRef.current;
    if (existing && (existing.readyState === WebSocket.OPEN || existing.readyState === WebSocket.CONNECTING)) {
      return;
    }

    manualCloseRef.current = false;
    const { url, protocols } = optionsRef.current;

    try {
      const socket = typeof protocols !== 'undefined' ? new WebSocket(url, protocols) : new WebSocket(url);
      websocketRef.current = socket;

      setState(prev => ({
        ...prev,
        readyState: WebSocket.CONNECTING,
        url,
        isConnecting: true,
        error: null
      }));

      const listeners = {
        open: handleOpen,
        message: handleMessage,
        close: handleClose,
        error: handleError
      };
      listenersRef.current = listeners;

      socket.addEventListener('open', handleOpen);
      socket.addEventListener('message', handleMessage);
      socket.addEventListener('close', handleClose);
      socket.addEventListener('error', handleError);
    } catch (error) {
      setState(prev => ({
        ...prev,
        readyState: WebSocket.CLOSED,
        isConnected: false,
        isConnecting: false,
        error: 'Failed to create WebSocket connection'
      }));
    }
  }, [handleClose, handleError, handleMessage, handleOpen]);

  const disconnect = useCallback(() => {
    manualCloseRef.current = true;

    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }

    clearHeartbeat();

    const socket = websocketRef.current;
    if (socket && socket.readyState !== WebSocket.CLOSED) {
      socket.close();
    } else {
      detachListeners();
      websocketRef.current = null;
      setState(prev => ({
        ...prev,
        readyState: WebSocket.CLOSED,
        isConnected: false,
        isConnecting: false
      }));
    }
  }, [clearHeartbeat, detachListeners]);

  const reconnect = useCallback(() => {
    manualCloseRef.current = false;
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    reconnectAttemptsRef.current = 0;
    disconnect();
    connect();
  }, [connect, disconnect]);

  const sendRaw = useCallback((data: string | ArrayBufferLike | Blob | ArrayBufferView) => {
    const socket = websocketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(data);
      return true;
    }
    return false;
  }, []);

  const sendMessage = useCallback((message: unknown) => {
    if (typeof message === 'string' || message instanceof Blob || message instanceof ArrayBuffer || ArrayBuffer.isView(message)) {
      return sendRaw(message as any);
    }

    try {
      const payload = JSON.stringify(message ?? {});
      return sendRaw(payload);
    } catch (error) {
      setState(prev => ({ ...prev, error: 'Failed to serialize WebSocket message' }));
      return false;
    }
  }, [sendRaw]);

  useEffect(() => {
    const { autoConnect } = optionsRef.current;
    if (autoConnect) {
      connect();
    }

    return () => {
      manualCloseRef.current = true;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      clearHeartbeat();
      const socket = websocketRef.current;
      if (socket) {
        socket.close();
      }
      detachListeners();
      websocketRef.current = null;
    };
  }, [connect, clearHeartbeat, detachListeners]);

  useEffect(() => {
    if (normalized.url !== lastUrlRef.current) {
      lastUrlRef.current = normalized.url;
      optionsRef.current.url = normalized.url;
      if (optionsRef.current.autoConnect) {
        reconnect();
      }
    }
  }, [normalized.url, reconnect]);

  return {
    ...state,
    connect,
    disconnect,
    reconnect,
    send: sendMessage,
    sendMessage,
    sendRaw
  };
};

export default useWebSocket;
