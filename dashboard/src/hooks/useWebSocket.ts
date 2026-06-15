import { useCallback, useEffect, useRef, useState } from "react";
import type { FatigueScore, Alert, WSMessage } from "@/types";

interface UseLiveGameReturn {
  scores: Record<string, FatigueScore>;
  alerts: Alert[];
  connected: boolean;
  error: string | null;
}

export function useLiveGame(gameId: string | null): UseLiveGameReturn {
  const [scores, setScores] = useState<Record<string, FatigueScore>>({});
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<number>();

  const connect = useCallback(() => {
    if (!gameId) return;

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/games/${gameId}/live`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);

        if (msg.type === "fatigue_update") {
          const parsed = JSON.parse(msg.data.scores) as Record<string, FatigueScore>;
          setScores(parsed);
        } else if (msg.type === "alert") {
          setAlerts((prev) => [msg.data, ...prev].slice(0, 50));
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      setConnected(false);
      // Auto-reconnect after 3 seconds
      reconnectTimeout.current = window.setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      setError("WebSocket connection failed");
      ws.close();
    };
  }, [gameId]);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
      clearTimeout(reconnectTimeout.current);
    };
  }, [connect]);

  return { scores, alerts, connected, error };
}
