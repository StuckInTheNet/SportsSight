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
  // Track the gameId that the current connection belongs to, so stale
  // reconnect timers from a previous gameId are discarded.
  const activeGameId = useRef<string | null>(null);

  useEffect(() => {
    activeGameId.current = gameId;

    // Clean up any previous connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    clearTimeout(reconnectTimeout.current);

    if (!gameId) {
      setConnected(false);
      setScores({});
      setAlerts([]);
      return;
    }

    function connect() {
      // Bail if the gameId changed since this connect was scheduled
      if (activeGameId.current !== gameId) return;

      const apiKey = localStorage.getItem("sportssight_api_key") || "";
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${protocol}//${window.location.host}/ws/games/${gameId}/live?api_key=${encodeURIComponent(apiKey)}`;

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
        // Only reconnect if this gameId is still the active one
        if (activeGameId.current === gameId) {
          reconnectTimeout.current = window.setTimeout(connect, 3000);
        }
      };

      ws.onerror = () => {
        setError("WebSocket connection failed");
        ws.close();
      };
    }

    connect();

    return () => {
      activeGameId.current = null;
      wsRef.current?.close();
      wsRef.current = null;
      clearTimeout(reconnectTimeout.current);
    };
  }, [gameId]);

  return { scores, alerts, connected, error };
}
