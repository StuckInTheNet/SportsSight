import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useLiveGame } from "@/hooks/useWebSocket";
import PlayerCard from "@/components/PlayerCard";
import FatigueTimeline from "@/components/FatigueTimeline";
import AlertFeed from "@/components/AlertFeed";
import type { Player, FatigueScore } from "@/types";
import {
  api,
  setApiKey,
  getApiKey,
  type GameInfo,
  type PlayerInfo,
  type FatigueRecord,
} from "@/services/api";
import { Activity, Key, Loader2, Radio, Users, Wifi, WifiOff } from "lucide-react";

export default function App() {
  // Auth
  const [apiKey, setApiKeyState] = useState(getApiKey());
  const [authError, setAuthError] = useState<string | null>(null);
  const [authenticated, setAuthenticated] = useState(false);

  // Data
  const [games, setGames] = useState<GameInfo[]>([]);
  const [selectedGameId, setSelectedGameId] = useState<string | null>(null);
  const [dbPlayers, setDbPlayers] = useState<Map<string, PlayerInfo>>(new Map());
  const [dbFatigue, setDbFatigue] = useState<Map<string, FatigueRecord[]>>(new Map());
  const [loading, setLoading] = useState(false);

  // Live
  const { scores: liveScores, alerts, connected, error: wsError } = useLiveGame(
    authenticated ? selectedGameId : null
  );
  const [timelineData, setTimelineData] = useState<any[]>([]);

  // --- Auth flow ---
  const handleAuth = useCallback(async (key: string) => {
    setApiKey(key);
    setApiKeyState(key);
    try {
      await api.getTeam();
      setAuthenticated(true);
      setAuthError(null);

      // Load games + players
      const [gamesList, playersList] = await Promise.all([
        api.listGames(),
        api.listPlayers(),
      ]);
      setGames(gamesList);

      const pMap = new Map<string, PlayerInfo>();
      for (const p of playersList) {
        pMap.set(p.id, p);
      }
      setDbPlayers(pMap);
    } catch (e: any) {
      setAuthError(e.message || "Authentication failed");
      setAuthenticated(false);
    }
  }, []);

  // Auto-auth on mount if key exists
  useEffect(() => {
    if (apiKey) handleAuth(apiKey);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // --- Load fatigue data when game is selected ---
  useEffect(() => {
    if (!selectedGameId || !authenticated) return;

    setLoading(true);
    setTimelineData([]);

    api
      .getGameFatigue(selectedGameId, { limit: "10000" })
      .then((records) => {
        // Group by timestamp
        const byTime = new Map<number, Map<string, FatigueRecord>>();
        const byPlayer = new Map<string, FatigueRecord[]>();

        for (const r of records) {
          // By time
          if (!byTime.has(r.timestamp_ms)) {
            byTime.set(r.timestamp_ms, new Map());
          }
          byTime.get(r.timestamp_ms)!.set(r.player_id, r);

          // By player
          if (!byPlayer.has(r.player_id)) {
            byPlayer.set(r.player_id, []);
          }
          byPlayer.get(r.player_id)!.push(r);
        }
        setDbFatigue(byPlayer);

        // Build timeline from DB records (sampled to ~200 points)
        const timestamps = Array.from(byTime.keys()).sort((a, b) => a - b);
        const step = Math.max(1, Math.floor(timestamps.length / 200));
        const timeline: any[] = [];

        for (let i = 0; i < timestamps.length; i += step) {
          const ts = timestamps[i];
          const playerScores = byTime.get(ts)!;
          const entry: any = {
            time: formatGameTime(ts),
            timestamp_ms: ts,
          };
          for (const [pid, rec] of playerScores) {
            entry[pid] = rec.fatigue_score;
          }
          timeline.push(entry);
        }
        setTimelineData(timeline);
      })
      .catch((e) => console.error("Failed to load fatigue data:", e))
      .finally(() => setLoading(false));
  }, [selectedGameId, authenticated]);

  // Accumulate live WebSocket data into timeline
  useEffect(() => {
    if (Object.keys(liveScores).length === 0) return;

    const entry: any = {
      time: new Date().toLocaleTimeString(),
      timestamp_ms: Date.now(),
    };
    for (const [pid, score] of Object.entries(liveScores)) {
      entry[pid] = score.score;
    }
    setTimelineData((prev) => [...prev.slice(-200), entry]);
  }, [liveScores]);

  // --- Compute display data ---

  // Find the latest fatigue score per player (from DB or live)
  const latestScores: Map<string, { score: number; confidence: number; trend: string; level: string; predicted_score_5min: number; contributing_factors: Record<string, number> }> = useMemo(() => {
    const result = new Map<string, any>();

    // From DB records: take the last record per player
    for (const [pid, records] of dbFatigue) {
      if (records.length > 0) {
        const last = records[records.length - 1];
        result.set(pid, {
          score: last.fatigue_score,
          confidence: last.confidence,
          trend: last.trend,
          level: last.fatigue_score >= 75 ? "critical" : last.fatigue_score >= 55 ? "high" : last.fatigue_score >= 30 ? "moderate" : "low",
          predicted_score_5min: 0,
          contributing_factors: last.contributing_factors || {},
        });
      }
    }

    // Live scores override DB
    for (const [pid, score] of Object.entries(liveScores)) {
      result.set(pid, score);
    }

    return result;
  }, [dbFatigue, liveScores]);

  // Top 10 players by fatigue score
  const displayPlayers: Player[] = useMemo(() => {
    const entries = Array.from(latestScores.entries())
      .sort(([, a], [, b]) => b.score - a.score)
      .slice(0, 10);

    return entries.map(([pid]) => {
      const dbPlayer = dbPlayers.get(pid);
      if (dbPlayer) return { ...dbPlayer };

      const num = parseInt(pid, 10);
      return {
        id: pid,
        name: `Player ${pid.slice(0, 8)}`,
        jersey_number: isNaN(num) || num >= 100 ? null : num,
        position: null,
      };
    });
  }, [latestScores, dbPlayers]);

  // Player IDs for timeline chart
  const timelinePlayerIds = displayPlayers.map((p) => p.id);

  // --- Auth screen ---
  if (!authenticated) {
    return (
      <div className="min-h-screen bg-gray-950 text-gray-100 flex items-center justify-center">
        <div className="bg-gray-900 border border-gray-800 rounded-xl p-8 w-full max-w-md">
          <div className="flex items-center gap-3 mb-6">
            <Activity className="w-8 h-8 text-blue-500" />
            <div>
              <h1 className="text-2xl font-bold">SportsSight</h1>
              <p className="text-sm text-gray-500">Fatigue Analytics Dashboard</p>
            </div>
          </div>

          <label className="block text-sm text-gray-400 mb-2">API Key</label>
          <div className="flex gap-2">
            <div className="flex-1 relative">
              <Key className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
              <input
                type="password"
                placeholder="ss_..."
                defaultValue={apiKey}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg pl-10 pr-3 py-2 text-sm focus:outline-none focus:border-blue-500"
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleAuth((e.target as HTMLInputElement).value.trim());
                  }
                }}
              />
            </div>
            <button
              className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium"
              onClick={(e) => {
                const input = (e.target as HTMLElement)
                  .closest("div")
                  ?.querySelector("input") as HTMLInputElement;
                if (input) handleAuth(input.value.trim());
              }}
            >
              Connect
            </button>
          </div>
          {authError && (
            <p className="text-red-400 text-sm mt-3">{authError}</p>
          )}
        </div>
      </div>
    );
  }

  // --- Main dashboard ---
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-blue-500" />
            <h1 className="text-xl font-bold">SportsSight</h1>
            <span className="text-sm text-gray-500">Fatigue Analytics</span>
          </div>

          <div className="flex items-center gap-4">
            {/* Game selector */}
            <select
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-blue-500"
              value={selectedGameId || ""}
              onChange={(e) => setSelectedGameId(e.target.value || null)}
            >
              <option value="">Select a game...</option>
              {games.map((g) => (
                <option key={g.id} value={g.id}>
                  {g.opponent || "Game"} — {new Date(g.date).toLocaleDateString()}{" "}
                  ({g.status})
                </option>
              ))}
            </select>

            {/* Connection status */}
            <div className="flex items-center gap-1.5">
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />
                  <span className="text-xs text-blue-400">Loading...</span>
                </>
              ) : connected ? (
                <>
                  <Wifi className="w-4 h-4 text-green-400" />
                  <span className="text-xs text-green-400">Live</span>
                </>
              ) : selectedGameId ? (
                <>
                  <Radio className="w-4 h-4 text-gray-400" />
                  <span className="text-xs text-gray-400">Historical</span>
                </>
              ) : (
                <>
                  <Radio className="w-4 h-4 text-gray-500" />
                  <span className="text-xs text-gray-500">Standby</span>
                </>
              )}
              {latestScores.size > 0 && (
                <span className="text-xs text-gray-500 ml-1">
                  <Users className="w-3 h-3 inline mr-0.5" />
                  {latestScores.size}
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {wsError && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-sm text-red-400">
            {wsError}
          </div>
        )}

        {/* Player cards */}
        <section>
          <h2 className="text-lg font-semibold mb-3">
            Players
            {displayPlayers.length > 0 && (
              <span className="text-sm font-normal text-gray-500 ml-2">
                top {displayPlayers.length} by fatigue
              </span>
            )}
          </h2>
          {displayPlayers.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
              {displayPlayers.map((player) => (
                <PlayerCard
                  key={player.id}
                  player={player}
                  fatigue={latestScores.get(player.id) ?? null}
                />
              ))}
            </div>
          ) : (
            <div className="text-gray-500 text-sm py-12 text-center border border-gray-800 rounded-xl">
              {selectedGameId
                ? "Loading fatigue data..."
                : "Select a game above to view fatigue analytics"}
            </div>
          )}
        </section>

        {/* Timeline + Alerts */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <FatigueTimeline data={timelineData} playerIds={timelinePlayerIds} />
          </div>
          <div>
            <AlertFeed alerts={alerts} />
          </div>
        </div>
      </main>
    </div>
  );
}

function formatGameTime(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${min}:${sec.toString().padStart(2, "0")}`;
}
