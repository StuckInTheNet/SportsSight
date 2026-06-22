import { useCallback, useEffect, useMemo, useState } from "react";
import { useLiveGame } from "@/hooks/useWebSocket";
import PlayerCard from "@/components/PlayerCard";
import FatigueTimeline from "@/components/FatigueTimeline";
import AlertFeed from "@/components/AlertFeed";
import VideoPlayer from "@/components/VideoPlayer";
import type { Player } from "@/types";
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

  // Video sync
  const [videoTimeMs, setVideoTimeMs] = useState(0);

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
      const [gamesList, playersList] = await Promise.all([
        api.listGames(),
        api.listPlayers(),
      ]);
      setGames(gamesList);
      const pMap = new Map<string, PlayerInfo>();
      for (const p of playersList) pMap.set(p.id, p);
      setDbPlayers(pMap);
    } catch (e: any) {
      setAuthError(e.message || "Authentication failed");
      setAuthenticated(false);
    }
  }, []);

  useEffect(() => {
    if (apiKey) handleAuth(apiKey);
  }, []); // eslint-disable-line

  // --- Load fatigue data ---
  useEffect(() => {
    if (!selectedGameId || !authenticated) return;
    setLoading(true);
    setTimelineData([]);

    api
      .getGameFatigue(selectedGameId, { limit: "10000" })
      .then((records) => {
        const byTime = new Map<number, Map<string, FatigueRecord>>();
        const byPlayer = new Map<string, FatigueRecord[]>();

        for (const r of records) {
          if (!byTime.has(r.timestamp_ms)) byTime.set(r.timestamp_ms, new Map());
          byTime.get(r.timestamp_ms)!.set(r.player_id, r);
          if (!byPlayer.has(r.player_id)) byPlayer.set(r.player_id, []);
          byPlayer.get(r.player_id)!.push(r);
        }
        setDbFatigue(byPlayer);

        const timestamps = Array.from(byTime.keys()).sort((a, b) => a - b);
        const step = Math.max(1, Math.floor(timestamps.length / 200));
        const timeline: any[] = [];
        for (let i = 0; i < timestamps.length; i += step) {
          const ts = timestamps[i];
          const entry: any = { time: fmtTime(ts), timestamp_ms: ts };
          for (const [pid, rec] of byTime.get(ts)!) {
            entry[pid] = rec.fatigue_score;
          }
          timeline.push(entry);
        }
        setTimelineData(timeline);
      })
      .catch((e) => console.error("Load failed:", e))
      .finally(() => setLoading(false));
  }, [selectedGameId, authenticated]);

  // Live scores into timeline
  useEffect(() => {
    if (Object.keys(liveScores).length === 0) return;
    const entry: any = { time: new Date().toLocaleTimeString(), timestamp_ms: Date.now() };
    for (const [pid, score] of Object.entries(liveScores)) entry[pid] = score.score;
    setTimelineData((prev) => [...prev.slice(-200), entry]);
  }, [liveScores]);

  // --- Compute scores at current video time (or latest) ---
  const scoresAtTime = useMemo(() => {
    const result = new Map<string, any>();

    for (const [pid, records] of dbFatigue) {
      // Find the record closest to video time (or last record if no video sync)
      let best = records[records.length - 1];
      if (videoTimeMs > 0) {
        for (const r of records) {
          if (r.timestamp_ms <= videoTimeMs) best = r;
          else break;
        }
      }
      if (best) {
        result.set(pid, {
          score: best.fatigue_score,
          confidence: best.confidence,
          trend: best.trend,
          level: best.fatigue_score >= 75 ? "critical" : best.fatigue_score >= 55 ? "high" : best.fatigue_score >= 30 ? "moderate" : "low",
          predicted_score_5min: 0,
          contributing_factors: best.contributing_factors || {},
        });
      }
    }

    // Live overrides
    for (const [pid, score] of Object.entries(liveScores)) result.set(pid, score);
    return result;
  }, [dbFatigue, liveScores, videoTimeMs]);

  // Top 10 players
  const displayPlayers: Player[] = useMemo(() => {
    return Array.from(scoresAtTime.entries())
      .sort(([, a], [, b]) => b.score - a.score)
      .slice(0, 10)
      .map(([pid]) => {
        const p = dbPlayers.get(pid);
        if (p) return { ...p };
        const num = parseInt(pid, 10);
        return { id: pid, name: `Player ${pid.slice(0, 8)}`, jersey_number: isNaN(num) || num >= 100 ? null : num, position: null };
      });
  }, [scoresAtTime, dbPlayers]);

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
                  if (e.key === "Enter") handleAuth((e.target as HTMLInputElement).value.trim());
                }}
              />
            </div>
            <button
              className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-2 rounded-lg text-sm font-medium"
              onClick={(e) => {
                const input = (e.target as HTMLElement).closest("div")?.querySelector("input") as HTMLInputElement;
                if (input) handleAuth(input.value.trim());
              }}
            >
              Connect
            </button>
          </div>
          {authError && <p className="text-red-400 text-sm mt-3">{authError}</p>}
        </div>
      </div>
    );
  }

  // --- Main dashboard ---
  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-3">
        <div className="flex items-center justify-between max-w-[1600px] mx-auto">
          <div className="flex items-center gap-3">
            <Activity className="w-6 h-6 text-blue-500" />
            <h1 className="text-xl font-bold">SportsSight</h1>
          </div>
          <div className="flex items-center gap-4">
            <select
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm focus:outline-none focus:border-blue-500"
              value={selectedGameId || ""}
              onChange={(e) => {
                setSelectedGameId(e.target.value || null);
                setVideoTimeMs(0);
              }}
            >
              <option value="">Select a game...</option>
              {games.map((g) => (
                <option key={g.id} value={g.id}>
                  {g.opponent || "Game"} — {new Date(g.date).toLocaleDateString()} ({g.status})
                </option>
              ))}
            </select>
            <div className="flex items-center gap-1.5">
              {loading ? (
                <><Loader2 className="w-4 h-4 text-blue-400 animate-spin" /><span className="text-xs text-blue-400">Loading...</span></>
              ) : connected ? (
                <><Wifi className="w-4 h-4 text-green-400" /><span className="text-xs text-green-400">Live</span></>
              ) : selectedGameId ? (
                <><Radio className="w-4 h-4 text-gray-400" /><span className="text-xs text-gray-400">Historical</span></>
              ) : (
                <><Radio className="w-4 h-4 text-gray-500" /><span className="text-xs text-gray-500">Standby</span></>
              )}
              {scoresAtTime.size > 0 && (
                <span className="text-xs text-gray-500 ml-1">
                  <Users className="w-3 h-3 inline mr-0.5" />{scoresAtTime.size}
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Two-column layout: analytics left, video+alerts right */}
      <main className="max-w-[1600px] mx-auto px-6 py-5">
        {wsError && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-sm text-red-400 mb-4">
            {wsError}
          </div>
        )}

        {!selectedGameId ? (
          <div className="text-gray-500 text-sm py-20 text-center border border-gray-800 rounded-xl">
            Select a game above to view fatigue analytics
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
            {/* LEFT COLUMN — analytics (2/3) */}
            <div className="lg:col-span-2 space-y-5">
              {/* Player cards */}
              <section>
                <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wide mb-3">
                  Top Players by Fatigue
                  {videoTimeMs > 0 && (
                    <span className="text-blue-400 ml-2 normal-case tracking-normal">
                      at {fmtTime(videoTimeMs)}
                    </span>
                  )}
                </h2>
                {displayPlayers.length > 0 ? (
                  <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3">
                    {displayPlayers.map((player) => (
                      <PlayerCard
                        key={player.id}
                        player={player}
                        fatigue={scoresAtTime.get(player.id) ?? null}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="text-gray-500 text-sm py-8 text-center border border-gray-800 rounded-xl">
                    Loading fatigue data...
                  </div>
                )}
              </section>

              {/* Timeline */}
              <FatigueTimeline data={timelineData} playerIds={timelinePlayerIds} />
            </div>

            {/* RIGHT COLUMN — video + alerts (1/3) */}
            <div className="space-y-5">
              <VideoPlayer onTimeUpdate={setVideoTimeMs} />
              <AlertFeed alerts={alerts} />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function fmtTime(ms: number): string {
  const s = Math.floor(ms / 1000);
  return `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;
}
