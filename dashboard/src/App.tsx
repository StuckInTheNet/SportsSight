import { useEffect, useState } from "react";
import { useLiveGame } from "@/hooks/useWebSocket";
import PlayerCard from "@/components/PlayerCard";
import FatigueTimeline from "@/components/FatigueTimeline";
import AlertFeed from "@/components/AlertFeed";
import type { Player, FatigueScore, Alert } from "@/types";
import { Activity, Radio, Wifi, WifiOff } from "lucide-react";

// Demo players for initial UI (replaced by API data in production)
const DEMO_PLAYERS: Player[] = [
  { id: "1", name: "Demo Player 1", jersey_number: 23, position: "SF" },
  { id: "2", name: "Demo Player 2", jersey_number: 30, position: "PG" },
  { id: "3", name: "Demo Player 3", jersey_number: 34, position: "PF" },
  { id: "4", name: "Demo Player 4", jersey_number: 11, position: "SG" },
  { id: "5", name: "Demo Player 5", jersey_number: 6, position: "C" },
];

export default function App() {
  const [gameId, setGameId] = useState<string | null>(null);
  const [players] = useState<Player[]>(DEMO_PLAYERS);
  const [timelineData, setTimelineData] = useState<any[]>([]);
  const { scores, alerts, connected, error } = useLiveGame(gameId);

  // Accumulate timeline data from live scores
  useEffect(() => {
    if (Object.keys(scores).length === 0) return;

    const entry: any = {
      time: new Date().toLocaleTimeString(),
      timestamp_ms: Date.now(),
    };
    for (const [pid, score] of Object.entries(scores)) {
      entry[pid] = score.score;
    }

    setTimelineData((prev) => [...prev.slice(-200), entry]);
  }, [scores]);

  const playerIds = Object.keys(scores).length > 0
    ? Object.keys(scores)
    : players.map((p) => p.id);

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
            <input
              type="text"
              placeholder="Enter Game ID..."
              className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm w-48 focus:outline-none focus:border-blue-500"
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  setGameId((e.target as HTMLInputElement).value || null);
                }
              }}
            />

            {/* Connection status */}
            <div className="flex items-center gap-1.5">
              {connected ? (
                <>
                  <Wifi className="w-4 h-4 text-green-400" />
                  <span className="text-xs text-green-400">Live</span>
                </>
              ) : gameId ? (
                <>
                  <WifiOff className="w-4 h-4 text-red-400" />
                  <span className="text-xs text-red-400">Disconnected</span>
                </>
              ) : (
                <>
                  <Radio className="w-4 h-4 text-gray-500" />
                  <span className="text-xs text-gray-500">Standby</span>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-6 py-6 space-y-6">
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-sm text-red-400">
            {error}
          </div>
        )}

        {/* Player cards grid */}
        <section>
          <h2 className="text-lg font-semibold mb-3">Players</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
            {players.map((player) => (
              <PlayerCard
                key={player.id}
                player={player}
                fatigue={scores[player.id] ?? null}
              />
            ))}
          </div>
        </section>

        {/* Timeline + Alerts row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <FatigueTimeline
              data={timelineData}
              playerIds={playerIds}
            />
          </div>
          <div>
            <AlertFeed alerts={alerts} />
          </div>
        </div>
      </main>
    </div>
  );
}
