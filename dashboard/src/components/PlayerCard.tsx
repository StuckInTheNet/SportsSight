import type { Player } from "@/types";
import { TrendingUp, TrendingDown, Minus, AlertTriangle } from "lucide-react";

interface FatigueData {
  score: number;
  confidence: number;
  trend: string;
  level: string;
  predicted_score_5min?: number;
  contributing_factors?: Record<string, number>;
}

interface Props {
  player: Player;
  fatigue: FatigueData | null;
}

function levelColor(level: string): string {
  switch (level) {
    case "low": return "text-green-400";
    case "moderate": return "text-yellow-400";
    case "high": return "text-orange-400";
    case "critical": return "text-red-400";
    default: return "text-gray-400";
  }
}

function levelBg(level: string): string {
  switch (level) {
    case "low": return "bg-green-500/10 border-green-500/30";
    case "moderate": return "bg-yellow-500/10 border-yellow-500/30";
    case "high": return "bg-orange-500/10 border-orange-500/30";
    case "critical": return "bg-red-500/10 border-red-500/30 animate-pulse";
    default: return "bg-gray-500/10 border-gray-500/30";
  }
}

function barColor(level: string): string {
  switch (level) {
    case "critical": return "bg-red-500";
    case "high": return "bg-orange-500";
    case "moderate": return "bg-yellow-500";
    default: return "bg-green-500";
  }
}

function TrendIcon({ trend }: { trend: string }) {
  switch (trend) {
    case "rising": return <TrendingUp className="w-4 h-4 text-red-400" />;
    case "declining": return <TrendingDown className="w-4 h-4 text-green-400" />;
    default: return <Minus className="w-4 h-4 text-gray-400" />;
  }
}

export default function PlayerCard({ player, fatigue }: Props) {
  const level = fatigue?.level ?? "low";
  const score = fatigue?.score ?? 0;

  return (
    <div className={`rounded-xl border p-4 ${levelBg(level)}`}>
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="flex items-center gap-2">
            {player.jersey_number != null && (
              <span className="text-2xl font-bold text-gray-300">
                #{player.jersey_number}
              </span>
            )}
            <span className="font-semibold text-sm">{player.name}</span>
          </div>
          {player.position && (
            <span className="text-xs text-gray-500">{player.position}</span>
          )}
        </div>
        {level === "critical" && (
          <AlertTriangle className="w-5 h-5 text-red-400" />
        )}
      </div>

      {/* Fatigue score + bar */}
      <div className="mb-2">
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm text-gray-400">Fatigue</span>
          <div className="flex items-center gap-1">
            <TrendIcon trend={fatigue?.trend ?? "stable"} />
            <span className={`text-xl font-bold ${levelColor(level)}`}>
              {score.toFixed(0)}
            </span>
          </div>
        </div>
        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${barColor(level)}`}
            style={{ width: `${Math.min(score, 100)}%` }}
          />
        </div>
      </div>

      {/* Metrics */}
      {fatigue && (
        <div className="grid grid-cols-2 gap-2 mt-3 text-xs text-gray-400">
          <div>
            <span className="block text-gray-500">Confidence</span>
            {(fatigue.confidence * 100).toFixed(0)}%
          </div>
          <div>
            <span className="block text-gray-500">Level</span>
            <span className={levelColor(level)}>{level}</span>
          </div>
        </div>
      )}

      {/* Contributing factors */}
      {fatigue?.contributing_factors && Object.keys(fatigue.contributing_factors).length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-700/50">
          <span className="text-xs text-gray-500 block mb-1">Top Factors</span>
          <div className="flex flex-wrap gap-1">
            {Object.entries(fatigue.contributing_factors)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 3)
              .map(([name, value]) => (
                <span
                  key={name}
                  className="text-xs px-2 py-0.5 rounded-full bg-gray-800 text-gray-300"
                >
                  {name}
                </span>
              ))}
          </div>
        </div>
      )}
    </div>
  );
}
