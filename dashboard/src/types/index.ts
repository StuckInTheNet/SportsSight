export interface FatigueScore {
  player_id: string | number;
  timestamp_ms: number;
  score: number;
  confidence: number;
  level: "low" | "moderate" | "high" | "critical";
  trend: "rising" | "stable" | "declining";
  baseline_deviation: number;
  contributing_factors: Record<string, number>;
  predicted_score_5min: number;
}

export interface Player {
  id: string;
  name: string;
  jersey_number: number | null;
  position: string | null;
}

export interface Game {
  id: string;
  opponent: string | null;
  date: string;
  status: "pending" | "live" | "completed";
  venue: string | null;
}

export interface Alert {
  player_id: number;
  level: "moderate" | "high" | "critical";
  score: number;
  message: string;
  timestamp: number;
}

export interface FatigueUpdate {
  type: "fatigue_update";
  data: {
    game_id: string;
    frame: number;
    timestamp_ms: number;
    scores: string; // JSON string of Record<string, FatigueScore>
  };
}

export interface AlertUpdate {
  type: "alert";
  data: Alert;
}

export type WSMessage = FatigueUpdate | AlertUpdate;
