const BASE = "/api";

let _apiKey = localStorage.getItem("sportssight_api_key") || "";

export function setApiKey(key: string) {
  _apiKey = key;
  localStorage.setItem("sportssight_api_key", key);
}

export function getApiKey(): string {
  return _apiKey;
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": _apiKey,
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export interface TeamInfo {
  id: string;
  name: string;
  sport: string;
}

export interface PlayerInfo {
  id: string;
  name: string;
  jersey_number: number | null;
  position: string | null;
}

export interface GameInfo {
  id: string;
  opponent: string | null;
  date: string;
  status: string;
  venue: string | null;
}

export interface FatigueRecord {
  player_id: string;
  timestamp_ms: number;
  fatigue_score: number;
  confidence: number;
  trend: string;
  speed: number | null;
  contributing_factors: Record<string, number>;
}

export const api = {
  getTeam: () => request<TeamInfo>("/teams/me"),
  listPlayers: () => request<PlayerInfo[]>("/players"),
  listGames: (status?: string) =>
    request<GameInfo[]>(`/games${status ? `?status=${status}` : ""}`),
  getGame: (id: string) => request<GameInfo>(`/games/${id}`),

  getGameFatigue: (gameId: string, params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<FatigueRecord[]>(`/games/${gameId}/fatigue${qs}`);
  },

  getPlayerHistory: (playerId: string, lastN = 10) =>
    request<any[]>(`/players/${playerId}/fatigue-history?last_n_games=${lastN}`),

  getGameAlerts: (gameId: string) => request<any[]>(`/games/${gameId}/alerts`),
};
