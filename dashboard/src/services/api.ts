const BASE = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const apiKey = localStorage.getItem("sportssight_api_key") || "";
  const res = await fetch(`${BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-API-Key": apiKey,
      ...options?.headers,
    },
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export const api = {
  // Teams
  getTeam: () => request<any>("/teams/me"),

  // Players
  listPlayers: () => request<any[]>("/players"),
  createPlayer: (data: any) =>
    request<any>("/players", { method: "POST", body: JSON.stringify(data) }),

  // Games
  listGames: (status?: string) =>
    request<any[]>(`/games${status ? `?status=${status}` : ""}`),
  getGame: (id: string) => request<any>(`/games/${id}`),
  createGame: (data: any) =>
    request<any>("/games", { method: "POST", body: JSON.stringify(data) }),

  // Fatigue data
  getGameFatigue: (gameId: string, params?: Record<string, string>) => {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    return request<any[]>(`/games/${gameId}/fatigue${qs}`);
  },
  getPlayerHistory: (playerId: string, lastN = 10) =>
    request<any[]>(`/players/${playerId}/fatigue-history?last_n_games=${lastN}`),

  // Alerts
  getGameAlerts: (gameId: string) => request<any[]>(`/games/${gameId}/alerts`),
  configureAlerts: (config: any) =>
    request<any>("/alerts/configure", {
      method: "POST",
      body: JSON.stringify(config),
    }),
};
