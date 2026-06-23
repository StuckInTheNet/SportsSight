import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";

interface TimelineEntry {
  time: string;
  timestamp_ms: number;
  [playerId: string]: number | string;
}

interface Props {
  data: TimelineEntry[];
  playerIds: string[];
  playerNames?: Record<string, string>;
  thresholds?: { moderate: number; high: number; critical: number };
}

const PLAYER_COLORS = [
  "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
  "#ec4899", "#06b6d4", "#f97316", "#14b8a6", "#6366f1",
];

export default function FatigueTimeline({ data, playerIds, playerNames, thresholds }: Props) {
  const t = thresholds ?? { moderate: 55, high: 75, critical: 90 };

  return (
    <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
      <h3 className="text-lg font-semibold mb-4">Fatigue Timeline</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="time" stroke="#9ca3af" fontSize={12} />
          <YAxis domain={[0, 100]} stroke="#9ca3af" fontSize={12} />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1f2937",
              border: "1px solid #374151",
              borderRadius: "8px",
            }}
          />
          <Legend />

          {/* Threshold zones */}
          <ReferenceLine y={t.moderate} stroke="#f59e0b" strokeDasharray="5 5" label="Moderate" />
          <ReferenceLine y={t.high} stroke="#ef4444" strokeDasharray="5 5" label="High" />
          <ReferenceLine y={t.critical} stroke="#dc2626" strokeDasharray="3 3" label="Critical" />

          {/* Player lines */}
          {playerIds.map((pid, i) => (
            <Line
              key={pid}
              type="monotone"
              dataKey={pid}
              name={playerNames?.[pid] ?? `Player ${pid}`}
              stroke={PLAYER_COLORS[i % PLAYER_COLORS.length]}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
