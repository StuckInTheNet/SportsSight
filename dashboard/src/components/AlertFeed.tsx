import type { Alert } from "@/types";
import { AlertTriangle, AlertCircle, Info } from "lucide-react";

interface Props {
  alerts: Alert[];
}

function alertIcon(level: string) {
  switch (level) {
    case "critical": return <AlertTriangle className="w-4 h-4 text-red-400" />;
    case "high": return <AlertCircle className="w-4 h-4 text-orange-400" />;
    default: return <Info className="w-4 h-4 text-yellow-400" />;
  }
}

function alertBorder(level: string): string {
  switch (level) {
    case "critical": return "border-l-red-500";
    case "high": return "border-l-orange-500";
    default: return "border-l-yellow-500";
  }
}

export default function AlertFeed({ alerts }: Props) {
  if (alerts.length === 0) {
    return (
      <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
        <h3 className="text-lg font-semibold mb-2">Alerts</h3>
        <p className="text-gray-500 text-sm">No alerts yet</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-xl p-4 border border-gray-800">
      <h3 className="text-lg font-semibold mb-3">
        Alerts <span className="text-sm text-gray-500">({alerts.length})</span>
      </h3>
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {alerts.map((alert, i) => (
          <div
            key={`${alert.player_id}-${alert.timestamp}-${i}`}
            className={`flex items-start gap-2 p-3 bg-gray-800/50 rounded-lg border-l-2 ${alertBorder(alert.level)}`}
          >
            {alertIcon(alert.level)}
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium">{alert.message}</p>
              <p className="text-xs text-gray-500 mt-1">
                Score: {alert.score.toFixed(0)} | {new Date(alert.timestamp * 1000).toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
