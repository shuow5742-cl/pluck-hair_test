import { formatTimestamp } from "@/lib/utils";
import { RobotLog } from "@/store/robot-store";

interface LogPanelProps {
  logs: RobotLog[];
}

export function LogPanel({ logs }: LogPanelProps) {
  return (
    <div className="rounded-lg border border-border bg-panel p-4 shadow-panel">
      <div className="flex items-center justify-between">
        <p className="text-sm font-semibold text-text">日志</p>
        <p className="text-xs text-muted">最近 {logs.length} 条</p>
      </div>
      <div className="mt-3 h-48 overflow-y-auto rounded-md bg-muted/10 px-3 py-2">
        {logs.length === 0 ? (
          <p className="text-sm text-muted">暂无日志</p>
        ) : (
          <ul className="space-y-2 text-sm">
            {logs
              .slice()
              .reverse()
              .map((log) => (
                <li
                  key={`${log.timestamp}-${log.message}`}
                  className="border-b border-border/40 pb-2 last:border-b-0 last:pb-0"
                >
                  <div className="flex items-center gap-2 text-xs text-muted">
                    <span className="inline-block rounded-full bg-white/10 px-2 py-0.5 uppercase tracking-wide">
                      {log.level}
                    </span>
                    <span>{formatTimestamp(log.timestamp)}</span>
                  </div>
                  <p className="mt-1 text-text">{log.message}</p>
                </li>
              ))}
          </ul>
        )}
      </div>
    </div>
  );
}
