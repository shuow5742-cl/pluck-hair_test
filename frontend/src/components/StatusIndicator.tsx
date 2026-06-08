import { cn } from "@/lib/utils";

interface StatusIndicatorProps {
  running: boolean;
}

export function StatusIndicator({ running }: StatusIndicatorProps) {
  return (
    <div className="rounded-lg border border-border bg-panel px-4 py-3 shadow-panel">
      <div className="flex items-center gap-3">
        <span
          className={cn(
            "h-3 w-3 rounded-full shadow-[0_0_0_6px_rgba(22,163,74,0.18)]",
            running ? "bg-accent" : "bg-danger shadow-[0_0_0_6px_rgba(220,38,38,0.18)]",
          )}
          aria-hidden
        />
        <div className="flex flex-col gap-1">
          <p className="text-xs uppercase tracking-wide text-muted">AI 算法</p>
          <p className="text-base font-semibold leading-tight">
            {running ? "AI算法正在运行中..." : "AI算法已停止"}
          </p>
        </div>
      </div>
    </div>
  );
}
