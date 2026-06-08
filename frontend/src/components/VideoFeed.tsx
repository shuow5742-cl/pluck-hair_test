"use client";

import { useMemo, useState } from "react";
import { cn } from "@/lib/utils";
import { buildApiUrl } from "@/lib/api";
import type { RobotStats } from "@/store/robot-store";

interface VideoFeedProps {
  stats: RobotStats;
  maskActive?: boolean;
}

export function VideoFeed({ stats, maskActive = false }: VideoFeedProps) {
  const [errored, setErrored] = useState(false);
  const videoUrl = useMemo(() => buildApiUrl("/stream/video"), []);

  return (
    <div className="relative aspect-video overflow-hidden rounded-xl border border-border bg-panel shadow-panel">
      <div className="absolute inset-0">
        {!errored ? (
          <img
            src={videoUrl}
            alt="实时视频流"
            className="h-full w-full object-cover"
            onError={() => setErrored(true)}
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-[#0b1323] to-[#13213b] text-muted">
            视频流加载失败，使用占位图
          </div>
        )}
      </div>

      <div className="absolute inset-0 bg-gradient-to-t from-black/35 via-transparent to-black/10" />

      <div className="absolute left-4 top-3 flex items-center gap-2 rounded-full bg-black/40 px-3 py-1 text-xs text-white backdrop-blur">
        <span className="h-2 w-2 rounded-full bg-accent shadow-[0_0_0_4px_rgba(22,163,74,0.18)]" />
        高清默认视图
      </div>

      <div className="absolute bottom-3 left-4 rounded-md bg-black/50 px-3 py-1.5 text-sm text-white backdrop-blur">
        实时帧率 (FPS): {stats.fps.toFixed(0)}
      </div>

      <div className="absolute right-4 top-4 flex flex-col items-end gap-2">
        <div
          className={cn(
            "rounded-md px-3 py-1.5 text-xs font-medium",
            stats.confidence >= 0.95
              ? "bg-accent/15 text-accent"
              : "bg-yellow-500/20 text-yellow-200",
          )}
        >
          平均置信度 {(stats.confidence * 100).toFixed(1)}%
        </div>
        <div className="rounded-md bg-black/55 px-3 py-2 text-right text-xs font-medium text-white backdrop-blur">
          <div>CID {stats.clusterId ?? "-"}</div>
          <div>Track ID {stats.trackId ?? "-"}</div>
        </div>
      </div>

      <div className="absolute bottom-3 right-4 max-w-[260px] rounded-lg border border-white/10 bg-black/55 px-3 py-2 text-white backdrop-blur">
        <div className="text-[11px] uppercase tracking-[0.18em] text-white/60">
          Current Target
        </div>
        <div className="mt-1 text-sm font-semibold">
          {stats.targetObjectType ?? "未收到目标"}
        </div>
        <div className="mt-1 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-white/85">
          <span>
            X: {stats.targetXMm !== null ? `${stats.targetXMm.toFixed(3)} mm` : "-"}
          </span>
          <span>
            Y: {stats.targetYMm !== null ? `${stats.targetYMm.toFixed(3)} mm` : "-"}
          </span>
          <span>状态: {stats.targetState ?? "-"}</span>
          <span>目标数: {stats.currentTargets}</span>
        </div>
      </div>

      {maskActive ? (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-slate-950/55 text-white backdrop-blur-md">
          <div className="rounded-full border border-white/25 bg-white/10 px-6 py-2 text-lg font-semibold tracking-wide shadow-[0_12px_30px_rgba(0,0,0,0.35)]">
            机械臂移动中
          </div>
        </div>
      ) : null}
    </div>
  );
}
