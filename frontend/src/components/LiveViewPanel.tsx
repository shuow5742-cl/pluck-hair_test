"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import { cn } from "@/lib/utils";
import { type LiveState, swrFetcher, testStreamUrl } from "@/lib/controlApi";

// Left half: live camera (MJPEG with 异物 detection + tweezer cross baked in by
// the backend) plus a telemetry strip: tweezer open/closed, tip pixel coords,
// and the real-mm distance from the tweezer tip to the nearest predicted pick.
export function LiveViewPanel() {
  const [errored, setErrored] = useState(false);
  const videoUrl = useMemo(() => testStreamUrl(), []);
  const { data } = useSWR<LiveState>("/test/state", swrFetcher, {
    refreshInterval: 200,
  });

  const tw = data?.tweezer;
  const tipMm = data?.tip_to_pick_mm ?? null;
  const state = tw?.found ? (tw.is_open ? "张开" : "闭合") : "未识别";
  const stateColor = !tw?.found
    ? "bg-yellow-500/20 text-yellow-700"
    : tw?.is_open
      ? "bg-sky-500/15 text-sky-700"
      : "bg-accent/15 text-accent";

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-border bg-panel p-3 shadow-panel">
      <div className="relative aspect-[4/3] overflow-hidden rounded-lg border border-border bg-black">
        {!errored ? (
          <img
            src={videoUrl}
            alt="实时视频流"
            className="h-full w-full object-contain"
            onError={() => setErrored(true)}
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center bg-gradient-to-br from-slate-200 to-slate-100 text-muted">
            视频流未连接（确认后端 --mode test 已启动）
          </div>
        )}
        <div className="absolute left-3 top-2 flex items-center gap-2 rounded-full bg-black/45 px-3 py-1 text-xs text-white backdrop-blur">
          <span className="h-2 w-2 rounded-full bg-accent" />
          实时画面 · {data?.ar_mode ?? "—"}
        </div>
        <div className="absolute bottom-2 left-3 rounded bg-black/50 px-2 py-1 text-xs text-white backdrop-blur">
          FPS {data ? data.fps.toFixed(0) : "—"} · 目标 {data?.detection_count ?? 0}
        </div>
      </div>

      {/* Tweezer telemetry strip */}
      <div className="grid grid-cols-3 gap-2">
        <div className="rounded-md border border-border bg-muted/5 px-3 py-2">
          <div className="text-[11px] text-muted">镊子状态</div>
          <div className={cn("mt-1 inline-block rounded px-2 py-0.5 text-sm font-semibold", stateColor)}>
            {state}
          </div>
        </div>
        <div className="rounded-md border border-border bg-muted/5 px-3 py-2">
          <div className="text-[11px] text-muted">镊子尖像素</div>
          <div className="mt-1 font-mono text-sm tabular-nums">
            {tw?.tip_xy ? `${Math.round(tw.tip_xy[0])}, ${Math.round(tw.tip_xy[1])}` : "—"}
          </div>
        </div>
        <div className="rounded-md border border-border bg-muted/5 px-3 py-2">
          <div className="text-[11px] text-muted">尖↔预测点距离</div>
          <div
            className={cn(
              "mt-1 font-mono text-lg font-bold tabular-nums",
              tipMm === null
                ? "text-muted"
                : tipMm <= 0.5
                  ? "text-accent"
                  : "text-danger",
            )}
          >
            {tipMm === null ? "—" : `${tipMm.toFixed(2)} mm`}
          </div>
        </div>
      </div>
    </div>
  );
}
