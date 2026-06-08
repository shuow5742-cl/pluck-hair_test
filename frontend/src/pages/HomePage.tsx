import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { Settings } from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { StatusIndicator } from "@/components/StatusIndicator";
import { StatCard } from "@/components/StatCard";
import { ControlButtons } from "@/components/ControlButtons";
import { LogPanel } from "@/components/LogPanel";
import { VideoFeed } from "@/components/VideoFeed";
import { cn } from "@/lib/utils";
import { buildWsUrl } from "@/lib/api";
import type { RobotStats, RobotTarget } from "@/store/robot-store";
import { useRobotStore } from "@/store/robot-store";

function readNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readString(value: unknown) {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function normalizeTarget(raw: Record<string, unknown>): RobotTarget {
  return {
    trackId: readNumber(raw.track_id),
    clusterId: readString(raw.cluster_id),
    objectType: readString(raw.object_type),
    confidence: readNumber(raw.confidence) ?? 0,
    state: readString(raw.state),
    xPx: readNumber(raw.pixel_x) ?? readNumber(raw.x_px),
    yPx: readNumber(raw.pixel_y) ?? readNumber(raw.y_px),
    worldXMm:
      readNumber(raw.world_x_mm) ??
      (readNumber(raw.x) !== null && readNumber(raw.pixel_x) === null
        ? readNumber(raw.x)
        : null),
    worldYMm:
      readNumber(raw.world_y_mm) ??
      (readNumber(raw.y) !== null && readNumber(raw.pixel_y) === null
        ? readNumber(raw.y)
        : null),
  };
}

function buildTargetPatch(target: RobotTarget | null): Partial<RobotStats> {
  if (!target) {
    return {
      trackId: null,
      clusterId: null,
      targetXMm: null,
      targetYMm: null,
      targetObjectType: null,
      targetState: null,
    };
  }

  return {
    trackId: target.trackId,
    clusterId: target.clusterId,
    targetXMm: target.worldXMm,
    targetYMm: target.worldYMm,
    targetObjectType: target.objectType,
    targetState: target.state,
    confidence: target.confidence,
  };
}

function pickDisplayedTarget(targets: RobotTarget[]) {
  if (targets.length === 0) {
    return null;
  }

  const currentTrackId = useRobotStore.getState().stats.trackId;
  if (currentTrackId !== null) {
    const matched = targets.find((target) => target.trackId === currentTrackId);
    if (matched) {
      return matched;
    }
  }

  return targets.find((target) => target.state === "pending") ?? targets[0];
}

export default function HomePage() {
  const status = useRobotStore((state) => state.status);
  const stats = useRobotStore((state) => state.stats);
  const logs = useRobotStore((state) => state.logs);
  const setStatus = useRobotStore((state) => state.setStatus);
  const setStats = useRobotStore((state) => state.setStats);
  const setTargets = useRobotStore((state) => state.setTargets);
  const pushLog = useRobotStore((state) => state.pushLog);
  const mounted = useRef(false);
  const [movementMaskActive, setMovementMaskActive] = useState(false);
  const [pickDoneTargetId, setPickDoneTargetId] = useState("");
  const wsRef = useRef<WebSocket | null>(null);
  const wsReconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wsReconnectAttemptRef = useRef(0);

  useEffect(() => {
    if (mounted.current) return;
    pushLog({ level: "info", message: "前端面板已加载" });
    mounted.current = true;
  }, [pushLog]);

  useEffect(() => {
    let cancelled = false;

    const scheduleReconnect = (reason: string) => {
      if (cancelled) return;
      if (wsReconnectTimerRef.current) return;

      wsReconnectAttemptRef.current += 1;
      const attempt = wsReconnectAttemptRef.current;
      const delayMs = Math.min(10_000, 500 * 2 ** Math.min(attempt, 4));
      pushLog({
        level: "error",
        message: `${reason}，${Math.round(delayMs / 1000)}s 后重试（第 ${attempt} 次）`,
      });

      wsReconnectTimerRef.current = setTimeout(() => {
        wsReconnectTimerRef.current = null;
        connect();
      }, delayMs);
    };

    const connect = () => {
      if (cancelled) return;
      if (
        wsRef.current &&
        (wsRef.current.readyState === WebSocket.OPEN ||
          wsRef.current.readyState === WebSocket.CONNECTING)
      ) {
        return;
      }

      const wsUrl = buildWsUrl();
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.addEventListener("open", () => {
        wsReconnectAttemptRef.current = 0;
        setStatus("running");
        pushLog({ level: "info", message: "WebSocket 已连接" });
      });

      ws.addEventListener("message", (event) => {
        try {
          const message = JSON.parse(event.data as string) as Record<string, unknown>;

          if (message.type === "detection") {
            const patch: Partial<RobotStats> = {};
            setStatus("running");
            if (typeof message.region_picked === "number") {
              patch.totalImpurities = message.region_picked;
            }
            if (typeof message.current_pending === "number") {
              patch.currentTargets = message.current_pending;
            }
            if (typeof message.fps === "number") {
              patch.fps = message.fps;
            }
            if (
              typeof message.phase === "string" &&
              !useRobotStore.getState().stats.targetState
            ) {
              patch.targetState = message.phase;
            }
            if (Array.isArray(message.targets)) {
              const targets = message.targets
                .filter(
                  (item): item is Record<string, unknown> =>
                    typeof item === "object" && item !== null,
                )
                .map(normalizeTarget);
              setTargets(targets);
              const displayedTarget = pickDisplayedTarget(targets);
              Object.assign(patch, buildTargetPatch(displayedTarget));
              if (targets.length > 0) {
                const avgConfidence =
                  targets.reduce((sum, target) => sum + target.confidence, 0) /
                  targets.length;
                patch.confidence = avgConfidence;
              }
            }
            if (Object.keys(patch).length) {
              setStats(patch);
            }
          } else if (message.type === "target") {
            const target = normalizeTarget(message);
            setStats(buildTargetPatch(target));
            pushLog({
              level: "info",
              message: `已收到目标 ${target.clusterId ?? "-"} / ${target.trackId ?? "-"} 坐标`,
            });
          } else if (message.type === "error") {
            const errorMessage =
              typeof message.message === "string" ? message.message : "后端返回错误";
            pushLog({ level: "error", message: errorMessage });
          } else if (message.type === "ack") {
            const statusText =
              typeof message.status === "string" ? message.status : "ok";
            pushLog({ level: "info", message: `ACK: ${statusText}` });
          }
        } catch (error) {
          console.warn("Failed to parse websocket message", error);
        }
      });

      ws.addEventListener("error", () => {
        scheduleReconnect(`WebSocket 连接失败（${wsUrl}）`);
      });

      ws.addEventListener("close", () => {
        if (wsRef.current === ws) {
          wsRef.current = null;
        }
        setStatus("stopped");
        setTargets([]);
        setStats({
          currentTargets: 0,
          ...buildTargetPatch(null),
        });
        scheduleReconnect("WebSocket 已断开");
      });
    };

    connect();

    return () => {
      cancelled = true;
      if (wsReconnectTimerRef.current) {
        clearTimeout(wsReconnectTimerRef.current);
        wsReconnectTimerRef.current = null;
      }
      wsReconnectAttemptRef.current = 0;
      setStatus("stopped");
      setTargets([]);
      setStats({
        currentTargets: 0,
        ...buildTargetPatch(null),
      });
      const ws = wsRef.current;
      wsRef.current = null;
      ws?.close();
    };
  }, [pushLog, setStats, setStatus, setTargets]);

  const sendWsAction = (payload: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify(payload));
  };

  return (
    <div className="min-h-screen bg-surface text-text">
      <div className="mx-auto flex max-w-screen-2xl flex-col gap-4 p-4 lg:p-6">
        <header className="flex flex-col gap-3 rounded-xl border border-border bg-panel px-5 py-4 shadow-panel md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-accent/10 text-lg font-semibold text-accent shadow-panel">
              燕
            </div>
            <div>
              <p className="text-lg font-semibold">
                燕窝智能挑毛监控系统{" "}
                <span className="text-sm font-normal text-muted">v1.0</span>
              </p>
              <p className="text-sm text-muted">
                Bird Nest Intelligent Plucking Monitoring
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="rounded-lg border border-border bg-muted/10 px-3 py-2 text-sm text-muted">
              实时监控 · 工业模式
            </div>
            <Link
              to="/settings"
              className={cn(
                buttonVariants({ variant: "secondary", size: "sm" }),
                "gap-2",
              )}
            >
              <Settings className="h-4 w-4" />
              系统设置
            </Link>
          </div>
        </header>

        <div className="grid gap-4 xl:grid-cols-[1.65fr_1fr]">
          <div className="rounded-xl border border-border bg-panel p-3 shadow-panel">
            <div className="flex flex-col gap-3">
              <VideoFeed stats={stats} maskActive={movementMaskActive} />
              <div className="flex flex-wrap items-center gap-3">
                <Button
                  size="sm"
                  variant={movementMaskActive ? "danger" : "secondary"}
                  aria-pressed={movementMaskActive}
                  onClick={() =>
                    setMovementMaskActive((prevActive) => !prevActive)
                  }
                  className={cn(
                    "min-w-[140px] text-sm",
                    movementMaskActive
                      ? "shadow-sm"
                      : "border border-accent/30 bg-accent/10 text-accent hover:bg-accent/20",
                  )}
                >
                  {movementMaskActive ? "停止移动" : "开始移动"}
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={() => sendWsAction({ action: "request_target" })}
                  className="min-w-[140px] border border-border bg-muted/10 text-sm hover:bg-muted/20"
                >
                  请求目标
                </Button>
                <div className="flex items-center gap-2">
                  <Input
                    value={pickDoneTargetId}
                    onChange={(event) => setPickDoneTargetId(event.target.value)}
                    placeholder="target_id"
                    inputMode="numeric"
                    className="h-9 w-[140px] bg-muted/10"
                  />
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => {
                      const payload: Record<string, unknown> = {
                        action: "pick_done",
                      };
                      if (pickDoneTargetId.trim()) {
                        payload.track_id = Number.parseInt(pickDoneTargetId, 10);
                      }
                      sendWsAction(payload);
                    }}
                    className="min-w-[140px] border border-border bg-muted/10 text-sm hover:bg-muted/20"
                  >
                    PICK_DONE
                  </Button>
                </div>
              </div>
            </div>
          </div>

          <div className="grid gap-3">
            <StatusIndicator running={status === "running"} />

            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
              <StatCard
                title="累计挑出杂质"
                value={stats.totalImpurities}
                unit="个"
              />
              <StatCard
                title="当前视野目标"
                value={stats.currentTargets}
                unit="个"
              />
            </div>

            <ControlButtons />

            <LogPanel logs={logs} />
          </div>
        </div>
      </div>
    </div>
  );
}
