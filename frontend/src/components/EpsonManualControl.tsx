"use client";

import { useEffect, useMemo, useState } from "react";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import {
  type EpsonDescribe,
  epsonJog,
  epsonMove,
  swrFetcher,
} from "@/lib/controlApi";

interface Props {
  onLog?: (level: "info" | "error", message: string) => void;
}

const STEP_LABELS: Record<string, string> = {
  short: "短",
  medium: "中",
  long: "长",
  continuous: "连续",
};

export function EpsonManualControl({ onLog }: Props) {
  const { data: describe } = useSWR<EpsonDescribe>("/epson/describe", swrFetcher, {
    revalidateOnFocus: false,
  });
  const { data: poseData, mutate: mutatePose } = useSWR<{
    pose: Record<string, number>;
  }>("/epson/pose", swrFetcher, { refreshInterval: 400 });

  const axes = describe?.axes ?? ["X", "Y", "Z", "U"];
  const units = describe?.units ?? {};
  const pose = poseData?.pose ?? {};
  const points = describe?.points ?? [];
  const stepModes = describe?.step_modes ?? ["short", "medium", "long", "continuous"];
  const commands = describe?.commands ?? ["Move", "Go"];

  const [stepMode, setStepMode] = useState("medium");
  const [command, setCommand] = useState("Move");
  const [selectedPoint, setSelectedPoint] = useState("");
  const [target, setTarget] = useState<Record<string, string>>({});

  useEffect(() => {
    if (stepModes.length && !stepModes.includes(stepMode)) setStepMode(stepModes[0]);
  }, [stepModes, stepMode]);

  // When a taught point is chosen, prefill the target boxes from it.
  const pointMap = useMemo(
    () => Object.fromEntries(points.map((p) => [p.name, p])),
    [points],
  );
  useEffect(() => {
    if (!selectedPoint) return;
    const p = pointMap[selectedPoint];
    if (!p) return;
    const next: Record<string, string> = {};
    for (const a of axes) next[a] = String(p[a] ?? "");
    setTarget(next);
  }, [selectedPoint, pointMap, axes]);

  const doJog = async (axis: string, direction: "+" | "-") => {
    try {
      const r = await epsonJog(axis, direction, stepMode);
      if (r.pose) mutatePose({ pose: r.pose }, { revalidate: false });
      if (!r.ok) onLog?.("error", `点动失败: ${r.error ?? "?"}`);
      else onLog?.("info", `点动 ${axis}${direction} (${STEP_LABELS[stepMode] ?? stepMode})`);
    } catch (e) {
      onLog?.("error", `点动请求失败 ${axis}${direction}`);
    }
  };

  const doExecute = async () => {
    try {
      let r;
      const hasTarget = axes.some((a) => target[a]?.trim());
      if (hasTarget) {
        const t: Record<string, number> = {};
        for (const a of axes) {
          const v = target[a];
          if (v?.trim()) t[a] = Number(v);
        }
        r = await epsonMove({ target: t, command });
      } else if (selectedPoint) {
        r = await epsonMove({ point: selectedPoint, command });
      } else {
        onLog?.("error", "请先选择示教点或输入目标坐标");
        return;
      }
      if (r.pose) mutatePose({ pose: r.pose }, { revalidate: false });
      onLog?.(
        r.ok ? "info" : "error",
        r.ok ? `执行 ${command} → 目标位置` : `执行失败: ${r.error ?? "?"}`,
      );
    } catch (e) {
      onLog?.("error", "执行运动请求失败");
    }
  };

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-border bg-panel p-3 shadow-panel">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Epson LS6 手动控制</h3>
        <span className="rounded bg-muted/10 px-2 py-0.5 text-[11px] text-muted">
          backend: {describe?.backend ?? "…"}
        </span>
      </div>

      {/* Current pose readout */}
      <div className="grid grid-cols-4 gap-2">
        {axes.map((a) => (
          <div key={a} className="rounded-md border border-border bg-muted/5 px-2 py-1.5">
            <div className="text-[11px] text-muted">
              {a} ({units[a] ?? "mm"})
            </div>
            <div className="font-mono text-sm font-semibold tabular-nums">
              {pose[a] !== undefined ? pose[a].toFixed(3) : "—"}
            </div>
          </div>
        ))}
      </div>

      {/* Jog buttons + step mode */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-muted">步进点动</span>
        <div className="flex items-center gap-1">
          <span className="text-[11px] text-muted">步距</span>
          <Select
            value={stepMode}
            onChange={(e) => setStepMode(e.target.value)}
            className="h-8 w-24 text-xs"
          >
            {stepModes.map((m) => (
              <option key={m} value={m}>
                {STEP_LABELS[m] ?? m}
              </option>
            ))}
          </Select>
        </div>
      </div>
      <div className="grid grid-cols-4 gap-2">
        {axes.map((a) => (
          <div key={a} className="flex flex-col gap-1">
            <Button size="sm" variant="secondary" onClick={() => doJog(a, "+")}>
              +{a}
            </Button>
            <Button size="sm" variant="secondary" onClick={() => doJog(a, "-")}>
              −{a}
            </Button>
          </div>
        ))}
      </div>

      {/* Target input boxes */}
      <div className="grid grid-cols-4 gap-2">
        {axes.map((a) => (
          <Input
            key={a}
            value={target[a] ?? ""}
            onChange={(e) => setTarget((t) => ({ ...t, [a]: e.target.value }))}
            placeholder={`${a} 目标`}
            inputMode="decimal"
            className="h-9 text-sm"
          />
        ))}
      </div>

      {/* Point select + command + execute */}
      <div className="flex flex-wrap items-center gap-2">
        <Select
          value={selectedPoint}
          onChange={(e) => setSelectedPoint(e.target.value)}
          className="h-9 min-w-[150px] flex-1 text-sm"
        >
          <option value="">— 选择示教点 —</option>
          {points.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name}
            </option>
          ))}
        </Select>
        <Select
          value={command}
          onChange={(e) => setCommand(e.target.value)}
          className="h-9 w-28 text-sm"
        >
          {commands.map((c) => (
            <option key={c} value={c}>
              {c === "Move" ? "Move(直线)" : c === "Go" ? "Go(点到点)" : c}
            </option>
          ))}
        </Select>
        <Button size="sm" onClick={doExecute} className="min-w-[88px]">
          执行
        </Button>
      </div>
    </div>
  );
}
