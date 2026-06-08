"use client";

import { useState } from "react";
import useSWR from "swr";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { type IoDeviceState, setIo, swrFetcher } from "@/lib/controlApi";

interface Props {
  onLog?: (level: "info" | "error", message: string) => void;
}

function FeedbackDot({ device }: { device: IoDeviceState }) {
  const color = !device.in_position
    ? "bg-yellow-400"
    : device.is_open
      ? "bg-accent"
      : "bg-sky-400";
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className={cn("h-2.5 w-2.5 rounded-full", color)} />
      <span className="text-xs text-muted">{device.feedback}</span>
    </span>
  );
}

function DeviceRow({
  device,
  onAction,
}: {
  device: IoDeviceState;
  onAction: (id: string, action: string) => void;
}) {
  const isCylinder = device.kind === "cylinder";
  // action verbs depend on device kind
  const openAction = isCylinder ? "extend" : "start";
  const closeAction = isCylinder ? "retract" : "stop";
  const openActive = device.is_open;

  return (
    <div className="grid grid-cols-[1fr_auto_auto_auto] items-center gap-2 border-b border-border/60 px-2 py-2 last:border-0">
      <div className="min-w-0">
        <div className="truncate text-sm font-medium">{device.name}</div>
        <div className="truncate font-mono text-[11px] text-muted">{device.plc_tag}</div>
      </div>
      <Button
        size="sm"
        variant={openActive ? "default" : "secondary"}
        className="min-w-[64px]"
        onClick={() => onAction(device.id, openAction)}
      >
        {device.open_label}
      </Button>
      <Button
        size="sm"
        variant={!openActive && device.in_position ? "danger" : "secondary"}
        className="min-w-[64px]"
        onClick={() => onAction(device.id, closeAction)}
      >
        {device.close_label}
      </Button>
      <div className="min-w-[88px] text-right">
        <FeedbackDot device={device} />
      </div>
    </div>
  );
}

export function IoControlPanel({ onLog }: Props) {
  const { data, mutate } = useSWR<{ devices: IoDeviceState[] }>(
    "/io/states",
    swrFetcher,
    { refreshInterval: 600 },
  );
  const [tab, setTab] = useState<"cylinder" | "other">("cylinder");
  const devices = data?.devices ?? [];
  const shown = devices.filter((d) => d.kind === tab);

  const onAction = async (id: string, action: string) => {
    try {
      const r = await setIo(id, action);
      if (r.device) {
        mutate(
          (prev) =>
            prev
              ? {
                  devices: prev.devices.map((d) =>
                    d.id === id ? r.device! : d,
                  ),
                }
              : prev,
          { revalidate: false },
        );
        onLog?.("info", `${r.device.name} → ${r.device.feedback}`);
      } else if (!r.ok) {
        onLog?.("error", `IO 操作失败: ${r.error ?? "?"}`);
      }
    } catch (e) {
      onLog?.("error", "IO 请求失败");
    }
  };

  return (
    <div className="flex flex-col gap-2 rounded-xl border border-border bg-panel p-3 shadow-panel">
      <div className="flex items-center gap-2">
        <h3 className="text-sm font-semibold">设备 IO 控制</h3>
        <div className="ml-auto flex gap-1">
          <Button
            size="sm"
            variant={tab === "cylinder" ? "default" : "secondary"}
            onClick={() => setTab("cylinder")}
          >
            气缸控制
          </Button>
          <Button
            size="sm"
            variant={tab === "other" ? "default" : "secondary"}
            onClick={() => setTab("other")}
          >
            其他控制
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-[1fr_auto_auto_auto] gap-2 px-2 text-[11px] uppercase tracking-wide text-muted">
        <span>设备名称</span>
        <span className="text-center">{tab === "cylinder" ? "伸出/打开" : "启动/打开"}</span>
        <span className="text-center">{tab === "cylinder" ? "收缩/关闭" : "停止/关闭"}</span>
        <span className="text-right">反馈状态</span>
      </div>

      <div className="rounded-lg border border-border/60">
        {shown.length === 0 ? (
          <div className="px-3 py-6 text-center text-sm text-muted">无设备</div>
        ) : (
          shown.map((d) => <DeviceRow key={d.id} device={d} onAction={onAction} />)
        )}
      </div>
    </div>
  );
}
