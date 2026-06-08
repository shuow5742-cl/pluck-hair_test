"use client";

import useSWR from "swr";
import { Button } from "@/components/ui/button";
import { type EpsonDescribe, swrFetcher, systemCommand } from "@/lib/controlApi";

interface Props {
  onLog?: (level: "info" | "error", message: string) => void;
}

// Button colouring mirrors the SENKE HMI: start/auto green, stop red, others neutral.
const VARIANT: Record<string, "default" | "danger" | "secondary"> = {
  auto: "default",
  start: "default",
  stop: "danger",
  init: "secondary",
  pause: "secondary",
  reset: "secondary",
  manual_pick_ok: "secondary",
};

const FALLBACK = [
  { id: "auto", name: "自动" },
  { id: "init", name: "初始化" },
  { id: "start", name: "启动" },
  { id: "stop", name: "停止" },
  { id: "pause", name: "暂停" },
  { id: "reset", name: "复位" },
  { id: "manual_pick_ok", name: "人工挑毛OK" },
];

export function SystemControlBar({ onLog }: Props) {
  const { data: describe } = useSWR<EpsonDescribe>("/epson/describe", swrFetcher, {
    revalidateOnFocus: false,
  });
  const buttons = describe?.system_buttons?.length ? describe.system_buttons : FALLBACK;

  const onClick = async (id: string, name: string) => {
    try {
      const r = await systemCommand(id);
      onLog?.(r.ok ? "info" : "error", r.ok ? `系统指令: ${name}` : `指令失败: ${r.error ?? "?"}`);
    } catch (e) {
      onLog?.("error", `系统指令失败: ${name}`);
    }
  };

  return (
    <div className="flex flex-wrap gap-2 rounded-xl border border-border bg-panel p-3 shadow-panel">
      {buttons.map((b) => (
        <Button
          key={b.id}
          size="md"
          variant={VARIANT[b.id] ?? "secondary"}
          onClick={() => onClick(b.id, b.name)}
          className="min-w-[84px] flex-1"
        >
          {b.name}
        </Button>
      ))}
    </div>
  );
}
