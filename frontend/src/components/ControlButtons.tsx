"use client";

import { useState } from "react";
import { Play, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { buildApiUrl } from "@/lib/api";
import { useRobotStore } from "@/store/robot-store";

export function ControlButtons() {
  const status = useRobotStore((state) => state.status);
  const setStatus = useRobotStore((state) => state.setStatus);
  const pushLog = useRobotStore((state) => state.pushLog);
  const [loading, setLoading] = useState<"start" | "stop" | null>(null);

  const toggle = async (running: boolean) => {
    setLoading(running ? "start" : "stop");
    try {
      const res = await fetch(buildApiUrl("/control/status"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ running }),
      });
      if (!res.ok) {
        throw new Error("请求失败");
      }
      const data = (await res.json()) as { running: boolean };
      setStatus(data.running ? "running" : "stopped");
      pushLog({
        level: "info",
        message: data.running ? "算法已启动" : "算法已停止",
      });
    } catch (error) {
      console.error(error);
      pushLog({
        level: "error",
        message: "切换算法状态失败，请稍后重试",
      });
    } finally {
      setLoading(null);
    }
  };

  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      <Button
        variant="secondary"
        size="lg"
        disabled={status === "running" || loading === "start"}
        onClick={() => toggle(true)}
        className="border border-accent/30 bg-accent/10 text-accent hover:bg-accent/20"
      >
        <Play className="h-4 w-4" />
        开始算法运行
      </Button>
      <Button
        variant="danger"
        size="lg"
        disabled={status === "stopped" || loading === "stop"}
        onClick={() => toggle(false)}
      >
        <Square className="h-4 w-4" />
        停止算法运行
      </Button>
    </div>
  );
}
