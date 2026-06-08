import { useCallback } from "react";
import { LiveViewPanel } from "@/components/LiveViewPanel";
import { EpsonManualControl } from "@/components/EpsonManualControl";
import { IoControlPanel } from "@/components/IoControlPanel";
import { SystemControlBar } from "@/components/SystemControlBar";
import { LogPanel } from "@/components/LogPanel";
import { useRobotStore } from "@/store/robot-store";

// pluck-hair_test console: left half = live camera + 异物 detection + tweezer
// tip overlay + tip↔pick distance; right half = Epson manual control (top) and
// system + device-IO control (bottom).
export default function TestConsolePage() {
  const logs = useRobotStore((s) => s.logs);
  const pushLog = useRobotStore((s) => s.pushLog);

  const onLog = useCallback(
    (level: "info" | "error", message: string) => pushLog({ level, message }),
    [pushLog],
  );

  return (
    <div className="min-h-screen bg-surface text-text">
      <div className="mx-auto flex max-w-screen-2xl flex-col gap-4 p-4 lg:p-6">
        <header className="flex items-center justify-between rounded-xl border border-border bg-panel px-5 py-3 shadow-panel">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent/10 text-lg font-semibold text-accent">
              燕
            </div>
            <div>
              <p className="text-base font-semibold">
                燕窝挑毛 · 测试调试台{" "}
                <span className="text-xs font-normal text-muted">pluck-hair_test</span>
              </p>
              <p className="text-xs text-muted">
                左：实时识别 + 镊子尖定位 ｜ 右：Epson 手动控制 + 设备 IO
              </p>
            </div>
          </div>
        </header>

        <div className="grid gap-4 xl:grid-cols-[1.35fr_1fr]">
          {/* LEFT: live view + logs */}
          <div className="flex flex-col gap-4">
            <LiveViewPanel />
            <LogPanel logs={logs} />
          </div>

          {/* RIGHT: top = Epson manual control, bottom = system + IO control */}
          <div className="flex flex-col gap-4">
            <EpsonManualControl onLog={onLog} />
            <SystemControlBar onLog={onLog} />
            <IoControlPanel onLog={onLog} />
          </div>
        </div>
      </div>
    </div>
  );
}
