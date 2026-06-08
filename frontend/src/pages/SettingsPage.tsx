"use client";

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, Save, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card } from "@/components/ui/card";
import {
  RobotSettings,
  defaultSettings,
  persistSettings,
  useRobotStore,
} from "@/store/robot-store";

type FormErrors = Record<string, string>;

export default function SettingsPage() {
  const settings = useRobotStore((state) => state.settings);
  const setSettings = useRobotStore((state) => state.setSettings);
  const pushLog = useRobotStore((state) => state.pushLog);
  const [formState, setFormState] = useState<RobotSettings>(settings);
  const [errors, setErrors] = useState<FormErrors>({});

  useEffect(() => {
    setFormState(settings);
  }, [settings]);

  const handleDetectionChange = (
    key: keyof RobotSettings["detection"],
    value: number,
  ) => {
    setFormState((prev) => ({
      ...prev,
      detection: {
        ...prev.detection,
        [key]: value,
      },
    }));
  };

  const validate = () => {
    const nextErrors: FormErrors = {};
    if (
      formState.detection.confidenceThreshold < 0 ||
      formState.detection.confidenceThreshold > 1
    ) {
      nextErrors.confidenceThreshold = "置信度阈值需在 0 - 1 之间";
    }
    if (formState.detection.minSizePx <= 0) {
      nextErrors.minSizePx = "最小目标尺寸需大于 0";
    }
    if (formState.detection.maxSizePx <= formState.detection.minSizePx) {
      nextErrors.maxSizePx = "最大尺寸需大于最小尺寸";
    }
    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!validate()) return;
    setSettings(formState);
    persistSettings(formState);
    pushLog({ level: "info", message: "设置已保存" });
  };

  const handleReset = () => {
    setFormState(defaultSettings);
    setSettings(defaultSettings);
    persistSettings(defaultSettings);
    pushLog({ level: "info", message: "设置已重置为默认" });
  };

  return (
    <div className="min-h-screen bg-surface text-text">
      <div className="mx-auto flex max-w-screen-lg flex-col gap-5 p-4 lg:p-6">
        <header className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <Link to="/">
              <Button variant="ghost" size="sm" className="gap-2">
                <ArrowLeft className="h-4 w-4" />
                返回面板
              </Button>
            </Link>
            <div>
              <p className="text-lg font-semibold">系统设置</p>
              <p className="text-sm text-muted">调整算法参数（仅本地保存）</p>
            </div>
          </div>
        </header>

        <form
          onSubmit={handleSubmit}
          className="flex flex-col gap-4 rounded-xl border border-border bg-panel p-5 shadow-panel"
        >
          <Card className="p-4">
            <p className="text-base font-semibold">检测算法</p>
            <p className="text-sm text-muted">阈值、目标尺寸</p>
            <div className="mt-4 space-y-4">
              <div className="space-y-2">
                <Label htmlFor="threshold">置信度阈值 (0-1)</Label>
                <Input
                  id="threshold"
                  type="number"
                  step="0.01"
                  min={0}
                  max={1}
                  value={formState.detection.confidenceThreshold}
                  onChange={(e) =>
                    handleDetectionChange(
                      "confidenceThreshold",
                      Number(e.target.value),
                    )
                  }
                />
                {errors.confidenceThreshold ? (
                  <p className="text-xs text-danger">
                    {errors.confidenceThreshold}
                  </p>
                ) : null}
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="minSize">最小目标尺寸 (px)</Label>
                  <Input
                    id="minSize"
                    type="number"
                    value={formState.detection.minSizePx}
                    onChange={(e) =>
                      handleDetectionChange("minSizePx", Number(e.target.value))
                    }
                  />
                  {errors.minSizePx ? (
                    <p className="text-xs text-danger">{errors.minSizePx}</p>
                  ) : null}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="maxSize">最大目标尺寸 (px)</Label>
                  <Input
                    id="maxSize"
                    type="number"
                    value={formState.detection.maxSizePx}
                    onChange={(e) =>
                      handleDetectionChange("maxSizePx", Number(e.target.value))
                    }
                  />
                  {errors.maxSizePx ? (
                    <p className="text-xs text-danger">{errors.maxSizePx}</p>
                  ) : null}
                </div>
              </div>
            </div>
          </Card>

          <div className="flex flex-wrap items-center justify-end gap-3">
            <Button
              type="button"
              variant="secondary"
              size="md"
              className="gap-2"
              onClick={handleReset}
            >
              <RotateCcw className="h-4 w-4" />
              重置默认
            </Button>
            <Button type="submit" size="md" className="gap-2">
              <Save className="h-4 w-4" />
              保存设置
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
