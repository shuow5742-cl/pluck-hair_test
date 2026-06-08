import { useEffect, useRef } from "react";
import { create } from "zustand";

export type AlgorithmStatus = "running" | "stopped";

export interface RobotTarget {
  trackId: number | null;
  clusterId: string | null;
  objectType: string | null;
  confidence: number;
  state: string | null;
  xPx: number | null;
  yPx: number | null;
  worldXMm: number | null;
  worldYMm: number | null;
}

export interface RobotStats {
  fps: number;
  totalImpurities: number;
  currentTargets: number;
  durationSec: number;
  confidence: number;
  trackId: number | null;
  clusterId: string | null;
  targetXMm: number | null;
  targetYMm: number | null;
  targetObjectType: string | null;
  targetState: string | null;
}

export interface RobotSettings {
  detection: {
    confidenceThreshold: number;
    minSizePx: number;
    maxSizePx: number;
  };
}

export interface RobotLog {
  timestamp: string;
  level: "info" | "error" | "debug";
  message: string;
}

interface RobotState {
  status: AlgorithmStatus;
  stats: RobotStats;
  targets: RobotTarget[];
  settings: RobotSettings;
  logs: RobotLog[];
  setStatus: (status: AlgorithmStatus) => void;
  setStats: (stats: Partial<RobotStats>) => void;
  setTargets: (targets: RobotTarget[]) => void;
  setSettings: (settings: RobotSettings) => void;
  pushLog: (log: Omit<RobotLog, "timestamp"> & { timestamp?: string }) => void;
  startAlgo: () => void;
  stopAlgo: () => void;
}

export const defaultStats: RobotStats = {
  fps: 30,
  totalImpurities: 0,
  currentTargets: 0,
  durationSec: 0,
  confidence: 0.98,
  trackId: null,
  clusterId: null,
  targetXMm: null,
  targetYMm: null,
  targetObjectType: null,
  targetState: null,
};

export const defaultSettings: RobotSettings = {
  detection: {
    confidenceThreshold: 0.5,
    minSizePx: 24,
    maxSizePx: 240,
  },
};

const MAX_LOGS = 100;
const SETTINGS_STORAGE_KEY = "pluck-robot-settings";

export const useRobotStore = create<RobotState>((set) => ({
  status: "stopped",
  stats: defaultStats,
  targets: [],
  settings: defaultSettings,
  logs: [],
  setStatus: (status) => set({ status }),
  setStats: (stats) =>
    set((state) => ({
      stats: { ...state.stats, ...stats },
    })),
  setTargets: (targets) => set({ targets }),
  setSettings: (settings) => set({ settings }),
  pushLog: (log) =>
    set((state) => {
      const nextLog = {
        timestamp: log.timestamp ?? new Date().toISOString(),
        level: log.level,
        message: log.message,
      };
      const next = [...state.logs, nextLog];
      return { logs: next.slice(-MAX_LOGS) };
    }),
  startAlgo: () => set({ status: "running" }),
  stopAlgo: () => set({ status: "stopped" }),
}));

export function loadSettingsFromStorage(): RobotSettings | null {
  if (typeof window === "undefined") return null;
  const raw = localStorage.getItem(SETTINGS_STORAGE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw) as RobotSettings;
    return {
      detection: { ...defaultSettings.detection, ...parsed.detection },
    };
  } catch (error) {
    console.warn("Failed to parse stored settings", error);
    return null;
  }
}

export function persistSettings(settings: RobotSettings) {
  if (typeof window === "undefined") return;
  localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
}

export function useSettingsPersistence() {
  const setSettings = useRobotStore((state) => state.setSettings);
  const settings = useRobotStore((state) => state.settings);
  const hasHydrated = useRef(false);

  useEffect(() => {
    if (hasHydrated.current) return;
    const stored = loadSettingsFromStorage();
    if (stored) {
      setSettings(stored);
    }
    hasHydrated.current = true;
  }, [setSettings]);

  useEffect(() => {
    if (!hasHydrated.current) return;
    persistSettings(settings);
  }, [settings]);
}
