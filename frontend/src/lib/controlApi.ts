// API client for the pluck-hair_test console (NEW).
// Wraps the /api/test, /api/epson, /api/io and /api/system endpoints.

import { buildApiUrl } from "@/lib/api";

// ---- types ------------------------------------------------------------------
export interface TweezerState {
  found: boolean;
  is_open: boolean | null;
  state: "open" | "closed" | null;
  tip_xy: [number, number] | null;
  tips: number[][];
  entry_side: string | null;
  confidence: number;
  held: boolean;
}

export interface LiveState {
  fps: number;
  frame: number;
  detection_count: number;
  ar_mode: string;
  tweezer: TweezerState;
  tip_to_pick_mm: number | null;
  updated_at: number;
}

export interface EpsonDescribe {
  backend: string;
  axes: string[];
  units: Record<string, string>;
  limits: Record<string, [number, number]>;
  points: Array<{ name: string } & Record<string, number>>;
  commands: string[];
  step_modes: string[];
  devices: IoDeviceMeta[];
  system_buttons: Array<{ id: string; name: string }>;
}

export interface IoDeviceMeta {
  id: string;
  name: string;
  plc_tag: string;
  kind: "cylinder" | "other";
  open_label: string;
  close_label: string;
}

export interface IoDeviceState extends IoDeviceMeta {
  state: string;
  feedback: string;
  in_position: boolean;
  is_open: boolean;
}

// ---- fetch helpers ----------------------------------------------------------
async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(buildApiUrl(path));
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(buildApiUrl(path), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json() as Promise<T>;
}

export const swrFetcher = <T>(path: string): Promise<T> => getJson<T>(path);

// ---- left half: video + live state -----------------------------------------
export function testStreamUrl() {
  return buildApiUrl("/test/stream/video");
}

export const getLiveState = () => getJson<LiveState>("/test/state");

// ---- right half: Epson ------------------------------------------------------
export const getEpsonDescribe = () => getJson<EpsonDescribe>("/epson/describe");
export const getEpsonPose = () =>
  getJson<{ pose: Record<string, number>; units: Record<string, string> }>("/epson/pose");

export const epsonJog = (axis: string, direction: "+" | "-", step_mode: string) =>
  postJson<{ ok: boolean; pose?: Record<string, number>; error?: string }>(
    "/epson/jog",
    { axis, direction, step_mode },
  );

export const epsonMove = (
  payload: { target?: Record<string, number>; point?: string; command: string },
) =>
  postJson<{ ok: boolean; pose?: Record<string, number>; error?: string }>(
    "/epson/move",
    payload,
  );

// ---- right half: IO + system ------------------------------------------------
export const getIoStates = () => getJson<{ devices: IoDeviceState[] }>("/io/states");

export const setIo = (device_id: string, action: string) =>
  postJson<{ ok: boolean; device?: IoDeviceState; error?: string }>("/io/set", {
    device_id,
    action,
  });

export const systemCommand = (command: string) =>
  postJson<{ ok: boolean; command?: string; error?: string }>("/system", { command });
