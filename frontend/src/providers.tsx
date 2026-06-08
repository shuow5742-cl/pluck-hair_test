"use client";

import { ReactNode } from "react";
import { SWRConfig } from "swr";
import { buildApiUrl } from "@/lib/api";
import { useSettingsPersistence } from "@/store/robot-store";

const fetcher = async (path: string) => {
  const url = buildApiUrl(path);
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`请求失败：${res.status}`);
  }
  return res.json();
};

export function Providers({ children }: { children: ReactNode }) {
  useSettingsPersistence();

  return (
    <SWRConfig
      value={{
        fetcher,
        revalidateOnFocus: false,
        refreshWhenHidden: false,
        errorRetryInterval: 4000,
      }}
    >
      {children}
    </SWRConfig>
  );
}
