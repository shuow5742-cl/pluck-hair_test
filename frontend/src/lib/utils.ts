import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(value: number) {
  return value.toLocaleString("zh-CN");
}

export function formatTimestamp(value: string) {
  return new Date(value).toLocaleString("zh-CN", {
    hour12: false,
  });
}
