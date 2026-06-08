import { cn, formatNumber } from "@/lib/utils";

interface StatCardProps {
  title: string;
  value: number;
  unit?: string;
  highlight?: "accent" | "danger";
}

export function StatCard({
  title,
  value,
  unit,
  highlight = "accent",
}: StatCardProps) {
  return (
    <div className="rounded-lg border border-border bg-panel px-4 py-3 shadow-panel">
      <p className="text-sm text-muted">{title}</p>
      <div className="mt-2 flex items-end gap-2">
        <span
          className={cn(
            "text-3xl font-semibold leading-none",
            highlight === "accent" ? "text-text" : "text-danger",
          )}
        >
          {formatNumber(value)}
        </span>
        {unit ? <span className="pb-1 text-sm text-muted">{unit}</span> : null}
      </div>
    </div>
  );
}
