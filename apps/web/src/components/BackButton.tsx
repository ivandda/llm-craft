"use client";

import { ArrowLeft } from "lucide-react";

/**
 * A single, labeled back control used across screens so "go back" is always
 * an arrow + word (never a bare icon or an ambiguous menu button).
 */
export function BackButton({
  label = "Back",
  onClick
}: {
  label?: string;
  onClick: () => void;
}) {
  return (
    <button
      className="flex h-10 items-center gap-1.5 rounded-md border border-linen bg-surface px-3 text-sm font-medium text-soot transition hover:bg-paper hover:text-ink"
      onClick={onClick}
      type="button"
    >
      <ArrowLeft size={16} />
      {label}
    </button>
  );
}
