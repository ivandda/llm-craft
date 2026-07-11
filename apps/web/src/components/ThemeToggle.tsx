"use client";

import { toggleTheme, useTheme } from "@/lib/theme";
import { Moon, Sun } from "lucide-react";

/**
 * A single, app-wide theme toggle. Because theme lives in a shared store and
 * is applied to <html>, flipping it here re-themes every screen consistently.
 */
export function ThemeToggle({ className = "" }: { className?: string }) {
  const isDark = useTheme() === "dark";

  return (
    <button
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
      className={`grid size-10 place-items-center rounded-md border border-linen bg-surface text-soot transition hover:bg-paper hover:text-ink ${className}`}
      onClick={() => toggleTheme()}
      title={isDark ? "Switch to light mode" : "Switch to dark mode"}
      type="button"
    >
      {isDark ? <Sun size={17} /> : <Moon size={17} />}
    </button>
  );
}
