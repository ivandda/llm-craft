"use client";

import { useEffect, useState } from "react";

export type Theme = "light" | "dark";

const STORAGE_KEY = "llm-craft.theme";

function getStoredTheme(): Theme | null {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

function getSystemTheme(): Theme {
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

/** The theme to use on first load: explicit choice, else OS preference. */
export function resolveInitialTheme(): Theme {
  return getStoredTheme() ?? getSystemTheme();
}

function applyTheme(theme: Theme): void {
  document.documentElement.classList.toggle("dark", theme === "dark");
}

function storeTheme(theme: Theme): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, theme);
  } catch {
    // Ignore: theme just won't persist across reloads.
  }
}

// Module-level store so every mounted toggle stays in sync when any one of
// them flips the theme. Components start from "light" (matching the SSR/first
// client render to avoid hydration mismatch) and correct to the real theme in
// an effect; the no-FOUC inline script has already set the <html> class so
// there is no visible flash.
let currentTheme: Theme = "light";
let initialized = false;
const listeners = new Set<(theme: Theme) => void>();

function ensureInitialized(): void {
  if (initialized || typeof window === "undefined") {
    return;
  }

  currentTheme = resolveInitialTheme();
  initialized = true;
}

export function setTheme(theme: Theme): void {
  currentTheme = theme;
  initialized = true;
  applyTheme(theme);
  storeTheme(theme);
  listeners.forEach((listener) => listener(theme));
}

export function toggleTheme(): void {
  ensureInitialized();
  setTheme(currentTheme === "dark" ? "light" : "dark");
}

export function useTheme(): Theme {
  const [theme, setLocalTheme] = useState<Theme>("light");

  useEffect(() => {
    ensureInitialized();
    setLocalTheme(currentTheme);

    const listener = (next: Theme) => setLocalTheme(next);
    listeners.add(listener);

    return () => {
      listeners.delete(listener);
    };
  }, []);

  return theme;
}
