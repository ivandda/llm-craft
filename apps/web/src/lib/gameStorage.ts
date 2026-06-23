import type { GameMode } from "@/lib/types";

export type GameStorageKind =
  | "inventory"
  | "history"
  | "board"
  | "darkMode"
  | "consumeInputs";

const GAME_STORAGE_VERSION = "v2";

export function createGameStorageKey(
  userId: string,
  mode: GameMode,
  kind: GameStorageKind
): string {
  return `llm-craft.${GAME_STORAGE_VERSION}.${normalizeStorageSegment(userId)}.${mode}.${kind}`;
}

function normalizeStorageSegment(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "-") || "anonymous";
}
