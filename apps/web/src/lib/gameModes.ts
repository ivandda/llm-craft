import { BASE_ELEMENTS, createElementToken } from "@/lib/craft";
import type { ElementToken, GameMode, GoalPreset } from "@/lib/types";

const GOAL_STARTERS: ElementToken[] = [
  createElementToken("earth"),
  createElementToken("rain")
];

export const GOAL_PRESET: GoalPreset = {
  id: "fallback-first-plant",
  mode: "goal",
  title: "First Plant",
  description: "Fallback goal used when random generation is unavailable.",
  objective: "Discover plant from earth and rain.",
  target: createElementToken("plant"),
  metadata: {
    difficulty: "intro",
    status: "mock",
    depth: 1,
    minDepth: 1,
    strategy: "fallback",
    initialInventoryId: "growth"
  },
  initialInventory: GOAL_STARTERS
};

export function getInitialInventoryForMode(mode: GameMode): ElementToken[] {
  return mode === "goal" ? GOAL_PRESET.initialInventory : BASE_ELEMENTS;
}
