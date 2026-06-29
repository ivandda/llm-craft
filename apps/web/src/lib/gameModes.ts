import { BASE_ELEMENTS, createElementToken } from "@/lib/craft";
import type { ElementToken, GameMode, GoalPreset } from "@/lib/types";

const GOAL_STARTERS: ElementToken[] = [
  createElementToken("earth", "🌍"),
  createElementToken("rain", "🌧️")
];

export const GOAL_PRESET: GoalPreset = {
  id: "first-garden",
  mode: "goal",
  title: "First Garden",
  description: "A static mock goal ready for future backend validation.",
  objective: "Discover grass from earth and rain.",
  target: createElementToken("grass", "🌱"),
  metadata: {
    difficulty: "intro",
    status: "mock"
  },
  initialInventory: GOAL_STARTERS
};

export function getInitialInventoryForMode(mode: GameMode): ElementToken[] {
  return mode === "goal" ? GOAL_PRESET.initialInventory : BASE_ELEMENTS;
}
