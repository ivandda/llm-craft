import { mergeInventory } from "@/lib/craft";
import type { AgentTestStep, ElementToken } from "@/lib/types";

export function buildAgentPlaybackInventory(
  initialInventory: ElementToken[],
  steps: AgentTestStep[],
  playbackIndex: number
): ElementToken[] {
  const safeIndex = Math.max(0, Math.min(playbackIndex, steps.length));

  return steps
    .slice(0, safeIndex)
    .reduce(
      (inventory, step) => mergeInventory(inventory, step.output),
      initialInventory
    );
}
