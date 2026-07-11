import type { ElementToken } from "@/lib/types";

const MAX_DPO_OPTIONS = 3;

export function selectDpoCandidates(
  knownOutputs: ElementToken[] | undefined,
  random = Math.random
): ElementToken[] {
  const uniqueOutputs = dedupeOutputs(knownOutputs ?? []);

  if (uniqueOutputs.length < 2) {
    return [];
  }

  // Always keep the canonical (rank-1) output among the choices: goal-mode
  // paths are computed from rank-1 recipes, so hiding it could make a goal
  // unreachable in the promised number of combinations.
  const [topOutput, ...alternatives] = uniqueOutputs;
  const shownOutputs = [
    topOutput,
    ...shuffleOutputs(alternatives, random).slice(0, MAX_DPO_OPTIONS - 1)
  ];

  return shuffleOutputs(shownOutputs, random);
}

function dedupeOutputs(outputs: ElementToken[]): ElementToken[] {
  const seenIds = new Set<string>();
  const uniqueOutputs: ElementToken[] = [];

  for (const output of outputs) {
    if (!seenIds.has(output.id)) {
      seenIds.add(output.id);
      uniqueOutputs.push(output);
    }
  }

  return uniqueOutputs;
}

function shuffleOutputs(
  outputs: ElementToken[],
  random: () => number
): ElementToken[] {
  const shuffledOutputs = [...outputs];

  for (let index = shuffledOutputs.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [shuffledOutputs[index], shuffledOutputs[swapIndex]] = [
      shuffledOutputs[swapIndex],
      shuffledOutputs[index]
    ];
  }

  return shuffledOutputs;
}
