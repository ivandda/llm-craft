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

  return shuffleOutputs(uniqueOutputs, random).slice(
    0,
    Math.min(MAX_DPO_OPTIONS, uniqueOutputs.length)
  );
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
