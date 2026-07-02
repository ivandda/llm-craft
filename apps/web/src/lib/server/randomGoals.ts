import { BASE_ELEMENTS, createElementToken, normalizeConcept } from "@/lib/craft";
import { GOAL_PRESET } from "@/lib/gameModes";
import { query } from "@/lib/server/db";
import type { ElementToken, GoalPreset } from "@/lib/types";

const FINAL_DATASET_NAME = "final-10k";
export const MIN_GOAL_DEPTH = 1;
export const MAX_GOAL_DEPTH = 20;
const GOAL_STRATEGY = "bfs-depth";

export type GoalRecipe = {
  inputA: string;
  inputB: string;
  output: string;
};

export type GoalPathStep = {
  inputA: string;
  inputB: string;
  output: string;
};

export type GoalCandidate = {
  target: ElementToken;
  initialInventory: ElementToken[];
  initialInventoryId: string;
  minDepth: number;
  witnessPath: GoalPathStep[];
  seed: string;
  strategy: string;
};

type InitialInventoryPreset = {
  id: string;
  elements: ElementToken[];
  minDepth: number;
  maxDepth: number;
};

type GoalRecipeRow = {
  input_a: string;
  input_b: string;
  output: string;
};

type BuildGoalOptions = {
  random?: () => number;
  seed?: string;
};

const INITIAL_INVENTORY_PRESETS: InitialInventoryPreset[] = [
  {
    id: "classic",
    elements: BASE_ELEMENTS,
    minDepth: 1,
    maxDepth: MAX_GOAL_DEPTH
  },
  {
    id: "growth",
    elements: [createElementToken("earth"), createElementToken("rain")],
    minDepth: 1,
    maxDepth: 10
  },
  {
    id: "thermal",
    elements: [
      createElementToken("fire"),
      createElementToken("water"),
      createElementToken("earth")
    ],
    minDepth: 1,
    maxDepth: 14
  },
  {
    id: "atmosphere",
    elements: [
      createElementToken("air"),
      createElementToken("water"),
      createElementToken("fire")
    ],
    minDepth: 1,
    maxDepth: 14
  }
];

export function isValidGoalDepth(depth: unknown): depth is number {
  return (
    Number.isInteger(depth) &&
    typeof depth === "number" &&
    depth >= MIN_GOAL_DEPTH &&
    depth <= MAX_GOAL_DEPTH
  );
}

export async function generateRandomGoal(depth: number): Promise<GoalPreset> {
  if (!isValidGoalDepth(depth)) {
    throw new Error("Invalid goal depth");
  }

  const recipes = await listGoalRecipes();
  return (
    buildRandomGoalFromRecipes(recipes, depth, {
      seed: createDailyGoalSeed(depth)
    }) ?? GOAL_PRESET
  );
}

export function buildRandomGoalFromRecipes(
  recipes: GoalRecipe[],
  depth: number,
  options: BuildGoalOptions | (() => number) = {}
): GoalPreset | null {
  const candidate = buildRandomGoalCandidate(recipes, depth, options);

  if (!candidate) {
    return null;
  }

  return toGoalPreset(candidate, depth);
}

export function buildRandomGoalCandidate(
  recipes: GoalRecipe[],
  depth: number,
  options: BuildGoalOptions | (() => number) = {}
): GoalCandidate | null {
  if (!isValidGoalDepth(depth)) {
    return null;
  }

  const normalizedOptions =
    typeof options === "function" ? { random: options } : options;
  const seed = normalizedOptions.seed ?? `test-depth-${depth}`;
  const random = normalizedOptions.random ?? createSeededRandom(seed);
  const normalizedRecipes = normalizeGoalRecipes(recipes);
  const compatiblePresets = INITIAL_INVENTORY_PRESETS.filter(
    (preset) => depth >= preset.minDepth && depth <= preset.maxDepth
  );
  const candidateDepths = [depth, depth - 1, depth + 1].filter(isValidGoalDepth);

  for (const candidateDepth of candidateDepths) {
    const candidates = compatiblePresets.flatMap((preset) =>
      buildCandidatesForPreset(normalizedRecipes, preset, candidateDepth, seed)
    );

    if (candidates.length === 0) {
      continue;
    }

    return candidates[Math.floor(random() * candidates.length)];
  }

  return null;
}

async function listGoalRecipes(): Promise<GoalRecipe[]> {
  const result = await query<GoalRecipeRow>(
    `
    SELECT DISTINCT rp.input_a, rp.input_b, rc.output
    FROM recipe_pairs rp
    JOIN recipe_candidates rc ON rc.pair_id = rp.pair_id
    WHERE rp.dataset_name = $1
      AND rc.rank = 1
    `,
    [FINAL_DATASET_NAME]
  );

  return result.rows.map((row) => ({
    inputA: normalizeConcept(row.input_a),
    inputB: normalizeConcept(row.input_b),
    output: normalizeConcept(row.output)
  }));
}

function buildCandidatesForPreset(
  recipes: GoalRecipe[],
  preset: InitialInventoryPreset,
  depth: number,
  seed: string
): GoalCandidate[] {
  const paths = buildReachablePaths(recipes, preset.elements, depth);

  return [...paths.entries()]
    .filter(([, path]) => path.length === depth && validatePath(preset.elements, path))
    .map(([targetName, witnessPath]) => ({
      target: createElementToken(targetName),
      initialInventory: preset.elements,
      initialInventoryId: preset.id,
      minDepth: witnessPath.length,
      witnessPath,
      seed,
      strategy: GOAL_STRATEGY
    }));
}

function buildReachablePaths(
  recipes: GoalRecipe[],
  initialInventory: ElementToken[],
  maxDepth: number
): Map<string, GoalPathStep[]> {
  const initialIds = new Set(initialInventory.map((element) => element.id));
  const paths = new Map<string, GoalPathStep[]>();

  for (let pass = 1; pass <= maxDepth; pass += 1) {
    let didImprove = false;

    for (const recipe of recipes) {
      if (initialIds.has(recipe.output)) {
        continue;
      }

      const leftPath = getPathForInput(paths, initialIds, recipe.inputA);
      const rightPath = getPathForInput(paths, initialIds, recipe.inputB);

      if (!leftPath || !rightPath) {
        continue;
      }

      const nextPath = mergePlans(leftPath, rightPath, recipe);
      const currentPath = paths.get(recipe.output);

      if (nextPath.length <= maxDepth && (!currentPath || nextPath.length < currentPath.length)) {
        paths.set(recipe.output, nextPath);
        didImprove = true;
      }
    }

    if (!didImprove) {
      break;
    }
  }

  return paths;
}

function getPathForInput(
  paths: Map<string, GoalPathStep[]>,
  initialIds: Set<string>,
  input: string
): GoalPathStep[] | null {
  if (initialIds.has(input)) {
    return [];
  }

  return paths.get(input) ?? null;
}

function mergePlans(
  leftPath: GoalPathStep[],
  rightPath: GoalPathStep[],
  recipe: GoalRecipe
): GoalPathStep[] {
  const seenOutputs = new Set<string>();
  const mergedPath: GoalPathStep[] = [];

  for (const step of [...leftPath, ...rightPath, recipe]) {
    if (!seenOutputs.has(step.output)) {
      seenOutputs.add(step.output);
      mergedPath.push(step);
    }
  }

  return mergedPath;
}

function validatePath(
  initialInventory: ElementToken[],
  witnessPath: GoalPathStep[]
): boolean {
  const inventory = new Set(initialInventory.map((element) => element.id));

  for (const step of witnessPath) {
    if (!inventory.has(step.inputA) || !inventory.has(step.inputB)) {
      return false;
    }

    inventory.add(step.output);
  }

  return witnessPath.length > 0;
}

function toGoalPreset(candidate: GoalCandidate, requestedDepth: number): GoalPreset {
  return {
    id: `random-${candidate.initialInventoryId}-depth-${candidate.minDepth}-${candidate.target.id}-${hashString(candidate.seed).toString(36)}`,
    mode: "goal",
    title: `Random depth ${candidate.minDepth}`,
    description: "Generated from real recipe data.",
    objective: `Discover ${candidate.target.name} in ${candidate.minDepth} combinations.`,
    target: candidate.target,
    metadata: {
      difficulty: `depth-${candidate.minDepth}`,
      status: "generated",
      depth: requestedDepth,
      minDepth: candidate.minDepth,
      seed: candidate.seed,
      strategy: candidate.strategy,
      initialInventoryId: candidate.initialInventoryId
    },
    initialInventory: candidate.initialInventory
  };
}

function normalizeGoalRecipes(recipes: GoalRecipe[]): GoalRecipe[] {
  const uniqueRecipes = new Map<string, GoalRecipe>();

  for (const recipe of recipes) {
    const normalizedRecipe = {
      inputA: normalizeConcept(recipe.inputA),
      inputB: normalizeConcept(recipe.inputB),
      output: normalizeConcept(recipe.output)
    };
    const key = `${normalizedRecipe.inputA}+${normalizedRecipe.inputB}=>${normalizedRecipe.output}`;

    if (
      normalizedRecipe.inputA &&
      normalizedRecipe.inputB &&
      normalizedRecipe.output &&
      normalizedRecipe.inputA !== normalizedRecipe.output &&
      normalizedRecipe.inputB !== normalizedRecipe.output &&
      !uniqueRecipes.has(key)
    ) {
      uniqueRecipes.set(key, normalizedRecipe);
    }
  }

  return [...uniqueRecipes.values()];
}

function createDailyGoalSeed(depth: number): string {
  return `goal:${new Date().toISOString().slice(0, 10)}:${depth}`;
}

function createSeededRandom(seed: string): () => number {
  let state = hashString(seed);

  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function hashString(value: string): number {
  let hash = 2166136261;

  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }

  return hash >>> 0;
}
