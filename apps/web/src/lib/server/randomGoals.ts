import { BASE_ELEMENTS, createElementToken, normalizeConcept } from "@/lib/craft";
import { GOAL_PRESET } from "@/lib/gameModes";
import { query } from "@/lib/server/db";
import type { GoalPreset } from "@/lib/types";

const FINAL_DATASET_NAME = "final-10k";
export const MIN_GOAL_DEPTH = 1;
export const MAX_GOAL_DEPTH = 20;

export type GoalRecipe = {
  inputA: string;
  inputB: string;
  output: string;
};

type GoalRecipeRow = {
  input_a: string;
  input_b: string;
  output: string;
};

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
  return buildRandomGoalFromRecipes(recipes, depth) ?? GOAL_PRESET;
}

export function buildRandomGoalFromRecipes(
  recipes: GoalRecipe[],
  depth: number,
  random = Math.random
): GoalPreset | null {
  if (!isValidGoalDepth(depth)) {
    return null;
  }

  const inventory = new Set(BASE_ELEMENTS.map((element) => element.id));
  let targetName = "";

  for (let step = 1; step <= depth; step += 1) {
    const candidates = recipes.filter(
      (recipe) =>
        inventory.has(recipe.inputA) &&
        inventory.has(recipe.inputB) &&
        !inventory.has(recipe.output)
    );

    if (candidates.length === 0) {
      return null;
    }

    const nextRecipe = candidates[Math.floor(random() * candidates.length)];
    inventory.add(nextRecipe.output);
    targetName = nextRecipe.output;
  }

  const target = createElementToken(targetName);

  return {
    id: `random-depth-${depth}-${target.id}`,
    mode: "goal",
    title: `Random depth ${depth}`,
    description: "Generated from real recipe data.",
    objective: `Discover ${target.name} in ${depth} combinations.`,
    target,
    metadata: {
      difficulty: `depth-${depth}`,
      status: "generated",
      depth
    },
    initialInventory: BASE_ELEMENTS
  };
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
