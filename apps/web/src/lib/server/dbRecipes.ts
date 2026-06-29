import { combineElements, createElementToken, normalizeConcept } from "@/lib/craft";
import { query } from "@/lib/server/db";
import type { CombineRequest, CombineResponse, ElementToken } from "@/lib/types";

const FINAL_DATASET_NAME = "final-10k";

type CandidateRow = {
  output: string;
  rationale: string | null;
  rank: number;
};

export async function combineElementsWithDataset(
  request: CombineRequest
): Promise<CombineResponse> {
  const recipe = await findFinalDatasetRecipe(request.inputA.name, request.inputB.name);

  if (recipe) {
    return recipe;
  }

  return combineElements(request);
}

async function findFinalDatasetRecipe(
  inputA: string,
  inputB: string
): Promise<CombineResponse | null> {
  const [leftInput, rightInput] = [normalizeConcept(inputA), normalizeConcept(inputB)].sort();
  const result = await query<CandidateRow>(
    `
    SELECT rc.output, rc.rationale, rc.rank
    FROM recipe_pairs rp
    JOIN recipe_candidates rc ON rc.pair_id = rp.pair_id
    WHERE rp.dataset_name = $1
      AND rp.input_a = $2
      AND rp.input_b = $3
    ORDER BY rc.rank ASC
    `,
    [FINAL_DATASET_NAME, leftInput, rightInput]
  );

  if (result.rows.length === 0) {
    return null;
  }

  const knownOutputs = result.rows.map((row) => toElementToken(row.output));
  const topCandidate = result.rows[0];

  return {
    result: knownOutputs[0],
    source: "known_recipe",
    confidence: 0.98,
    knownOutputs,
    rationale: topCandidate.rationale ?? undefined
  };
}

function toElementToken(name: string): ElementToken {
  return createElementToken(name);
}
