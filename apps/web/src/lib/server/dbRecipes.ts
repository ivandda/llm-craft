import { createElementToken, normalizeConcept } from "@/lib/craft";
import { getEmojiForConcept } from "@/lib/emoji";
import {
  DEFAULT_VERTEX_MODEL,
  isQwenCombinerModel,
  QWEN_COMBINER_MODEL
} from "@/lib/agentModels";
import { query, transaction } from "@/lib/server/db";
import {
  generateCombinationWithQwen,
  QwenConfigurationError,
  QwenGenerationError
} from "@/lib/server/qwenCombiner";
import {
  generateCombinationWithVertex,
  getVertexModel
} from "@/lib/server/vertexCombiner";
import type {
  CombineRequest,
  CombineResponse,
  DpoCandidate,
  DpoCandidatesResponse,
  ElementToken
} from "@/lib/types";
import { createHash, randomBytes } from "crypto";
import type { PoolClient } from "pg";

const FINAL_DATASET_NAME = "final-10k";
const WEB_GENERATED_DATASET_NAME = "web-generated";

type CandidateRow = {
  dataset_name: string;
  output: string;
  rationale: string | null;
  rank: number;
  raw_candidate: unknown;
};

export async function combineElementsWithDataset(
  request: CombineRequest
): Promise<CombineResponse> {
  const model = request.model ?? getVertexModel();
  const recipe = await findStoredRecipe(
    request.inputA.name,
    request.inputB.name,
    model
  );

  if (recipe) {
    return recipe;
  }

  const { generatedRecipe, effectiveModel } = await generateWithFallback(
    request,
    model
  );
  await saveGeneratedRecipe(
    request,
    generatedRecipe,
    effectiveModel,
    getGenerationSource(effectiveModel)
  );
  return generatedRecipe;
}

const MAX_DPO_CANDIDATES = 3;

/**
 * Assembles a blind set of candidates for a "Help train the AI" round: the
 * stored canonical output (if any) plus live generations from models that did
 * not produce it. Each candidate carries `generatedBy` for training-data
 * attribution; the UI must not display it.
 */
export async function buildDpoCandidates(
  request: CombineRequest
): Promise<DpoCandidatesResponse> {
  const model = request.model ?? getVertexModel();
  const stored = await findStoredRecipe(
    request.inputA.name,
    request.inputB.name,
    model
  );
  const candidates: DpoCandidate[] = [];
  const canonicalModel = stored?.source === "model_generated" ? stored.model : undefined;

  if (stored) {
    candidates.push({
      ...stored.result,
      generatedBy: stored.source === "known_recipe" ? "dataset" : canonicalModel
    });
  }

  const liveModels = [QWEN_COMBINER_MODEL, getVertexModel()].filter(
    (liveModel) => liveModel !== canonicalModel
  );
  const attempts = await Promise.allSettled(
    liveModels.map((liveModel) =>
      (isQwenCombinerModel(liveModel)
        ? generateCombinationWithQwen(request, liveModel)
        : generateCombinationWithVertex(request, liveModel)
      ).then(async (response) => {
        await saveGeneratedRecipe(
          request,
          response,
          liveModel,
          getGenerationSource(liveModel)
        );
        return { response, liveModel };
      })
    )
  );

  for (const attempt of attempts) {
    if (attempt.status === "fulfilled") {
      candidates.push({
        ...attempt.value.response.result,
        generatedBy: attempt.value.liveModel
      });
    }
  }

  if (stored?.knownOutputs) {
    for (const alternative of stored.knownOutputs.slice(1)) {
      candidates.push({
        ...alternative,
        generatedBy: stored.source === "known_recipe" ? "dataset" : canonicalModel
      });
    }
  }

  return {
    candidates: dedupeCandidates(candidates).slice(0, MAX_DPO_CANDIDATES),
    source: stored ? stored.source : "model_generated"
  };
}

function dedupeCandidates(candidates: DpoCandidate[]): DpoCandidate[] {
  const seenIds = new Set<string>();
  const uniqueCandidates: DpoCandidate[] = [];

  for (const candidate of candidates) {
    if (!seenIds.has(candidate.id)) {
      seenIds.add(candidate.id);
      uniqueCandidates.push(candidate);
    }
  }

  return uniqueCandidates;
}

async function generateWithFallback(
  request: CombineRequest,
  model: string
): Promise<{ generatedRecipe: CombineResponse; effectiveModel: string }> {
  if (!isQwenCombinerModel(model)) {
    return {
      generatedRecipe: await generateCombinationWithVertex(request, model),
      effectiveModel: model
    };
  }

  try {
    return {
      generatedRecipe: await generateCombinationWithQwen(request, model),
      effectiveModel: model
    };
  } catch (error) {
    // The Qwen VM is expected to be off most of the time (GPU cost); fall
    // back to Vertex so the game keeps working instead of erroring.
    if (
      error instanceof QwenConfigurationError ||
      error instanceof QwenGenerationError
    ) {
      const fallbackModel = getVertexModel();

      return {
        generatedRecipe: await generateCombinationWithVertex(request, fallbackModel),
        effectiveModel: fallbackModel
      };
    }

    throw error;
  }
}

async function findStoredRecipe(
  inputA: string,
  inputB: string,
  model: string
): Promise<CombineResponse | null> {
  const [leftInput, rightInput] = [normalizeConcept(inputA), normalizeConcept(inputB)].sort();
  const generatedDatasets = getGeneratedDatasetLookupOrder(model);
  const result = await query<CandidateRow>(
    `
    SELECT rp.dataset_name, rc.output, rc.rationale, rc.rank, rc.raw_candidate
    FROM recipe_pairs rp
    JOIN recipe_candidates rc ON rc.pair_id = rp.pair_id
    WHERE rp.dataset_name = ANY($2::text[])
      AND rp.input_a = $3
      AND rp.input_b = $4
    ORDER BY
      CASE rp.dataset_name
        WHEN $1 THEN 0
        ELSE 2
      END,
      rc.rank ASC
    `,
    [
      FINAL_DATASET_NAME,
      [FINAL_DATASET_NAME, ...generatedDatasets],
      leftInput,
      rightInput
    ]
  );

  if (result.rows.length === 0) {
    return null;
  }

  const datasetName = result.rows[0].dataset_name;
  const datasetRows = result.rows.filter((row) => row.dataset_name === datasetName);
  const knownOutputs = datasetRows.map(toElementToken);
  const topCandidate = datasetRows[0];

  return {
    result: knownOutputs[0],
    source: datasetName === FINAL_DATASET_NAME ? "known_recipe" : "model_generated",
    model: datasetName === FINAL_DATASET_NAME ? undefined : readGeneratedModel(topCandidate, model),
    confidence: datasetName === FINAL_DATASET_NAME ? 0.98 : 0.72,
    knownOutputs,
    rationale: topCandidate.rationale ?? undefined
  };
}

async function saveGeneratedRecipe(
  request: CombineRequest,
  response: CombineResponse,
  model: string,
  generationSource: string
): Promise<void> {
  const [leftInput, rightInput] = [
    normalizeConcept(request.inputA.name),
    normalizeConcept(request.inputB.name)
  ].sort();
  const datasetName = getGeneratedDatasetName(model);
  const pairId = createPairId(datasetName, leftInput, rightInput);

  await transaction(async (client) => {
    await ensureGeneratedDataset(client, datasetName, model, generationSource);
    const storedPairId = await upsertGeneratedPair(client, {
      datasetName,
      pairId,
      leftInput,
      rightInput,
      model
    });
    await upsertGeneratedCandidate(client, storedPairId, response, model);
  });
}

async function ensureGeneratedDataset(
  client: PoolClient,
  datasetName: string,
  model: string,
  generationSource: string
): Promise<void> {
  await client.query(
    `
    INSERT INTO dataset_imports (dataset_name, source_dir, raw_metadata)
    VALUES ($1, $2, $3::jsonb)
    ON CONFLICT (dataset_name) DO NOTHING
    `,
    [
      datasetName,
      "apps/web",
      toJsonb({ source: generationSource, model })
    ]
  );
}

async function upsertGeneratedPair(
  client: PoolClient,
  input: {
    datasetName: string;
    pairId: string;
    leftInput: string;
    rightInput: string;
    model: string;
  }
): Promise<string> {
  const result = await client.query<{ pair_id: string }>(
    `
    INSERT INTO recipe_pairs (
      pair_id, dataset_name, split, input_a, input_b, raw_record
    )
    VALUES ($1, $2, 'train', $3, $4, $5::jsonb)
    ON CONFLICT (dataset_name, input_a, input_b)
    DO UPDATE SET raw_record = EXCLUDED.raw_record
    RETURNING pair_id
    `,
    [
      input.pairId,
      input.datasetName,
      input.leftInput,
      input.rightInput,
      toJsonb({
        input: [input.leftInput, input.rightInput],
        model: input.model,
        generatedAt: new Date().toISOString()
      })
    ]
  );

  return result.rows[0].pair_id;
}

async function upsertGeneratedCandidate(
  client: PoolClient,
  pairId: string,
  response: CombineResponse,
  model: string
): Promise<void> {
  await client.query(
    `
    INSERT INTO recipe_candidates (
      candidate_id, pair_id, output, source, rationale, rank, raw_candidate
    )
    VALUES ($1, $2, $3, 'teacher', $4, 1, $5::jsonb)
    ON CONFLICT (pair_id, rank)
    DO UPDATE SET
      output = EXCLUDED.output,
      rationale = EXCLUDED.rationale,
      raw_candidate = EXCLUDED.raw_candidate
    `,
    [
      `${pairId}:rank-1`,
      pairId,
      response.result.name,
      response.rationale ?? null,
      toJsonb(createRawCandidate(response, model))
    ]
  );
}

function createRawCandidate(response: CombineResponse, model: string) {
  return {
    id: randomBytes(12).toString("hex"),
    model,
    output: response.result,
    source: response.source
  };
}

function toElementToken(row: CandidateRow): ElementToken {
  const rawOutput = readRawOutput(row.raw_candidate);
  return createElementToken(
    row.output,
    rawOutput?.emoji ?? getEmojiForConcept(row.output)
  );
}

function createPairId(datasetName: string, inputA: string, inputB: string): string {
  return `${datasetName}:${createHash("sha256")
    .update(`${inputA}+${inputB}`)
    .digest("hex")
    .slice(0, 24)}`;
}

function getGeneratedDatasetName(model: string): string {
  return `${WEB_GENERATED_DATASET_NAME}-${model}`;
}

function getGeneratedDatasetLookupOrder(model: string): string[] {
  const datasets = [getGeneratedDatasetName(model)];

  if (model === DEFAULT_VERTEX_MODEL) {
    datasets.push(WEB_GENERATED_DATASET_NAME);
  }

  return datasets;
}

function getGenerationSource(model: string): string {
  return isQwenCombinerModel(model)
    ? "qwen-vm-web-combinator"
    : "vertex-web-combinator";
}

function toJsonb(value: unknown): string {
  return JSON.stringify(value);
}

function readRawOutput(rawCandidate: unknown): ElementToken | null {
  if (!rawCandidate || typeof rawCandidate !== "object") {
    return null;
  }

  const output = (rawCandidate as { output?: unknown }).output;

  if (!output || typeof output !== "object") {
    return null;
  }

  return output as ElementToken;
}

function readGeneratedModel(row: CandidateRow, fallbackModel: string): string {
  if (row.raw_candidate && typeof row.raw_candidate === "object") {
    const model = (row.raw_candidate as { model?: unknown }).model;

    if (typeof model === "string" && model.trim()) {
      return model;
    }
  }

  return row.dataset_name === WEB_GENERATED_DATASET_NAME
    ? DEFAULT_VERTEX_MODEL
    : fallbackModel;
}
