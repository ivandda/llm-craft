import { combineElementsWithDataset } from "@/lib/server/dbRecipes";
import {
  VertexConfigurationError,
  VertexGenerationError
} from "@/lib/server/vertexCombiner";
import {
  QwenConfigurationError,
  QwenGenerationError
} from "@/lib/server/qwenCombiner";
import { getClientIp, getSessionFromRequest } from "@/lib/server/requestAuth";
import { enforceRateLimits, RATE_LIMITS } from "@/lib/server/rateLimit";
import { isKnownCombinerModel } from "@/lib/agentModels";
import type { CombineRequest } from "@/lib/types";
import { NextResponse } from "next/server";

const MAX_INVENTORY_PAYLOAD = 600;
const MAX_INVENTORY_FOR_MODEL = 120;

export async function POST(request: Request) {
  const session = await getSessionFromRequest(request);

  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const payload = (await request.json().catch(() => null)) as CombineRequest | null;

  if (!isCombineRequest(payload)) {
    return NextResponse.json(
      { error: "Invalid combine request" },
      { status: 400 }
    );
  }

  const rateLimited = await enforceRateLimits([
    { scope: "combine:user", key: session.user.id, rule: RATE_LIMITS.combineUser },
    { scope: "combine:ip", key: getClientIp(request), rule: RATE_LIMITS.combineIp }
  ]);

  if (rateLimited) {
    return rateLimited;
  }

  try {
    return NextResponse.json(
      await combineElementsWithDataset({
        ...payload,
        inventory: payload.inventory.slice(-MAX_INVENTORY_FOR_MODEL)
      })
    );
  } catch (error) {
    if (
      error instanceof VertexConfigurationError ||
      error instanceof VertexGenerationError ||
      error instanceof QwenConfigurationError ||
      error instanceof QwenGenerationError
    ) {
      return NextResponse.json(
        { error: "Model-backed combination is unavailable" },
        { status: 503 }
      );
    }

    throw error;
  }
}

function isCombineRequest(value: CombineRequest | null): value is CombineRequest {
  return Boolean(
    value?.inputA?.id &&
      value.inputA.name &&
      value?.inputB?.id &&
      value.inputB.name &&
      Array.isArray(value.inventory) &&
      value.inventory.length <= MAX_INVENTORY_PAYLOAD &&
      (value.model === undefined || isKnownCombinerModel(value.model))
  );
}
