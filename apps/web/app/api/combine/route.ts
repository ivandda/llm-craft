import { combineElementsWithDataset } from "@/lib/server/dbRecipes";
import {
  VertexConfigurationError,
  VertexGenerationError
} from "@/lib/server/vertexCombiner";
import { isKnownVertexModel } from "@/lib/agentModels";
import type { CombineRequest } from "@/lib/types";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => null)) as CombineRequest | null;

  if (!isCombineRequest(payload)) {
    return NextResponse.json(
      { error: "Invalid combine request" },
      { status: 400 }
    );
  }

  try {
    return NextResponse.json(await combineElementsWithDataset(payload));
  } catch (error) {
    if (
      error instanceof VertexConfigurationError ||
      error instanceof VertexGenerationError
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
      (value.model === undefined || isKnownVertexModel(value.model))
  );
}
