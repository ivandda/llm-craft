import { isKnownCombinerModel } from "@/lib/agentModels";
import { buildDpoCandidates } from "@/lib/server/dbRecipes";
import { getClientIp, getSessionFromRequest } from "@/lib/server/requestAuth";
import { enforceRateLimits, RATE_LIMITS } from "@/lib/server/rateLimit";
import type { DpoCandidatesRequest } from "@/lib/types";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const session = await getSessionFromRequest(request);

  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const payload = (await request.json().catch(() => null)) as
    | DpoCandidatesRequest
    | null;

  if (!isDpoCandidatesRequest(payload)) {
    return NextResponse.json(
      { error: "Invalid candidates request" },
      { status: 400 }
    );
  }

  const rateLimited = await enforceRateLimits([
    {
      scope: "dpo-candidates:user",
      key: session.user.id,
      rule: RATE_LIMITS.dpoCandidatesUser
    },
    {
      scope: "dpo-candidates:ip",
      key: getClientIp(request),
      rule: RATE_LIMITS.dpoCandidatesIp
    }
  ]);

  if (rateLimited) {
    return rateLimited;
  }

  // Cap the inventory sent to the model, mirroring the combine route: the
  // request allows up to 600 items, but only the most recent matter for
  // generation and the full list inflates prompt token cost/latency.
  return NextResponse.json(
    await buildDpoCandidates({
      ...payload,
      inventory: payload.inventory.slice(-MAX_INVENTORY_FOR_MODEL)
    })
  );
}

const MAX_INVENTORY_FOR_MODEL = 120;

function isDpoCandidatesRequest(
  value: DpoCandidatesRequest | null
): value is DpoCandidatesRequest {
  return Boolean(
    value?.inputA?.id &&
      value.inputA.name &&
      value?.inputB?.id &&
      value.inputB.name &&
      Array.isArray(value.inventory) &&
      value.inventory.length <= 600 &&
      (value.model === undefined || isKnownCombinerModel(value.model))
  );
}
