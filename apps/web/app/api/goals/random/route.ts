import { generateRandomGoal, isValidGoalDepth } from "@/lib/server/randomGoals";
import { getClientIp, getSessionFromRequest } from "@/lib/server/requestAuth";
import { enforceRateLimits, RATE_LIMITS } from "@/lib/server/rateLimit";
import type { RandomGoalRequest } from "@/lib/types";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const session = await getSessionFromRequest(request);

  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const payload = (await request.json().catch(() => null)) as
    | RandomGoalRequest
    | null;

  if (
    !isValidGoalDepth(payload?.depth) ||
    (payload?.seed !== undefined &&
      (typeof payload.seed !== "string" || payload.seed.length > 80))
  ) {
    return NextResponse.json(
      { error: "Invalid random goal payload" },
      { status: 400 }
    );
  }

  const rateLimited = await enforceRateLimits([
    { scope: "random-goal:user", key: session.user.id, rule: RATE_LIMITS.randomGoalUser },
    { scope: "random-goal:ip", key: getClientIp(request), rule: RATE_LIMITS.randomGoalIp }
  ]);

  if (rateLimited) {
    return rateLimited;
  }

  return NextResponse.json({
    goal: await generateRandomGoal(
      payload.depth,
      payload.seed ? `goal:${payload.seed}:${payload.depth}` : undefined
    )
  });
}
