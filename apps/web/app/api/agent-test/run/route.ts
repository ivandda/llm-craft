import { isKnownAgentModel } from "@/lib/agentModels";
import { saveAgentRun } from "@/lib/server/agentRuns";
import { runAgentGoalTest } from "@/lib/server/agentTestRunner";
import { isValidGoalDepth } from "@/lib/server/randomGoals";
import { getClientIp, getSessionFromRequest } from "@/lib/server/requestAuth";
import { enforceRateLimits, RATE_LIMITS } from "@/lib/server/rateLimit";
import type { AgentTestRequest } from "@/lib/types";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const session = await getSessionFromRequest(request);

  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const payload = (await request.json().catch(() => null)) as
    | AgentTestRequest
    | null;

  if (
    !isValidGoalDepth(payload?.depth) ||
    (payload?.model !== undefined && !isKnownAgentModel(payload.model))
  ) {
    return NextResponse.json(
      { error: "Invalid agent test payload" },
      { status: 400 }
    );
  }

  const rateLimited = await enforceRateLimits([
    { scope: "agent-test:user", key: session.user.id, rule: RATE_LIMITS.agentTestUser },
    { scope: "agent-test:ip", key: getClientIp(request), rule: RATE_LIMITS.agentTestIp }
  ]);

  if (rateLimited) {
    return rateLimited;
  }

  const report = await runAgentGoalTest(payload);

  try {
    await saveAgentRun(report, session.user.id);
  } catch (error) {
    console.error("Failed to persist agent run", error);
  }

  return NextResponse.json({ report });
}
