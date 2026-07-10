import { isKnownAgentModel } from "@/lib/agentModels";
import { runAgentGoalTest } from "@/lib/server/agentTestRunner";
import { isValidGoalDepth } from "@/lib/server/randomGoals";
import type { AgentTestRequest } from "@/lib/types";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
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

  return NextResponse.json({ report: await runAgentGoalTest(payload) });
}
