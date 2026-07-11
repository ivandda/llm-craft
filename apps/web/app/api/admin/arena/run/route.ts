import { isKnownAgentModel } from "@/lib/agentModels";
import {
  adminUnauthorizedResponse,
  isAuthorizedAdminRequest
} from "@/lib/server/adminAuth";
import { saveAgentRun } from "@/lib/server/agentRuns";
import { runAgentGoalTest } from "@/lib/server/agentTestRunner";
import { isValidGoalDepth } from "@/lib/server/randomGoals";
import type { AgentTestRequest } from "@/lib/types";
import { NextResponse } from "next/server";

// Arena runs cost real model calls, so only admins can trigger them.
// proxy.ts already gates /api/admin/*; this check is defense in depth.
export async function POST(request: Request) {
  if (!isAuthorizedAdminRequest(request)) {
    return adminUnauthorizedResponse();
  }

  const payload = (await request.json().catch(() => null)) as
    | AgentTestRequest
    | null;

  if (
    !isValidGoalDepth(payload?.depth) ||
    (payload?.model !== undefined && !isKnownAgentModel(payload.model))
  ) {
    return NextResponse.json(
      { error: "Invalid arena run payload" },
      { status: 400 }
    );
  }

  const report = await runAgentGoalTest(payload);

  try {
    await saveAgentRun(report, null);
  } catch (error) {
    console.error("Failed to persist arena run", error);
  }

  return NextResponse.json({ report });
}
