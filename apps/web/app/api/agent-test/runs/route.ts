import { listRecentAgentRuns } from "@/lib/server/agentRuns";
import { createAgentTestSeed } from "@/lib/server/agentTestRunner";
import { generateRandomGoal, isValidGoalDepth } from "@/lib/server/randomGoals";
import { NextResponse } from "next/server";

// Public, read-only arena feed: today's challenge + recent runs at a depth.
export async function GET(request: Request) {
  const depth = Number(new URL(request.url).searchParams.get("depth"));

  if (!isValidGoalDepth(depth)) {
    return NextResponse.json({ error: "Invalid depth" }, { status: 400 });
  }

  const [goal, runs] = await Promise.all([
    generateRandomGoal(depth, createAgentTestSeed(depth)),
    listRecentAgentRuns(depth)
  ]);

  return NextResponse.json({ goal, runs });
}
