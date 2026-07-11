import { listAgentRankings } from "@/lib/server/agentRuns";
import { isValidGoalDepth } from "@/lib/server/randomGoals";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const depth = Number(new URL(request.url).searchParams.get("depth"));

  if (!isValidGoalDepth(depth)) {
    return NextResponse.json({ error: "Invalid depth" }, { status: 400 });
  }

  return NextResponse.json({ rankings: await listAgentRankings(depth) });
}
