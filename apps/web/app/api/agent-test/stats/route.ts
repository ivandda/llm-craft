import { aggregateOverallStats } from "@/lib/arenaStats";
import {
  listAgentRankings,
  listArenaDailyStats
} from "@/lib/server/agentRuns";
import { isValidGoalDepth } from "@/lib/server/randomGoals";
import { NextResponse } from "next/server";
import type { ArenaStats } from "@/lib/types";

// Public arena statistics: per-day history at the requested depth plus a
// cross-depth overall leaderboard (per-day-averaged win rates).
export async function GET(request: Request) {
  const depth = Number(new URL(request.url).searchParams.get("depth"));

  if (!isValidGoalDepth(depth)) {
    return NextResponse.json({ error: "Invalid depth" }, { status: 400 });
  }

  const [daily, allDepths, rankings] = await Promise.all([
    listArenaDailyStats(depth),
    listArenaDailyStats(),
    listAgentRankings(depth)
  ]);

  const stats: ArenaStats = {
    days: [...new Set(daily.map((stat) => stat.day))],
    daily,
    overall: aggregateOverallStats(allDepths),
    rankings
  };

  return NextResponse.json(stats);
}
