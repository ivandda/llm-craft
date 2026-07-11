import {
  getStoredArenaGoal,
  listAgentRunsForDay
} from "@/lib/server/agentRuns";
import { createAgentTestSeed } from "@/lib/server/agentTestRunner";
import { generateRandomGoal, isValidGoalDepth } from "@/lib/server/randomGoals";
import { NextResponse } from "next/server";

const DAY_PATTERN = /^\d{4}-\d{2}-\d{2}$/;

function currentUtcDay(): string {
  return new Date().toISOString().slice(0, 10);
}

// Public, read-only arena feed: a day's challenge + that day's runs at a
// depth. Defaults to today; past days replay the goal stored with the runs.
export async function GET(request: Request) {
  const url = new URL(request.url);
  const depth = Number(url.searchParams.get("depth"));

  if (!isValidGoalDepth(depth)) {
    return NextResponse.json({ error: "Invalid depth" }, { status: 400 });
  }

  const today = currentUtcDay();
  const day = url.searchParams.get("day") ?? today;

  if (!DAY_PATTERN.test(day) || day > today) {
    return NextResponse.json({ error: "Invalid day" }, { status: 400 });
  }

  const [goal, runs] = await Promise.all([
    day === today
      ? generateRandomGoal(depth, createAgentTestSeed(depth))
      : getStoredArenaGoal(depth, day),
    listAgentRunsForDay(depth, day)
  ]);

  return NextResponse.json({ day, goal, runs });
}
