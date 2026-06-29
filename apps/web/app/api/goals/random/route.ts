import { generateRandomGoal, isValidGoalDepth } from "@/lib/server/randomGoals";
import type { RandomGoalRequest } from "@/lib/types";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => null)) as
    | RandomGoalRequest
    | null;

  if (!isValidGoalDepth(payload?.depth)) {
    return NextResponse.json(
      { error: "Invalid random goal payload" },
      { status: 400 }
    );
  }

  return NextResponse.json({ goal: await generateRandomGoal(payload.depth) });
}
