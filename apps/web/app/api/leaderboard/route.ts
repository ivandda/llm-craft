import { listLeaderboard, saveLeaderboardEntry } from "@/lib/server/mockLeaderboard";
import { getUserBySession, SESSION_COOKIE_NAME } from "@/lib/server/mockAuth";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const goalId = searchParams.get("goalId");

  if (!goalId) {
    return NextResponse.json({ error: "Missing goalId" }, { status: 400 });
  }

  return NextResponse.json({ entries: await listLeaderboard(goalId) });
}

export async function POST(request: Request) {
  const session = await getUserBySession(readCookie(request, SESSION_COOKIE_NAME));

  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const payload = (await request.json().catch(() => null)) as {
    goalId?: string;
    goalTitle?: string;
    combinationsUsed?: number;
  } | null;

  if (
    !payload?.goalId ||
    !payload.goalTitle ||
    typeof payload.combinationsUsed !== "number" ||
    payload.combinationsUsed < 1
  ) {
    return NextResponse.json(
      { error: "Invalid leaderboard payload" },
      { status: 400 }
    );
  }

  const entry = await saveLeaderboardEntry({
    user: session.user,
    goalId: payload.goalId,
    goalTitle: payload.goalTitle,
    combinationsUsed: payload.combinationsUsed
  });

  return NextResponse.json({
    entry,
    entries: await listLeaderboard(payload.goalId)
  });
}

function readCookie(request: Request, name: string): string | undefined {
  return request.headers
    .get("cookie")
    ?.split(";")
    .map((cookie) => cookie.trim())
    .find((cookie) => cookie.startsWith(`${name}=`))
    ?.slice(name.length + 1);
}
