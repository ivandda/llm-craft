import { SESSION_COOKIE_NAME, updateProfile } from "@/lib/server/mockAuth";
import type { FeaturedAchievement } from "@/lib/types";
import { NextResponse } from "next/server";

export async function PATCH(request: Request) {
  const payload = (await request.json().catch(() => null)) as {
    displayName?: string;
    featuredAchievements?: FeaturedAchievement[];
  } | null;

  if (!payload) {
    return NextResponse.json({ error: "Invalid profile payload" }, { status: 400 });
  }

  const result = updateProfile(readCookie(request, SESSION_COOKIE_NAME), payload);

  if ("error" in result) {
    return NextResponse.json({ error: result.error }, { status: result.status });
  }

  return NextResponse.json(result);
}

function readCookie(request: Request, name: string): string | undefined {
  return request.headers
    .get("cookie")
    ?.split(";")
    .map((cookie) => cookie.trim())
    .find((cookie) => cookie.startsWith(`${name}=`))
    ?.slice(name.length + 1);
}
