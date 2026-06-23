import { registerUser, SESSION_COOKIE_NAME } from "@/lib/server/mockAuth";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => null)) as {
    username?: string;
    password?: string;
    displayName?: string;
  } | null;

  if (!payload?.username || !payload.password || !payload.displayName) {
    return NextResponse.json(
      { error: "Invalid registration payload" },
      { status: 400 }
    );
  }

  const result = registerUser({
    username: payload.username,
    password: payload.password,
    displayName: payload.displayName
  });

  if ("error" in result) {
    return NextResponse.json({ error: result.error }, { status: result.status });
  }

  const response = NextResponse.json({ user: result.user });
  response.cookies.set(SESSION_COOKIE_NAME, result.sessionId, {
    httpOnly: true,
    sameSite: "lax",
    path: "/"
  });

  return response;
}
