import { loginUser, SESSION_COOKIE_NAME } from "@/lib/server/mockAuth";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => null)) as {
    username?: string;
    password?: string;
  } | null;

  if (!payload?.username || !payload.password) {
    return NextResponse.json(
      { error: "Invalid login payload" },
      { status: 400 }
    );
  }

  const result = loginUser({
    username: payload.username,
    password: payload.password
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
