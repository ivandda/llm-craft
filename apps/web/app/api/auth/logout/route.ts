import { clearSession, SESSION_COOKIE_NAME } from "@/lib/server/mockAuth";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const sessionId = request.headers
    .get("cookie")
    ?.split(";")
    .map((cookie) => cookie.trim())
    .find((cookie) => cookie.startsWith(`${SESSION_COOKIE_NAME}=`))
    ?.split("=")[1];

  await clearSession(sessionId);

  const response = NextResponse.json({ ok: true });
  response.cookies.set(SESSION_COOKIE_NAME, "", {
    httpOnly: true,
    maxAge: 0,
    path: "/"
  });

  return response;
}
