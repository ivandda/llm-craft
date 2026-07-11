import {
  clearSession,
  SESSION_COOKIE_NAME,
  sessionCookieOptions
} from "@/lib/server/mockAuth";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const cookiePrefix = `${SESSION_COOKIE_NAME}=`;
  const sessionId = request.headers
    .get("cookie")
    ?.split(";")
    .map((cookie) => cookie.trim())
    .find((cookie) => cookie.startsWith(cookiePrefix))
    ?.slice(cookiePrefix.length);

  await clearSession(sessionId);

  const response = NextResponse.json({ ok: true });
  // Clear with the same attributes used to set the cookie (secure/sameSite),
  // so the browser reliably matches and removes it.
  response.cookies.set(SESSION_COOKIE_NAME, "", {
    ...sessionCookieOptions(),
    maxAge: 0
  });

  return response;
}
