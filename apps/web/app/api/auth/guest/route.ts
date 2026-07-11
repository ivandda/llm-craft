import { createGuestSession, sessionCookieOptions, SESSION_COOKIE_NAME } from "@/lib/server/mockAuth";
import { getClientIp, getSessionFromRequest } from "@/lib/server/requestAuth";
import { enforceRateLimits, RATE_LIMITS } from "@/lib/server/rateLimit";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const existingSession = await getSessionFromRequest(request);

  if (existingSession) {
    return NextResponse.json(existingSession);
  }

  const rateLimited = await enforceRateLimits([
    {
      scope: "guest-session:ip",
      key: getClientIp(request),
      rule: RATE_LIMITS.guestSessionIp
    }
  ]);

  if (rateLimited) {
    return rateLimited;
  }

  const { user, profile, sessionId } = await createGuestSession();
  const response = NextResponse.json({ user, profile });
  response.cookies.set(SESSION_COOKIE_NAME, sessionId, sessionCookieOptions());

  return response;
}
