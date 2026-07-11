import { getUserBySession, SESSION_COOKIE_NAME } from "@/lib/server/mockAuth";
import type { AuthUser, UserProfile } from "@/lib/types";

export function readSessionCookie(request: Request): string | undefined {
  return request.headers
    .get("cookie")
    ?.split(";")
    .map((cookie) => cookie.trim())
    .find((cookie) => cookie.startsWith(`${SESSION_COOKIE_NAME}=`))
    ?.slice(SESSION_COOKIE_NAME.length + 1);
}

export async function getSessionFromRequest(request: Request): Promise<{
  user: AuthUser;
  profile: UserProfile;
} | null> {
  return getUserBySession(readSessionCookie(request));
}

export function getClientIp(request: Request): string {
  const forwarded = request.headers.get("x-forwarded-for");

  if (forwarded) {
    return forwarded.split(",")[0].trim() || "unknown";
  }

  return request.headers.get("x-real-ip")?.trim() || "unknown";
}
