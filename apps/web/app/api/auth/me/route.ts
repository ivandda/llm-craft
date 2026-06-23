import { getUserBySession, SESSION_COOKIE_NAME } from "@/lib/server/mockAuth";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  const session = getUserBySession(readCookie(request, SESSION_COOKIE_NAME));

  if (!session) {
    return NextResponse.json({ user: null, profile: null }, { status: 401 });
  }

  return NextResponse.json(session);
}

function readCookie(request: Request, name: string): string | undefined {
  return request.headers
    .get("cookie")
    ?.split(";")
    .map((cookie) => cookie.trim())
    .find((cookie) => cookie.startsWith(`${name}=`))
    ?.slice(name.length + 1);
}
