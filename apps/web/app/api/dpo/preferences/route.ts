import { getUserBySession, SESSION_COOKIE_NAME } from "@/lib/server/mockAuth";
import { saveDpoPreference } from "@/lib/server/dpoPreferences";
import type { DpoPreferenceRequest, ElementToken } from "@/lib/types";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const session = await getUserBySession(readCookie(request, SESSION_COOKIE_NAME));

  if (!session) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const payload = (await request.json().catch(() => null)) as
    | DpoPreferenceRequest
    | null;

  if (!isDpoPreferenceRequest(payload)) {
    return NextResponse.json(
      { error: "Invalid DPO preference payload" },
      { status: 400 }
    );
  }

  const event = await saveDpoPreference({
    user: session.user,
    preference: payload
  });

  return NextResponse.json({ event });
}

function isDpoPreferenceRequest(
  value: DpoPreferenceRequest | null
): value is DpoPreferenceRequest {
  return Boolean(
    value &&
      (value.mode === "sandbox" || value.mode === "goal") &&
      isElementToken(value.inputA) &&
      isElementToken(value.inputB) &&
      Array.isArray(value.shownOutputs) &&
      value.shownOutputs.length >= 2 &&
      value.shownOutputs.every(isElementToken) &&
      isElementToken(value.selectedOutput) &&
      value.shownOutputs.some((output) => output.id === value.selectedOutput.id) &&
      Array.isArray(value.inventorySnapshot) &&
      value.inventorySnapshot.every(isElementToken) &&
      Number.isInteger(value.combinationIndex) &&
      value.combinationIndex > 0 &&
      (value.source === "known_recipe" || value.source === "model_generated")
  );
}

function isElementToken(value: ElementToken | null | undefined): value is ElementToken {
  return Boolean(value?.id && value.name);
}

function readCookie(request: Request, name: string): string | undefined {
  return request.headers
    .get("cookie")
    ?.split(";")
    .map((cookie) => cookie.trim())
    .find((cookie) => cookie.startsWith(`${name}=`))
    ?.slice(name.length + 1);
}
