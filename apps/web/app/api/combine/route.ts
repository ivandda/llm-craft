import { combineElements } from "@/lib/craft";
import type { CombineRequest } from "@/lib/types";
import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const payload = (await request.json().catch(() => null)) as CombineRequest | null;

  if (!isCombineRequest(payload)) {
    return NextResponse.json(
      { error: "Invalid combine request" },
      { status: 400 }
    );
  }

  return NextResponse.json(combineElements(payload));
}

function isCombineRequest(value: CombineRequest | null): value is CombineRequest {
  return Boolean(
    value?.inputA?.id &&
      value.inputA.name &&
      value?.inputB?.id &&
      value.inputB.name &&
      Array.isArray(value.inventory)
  );
}
