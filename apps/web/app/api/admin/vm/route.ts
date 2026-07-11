import {
  adminUnauthorizedResponse,
  isAuthorizedAdminRequest
} from "@/lib/server/adminAuth";
import { GcpVmError, getVmStatus, startVm, stopVm } from "@/lib/server/gcpVm";
import { NextResponse } from "next/server";

export async function GET(request: Request) {
  // proxy.ts already gates /api/admin/*; this is defense in depth.
  if (!isAuthorizedAdminRequest(request)) {
    return adminUnauthorizedResponse();
  }

  try {
    return NextResponse.json({ vm: await getVmStatus() });
  } catch (error) {
    return vmErrorResponse(error);
  }
}

export async function POST(request: Request) {
  if (!isAuthorizedAdminRequest(request)) {
    return adminUnauthorizedResponse();
  }

  const payload = (await request.json().catch(() => null)) as
    | { action?: string }
    | null;

  if (payload?.action !== "start" && payload?.action !== "stop") {
    return NextResponse.json(
      { error: 'Invalid payload: expected {"action":"start"|"stop"}' },
      { status: 400 }
    );
  }

  try {
    if (payload.action === "start") {
      await startVm();
    } else {
      await stopVm();
    }

    return NextResponse.json({ vm: await getVmStatus() });
  } catch (error) {
    return vmErrorResponse(error);
  }
}

function vmErrorResponse(error: unknown) {
  const message =
    error instanceof GcpVmError || error instanceof Error
      ? error.message
      : "Unexpected VM control error";

  return NextResponse.json({ error: message }, { status: 502 });
}
