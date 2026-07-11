import {
  adminUnauthorizedResponse,
  isAdminDashboardConfigured,
  isAuthorizedAdminRequest
} from "@/lib/server/adminAuth";
import { NextResponse } from "next/server";

export default function proxy(request: Request) {
  if (!isAdminDashboardConfigured()) {
    return new Response("Admin dashboard is not configured", { status: 404 });
  }

  if (!isAuthorizedAdminRequest(request)) {
    return adminUnauthorizedResponse();
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/admin", "/admin/:path*", "/api/admin/:path*"]
};
