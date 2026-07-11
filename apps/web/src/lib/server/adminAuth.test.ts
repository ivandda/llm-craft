import {
  isAdminDashboardConfigured,
  isAuthorizedAdminRequest
} from "@/lib/server/adminAuth";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

function basicAuthRequest(user: string, password: string): Request {
  const encoded = Buffer.from(`${user}:${password}`).toString("base64");

  return new Request("http://localhost/admin", {
    headers: { Authorization: `Basic ${encoded}` }
  });
}

describe("adminAuth", () => {
  beforeEach(() => {
    process.env.ADMIN_DASH_USER = "ivan";
    process.env.ADMIN_DASH_PASSWORD = "s3cret";
  });

  afterEach(() => {
    delete process.env.ADMIN_DASH_USER;
    delete process.env.ADMIN_DASH_PASSWORD;
  });

  it("is disabled when credentials are not configured", () => {
    delete process.env.ADMIN_DASH_PASSWORD;

    expect(isAdminDashboardConfigured()).toBe(false);
    expect(isAuthorizedAdminRequest(basicAuthRequest("ivan", "s3cret"))).toBe(false);
  });

  it("accepts matching basic auth credentials", () => {
    expect(isAuthorizedAdminRequest(basicAuthRequest("ivan", "s3cret"))).toBe(true);
  });

  it("rejects wrong password, missing header and malformed values", () => {
    expect(isAuthorizedAdminRequest(basicAuthRequest("ivan", "wrong"))).toBe(false);
    expect(isAuthorizedAdminRequest(new Request("http://localhost/admin"))).toBe(false);
    expect(
      isAuthorizedAdminRequest(
        new Request("http://localhost/admin", {
          headers: { Authorization: "Basic not-base64!!" }
        })
      )
    ).toBe(false);
  });

  it("allows a colon inside the password", () => {
    process.env.ADMIN_DASH_PASSWORD = "pa:ss";

    expect(isAuthorizedAdminRequest(basicAuthRequest("ivan", "pa:ss"))).toBe(true);
  });
});
