import { query } from "@/lib/server/db";
import { NextResponse } from "next/server";

export type RateLimitRule = {
  limit: number;
  windowSeconds: number;
};

export type RateLimitResult = {
  allowed: boolean;
  retryAfterSeconds: number;
};

export const RATE_LIMITS = {
  combineUser: { limit: 60, windowSeconds: 60 },
  combineIp: { limit: 300, windowSeconds: 60 },
  randomGoalUser: { limit: 15, windowSeconds: 60 },
  randomGoalIp: { limit: 60, windowSeconds: 60 },
  agentTestUser: { limit: 4, windowSeconds: 3600 },
  agentTestIp: { limit: 12, windowSeconds: 3600 },
  dpoCandidatesUser: { limit: 6, windowSeconds: 60 },
  dpoCandidatesIp: { limit: 40, windowSeconds: 60 },
  guestSessionIp: { limit: 120, windowSeconds: 3600 }
} as const satisfies Record<string, RateLimitRule>;

export async function checkRateLimit(
  scope: string,
  key: string,
  rule: RateLimitRule
): Promise<RateLimitResult> {
  const windowMs = rule.windowSeconds * 1000;
  const windowStartMs = Math.floor(Date.now() / windowMs) * windowMs;
  const windowStart = new Date(windowStartMs).toISOString();

  const result = await query<{ count: number }>(
    `
    INSERT INTO rate_limit_counters (scope, key, window_start, count)
    VALUES ($1, $2, $3, 1)
    ON CONFLICT (scope, key, window_start)
    DO UPDATE SET count = rate_limit_counters.count + 1
    RETURNING count
    `,
    [scope, key, windowStart]
  );
  const count = result.rows[0].count;

  if (count === 1) {
    await query(
      "DELETE FROM rate_limit_counters WHERE scope = $1 AND window_start < $2",
      [scope, windowStart]
    );
  }

  return {
    allowed: count <= rule.limit,
    retryAfterSeconds: Math.max(
      1,
      Math.ceil((windowStartMs + windowMs - Date.now()) / 1000)
    )
  };
}

export async function enforceRateLimits(
  checks: Array<{ scope: string; key: string; rule: RateLimitRule }>
): Promise<NextResponse | null> {
  for (const check of checks) {
    const result = await checkRateLimit(check.scope, check.key, check.rule);

    if (!result.allowed) {
      return NextResponse.json(
        { error: "Too many requests. Please slow down." },
        {
          status: 429,
          headers: { "Retry-After": String(result.retryAfterSeconds) }
        }
      );
    }
  }

  return null;
}
