import type {
  AuthUser,
  CombineRequest,
  CombineResponse,
  DpoPreferenceRequest,
  FeaturedAchievement,
  GoalPreset,
  LeaderboardEntry,
  RandomGoalRequest,
  UserProfile
} from "@/lib/types";

export type AuthSession = {
  user: AuthUser;
  profile: UserProfile;
};

export async function requestCombination(
  request: CombineRequest
): Promise<CombineResponse> {
  const response = await fetch("/api/combine", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(request)
  });

  if (!response.ok) {
    throw new Error("Combination request failed");
  }

  return response.json() as Promise<CombineResponse>;
}

export async function requestRandomGoal(
  request: RandomGoalRequest
): Promise<GoalPreset> {
  const payload = await requestJson<{ goal: GoalPreset }>(
    "/api/goals/random",
    request
  );

  return payload.goal;
}

export async function requestDpoPreference(
  request: DpoPreferenceRequest
): Promise<void> {
  await requestJson("/api/dpo/preferences", request);
}

export async function requestCurrentSession(): Promise<AuthSession | null> {
  const response = await fetch("/api/auth/me", {
    cache: "no-store"
  });

  if (response.status === 401) {
    return null;
  }

  if (!response.ok) {
    throw new Error("Session request failed");
  }

  return response.json() as Promise<AuthSession>;
}

export async function requestRegister(input: {
  username: string;
  password: string;
  displayName: string;
}): Promise<{ user: AuthUser }> {
  return requestJson("/api/auth/register", input);
}

export async function requestLogin(input: {
  username: string;
  password: string;
}): Promise<{ user: AuthUser }> {
  return requestJson("/api/auth/login", input);
}

export async function requestLogout(): Promise<void> {
  const response = await fetch("/api/auth/logout", {
    method: "POST"
  });

  if (!response.ok) {
    throw new Error("Logout request failed");
  }
}

export async function requestProfileUpdate(input: {
  displayName?: string;
  featuredAchievements?: FeaturedAchievement[];
}): Promise<AuthSession> {
  return requestJson("/api/profile", input, "PATCH");
}

export async function requestLeaderboard(
  goalId: string
): Promise<LeaderboardEntry[]> {
  const response = await fetch(`/api/leaderboard?goalId=${encodeURIComponent(goalId)}`, {
    cache: "no-store"
  });

  if (!response.ok) {
    throw new Error("Leaderboard request failed");
  }

  const payload = (await response.json()) as { entries: LeaderboardEntry[] };
  return payload.entries;
}

export async function requestLeaderboardSubmission(input: {
  goalId: string;
  goalTitle: string;
  combinationsUsed: number;
}): Promise<{
  entry: LeaderboardEntry;
  entries: LeaderboardEntry[];
}> {
  return requestJson("/api/leaderboard", input);
}

async function requestJson<TResponse>(
  path: string,
  body: unknown,
  method = "POST"
): Promise<TResponse> {
  const response = await fetch(path, {
    method,
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body)
  });

  if (!response.ok) {
    throw new Error("Request failed");
  }

  return response.json() as Promise<TResponse>;
}
