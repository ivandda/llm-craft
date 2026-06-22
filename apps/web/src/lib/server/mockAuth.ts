import { createDefaultProfile } from "@/lib/profile";
import type { AuthUser, FeaturedAchievement, UserProfile } from "@/lib/types";
import { createHash, randomBytes, timingSafeEqual } from "crypto";

export const SESSION_COOKIE_NAME = "llm-craft.session";
const SEEDED_ADMIN_USERNAME = "admin";
const SEEDED_ADMIN_PASSWORD = "admin";
const SEEDED_ADMIN_SALT = "llm-craft-seeded-admin";

type StoredUser = {
  id: string;
  username: string;
  displayName: string;
  passwordHash: string;
  passwordSalt: string;
  profile: UserProfile;
};

type MockAuthStore = {
  usersByUsername: Map<string, StoredUser>;
  sessionsById: Map<string, string>;
};

declare global {
  var llmCraftMockAuthStore: MockAuthStore | undefined;
}

function getStore(): MockAuthStore {
  globalThis.llmCraftMockAuthStore ??= {
    usersByUsername: new Map(),
    sessionsById: new Map()
  };

  seedAdminUser(globalThis.llmCraftMockAuthStore);

  return globalThis.llmCraftMockAuthStore;
}

export function registerUser(input: {
  username: string;
  password: string;
  displayName: string;
}): { user: AuthUser; sessionId: string } | { error: string; status: number } {
  const username = normalizeUsername(input.username);
  const displayName = input.displayName.trim();

  if (!username || input.password.length < 6 || !displayName) {
    return { error: "Invalid registration payload", status: 400 };
  }

  const store = getStore();

  if (store.usersByUsername.has(username)) {
    return { error: "Username already exists", status: 409 };
  }

  const passwordSalt = randomBytes(16).toString("hex");
  const user: StoredUser = {
    id: randomBytes(12).toString("hex"),
    username,
    displayName,
    passwordHash: hashPassword(input.password, passwordSalt),
    passwordSalt,
    profile: createDefaultProfile("", displayName)
  };

  user.profile = createDefaultProfile(user.id, displayName);
  store.usersByUsername.set(username, user);

  return createSession(user);
}

export function loginUser(input: {
  username: string;
  password: string;
}): { user: AuthUser; sessionId: string } | { error: string; status: number } {
  const username = normalizeUsername(input.username);
  const user = getStore().usersByUsername.get(username);

  if (!user || !verifyPassword(input.password, user)) {
    return { error: "Invalid credentials", status: 401 };
  }

  return createSession(user);
}

export function getUserBySession(sessionId: string | undefined): {
  user: AuthUser;
  profile: UserProfile;
} | null {
  if (!sessionId) {
    return null;
  }

  const store = getStore();
  const userId = store.sessionsById.get(sessionId);

  if (!userId) {
    return null;
  }

  const user = [...store.usersByUsername.values()].find(
    (candidate) => candidate.id === userId
  );

  if (!user) {
    store.sessionsById.delete(sessionId);
    return null;
  }

  return {
    user: toAuthUser(user),
    profile: user.profile
  };
}

export function clearSession(sessionId: string | undefined): void {
  if (sessionId) {
    getStore().sessionsById.delete(sessionId);
  }
}

export function updateProfile(
  sessionId: string | undefined,
  input: {
    displayName?: string;
    featuredAchievements?: FeaturedAchievement[];
  }
): { user: AuthUser; profile: UserProfile } | { error: string; status: number } {
  const session = getUserBySession(sessionId);

  if (!session) {
    return { error: "Unauthorized", status: 401 };
  }

  const store = getStore();
  const storedUser = [...store.usersByUsername.values()].find(
    (candidate) => candidate.id === session.user.id
  );

  if (!storedUser) {
    return { error: "Unauthorized", status: 401 };
  }

  const displayName = input.displayName?.trim() || storedUser.displayName;
  const featuredAchievements = Array.isArray(input.featuredAchievements)
    ? input.featuredAchievements
    : storedUser.profile.featuredAchievements;

  storedUser.displayName = displayName;
  storedUser.profile = {
    userId: storedUser.id,
    displayName,
    featuredAchievements: featuredAchievements.slice(0, 6),
    updatedAt: new Date().toISOString()
  };

  return {
    user: toAuthUser(storedUser),
    profile: storedUser.profile
  };
}

function createSession(user: StoredUser): { user: AuthUser; sessionId: string } {
  const sessionId = randomBytes(24).toString("hex");
  getStore().sessionsById.set(sessionId, user.id);

  return {
    user: toAuthUser(user),
    sessionId
  };
}

function seedAdminUser(store: MockAuthStore): void {
  if (store.usersByUsername.has(SEEDED_ADMIN_USERNAME)) {
    return;
  }

  const displayName = "Admin";
  const user: StoredUser = {
    id: "seed-admin",
    username: SEEDED_ADMIN_USERNAME,
    displayName,
    passwordHash: hashPassword(SEEDED_ADMIN_PASSWORD, SEEDED_ADMIN_SALT),
    passwordSalt: SEEDED_ADMIN_SALT,
    profile: createDefaultProfile("seed-admin", displayName)
  };

  store.usersByUsername.set(user.username, user);
}

function toAuthUser(user: StoredUser): AuthUser {
  return {
    id: user.id,
    username: user.username,
    displayName: user.displayName
  };
}

function normalizeUsername(value: string): string {
  return value.trim().toLowerCase();
}

function hashPassword(password: string, salt: string): string {
  return createHash("sha256").update(`${salt}:${password}`).digest("hex");
}

function verifyPassword(password: string, user: StoredUser): boolean {
  const expected = Buffer.from(user.passwordHash, "hex");
  const actual = Buffer.from(hashPassword(password, user.passwordSalt), "hex");

  return expected.length === actual.length && timingSafeEqual(expected, actual);
}
