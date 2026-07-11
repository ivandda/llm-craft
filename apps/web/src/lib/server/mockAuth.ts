import { GUEST_USERNAME_PREFIX } from "@/lib/guests";
import { createDefaultProfile } from "@/lib/profile";
import { query, transaction } from "@/lib/server/db";
import type { AuthUser, FeaturedAchievement, UserProfile } from "@/lib/types";
import { createHash, randomBytes, timingSafeEqual } from "crypto";

export const SESSION_COOKIE_NAME = "llm-craft.session";
export const SESSION_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365;
const SEEDED_ADMIN_USERNAME = "admin";
const SEEDED_ADMIN_PASSWORD = "admin";
const SEEDED_ADMIN_SALT = "llm-craft-seeded-admin";

export function sessionCookieOptions() {
  return {
    httpOnly: true,
    sameSite: "lax" as const,
    path: "/",
    maxAge: SESSION_COOKIE_MAX_AGE_SECONDS,
    secure: process.env.NODE_ENV === "production"
  };
}

type StoredUser = {
  id: string;
  username: string;
  displayName: string;
  passwordHash: string;
  passwordSalt: string;
};

type UserRow = {
  id: string;
  username: string;
  display_name: string;
  password_hash: string;
  password_salt: string;
};

type ProfileRow = {
  user_id: string;
  display_name: string;
  updated_at: Date;
};

type FeaturedAchievementRow = {
  element_id: string;
  name: string;
  emoji: string | null;
  featured_at: Date;
};

export async function registerUser(input: {
  username: string;
  password: string;
  displayName: string;
}): Promise<{ user: AuthUser; sessionId: string } | { error: string; status: number }> {
  await ensureAdminSeeded();
  const username = normalizeUsername(input.username);
  const displayName = input.displayName.trim();

  if (!username || input.password.length < 6 || !displayName) {
    return { error: "Invalid registration payload", status: 400 };
  }

  if (username.startsWith(GUEST_USERNAME_PREFIX)) {
    return { error: "That username prefix is reserved", status: 400 };
  }

  if (await findUserByUsername(username)) {
    return { error: "Username already exists", status: 409 };
  }

  const passwordSalt = randomBytes(16).toString("hex");
  const user: StoredUser = {
    id: randomBytes(12).toString("hex"),
    username,
    displayName,
    passwordHash: hashPassword(input.password, passwordSalt),
    passwordSalt
  };

  await transaction(async (client) => {
    await client.query(
      `
      INSERT INTO users (
        id, username, display_name, password_hash, password_salt
      )
      VALUES ($1, $2, $3, $4, $5)
      `,
      [user.id, user.username, user.displayName, user.passwordHash, user.passwordSalt]
    );
    await client.query(
      "INSERT INTO user_profiles (user_id, display_name) VALUES ($1, $2)",
      [user.id, displayName]
    );
  });

  return createSession(user);
}

export async function loginUser(input: {
  username: string;
  password: string;
}): Promise<{ user: AuthUser; sessionId: string } | { error: string; status: number }> {
  await ensureAdminSeeded();
  const user = await findUserByUsername(normalizeUsername(input.username));

  if (!user || !verifyPassword(input.password, user)) {
    return { error: "Invalid credentials", status: 401 };
  }

  return createSession(user);
}

export async function getUserBySession(sessionId: string | undefined): Promise<{
  user: AuthUser;
  profile: UserProfile;
} | null> {
  if (!sessionId) {
    return null;
  }

  const session = await query<UserRow>(
    `
    SELECT u.id, u.username, u.display_name, u.password_hash, u.password_salt
    FROM sessions s
    JOIN users u ON u.id = s.user_id
    WHERE s.id = $1
    `,
    [sessionId]
  );

  if (session.rows.length === 0) {
    return null;
  }

  const user = fromUserRow(session.rows[0]);

  return {
    user: toAuthUser(user),
    profile: await getProfile(user.id, user.displayName)
  };
}

export async function clearSession(sessionId: string | undefined): Promise<void> {
  if (sessionId) {
    await query("DELETE FROM sessions WHERE id = $1", [sessionId]);
  }
}

export async function updateProfile(
  sessionId: string | undefined,
  input: {
    displayName?: string;
    featuredAchievements?: FeaturedAchievement[];
  }
): Promise<{ user: AuthUser; profile: UserProfile } | { error: string; status: number }> {
  const session = await getUserBySession(sessionId);

  if (!session) {
    return { error: "Unauthorized", status: 401 };
  }

  const displayName = input.displayName?.trim() || session.user.displayName;
  const featuredAchievements = Array.isArray(input.featuredAchievements)
    ? input.featuredAchievements
    : session.profile.featuredAchievements;
  const nextAchievements = featuredAchievements.slice(0, 6);

  await transaction(async (client) => {
    await client.query(
      "UPDATE users SET display_name = $1, updated_at = now() WHERE id = $2",
      [displayName, session.user.id]
    );
    await client.query(
      `
      INSERT INTO user_profiles (user_id, display_name, updated_at)
      VALUES ($1, $2, now())
      ON CONFLICT (user_id) DO UPDATE SET
        display_name = EXCLUDED.display_name,
        updated_at = EXCLUDED.updated_at
      `,
      [session.user.id, displayName]
    );
    await client.query("DELETE FROM featured_achievements WHERE user_id = $1", [
      session.user.id
    ]);

    for (const [position, achievement] of nextAchievements.entries()) {
      await client.query(
        `
        INSERT INTO featured_achievements (
          user_id, position, element_id, name, emoji, featured_at
        )
        VALUES ($1, $2, $3, $4, $5, $6)
        `,
        [
          session.user.id,
          position,
          achievement.elementId,
          achievement.name,
          achievement.emoji ?? null,
          achievement.featuredAt
        ]
      );
    }
  });

  const user = {
    ...session.user,
    displayName
  };

  return {
    user,
    profile: await getProfile(user.id, displayName)
  };
}

async function createSession(
  user: StoredUser
): Promise<{ user: AuthUser; sessionId: string }> {
  const sessionId = randomBytes(24).toString("hex");
  await query("INSERT INTO sessions (id, user_id) VALUES ($1, $2)", [
    sessionId,
    user.id
  ]);

  return {
    user: toAuthUser(user),
    sessionId
  };
}

export async function createGuestSession(): Promise<{
  user: AuthUser;
  profile: UserProfile;
  sessionId: string;
}> {
  const suffix = randomBytes(4).toString("hex");
  const passwordSalt = randomBytes(16).toString("hex");
  const user: StoredUser = {
    id: randomBytes(12).toString("hex"),
    username: `${GUEST_USERNAME_PREFIX}${suffix}`,
    displayName: `Guest ${suffix}`,
    passwordHash: hashPassword(randomBytes(24).toString("hex"), passwordSalt),
    passwordSalt
  };

  await transaction(async (client) => {
    await client.query(
      `
      INSERT INTO users (
        id, username, display_name, password_hash, password_salt
      )
      VALUES ($1, $2, $3, $4, $5)
      `,
      [user.id, user.username, user.displayName, user.passwordHash, user.passwordSalt]
    );
    await client.query(
      "INSERT INTO user_profiles (user_id, display_name) VALUES ($1, $2)",
      [user.id, user.displayName]
    );
  });

  const session = await createSession(user);

  return {
    ...session,
    profile: await getProfile(user.id, user.displayName)
  };
}

declare global {
  var llmCraftAdminSeeded: Promise<void> | undefined;
}

function ensureAdminSeeded(): Promise<void> {
  globalThis.llmCraftAdminSeeded ??= seedAdminUser().catch((error) => {
    globalThis.llmCraftAdminSeeded = undefined;
    throw error;
  });

  return globalThis.llmCraftAdminSeeded;
}

async function seedAdminUser(): Promise<void> {
  const displayName = "Admin";
  const user: StoredUser = {
    id: "seed-admin",
    username: SEEDED_ADMIN_USERNAME,
    displayName,
    passwordHash: hashPassword(SEEDED_ADMIN_PASSWORD, SEEDED_ADMIN_SALT),
    passwordSalt: SEEDED_ADMIN_SALT
  };

  await transaction(async (client) => {
    await client.query(
      `
      INSERT INTO users (
        id, username, display_name, password_hash, password_salt
      )
      VALUES ($1, $2, $3, $4, $5)
      ON CONFLICT (username) DO NOTHING
      `,
      [user.id, user.username, user.displayName, user.passwordHash, user.passwordSalt]
    );
    await client.query(
      `
      INSERT INTO user_profiles (user_id, display_name)
      VALUES ($1, $2)
      ON CONFLICT (user_id) DO NOTHING
      `,
      [user.id, user.displayName]
    );
  });
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

async function findUserByUsername(username: string): Promise<StoredUser | null> {
  const result = await query<UserRow>(
    `
    SELECT id, username, display_name, password_hash, password_salt
    FROM users
    WHERE username = $1
    `,
    [username]
  );

  if (result.rows.length === 0) {
    return null;
  }

  return fromUserRow(result.rows[0]);
}

async function getProfile(
  userId: string,
  fallbackDisplayName: string
): Promise<UserProfile> {
  const profileResult = await query<ProfileRow>(
    "SELECT user_id, display_name, updated_at FROM user_profiles WHERE user_id = $1",
    [userId]
  );
  const fallbackProfile = createDefaultProfile(userId, fallbackDisplayName);
  const profile = profileResult.rows[0];
  const achievements = await query<FeaturedAchievementRow>(
    `
    SELECT element_id, name, emoji, featured_at
    FROM featured_achievements
    WHERE user_id = $1
    ORDER BY position ASC
    `,
    [userId]
  );

  return {
    userId,
    displayName: profile?.display_name ?? fallbackProfile.displayName,
    featuredAchievements: achievements.rows.map((achievement) => ({
      elementId: achievement.element_id,
      name: achievement.name,
      emoji: achievement.emoji ?? undefined,
      featuredAt: achievement.featured_at.toISOString()
    })),
    updatedAt: profile?.updated_at.toISOString() ?? fallbackProfile.updatedAt
  };
}

function fromUserRow(row: UserRow): StoredUser {
  return {
    id: row.id,
    username: row.username,
    displayName: row.display_name,
    passwordHash: row.password_hash,
    passwordSalt: row.password_salt
  };
}

function hashPassword(password: string, salt: string): string {
  return createHash("sha256").update(`${salt}:${password}`).digest("hex");
}

function verifyPassword(password: string, user: StoredUser): boolean {
  const expected = Buffer.from(user.passwordHash, "hex");
  const actual = Buffer.from(hashPassword(password, user.passwordSalt), "hex");

  return expected.length === actual.length && timingSafeEqual(expected, actual);
}
