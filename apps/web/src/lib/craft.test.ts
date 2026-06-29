import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  BASE_ELEMENTS,
  combineElements,
  createElementToken,
  makePairKey,
  mergeInventory,
  normalizeConcept
} from "@/lib/craft";
import { GOAL_PRESET, getInitialInventoryForMode } from "@/lib/gameModes";
import { createGameStorageKey } from "@/lib/gameStorage";
import {
  FEATURED_ACHIEVEMENT_LIMIT,
  selectFeaturedAchievements
} from "@/lib/profile";
import { loginUser } from "@/lib/server/mockAuth";
import {
  listLeaderboard,
  saveLeaderboardEntry
} from "@/lib/server/mockLeaderboard";

const dbMock = vi.hoisted(() => {
  type UserRow = {
    id: string;
    username: string;
    display_name: string;
    password_hash: string;
    password_salt: string;
  };
  type LeaderboardRow = {
    id: string;
    goal_id: string;
    goal_title: string;
    user_id: string;
    username: string;
    display_name: string;
    combinations_used: number;
    completed_at: Date;
  };

  const users = new Map<string, UserRow>();
  const sessions = new Map<string, string>();
  const leaderboard = new Map<string, LeaderboardRow>();

  async function query(sql: string, params: unknown[] = []) {
    const normalizedSql = sql.toLowerCase().replace(/\s+/g, " ");

    if (normalizedSql.includes("insert into users")) {
      const [id, username, displayName, passwordHash, passwordSalt] = params as string[];
      if (![...users.values()].some((user) => user.username === username)) {
        users.set(id, {
          id,
          username,
          display_name: displayName,
          password_hash: passwordHash,
          password_salt: passwordSalt
        });
      }
      return { rows: [] };
    }

    if (normalizedSql.includes("insert into user_profiles")) {
      return { rows: [] };
    }

    if (
      normalizedSql.includes("from users") &&
      normalizedSql.includes("where username")
    ) {
      const username = params[0] as string;
      return {
        rows: [...users.values()].filter((user) => user.username === username)
      };
    }

    if (normalizedSql.includes("insert into sessions")) {
      const [sessionId, userId] = params as string[];
      sessions.set(sessionId, userId);
      return { rows: [] };
    }

    if (normalizedSql.includes("insert into leaderboard_entries")) {
      const [
        id,
        goalId,
        goalTitle,
        userId,
        username,
        displayName,
        combinationsUsed
      ] = params as [string, string, string, string, string, string, number];
      const key = `${goalId}:${userId}`;
      const currentEntry = leaderboard.get(key);

      if (currentEntry && currentEntry.combinations_used <= combinationsUsed) {
        return { rows: [] };
      }

      const entry = {
        id,
        goal_id: goalId,
        goal_title: goalTitle,
        user_id: userId,
        username,
        display_name: displayName,
        combinations_used: combinationsUsed,
        completed_at: new Date("2026-06-29T00:00:00.000Z")
      };
      leaderboard.set(key, entry);
      return { rows: [entry] };
    }

    if (
      normalizedSql.includes("from leaderboard_entries") &&
      normalizedSql.includes("and user_id")
    ) {
      return { rows: [leaderboard.get(`${params[0]}:${params[1]}`)] };
    }

    if (normalizedSql.includes("from leaderboard_entries")) {
      const goalId = params[0] as string;
      return {
        rows: [...leaderboard.values()]
          .filter((entry) => entry.goal_id === goalId)
          .sort((left, right) => left.combinations_used - right.combinations_used)
      };
    }

    return { rows: [] };
  }

  return {
    query: vi.fn(query),
    reset() {
      users.clear();
      sessions.clear();
      leaderboard.clear();
    },
    transaction: vi.fn(async (callback) => callback({ query }))
  };
});

vi.mock("@/lib/server/db", () => ({
  query: dbMock.query,
  transaction: dbMock.transaction
}));

describe("craft utilities", () => {
  beforeEach(() => {
    dbMock.reset();
  });

  it("normalizes concept names", () => {
    expect(normalizeConcept("  Hot   Spring  ")).toBe("hot spring");
  });

  it("builds commutative pair keys", () => {
    expect(makePairKey("Water", "Fire")).toBe("fire+water");
    expect(makePairKey("fire", "water")).toBe("fire+water");
  });

  it("returns known recipe metadata for a seeded pair", () => {
    const response = combineElements({
      inputA: createElementToken("water"),
      inputB: createElementToken("fire"),
      inventory: BASE_ELEMENTS
    });

    expect(response.source).toBe("known_recipe");
    expect(response.result.name).toBe("steam");
    expect(response.knownOutputs?.map((element) => element.name)).toContain("mist");
  });

  it("returns a deterministic mock result for unknown pairs", () => {
    const request = {
      inputA: createElementToken("tree"),
      inputB: createElementToken("cloud"),
      inventory: BASE_ELEMENTS
    };

    const firstResponse = combineElements(request);
    const secondResponse = combineElements(request);

    expect(firstResponse.source).toBe("mock_model");
    expect(firstResponse.result).toEqual(secondResponse.result);
  });

  it("adds only new elements to inventory", () => {
    const inventory = mergeInventory(BASE_ELEMENTS, createElementToken("steam"));
    const unchangedInventory = mergeInventory(inventory, createElementToken("steam"));

    expect(inventory).toHaveLength(BASE_ELEMENTS.length + 1);
    expect(unchangedInventory).toHaveLength(inventory.length);
  });

  it("namespaces storage keys by normalized user and mode", () => {
    expect(createGameStorageKey(" User One ", "sandbox", "inventory")).toBe(
      "llm-craft.v2.user-one.sandbox.inventory"
    );
    expect(createGameStorageKey("User One", "goal", "inventory")).toBe(
      "llm-craft.v2.user-one.goal.inventory"
    );
  });

  it("limits featured achievements to available unique elements", () => {
    const inventory = [
      ...BASE_ELEMENTS,
      createElementToken("steam"),
      createElementToken("rain"),
      createElementToken("mud"),
      createElementToken("plant"),
      createElementToken("brick")
    ];
    const achievements = selectFeaturedAchievements(
      inventory,
      ["steam", "steam", "missing", "rain", "mud", "plant", "brick", "fire"],
      "2026-06-16T00:00:00.000Z"
    );

    expect(achievements).toHaveLength(FEATURED_ACHIEVEMENT_LIMIT);
    expect(achievements.map((achievement) => achievement.elementId)).toEqual([
      "steam",
      "rain",
      "mud",
      "plant",
      "brick",
      "fire"
    ]);
  });

  it("defines the static goal preset inventory and objective", () => {
    expect(GOAL_PRESET.mode).toBe("goal");
    expect(GOAL_PRESET.target.id).toBe("grass");
    expect(GOAL_PRESET.objective).toContain("grass");
    expect(getInitialInventoryForMode("goal").map((element) => element.id)).toEqual([
      "earth",
      "rain"
    ]);
    expect(getInitialInventoryForMode("sandbox")).toEqual(BASE_ELEMENTS);
  });

  it("seeds admin credentials for database auth", async () => {
    const result = await loginUser({
      username: "admin",
      password: "admin"
    });

    expect("user" in result ? result.user.username : null).toBe("admin");
  });

  it("keeps the best leaderboard score per user", async () => {
    const user = {
      id: "leaderboard-test-user",
      username: "runner",
      displayName: "Runner"
    };

    await saveLeaderboardEntry({
      user,
      goalId: "test-goal",
      goalTitle: "Test Goal",
      combinationsUsed: 5
    });
    await saveLeaderboardEntry({
      user,
      goalId: "test-goal",
      goalTitle: "Test Goal",
      combinationsUsed: 2
    });
    await saveLeaderboardEntry({
      user: {
        id: "leaderboard-test-other",
        username: "other",
        displayName: "Other"
      },
      goalId: "test-goal",
      goalTitle: "Test Goal",
      combinationsUsed: 4
    });

    expect(
      (await listLeaderboard("test-goal")).map((entry) => [
        entry.displayName,
        entry.combinationsUsed
      ])
    ).toEqual([
      ["Runner", 2],
      ["Other", 4]
    ]);
  });
});
