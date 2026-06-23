import { describe, expect, it } from "vitest";
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

describe("craft utilities", () => {
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
    expect(GOAL_PRESET.target.id).toBe("plant");
    expect(GOAL_PRESET.objective).toContain("plant");
    expect(getInitialInventoryForMode("goal").map((element) => element.id)).toEqual([
      "earth",
      "rain"
    ]);
    expect(getInitialInventoryForMode("sandbox")).toEqual(BASE_ELEMENTS);
  });

  it("seeds admin credentials for mock auth", () => {
    const result = loginUser({
      username: "admin",
      password: "admin"
    });

    expect("user" in result ? result.user.username : null).toBe("admin");
  });

  it("keeps the best leaderboard score per user", () => {
    const user = {
      id: "leaderboard-test-user",
      username: "runner",
      displayName: "Runner"
    };

    saveLeaderboardEntry({
      user,
      goalId: "test-goal",
      goalTitle: "Test Goal",
      combinationsUsed: 5
    });
    saveLeaderboardEntry({
      user,
      goalId: "test-goal",
      goalTitle: "Test Goal",
      combinationsUsed: 2
    });
    saveLeaderboardEntry({
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
      listLeaderboard("test-goal").map((entry) => [
        entry.displayName,
        entry.combinationsUsed
      ])
    ).toEqual([
      ["Runner", 2],
      ["Other", 4]
    ]);
  });
});
