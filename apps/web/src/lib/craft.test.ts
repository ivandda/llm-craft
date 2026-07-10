import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  BASE_ELEMENTS,
  combineElements,
  createElementToken,
  makePairKey,
  mergeInventory,
  normalizeConcept
} from "@/lib/craft";
import { buildAgentPlaybackInventory } from "@/lib/agentPlayback";
import { requestCombination } from "@/lib/api";
import { selectDpoCandidates } from "@/lib/dpo";
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
import {
  buildRandomGoalCandidate,
  buildRandomGoalFromRecipes,
  generateRandomGoal,
  isValidGoalDepth
} from "@/lib/server/randomGoals";
import type { GoalPathStep } from "@/lib/server/randomGoals";
import { saveDpoPreference } from "@/lib/server/dpoPreferences";
import { combineElementsWithDataset } from "@/lib/server/dbRecipes";
import { runAgentGoalTest } from "@/lib/server/agentTestRunner";
import {
  getCombinerModelLabel,
  isKnownCombinerModel,
  isQwenCombinerModel
} from "@/lib/agentModels";
import { parseQwenConcept } from "@/lib/server/qwenCombiner";
import { POST as combinePost } from "../../app/api/combine/route";
import type { GoalPreset } from "@/lib/types";

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
  type RecipeRow = {
    input_a: string;
    input_b: string;
    output: string;
  };
  type StoredRecipeRow = {
    dataset_name: string;
    output: string;
    rationale: string | null;
    rank: number;
    raw_candidate?: unknown;
  };
  type DpoPreferenceRow = {
    id: string;
    user_id: string;
    mode: string;
    goal_id: string | null;
    input_a: unknown;
    input_b: unknown;
    shown_outputs: unknown[];
    selected_output: unknown;
    rejected_outputs: unknown[];
    inventory_snapshot: unknown[];
    combination_index: number;
    source: string;
    created_at: Date;
  };

  const users = new Map<string, UserRow>();
  const sessions = new Map<string, string>();
  const leaderboard = new Map<string, LeaderboardRow>();
  const recipeRows: RecipeRow[] = [];
  const storedRecipeRows: StoredRecipeRow[] = [];
  const dpoPreferenceEvents: DpoPreferenceRow[] = [];
  const generatedCandidates: unknown[] = [];
  const generatedDatasetImports: unknown[] = [];
  const generatedPairs: unknown[] = [];

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

    if (
      normalizedSql.includes("from recipe_pairs") &&
      normalizedSql.includes("join recipe_candidates")
    ) {
      if (normalizedSql.includes("select rp.dataset_name")) {
        return { rows: storedRecipeRows };
      }

      return { rows: recipeRows };
    }

    if (normalizedSql.includes("insert into dataset_imports")) {
      generatedDatasetImports.push(params);
      return { rows: [] };
    }

    if (normalizedSql.includes("insert into recipe_pairs")) {
      generatedPairs.push(params);
      return { rows: [{ pair_id: params[0] }] };
    }

    if (normalizedSql.includes("insert into recipe_candidates")) {
      generatedCandidates.push(params);
      return { rows: [] };
    }

    if (normalizedSql.includes("insert into dpo_preference_events")) {
      const [
        id,
        userId,
        mode,
        goalId,
        inputA,
        inputB,
        shownOutputs,
        selectedOutput,
        rejectedOutputs,
        inventorySnapshot,
        combinationIndex,
        source
      ] = params as [
        string,
        string,
        string,
        string | null,
        unknown,
        unknown,
        unknown[],
        unknown,
        unknown[],
        unknown[],
        number,
        string
      ];
      const row = {
        id,
        user_id: userId,
        mode,
        goal_id: goalId,
        input_a: JSON.parse(inputA as string),
        input_b: JSON.parse(inputB as string),
        shown_outputs: JSON.parse(shownOutputs as unknown as string),
        selected_output: JSON.parse(selectedOutput as string),
        rejected_outputs: JSON.parse(rejectedOutputs as unknown as string),
        inventory_snapshot: JSON.parse(inventorySnapshot as unknown as string),
        combination_index: combinationIndex,
        source,
        created_at: new Date("2026-06-29T00:00:00.000Z")
      };
      dpoPreferenceEvents.push(row);
      return { rows: [row] };
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
      recipeRows.splice(0);
      storedRecipeRows.splice(0);
      dpoPreferenceEvents.splice(0);
      generatedCandidates.splice(0);
      generatedDatasetImports.splice(0);
      generatedPairs.splice(0);
    },
    setRecipeRows(rows: RecipeRow[]) {
      recipeRows.splice(0, recipeRows.length, ...rows);
    },
    setStoredRecipeRows(rows: StoredRecipeRow[]) {
      storedRecipeRows.splice(0, storedRecipeRows.length, ...rows);
    },
    dpoPreferenceEvents,
    generatedCandidates,
    generatedDatasetImports,
    generatedPairs,
    transaction: vi.fn(async (callback) => callback({ query }))
  };
});

vi.mock("@/lib/server/db", () => ({
  query: dbMock.query,
  transaction: dbMock.transaction
}));

function replayGoalPath(
  initialInventory: Array<{ id: string }>,
  witnessPath: GoalPathStep[]
): string | null {
  const inventory = new Set(initialInventory.map((element) => element.id));
  let target: string | null = null;

  for (const step of witnessPath) {
    expect(inventory.has(step.inputA)).toBe(true);
    expect(inventory.has(step.inputB)).toBe(true);
    inventory.add(step.output);
    target = step.output;
  }

  return target;
}

function createAgentGoal(targetName: string, minDepth = 2): GoalPreset {
  return {
    id: `agent-test-${targetName}`,
    mode: "goal",
    title: `Agent ${targetName}`,
    description: "Agent test goal.",
    objective: `Discover ${targetName}.`,
    target: createElementToken(targetName),
    metadata: {
      difficulty: `depth-${minDepth}`,
      status: "generated",
      depth: minDepth,
      minDepth,
      strategy: "test",
      initialInventoryId: "classic"
    },
    initialInventory: BASE_ELEMENTS
  };
}

describe("craft utilities", () => {
  beforeEach(() => {
    dbMock.reset();
    vi.unstubAllGlobals();
    delete process.env.VERTEX_API_KEY;
    delete process.env.VERTEX_MODEL;
    delete process.env.QWEN_COMBINER_BASE_URL;
    delete process.env.QWEN_COMBINER_API_KEY;
    delete process.env.QWEN_COMBINER_MODEL;
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

  it("requires the server model for unknown local pairs", () => {
    const request = {
      inputA: createElementToken("tree"),
      inputB: createElementToken("cloud"),
      inventory: BASE_ELEMENTS
    };

    expect(() => combineElements(request)).toThrow(
      "Unknown combinations must be resolved by the server model"
    );
  });

  it("returns stored dataset recipes before generating", async () => {
    dbMock.setStoredRecipeRows([
      {
        dataset_name: "final-10k",
        output: "steam",
        rationale: "fire heats water into steam",
        rank: 1
      }
    ]);
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const response = await combineElementsWithDataset({
      inputA: createElementToken("water"),
      inputB: createElementToken("fire"),
      inventory: BASE_ELEMENTS
    });

    expect(response.source).toBe("known_recipe");
    expect(response.result.name).toBe("steam");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("keeps final-10k recipes deterministic when a model is selected", async () => {
    dbMock.setStoredRecipeRows([
      {
        dataset_name: "final-10k",
        output: "steam",
        rationale: "fire heats water into steam",
        rank: 1
      }
    ]);
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const response = await combineElementsWithDataset({
      inputA: createElementToken("water"),
      inputB: createElementToken("fire"),
      inventory: BASE_ELEMENTS,
      model: "gemini-2.5-pro"
    });

    expect(response.source).toBe("known_recipe");
    expect(response.model).toBeUndefined();
    expect(response.result.name).toBe("steam");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("rejects unknown combine models at the API boundary", async () => {
    const response = await combinePost(
      new Request("http://localhost/api/combine", {
        method: "POST",
        body: JSON.stringify({
          inputA: createElementToken("water"),
          inputB: createElementToken("fire"),
          inventory: BASE_ELEMENTS,
          model: "unknown-model"
        })
      })
    );

    expect(response.status).toBe(400);
  });

  it("recognizes Gemini and Qwen combiner models separately from agent models", () => {
    expect(isKnownCombinerModel("gemini-2.5-flash")).toBe(true);
    expect(isKnownCombinerModel("qwen3-4b-dpo-softce")).toBe(true);
    expect(isQwenCombinerModel("qwen3-4b-dpo-softce")).toBe(true);
    expect(isQwenCombinerModel("gemini-2.5-flash")).toBe(false);
    expect(getCombinerModelLabel("qwen3-4b-dpo-softce")).toBe("Qwen DPO SoftCE");
  });

  it("parses Qwen combiner outputs into clean concepts", () => {
    expect(parseQwenConcept("steam<|im_end|>")).toBe("steam");
    expect(parseQwenConcept("\"Concept: Storm   Cloud\"")).toBe("storm cloud");
    expect(parseQwenConcept("<think>hidden</think>\nResulting concept: Ice Cream")).toBe(
      "ice cream"
    );
  });

  it("generates and stores missing dataset recipes with Vertex", async () => {
    process.env.VERTEX_API_KEY = "test-key";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          candidates: [
            {
              content: {
                parts: [
                  {
                    text: JSON.stringify({
                      name: "storm cloud",
                      emoji: "⛈️",
                      rationale: "tree moisture and cloud air gather into a storm cloud"
                    })
                  }
                ]
              }
            }
          ]
        })
      }))
    );

    const response = await combineElementsWithDataset({
      inputA: createElementToken("tree"),
      inputB: createElementToken("cloud"),
      inventory: BASE_ELEMENTS
    });

    expect(response.source).toBe("model_generated");
    expect(response.result.name).toBe("storm cloud");
    expect(response.result.emoji).toBe("⛈️");
    expect(response.model).toBe("gemini-2.5-flash");
    expect(dbMock.generatedCandidates).toHaveLength(1);
  });

  it("passes the selected combine model to Vertex", async () => {
    process.env.VERTEX_API_KEY = "test-key";
    let capturedUrl = "";
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        capturedUrl = url;
        return {
          ok: true,
          json: async () => ({
            candidates: [
              {
                content: {
                  parts: [
                    {
                      text: JSON.stringify({
                        name: "signal mist",
                        emoji: "*",
                        rationale: "cloud and tree make a signal mist"
                      })
                    }
                  ]
                }
              }
            ]
          })
        };
      })
    );

    const response = await combineElementsWithDataset({
      inputA: createElementToken("tree"),
      inputB: createElementToken("cloud"),
      inventory: BASE_ELEMENTS,
      model: "gemini-2.5-flash-lite"
    });

    expect(response.model).toBe("gemini-2.5-flash-lite");
    expect(capturedUrl).toContain("gemini-2.5-flash-lite");
  });

  it("generates and stores missing dataset recipes with Qwen", async () => {
    process.env.QWEN_COMBINER_BASE_URL = "http://qwen.test/v1";
    process.env.QWEN_COMBINER_API_KEY = "qwen-key";
    process.env.QWEN_COMBINER_MODEL = "qwen3-4b-dpo-softce";
    let capturedUrl = "";
    let capturedBody: unknown;
    let capturedAuth = "";
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: RequestInit) => {
        capturedUrl = url;
        capturedBody = JSON.parse(String(init.body));
        capturedAuth = String((init.headers as Record<string, string>).Authorization);
        return {
          ok: true,
          json: async () => ({
            choices: [
              {
                message: {
                  content: "steam",
                  raw_content: "steam<|im_end|>"
                }
              }
            ]
          })
        };
      })
    );

    const response = await combineElementsWithDataset({
      inputA: createElementToken("fire"),
      inputB: createElementToken("water"),
      inventory: BASE_ELEMENTS,
      model: "qwen3-4b-dpo-softce"
    });

    expect(response.source).toBe("model_generated");
    expect(response.result.name).toBe("steam");
    expect(response.result.emoji).toBeUndefined();
    expect(response.model).toBe("qwen3-4b-dpo-softce");
    expect(response.rationale).toBeUndefined();
    expect(capturedUrl).toBe("http://qwen.test/v1/chat/completions");
    expect(capturedAuth).toBe("Bearer qwen-key");
    expect(capturedBody).toMatchObject({
      model: "qwen3-4b-dpo-softce",
      max_tokens: 16,
      temperature: 0,
      messages: [
        {
          role: "system",
          content: "You combine two concepts into one resulting concept."
        },
        {
          role: "user",
          content:
            "Given two concepts, combine them into one resulting concept.\n\nConcept A: fire\nConcept B: water\n\nReturn only the resulting concept."
        }
      ]
    });
    expect(dbMock.generatedDatasetImports[0]).toEqual([
      "web-generated-qwen3-4b-dpo-softce",
      "apps/web",
      JSON.stringify({
        source: "qwen-vm-web-combinator",
        model: "qwen3-4b-dpo-softce"
      })
    ]);
  });

  it("rejects invalid Qwen outputs before storing generated recipes", async () => {
    process.env.QWEN_COMBINER_BASE_URL = "http://qwen.test/v1";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          choices: [
            {
              message: {
                content: "fire"
              }
            }
          ]
        })
      }))
    );

    await expect(
      combineElementsWithDataset({
        inputA: createElementToken("fire"),
        inputB: createElementToken("water"),
        inventory: BASE_ELEMENTS,
        model: "qwen3-4b-dpo-softce"
      })
    ).rejects.toThrow("Qwen returned one of the input concepts");
    expect(dbMock.generatedCandidates).toHaveLength(0);
  });

  it("uses Pro combine generation settings without zero thinking budget", async () => {
    process.env.VERTEX_API_KEY = "test-key";
    const signal = new AbortController().signal;
    const timeoutSpy = vi.spyOn(AbortSignal, "timeout").mockReturnValue(signal);
    let capturedBody: unknown;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init: RequestInit) => {
        capturedBody = JSON.parse(String(init.body));
        return {
          ok: true,
          json: async () => ({
            candidates: [
              {
                content: {
                  parts: [
                    {
                      text: JSON.stringify({
                        name: "ancient grove",
                        emoji: "*",
                        rationale: "tree and cloud age into an ancient grove"
                      })
                    }
                  ]
                }
              }
            ]
          })
        };
      })
    );

    try {
      const response = await combineElementsWithDataset({
        inputA: createElementToken("tree"),
        inputB: createElementToken("cloud"),
        inventory: BASE_ELEMENTS,
        model: "gemini-2.5-pro"
      });

      expect(response.model).toBe("gemini-2.5-pro");
      expect(timeoutSpy).toHaveBeenCalledWith(45_000);
      expect(capturedBody).toMatchObject({
        generationConfig: {
          maxOutputTokens: 1024,
          responseMimeType: "application/json",
          responseSchema: {
            type: "object",
            properties: {
              name: { type: "string" },
              emoji: { type: "string" },
              rationale: { type: "string" }
            },
            required: ["name", "emoji", "rationale"]
          }
        }
      });
      expect(
        (capturedBody as { generationConfig: { thinkingConfig?: unknown } })
          .generationConfig.thinkingConfig
      ).toBeUndefined();
    } finally {
      timeoutSpy.mockRestore();
    }
  });

  it("stores generated recipes in a model-specific dataset", async () => {
    process.env.VERTEX_API_KEY = "test-key";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          candidates: [
            {
              content: {
                parts: [
                  {
                    text: JSON.stringify({
                      name: "storm archive",
                      emoji: "*",
                      rationale: "tree and cloud preserve storm traces"
                    })
                  }
                ]
              }
            }
          ]
        })
      }))
    );

    await combineElementsWithDataset({
      inputA: createElementToken("tree"),
      inputB: createElementToken("cloud"),
      inventory: BASE_ELEMENTS,
      model: "gemini-2.5-pro"
    });

    expect(dbMock.generatedDatasetImports[0]).toEqual([
      "web-generated-gemini-2.5-pro",
      "apps/web",
      JSON.stringify({
        source: "vertex-web-combinator",
        model: "gemini-2.5-pro"
      })
    ]);
    expect((dbMock.generatedPairs[0] as unknown[])[1]).toBe(
      "web-generated-gemini-2.5-pro"
    );
    expect(
      JSON.parse((dbMock.generatedCandidates[0] as unknown[])[4] as string).model
    ).toBe("gemini-2.5-pro");
  });

  it("serializes the selected model in combination requests", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => ({
      ok: true,
      json: async () => ({
        result: createElementToken("steam"),
        source: "known_recipe"
      })
    }));
    vi.stubGlobal("fetch", fetchMock);

    await requestCombination({
      inputA: createElementToken("water"),
      inputB: createElementToken("fire"),
      inventory: BASE_ELEMENTS,
      model: "gemini-2.5-pro"
    });

    const requestInit = fetchMock.mock.calls[0][1] as RequestInit;

    expect(JSON.parse(String(requestInit.body))).toMatchObject({
      model: "gemini-2.5-pro"
    });
  });

  it("uses a visible fallback emoji when Vertex omits one", async () => {
    process.env.VERTEX_API_KEY = "test-key";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          candidates: [
            {
              content: {
                parts: [
                  {
                    text: JSON.stringify({
                      name: "mystery signal",
                      emoji: null,
                      rationale: "cloud and tree form a strange signal"
                    })
                  }
                ]
              }
            }
          ]
        })
      }))
    );

    const response = await combineElementsWithDataset({
      inputA: createElementToken("tree"),
      inputB: createElementToken("cloud"),
      inventory: BASE_ELEMENTS
    });

    expect(response.source).toBe("model_generated");
    expect(response.result.name).toBe("mystery signal");
    expect(response.result.emoji).toBe("🤖");
  });

  it("fails unknown dataset recipes when Vertex is not configured", async () => {
    await expect(
      combineElementsWithDataset({
        inputA: createElementToken("tree"),
        inputB: createElementToken("cloud"),
        inventory: BASE_ELEMENTS
      })
    ).rejects.toThrow(
      "VERTEX_API_KEY or GOOGLE_APPLICATION_CREDENTIALS is not configured"
    );
  });

  it("adds only new elements to inventory", () => {
    const inventory = mergeInventory(BASE_ELEMENTS, createElementToken("steam"));
    const unchangedInventory = mergeInventory(inventory, createElementToken("steam"));

    expect(inventory).toHaveLength(BASE_ELEMENTS.length + 1);
    expect(unchangedInventory).toHaveLength(inventory.length);
  });

  it("reconstructs agent playback inventory from visible steps", () => {
    const steps = [
      {
        index: 1,
        inputA: createElementToken("water"),
        inputB: createElementToken("fire"),
        output: createElementToken("steam"),
        source: "known_recipe" as const
      },
      {
        index: 2,
        inputA: createElementToken("steam"),
        inputB: createElementToken("earth"),
        output: createElementToken("geyser"),
        source: "known_recipe" as const
      }
    ];

    expect(buildAgentPlaybackInventory(BASE_ELEMENTS, steps, 0).map((item) => item.id)).toEqual(
      BASE_ELEMENTS.map((item) => item.id)
    );
    expect(buildAgentPlaybackInventory(BASE_ELEMENTS, steps, 1).map((item) => item.id)).toContain(
      "steam"
    );
    expect(buildAgentPlaybackInventory(BASE_ELEMENTS, steps, 2).map((item) => item.id)).toContain(
      "geyser"
    );
  });

  it("does not duplicate existing elements during agent playback", () => {
    const inventory = buildAgentPlaybackInventory(
      BASE_ELEMENTS,
      [
        {
          index: 1,
          inputA: createElementToken("water"),
          inputB: createElementToken("fire"),
          output: createElementToken("fire"),
          source: "known_recipe"
        }
      ],
      1
    );

    expect(inventory.filter((element) => element.id === "fire")).toHaveLength(1);
    expect(inventory).toHaveLength(BASE_ELEMENTS.length);
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
    expect(GOAL_PRESET.metadata.depth).toBe(1);
    expect(getInitialInventoryForMode("goal").map((element) => element.id)).toEqual([
      "earth",
      "rain"
    ]);
    expect(getInitialInventoryForMode("sandbox")).toEqual(BASE_ELEMENTS);
  });

  it("generates a random goal with an exact playable depth", () => {
    const goal = buildRandomGoalFromRecipes(
      [
        { inputA: "water", inputB: "fire", output: "steam" },
        { inputA: "earth", inputB: "water", output: "mud" },
        { inputA: "steam", inputB: "mud", output: "geyser" }
      ],
      3,
      () => 0
    );

    expect(goal?.metadata.status).toBe("generated");
    expect(goal?.metadata.depth).toBe(3);
    expect(goal?.metadata.minDepth).toBe(3);
    expect(goal?.metadata.strategy).toBe("bfs-depth");
    expect(goal?.target.id).toBe("geyser");
    expect(goal?.initialInventory).toEqual(BASE_ELEMENTS);
  });

  it("builds a hidden witness path that reaches the goal", () => {
    const candidate = buildRandomGoalCandidate(
      [
        { inputA: "water", inputB: "fire", output: "steam" },
        { inputA: "earth", inputB: "water", output: "mud" },
        { inputA: "steam", inputB: "mud", output: "geyser" }
      ],
      3,
      () => 0
    );

    expect(candidate?.target.id).toBe("geyser");
    expect(candidate?.minDepth).toBe(3);
    expect(candidate?.witnessPath).toHaveLength(3);
    expect(replayGoalPath(candidate?.initialInventory ?? [], candidate?.witnessPath ?? [])).toBe(
      "geyser"
    );
    expect(candidate?.initialInventory.map((element) => element.id)).not.toContain(
      candidate?.target.id
    );
  });

  it("selects generated goals deterministically with a fixed seed", () => {
    const recipes = [
      { inputA: "water", inputB: "fire", output: "steam" },
      { inputA: "earth", inputB: "water", output: "mud" },
      { inputA: "earth", inputB: "fire", output: "lava" }
    ];
    const firstGoal = buildRandomGoalFromRecipes(recipes, 1, { seed: "fixed-seed" });
    const secondGoal = buildRandomGoalFromRecipes(recipes, 1, { seed: "fixed-seed" });

    expect(firstGoal?.id).toBe(secondGoal?.id);
    expect(firstGoal?.target.id).toBe(secondGoal?.target.id);
  });

  it("generates random goals with an explicit seed", async () => {
    dbMock.setRecipeRows([
      { input_a: "water", input_b: "fire", output: "steam" },
      { input_a: "earth", input_b: "water", output: "mud" }
    ]);

    const firstGoal = await generateRandomGoal(1, "seed-a");
    const secondGoal = await generateRandomGoal(1, "seed-a");

    expect(firstGoal.id).toBe(secondGoal.id);
    expect(firstGoal.metadata.seed).toBe("seed-a");
  });

  it("rejects invalid random goal depths", () => {
    expect(isValidGoalDepth(0)).toBe(false);
    expect(isValidGoalDepth(21)).toBe(false);
    expect(isValidGoalDepth(3.5)).toBe(false);
    expect(isValidGoalDepth(3)).toBe(true);
  });

  it("falls back when no dataset recipes can build a random goal", async () => {
    dbMock.setRecipeRows([]);

    await expect(generateRandomGoal(3)).resolves.toEqual(GOAL_PRESET);
  });

  it("selects DPO candidates only when multiple real outputs exist", () => {
    expect(selectDpoCandidates([createElementToken("steam")])).toEqual([]);

    const twoCandidates = selectDpoCandidates(
      [createElementToken("steam"), createElementToken("mist")],
      () => 0
    );
    const threeCandidates = selectDpoCandidates(
      [
        createElementToken("steam"),
        createElementToken("mist"),
        createElementToken("hot spring"),
        createElementToken("cloud")
      ],
      () => 0
    );

    expect(twoCandidates).toHaveLength(2);
    expect(threeCandidates).toHaveLength(3);
  });

  it("stores selected and rejected DPO outputs", async () => {
    const user = {
      id: "dpo-test-user",
      username: "dpo",
      displayName: "DPO"
    };
    const selectedOutput = createElementToken("steam");

    const event = await saveDpoPreference({
      user,
      preference: {
        mode: "goal",
        goalId: "random-depth-3-steam",
        inputA: createElementToken("fire"),
        inputB: createElementToken("water"),
        shownOutputs: [
          selectedOutput,
          createElementToken("mist"),
          createElementToken("hot spring")
        ],
        selectedOutput,
        inventorySnapshot: BASE_ELEMENTS,
        combinationIndex: 1,
        source: "known_recipe"
      }
    });

    expect(event.selectedOutput.id).toBe("steam");
    expect(event.rejectedOutputs.map((output) => output.id)).toEqual([
      "mist",
      "hot spring"
    ]);
    expect(dbMock.dpoPreferenceEvents).toHaveLength(1);
  });

  it("limits agent tests to 20 combinations", async () => {
    const combine = vi.fn(async () => ({
      result: createElementToken("not the target"),
      source: "known_recipe" as const,
      knownOutputs: [createElementToken("not the target")]
    }));

    const report = await runAgentGoalTest(
      { depth: 2 },
      {
        generateGoal: async () => createAgentGoal("target", 2),
        planAction: async () => ({
          inputA: "water",
          inputB: "fire",
          reason: "try a basic pair"
        }),
        combine
      }
    );

    expect(report.requestedDepth).toBe(2);
    expect(report.minDepth).toBe(2);
    expect(report.maxCombinations).toBe(20);
    expect(report.combinationsUsed).toBe(20);
    expect(report.stopReason).toBe("budget_exhausted");
    expect(combine).toHaveBeenCalledTimes(20);
  });

  it("stops an agent test when the target is reached", async () => {
    const report = await runAgentGoalTest(
      { depth: 2 },
      {
        generateGoal: async () => createAgentGoal("steam", 2),
        planAction: async () => ({
          inputA: "water",
          inputB: "fire"
        }),
        combine: async () => ({
          result: createElementToken("steam"),
          source: "known_recipe",
          knownOutputs: [createElementToken("steam")]
        })
      }
    );

    expect(report.success).toBe(true);
    expect(report.stopReason).toBe("target_reached");
    expect(report.combinationsUsed).toBe(1);
  });

  it("passes the selected model to the agent planner", async () => {
    const seenModels: string[] = [];
    const report = await runAgentGoalTest(
      { depth: 2, model: "gemini-2.5-flash-lite" },
      {
        generateGoal: async () => createAgentGoal("steam", 2),
        planAction: async (state) => {
          seenModels.push(state.model);
          return {
            inputA: "water",
            inputB: "fire"
          };
        },
        combine: async () => ({
          result: createElementToken("steam"),
          source: "known_recipe",
          knownOutputs: [createElementToken("steam")]
        })
      }
    );

    expect(report.model).toBe("gemini-2.5-flash-lite");
    expect(seenModels).toEqual(["gemini-2.5-flash-lite"]);
  });

  it("accepts Gemini 2.5 Pro as an agent test model", async () => {
    const report = await runAgentGoalTest(
      { depth: 2, model: "gemini-2.5-pro" },
      {
        generateGoal: async () => createAgentGoal("steam", 2),
        planAction: async () => ({
          inputA: "water",
          inputB: "fire",
          reason: "try the known steam recipe"
        }),
        combine: async () => ({
          result: createElementToken("steam"),
          source: "known_recipe",
          knownOutputs: [createElementToken("steam")]
        })
      }
    );

    expect(report.model).toBe("gemini-2.5-pro");
    expect(report.success).toBe(true);
  });

  it("sends a structured action schema to Vertex for agent planning", async () => {
    process.env.VERTEX_API_KEY = "test-key";
    const signal = new AbortController().signal;
    const timeoutSpy = vi.spyOn(AbortSignal, "timeout").mockReturnValue(signal);
    let capturedBody: unknown;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string, init: RequestInit) => {
        capturedBody = JSON.parse(String(init.body));
        return {
          ok: true,
          json: async () => ({
            candidates: [
              {
                content: {
                  parts: [
                    {
                      text: JSON.stringify({
                        inputA: "water",
                        inputB: "fire",
                        reason: "known recipe for steam"
                      })
                    }
                  ]
                }
              }
            ]
          })
        };
      })
    );

    try {
      const report = await runAgentGoalTest(
        { depth: 2, model: "gemini-2.5-pro" },
        {
          generateGoal: async () => createAgentGoal("steam", 2),
          combine: async () => ({
            result: createElementToken("steam"),
            source: "known_recipe",
            knownOutputs: [createElementToken("steam")]
          })
        }
      );

      expect(report.success).toBe(true);
      expect(timeoutSpy).toHaveBeenCalledWith(18_000);
      expect(capturedBody).toMatchObject({
        generationConfig: {
          responseMimeType: "application/json",
          responseSchema: {
            type: "object",
            properties: {
              inputA: { type: "string" },
              inputB: { type: "string" },
              reason: { type: "string" }
            },
            required: ["inputA", "inputB", "reason"]
          },
          thinkingConfig: { thinkingBudget: 1024 }
        }
      });
    } finally {
      timeoutSpy.mockRestore();
    }
  });

  it("rejects empty structured agent actions before combining", async () => {
    process.env.VERTEX_API_KEY = "test-key";
    const combine = vi.fn(async () => ({
      result: createElementToken("steam"),
      source: "known_recipe" as const,
      knownOutputs: [createElementToken("steam")]
    }));
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({
          candidates: [
            {
              content: {
                parts: [
                  {
                    text: JSON.stringify({
                      inputA: "",
                      inputB: "fire",
                      reason: "invalid empty input"
                    })
                  }
                ]
              }
            }
          ]
        })
      }))
    );

    const report = await runAgentGoalTest(
      { depth: 2, model: "gemini-2.5-pro" },
      {
        generateGoal: async () => createAgentGoal("steam", 2),
        combine
      }
    );

    expect(report.success).toBe(false);
    expect(report.stopReason).toBe("model_error");
    expect(report.combinationsUsed).toBe(0);
    expect(combine).not.toHaveBeenCalled();
  });

  it("rejects unknown agent test models", async () => {
    await expect(
      runAgentGoalTest(
        { depth: 2, model: "unknown-model" },
        {
          generateGoal: async () => createAgentGoal("steam", 2)
        }
      )
    ).rejects.toThrow("Invalid agent test model");
  });

  it("rejects agent actions outside the current inventory", async () => {
    const report = await runAgentGoalTest(
      { depth: 2 },
      {
        generateGoal: async () => createAgentGoal("steam", 2),
        planAction: async () => ({
          inputA: "water",
          inputB: "missing"
        })
      }
    );

    expect(report.success).toBe(false);
    expect(report.stopReason).toBe("invalid_action");
    expect(report.combinationsUsed).toBe(0);
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
