import { mergeInventory, normalizeConcept } from "@/lib/craft";
import { DEFAULT_AGENT_MODEL, isKnownAgentModel } from "@/lib/agentModels";
import { combineElementsWithDataset } from "@/lib/server/dbRecipes";
import {
  generateRandomGoal,
  isValidGoalDepth
} from "@/lib/server/randomGoals";
import { requestVertexJson } from "@/lib/server/vertexCombiner";
import type {
  AgentTestReport,
  AgentTestRequest,
  AgentTestStep,
  CombineRequest,
  CombineResponse,
  ElementToken,
  GoalPreset
} from "@/lib/types";
import { randomUUID } from "crypto";

const MAX_AGENT_COMBINATIONS = 20;
const PRO_AGENT_REQUEST_TIMEOUT_MS = 18_000;
const PRO_AGENT_THINKING_BUDGET = 1024;
const AGENT_ACTION_RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    inputA: { type: "string" },
    inputB: { type: "string" },
    reason: { type: "string" }
  },
  required: ["inputA", "inputB", "reason"]
};

type AgentAction = {
  inputA: string;
  inputB: string;
  reason?: string;
};

type AgentState = {
  model: string;
  goal: GoalPreset;
  inventory: ElementToken[];
  steps: AgentTestStep[];
  remainingCombinations: number;
};

type AgentTestDependencies = {
  generateGoal?: (depth: number) => Promise<GoalPreset>;
  planAction?: (state: AgentState) => Promise<AgentAction>;
  combine?: (request: CombineRequest) => Promise<CombineResponse>;
};

type VertexResponse = {
  candidates?: Array<{
    content?: {
      parts?: Array<{ text?: string }>;
    };
  }>;
};

export async function runAgentGoalTest(
  request: AgentTestRequest,
  dependencies: AgentTestDependencies = {}
): Promise<AgentTestReport> {
  if (!isValidGoalDepth(request.depth)) {
    throw new Error("Invalid agent test depth");
  }
  const model = request.model ?? DEFAULT_AGENT_MODEL;

  if (!isKnownAgentModel(model)) {
    throw new Error("Invalid agent test model");
  }

  const generateGoal =
    dependencies.generateGoal ??
    ((depth: number) => generateRandomGoal(depth, request.seed ?? createAgentTestSeed(depth)));
  const planAction = dependencies.planAction ?? planActionWithVertex;
  const combine = dependencies.combine ?? combineElementsWithDataset;
  const goal = await generateGoal(request.depth);
  const minDepth = goal.metadata.minDepth ?? goal.metadata.depth;
  const maxCombinations = MAX_AGENT_COMBINATIONS;
  let inventory = [...goal.initialInventory];
  const steps: AgentTestStep[] = [];

  for (let index = 1; index <= maxCombinations; index += 1) {
    const action = await readAgentAction(planAction, {
      model,
      goal,
      inventory,
      steps,
      remainingCombinations: maxCombinations - steps.length
    });

    if (!action.ok) {
      return buildReport(request, goal, minDepth, maxCombinations, inventory, steps, {
        stopReason: "model_error",
        errorMessage: action.errorMessage
      });
    }

    const inputA = findInventoryElement(inventory, action.value.inputA);
    const inputB = findInventoryElement(inventory, action.value.inputB);

    if (!inputA || !inputB) {
      return buildReport(request, goal, minDepth, maxCombinations, inventory, steps, {
        stopReason: "invalid_action",
        errorMessage: "Agent selected an element outside the current inventory."
      });
    }

    const response = await readCombination(combine, { inputA, inputB, inventory });

    if (!response.ok) {
      return buildReport(request, goal, minDepth, maxCombinations, inventory, steps, {
        stopReason: "combine_error",
        errorMessage: response.errorMessage
      });
    }

    const step = {
      index,
      inputA,
      inputB,
      output: response.value.result,
      source: response.value.source,
      rationale: response.value.rationale,
      agentReason: action.value.reason
    };
    steps.push(step);
    inventory = mergeInventory(inventory, response.value.result);

    if (response.value.result.id === goal.target.id) {
      return buildReport(request, goal, minDepth, maxCombinations, inventory, steps, {
        stopReason: "target_reached"
      });
    }
  }

  return buildReport(request, goal, minDepth, maxCombinations, inventory, steps, {
    stopReason: "budget_exhausted"
  });
}

async function planActionWithVertex(state: AgentState): Promise<AgentAction> {
  return parseAgentAction(
    extractVertexText(
      (await requestVertexJson(
        buildVertexRequest(state),
        state.model,
        getAgentRequestTimeoutMs(state.model)
      )) as VertexResponse
    )
  );
}

function buildVertexRequest(state: AgentState) {
  return {
    systemInstruction: {
      parts: [
        {
          text: [
            "You are an agent playing an Infinite Craft style goal test.",
            "You may only choose two elements already present in inventory.",
            "The app will call the combine_elements tool for your chosen pair.",
            "Do not invent the output yourself.",
            "Avoid repeating a pair that already appears in previous tool calls.",
            "Return only JSON: {\"inputA\":\"name\",\"inputB\":\"name\",\"reason\":\"short reason\"}."
          ].join(" ")
        }
      ]
    },
    contents: [{ role: "user", parts: [{ text: buildAgentPrompt(state) }] }],
    generationConfig: buildAgentGenerationConfig(state.model)
  };
}

function buildAgentGenerationConfig(model: string) {
  const config: {
    temperature: number;
    maxOutputTokens: number;
    responseMimeType: string;
    responseSchema: typeof AGENT_ACTION_RESPONSE_SCHEMA;
    thinkingConfig?: { thinkingBudget: number };
  } = {
    temperature: 0.2,
    maxOutputTokens: model.includes("pro") ? 2048 : 160,
    responseMimeType: "application/json",
    responseSchema: AGENT_ACTION_RESPONSE_SCHEMA
  };

  config.thinkingConfig = {
    thinkingBudget: model.includes("pro") ? PRO_AGENT_THINKING_BUDGET : 0
  };

  return config;
}

function buildAgentPrompt(state: AgentState): string {
  const inventory = state.inventory.map((element) => element.name).join(", ");
  const history = state.steps
    .map((step) => `${step.inputA.name} + ${step.inputB.name} -> ${step.output.name}`)
    .join("\n");

  return [
    `Target: ${state.goal.target.name}`,
    `Agent model: ${state.model}`,
    `Remaining tool calls: ${state.remainingCombinations}`,
    `Inventory: ${inventory}`,
    `Previous tool calls:\n${history || "none"}`,
    "Choose the next pair most likely to reach the target.",
    "Return the JSON object directly, with no markdown, label, or preamble."
  ].join("\n");
}

function findInventoryElement(
  inventory: ElementToken[],
  selectedName: string
): ElementToken | null {
  const normalizedName = normalizeConcept(selectedName);

  return (
    inventory.find((element) => {
      return element.id === normalizedName || normalizeConcept(element.name) === normalizedName;
    }) ?? null
  );
}

async function readAgentAction(
  planAction: (state: AgentState) => Promise<AgentAction>,
  state: AgentState
): Promise<{ ok: true; value: AgentAction } | { ok: false; errorMessage: string }> {
  try {
    return { ok: true, value: await planAction(state) };
  } catch (error) {
    return { ok: false, errorMessage: getErrorMessage(error) };
  }
}

async function readCombination(
  combine: (request: CombineRequest) => Promise<CombineResponse>,
  request: CombineRequest
): Promise<{ ok: true; value: CombineResponse } | { ok: false; errorMessage: string }> {
  try {
    return { ok: true, value: await combine(request) };
  } catch (error) {
    return { ok: false, errorMessage: getErrorMessage(error) };
  }
}

function buildReport(
  request: AgentTestRequest,
  goal: GoalPreset,
  minDepth: number,
  maxCombinations: number,
  finalInventory: ElementToken[],
  steps: AgentTestStep[],
  result: {
    stopReason: AgentTestReport["stopReason"];
    errorMessage?: string;
  }
): AgentTestReport {
  return {
    model: request.model ?? DEFAULT_AGENT_MODEL,
    goal,
    requestedDepth: request.depth,
    minDepth,
    maxCombinations,
    combinationsUsed: steps.length,
    success: result.stopReason === "target_reached",
    stopReason: result.stopReason,
    steps,
    finalInventory,
    errorMessage: result.errorMessage
  };
}

function extractVertexText(response: VertexResponse): string {
  const text = response.candidates?.[0]?.content?.parts
    ?.map((part) => part.text ?? "")
    .join("")
    .trim();

  if (!text) {
    throw new Error("Vertex agent returned no text");
  }

  return text;
}

function parseAgentAction(rawPayload: string): AgentAction {
  const payload = JSON.parse(extractJson(rawPayload)) as Partial<AgentAction>;

  if (
    typeof payload.inputA !== "string" ||
    typeof payload.inputB !== "string" ||
    payload.inputA.trim().length === 0 ||
    payload.inputB.trim().length === 0
  ) {
    throw new Error("Vertex agent returned an invalid action");
  }

  return {
    inputA: payload.inputA.trim(),
    inputB: payload.inputB.trim(),
    reason: typeof payload.reason === "string" ? payload.reason.trim() : undefined
  };
}

function extractJson(value: string): string {
  const firstBrace = value.indexOf("{");
  const lastBrace = value.lastIndexOf("}");

  if (firstBrace < 0 || lastBrace <= firstBrace) {
    throw new Error(`Vertex agent did not return JSON: ${createErrorExcerpt(value)}`);
  }

  return value.slice(firstBrace, lastBrace + 1);
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Agent test failed";
}

function getAgentRequestTimeoutMs(model: string): number | undefined {
  return model.includes("pro") ? PRO_AGENT_REQUEST_TIMEOUT_MS : undefined;
}

// Daily seed: every model attempting the same depth on the same day faces the
// same goal, so arena rankings compare like with like. randomUUID stays
// available for explicit per-run seeds passed via the request.
function createAgentTestSeed(depth: number): string {
  return `agent-arena:${new Date().toISOString().slice(0, 10)}:${depth}`;
}

export function createRandomAgentSeed(depth: number): string {
  return `agent-test:${depth}:${randomUUID()}`;
}

function createErrorExcerpt(value: string): string {
  const excerpt = value.replace(/\s+/g, " ").trim().slice(0, 160);
  return excerpt || "empty response";
}
