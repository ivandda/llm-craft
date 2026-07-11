import { createElementToken, normalizeConcept } from "@/lib/craft";
import { getEmojiForConcept } from "@/lib/emoji";
import { QWEN_COMBINER_MODEL } from "@/lib/agentModels";
import type { CombineRequest, CombineResponse } from "@/lib/types";

const REQUEST_TIMEOUT_MS = 15_000;
const SYSTEM_PROMPT = "You combine two concepts into one resulting concept.";
const SPECIAL_TOKEN_PATTERN = /<\|[^|]+?\|>/g;
const THINK_PATTERN = /<think>.*?<\/think>/gis;
const PREFIX_PATTERN = /^(resulting\s+concept|concept)\s*:\s*/i;
const MAX_WORDS = 5;
const MAX_CHARS = 48;

type QwenChatResponse = {
  choices?: Array<{
    message?: {
      content?: unknown;
      raw_content?: unknown;
    };
  }>;
};

export class QwenConfigurationError extends Error {}
export class QwenGenerationError extends Error {}

export async function generateCombinationWithQwen(
  request: CombineRequest,
  model = getQwenModel()
): Promise<CombineResponse> {
  const rawPayload = await requestQwenGeneration(request, model);
  const name = parseQwenConcept(rawPayload);

  validateQwenConcept(name, request);

  const result = createElementToken(name, getEmojiForConcept(name));

  return {
    result,
    source: "model_generated",
    model,
    confidence: 0.65,
    knownOutputs: [result]
  };
}

export function getQwenModel(): string {
  return process.env.QWEN_COMBINER_MODEL?.trim() || QWEN_COMBINER_MODEL;
}

export function parseQwenConcept(rawPayload: string): string {
  const withoutThinking = rawPayload.replace(THINK_PATTERN, "");
  const text = withoutThinking.replace(SPECIAL_TOKEN_PATTERN, "").trim();

  for (const line of text.split(/\r?\n/)) {
    const cleaned = line.trim().replace(/^["'`]+|["'`]+$/g, "");
    const withoutPrefix = cleaned.replace(PREFIX_PATTERN, "").trim();

    if (withoutPrefix) {
      return normalizeConcept(withoutPrefix);
    }
  }

  return "";
}

async function requestQwenGeneration(
  request: CombineRequest,
  model: string
): Promise<string> {
  const response = await requestQwenJson(buildQwenRequest(request, model));
  return extractQwenText(response as QwenChatResponse);
}

async function requestQwenJson(body: unknown): Promise<unknown> {
  const baseUrl = process.env.QWEN_COMBINER_BASE_URL?.trim();

  if (!baseUrl) {
    throw new QwenConfigurationError("QWEN_COMBINER_BASE_URL is not configured");
  }

  const headers: Record<string, string> = {
    "Content-Type": "application/json"
  };
  const apiKey = process.env.QWEN_COMBINER_API_KEY?.trim();

  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`;
  }

  const response = await fetch(`${baseUrl.replace(/\/+$/, "")}/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)
  }).catch((error: unknown) => {
    throw new QwenGenerationError(getErrorMessage(error));
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new QwenGenerationError(
      `Qwen request failed with ${response.status}${detail ? `: ${detail}` : ""}`
    );
  }

  return response.json() as Promise<unknown>;
}

function buildQwenRequest(request: CombineRequest, model: string) {
  return {
    model,
    messages: [
      {
        role: "system",
        content: SYSTEM_PROMPT
      },
      {
        role: "user",
        content: buildUserPrompt(request)
      }
    ],
    max_tokens: 16,
    temperature: 0
  };
}

function buildUserPrompt(request: CombineRequest): string {
  return [
    "Given two concepts, combine them into one resulting concept.",
    "",
    `Concept A: ${normalizeConcept(request.inputA.name)}`,
    `Concept B: ${normalizeConcept(request.inputB.name)}`,
    "",
    "Return only the resulting concept."
  ].join("\n");
}

function extractQwenText(response: QwenChatResponse): string {
  const message = response.choices?.[0]?.message;
  const content = readString(message?.content);
  const rawContent = readString(message?.raw_content);
  const text = content || rawContent;

  if (!text) {
    throw new QwenGenerationError("Qwen returned no text");
  }

  return text;
}

function validateQwenConcept(name: string, request: CombineRequest): void {
  if (!name) {
    throw new QwenGenerationError("Qwen returned an empty concept");
  }

  const inputs = new Set([
    normalizeConcept(request.inputA.name),
    normalizeConcept(request.inputB.name)
  ]);

  if (inputs.has(name)) {
    throw new QwenGenerationError("Qwen returned one of the input concepts");
  }

  if (name.length > MAX_CHARS || name.split(" ").length > MAX_WORDS) {
    throw new QwenGenerationError("Qwen returned an invalid concept length");
  }
}

function readString(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Qwen request failed";
}
