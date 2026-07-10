import { createElementToken, normalizeConcept } from "@/lib/craft";
import { DEFAULT_VERTEX_MODEL } from "@/lib/agentModels";
import type { CombineRequest, CombineResponse } from "@/lib/types";
import { createSign } from "crypto";
import { readFile } from "fs/promises";

const VERTEX_ENDPOINT = "https://aiplatform.googleapis.com/v1";
const VERTEX_OAUTH_SCOPE = "https://www.googleapis.com/auth/cloud-platform";
const GCE_METADATA_TOKEN_URL =
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token";
const DEFAULT_VERTEX_LOCATION = "us-central1";
const REQUEST_TIMEOUT_MS = 15_000;
const PRO_REQUEST_TIMEOUT_MS = 45_000;
const COMBINER_RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    name: { type: "string" },
    emoji: { type: "string" },
    rationale: { type: "string" }
  },
  required: ["name", "emoji", "rationale"]
};
const FALLBACK_GENERATED_EMOJI = "🤖";

const SYSTEM_PROMPT = `
You generate new elements for a Little Alchemy style crafting game.

Rules:
- Combine the two input elements into one plausible, concrete result.
- Prefer short reusable nouns or noun phrases, 1 to 3 words.
- Do not return either input unchanged unless no better result exists.
- Do not use placeholder names, prefixes like "combined_", or abstract filler.
- Keep the rationale short and causal.
- Always return a visible emoji. If no specific emoji fits, use "🤖".
- Return only valid JSON with this exact shape:
  {"name":"element name","emoji":"single emoji","rationale":"short reason"}
`.trim();

type VertexPart = {
  text?: string;
};

type VertexResponse = {
  candidates?: Array<{
    content?: {
      parts?: VertexPart[];
    };
  }>;
};

type ModelPayload = {
  name?: unknown;
  emoji?: unknown;
  rationale?: unknown;
};

type ServiceAccountCredentials = {
  client_email?: string;
  private_key?: string;
  project_id?: string;
  token_uri?: string;
};

export class VertexConfigurationError extends Error {}
export class VertexGenerationError extends Error {}

export async function generateCombinationWithVertex(
  request: CombineRequest,
  model = getVertexModel()
): Promise<CombineResponse> {
  const rawPayload = await requestVertexGeneration(request, model);
  const parsedPayload = parseModelPayload(rawPayload);
  const result = createElementToken(parsedPayload.name, parsedPayload.emoji);

  return {
    result,
    source: "model_generated",
    model,
    confidence: 0.72,
    knownOutputs: [result],
    rationale: parsedPayload.rationale
  };
}

export function getVertexModel(): string {
  return process.env.VERTEX_MODEL?.trim() || DEFAULT_VERTEX_MODEL;
}

async function requestVertexGeneration(
  request: CombineRequest,
  model: string
): Promise<string> {
  return extractVertexText(
    (await requestVertexJson(
      buildVertexRequest(request, model),
      model,
      getRequestTimeoutMs(model)
    )) as VertexResponse
  );
}

export async function requestVertexJson(
  body: unknown,
  model = getVertexModel(),
  timeoutMs = REQUEST_TIMEOUT_MS
): Promise<unknown> {
  const apiKey = process.env.VERTEX_API_KEY?.trim();
  const response = apiKey
    ? await requestWithApiKey(body, apiKey, model, timeoutMs)
    : await requestWithGoogleCredentials(body, model, timeoutMs);

  return response.json() as Promise<unknown>;
}

async function requestWithApiKey(
  body: unknown,
  apiKey: string,
  model: string,
  timeoutMs: number
): Promise<Response> {
  const response = await fetch(buildApiKeyUrl(model, apiKey), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs)
  }).catch((error: unknown) => {
    throw new VertexGenerationError(getErrorMessage(error));
  });

  if (!response.ok) {
    throw new VertexGenerationError(`Vertex request failed with ${response.status}`);
  }

  return response;
}

async function requestWithGoogleCredentials(
  body: unknown,
  model: string,
  timeoutMs: number
): Promise<Response> {
  const credentialsPath = process.env.GOOGLE_APPLICATION_CREDENTIALS?.trim();

  if (credentialsPath) {
    return requestWithServiceAccount(body, model, timeoutMs);
  }

  if (usesGceMetadataCredentials()) {
    return requestWithGceMetadata(body, model, timeoutMs);
  }

  throw new VertexConfigurationError(
    "VERTEX_API_KEY, GOOGLE_APPLICATION_CREDENTIALS, or VERTEX_USE_GCE_METADATA is not configured"
  );
}

async function requestWithServiceAccount(
  body: unknown,
  model: string,
  timeoutMs: number
): Promise<Response> {
  const credentials = await loadServiceAccountCredentials();
  const accessToken = await createAccessToken(credentials);
  const response = await fetch(buildServiceAccountUrl(credentials, model), {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs)
  }).catch((error: unknown) => {
    throw new VertexGenerationError(getErrorMessage(error));
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new VertexGenerationError(
      `Vertex request failed with ${response.status}${detail ? `: ${detail}` : ""}`
    );
  }

  return response;
}

async function requestWithGceMetadata(
  body: unknown,
  model: string,
  timeoutMs: number
): Promise<Response> {
  const accessToken = await createMetadataAccessToken();
  const response = await fetch(buildServiceAccountUrl({}, model), {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs)
  }).catch((error: unknown) => {
    throw new VertexGenerationError(getErrorMessage(error));
  });

  if (!response.ok) {
    const detail = await response.text().catch(() => "");
    throw new VertexGenerationError(
      `Vertex request failed with ${response.status}${detail ? `: ${detail}` : ""}`
    );
  }

  return response;
}

function buildApiKeyUrl(model: string, apiKey: string): string {
  const modelPath = encodeURIComponent(model);
  const encodedKey = encodeURIComponent(apiKey);
  return `${VERTEX_ENDPOINT}/publishers/google/models/${modelPath}:generateContent?key=${encodedKey}`;
}

function buildServiceAccountUrl(
  credentials: ServiceAccountCredentials,
  modelName: string
): string {
  const projectId = getProjectId(credentials);
  const location = process.env.VERTEX_LOCATION?.trim() || DEFAULT_VERTEX_LOCATION;
  const model = encodeURIComponent(modelName);

  return `https://${location}-aiplatform.googleapis.com/v1/projects/${projectId}/locations/${location}/publishers/google/models/${model}:generateContent`;
}

function buildVertexRequest(request: CombineRequest, model: string) {
  return {
    systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
    contents: [
      {
        role: "user",
        parts: [{ text: buildUserPrompt(request) }]
      }
    ],
    generationConfig: buildGenerationConfig(model)
  };
}

function buildGenerationConfig(model: string) {
  const config: {
    temperature: number;
    maxOutputTokens: number;
    responseMimeType: string;
    responseSchema: typeof COMBINER_RESPONSE_SCHEMA;
    thinkingConfig?: { thinkingBudget: number };
  } = {
    temperature: 0.7,
    maxOutputTokens: model.includes("pro") ? 1024 : 256,
    responseMimeType: "application/json",
    responseSchema: COMBINER_RESPONSE_SCHEMA
  };

  if (!model.includes("pro")) {
    config.thinkingConfig = { thinkingBudget: 0 };
  }

  return config;
}

function buildUserPrompt(request: CombineRequest): string {
  const inventory = request.inventory.map((element) => element.name).join(", ");

  return [
    `Input A: ${normalizeConcept(request.inputA.name)}`,
    `Input B: ${normalizeConcept(request.inputB.name)}`,
    `Current inventory: ${inventory || "empty"}`
  ].join("\n");
}

async function loadServiceAccountCredentials(): Promise<ServiceAccountCredentials> {
  const credentialsPath = process.env.GOOGLE_APPLICATION_CREDENTIALS?.trim();

  if (!credentialsPath) {
    throw new VertexConfigurationError(
      "VERTEX_API_KEY or GOOGLE_APPLICATION_CREDENTIALS is not configured"
    );
  }

  const rawCredentials = await readFile(credentialsPath, "utf8").catch(() => {
    throw new VertexConfigurationError("GOOGLE_APPLICATION_CREDENTIALS is not readable");
  });
  const credentials = JSON.parse(rawCredentials) as ServiceAccountCredentials;

  if (!credentials.client_email || !credentials.private_key) {
    throw new VertexConfigurationError("Invalid service account credentials");
  }

  return credentials;
}

async function createAccessToken(
  credentials: ServiceAccountCredentials
): Promise<string> {
  const tokenUri = credentials.token_uri ?? "https://oauth2.googleapis.com/token";
  const assertion = createJwtAssertion(credentials, tokenUri);
  const response = await fetch(tokenUri, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer",
      assertion
    }),
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)
  });

  if (!response.ok) {
    throw new VertexGenerationError(`OAuth token request failed with ${response.status}`);
  }

  const payload = (await response.json()) as { access_token?: string };

  if (!payload.access_token) {
    throw new VertexGenerationError("OAuth token response did not include access_token");
  }

  return payload.access_token;
}

async function createMetadataAccessToken(): Promise<string> {
  const response = await fetch(GCE_METADATA_TOKEN_URL, {
    headers: { "Metadata-Flavor": "Google" },
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS)
  }).catch((error: unknown) => {
    throw new VertexGenerationError(getErrorMessage(error));
  });

  if (!response.ok) {
    throw new VertexGenerationError(
      `Metadata token request failed with ${response.status}`
    );
  }

  const payload = (await response.json()) as { access_token?: string };

  if (!payload.access_token) {
    throw new VertexGenerationError("Metadata token response did not include access_token");
  }

  return payload.access_token;
}

function createJwtAssertion(
  credentials: ServiceAccountCredentials,
  tokenUri: string
): string {
  const now = Math.floor(Date.now() / 1000);
  const header = encodeBase64Url({ alg: "RS256", typ: "JWT" });
  const claim = encodeBase64Url({
    iss: credentials.client_email,
    scope: VERTEX_OAUTH_SCOPE,
    aud: tokenUri,
    iat: now,
    exp: now + 3600
  });
  const signingInput = `${header}.${claim}`;
  const signature = createSign("RSA-SHA256")
    .update(signingInput)
    .sign(credentials.private_key ?? "", "base64url");

  return `${signingInput}.${signature}`;
}

function encodeBase64Url(value: unknown): string {
  return Buffer.from(JSON.stringify(value)).toString("base64url");
}

function getProjectId(credentials: ServiceAccountCredentials): string {
  const projectId =
    process.env.GOOGLE_CLOUD_PROJECT?.trim() || credentials.project_id?.trim();

  if (!projectId) {
    throw new VertexConfigurationError("GOOGLE_CLOUD_PROJECT is not configured");
  }

  return projectId;
}

function usesGceMetadataCredentials(): boolean {
  return ["1", "true", "yes"].includes(
    process.env.VERTEX_USE_GCE_METADATA?.trim().toLowerCase() ?? ""
  );
}

function extractVertexText(response: VertexResponse): string {
  const text = response.candidates?.[0]?.content?.parts
    ?.map((part) => part.text ?? "")
    .join("")
    .trim();

  if (!text) {
    throw new VertexGenerationError("Vertex returned no text");
  }

  return text;
}

function parseModelPayload(rawPayload: string): {
  name: string;
  emoji?: string;
  rationale: string;
} {
  const payload = parseJsonPayload(extractJson(rawPayload));
  const name = typeof payload.name === "string" ? normalizeConcept(payload.name) : "";
  const rationale =
    typeof payload.rationale === "string" ? payload.rationale.trim() : "";

  if (!name || !rationale) {
    throw new VertexGenerationError("Vertex returned an invalid element");
  }

  return {
    name,
    emoji: normalizeEmoji(payload.emoji),
    rationale
  };
}

function parseJsonPayload(value: string): ModelPayload {
  try {
    return JSON.parse(value) as ModelPayload;
  } catch {
    throw new VertexGenerationError("Vertex returned malformed JSON");
  }
}

function extractJson(value: string): string {
  const firstBrace = value.indexOf("{");
  const lastBrace = value.lastIndexOf("}");

  if (firstBrace < 0 || lastBrace <= firstBrace) {
    throw new VertexGenerationError("Vertex did not return JSON");
  }

  return value.slice(firstBrace, lastBrace + 1);
}

function normalizeEmoji(value: unknown): string | undefined {
  if (typeof value !== "string") {
    return FALLBACK_GENERATED_EMOJI;
  }

  const emoji = value.trim();
  return emoji.length > 0 && emoji.length <= 8 ? emoji : FALLBACK_GENERATED_EMOJI;
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Vertex request failed";
}

function getRequestTimeoutMs(model: string): number {
  return model.includes("pro") ? PRO_REQUEST_TIMEOUT_MS : REQUEST_TIMEOUT_MS;
}
