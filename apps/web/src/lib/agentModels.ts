export type VertexModelOption = {
  id: string;
  label: string;
};

export type CombinerModelOption = VertexModelOption & {
  provider: "vertex" | "qwen";
};

export const VERTEX_MODEL_OPTIONS: VertexModelOption[] = [
  { id: "gemini-2.5-flash", label: "Gemini 2.5 Flash" },
  { id: "gemini-2.5-pro", label: "Gemini 2.5 Pro" },
  { id: "gemini-2.5-flash-lite", label: "Gemini 2.5 Flash Lite" }
];

export const DEFAULT_VERTEX_MODEL = VERTEX_MODEL_OPTIONS[0].id;
export const QWEN_COMBINER_MODEL = "qwen3-4b-dpo-softce";
export const DEFAULT_COMBINER_MODEL = DEFAULT_VERTEX_MODEL;

export const COMBINER_MODEL_OPTIONS: CombinerModelOption[] = [
  ...VERTEX_MODEL_OPTIONS.map((model) => ({
    ...model,
    provider: "vertex" as const
  })),
  {
    id: QWEN_COMBINER_MODEL,
    label: "Qwen DPO SoftCE",
    provider: "qwen"
  }
];

export const AGENT_MODEL_OPTIONS = VERTEX_MODEL_OPTIONS;
export const DEFAULT_AGENT_MODEL = DEFAULT_VERTEX_MODEL;

export function isKnownVertexModel(model: unknown): model is string {
  return (
    typeof model === "string" &&
    VERTEX_MODEL_OPTIONS.some((option) => option.id === model)
  );
}

export function getVertexModelLabel(model: string): string {
  return VERTEX_MODEL_OPTIONS.find((option) => option.id === model)?.label ?? model;
}

export function isKnownCombinerModel(model: unknown): model is string {
  return (
    typeof model === "string" &&
    COMBINER_MODEL_OPTIONS.some((option) => option.id === model)
  );
}

export function isQwenCombinerModel(model: string): boolean {
  return (
    COMBINER_MODEL_OPTIONS.find((option) => option.id === model)?.provider === "qwen"
  );
}

export function getCombinerModelLabel(model: string): string {
  return COMBINER_MODEL_OPTIONS.find((option) => option.id === model)?.label ?? model;
}

export const isKnownAgentModel = isKnownVertexModel;
export const getAgentModelLabel = getVertexModelLabel;
