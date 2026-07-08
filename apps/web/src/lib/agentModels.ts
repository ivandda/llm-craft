export type VertexModelOption = {
  id: string;
  label: string;
};

export const VERTEX_MODEL_OPTIONS: VertexModelOption[] = [
  { id: "gemini-2.5-flash", label: "Gemini 2.5 Flash" },
  { id: "gemini-2.5-pro", label: "Gemini 2.5 Pro" },
  { id: "gemini-2.5-flash-lite", label: "Gemini 2.5 Flash Lite" }
];

export const DEFAULT_VERTEX_MODEL = VERTEX_MODEL_OPTIONS[0].id;

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

export const isKnownAgentModel = isKnownVertexModel;
export const getAgentModelLabel = getVertexModelLabel;
