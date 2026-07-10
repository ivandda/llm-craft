export type CombineSource = "known_recipe" | "model_generated";

export type ElementToken = {
  id: string;
  name: string;
  emoji?: string;
  discoveredAt?: string;
};

export type CombineRequest = {
  inputA: ElementToken;
  inputB: ElementToken;
  inventory: ElementToken[];
  model?: string;
};

export type CombineResponse = {
  result: ElementToken;
  source: CombineSource;
  model?: string;
  confidence?: number;
  knownOutputs?: ElementToken[];
  rationale?: string;
};

export type RecipeHistoryItem = {
  id: string;
  inputA: ElementToken;
  inputB: ElementToken;
  output: ElementToken;
  source: CombineSource;
  createdAt: string;
};

export type GameMode = "sandbox" | "goal" | "agent-test";

export type FeaturedAchievement = {
  elementId: string;
  name: string;
  emoji?: string;
  featuredAt: string;
};

export type UserProfile = {
  userId: string;
  displayName: string;
  featuredAchievements: FeaturedAchievement[];
  updatedAt: string;
};

export type AuthUser = {
  id: string;
  username: string;
  displayName: string;
};

export type GoalPreset = {
  id: string;
  mode: "goal";
  title: string;
  description: string;
  objective: string;
  target: ElementToken;
  metadata: {
    difficulty: string;
    status: "mock" | "generated";
    depth: number;
    minDepth?: number;
    seed?: string;
    strategy?: string;
    initialInventoryId?: string;
  };
  initialInventory: ElementToken[];
};

export type RandomGoalRequest = {
  depth: number;
};

export type AgentTestRequest = {
  depth: number;
  model?: string;
  seed?: string;
};

export type AgentTestStep = {
  index: number;
  inputA: ElementToken;
  inputB: ElementToken;
  output: ElementToken;
  source: CombineSource;
  rationale?: string;
  agentReason?: string;
};

export type AgentTestStopReason =
  | "target_reached"
  | "budget_exhausted"
  | "invalid_action"
  | "model_error"
  | "combine_error";

export type AgentTestReport = {
  model: string;
  goal: GoalPreset;
  requestedDepth: number;
  minDepth: number;
  maxCombinations: number;
  combinationsUsed: number;
  success: boolean;
  stopReason: AgentTestStopReason;
  steps: AgentTestStep[];
  finalInventory: ElementToken[];
  errorMessage?: string;
};

export type DpoPreferenceRequest = {
  mode: GameMode;
  goalId?: string;
  inputA: ElementToken;
  inputB: ElementToken;
  shownOutputs: ElementToken[];
  selectedOutput: ElementToken;
  inventorySnapshot: ElementToken[];
  combinationIndex: number;
  source: CombineSource;
};

export type GameSnapshot = {
  inventory: ElementToken[];
  history: RecipeHistoryItem[];
};

export type LeaderboardEntry = {
  id: string;
  goalId: string;
  goalTitle: string;
  userId: string;
  username: string;
  displayName: string;
  combinationsUsed: number;
  completedAt: string;
};
