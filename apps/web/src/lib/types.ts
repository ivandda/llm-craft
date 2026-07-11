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
  seed?: string;
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

export type AgentRunSummary = {
  id: string;
  model: string;
  success: boolean;
  stopReason: string;
  combinationsUsed: number;
  createdAt: string;
  report: AgentTestReport;
};

export type AgentRankingEntry = {
  model: string;
  runs: number;
  wins: number;
  winRate: number;
  avgCombinations: number | null;
};

export type ArenaDailyModelStat = {
  /** UTC calendar day, YYYY-MM-DD — matches the daily goal seed. */
  day: string;
  model: string;
  runs: number;
  wins: number;
  winRate: number;
  /** Average combinations used, successful runs only. */
  avgCombinations: number | null;
};

export type ArenaOverallEntry = {
  model: string;
  daysPlayed: number;
  runs: number;
  wins: number;
  /** Mean of per-day win rates over the days the model actually ran. */
  score: number;
  avgCombinations: number | null;
};

export type ArenaStats = {
  /** Days with runs at the requested depth, newest first. */
  days: string[];
  /** Per-day, per-model stats at the requested depth. */
  daily: ArenaDailyModelStat[];
  /** Cross-depth leaderboard, per-day-averaged so sparse days weigh equally. */
  overall: ArenaOverallEntry[];
  /** All-time ranking at the requested depth. */
  rankings: AgentRankingEntry[];
};

export type DpoCandidate = ElementToken & {
  generatedBy?: string;
};

export type DpoCandidatesRequest = {
  inputA: ElementToken;
  inputB: ElementToken;
  inventory: ElementToken[];
  model?: string;
};

export type DpoCandidatesResponse = {
  candidates: DpoCandidate[];
  source: CombineSource;
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
