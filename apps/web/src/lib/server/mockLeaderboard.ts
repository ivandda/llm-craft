import type { AuthUser, LeaderboardEntry } from "@/lib/types";

type LeaderboardStore = {
  entriesByGoalAndUser: Map<string, LeaderboardEntry>;
};

declare global {
  var llmCraftMockLeaderboardStore: LeaderboardStore | undefined;
}

function getStore(): LeaderboardStore {
  globalThis.llmCraftMockLeaderboardStore ??= {
    entriesByGoalAndUser: new Map()
  };

  return globalThis.llmCraftMockLeaderboardStore;
}

export function listLeaderboard(goalId: string): LeaderboardEntry[] {
  return [...getStore().entriesByGoalAndUser.values()]
    .filter((entry) => entry.goalId === goalId)
    .sort((left, right) => {
      if (left.combinationsUsed !== right.combinationsUsed) {
        return left.combinationsUsed - right.combinationsUsed;
      }

      return left.completedAt.localeCompare(right.completedAt);
    })
    .slice(0, 20);
}

export function saveLeaderboardEntry(input: {
  user: AuthUser;
  goalId: string;
  goalTitle: string;
  combinationsUsed: number;
}): LeaderboardEntry {
  const store = getStore();
  const entryKey = `${input.goalId}:${input.user.id}`;
  const currentEntry = store.entriesByGoalAndUser.get(entryKey);

  if (currentEntry && currentEntry.combinationsUsed <= input.combinationsUsed) {
    return currentEntry;
  }

  const entry: LeaderboardEntry = {
    id: entryKey,
    goalId: input.goalId,
    goalTitle: input.goalTitle,
    userId: input.user.id,
    username: input.user.username,
    displayName: input.user.displayName,
    combinationsUsed: input.combinationsUsed,
    completedAt: new Date().toISOString()
  };

  store.entriesByGoalAndUser.set(entryKey, entry);
  return entry;
}
