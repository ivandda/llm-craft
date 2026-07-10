import { query } from "@/lib/server/db";
import type { AuthUser, LeaderboardEntry } from "@/lib/types";

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

export async function listLeaderboard(goalId: string): Promise<LeaderboardEntry[]> {
  const result = await query<LeaderboardRow>(
    `
    SELECT id, goal_id, goal_title, user_id, username, display_name,
      combinations_used, completed_at
    FROM leaderboard_entries
    WHERE goal_id = $1
    ORDER BY combinations_used ASC, completed_at ASC
    LIMIT 20
    `,
    [goalId]
  );

  return result.rows.map(toLeaderboardEntry);
}

export async function saveLeaderboardEntry(input: {
  user: AuthUser;
  goalId: string;
  goalTitle: string;
  combinationsUsed: number;
}): Promise<LeaderboardEntry> {
  const entryKey = `${input.goalId}:${input.user.id}`;
  const result = await query<LeaderboardRow>(
    `
    INSERT INTO leaderboard_entries (
      id, goal_id, goal_title, user_id, username, display_name, combinations_used
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7)
    ON CONFLICT (goal_id, user_id) DO UPDATE SET
      goal_title = EXCLUDED.goal_title,
      username = EXCLUDED.username,
      display_name = EXCLUDED.display_name,
      combinations_used = EXCLUDED.combinations_used,
      completed_at = now()
    WHERE leaderboard_entries.combinations_used > EXCLUDED.combinations_used
    RETURNING id, goal_id, goal_title, user_id, username, display_name,
      combinations_used, completed_at
    `,
    [
      entryKey,
      input.goalId,
      input.goalTitle,
      input.user.id,
      input.user.username,
      input.user.displayName,
      input.combinationsUsed
    ]
  );

  if (result.rows[0]) {
    return toLeaderboardEntry(result.rows[0]);
  }

  const current = await query<LeaderboardRow>(
    `
    SELECT id, goal_id, goal_title, user_id, username, display_name,
      combinations_used, completed_at
    FROM leaderboard_entries
    WHERE goal_id = $1 AND user_id = $2
    `,
    [input.goalId, input.user.id]
  );

  return toLeaderboardEntry(current.rows[0]);
}

function toLeaderboardEntry(row: LeaderboardRow): LeaderboardEntry {
  return {
    id: row.id,
    goalId: row.goal_id,
    goalTitle: row.goal_title,
    userId: row.user_id,
    username: row.username,
    displayName: row.display_name,
    combinationsUsed: row.combinations_used,
    completedAt: row.completed_at.toISOString()
  };
}
