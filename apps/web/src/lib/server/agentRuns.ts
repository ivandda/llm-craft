import { query } from "@/lib/server/db";
import type {
  AgentRankingEntry,
  AgentRunSummary,
  AgentTestReport,
  ArenaDailyModelStat,
  GoalPreset
} from "@/lib/types";
import { randomBytes } from "crypto";

type RankingRow = {
  model: string;
  runs: string;
  wins: string;
  avg_combinations: string | null;
};

// The daily goal seed is derived from the UTC calendar date, so all arena
// "day" groupings must use the same boundary.
const UTC_DAY_SQL = "to_char(created_at AT TIME ZONE 'UTC', 'YYYY-MM-DD')";

export async function saveAgentRun(
  report: AgentTestReport,
  userId: string | null
): Promise<void> {
  await query(
    `
    INSERT INTO agent_runs (
      id, model, goal_id, goal_title, target, requested_depth, min_depth,
      seed, success, stop_reason, combinations_used, max_combinations,
      user_id, raw_report
    )
    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14::jsonb)
    `,
    [
      randomBytes(12).toString("hex"),
      report.model,
      report.goal.id,
      report.goal.title,
      report.goal.target.name,
      report.requestedDepth,
      report.minDepth,
      report.goal.metadata.seed ?? null,
      report.success,
      report.stopReason,
      report.combinationsUsed,
      report.maxCombinations,
      userId,
      JSON.stringify(report)
    ]
  );
}

type RecentRunRow = {
  id: string;
  model: string;
  success: boolean;
  stop_reason: string;
  combinations_used: number;
  created_at: string;
  raw_report: AgentTestReport;
};

export async function listAgentRunsForDay(
  depth: number,
  day: string,
  limit = 30
): Promise<AgentRunSummary[]> {
  const result = await query<RecentRunRow>(
    `
    SELECT id, model, success, stop_reason, combinations_used, created_at, raw_report
    FROM agent_runs
    WHERE requested_depth = $1 AND ${UTC_DAY_SQL} = $2
    ORDER BY created_at DESC
    LIMIT $3
    `,
    [depth, day, limit]
  );

  return result.rows.map((row) => ({
    id: row.id,
    model: row.model,
    success: row.success,
    stopReason: row.stop_reason,
    combinationsUsed: row.combinations_used,
    createdAt: new Date(row.created_at).toISOString(),
    report: row.raw_report
  }));
}

/**
 * Recovers the goal every model faced on a past day at a depth from the
 * stored reports (goals are only generated on demand for the current day).
 */
export async function getStoredArenaGoal(
  depth: number,
  day: string
): Promise<GoalPreset | null> {
  const result = await query<{ goal: GoalPreset }>(
    `
    SELECT raw_report -> 'goal' AS goal
    FROM agent_runs
    WHERE requested_depth = $1 AND ${UTC_DAY_SQL} = $2
    ORDER BY created_at DESC
    LIMIT 1
    `,
    [depth, day]
  );

  return result.rows[0]?.goal ?? null;
}

type DailyStatRow = {
  day: string;
  model: string;
  runs: string;
  wins: string;
  avg_combinations: string | null;
};

/**
 * Per-day, per-model aggregates. When `depth` is given the stats are scoped
 * to it; otherwise runs at every depth are pooled (for the overall board).
 */
export async function listArenaDailyStats(
  depth?: number
): Promise<ArenaDailyModelStat[]> {
  const result = await query<DailyStatRow>(
    `
    SELECT
      ${UTC_DAY_SQL} AS day,
      model,
      COUNT(*) AS runs,
      COUNT(*) FILTER (WHERE success) AS wins,
      AVG(combinations_used) FILTER (WHERE success) AS avg_combinations
    FROM agent_runs
    WHERE $1::integer IS NULL OR requested_depth = $1
    GROUP BY 1, 2
    ORDER BY 1 DESC
    `,
    [depth ?? null]
  );

  return result.rows.map((row) => {
    const runs = Number(row.runs);
    const wins = Number(row.wins);

    return {
      day: row.day,
      model: row.model,
      runs,
      wins,
      winRate: runs > 0 ? wins / runs : 0,
      avgCombinations:
        row.avg_combinations === null ? null : Number(row.avg_combinations)
    };
  });
}

export async function listAgentRankings(
  depth: number
): Promise<AgentRankingEntry[]> {
  const result = await query<RankingRow>(
    `
    SELECT
      model,
      COUNT(*) AS runs,
      COUNT(*) FILTER (WHERE success) AS wins,
      AVG(combinations_used) FILTER (WHERE success) AS avg_combinations
    FROM agent_runs
    WHERE requested_depth = $1
    GROUP BY model
    ORDER BY COUNT(*) FILTER (WHERE success)::float / COUNT(*) DESC, AVG(combinations_used) FILTER (WHERE success) ASC NULLS LAST
    `,
    [depth]
  );

  return result.rows.map((row) => {
    const runs = Number(row.runs);
    const wins = Number(row.wins);

    return {
      model: row.model,
      runs,
      wins,
      winRate: runs > 0 ? wins / runs : 0,
      avgCombinations:
        row.avg_combinations === null ? null : Number(row.avg_combinations)
    };
  });
}
