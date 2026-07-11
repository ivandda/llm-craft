import { query } from "@/lib/server/db";
import type {
  AgentRankingEntry,
  AgentRunSummary,
  AgentTestReport
} from "@/lib/types";
import { randomBytes } from "crypto";

type RankingRow = {
  model: string;
  runs: string;
  wins: string;
  avg_combinations: string | null;
};

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

export async function listRecentAgentRuns(
  depth: number,
  limit = 12
): Promise<AgentRunSummary[]> {
  const result = await query<RecentRunRow>(
    `
    SELECT id, model, success, stop_reason, combinations_used, created_at, raw_report
    FROM agent_runs
    WHERE requested_depth = $1
    ORDER BY created_at DESC
    LIMIT $2
    `,
    [depth, limit]
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
