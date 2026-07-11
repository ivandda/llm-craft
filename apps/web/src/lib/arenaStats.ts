import type { ArenaDailyModelStat, ArenaOverallEntry } from "@/lib/types";

/**
 * Collapses per-day, per-model rows into one leaderboard entry per model.
 *
 * The score is the mean of the model's per-day win rates, computed only over
 * the days it actually ran — a model absent on some days is neither punished
 * for missing them nor able to farm a high-volume day for extra weight.
 */
export function aggregateOverallStats(
  rows: ArenaDailyModelStat[]
): ArenaOverallEntry[] {
  const byModel = new Map<string, ArenaDailyModelStat[]>();

  for (const row of rows) {
    const existing = byModel.get(row.model);

    if (existing) {
      existing.push(row);
    } else {
      byModel.set(row.model, [row]);
    }
  }

  const entries = [...byModel.entries()].map(([model, modelRows]) => {
    const runs = modelRows.reduce((total, row) => total + row.runs, 0);
    const wins = modelRows.reduce((total, row) => total + row.wins, 0);
    const score =
      modelRows.reduce((total, row) => total + row.winRate, 0) /
      modelRows.length;
    const combinationTotals = modelRows.reduce(
      (totals, row) =>
        row.avgCombinations === null
          ? totals
          : {
              combinations: totals.combinations + row.avgCombinations * row.wins,
              wins: totals.wins + row.wins
            },
      { combinations: 0, wins: 0 }
    );

    return {
      model,
      daysPlayed: new Set(modelRows.map((row) => row.day)).size,
      runs,
      wins,
      score,
      avgCombinations:
        combinationTotals.wins > 0
          ? combinationTotals.combinations / combinationTotals.wins
          : null
    };
  });

  return entries.sort(
    (a, b) =>
      b.score - a.score ||
      (a.avgCombinations ?? Infinity) - (b.avgCombinations ?? Infinity) ||
      b.runs - a.runs
  );
}
