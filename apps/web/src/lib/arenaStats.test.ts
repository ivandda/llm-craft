import { aggregateOverallStats } from "@/lib/arenaStats";
import type { ArenaDailyModelStat } from "@/lib/types";
import { describe, expect, it } from "vitest";

function stat(overrides: Partial<ArenaDailyModelStat>): ArenaDailyModelStat {
  return {
    day: "2026-07-10",
    model: "gemini-2.5-flash",
    runs: 1,
    wins: 1,
    winRate: 1,
    avgCombinations: 5,
    ...overrides
  };
}

describe("aggregateOverallStats", () => {
  it("averages win rates per day, not per run", () => {
    const [entry] = aggregateOverallStats([
      // 4 runs, 4 wins on a busy day…
      stat({ day: "2026-07-09", runs: 4, wins: 4, winRate: 1 }),
      // …must not outweigh a lost single-run day.
      stat({ day: "2026-07-10", runs: 1, wins: 0, winRate: 0, avgCombinations: null })
    ]);

    expect(entry.score).toBe(0.5);
    expect(entry.daysPlayed).toBe(2);
    expect(entry.runs).toBe(5);
    expect(entry.wins).toBe(4);
  });

  it("does not penalize models that skipped days", () => {
    const entries = aggregateOverallStats([
      stat({ model: "everyday", day: "2026-07-09", winRate: 0.5, wins: 1, runs: 2 }),
      stat({ model: "everyday", day: "2026-07-10", winRate: 0.5, wins: 1, runs: 2 }),
      stat({ model: "one-day", day: "2026-07-10", winRate: 1, wins: 1, runs: 1 })
    ]);

    const oneDay = entries.find((entry) => entry.model === "one-day");
    const everyday = entries.find((entry) => entry.model === "everyday");

    expect(oneDay?.score).toBe(1);
    expect(oneDay?.daysPlayed).toBe(1);
    expect(everyday?.score).toBe(0.5);
    expect(everyday?.daysPlayed).toBe(2);
    expect(entries[0]?.model).toBe("one-day");
  });

  it("weights avg combinations by wins and skips winless days", () => {
    const [entry] = aggregateOverallStats([
      stat({ day: "2026-07-08", runs: 2, wins: 2, winRate: 1, avgCombinations: 4 }),
      stat({ day: "2026-07-09", runs: 1, wins: 1, winRate: 1, avgCombinations: 10 }),
      stat({ day: "2026-07-10", runs: 1, wins: 0, winRate: 0, avgCombinations: null })
    ]);

    // (4*2 + 10*1) / 3 wins
    expect(entry.avgCombinations).toBe(6);
  });

  it("returns null avg combinations for models with no wins", () => {
    const [entry] = aggregateOverallStats([
      stat({ runs: 3, wins: 0, winRate: 0, avgCombinations: null })
    ]);

    expect(entry.avgCombinations).toBeNull();
  });

  it("ranks winless models below winners regardless of volume", () => {
    const entries = aggregateOverallStats([
      stat({ model: "loser", runs: 10, wins: 0, winRate: 0, avgCombinations: null }),
      stat({ model: "winner", runs: 1, wins: 1, winRate: 1 })
    ]);

    expect(entries.map((entry) => entry.model)).toEqual(["winner", "loser"]);
  });
});
