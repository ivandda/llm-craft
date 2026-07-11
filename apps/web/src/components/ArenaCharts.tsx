"use client";

import { AGENT_MODEL_OPTIONS, getAgentModelLabel } from "@/lib/agentModels";
import { useTheme, type Theme } from "@/lib/theme";
import type {
  AgentRankingEntry,
  ArenaDailyModelStat,
  ArenaOverallEntry
} from "@/lib/types";
import { useMemo } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps
} from "recharts";

// Categorical series palette validated (CVD + contrast) against the app's
// paper surfaces in both modes; slots are assigned per model in fixed order.
const SERIES_COLORS: Record<Theme, string[]> = {
  light: ["#2a78d6", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#e34948"],
  dark: ["#3987e5", "#199e70", "#c98500", "#008300", "#9085e9", "#e66767"]
};

// Chart chrome mirrors the app tokens (linen/soot/surface/ink) so the plots
// re-theme with the rest of the page.
const CHART_CHROME: Record<
  Theme,
  { grid: string; axis: string; ink: string; tooltipBg: string }
> = {
  light: { grid: "#E3DACA", axis: "#6F6759", ink: "#26221B", tooltipBg: "#FFFDF8" },
  dark: { grid: "#3A342A", axis: "#A89F8F", ink: "#F4F0E7", tooltipBg: "#201D16" }
};

type TrendPoint = { day: string } & Record<string, string | number | null>;

export function ArenaCharts({
  daily,
  depth,
  overall,
  rankings
}: {
  daily: ArenaDailyModelStat[];
  depth: number;
  overall: ArenaOverallEntry[];
  rankings: AgentRankingEntry[];
}) {
  const theme = useTheme();
  const chrome = CHART_CHROME[theme];

  // Colors follow the model across every chart and depth: known models keep
  // their fixed slot, unseen historical ones get the remaining slots.
  const modelColors = useMemo(() => {
    const knownModels = AGENT_MODEL_OPTIONS.map((option) => option.id);
    const extraModels = [
      ...new Set(
        [...overall.map((entry) => entry.model), ...daily.map((stat) => stat.model)]
      )
    ]
      .filter((model) => !knownModels.includes(model))
      .sort();
    const palette = SERIES_COLORS[theme];

    return new Map(
      [...knownModels, ...extraModels].map((model, index) => [
        model,
        palette[index % palette.length]
      ])
    );
  }, [daily, overall, theme]);

  const trendModels = useMemo(
    () =>
      [...modelColors.keys()].filter((model) =>
        daily.some((stat) => stat.model === model)
      ),
    [daily, modelColors]
  );

  const trendData = useMemo<TrendPoint[]>(() => {
    const days = [...new Set(daily.map((stat) => stat.day))].sort();

    return days.map((day) => {
      const point: TrendPoint = { day };

      for (const model of trendModels) {
        const stat = daily.find(
          (candidate) => candidate.day === day && candidate.model === model
        );
        point[model] = stat ? Math.round(stat.winRate * 100) : null;
        point[`${model}:runs`] = stat?.runs ?? null;
      }

      return point;
    });
  }, [daily, trendModels]);

  const efficiencyData = useMemo(
    () =>
      rankings
        .filter((entry) => entry.avgCombinations !== null)
        .map((entry) => ({
          model: entry.model,
          label: getAgentModelLabel(entry.model),
          avg: Number(entry.avgCombinations?.toFixed(1)),
          wins: entry.wins
        })),
    [rankings]
  );

  return (
    <section className="grid gap-4">
      <h2 className="font-mono text-xs font-semibold uppercase tracking-wider text-soot">
        Model comparison
      </h2>

      <div className="rounded-md border border-linen bg-surface p-5 shadow-hairline">
        <ChartTitle
          title="Overall leaderboard"
          subtitle="All depths and days · score is the average of each day's win rate over the days a model actually ran, so missing days neither punish nor reward it"
        />
        {overall.length === 0 ? (
          <EmptyChart message="No runs recorded yet." />
        ) : (
          <div className="mt-4 grid gap-3">
            {overall.map((entry, index) => (
              <LeaderboardRow
                color={modelColors.get(entry.model) ?? chrome.axis}
                entry={entry}
                key={entry.model}
                rank={index + 1}
              />
            ))}
          </div>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-md border border-linen bg-surface p-5 shadow-hairline">
          <ChartTitle
            title="Win rate by day"
            subtitle={`Depth ${depth} · each point is a model's win rate for that day's challenge`}
          />
          {trendData.length === 0 ? (
            <EmptyChart message="No runs at this depth yet." />
          ) : (
            <>
              <ChartLegend colors={modelColors} models={trendModels} />
              <div className="mt-3 h-56">
                <ResponsiveContainer height="100%" width="100%">
                  <LineChart
                    data={trendData}
                    margin={{ top: 8, right: 12, bottom: 0, left: -18 }}
                  >
                    <CartesianGrid stroke={chrome.grid} strokeWidth={1} vertical={false} />
                    <XAxis
                      axisLine={{ stroke: chrome.grid }}
                      dataKey="day"
                      interval="preserveStartEnd"
                      tick={{ fill: chrome.axis, fontSize: 11 }}
                      tickFormatter={formatDayTick}
                      tickLine={false}
                    />
                    <YAxis
                      axisLine={false}
                      domain={[0, 100]}
                      tick={{ fill: chrome.axis, fontSize: 11 }}
                      tickFormatter={(value: number) => `${value}%`}
                      tickLine={false}
                      ticks={[0, 25, 50, 75, 100]}
                    />
                    <Tooltip
                      content={(props) => (
                        <TrendTooltip {...props} colors={modelColors} />
                      )}
                      cursor={{ stroke: chrome.grid, strokeWidth: 1 }}
                    />
                    {trendModels.map((model) => (
                      <Line
                        connectNulls={false}
                        dataKey={model}
                        dot={{ fill: modelColors.get(model), r: 3.5, strokeWidth: 0 }}
                        activeDot={{ r: 5, strokeWidth: 0 }}
                        isAnimationActive={false}
                        key={model}
                        stroke={modelColors.get(model)}
                        strokeWidth={2}
                        type="monotone"
                      />
                    ))}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </div>

        <div className="rounded-md border border-linen bg-surface p-5 shadow-hairline">
          <ChartTitle
            title="Moves per win"
            subtitle={`Depth ${depth} · average combinations on successful runs — lower is more efficient`}
          />
          {efficiencyData.length === 0 ? (
            <EmptyChart message="No successful runs at this depth yet." />
          ) : (
            <div className="mt-3 h-56">
              <ResponsiveContainer height="100%" width="100%">
                <BarChart
                  data={efficiencyData}
                  margin={{ top: 20, right: 12, bottom: 0, left: -18 }}
                >
                  <CartesianGrid stroke={chrome.grid} strokeWidth={1} vertical={false} />
                  <XAxis
                    axisLine={{ stroke: chrome.grid }}
                    dataKey="label"
                    interval={0}
                    tick={{ fill: chrome.axis, fontSize: 11 }}
                    tickLine={false}
                  />
                  <YAxis
                    axisLine={false}
                    tick={{ fill: chrome.axis, fontSize: 11 }}
                    tickLine={false}
                  />
                  <Tooltip
                    content={(props) => <EfficiencyTooltip {...props} />}
                    cursor={{ fill: chrome.grid, fillOpacity: 0.35 }}
                  />
                  <Bar barSize={32} dataKey="avg" isAnimationActive={false} radius={[4, 4, 0, 0]}>
                    {efficiencyData.map((entry) => (
                      <Cell
                        fill={modelColors.get(entry.model) ?? chrome.axis}
                        key={entry.model}
                      />
                    ))}
                    <LabelList
                      dataKey="avg"
                      fill={chrome.ink}
                      fontSize={12}
                      position="top"
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

function LeaderboardRow({
  color,
  entry,
  rank
}: {
  color: string;
  entry: ArenaOverallEntry;
  rank: number;
}) {
  const scorePercent = Math.round(entry.score * 100);

  return (
    <div className="grid gap-1.5">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <span className="flex items-center gap-2 text-sm font-medium">
          <span className="font-mono text-xs text-soot">#{rank}</span>
          <span
            aria-hidden
            className="inline-block size-2.5 rounded-full"
            style={{ backgroundColor: color }}
          />
          {getAgentModelLabel(entry.model)}
        </span>
        <span className="font-mono text-xs text-soot">
          <span className="mr-2 text-sm font-semibold text-ink">{scorePercent}%</span>
          {entry.wins}/{entry.runs} wins · {entry.daysPlayed}{" "}
          {entry.daysPlayed === 1 ? "day" : "days"}
          {entry.avgCombinations !== null
            ? ` · ${entry.avgCombinations.toFixed(1)} moves/win`
            : ""}
        </span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-paper">
        <div
          className="h-full rounded-full"
          style={{ backgroundColor: color, width: `${Math.max(scorePercent, 2)}%` }}
        />
      </div>
    </div>
  );
}

function ChartTitle({ subtitle, title }: { subtitle: string; title: string }) {
  return (
    <div>
      <h3 className="text-sm font-semibold">{title}</h3>
      <p className="mt-0.5 text-xs text-soot">{subtitle}</p>
    </div>
  );
}

function ChartLegend({
  colors,
  models
}: {
  colors: Map<string, string>;
  models: string[];
}) {
  return (
    <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1">
      {models.map((model) => (
        <span className="flex items-center gap-1.5 text-xs text-ink" key={model}>
          <span
            aria-hidden
            className="inline-block size-2.5 rounded-full"
            style={{ backgroundColor: colors.get(model) }}
          />
          {getAgentModelLabel(model)}
        </span>
      ))}
    </div>
  );
}

function EmptyChart({ message }: { message: string }) {
  return (
    <p className="mt-4 rounded-md border border-dashed border-linen p-3 text-sm text-soot">
      {message}
    </p>
  );
}

function TrendTooltip({
  active,
  colors,
  label,
  payload
}: TooltipContentProps & { colors: Map<string, string> }) {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  const point = payload[0]?.payload as TrendPoint | undefined;

  return (
    <div className="rounded-md border border-linen bg-surface px-3 py-2 text-xs shadow-hairline">
      <p className="font-semibold text-ink">{formatDayTick(String(label))}</p>
      <div className="mt-1 grid gap-0.5">
        {payload.map((item) => {
          const model = String(item.dataKey);
          const runs = point?.[`${model}:runs`];

          return (
            <p className="flex items-center gap-1.5 text-soot" key={model}>
              <span
                aria-hidden
                className="inline-block size-2 rounded-full"
                style={{ backgroundColor: colors.get(model) }}
              />
              {getAgentModelLabel(model)}:{" "}
              <span className="font-semibold text-ink">{item.value}%</span>
              {typeof runs === "number"
                ? ` (${runs} ${runs === 1 ? "run" : "runs"})`
                : ""}
            </p>
          );
        })}
      </div>
    </div>
  );
}

function EfficiencyTooltip({ active, payload }: TooltipContentProps) {
  const entry = payload?.[0]?.payload as
    | { label: string; avg: number; wins: number }
    | undefined;

  if (!active || !entry) {
    return null;
  }

  return (
    <div className="rounded-md border border-linen bg-surface px-3 py-2 text-xs shadow-hairline">
      <p className="font-semibold text-ink">{entry.label}</p>
      <p className="mt-0.5 text-soot">
        {entry.avg} moves/win over {entry.wins} {entry.wins === 1 ? "win" : "wins"}
      </p>
    </div>
  );
}

function formatDayTick(day: string): string {
  const parsed = new Date(`${day}T00:00:00Z`);

  if (Number.isNaN(parsed.getTime())) {
    return day;
  }

  return parsed.toLocaleDateString("en-US", {
    day: "numeric",
    month: "short",
    timeZone: "UTC"
  });
}
