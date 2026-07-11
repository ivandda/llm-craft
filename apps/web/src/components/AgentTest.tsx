"use client";

import { requestArenaFeed, requestArenaStats } from "@/lib/api";
import { getAgentModelLabel } from "@/lib/agentModels";
import { buildAgentPlaybackInventory } from "@/lib/agentPlayback";
import { getHueForConcept } from "@/lib/emoji";
import { ArenaCharts } from "@/components/ArenaCharts";
import { BackButton } from "@/components/BackButton";
import { ThemeToggle } from "@/components/ThemeToggle";
import type {
  AgentRankingEntry,
  AgentRunSummary,
  AgentTestReport,
  ArenaStats,
  AuthUser,
  GoalPreset
} from "@/lib/types";
import {
  Check,
  LogOut,
  Pause,
  Play,
  RotateCcw,
  SkipForward,
  Swords,
  Target,
  X
} from "lucide-react";
import {
  type CSSProperties,
  type ReactNode,
  useEffect,
  useMemo,
  useState
} from "react";

const PLAYBACK_SPEEDS = [
  { label: "Slow", value: 1400 },
  { label: "Normal", value: 900 },
  { label: "Fast", value: 450 }
];

const FEED_REFRESH_MS = 30_000;
const PODIUM_MEDALS = ["🥇", "🥈", "🥉"];

/** Arena days follow the UTC calendar, matching the daily goal seed. */
function currentUtcDay(): string {
  return new Date().toISOString().slice(0, 10);
}

function formatDayLabel(day: string): string {
  const parsed = new Date(`${day}T00:00:00Z`);

  if (Number.isNaN(parsed.getTime())) {
    return day;
  }

  return parsed.toLocaleDateString("en-US", {
    day: "numeric",
    month: "short",
    timeZone: "UTC",
    year: "numeric"
  });
}

export function AgentTest({
  goalDepth,
  user,
  onBackToMenu,
  onGoalDepthChange,
  onLogout
}: {
  goalDepth: number;
  user: AuthUser;
  onBackToMenu: () => void;
  onGoalDepthChange: (depth: number) => void;
  onLogout: () => void;
}) {
  const [goal, setGoal] = useState<GoalPreset | null>(null);
  const [runs, setRuns] = useState<AgentRunSummary[]>([]);
  const [stats, setStats] = useState<ArenaStats | null>(null);
  const [selectedDay, setSelectedDay] = useState<string>(currentUtcDay);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [isLoadingFeed, setIsLoadingFeed] = useState(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [playbackIndex, setPlaybackIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeedMs, setPlaybackSpeedMs] = useState(900);

  const today = currentUtcDay();
  const dayOptions = useMemo(() => {
    const days = new Set<string>([today, ...(stats?.days ?? [])]);
    return [...days].sort().reverse();
  }, [stats, today]);

  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? null;
  const report = selectedRun?.report ?? null;
  const totalSteps = report?.steps.length ?? 0;
  const isPlaybackComplete = Boolean(report && playbackIndex >= totalSteps);
  const visibleSteps = report?.steps.slice(0, playbackIndex) ?? [];
  const visibleInventory = useMemo(
    () =>
      report
        ? buildAgentPlaybackInventory(
            report.goal.initialInventory,
            report.steps,
            playbackIndex
          )
        : [],
    [playbackIndex, report]
  );

  useEffect(() => {
    let cancelled = false;

    async function loadFeed(showSpinner: boolean) {
      if (showSpinner) {
        setIsLoadingFeed(true);
      }

      try {
        const [feed, statsPayload] = await Promise.all([
          requestArenaFeed(goalDepth, selectedDay),
          requestArenaStats(goalDepth)
        ]);

        if (cancelled) {
          return;
        }

        setGoal(feed.goal);
        setRuns(feed.runs);
        setStats(statsPayload);
        setErrorMessage(null);
      } catch {
        if (!cancelled) {
          setErrorMessage("Could not load the arena. Retrying shortly.");
        }
      } finally {
        if (!cancelled) {
          setIsLoadingFeed(false);
        }
      }
    }

    void loadFeed(true);
    const timer = window.setInterval(() => void loadFeed(false), FEED_REFRESH_MS);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [goalDepth, selectedDay]);

  useEffect(() => {
    if (!report || !isPlaying || playbackIndex >= report.steps.length) {
      return;
    }

    const timer = window.setTimeout(() => {
      setPlaybackIndex((currentIndex) =>
        Math.min(currentIndex + 1, report.steps.length)
      );
    }, playbackSpeedMs);

    return () => window.clearTimeout(timer);
  }, [isPlaying, playbackIndex, playbackSpeedMs, report]);

  useEffect(() => {
    if (report && playbackIndex >= report.steps.length) {
      setIsPlaying(false);
    }
  }, [playbackIndex, report]);

  function selectRun(run: AgentRunSummary) {
    setSelectedRunId(run.id);
    setPlaybackIndex(0);
    setIsPlaying(run.report.steps.length > 0);
  }

  function changeGoalDepth(depth: number) {
    setSelectedRunId(null);
    setIsPlaying(false);
    setPlaybackIndex(0);
    setSelectedDay(currentUtcDay());
    onGoalDepthChange(depth);
  }

  function changeDay(day: string) {
    setSelectedRunId(null);
    setIsPlaying(false);
    setPlaybackIndex(0);
    setSelectedDay(day);
  }

  function restartPlayback() {
    if (!report) {
      return;
    }

    setPlaybackIndex(0);
    setIsPlaying(report.steps.length > 0);
  }

  function stepForward() {
    if (!report) {
      return;
    }

    setIsPlaying(false);
    setPlaybackIndex((currentIndex) =>
      Math.min(currentIndex + 1, report.steps.length)
    );
  }

  function togglePlayback() {
    if (!report || report.steps.length === 0 || isPlaybackComplete) {
      return;
    }

    setIsPlaying((currentValue) => !currentValue);
  }

  return (
    <main className="min-h-[100dvh] bg-paper px-4 py-6 text-ink">
      <div className="mx-auto grid max-w-6xl gap-5">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-linen pb-4">
          <div>
            <h1 className="flex items-center gap-2 font-display text-xl font-semibold tracking-normal">
              <Swords size={20} />
              LLM Arena
            </h1>
            <p className="mt-1 text-sm text-soot">
              Each model faces the same daily crafting goal. Replay any day,
              watch how they reason — and see who wins over time.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <BackButton label="Modes" onClick={onBackToMenu} />
            <ThemeToggle />
            <IconButton label="Log out" onClick={onLogout}>
              <LogOut size={17} />
            </IconButton>
          </div>
        </header>

        <section className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
          <aside className="rounded-md border border-linen bg-surface p-5 shadow-hairline">
            <div className="grid grid-cols-2 gap-3">
              <label className="grid gap-2 text-sm font-semibold text-ink">
                <span>Goal depth</span>
                <select
                  className="h-10 rounded-md border border-linen bg-surface px-3 text-sm outline-none transition focus:border-cobalt"
                  onChange={(event) => changeGoalDepth(Number(event.target.value))}
                  value={goalDepth}
                >
                  {Array.from({ length: 10 }, (_, index) => index + 1).map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
              <label className="grid gap-2 text-sm font-semibold text-ink">
                <span>Day</span>
                <select
                  className="h-10 rounded-md border border-linen bg-surface px-3 text-sm outline-none transition focus:border-cobalt"
                  onChange={(event) => changeDay(event.target.value)}
                  value={selectedDay}
                >
                  {dayOptions.map((day) => (
                    <option key={day} value={day}>
                      {day === today ? "Today" : formatDayLabel(day)}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            <DayChallenge day={selectedDay} goal={goal} isToday={selectedDay === today} />
            <Podium entries={stats?.rankings ?? []} />

            <p className="mt-4 rounded-md border border-linen bg-paper px-3 py-2 text-xs text-soot">
              Runs are triggered by the admins to keep model costs in check —
              every result is public and updates here live.
            </p>

            {errorMessage ? (
              <p className="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                {errorMessage}
              </p>
            ) : null}

            {report ? (
              <PlaybackControls
                isComplete={isPlaybackComplete}
                isPlaying={isPlaying}
                playbackIndex={playbackIndex}
                playbackSpeedMs={playbackSpeedMs}
                totalSteps={totalSteps}
                onPlaybackSpeedChange={setPlaybackSpeedMs}
                onRestart={restartPlayback}
                onStepForward={stepForward}
                onTogglePlayback={togglePlayback}
              />
            ) : null}
          </aside>

          <section className="min-h-[520px] rounded-md border border-linen bg-surface p-5 shadow-hairline">
            <RecentRuns
              dayLabel={selectedDay === today ? "today" : formatDayLabel(selectedDay)}
              isLoading={isLoadingFeed}
              runs={runs}
              selectedRunId={selectedRunId}
              onSelect={selectRun}
            />

            {report ? (
              <div className="mt-5 border-t border-linen pt-5">
                <ReportDetails
                  inventory={visibleInventory}
                  isPlaybackComplete={isPlaybackComplete}
                  playbackIndex={playbackIndex}
                  report={report}
                  visibleSteps={visibleSteps}
                />
              </div>
            ) : (
              <div className="mt-10 grid place-items-center text-center">
                <div>
                  <div className="mx-auto grid size-16 place-items-center rounded-md border border-linen bg-paper text-soot">
                    <Swords size={28} />
                  </div>
                  <h2 className="mt-4 font-display text-lg font-semibold">
                    Pick a run to replay it
                  </h2>
                  <p className="mt-2 max-w-sm text-sm text-soot">
                    Select any run above to watch the model&apos;s path step by
                    step: which pairs it combined, and why.
                  </p>
                </div>
              </div>
            )}
          </section>
        </section>

        {stats ? (
          <ArenaCharts
            daily={stats.daily}
            depth={goalDepth}
            overall={stats.overall}
            rankings={stats.rankings}
          />
        ) : null}
      </div>
    </main>
  );
}

function DayChallenge({
  day,
  goal,
  isToday
}: {
  day: string;
  goal: GoalPreset | null;
  isToday: boolean;
}) {
  return (
    <div className="mt-5 rounded-md border border-linen bg-paper p-3">
      <p className="flex items-center gap-1.5 font-mono text-xs font-semibold uppercase tracking-wider text-soot">
        <Target size={13} />
        {isToday ? "Today's challenge" : `Challenge · ${formatDayLabel(day)}`}
      </p>
      {goal ? (
        <>
          <div className="mt-2">
            <ElementPill name={goal.target.name} emoji={goal.target.emoji} />
          </div>
          <p className="mt-2 text-xs text-soot">
            Doable in {goal.metadata.minDepth ?? goal.metadata.depth} steps ·
            starting from {goal.initialInventory.length} elements · same goal
            for every model {isToday ? "today" : "that day"}.
          </p>
        </>
      ) : (
        <p className="mt-2 text-xs text-soot">
          {isToday
            ? "Loading…"
            : "No recorded challenge for this day at this depth."}
        </p>
      )}
    </div>
  );
}

function Podium({ entries }: { entries: AgentRankingEntry[] }) {
  return (
    <div className="mt-4 rounded-md border border-linen bg-paper p-3">
      <p className="font-mono text-xs font-semibold uppercase tracking-wider text-soot">
        Ranking · this depth · all days
      </p>
      {entries.length === 0 ? (
        <p className="mt-2 text-xs text-soot">
          No runs recorded at this depth yet.
        </p>
      ) : (
        <div className="mt-2 grid gap-2">
          {entries.map((entry, index) => (
            <div
              className="rounded-md border border-linen bg-surface px-2.5 py-2"
              key={entry.model}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="flex min-w-0 items-center gap-2 text-sm">
                  <span aria-hidden>{PODIUM_MEDALS[index] ?? "·"}</span>
                  <span className="truncate font-medium">
                    {getAgentModelLabel(entry.model)}
                  </span>
                </span>
                <span className="shrink-0 font-mono text-sm font-semibold">
                  {Math.round(entry.winRate * 100)}%
                </span>
              </div>
              <p className="mt-0.5 font-mono text-xs text-soot">
                {entry.wins}/{entry.runs} wins
                {entry.avgCombinations !== null
                  ? ` · ${entry.avgCombinations.toFixed(1)} moves/win`
                  : ""}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function RecentRuns({
  dayLabel,
  isLoading,
  runs,
  selectedRunId,
  onSelect
}: {
  dayLabel: string;
  isLoading: boolean;
  runs: AgentRunSummary[];
  selectedRunId: string | null;
  onSelect: (run: AgentRunSummary) => void;
}) {
  return (
    <div>
      <h3 className="font-mono text-xs font-semibold uppercase tracking-wider text-soot">
        Runs · {dayLabel}
      </h3>
      {isLoading && runs.length === 0 ? (
        <p className="mt-3 text-sm text-soot">Loading runs…</p>
      ) : runs.length === 0 ? (
        <p className="mt-3 rounded-md border border-dashed border-linen p-3 text-sm text-soot">
          No runs at this depth {dayLabel === "today" ? "today" : `on ${dayLabel}`}{" "}
          — pick another day above, or check back after the next arena session.
        </p>
      ) : (
        <div className="mt-3 flex flex-wrap gap-2">
          {runs.map((run) => (
            <button
              className={`flex items-center gap-2 rounded-md border px-3 py-2 text-left text-sm transition ${
                run.id === selectedRunId
                  ? "border-cobalt bg-cobalt/5"
                  : "border-linen bg-paper hover:border-cobalt/50"
              }`}
              key={run.id}
              onClick={() => onSelect(run)}
              type="button"
            >
              <span
                className={`grid size-5 shrink-0 place-items-center rounded-full ${
                  run.success
                    ? "bg-emerald-100 text-emerald-700"
                    : "bg-red-100 text-red-600"
                }`}
              >
                {run.success ? <Check size={12} /> : <X size={12} />}
              </span>
              <span className="min-w-0">
                <span className="block truncate font-medium">
                  {getAgentModelLabel(run.model)}
                </span>
                <span className="block font-mono text-xs text-soot">
                  {run.combinationsUsed} moves · {formatTimeAgo(run.createdAt)}
                </span>
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function PlaybackControls({
  isComplete,
  isPlaying,
  playbackIndex,
  playbackSpeedMs,
  totalSteps,
  onPlaybackSpeedChange,
  onRestart,
  onStepForward,
  onTogglePlayback
}: {
  isComplete: boolean;
  isPlaying: boolean;
  playbackIndex: number;
  playbackSpeedMs: number;
  totalSteps: number;
  onPlaybackSpeedChange: (speedMs: number) => void;
  onRestart: () => void;
  onStepForward: () => void;
  onTogglePlayback: () => void;
}) {
  return (
    <div className="mt-5 rounded-md border border-linen bg-paper p-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-semibold">Playback</p>
        <p className="font-mono text-xs text-soot">
          {playbackIndex}/{totalSteps}
        </p>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2">
        <IconButton
          disabled={totalSteps === 0 || isComplete}
          label={isPlaying ? "Pause playback" : "Play playback"}
          onClick={onTogglePlayback}
        >
          {isPlaying ? <Pause size={16} /> : <Play size={16} />}
        </IconButton>
        <IconButton
          disabled={totalSteps === 0}
          label="Restart playback"
          onClick={onRestart}
        >
          <RotateCcw size={16} />
        </IconButton>
        <IconButton
          disabled={totalSteps === 0 || isComplete}
          label="Step forward"
          onClick={onStepForward}
        >
          <SkipForward size={16} />
        </IconButton>
      </div>
      <label className="mt-3 grid gap-1 text-xs font-semibold text-soot">
        <span>Speed</span>
        <select
          className="h-9 rounded-md border border-linen bg-surface px-2 text-sm outline-none transition focus:border-cobalt"
          onChange={(event) => onPlaybackSpeedChange(Number(event.target.value))}
          value={playbackSpeedMs}
        >
          {PLAYBACK_SPEEDS.map((speed) => (
            <option key={speed.value} value={speed.value}>
              {speed.label}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

function ReportDetails({
  inventory,
  isPlaybackComplete,
  playbackIndex,
  report,
  visibleSteps
}: {
  inventory: AgentTestReport["finalInventory"];
  isPlaybackComplete: boolean;
  playbackIndex: number;
  report: AgentTestReport;
  visibleSteps: AgentTestReport["steps"];
}) {
  return (
    <div className="grid gap-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex min-w-0 flex-wrap items-start gap-6">
          <div>
            <p className="font-mono text-xs font-semibold uppercase tracking-wider text-soot">
              {getAgentModelLabel(report.model)} · target
            </p>
            <h2 className="mt-1 font-display text-3xl font-semibold capitalize tracking-normal">
              {report.goal.target.emoji ? `${report.goal.target.emoji} ` : ""}
              {report.goal.target.name}
            </h2>
          </div>
          <div className="max-w-md">
            <p className="font-mono text-xs font-semibold uppercase tracking-wider text-soot">Initial inventory</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {report.goal.initialInventory.map((element) => (
                <ElementPill
                  key={element.id}
                  name={element.name}
                  emoji={element.emoji}
                />
              ))}
            </div>
          </div>
        </div>
        {isPlaybackComplete ? (
          <span
            className={`rounded-md border px-3 py-2 font-mono text-sm font-semibold ${
              report.success
                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                : "border-amber-200 bg-amber-50 text-amber-700"
            }`}
          >
            {formatStopReason(report.stopReason)}
          </span>
        ) : (
          <span className="rounded-md border border-cobalt/30 bg-cobalt/5 px-3 py-2 font-mono text-sm font-semibold text-accent">
            Replaying {playbackIndex}/{report.steps.length}
          </span>
        )}
      </div>

      {isPlaybackComplete && report.errorMessage ? (
        <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {report.errorMessage}
        </p>
      ) : null}

      <div>
        <h3 className="font-mono text-xs font-semibold uppercase tracking-wider text-soot">Visible inventory</h3>
        <div className="mt-2 flex flex-wrap gap-2">
          {inventory.map((element) => (
            <ElementPill key={element.id} name={element.name} emoji={element.emoji} />
          ))}
        </div>
      </div>

      <div>
        <h3 className="font-mono text-xs font-semibold uppercase tracking-wider text-soot">Agent path</h3>
        <div className="mt-3 grid gap-2">
          {report.steps.length === 0 ? (
            <p className="rounded-md border border-dashed border-linen p-3 text-sm text-soot">
              No combinations were completed.
            </p>
          ) : (
            visibleSteps.map((step, index) => (
              <div
                className={`rounded-md border px-3 py-2 transition ${
                  index === visibleSteps.length - 1 && !isPlaybackComplete
                    ? "border-cobalt/40 bg-cobalt/5"
                    : "border-linen bg-paper"
                }`}
                key={step.index}
              >
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <span className="font-mono font-semibold text-soot">#{step.index}</span>
                  <ElementPill name={step.inputA.name} emoji={step.inputA.emoji} />
                  <span className="text-soot">+</span>
                  <ElementPill name={step.inputB.name} emoji={step.inputB.emoji} />
                  <span className="text-soot">=</span>
                  <ElementPill name={step.output.name} emoji={step.output.emoji} />
                </div>
                {step.agentReason ? (
                  <p className="mt-2 text-xs text-soot">{step.agentReason}</p>
                ) : null}
              </div>
            ))
          )}
          {report.steps.length > 0 && visibleSteps.length === 0 ? (
            <p className="rounded-md border border-dashed border-linen p-3 text-sm text-soot">
              Playback is ready.
            </p>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function ElementPill({ emoji, name }: { emoji?: string; name: string }) {
  return (
    <span
      className="element-card inline-flex max-w-full items-center gap-1 rounded-md border px-2 py-1 text-xs font-medium capitalize text-ink"
      style={{ "--el-hue": getHueForConcept(name) } as CSSProperties}
    >
      <span>{emoji ?? "·"}</span>
      <span className="truncate">{name}</span>
    </span>
  );
}

function IconButton({
  children,
  disabled = false,
  label,
  onClick
}: {
  children: ReactNode;
  disabled?: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      aria-label={label}
      className="grid size-10 place-items-center rounded-md border border-linen bg-surface text-soot transition hover:bg-paper hover:text-ink disabled:cursor-not-allowed disabled:opacity-50"
      disabled={disabled}
      onClick={onClick}
      title={label}
      type="button"
    >
      {children}
    </button>
  );
}

function formatStopReason(reason: AgentTestReport["stopReason"]): string {
  return reason.replace(/_/g, " ");
}

function formatTimeAgo(isoDate: string): string {
  const elapsedMs = Date.now() - new Date(isoDate).getTime();
  const minutes = Math.floor(elapsedMs / 60_000);

  if (minutes < 1) {
    return "just now";
  }

  if (minutes < 60) {
    return `${minutes}m ago`;
  }

  const hours = Math.floor(minutes / 60);

  if (hours < 24) {
    return `${hours}h ago`;
  }

  return `${Math.floor(hours / 24)}d ago`;
}
