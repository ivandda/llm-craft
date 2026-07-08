"use client";

import { requestAgentTestRun } from "@/lib/api";
import {
  AGENT_MODEL_OPTIONS,
  DEFAULT_AGENT_MODEL,
  getAgentModelLabel
} from "@/lib/agentModels";
import { buildAgentPlaybackInventory } from "@/lib/agentPlayback";
import type { AgentTestReport, AuthUser } from "@/lib/types";
import {
  ArrowLeft,
  Bot,
  LogOut,
  Pause,
  Play,
  RotateCcw,
  SkipForward
} from "lucide-react";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";

const PLAYBACK_SPEEDS = [
  { label: "Slow", value: 1400 },
  { label: "Normal", value: 900 },
  { label: "Fast", value: 450 }
];

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
  const [report, setReport] = useState<AgentTestReport | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [playbackIndex, setPlaybackIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeedMs, setPlaybackSpeedMs] = useState(900);
  const [selectedModel, setSelectedModel] = useState(DEFAULT_AGENT_MODEL);
  const activeRunId = useRef(0);
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

  async function runTest() {
    const runId = activeRunId.current + 1;
    activeRunId.current = runId;
    setIsRunning(true);
    setErrorMessage(null);
    setIsPlaying(false);
    setPlaybackIndex(0);
    setReport(null);

    try {
      const nextReport = await requestAgentTestRun({
        depth: goalDepth,
        model: selectedModel
      });
      if (activeRunId.current !== runId) {
        return;
      }

      setReport(nextReport);
      setIsPlaying(nextReport.steps.length > 0);
    } catch {
      if (activeRunId.current !== runId) {
        return;
      }

      setErrorMessage("Agent test could not be completed.");
    } finally {
      if (activeRunId.current === runId) {
        setIsRunning(false);
      }
    }
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

  function changeGoalDepth(depth: number) {
    activeRunId.current += 1;
    setIsPlaying(false);
    setPlaybackIndex(0);
    onGoalDepthChange(depth);
  }

  function changeModel(model: string) {
    activeRunId.current += 1;
    setIsPlaying(false);
    setPlaybackIndex(0);
    setSelectedModel(model);
  }

  return (
    <main className="min-h-screen bg-stone-100 px-4 py-6 text-zinc-950">
      <div className="mx-auto grid max-w-6xl gap-5">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-200 pb-4">
          <div>
            <h1 className="flex items-center gap-2 text-xl font-semibold tracking-normal">
              <Bot size={20} />
              Agent Test
            </h1>
            <p className="mt-1 text-sm text-zinc-500">{user.displayName}</p>
          </div>
          <div className="flex items-center gap-2">
            <IconButton label="Back to modes" onClick={onBackToMenu}>
              <ArrowLeft size={17} />
            </IconButton>
            <IconButton label="Log out" onClick={onLogout}>
              <LogOut size={17} />
            </IconButton>
          </div>
        </header>

        <section className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
          <aside className="rounded-md border border-zinc-200 bg-white p-5 shadow-hairline">
            <label className="grid gap-2 text-sm font-semibold text-zinc-700">
              <span>Goal depth</span>
              <select
                className="h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none transition focus:border-zinc-500"
                disabled={isRunning}
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

            <label className="mt-4 grid gap-2 text-sm font-semibold text-zinc-700">
              <span>Agent model</span>
              <select
                className="h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none transition focus:border-zinc-500"
                disabled={isRunning}
                onChange={(event) => changeModel(event.target.value)}
                value={selectedModel}
              >
                {AGENT_MODEL_OPTIONS.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.label}
                  </option>
                ))}
              </select>
            </label>

            <button
              className="mt-4 flex h-10 w-full items-center justify-center gap-2 rounded-md bg-zinc-950 px-3 text-sm font-semibold text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-60"
              disabled={isRunning}
              onClick={() => void runTest()}
              type="button"
            >
              {report ? <RotateCcw size={16} /> : <Play size={16} />}
              {isRunning ? "Running" : report ? "Run again" : "Run test"}
            </button>

            {errorMessage ? (
              <p className="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                {errorMessage}
              </p>
            ) : null}

            {report ? <ReportMetrics report={report} /> : null}
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

          <section className="min-h-[520px] rounded-md border border-zinc-200 bg-white p-5 shadow-hairline">
            {!report && isRunning ? (
              <div className="grid h-full place-items-center text-sm text-zinc-500">
                Running agent test.
              </div>
            ) : null}

            {!report && !isRunning ? (
              <div className="grid h-full place-items-center text-center">
                <div>
                  <div className="mx-auto grid size-16 place-items-center rounded-md border border-zinc-200 bg-zinc-50 text-zinc-700">
                    <Bot size={28} />
                  </div>
                  <h2 className="mt-4 text-lg font-semibold">Agent ready</h2>
                  <p className="mt-2 max-w-sm text-sm text-zinc-500">
                    Choose a depth and model, then run the test to generate a goal.
                  </p>
                </div>
              </div>
            ) : null}

            {report ? (
              <ReportDetails
                inventory={visibleInventory}
                isPlaybackComplete={isPlaybackComplete}
                playbackIndex={playbackIndex}
                report={report}
                visibleSteps={visibleSteps}
              />
            ) : null}
          </section>
        </section>
      </div>
    </main>
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
    <div className="mt-5 rounded-md border border-zinc-200 bg-zinc-50 p-3">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-semibold">Playback</p>
        <p className="text-xs text-zinc-500">
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
      <label className="mt-3 grid gap-1 text-xs font-semibold text-zinc-600">
        <span>Speed</span>
        <select
          className="h-9 rounded-md border border-zinc-300 bg-white px-2 text-sm outline-none transition focus:border-zinc-500"
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

function ReportMetrics({ report }: { report: AgentTestReport }) {
  return (
    <div className="mt-5 grid grid-cols-2 gap-3">
      <MetricCard label="Min depth" value={report.minDepth} />
      <MetricCard label="Limit" value={report.maxCombinations} />
      <MetricCard label="Used" value={report.combinationsUsed} />
      <MetricCard label="Status" value={report.success ? "Pass" : "Fail"} />
      <div className="col-span-2 rounded-md border border-zinc-200 bg-zinc-50 p-3">
        <p className="truncate text-sm font-semibold">
          {getAgentModelLabel(report.model)}
        </p>
        <p className="mt-1 text-xs text-zinc-500">Agent model</p>
      </div>
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
            <p className="text-sm font-semibold text-zinc-500">Target</p>
            <h2 className="mt-1 text-3xl font-semibold capitalize tracking-normal">
              {report.goal.target.emoji ? `${report.goal.target.emoji} ` : ""}
              {report.goal.target.name}
            </h2>
          </div>
          <div className="max-w-md">
            <p className="text-sm font-semibold text-zinc-500">Initial inventory</p>
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
            className={`rounded-md border px-3 py-2 text-sm font-semibold ${
              report.success
                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                : "border-amber-200 bg-amber-50 text-amber-700"
            }`}
          >
            {formatStopReason(report.stopReason)}
          </span>
        ) : (
          <span className="rounded-md border border-sky-200 bg-sky-50 px-3 py-2 text-sm font-semibold text-sky-700">
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
        <h3 className="text-sm font-semibold">Visible inventory</h3>
        <div className="mt-2 flex flex-wrap gap-2">
          {inventory.map((element) => (
            <ElementPill key={element.id} name={element.name} emoji={element.emoji} />
          ))}
        </div>
      </div>

      <div>
        <h3 className="text-sm font-semibold">Agent path</h3>
        <div className="mt-3 grid gap-2">
          {report.steps.length === 0 ? (
            <p className="rounded-md border border-dashed border-zinc-200 p-3 text-sm text-zinc-500">
              No combinations were completed.
            </p>
          ) : (
            visibleSteps.map((step, index) => (
              <div
                className={`rounded-md border px-3 py-2 transition ${
                  index === visibleSteps.length - 1 && !isPlaybackComplete
                    ? "border-sky-300 bg-sky-50"
                    : "border-zinc-200 bg-zinc-50"
                }`}
                key={step.index}
              >
                <div className="flex flex-wrap items-center gap-2 text-sm">
                  <span className="font-semibold text-zinc-500">#{step.index}</span>
                  <ElementPill name={step.inputA.name} emoji={step.inputA.emoji} />
                  <span className="text-zinc-400">+</span>
                  <ElementPill name={step.inputB.name} emoji={step.inputB.emoji} />
                  <span className="text-zinc-400">=</span>
                  <ElementPill name={step.output.name} emoji={step.output.emoji} />
                </div>
                {step.agentReason ? (
                  <p className="mt-2 text-xs text-zinc-500">{step.agentReason}</p>
                ) : null}
              </div>
            ))
          )}
          {report.steps.length > 0 && visibleSteps.length === 0 ? (
            <p className="rounded-md border border-dashed border-zinc-200 p-3 text-sm text-zinc-500">
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
    <span className="inline-flex max-w-full items-center gap-1 rounded-md border border-zinc-200 bg-white px-2 py-1 text-xs font-medium capitalize text-zinc-700">
      <span>{emoji ?? "·"}</span>
      <span className="truncate">{name}</span>
    </span>
  );
}

function MetricCard({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-md border border-zinc-200 bg-zinc-50 p-3">
      <p className="text-2xl font-semibold">{value}</p>
      <p className="mt-1 text-xs text-zinc-500">{label}</p>
    </div>
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
      className="grid size-10 place-items-center rounded-md border border-zinc-200 bg-white text-zinc-600 transition hover:bg-zinc-50 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
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
