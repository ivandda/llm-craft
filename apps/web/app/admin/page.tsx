"use client";

import { AGENT_MODEL_OPTIONS, getAgentModelLabel } from "@/lib/agentModels";
import type { AgentTestReport } from "@/lib/types";
import {
  Activity,
  ExternalLink,
  Play,
  RefreshCw,
  ShieldCheck,
  Square,
  Swords
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";

type VmStatus = {
  name: string;
  zone: string;
  project: string;
  machineType: string;
  accelerator?: string;
  status: string;
  internalIp?: string;
  externalIp?: string;
  lastStartTimestamp?: string;
  lastStopTimestamp?: string;
  consoleUrl: string;
};

const REFRESH_INTERVAL_MS = 10_000;
const GPU_HOURLY_COST_USD = 0.85;

const STATUS_STYLES: Record<string, string> = {
  RUNNING: "bg-emerald-100 text-emerald-800 border-emerald-300",
  TERMINATED: "bg-paper text-soot border-linen",
  STOPPING: "bg-amber-100 text-amber-800 border-amber-300",
  STAGING: "bg-amber-100 text-amber-800 border-amber-300",
  PROVISIONING: "bg-amber-100 text-amber-800 border-amber-300"
};

export default function AdminDashboardPage() {
  const [vm, setVm] = useState<VmStatus | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [pendingAction, setPendingAction] = useState<"start" | "stop" | null>(null);

  const refreshStatus = useCallback(async () => {
    setIsRefreshing(true);

    try {
      const response = await fetch("/api/admin/vm", { cache: "no-store" });
      const payload = (await response.json()) as { vm?: VmStatus; error?: string };

      if (!response.ok || !payload.vm) {
        throw new Error(payload.error ?? `Status request failed (${response.status})`);
      }

      setVm(payload.vm);
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Status request failed");
    } finally {
      setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refreshStatus();
    const timer = window.setInterval(() => void refreshStatus(), REFRESH_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [refreshStatus]);

  async function sendVmAction(action: "start" | "stop") {
    setPendingAction(action);
    setErrorMessage(null);

    try {
      const response = await fetch("/api/admin/vm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action })
      });
      const payload = (await response.json()) as { vm?: VmStatus; error?: string };

      if (!response.ok || !payload.vm) {
        throw new Error(payload.error ?? `${action} request failed (${response.status})`);
      }

      setVm(payload.vm);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : `${action} request failed`);
    } finally {
      setPendingAction(null);
    }
  }

  const isRunning = vm?.status === "RUNNING";
  const isTransitional =
    vm !== null && !["RUNNING", "TERMINATED"].includes(vm.status);

  return (
    <main className="min-h-[100dvh] bg-paper px-4 py-8 text-ink">
      <div className="mx-auto max-w-2xl">
        <header className="flex items-center justify-between border-b border-linen pb-4">
          <div>
            <h1 className="flex items-center gap-2 font-display text-xl font-semibold">
              <ShieldCheck className="size-5 text-accent" aria-hidden />
              Admin · model server
            </h1>
            <p className="mt-1 text-sm text-soot">
              Start the GPU VM for demos, stop it when you are done.
            </p>
          </div>
          <button
            type="button"
            onClick={() => void refreshStatus()}
            className="flex h-9 items-center gap-2 rounded-md border border-linen bg-surface px-3 text-sm text-soot transition hover:border-cobalt hover:text-ink"
          >
            <RefreshCw
              className={`size-4 ${isRefreshing ? "animate-spin" : ""}`}
              aria-hidden
            />
            Refresh
          </button>
        </header>

        <section className="mt-6 rounded-md border border-linen bg-surface p-5 shadow-hairline">
          {vm ? (
            <>
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-3">
                  <span
                    className={`rounded-md border px-2.5 py-1 font-mono text-xs font-semibold ${
                      STATUS_STYLES[vm.status] ?? STATUS_STYLES.TERMINATED
                    }`}
                  >
                    {vm.status}
                  </span>
                  <div>
                    <p className="font-semibold">{vm.name}</p>
                    <p className="font-mono text-xs text-soot">
                      {vm.machineType}
                      {vm.accelerator ? ` · ${vm.accelerator}` : ""} · {vm.zone}
                    </p>
                  </div>
                </div>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => void sendVmAction("start")}
                    disabled={isRunning || isTransitional || pendingAction !== null}
                    className="flex h-10 items-center gap-2 rounded-md bg-cobalt px-4 text-sm font-semibold text-white transition hover:bg-cobalt-deep disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <Play className="size-4" aria-hidden />
                    {pendingAction === "start" ? "Starting…" : "Start"}
                  </button>
                  <button
                    type="button"
                    onClick={() => void sendVmAction("stop")}
                    disabled={!isRunning || pendingAction !== null}
                    className="flex h-10 items-center gap-2 rounded-md border border-linen bg-surface px-4 text-sm font-semibold text-ink transition hover:border-cobalt disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <Square className="size-4" aria-hidden />
                    {pendingAction === "stop" ? "Stopping…" : "Stop"}
                  </button>
                </div>
              </div>

              <dl className="mt-5 grid grid-cols-2 gap-x-4 gap-y-2 border-t border-linen pt-4 font-mono text-xs text-soot sm:grid-cols-3">
                <MetaItem label="Internal IP" value={vm.internalIp ?? "—"} />
                <MetaItem label="External IP" value={vm.externalIp ?? "—"} />
                <MetaItem
                  label="Cost while on"
                  value={`~$${GPU_HOURLY_COST_USD.toFixed(2)}/hr`}
                />
                <MetaItem
                  label="Last started"
                  value={formatTimestamp(vm.lastStartTimestamp)}
                />
                <MetaItem
                  label="Last stopped"
                  value={formatTimestamp(vm.lastStopTimestamp)}
                />
              </dl>

              {isRunning ? (
                <p className="mt-4 flex items-center gap-2 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800">
                  <Activity className="size-4 shrink-0" aria-hidden />
                  The GPU is billing right now. Stop it when the demo is over —
                  the game keeps working on Gemini.
                </p>
              ) : (
                <p className="mt-4 rounded-md border border-linen bg-paper px-3 py-2 text-xs text-soot">
                  GPU off: combines fall back to Gemini automatically. Starting
                  takes about a minute.
                </p>
              )}
            </>
          ) : errorMessage ? null : (
            <p className="text-sm text-soot">Loading VM status…</p>
          )}

          {errorMessage ? (
            <p className="mt-4 rounded-md border border-red-300 bg-red-50 px-3 py-2 text-xs text-red-800">
              {errorMessage}
            </p>
          ) : null}
        </section>

        <ArenaControl />

        <section className="mt-6 rounded-md border border-linen bg-surface p-5 shadow-hairline">
          <h2 className="font-display text-sm font-semibold">
            Verify in Google Cloud
          </h2>
          <ul className="mt-3 grid gap-2 text-sm">
            <ConsoleLink
              href={vm?.consoleUrl ?? buildDefaultVmConsoleUrl()}
              label="VM instance (live status, monitoring)"
            />
            <ConsoleLink
              href="https://console.cloud.google.com/run?project=nlp2026-498021"
              label="Cloud Run services (the web app)"
            />
            <ConsoleLink
              href="https://console.cloud.google.com/billing"
              label="Billing (what everything costs)"
            />
          </ul>
        </section>
      </div>
    </main>
  );
}

type ArenaModelState =
  | { state: "idle" }
  | { state: "running" }
  | { state: "done"; success: boolean; combinationsUsed: number }
  | { state: "error"; message: string };

function ArenaControl() {
  const [depth, setDepth] = useState(3);
  const [isRacing, setIsRacing] = useState(false);
  const [progress, setProgress] = useState<Record<string, ArenaModelState>>({});

  async function raceAllModels() {
    setIsRacing(true);
    setProgress(
      Object.fromEntries(
        AGENT_MODEL_OPTIONS.map((model) => [model.id, { state: "idle" }])
      )
    );

    // Sequential on purpose: parallel runs would race each other for the
    // same combiner and blur per-model latency in the reports.
    for (const model of AGENT_MODEL_OPTIONS) {
      setProgress((current) => ({
        ...current,
        [model.id]: { state: "running" }
      }));

      try {
        const response = await fetch("/api/admin/arena/run", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ depth, model: model.id })
        });
        const payload = (await response.json()) as {
          report?: AgentTestReport;
          error?: string;
        };

        if (!response.ok || !payload.report) {
          throw new Error(payload.error ?? `Run failed (${response.status})`);
        }

        const { report } = payload;
        setProgress((current) => ({
          ...current,
          [model.id]: {
            state: "done",
            success: report.success,
            combinationsUsed: report.combinationsUsed
          }
        }));
      } catch (error) {
        setProgress((current) => ({
          ...current,
          [model.id]: {
            state: "error",
            message: error instanceof Error ? error.message : "Run failed"
          }
        }));
      }
    }

    setIsRacing(false);
  }

  return (
    <section className="mt-6 rounded-md border border-linen bg-surface p-5 shadow-hairline">
      <h2 className="flex items-center gap-2 font-display text-sm font-semibold">
        <Swords className="size-4 text-accent" aria-hidden />
        LLM Arena · run today&apos;s challenge
      </h2>
      <p className="mt-1 text-xs text-soot">
        Races all models against the same daily goal, one after another. Each
        run costs real model calls; results appear immediately in the public
        Arena mode.
      </p>

      <div className="mt-4 flex flex-wrap items-end gap-3">
        <label className="grid gap-1 text-xs font-semibold text-soot">
          <span>Goal depth</span>
          <select
            className="h-10 rounded-md border border-linen bg-surface px-3 text-sm text-ink outline-none transition focus:border-cobalt"
            disabled={isRacing}
            onChange={(event) => setDepth(Number(event.target.value))}
            value={depth}
          >
            {Array.from({ length: 10 }, (_, index) => index + 1).map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          onClick={() => void raceAllModels()}
          disabled={isRacing}
          className="flex h-10 items-center gap-2 rounded-md bg-cobalt px-4 text-sm font-semibold text-white transition hover:bg-cobalt-deep disabled:cursor-not-allowed disabled:opacity-50"
        >
          <Play className="size-4" aria-hidden />
          {isRacing ? "Racing…" : `Race ${AGENT_MODEL_OPTIONS.length} models`}
        </button>
      </div>

      {Object.keys(progress).length > 0 ? (
        <ul className="mt-4 grid gap-2">
          {AGENT_MODEL_OPTIONS.map((model) => {
            const modelState = progress[model.id] ?? { state: "idle" };

            return (
              <li
                className="flex items-center justify-between gap-2 rounded-md border border-linen bg-paper px-3 py-2 text-sm"
                key={model.id}
              >
                <span className="font-medium">{getAgentModelLabel(model.id)}</span>
                <span className="font-mono text-xs text-soot">
                  {modelState.state === "idle" ? "waiting" : null}
                  {modelState.state === "running" ? "running…" : null}
                  {modelState.state === "done"
                    ? `${modelState.success ? "✓ solved" : "✗ failed"} · ${modelState.combinationsUsed} moves`
                    : null}
                  {modelState.state === "error" ? `error: ${modelState.message}` : null}
                </span>
              </li>
            );
          })}
        </ul>
      ) : null}
    </section>
  );
}

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="uppercase tracking-wide">{label}</dt>
      <dd className="mt-0.5 text-ink">{value}</dd>
    </div>
  );
}

function ConsoleLink({ href, label }: { href: string; label: string }) {
  return (
    <li>
      <a
        href={href}
        target="_blank"
        rel="noreferrer"
        className="flex items-center gap-2 rounded-md border border-linen bg-paper px-3 py-2 text-ink transition hover:border-cobalt"
      >
        <ExternalLink className="size-4 text-soot" aria-hidden />
        {label}
      </a>
    </li>
  );
}

function formatTimestamp(value?: string): string {
  if (!value) {
    return "—";
  }

  const parsed = new Date(value);

  return Number.isNaN(parsed.getTime()) ? "—" : parsed.toLocaleString();
}

function buildDefaultVmConsoleUrl(): string {
  return "https://console.cloud.google.com/compute/instances?project=nlp2026-498021";
}
