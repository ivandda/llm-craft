"use client";

import {
  requestAppConfig,
  requestCurrentSession,
  requestGuestSession,
  requestLogin,
  requestLogout,
  requestProfileUpdate,
  requestRandomGoal,
  requestRegister,
  type AuthSession
} from "@/lib/api";
import { isGuestUser } from "@/lib/guests";
import {
  COMBINER_MODEL_OPTIONS,
  DEFAULT_COMBINER_MODEL,
  isKnownCombinerModel,
  QWEN_COMBINER_MODEL
} from "@/lib/agentModels";
import { GOAL_PRESET } from "@/lib/gameModes";
import {
  FEATURED_ACHIEVEMENT_LIMIT,
  selectFeaturedAchievements
} from "@/lib/profile";
import type {
  AuthUser,
  GameMode,
  GameSnapshot,
  GoalPreset,
  UserProfile
} from "@/lib/types";
import { AgentTest } from "@/components/AgentTest";
import { CraftGame } from "@/components/CraftGame";
import { BackButton } from "@/components/BackButton";
import { ThemeToggle } from "@/components/ThemeToggle";
import {
  BadgeCheck,
  Bot,
  LogIn,
  LogOut,
  Play,
  Sparkles,
  Save,
  Target,
  UserPlus
} from "lucide-react";
import {
  type CSSProperties,
  type FormEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useState
} from "react";
import { getHueForConcept } from "@/lib/emoji";

const EMPTY_SNAPSHOT: GameSnapshot = {
  inventory: [],
  history: []
};

export function AppShell() {
  const [session, setSession] = useState<AuthSession | null>(null);
  const [selectedMode, setSelectedMode] = useState<GameMode | null>(null);
  const [goalDepth, setGoalDepth] = useState(3);
  const [selectedCombinerModel, setSelectedCombinerModel] =
    useState(DEFAULT_COMBINER_MODEL);
  const [hasHydratedCombinerModel, setHasHydratedCombinerModel] = useState(false);
  const [qwenAvailable, setQwenAvailable] = useState(false);
  const [hasLoadedConfig, setHasLoadedConfig] = useState(false);
  const [activeGoalPreset, setActiveGoalPreset] = useState<GoalPreset>(GOAL_PRESET);
  const [snapshot, setSnapshot] = useState<GameSnapshot>(EMPTY_SNAPSHOT);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isLoadingSession, setIsLoadingSession] = useState(true);
  const [isAuthPanelOpen, setIsAuthPanelOpen] = useState(false);
  const [isGeneratingGoal, setIsGeneratingGoal] = useState(false);
  const [goalGenerationMessage, setGoalGenerationMessage] = useState<string | null>(
    null
  );

  useEffect(() => {
    let cancelled = false;

    async function bootstrapSession() {
      try {
        const existingSession = await requestCurrentSession();
        const nextSession = existingSession ?? (await requestGuestSession());

        if (!cancelled) {
          setSession(nextSession);
        }
      } catch {
        // Leave session null; the auth panel becomes the fallback.
      } finally {
        if (!cancelled) {
          setIsLoadingSession(false);
        }
      }
    }

    bootstrapSession();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;

    requestAppConfig().then((config) => {
      if (!cancelled) {
        setQwenAvailable(config.qwenAvailable);
        setHasLoadedConfig(true);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!session) {
      setHasHydratedCombinerModel(false);
      return;
    }

    // Wait for the availability flag so the first-time default lands on the
    // fine-tuned Qwen model whenever it's on offer, without clobbering an
    // explicit choice the user has already saved.
    if (!hasLoadedConfig) {
      return;
    }

    const storedModel = readStoredModel(
      createCombinerModelStorageKey(session.user.id)
    );
    const defaultModel =
      qwenAvailable && isKnownCombinerModel(QWEN_COMBINER_MODEL)
        ? QWEN_COMBINER_MODEL
        : DEFAULT_COMBINER_MODEL;

    setSelectedCombinerModel(storedModel ?? defaultModel);
    setHasHydratedCombinerModel(true);
  }, [session?.user.id, hasLoadedConfig, qwenAvailable]);

  useEffect(() => {
    if (!session || !hasHydratedCombinerModel) {
      return;
    }

    window.localStorage.setItem(
      createCombinerModelStorageKey(session.user.id),
      JSON.stringify(selectedCombinerModel)
    );
  }, [hasHydratedCombinerModel, selectedCombinerModel, session]);

  async function handleLogout() {
    await requestLogout();
    setSelectedMode(null);
    setActiveGoalPreset(GOAL_PRESET);
    setSnapshot(EMPTY_SNAPSHOT);
    setIsProfileOpen(false);
    setIsAuthPanelOpen(false);

    try {
      setSession(await requestGuestSession());
    } catch {
      setSession(null);
    }
  }

  async function generateNewGoal(options?: { fresh?: boolean }): Promise<GoalPreset> {
    setIsGeneratingGoal(true);
    setGoalGenerationMessage(null);

    try {
      // The default (daily) seed gives everyone the same goal for the
      // leaderboard; "New goal" asks for a fresh random one instead.
      const nextGoal = await requestRandomGoal({
        depth: goalDepth,
        seed: options?.fresh
          ? Math.random().toString(36).slice(2, 10)
          : undefined
      });
      setActiveGoalPreset(nextGoal);
      return nextGoal;
    } catch {
      setActiveGoalPreset(GOAL_PRESET);
      setGoalGenerationMessage("Random goal unavailable. Using fallback goal.");
      return GOAL_PRESET;
    } finally {
      setIsGeneratingGoal(false);
    }
  }

  async function handleSelectMode(mode: GameMode) {
    if (mode === "sandbox") {
      setSelectedMode("sandbox");
      return;
    }

    if (mode === "agent-test") {
      setSelectedMode("agent-test");
      return;
    }

    await generateNewGoal();
    setSelectedMode("goal");
  }

  if (isLoadingSession) {
    return <LoadingScreen />;
  }

  if (!session || isAuthPanelOpen) {
    return (
      <AuthPanel
        onAuthenticated={(nextSession) => {
          setSession(nextSession);
          setIsAuthPanelOpen(false);
        }}
        onDismiss={session ? () => setIsAuthPanelOpen(false) : undefined}
      />
    );
  }

  if (isProfileOpen) {
    return (
      <ProfileView
        profile={session.profile}
        snapshot={snapshot}
        user={session.user}
        onBack={() => setIsProfileOpen(false)}
        onLogout={handleLogout}
        onProfileUpdated={setSession}
      />
    );
  }

  if (!selectedMode) {
    return (
      <ModeMenu
        user={session.user}
        goalDepth={goalDepth}
        goalGenerationMessage={goalGenerationMessage}
        isGeneratingGoal={isGeneratingGoal}
        selectedCombinerModel={selectedCombinerModel}
        onGoalDepthChange={setGoalDepth}
        onLogout={handleLogout}
        onCombinerModelChange={setSelectedCombinerModel}
        onOpenProfile={() => setIsProfileOpen(true)}
        onSelectMode={handleSelectMode}
        onSignIn={
          isGuestUser(session.user) ? () => setIsAuthPanelOpen(true) : undefined
        }
      />
    );
  }

  if (selectedMode === "agent-test") {
    return (
      <AgentTest
        goalDepth={goalDepth}
        user={session.user}
        onBackToMenu={() => setSelectedMode(null)}
        onGoalDepthChange={setGoalDepth}
        onLogout={handleLogout}
      />
    );
  }

  return (
    <CraftGame
      goalPreset={activeGoalPreset}
      goalDepth={goalDepth}
      goalGenerationMessage={goalGenerationMessage}
      selectedCombinerModel={selectedCombinerModel}
      isGeneratingGoal={isGeneratingGoal}
      mode={selectedMode}
      user={session.user}
      onBackToMenu={() => setSelectedMode(null)}
      onGenerateNewGoal={() => generateNewGoal({ fresh: true })}
      onGoalDepthChange={setGoalDepth}
      onLogout={handleLogout}
      onOpenProfile={(nextSnapshot) => {
        setSnapshot(nextSnapshot);
        setIsProfileOpen(true);
      }}
      onSnapshotChange={setSnapshot}
    />
  );
}

function LoadingScreen() {
  return (
    <main className="grid min-h-[100dvh] place-items-center bg-paper px-6 text-ink">
      <div className="rounded-md border border-linen bg-surface px-5 py-4 text-sm text-soot shadow-hairline">
        Setting up your workbench…
      </div>
    </main>
  );
}

function Wordmark({ className = "text-xl" }: { className?: string }) {
  return (
    <span className={`font-display font-bold tracking-tight ${className}`}>
      llm<span className="text-accent">·</span>craft
    </span>
  );
}

function AuthPanel({
  onAuthenticated,
  onDismiss
}: {
  onAuthenticated: (session: AuthSession) => void;
  onDismiss?: () => void;
}) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setIsSubmitting(true);
    setErrorMessage(null);

    try {
      if (mode === "register") {
        await requestRegister({ username, password, displayName });
      } else {
        await requestLogin({ username, password });
      }

      const nextSession = await requestCurrentSession();

      if (!nextSession) {
        throw new Error("Session missing after auth");
      }

      onAuthenticated(nextSession);
    } catch {
      setErrorMessage(
        mode === "register"
          ? "Could not create that user."
          : "Username or password is incorrect."
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <main className="grid min-h-[100dvh] place-items-center bg-paper px-4 py-8 text-ink">
      <form
        className="w-full max-w-sm rounded-md border border-linen bg-surface p-5 shadow-lift"
        onSubmit={handleSubmit}
      >
        <div>
          <h1>
            <Wordmark className="text-2xl" />
          </h1>
          <p className="mt-1 text-sm text-soot">
            Sign in with an account to keep your name and progress.
          </p>
        </div>

        <div className="mt-5 grid grid-cols-2 rounded-md border border-linen bg-paper p-1">
          <button
            className={`h-9 rounded text-sm font-medium ${
              mode === "login" ? "bg-surface shadow-sm" : "text-soot"
            }`}
            onClick={() => setMode("login")}
            type="button"
          >
            Log in
          </button>
          <button
            className={`h-9 rounded text-sm font-medium ${
              mode === "register" ? "bg-surface shadow-sm" : "text-soot"
            }`}
            onClick={() => setMode("register")}
            type="button"
          >
            Create account
          </button>
        </div>

        <div className="mt-5 grid gap-3">
          <TextField
            label="Username"
            onChange={setUsername}
            value={username}
          />
          {mode === "register" ? (
            <TextField
              label="Display name"
              onChange={setDisplayName}
              value={displayName}
            />
          ) : null}
          <TextField
            label="Password"
            onChange={setPassword}
            type="password"
            value={password}
          />
        </div>

        {errorMessage ? (
          <p className="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {errorMessage}
          </p>
        ) : null}

        <button
          className="mt-5 flex h-10 w-full items-center justify-center gap-2 rounded-md bg-cobalt px-3 text-sm font-semibold text-white transition hover:bg-cobalt-deep disabled:cursor-not-allowed disabled:opacity-60"
          disabled={isSubmitting}
          type="submit"
        >
          {mode === "register" ? <UserPlus size={16} /> : <LogIn size={16} />}
          {mode === "register" ? "Create account" : "Log in"}
        </button>

        {onDismiss ? (
          <button
            className="mt-3 h-10 w-full rounded-md border border-linen bg-surface px-3 text-sm font-medium text-soot transition hover:bg-paper hover:text-ink"
            onClick={onDismiss}
            type="button"
          >
            Keep playing as guest
          </button>
        ) : null}
      </form>
    </main>
  );
}

function ModeMenu({
  goalDepth,
  goalGenerationMessage,
  isGeneratingGoal,
  selectedCombinerModel,
  user,
  onGoalDepthChange,
  onCombinerModelChange,
  onLogout,
  onOpenProfile,
  onSelectMode,
  onSignIn
}: {
  goalDepth: number;
  goalGenerationMessage: string | null;
  isGeneratingGoal: boolean;
  selectedCombinerModel: string;
  user: AuthUser;
  onGoalDepthChange: (depth: number) => void;
  onCombinerModelChange: (model: string) => void;
  onLogout: () => void;
  onOpenProfile: () => void;
  onSelectMode: (mode: GameMode) => void | Promise<void>;
  onSignIn?: () => void;
}) {
  return (
    <main className="min-h-[100dvh] bg-paper px-4 py-6 text-ink">
      <div className="mx-auto flex min-h-[calc(100dvh-3rem)] max-w-6xl flex-col gap-5">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-linen pb-4">
          <div>
            <h1>
              <Wordmark className="text-3xl" />
            </h1>
            <p className="mt-1 text-sm text-soot">
              Combine two ideas. A fine-tuned model invents the result.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="hidden h-10 items-center rounded-md border border-linen bg-surface px-3 text-sm text-soot sm:flex">
              {user.displayName}
            </span>
            {onSignIn ? (
              <button
                className="flex h-10 items-center gap-2 rounded-md border border-linen bg-surface px-3 text-sm font-medium text-soot transition hover:bg-paper hover:text-ink"
                onClick={onSignIn}
                type="button"
              >
                <LogIn size={16} />
                Sign in
              </button>
            ) : null}
            <ThemeToggle />
            <IconButton label="Profile" onClick={onOpenProfile}>
              <BadgeCheck size={17} />
            </IconButton>
            <IconButton label={onSignIn ? "New guest session" : "Log out"} onClick={onLogout}>
              <LogOut size={17} />
            </IconButton>
          </div>
        </header>

        <section className="grid flex-1 gap-4 lg:grid-cols-3">
          <ModeCard
            hue={38}
            icon={<Sparkles size={22} />}
            modeLabel="Mix anything with anything, no rules"
            previewElements={["💧", "🔥", "🌍", "💨"]}
            resultElement="✨"
            label="Sandbox"
            onClick={() => onSelectMode("sandbox")}
          />
          <ModeCard
            hue={222}
            disabled={isGeneratingGoal}
            icon={<Target size={22} />}
            modeLabel={`Reach a target element in as few combinations as you can — depth ${goalDepth}`}
            previewElements={["🧭", "🛤️", "📍", "🎯"]}
            resultElement="🏁"
            label={isGeneratingGoal ? "Generating…" : "Goal"}
            onClick={() => onSelectMode("goal")}
          />
          <ModeCard
            hue={152}
            compactPreview
            icon={<Bot size={22} />}
            modeLabel={`Watch LLMs race toward the daily goal and compare their results — depth ${goalDepth}`}
            previewElements={["🤖"]}
            resultElement="🤖"
            label="LLM Arena"
            onClick={() => onSelectMode("agent-test")}
          />
        </section>
        <div className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-md border border-linen bg-surface px-4 py-3 shadow-hairline">
          <label className="flex items-center gap-3">
            <span className="font-mono text-xs font-medium uppercase tracking-wider text-soot">
              Goal depth
            </span>
            <select
              className="h-9 rounded-md border border-linen bg-surface px-3 font-mono text-sm outline-none transition focus:border-cobalt"
              disabled={isGeneratingGoal}
              onChange={(event) => onGoalDepthChange(Number(event.target.value))}
              value={goalDepth}
            >
              {Array.from({ length: 10 }, (_, index) => index + 1).map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </label>
          <label className="flex min-w-[240px] items-center gap-3">
            <span className="font-mono text-xs font-medium uppercase tracking-wider text-soot">
              Model
            </span>
            <select
              className="h-9 min-w-0 flex-1 rounded-md border border-linen bg-surface px-3 font-mono text-sm outline-none transition focus:border-cobalt"
              onChange={(event) => onCombinerModelChange(event.target.value)}
              value={selectedCombinerModel}
            >
              {COMBINER_MODEL_OPTIONS.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </select>
          </label>
          {goalGenerationMessage ? (
            <p className="text-sm text-amber-700">{goalGenerationMessage}</p>
          ) : null}
        </div>
      </div>
    </main>
  );
}

function createCombinerModelStorageKey(userId: string): string {
  return `llm-craft.v2.${normalizeStorageSegment(userId)}.combinerModel`;
}

// Returns the user's saved combiner model, or null when they've never picked
// one — the caller decides the default so it can factor in model availability.
function readStoredModel(key: string): string | null {
  const rawValue = window.localStorage.getItem(key);

  if (!rawValue) {
    return null;
  }

  try {
    const model = JSON.parse(rawValue) as unknown;
    return isKnownCombinerModel(model) ? model : null;
  } catch {
    return null;
  }
}

function normalizeStorageSegment(value: string): string {
  return value.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, "-") || "anonymous";
}

function ModeCard({
  hue,
  disabled = false,
  compactPreview = false,
  icon,
  label,
  modeLabel,
  onClick,
  previewElements,
  resultElement
}: {
  hue: number;
  disabled?: boolean;
  compactPreview?: boolean;
  icon: ReactNode;
  label: string;
  modeLabel: string;
  onClick: () => void;
  previewElements: string[];
  resultElement: string;
}) {
  return (
    <button
      className="element-card group relative min-h-[280px] overflow-hidden rounded-md border-2 p-5 text-left shadow-hairline transition duration-200 hover:-translate-y-1 hover:shadow-lift active:translate-y-0 sm:min-h-[360px]"
      disabled={disabled}
      onClick={onClick}
      style={{ "--el-hue": hue } as CSSProperties}
      type="button"
    >
      <div className="relative z-10 flex h-full flex-col justify-between gap-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="font-display text-4xl font-black tracking-tight sm:text-5xl">
              {label}
            </h2>
            <p className="mt-3 max-w-xs text-base font-medium text-soot">
              {modeLabel}
            </p>
          </div>
          <span className="grid size-12 place-items-center rounded-md border border-linen bg-surface/85 text-ink transition group-hover:rotate-6 group-hover:scale-110">
            {icon}
          </span>
        </div>

        <div className="grid gap-5">
          {compactPreview ? (
            <div className="grid place-items-center py-6">
              <span className="grid size-36 place-items-center rounded-md border border-linen bg-surface text-7xl shadow-md transition duration-200 group-hover:-translate-y-2 group-hover:rotate-3">
                {resultElement}
              </span>
            </div>
          ) : (
            <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3">
              <div className="grid grid-cols-2 gap-2">
                {previewElements.map((element, index) => (
                  <span
                    className="grid aspect-square place-items-center rounded-md border border-linen bg-surface/90 text-3xl shadow-sm transition duration-200 group-hover:-translate-y-1 group-hover:rotate-2"
                    key={`${element}-${index}`}
                  >
                    {element}
                  </span>
                ))}
              </div>
              <span className="grid size-12 place-items-center rounded-md border border-linen bg-surface/85 text-soot transition group-hover:scale-110">
                <Play size={18} />
              </span>
              <span className="grid aspect-square place-items-center rounded-md border border-linen bg-surface text-5xl shadow-md transition duration-200 group-hover:-translate-y-2 group-hover:rotate-3">
                {resultElement}
              </span>
            </div>
          )}

          <div className="flex items-center justify-between gap-3 rounded-md border border-linen bg-surface/85 px-4 py-3">
            <span className="text-sm font-semibold text-ink">Play</span>
            <span className="grid size-8 place-items-center rounded-md bg-cobalt text-white transition group-hover:translate-x-1">
              <Play size={15} />
            </span>
          </div>
        </div>
      </div>

      {!compactPreview ? (
        <div className="pointer-events-none absolute -bottom-10 -right-8 grid grid-cols-3 gap-2 opacity-30 transition duration-200 group-hover:-translate-y-2 group-hover:opacity-45">
          {previewElements.concat(resultElement).map((element, index) => (
            <span
              className="grid size-12 place-items-center rounded-md border border-linen bg-surface text-xl"
              key={`${element}-ghost-${index}`}
            >
              {element}
            </span>
          ))}
        </div>
      ) : null}
    </button>
  );
}
function ProfileView({
  profile,
  snapshot,
  user,
  onBack,
  onLogout,
  onProfileUpdated
}: {
  profile: UserProfile;
  snapshot: GameSnapshot;
  user: AuthUser;
  onBack: () => void;
  onLogout: () => void;
  onProfileUpdated: (session: AuthSession) => void;
}) {
  const [selectedIds, setSelectedIds] = useState(
    profile.featuredAchievements.map((achievement) => achievement.elementId)
  );
  const [displayName, setDisplayName] = useState(profile.displayName);
  const [isSaving, setIsSaving] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const selectedAchievements = useMemo(
    () => {
      const achievements = selectFeaturedAchievements(snapshot.inventory, selectedIds);

      if (achievements.length > 0 || snapshot.inventory.length > 0) {
        return achievements;
      }

      return profile.featuredAchievements.filter((achievement) =>
        selectedIds.includes(achievement.elementId)
      );
    },
    [profile.featuredAchievements, selectedIds, snapshot.inventory]
  );

  function toggleAchievement(elementId: string) {
    setSelectedIds((currentIds) => {
      if (currentIds.includes(elementId)) {
        return currentIds.filter((id) => id !== elementId);
      }

      if (currentIds.length >= FEATURED_ACHIEVEMENT_LIMIT) {
        return currentIds;
      }

      return [...currentIds, elementId];
    });
  }

  async function saveProfile() {
    setIsSaving(true);
    setMessage(null);

    try {
      const nextSession = await requestProfileUpdate({
        displayName,
        featuredAchievements: selectedAchievements
      });
      onProfileUpdated(nextSession);
      setMessage("Profile saved.");
    } catch {
      setMessage("Profile is local until the mock server session is available.");
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <main className="min-h-[100dvh] bg-paper px-4 py-6 text-ink">
      <div className="mx-auto grid max-w-5xl gap-5">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-linen pb-4">
          <div>
            <h1 className="font-display text-xl font-bold tracking-tight">Profile</h1>
            <p className="mt-1 font-mono text-sm text-soot">@{user.username}</p>
          </div>
          <div className="flex items-center gap-2">
            <BackButton onClick={onBack} />
            <ThemeToggle />
            <IconButton label="Log out" onClick={onLogout}>
              <LogOut size={17} />
            </IconButton>
          </div>
        </header>

        <section className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
          <div className="rounded-md border border-linen bg-surface p-5 shadow-hairline">
            <TextField
              label="Display name"
              onChange={setDisplayName}
              value={displayName}
            />
            <div className="mt-5 grid grid-cols-2 gap-3">
              <MetricCard label="Discovered" value={snapshot.inventory.length} />
              <MetricCard label="Recent" value={snapshot.history.length} />
            </div>

            <div className="mt-5">
              <h2 className="text-sm font-semibold">Showcase</h2>
              <div className="mt-3 grid gap-2">
                {selectedAchievements.length === 0 ? (
                  <p className="rounded-md border border-dashed border-linen p-3 text-sm text-soot">
                    Feature up to six discoveries here.
                  </p>
                ) : (
                  selectedAchievements.map((achievement) => (
                    <div
                      className="element-card flex items-center gap-2 rounded-md border px-3 py-2 text-sm capitalize"
                      key={achievement.elementId}
                      style={{ "--el-hue": getHueForConcept(achievement.name) } as CSSProperties}
                    >
                      <span>{achievement.emoji ?? "·"}</span>
                      <span className="truncate">{achievement.name}</span>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="mt-5">
              <h2 className="text-sm font-semibold">Recent</h2>
              <div className="mt-3 grid gap-2">
                {snapshot.history.length === 0 ? (
                  <p className="rounded-md border border-dashed border-linen p-3 text-sm text-soot">
                    No recent discoveries in this mode.
                  </p>
                ) : (
                  snapshot.history.slice(0, 5).map((item) => (
                    <div
                      className="element-card rounded-md border px-3 py-2 text-sm"
                      key={item.id}
                      style={{ "--el-hue": getHueForConcept(item.output.name) } as CSSProperties}
                    >
                      <span className="capitalize">{item.output.name}</span>
                    </div>
                  ))
                )}
              </div>
            </div>

            <button
              className="mt-5 flex h-10 w-full items-center justify-center gap-2 rounded-md bg-cobalt px-3 text-sm font-semibold text-white transition hover:bg-cobalt-deep disabled:cursor-not-allowed disabled:opacity-60"
              disabled={isSaving}
              onClick={saveProfile}
              type="button"
            >
              <Save size={16} />
              Save profile
            </button>
            {message ? <p className="mt-3 text-sm text-soot">{message}</p> : null}
          </div>

          <div className="rounded-md border border-linen bg-surface p-5 shadow-hairline">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-sm font-semibold">Achievements</h2>
              <span className="font-mono text-sm text-soot">
                {selectedIds.length}/{FEATURED_ACHIEVEMENT_LIMIT}
              </span>
            </div>

            <div className="mt-4 grid max-h-[60vh] grid-cols-2 gap-2 overflow-y-auto pr-1 md:grid-cols-3">
              {snapshot.inventory.length === 0 ? (
                <p className="col-span-full rounded-md border border-dashed border-linen p-4 text-sm text-soot">
                  Enter a mode to load this user's inventory.
                </p>
              ) : (
                snapshot.inventory.map((element) => {
                  const isSelected = selectedIds.includes(element.id);

                  return (
                    <button
                      className={`element-card min-h-20 rounded-md border px-3 py-2 text-center transition hover:shadow-hairline ${
                        isSelected ? "ring-2 ring-cobalt" : ""
                      }`}
                      key={element.id}
                      onClick={() => toggleAchievement(element.id)}
                      style={{ "--el-hue": getHueForConcept(element.name) } as CSSProperties}
                      type="button"
                    >
                      <span className="text-2xl">{element.emoji ?? "·"}</span>
                      <span className="mt-1 block truncate text-xs font-medium capitalize">
                        {element.name}
                      </span>
                    </button>
                  );
                })
              )}
            </div>
          </div>
        </section>
      </div>
    </main>
  );
}

function TextField({
  label,
  onChange,
  type = "text",
  value
}: {
  label: string;
  onChange: (value: string) => void;
  type?: string;
  value: string;
}) {
  return (
    <label className="grid gap-1 text-sm font-medium text-ink">
      <span>{label}</span>
      <input
        className="h-10 rounded-md border border-linen bg-paper px-3 text-sm outline-none transition focus:border-cobalt focus:bg-surface"
        onChange={(event) => onChange(event.target.value)}
        type={type}
        value={value}
      />
    </label>
  );
}

function IconButton({
  children,
  label,
  onClick
}: {
  children: ReactNode;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      aria-label={label}
      className="grid size-10 place-items-center rounded-md border border-linen bg-surface text-soot transition hover:bg-paper hover:text-ink"
      onClick={onClick}
      title={label}
      type="button"
    >
      {children}
    </button>
  );
}

function MetricCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-md border border-linen bg-paper p-3">
      <p className="font-mono text-2xl font-semibold">{value}</p>
      <p className="mt-1 font-mono text-xs uppercase tracking-wider text-soot">{label}</p>
    </div>
  );
}
