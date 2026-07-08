"use client";

import {
  requestCurrentSession,
  requestLogin,
  requestLogout,
  requestProfileUpdate,
  requestRandomGoal,
  requestRegister,
  type AuthSession
} from "@/lib/api";
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
import {
  ArrowLeft,
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
  type FormEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useState
} from "react";

const EMPTY_SNAPSHOT: GameSnapshot = {
  inventory: [],
  history: []
};

export function AppShell() {
  const [session, setSession] = useState<AuthSession | null>(null);
  const [selectedMode, setSelectedMode] = useState<GameMode | null>(null);
  const [goalDepth, setGoalDepth] = useState(3);
  const [activeGoalPreset, setActiveGoalPreset] = useState<GoalPreset>(GOAL_PRESET);
  const [snapshot, setSnapshot] = useState<GameSnapshot>(EMPTY_SNAPSHOT);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [isLoadingSession, setIsLoadingSession] = useState(true);
  const [isGeneratingGoal, setIsGeneratingGoal] = useState(false);
  const [goalGenerationMessage, setGoalGenerationMessage] = useState<string | null>(
    null
  );

  useEffect(() => {
    requestCurrentSession()
      .then((nextSession) => setSession(nextSession))
      .finally(() => setIsLoadingSession(false));
  }, []);

  async function handleLogout() {
    await requestLogout();
    setSession(null);
    setSelectedMode(null);
    setActiveGoalPreset(GOAL_PRESET);
    setSnapshot(EMPTY_SNAPSHOT);
    setIsProfileOpen(false);
  }

  async function generateNewGoal(): Promise<GoalPreset> {
    setIsGeneratingGoal(true);
    setGoalGenerationMessage(null);

    try {
      const nextGoal = await requestRandomGoal({ depth: goalDepth });
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

  if (!session) {
    return <AuthPanel onAuthenticated={setSession} />;
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
        onGoalDepthChange={setGoalDepth}
        onLogout={handleLogout}
        onOpenProfile={() => setIsProfileOpen(true)}
        onSelectMode={handleSelectMode}
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
      isGeneratingGoal={isGeneratingGoal}
      mode={selectedMode}
      user={session.user}
      onBackToMenu={() => setSelectedMode(null)}
      onGenerateNewGoal={generateNewGoal}
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
    <main className="grid min-h-screen place-items-center bg-stone-100 px-6 text-zinc-950">
      <div className="rounded-md border border-zinc-200 bg-white px-5 py-4 text-sm text-zinc-600 shadow-hairline">
        Loading session.
      </div>
    </main>
  );
}

function AuthPanel({ onAuthenticated }: { onAuthenticated: (session: AuthSession) => void }) {
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
    <main className="grid min-h-screen place-items-center bg-stone-100 px-4 py-8 text-zinc-950">
      <form
        className="w-full max-w-sm rounded-md border border-zinc-200 bg-white p-5 shadow-hairline"
        onSubmit={handleSubmit}
      >
        <div>
          <h1 className="text-xl font-semibold tracking-normal">llm-craft</h1>
          <p className="mt-1 text-sm text-zinc-500">Mock user session</p>
        </div>

        <div className="mt-5 grid grid-cols-2 rounded-md border border-zinc-200 bg-zinc-50 p-1">
          <button
            className={`h-9 rounded text-sm font-medium ${
              mode === "login" ? "bg-white shadow-sm" : "text-zinc-500"
            }`}
            onClick={() => setMode("login")}
            type="button"
          >
            Login
          </button>
          <button
            className={`h-9 rounded text-sm font-medium ${
              mode === "register" ? "bg-white shadow-sm" : "text-zinc-500"
            }`}
            onClick={() => setMode("register")}
            type="button"
          >
            Register
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
          className="mt-5 flex h-10 w-full items-center justify-center gap-2 rounded-md bg-zinc-950 px-3 text-sm font-semibold text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-60"
          disabled={isSubmitting}
          type="submit"
        >
          {mode === "register" ? <UserPlus size={16} /> : <LogIn size={16} />}
          {mode === "register" ? "Create user" : "Log in"}
        </button>
      </form>
    </main>
  );
}

function ModeMenu({
  goalDepth,
  goalGenerationMessage,
  isGeneratingGoal,
  user,
  onGoalDepthChange,
  onLogout,
  onOpenProfile,
  onSelectMode
}: {
  goalDepth: number;
  goalGenerationMessage: string | null;
  isGeneratingGoal: boolean;
  user: AuthUser;
  onGoalDepthChange: (depth: number) => void;
  onLogout: () => void;
  onOpenProfile: () => void;
  onSelectMode: (mode: GameMode) => void | Promise<void>;
}) {
  return (
    <main className="min-h-screen bg-[#f6f2e8] px-4 py-6 text-zinc-950">
      <div className="mx-auto flex min-h-[calc(100vh-3rem)] max-w-6xl flex-col gap-5">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-200 pb-4">
          <div>
            <h1 className="text-xl font-semibold tracking-normal">llm-craft</h1>
            <p className="mt-1 text-sm text-zinc-500">{user.displayName}</p>
          </div>
          <div className="flex items-center gap-2">
            <IconButton label="Profile" onClick={onOpenProfile}>
              <BadgeCheck size={17} />
            </IconButton>
            <IconButton label="Log out" onClick={onLogout}>
              <LogOut size={17} />
            </IconButton>
          </div>
        </header>

        <section className="grid flex-1 gap-4 lg:grid-cols-3">
          <ModeCard
            accentClassName="border-amber-300 bg-amber-50 hover:border-amber-400 hover:bg-amber-100"
            icon={<Sparkles size={22} />}
            modeLabel="Free mix"
            previewElements={["💧", "🔥", "🌍", "💨"]}
            resultElement="✨"
            label="Sandbox"
            onClick={() => onSelectMode("sandbox")}
          />
          <ModeCard
            accentClassName="border-sky-300 bg-sky-50 hover:border-sky-400 hover:bg-sky-100"
            disabled={isGeneratingGoal}
            icon={<Target size={22} />}
            modeLabel={`Random goal at depth ${goalDepth}`}
            previewElements={["🧭", "🛤️", "📍", "🎯"]}
            resultElement="🏁"
            label={isGeneratingGoal ? "Generating" : "Goal"}
            onClick={() => onSelectMode("goal")}
          />
          <ModeCard
            accentClassName="border-emerald-300 bg-emerald-50 hover:border-emerald-400 hover:bg-emerald-100"
            compactPreview
            icon={<Bot size={22} />}
            modeLabel={`Agent run at depth ${goalDepth}`}
            previewElements={["🤖"]}
            resultElement="🤖"
            label="Agent Test"
            onClick={() => onSelectMode("agent-test")}
          />
        </section>
        <div className="flex flex-wrap items-center gap-3 rounded-md border border-sky-200 bg-white/80 px-4 py-3 shadow-hairline">
          <label className="flex items-center gap-3 text-sm font-semibold text-zinc-700">
            <span>Goal depth</span>
            <select
              className="h-9 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none transition focus:border-zinc-500"
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
          {goalGenerationMessage ? (
            <p className="text-sm text-amber-700">{goalGenerationMessage}</p>
          ) : null}
        </div>
      </div>
    </main>
  );
}

function ModeCard({
  accentClassName,
  disabled = false,
  compactPreview = false,
  icon,
  label,
  modeLabel,
  onClick,
  previewElements,
  resultElement
}: {
  accentClassName: string;
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
      className={`group relative min-h-[360px] overflow-hidden rounded-md border-2 p-5 text-left shadow-hairline transition duration-200 hover:-translate-y-1 hover:shadow-xl active:translate-y-0 ${accentClassName}`}
      disabled={disabled}
      onClick={onClick}
      type="button"
    >
      <div className="relative z-10 flex h-full flex-col justify-between gap-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h2 className="text-4xl font-black tracking-normal sm:text-5xl">
              {label}
            </h2>
            <p className="mt-3 max-w-xs text-base font-semibold text-zinc-600">
              {modeLabel}
            </p>
          </div>
          <span className="grid size-12 place-items-center rounded-md border border-zinc-300 bg-white/80 text-zinc-700 transition group-hover:rotate-6 group-hover:scale-110">
            {icon}
          </span>
        </div>

        <div className="grid gap-5">
          {compactPreview ? (
            <div className="grid place-items-center py-6">
              <span className="grid size-36 place-items-center rounded-md border border-emerald-300 bg-white text-7xl shadow-md transition duration-200 group-hover:-translate-y-2 group-hover:rotate-3">
                {resultElement}
              </span>
            </div>
          ) : (
            <div className="grid grid-cols-[1fr_auto_1fr] items-center gap-3">
              <div className="grid grid-cols-2 gap-2">
                {previewElements.map((element, index) => (
                  <span
                    className="grid aspect-square place-items-center rounded-md border border-zinc-300 bg-white/85 text-3xl shadow-sm transition duration-200 group-hover:-translate-y-1 group-hover:rotate-2"
                    key={`${element}-${index}`}
                  >
                    {element}
                  </span>
                ))}
              </div>
              <span className="grid size-12 place-items-center rounded-md border border-zinc-300 bg-white/80 text-zinc-600 transition group-hover:scale-110">
                <Play size={18} />
              </span>
              <span className="grid aspect-square place-items-center rounded-md border border-zinc-300 bg-white text-5xl shadow-md transition duration-200 group-hover:-translate-y-2 group-hover:rotate-3">
                {resultElement}
              </span>
            </div>
          )}

          <div className="flex items-center justify-between gap-3 rounded-md border border-zinc-300 bg-white/80 px-4 py-3">
            <span className="text-sm font-semibold text-zinc-700">Play</span>
            <span className="grid size-8 place-items-center rounded-md bg-zinc-950 text-white transition group-hover:translate-x-1">
              <Play size={15} />
            </span>
          </div>
        </div>
      </div>

      {!compactPreview ? (
        <div className="pointer-events-none absolute -bottom-10 -right-8 grid grid-cols-3 gap-2 opacity-30 transition duration-200 group-hover:-translate-y-2 group-hover:opacity-45">
          {previewElements.concat(resultElement).map((element, index) => (
            <span
              className="grid size-12 place-items-center rounded-md border border-zinc-300 bg-white text-xl"
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
    <main className="min-h-screen bg-stone-100 px-4 py-6 text-zinc-950">
      <div className="mx-auto grid max-w-5xl gap-5">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-zinc-200 pb-4">
          <div>
            <h1 className="text-xl font-semibold tracking-normal">Profile</h1>
            <p className="mt-1 text-sm text-zinc-500">@{user.username}</p>
          </div>
          <div className="flex items-center gap-2">
            <IconButton label="Back" onClick={onBack}>
              <ArrowLeft size={17} />
            </IconButton>
            <IconButton label="Log out" onClick={onLogout}>
              <LogOut size={17} />
            </IconButton>
          </div>
        </header>

        <section className="grid gap-4 lg:grid-cols-[320px_minmax(0,1fr)]">
          <div className="rounded-md border border-zinc-200 bg-white p-5 shadow-hairline">
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
                  <p className="rounded-md border border-dashed border-zinc-200 p-3 text-sm text-zinc-500">
                    No featured achievements yet.
                  </p>
                ) : (
                  selectedAchievements.map((achievement) => (
                    <div
                      className="flex items-center gap-2 rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm capitalize"
                      key={achievement.elementId}
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
                  <p className="rounded-md border border-dashed border-zinc-200 p-3 text-sm text-zinc-500">
                    No recent discoveries in this mode.
                  </p>
                ) : (
                  snapshot.history.slice(0, 5).map((item) => (
                    <div
                      className="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm"
                      key={item.id}
                    >
                      <span className="capitalize">{item.output.name}</span>
                    </div>
                  ))
                )}
              </div>
            </div>

            <button
              className="mt-5 flex h-10 w-full items-center justify-center gap-2 rounded-md bg-zinc-950 px-3 text-sm font-semibold text-white transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-60"
              disabled={isSaving}
              onClick={saveProfile}
              type="button"
            >
              <Save size={16} />
              Save profile
            </button>
            {message ? <p className="mt-3 text-sm text-zinc-500">{message}</p> : null}
          </div>

          <div className="rounded-md border border-zinc-200 bg-white p-5 shadow-hairline">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-sm font-semibold">Achievements</h2>
              <span className="text-sm text-zinc-500">
                {selectedIds.length}/{FEATURED_ACHIEVEMENT_LIMIT}
              </span>
            </div>

            <div className="mt-4 grid max-h-[60vh] grid-cols-2 gap-2 overflow-y-auto pr-1 md:grid-cols-3">
              {snapshot.inventory.length === 0 ? (
                <p className="col-span-full rounded-md border border-dashed border-zinc-200 p-4 text-sm text-zinc-500">
                  Enter a mode to load this user's inventory.
                </p>
              ) : (
                snapshot.inventory.map((element) => {
                  const isSelected = selectedIds.includes(element.id);

                  return (
                    <button
                      className={`min-h-20 rounded-md border px-3 py-2 text-center transition ${
                        isSelected
                          ? "border-zinc-950 bg-zinc-100"
                          : "border-zinc-200 bg-zinc-50 hover:bg-white"
                      }`}
                      key={element.id}
                      onClick={() => toggleAchievement(element.id)}
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
    <label className="grid gap-1 text-sm font-medium text-zinc-700">
      <span>{label}</span>
      <input
        className="h-10 rounded-md border border-zinc-200 bg-zinc-50 px-3 text-sm outline-none transition focus:border-zinc-400 focus:bg-white"
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
      className="grid size-10 place-items-center rounded-md border border-zinc-200 bg-white text-zinc-600 transition hover:bg-zinc-50 hover:text-zinc-950"
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
    <div className="rounded-md border border-zinc-200 bg-zinc-50 p-3">
      <p className="text-2xl font-semibold">{value}</p>
      <p className="mt-1 text-xs text-zinc-500">{label}</p>
    </div>
  );
}
