"use client";

import {
  requestCombination,
  requestDpoPreference,
  requestLeaderboard,
  requestLeaderboardSubmission
} from "@/lib/api";
import { getCombinerModelLabel } from "@/lib/agentModels";
import { mergeInventory } from "@/lib/craft";
import { selectDpoCandidates } from "@/lib/dpo";
import { getInitialInventoryForMode } from "@/lib/gameModes";
import { createGameStorageKey } from "@/lib/gameStorage";
import type {
  AuthUser,
  CombineResponse,
  ElementToken,
  GameMode,
  GameSnapshot,
  GoalPreset,
  LeaderboardEntry,
  RecipeHistoryItem
} from "@/lib/types";
import {
  Brush,
  LogOut,
  Menu,
  Moon,
  RotateCcw,
  Search,
  Sparkles,
  Sun,
  Target,
  Trophy,
  UserCircle
} from "lucide-react";
import {
  forwardRef,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent
} from "react";

const TOKEN_WIDTH = 132;
const TOKEN_HEIGHT = 48;
const RESULT_OFFSET = 18;

type CraftGameProps = {
  user: AuthUser;
  mode: GameMode;
  goalPreset?: GoalPreset;
  goalDepth: number;
  selectedCombinerModel: string;
  isGeneratingGoal: boolean;
  goalGenerationMessage: string | null;
  onBackToMenu: () => void;
  onGenerateNewGoal: () => Promise<GoalPreset>;
  onGoalDepthChange: (depth: number) => void;
  onLogout: () => void;
  onOpenProfile: (snapshot: GameSnapshot) => void;
  onSnapshotChange?: (snapshot: GameSnapshot) => void;
};

type BoardElement = ElementToken & {
  instanceId: string;
  x: number;
  y: number;
  zIndex: number;
};

type DragState = {
  kind: "inventory" | "board";
  element: ElementToken;
  instanceId?: string;
  offsetX: number;
  offsetY: number;
  pointerX: number;
  pointerY: number;
};

type Point = {
  x: number;
  y: number;
};

type GoalCompletion = {
  combinationsUsed: number;
  isSaving: boolean;
  errorMessage?: string;
};

type PendingDpoChoice = {
  firstInput: ElementToken;
  secondInput: ElementToken;
  response: CombineResponse;
  candidates: ElementToken[];
  clientX: number;
  clientY: number;
  usedInstanceIds: string[];
  combinationIndex: number;
  isSaving: boolean;
  errorMessage?: string;
};

export function CraftGame({
  user,
  mode,
  goalPreset,
  goalDepth,
  selectedCombinerModel,
  isGeneratingGoal,
  goalGenerationMessage,
  onBackToMenu,
  onGenerateNewGoal,
  onGoalDepthChange,
  onLogout,
  onOpenProfile,
  onSnapshotChange
}: CraftGameProps) {
  const boardRef = useRef<HTMLDivElement | null>(null);
  const blackHoleRef = useRef<HTMLDivElement | null>(null);
  const sweepTimeoutRef = useRef<number | null>(null);
  const initialInventory = useMemo(
    () =>
      mode === "goal" && goalPreset
        ? goalPreset.initialInventory
        : getInitialInventoryForMode(mode),
    [goalPreset, mode]
  );
  const storageKeys = useMemo(
    () => ({
      inventory: createGameStorageKey(user.id, mode, "inventory"),
      history: createGameStorageKey(user.id, mode, "history"),
      board: createGameStorageKey(user.id, mode, "board"),
      darkMode: createGameStorageKey(user.id, mode, "darkMode"),
      consumeInputs: createGameStorageKey(user.id, mode, "consumeInputs"),
      dpoTestMode: createGameStorageKey(user.id, mode, "dpoTestMode")
    }),
    [mode, user.id]
  );
  const [inventory, setInventory] = useState<ElementToken[]>(initialInventory);
  const [boardElements, setBoardElements] = useState<BoardElement[]>(() =>
    createInitialBoard(initialInventory)
  );
  const [history, setHistory] = useState<RecipeHistoryItem[]>([]);
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<CombineResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isCombining, setIsCombining] = useState(false);
  const [dragState, setDragState] = useState<DragState | null>(null);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [consumeInputsOnCombine, setConsumeInputsOnCombine] = useState(true);
  const [isDpoTestMode, setIsDpoTestMode] = useState(false);
  const [pendingDpoChoice, setPendingDpoChoice] = useState<PendingDpoChoice | null>(
    null
  );
  const [isSweeping, setIsSweeping] = useState(false);
  const [sweepTarget, setSweepTarget] = useState<Point | null>(null);
  const [hasHydratedStorage, setHasHydratedStorage] = useState(false);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [goalCompletion, setGoalCompletion] = useState<GoalCompletion | null>(null);

  useEffect(() => {
    setHasHydratedStorage(false);
    const shouldRestoreState = mode !== "goal";
    const storedInventory = shouldRestoreState
      ? readStoredValue(storageKeys.inventory, initialInventory)
      : initialInventory;

    setInventory(storedInventory);
    setHistory(shouldRestoreState ? readStoredValue(storageKeys.history, []) : []);
    setBoardElements(
      shouldRestoreState
        ? readStoredValue(storageKeys.board, createInitialBoard(storedInventory))
        : createInitialBoard(storedInventory)
    );
    setIsDarkMode(readStoredValue(storageKeys.darkMode, false));
    setConsumeInputsOnCombine(readStoredValue(storageKeys.consumeInputs, true));
    setIsDpoTestMode(readStoredValue(storageKeys.dpoTestMode, false));
    setPendingDpoChoice(null);
    setResult(null);
    setErrorMessage(null);
    setQuery("");
    setDragState(null);
    setGoalCompletion(null);
    setHasHydratedStorage(true);
  }, [goalPreset?.id, initialInventory, mode, storageKeys]);

  useEffect(() => {
    if (mode !== "goal" || !goalPreset) {
      setLeaderboard([]);
      return;
    }

    requestLeaderboard(goalPreset.id)
      .then(setLeaderboard)
      .catch(() => setLeaderboard([]));
  }, [goalPreset, mode]);

  useEffect(() => {
    if (hasHydratedStorage) {
      window.localStorage.setItem(storageKeys.inventory, JSON.stringify(inventory));
    }
  }, [hasHydratedStorage, inventory, storageKeys.inventory]);

  useEffect(() => {
    if (hasHydratedStorage) {
      window.localStorage.setItem(storageKeys.board, JSON.stringify(boardElements));
    }
  }, [boardElements, hasHydratedStorage, storageKeys.board]);

  useEffect(() => {
    if (hasHydratedStorage) {
      window.localStorage.setItem(storageKeys.history, JSON.stringify(history));
    }
  }, [hasHydratedStorage, history, storageKeys.history]);

  useEffect(() => {
    if (hasHydratedStorage) {
      window.localStorage.setItem(storageKeys.darkMode, JSON.stringify(isDarkMode));
    }
  }, [hasHydratedStorage, isDarkMode, storageKeys.darkMode]);

  useEffect(() => {
    if (hasHydratedStorage) {
      window.localStorage.setItem(
        storageKeys.consumeInputs,
        JSON.stringify(consumeInputsOnCombine)
      );
    }
  }, [consumeInputsOnCombine, hasHydratedStorage, storageKeys.consumeInputs]);

  useEffect(() => {
    if (hasHydratedStorage) {
      window.localStorage.setItem(
        storageKeys.dpoTestMode,
        JSON.stringify(isDpoTestMode)
      );
    }
  }, [hasHydratedStorage, isDpoTestMode, storageKeys.dpoTestMode]);

  useEffect(() => {
    onSnapshotChange?.({ inventory, history });
  }, [history, inventory, onSnapshotChange]);

  useEffect(() => {
    return () => {
      if (sweepTimeoutRef.current !== null) {
        window.clearTimeout(sweepTimeoutRef.current);
      }
    };
  }, []);

  const filteredInventory = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();

    if (!normalizedQuery) {
      return inventory;
    }

    return inventory.filter((element) =>
      element.name.toLowerCase().includes(normalizedQuery)
    );
  }, [inventory, query]);

  function resetGame(nextInitialInventory = initialInventory) {
    setInventory(nextInitialInventory);
    setBoardElements(createInitialBoard(nextInitialInventory));
    setHistory([]);
    setQuery("");
    setResult(null);
    setErrorMessage(null);
    setDragState(null);
    setPendingDpoChoice(null);
    setGoalCompletion(null);
  }

  async function generateNewGoal() {
    const nextGoal = await onGenerateNewGoal();
    resetGame(nextGoal.initialInventory);
  }

  function clearSandboxWithSweep() {
    if (boardElements.length === 0 || isSweeping) {
      return;
    }

    setDragState(null);
    setErrorMessage(null);
    setSweepTarget(getBlackHoleCenterInBoard());
    setIsSweeping(true);

    sweepTimeoutRef.current = window.setTimeout(() => {
      setBoardElements([]);
      setIsSweeping(false);
      setSweepTarget(null);
      sweepTimeoutRef.current = null;
    }, 700);
  }

  function addElementToBoard(
    element: ElementToken,
    clientX: number,
    clientY: number
  ) {
    const position = getBoardPosition(clientX, clientY);

    if (!position) {
      return;
    }

    setBoardElements((currentElements) => [
      ...currentElements,
      createBoardElement(element, position.x, position.y, getNextZIndex(currentElements))
    ]);
  }

  async function combineElementsOnBoard(
    inputA: ElementToken,
    inputB: ElementToken,
    clientX: number,
    clientY: number,
    usedInstanceIds: string[]
  ) {
    if (isCombining) {
      return;
    }

    setIsCombining(true);
    setErrorMessage(null);

    try {
      const firstInput = toElementToken(inputA);
      const secondInput = toElementToken(inputB);
      const response = await requestCombination({
        inputA: firstInput,
        inputB: secondInput,
        inventory,
        model: selectedCombinerModel
      });
      const combinationsUsed = history.length + 1;
      const dpoCandidates = isDpoTestMode
        ? selectDpoCandidates(response.knownOutputs)
        : [];

      if (dpoCandidates.length >= 2) {
        setPendingDpoChoice({
          firstInput,
          secondInput,
          response,
          candidates: dpoCandidates,
          clientX,
          clientY,
          usedInstanceIds,
          combinationIndex: combinationsUsed,
          isSaving: false
        });
        return;
      }

      applyCombinationResult({
        firstInput,
        secondInput,
        response,
        output: response.result,
        clientX,
        clientY,
        usedInstanceIds,
        combinationsUsed
      });
    } catch {
      setErrorMessage("The combination could not be completed.");
    } finally {
      setIsCombining(false);
    }
  }

  async function selectDpoOutput(output: ElementToken) {
    if (!pendingDpoChoice) {
      return;
    }

    setPendingDpoChoice({ ...pendingDpoChoice, isSaving: true, errorMessage: undefined });

    try {
      await requestDpoPreference({
        mode,
        goalId: mode === "goal" ? goalPreset?.id : undefined,
        inputA: pendingDpoChoice.firstInput,
        inputB: pendingDpoChoice.secondInput,
        shownOutputs: pendingDpoChoice.candidates,
        selectedOutput: output,
        inventorySnapshot: inventory,
        combinationIndex: pendingDpoChoice.combinationIndex,
        source: pendingDpoChoice.response.source
      });
      applyCombinationResult({
        firstInput: pendingDpoChoice.firstInput,
        secondInput: pendingDpoChoice.secondInput,
        response: pendingDpoChoice.response,
        output,
        clientX: pendingDpoChoice.clientX,
        clientY: pendingDpoChoice.clientY,
        usedInstanceIds: pendingDpoChoice.usedInstanceIds,
        combinationsUsed: pendingDpoChoice.combinationIndex
      });
      setPendingDpoChoice(null);
    } catch {
      setPendingDpoChoice({
        ...pendingDpoChoice,
        isSaving: false,
        errorMessage: "Preference could not be saved."
      });
    }
  }

  function applyCombinationResult(input: {
    firstInput: ElementToken;
    secondInput: ElementToken;
    response: CombineResponse;
    output: ElementToken;
    clientX: number;
    clientY: number;
    usedInstanceIds: string[];
    combinationsUsed: number;
  }) {
    const discoveredAt = new Date().toISOString();
    const output = { ...input.output, discoveredAt };
    const nextResult = { ...input.response, result: output };

    setResult(nextResult);
    setInventory((currentInventory) => mergeInventory(currentInventory, output));
    setBoardElements((currentElements) => {
      const remainingElements = consumeInputsOnCombine
        ? currentElements.filter(
            (element) => !input.usedInstanceIds.includes(element.instanceId)
          )
        : currentElements;

      return [
        ...remainingElements,
        createBoardElement(
          output,
          ...getResultPosition(input.clientX, input.clientY),
          getNextZIndex(remainingElements)
        )
      ];
    });
    setHistory((currentHistory) =>
      [
        {
          id: `${input.firstInput.id}+${input.secondInput.id}=>${output.id}:${discoveredAt}`,
          inputA: input.firstInput,
          inputB: input.secondInput,
          output,
          source: input.response.source,
          createdAt: discoveredAt
        },
        ...currentHistory
      ].slice(0, 30)
    );

    if (isGoalTarget(output) && !goalCompletion) {
      void completeGoal(input.combinationsUsed);
    }
  }

  async function completeGoal(combinationsUsed: number) {
    if (mode !== "goal" || !goalPreset) {
      return;
    }

    setGoalCompletion({ combinationsUsed, isSaving: true });

    try {
      const payload = await requestLeaderboardSubmission({
        goalId: goalPreset.id,
        goalTitle: goalPreset.title,
        combinationsUsed
      });

      setLeaderboard(payload.entries);
      setGoalCompletion({ combinationsUsed, isSaving: false });
    } catch {
      setGoalCompletion({
        combinationsUsed,
        isSaving: false,
        errorMessage: "Leaderboard could not be saved."
      });
    }
  }

  function isGoalTarget(element: ElementToken): boolean {
    return mode === "goal" && goalPreset?.target.id === element.id;
  }

  function getBoardPosition(clientX: number, clientY: number) {
    const rect = boardRef.current?.getBoundingClientRect();

    if (!rect || !isPointInsideRect(clientX, clientY, rect)) {
      return null;
    }

    return clampBoardPosition(
      clientX - rect.left - TOKEN_WIDTH / 2,
      clientY - rect.top - TOKEN_HEIGHT / 2,
      rect
    );
  }

  function getResultPosition(clientX: number, clientY: number): [number, number] {
    const rect = boardRef.current?.getBoundingClientRect();

    if (!rect) {
      return [RESULT_OFFSET, RESULT_OFFSET];
    }

    const position = clampBoardPosition(
      clientX - rect.left + RESULT_OFFSET,
      clientY - rect.top + RESULT_OFFSET,
      rect
    );

    return [position.x, position.y];
  }

  function moveBoardElement(instanceId: string, clientX: number, clientY: number) {
    const rect = boardRef.current?.getBoundingClientRect();

    if (!rect || !dragState) {
      return;
    }

    const position = clampBoardPosition(
      clientX - rect.left - dragState.offsetX,
      clientY - rect.top - dragState.offsetY,
      rect
    );

    setBoardElements((currentElements) =>
      currentElements.map((element) =>
        element.instanceId === instanceId
          ? { ...element, x: position.x, y: position.y }
          : element
      )
    );
  }

  function bringElementForward(instanceId: string) {
    setBoardElements((currentElements) => {
      const nextZIndex = getNextZIndex(currentElements);

      return currentElements.map((element) =>
        element.instanceId === instanceId
          ? { ...element, zIndex: nextZIndex }
          : element
      );
    });
  }

  function findDropTarget(
    clientX: number,
    clientY: number,
    draggedInstanceId?: string
  ): BoardElement | null {
    const targetNode = document
      .elementsFromPoint(clientX, clientY)
      .map((node) =>
        node instanceof HTMLElement
          ? node.closest<HTMLElement>("[data-board-token-id]")
          : null
      )
      .find((node): node is HTMLElement => {
        if (!node?.dataset.boardTokenId) {
          return false;
        }

        return node.dataset.boardTokenId !== draggedInstanceId;
      });

    if (!targetNode?.dataset.boardTokenId) {
      return null;
    }

    return (
      boardElements.find(
        (element) => element.instanceId === targetNode.dataset.boardTokenId
      ) ?? null
    );
  }

  function isReleaseOnBlackHole(clientX: number, clientY: number): boolean {
    const blackHoleRect = blackHoleRef.current?.getBoundingClientRect();

    return Boolean(blackHoleRect && isPointInsideRect(clientX, clientY, blackHoleRect));
  }

  function getBlackHoleCenterInBoard(): Point {
    const boardRect = boardRef.current?.getBoundingClientRect();
    const blackHoleRect = blackHoleRef.current?.getBoundingClientRect();

    if (!boardRect || !blackHoleRect) {
      return { x: 0, y: 0 };
    }

    return {
      x: blackHoleRect.left - boardRect.left + blackHoleRect.width / 2,
      y: blackHoleRect.top - boardRect.top + blackHoleRect.height / 2
    };
  }

  function removeBoardElement(instanceId: string) {
    setBoardElements((currentElements) =>
      currentElements.filter((element) => element.instanceId !== instanceId)
    );
  }

  function beginInventoryDrag(
    event: PointerEvent<HTMLButtonElement>,
    element: ElementToken
  ) {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setErrorMessage(null);
    setDragState({
      kind: "inventory",
      element,
      offsetX: TOKEN_WIDTH / 2,
      offsetY: TOKEN_HEIGHT / 2,
      pointerX: event.clientX,
      pointerY: event.clientY
    });
  }

  function updateInventoryDrag(event: PointerEvent<HTMLButtonElement>) {
    if (dragState?.kind !== "inventory") {
      return;
    }

    setDragState({
      ...dragState,
      pointerX: event.clientX,
      pointerY: event.clientY
    });
  }

  async function finishInventoryDrag(event: PointerEvent<HTMLButtonElement>) {
    if (dragState?.kind !== "inventory") {
      return;
    }

    event.currentTarget.releasePointerCapture(event.pointerId);

    const target = findDropTarget(event.clientX, event.clientY);
    const element = dragState.element;

    setDragState(null);

    if (target) {
      await combineElementsOnBoard(
        element,
        target,
        event.clientX,
        event.clientY,
        [target.instanceId]
      );
      return;
    }

    addElementToBoard(element, event.clientX, event.clientY);
  }

  function beginBoardDrag(
    event: PointerEvent<HTMLButtonElement>,
    element: BoardElement
  ) {
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    const rect = event.currentTarget.getBoundingClientRect();

    setErrorMessage(null);
    bringElementForward(element.instanceId);
    setDragState({
      kind: "board",
      element,
      instanceId: element.instanceId,
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
      pointerX: event.clientX,
      pointerY: event.clientY
    });
  }

  function updateBoardDrag(event: PointerEvent<HTMLButtonElement>) {
    if (dragState?.kind !== "board" || !dragState.instanceId) {
      return;
    }

    setDragState({
      ...dragState,
      pointerX: event.clientX,
      pointerY: event.clientY
    });
    moveBoardElement(dragState.instanceId, event.clientX, event.clientY);
  }

  async function finishBoardDrag(event: PointerEvent<HTMLButtonElement>) {
    if (dragState?.kind !== "board" || !dragState.instanceId) {
      return;
    }

    event.currentTarget.releasePointerCapture(event.pointerId);

    const target = findDropTarget(
      event.clientX,
      event.clientY,
      dragState.instanceId
    );
    const element = dragState.element;
    const instanceId = dragState.instanceId;

    setDragState(null);

    if (isReleaseOnBlackHole(event.clientX, event.clientY)) {
      removeBoardElement(instanceId);
      return;
    }

    if (target) {
      await combineElementsOnBoard(
        element,
        target,
        event.clientX,
        event.clientY,
        [instanceId, target.instanceId]
      );
      return;
    }
  }

  return (
    <main
      className={`h-screen overflow-hidden ${
        isDarkMode ? "dark bg-zinc-950 text-zinc-50" : "bg-stone-100 text-zinc-950"
      }`}
    >
      <div className="grid h-screen grid-cols-1 grid-rows-[minmax(0,1fr)_minmax(0,42vh)] overflow-hidden lg:grid-cols-[minmax(0,1fr)_320px] lg:grid-rows-1">
        <section className="relative min-h-0 overflow-hidden border-b border-zinc-200 bg-[#f6f2e8] dark:border-zinc-800 dark:bg-zinc-900 lg:border-b-0 lg:border-r">
          <div className="absolute left-4 top-4 z-30 flex items-center gap-3 rounded-md border border-zinc-200 bg-white/90 px-3 py-2 shadow-hairline backdrop-blur dark:border-zinc-700 dark:bg-zinc-950/85">
            <div>
              <h1 className="text-base font-semibold tracking-normal">llm-craft</h1>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                {mode === "goal" ? "Goal" : "Sandbox"} · {boardElements.length} on board
              </p>
            </div>
            <button
              aria-label="Back to modes"
              className="grid size-9 place-items-center rounded-md border border-zinc-200 text-zinc-600 transition hover:bg-zinc-100 hover:text-zinc-950 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800 dark:hover:text-zinc-50"
              onClick={onBackToMenu}
              title="Modes"
              type="button"
            >
              <Menu size={16} />
            </button>
            <button
              aria-label="Profile"
              className="grid size-9 place-items-center rounded-md border border-zinc-200 text-zinc-600 transition hover:bg-zinc-100 hover:text-zinc-950 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800 dark:hover:text-zinc-50"
              onClick={() => onOpenProfile({ inventory, history })}
              title="Profile"
              type="button"
            >
              <UserCircle size={16} />
            </button>
            <button
              aria-label={isDarkMode ? "Use light mode" : "Use dark mode"}
              className="grid size-9 place-items-center rounded-md border border-zinc-200 text-zinc-600 transition hover:bg-zinc-100 hover:text-zinc-950 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800 dark:hover:text-zinc-50"
              onClick={() => setIsDarkMode((currentValue) => !currentValue)}
              title={isDarkMode ? "Light mode" : "Dark mode"}
              type="button"
            >
              {isDarkMode ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            {mode === "sandbox" ? (
              <>
                <button
                  aria-label="Clear sandbox"
                  className="grid size-9 place-items-center rounded-md border border-zinc-200 text-zinc-600 transition hover:bg-zinc-100 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-45 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800 dark:hover:text-zinc-50"
                  disabled={boardElements.length === 0 || isSweeping}
                  onClick={clearSandboxWithSweep}
                  title="Clear sandbox"
                  type="button"
                >
                  <Brush size={16} />
                </button>
                <button
                  aria-label="Reset"
                  className="grid size-9 place-items-center rounded-md border border-zinc-200 text-zinc-600 transition hover:bg-zinc-100 hover:text-zinc-950 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800 dark:hover:text-zinc-50"
                  onClick={() => resetGame()}
                  title="Reset"
                  type="button"
                >
                  <RotateCcw size={16} />
                </button>
              </>
            ) : null}
            <button
              aria-label="Log out"
              className="grid size-9 place-items-center rounded-md border border-zinc-200 text-zinc-600 transition hover:bg-zinc-100 hover:text-zinc-950 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800 dark:hover:text-zinc-50"
              onClick={onLogout}
              title="Log out"
              type="button"
            >
              <LogOut size={16} />
            </button>
          </div>

          <div
            className="absolute inset-0 z-0"
            ref={boardRef}
            style={{
              backgroundImage: isDarkMode
                ? "radial-gradient(circle at 1px 1px, rgba(244,244,245,0.11) 1px, transparent 0)"
                : "radial-gradient(circle at 1px 1px, rgba(39,39,42,0.1) 1px, transparent 0)",
              backgroundSize: "28px 28px"
            }}
          >
            <BlackHoleDropZone
              isActive={dragState?.kind === "board" || isSweeping}
              ref={blackHoleRef}
            />
            {isSweeping && sweepTarget ? (
              <SweepAnimation target={sweepTarget} />
            ) : null}
            {boardElements.map((element) => (
              <BoardToken
                element={element}
                isCombining={isCombining}
                isDragging={dragState?.instanceId === element.instanceId}
                isSweeping={isSweeping}
                key={element.instanceId}
                onPointerDown={beginBoardDrag}
                onPointerMove={updateBoardDrag}
                onPointerUp={finishBoardDrag}
                sweepTarget={sweepTarget}
              />
            ))}
          </div>

          <StatusToast
            errorMessage={errorMessage}
            isGoalMode={mode === "goal"}
            isCombining={isCombining}
            result={result}
          />
        </section>

        <aside
          className={`flex min-h-0 flex-col overflow-hidden bg-white p-4 transition-colors dark:bg-zinc-950 ${
            dragState?.kind === "board"
              ? "bg-zinc-50 dark:bg-zinc-900"
              : ""
          }`}
        >
          {mode === "goal" && goalPreset ? (
            <GoalSidePanel
              goalDepth={goalDepth}
              goalGenerationMessage={goalGenerationMessage}
              goalPreset={goalPreset}
              isGeneratingGoal={isGeneratingGoal}
              leaderboard={leaderboard}
              onGenerateNewGoal={generateNewGoal}
              onGoalDepthChange={onGoalDepthChange}
              onReset={() => resetGame()}
            />
          ) : null}

          <div className="mb-4">
            <div className="flex items-end justify-between gap-3">
              <div>
                <h2 className="text-base font-semibold">Inventory</h2>
                <p className="mt-1 text-sm text-zinc-500 dark:text-zinc-400">
                  {inventory.length} discovered by {user.displayName}
                </p>
              </div>
              <Sparkles className="text-zinc-400 dark:text-zinc-500" size={18} />
            </div>

            <label className="relative mt-4 block">
              <Search
                aria-hidden="true"
                className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400 dark:text-zinc-500"
                size={16}
              />
              <input
                className="h-10 w-full rounded-md border border-zinc-200 bg-zinc-50 pl-9 pr-3 text-sm outline-none transition placeholder:text-zinc-400 focus:border-zinc-400 focus:bg-white dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-50 dark:placeholder:text-zinc-500 dark:focus:border-zinc-500 dark:focus:bg-zinc-950"
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search"
                value={query}
              />
            </label>

            <label className="mt-3 flex min-h-11 items-center justify-between gap-3 rounded-md border border-zinc-200 bg-zinc-50 px-3 text-sm font-medium dark:border-zinc-700 dark:bg-zinc-900">
              <span>Consume inputs</span>
              <input
                checked={consumeInputsOnCombine}
                className="size-4 accent-zinc-950 dark:accent-zinc-50"
                onChange={(event) => setConsumeInputsOnCombine(event.target.checked)}
                type="checkbox"
              />
            </label>
            <label className="mt-2 flex min-h-11 items-center justify-between gap-3 rounded-md border border-zinc-200 bg-zinc-50 px-3 text-sm font-medium dark:border-zinc-700 dark:bg-zinc-900">
              <span>DPO test mode</span>
              <input
                checked={isDpoTestMode}
                className="size-4 accent-zinc-950 dark:accent-zinc-50"
                onChange={(event) => setIsDpoTestMode(event.target.checked)}
                type="checkbox"
              />
            </label>
          </div>

          <div className="grid min-h-0 flex-1 grid-cols-2 content-start gap-2 overflow-y-auto pr-1">
            {filteredInventory.map((element) => (
              <InventoryToken
                element={element}
                key={element.id}
                onPointerDown={beginInventoryDrag}
                onPointerMove={updateInventoryDrag}
                onPointerUp={finishInventoryDrag}
              />
            ))}
          </div>

          <div className="mt-5 min-h-0 shrink-0 border-t border-zinc-200 pt-4 dark:border-zinc-800">
            <div className="mb-3 flex items-center justify-between gap-3">
              <h2 className="text-base font-semibold">Recent</h2>
              <span className="text-sm text-zinc-500 dark:text-zinc-400">
                {history.length}
              </span>
            </div>

            <div className="grid max-h-52 gap-2 overflow-y-auto pr-1 lg:max-h-[220px]">
              {history.length === 0 ? (
                <div className="rounded-md border border-dashed border-zinc-200 p-4 text-sm text-zinc-500 dark:border-zinc-700 dark:text-zinc-400">
                  No recipes yet.
                </div>
              ) : (
                history.slice(0, 8).map((item) => (
                  <RecipeCard item={item} key={item.id} />
                ))
              )}
            </div>
          </div>
        </aside>
      </div>

      {dragState?.kind === "inventory" ? <DragPreview dragState={dragState} /> : null}
      {pendingDpoChoice ? (
        <DpoChoiceModal
          choice={pendingDpoChoice}
          onSelect={selectDpoOutput}
        />
      ) : null}
      {goalCompletion && goalPreset ? (
        <GoalCelebration
          completion={goalCompletion}
          leaderboard={leaderboard}
          target={goalPreset.target}
          userId={user.id}
          onBackToMenu={onBackToMenu}
          onReset={resetGame}
        />
      ) : null}
    </main>
  );
}

function GoalSidePanel({
  goalDepth,
  goalGenerationMessage,
  goalPreset,
  isGeneratingGoal,
  leaderboard,
  onGenerateNewGoal,
  onGoalDepthChange,
  onReset
}: {
  goalDepth: number;
  goalGenerationMessage: string | null;
  goalPreset: GoalPreset;
  isGeneratingGoal: boolean;
  leaderboard: LeaderboardEntry[];
  onGenerateNewGoal: () => void | Promise<void>;
  onGoalDepthChange: (depth: number) => void;
  onReset: () => void;
}) {
  return (
    <div className="mb-4 grid gap-3 rounded-md border border-emerald-200 bg-emerald-50 p-3 dark:border-emerald-900 dark:bg-emerald-950/30">
      <div className="flex items-start gap-3">
        <Target className="mt-0.5 shrink-0 text-emerald-700 dark:text-emerald-400" size={18} />
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold">{goalPreset.title}</h2>
          <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-300">
            {goalPreset.objective}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-md border border-emerald-200 bg-white px-3 py-2 dark:border-emerald-900 dark:bg-zinc-950">
          <p className="text-xs text-zinc-500 dark:text-zinc-400">Depth</p>
          <p className="mt-1 text-sm font-semibold">
            {goalPreset.metadata.minDepth ?? goalPreset.metadata.depth}
          </p>
        </div>
        <div className="rounded-md border border-emerald-200 bg-white px-3 py-2 dark:border-emerald-900 dark:bg-zinc-950">
          <p className="text-xs text-zinc-500 dark:text-zinc-400">Start</p>
          <p className="mt-1 truncate text-sm font-semibold capitalize">
            {goalPreset.metadata.initialInventoryId ?? "fallback"}
          </p>
        </div>
      </div>

      <div>
        <p className="mb-2 text-xs font-semibold text-zinc-500 dark:text-zinc-400">
          Initial inventory
        </p>
        <div className="flex flex-wrap gap-1.5">
          {goalPreset.initialInventory.map((element) => (
            <span
              className="max-w-full truncate rounded border border-emerald-200 bg-white px-2 py-1 text-xs capitalize dark:border-emerald-900 dark:bg-zinc-950"
              key={element.id}
            >
              {formatElementName(element)}
            </span>
          ))}
        </div>
      </div>

      <label className="grid gap-1 text-sm font-medium text-zinc-700 dark:text-zinc-200">
        <span>Goal depth</span>
        <select
          className="h-9 rounded-md border border-emerald-200 bg-white px-3 text-sm outline-none transition focus:border-emerald-500 disabled:cursor-wait disabled:opacity-60 dark:border-emerald-900 dark:bg-zinc-950"
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

      <div className="grid grid-cols-2 gap-2">
        <button
          className="flex h-10 items-center justify-center gap-2 rounded-md border border-emerald-200 bg-white px-3 text-sm font-semibold transition hover:bg-emerald-50 dark:border-emerald-900 dark:bg-zinc-950 dark:hover:bg-emerald-950/50"
          disabled={isGeneratingGoal}
          onClick={onReset}
          type="button"
        >
          <RotateCcw size={15} />
          Reset
        </button>
        <button
          className="flex h-10 items-center justify-center gap-2 rounded-md bg-emerald-700 px-3 text-sm font-semibold text-white transition hover:bg-emerald-800 disabled:cursor-wait disabled:opacity-60"
          disabled={isGeneratingGoal}
          onClick={() => void onGenerateNewGoal()}
          type="button"
        >
          <Target size={15} />
          {isGeneratingGoal ? "Generating" : "New goal"}
        </button>
      </div>

      {goalGenerationMessage ? (
        <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-700">
          {goalGenerationMessage}
        </p>
      ) : null}

      <LeaderboardPreview entries={leaderboard} />
    </div>
  );
}

function DpoChoiceModal({
  choice,
  onSelect
}: {
  choice: PendingDpoChoice;
  onSelect: (output: ElementToken) => void;
}) {
  return (
    <div className="fixed inset-0 z-[2147483647] grid place-items-center bg-zinc-950/45 px-4 backdrop-blur-sm">
      <div className="w-full max-w-xl rounded-md border border-zinc-200 bg-white p-5 text-zinc-950 shadow-2xl">
        <div>
          <p className="text-sm font-semibold text-zinc-500">DPO test mode</p>
          <h2 className="mt-1 text-2xl font-black tracking-normal">
            Choose the best result
          </h2>
          <p className="mt-2 text-sm text-zinc-600">
            {choice.firstInput.name} + {choice.secondInput.name}
          </p>
        </div>

        <div className="mt-5 grid gap-2">
          {choice.candidates.map((candidate) => (
            <button
              className="flex min-h-16 items-center justify-between gap-3 rounded-md border border-zinc-200 bg-zinc-50 px-4 py-3 text-left transition hover:border-zinc-400 hover:bg-white disabled:cursor-wait disabled:opacity-60"
              disabled={choice.isSaving}
              key={candidate.id}
              onClick={() => onSelect(candidate)}
              type="button"
            >
              <span className="truncate text-lg font-semibold capitalize">
                {candidate.name}
              </span>
              <span className="text-sm text-zinc-500">Select</span>
            </button>
          ))}
        </div>

        {choice.errorMessage ? (
          <p className="mt-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {choice.errorMessage}
          </p>
        ) : null}
      </div>
    </div>
  );
}

function LeaderboardPreview({ entries }: { entries: LeaderboardEntry[] }) {
  return (
    <div>
      <div className="flex items-center gap-2 text-xs font-semibold text-zinc-500 dark:text-zinc-400">
        <Trophy size={13} />
        Leaderboard
      </div>
      <div className="mt-2 grid gap-1">
        {entries.length === 0 ? (
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            No completed runs yet.
          </p>
        ) : (
          entries.slice(0, 3).map((entry, index) => (
            <div
              className="flex items-center justify-between gap-3 text-xs"
              key={entry.id}
            >
              <span className="truncate">
                {index + 1}. {entry.displayName}
              </span>
              <span className="shrink-0 font-semibold">
                {entry.combinationsUsed}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function GoalCelebration({
  completion,
  leaderboard,
  onBackToMenu,
  onReset,
  target,
  userId
}: {
  completion: GoalCompletion;
  leaderboard: LeaderboardEntry[];
  onBackToMenu: () => void;
  onReset: () => void;
  target: ElementToken;
  userId: string;
}) {
  const rank =
    leaderboard.findIndex((entry) => entry.userId === userId) >= 0
      ? leaderboard.findIndex((entry) => entry.userId === userId) + 1
      : null;

  return (
    <div className="fixed inset-0 z-[2147483647] grid place-items-center bg-zinc-950/45 px-4 backdrop-blur-sm">
      <div className="w-full max-w-lg rounded-md border border-emerald-200 bg-white p-5 text-zinc-950 shadow-2xl">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-sm font-semibold text-emerald-700">Goal complete</p>
            <h2 className="mt-1 text-3xl font-black tracking-normal">
              {target.emoji ?? "🎯"} {target.name}
            </h2>
          </div>
          <div className="grid size-14 place-items-center rounded-md border border-emerald-200 bg-emerald-50 text-3xl">
            🏁
          </div>
        </div>

        <div className="mt-5 grid grid-cols-2 gap-3">
          <div className="rounded-md border border-zinc-200 bg-zinc-50 p-3">
            <p className="text-2xl font-semibold">{completion.combinationsUsed}</p>
            <p className="mt-1 text-xs text-zinc-500">Combinations used</p>
          </div>
          <div className="rounded-md border border-zinc-200 bg-zinc-50 p-3">
            <p className="text-2xl font-semibold">{rank ? `#${rank}` : "..."}</p>
            <p className="mt-1 text-xs text-zinc-500">Leaderboard rank</p>
          </div>
        </div>

        {completion.isSaving ? (
          <p className="mt-4 rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-600">
            Saving leaderboard entry.
          </p>
        ) : null}
        {completion.errorMessage ? (
          <p className="mt-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {completion.errorMessage}
          </p>
        ) : null}

        <div className="mt-5">
          <div className="flex items-center gap-2 text-sm font-semibold">
            <Trophy size={16} />
            Top runs
          </div>
          <div className="mt-3 grid gap-2">
            {leaderboard.length === 0 ? (
              <p className="rounded-md border border-dashed border-zinc-200 p-3 text-sm text-zinc-500">
                No completed runs yet.
              </p>
            ) : (
              leaderboard.slice(0, 5).map((entry, index) => (
                <div
                  className={`flex items-center justify-between gap-3 rounded-md border px-3 py-2 text-sm ${
                    entry.userId === userId
                      ? "border-emerald-300 bg-emerald-50"
                      : "border-zinc-200 bg-zinc-50"
                  }`}
                  key={entry.id}
                >
                  <span className="truncate">
                    {index + 1}. {entry.displayName}
                  </span>
                  <span className="font-semibold">
                    {entry.combinationsUsed} combos
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="mt-5 grid gap-2 sm:grid-cols-2">
          <button
            className="h-10 rounded-md border border-zinc-200 bg-white px-3 text-sm font-semibold transition hover:bg-zinc-50"
            onClick={onReset}
            type="button"
          >
            Try again
          </button>
          <button
            className="h-10 rounded-md bg-zinc-950 px-3 text-sm font-semibold text-white transition hover:bg-zinc-800"
            onClick={onBackToMenu}
            type="button"
          >
            Back to modes
          </button>
        </div>
      </div>
    </div>
  );
}

const BlackHoleDropZone = forwardRef<
  HTMLDivElement,
  { isActive: boolean }
>(function BlackHoleDropZone({ isActive }, ref) {
  return (
    <div
      aria-label="Delete"
      className={`absolute bottom-5 right-5 z-[2147483647] grid size-32 place-items-center rounded-full transition duration-200 sm:size-36 ${
        isActive ? "scale-105 opacity-100" : "scale-100 opacity-80"
      }`}
      ref={ref}
      style={{
        background:
          "radial-gradient(circle at center, #020617 0 34%, #111827 35% 46%, #4c1d95 48% 55%, rgba(20,184,166,0.5) 56% 61%, transparent 62%)",
        boxShadow:
          "0 0 28px rgba(20,184,166,0.35), 0 0 46px rgba(124,58,237,0.35)"
      }}
      title="Delete"
    >
      <div className="size-14 rounded-full bg-black shadow-[inset_0_0_18px_rgba(255,255,255,0.08)] sm:size-16" />
    </div>
  );
});

function SweepAnimation({ target }: { target: Point }) {
  return (
    <div className="pointer-events-none absolute inset-0 z-[2147483000] overflow-hidden">
      <div
        className="sweep-dust absolute h-20 w-20 rounded-full bg-teal-300/30 blur-2xl"
        style={{
          left: target.x - 40,
          top: target.y - 40
        }}
      />
    </div>
  );
}

function BoardToken({
  element,
  isCombining,
  isDragging,
  isSweeping,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  sweepTarget
}: {
  element: BoardElement;
  isCombining: boolean;
  isDragging: boolean;
  isSweeping: boolean;
  onPointerDown: (
    event: PointerEvent<HTMLButtonElement>,
    element: BoardElement
  ) => void;
  onPointerMove: (event: PointerEvent<HTMLButtonElement>) => void;
  onPointerUp: (event: PointerEvent<HTMLButtonElement>) => void;
  sweepTarget: Point | null;
}) {
  const sweepTransform =
    isSweeping && sweepTarget
      ? `translate(${sweepTarget.x - element.x - TOKEN_WIDTH / 2}px, ${
          sweepTarget.y - element.y - TOKEN_HEIGHT / 2
        }px) scale(0.35)`
      : undefined;

  return (
    <button
      className={`absolute flex h-12 w-[132px] touch-none select-none items-center gap-2 rounded-md border bg-white px-3 text-left shadow-sm transition-shadow hover:shadow-md dark:bg-zinc-900 ${
        isDragging
          ? "border-zinc-950 opacity-90 shadow-lg dark:border-zinc-100"
          : "border-zinc-200 dark:border-zinc-700"
      } ${
        isCombining || isSweeping
          ? "cursor-wait"
          : "cursor-grab active:cursor-grabbing"
      } ${isSweeping ? "board-token-sweeping" : ""}`}
      data-board-token-id={element.instanceId}
      disabled={isCombining || isSweeping}
      onPointerDown={(event) => onPointerDown(event, element)}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      style={{
        left: element.x,
        top: element.y,
        transform: sweepTransform,
        zIndex: element.zIndex
      }}
      type="button"
    >
      <span className="grid size-8 shrink-0 place-items-center rounded border border-zinc-200 bg-zinc-50 text-lg dark:border-zinc-700 dark:bg-zinc-800">
        {element.emoji ?? "·"}
      </span>
      <span className="min-w-0 truncate text-sm font-medium capitalize">
        {element.name}
      </span>
    </button>
  );
}

function InventoryToken({
  element,
  onPointerDown,
  onPointerMove,
  onPointerUp
}: {
  element: ElementToken;
  onPointerDown: (
    event: PointerEvent<HTMLButtonElement>,
    element: ElementToken
  ) => void;
  onPointerMove: (event: PointerEvent<HTMLButtonElement>) => void;
  onPointerUp: (event: PointerEvent<HTMLButtonElement>) => void;
}) {
  return (
    <button
      className="flex min-h-16 touch-none select-none flex-col items-center justify-center rounded-md border border-zinc-200 bg-zinc-50 px-2 py-2 text-center transition hover:border-zinc-300 hover:bg-white active:cursor-grabbing dark:border-zinc-700 dark:bg-zinc-900 dark:hover:border-zinc-600 dark:hover:bg-zinc-800"
      onPointerDown={(event) => onPointerDown(event, element)}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      type="button"
    >
      <span className="text-2xl">{element.emoji ?? "·"}</span>
      <span className="mt-1 w-full truncate text-xs font-medium capitalize text-zinc-700 dark:text-zinc-200">
        {element.name}
      </span>
    </button>
  );
}

function DragPreview({ dragState }: { dragState: DragState }) {
  return (
    <div
      className="pointer-events-none fixed z-50 flex h-12 w-[132px] items-center gap-2 rounded-md border border-zinc-950 bg-white px-3 text-left opacity-90 shadow-lg dark:border-zinc-100 dark:bg-zinc-900"
      style={{
        left: dragState.pointerX - dragState.offsetX,
        top: dragState.pointerY - dragState.offsetY
      }}
    >
      <span className="grid size-8 shrink-0 place-items-center rounded border border-zinc-200 bg-zinc-50 text-lg dark:border-zinc-700 dark:bg-zinc-800">
        {dragState.element.emoji ?? "·"}
      </span>
      <span className="min-w-0 truncate text-sm font-medium capitalize">
        {dragState.element.name}
      </span>
    </div>
  );
}

function StatusToast({
  errorMessage,
  isGoalMode,
  isCombining,
  result
}: {
  errorMessage: string | null;
  isGoalMode: boolean;
  isCombining: boolean;
  result: CombineResponse | null;
}) {
  if (!errorMessage && !isCombining && !result) {
    return null;
  }

  return (
    <div
      className={`absolute left-4 z-40 max-w-[calc(100%-2rem)] rounded-md border border-zinc-200 bg-white/95 px-4 py-3 shadow-hairline backdrop-blur dark:border-zinc-700 dark:bg-zinc-950/90 sm:max-w-sm ${
        isGoalMode ? "bottom-32" : "bottom-4"
      }`}
    >
      {errorMessage ? (
        <p className="text-sm text-zinc-600 dark:text-zinc-300">{errorMessage}</p>
      ) : isCombining ? (
        <p className="text-sm text-zinc-600 dark:text-zinc-300">
          Resolving combination.
        </p>
      ) : result ? (
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xl">{result.result.emoji ?? "·"}</span>
            <span className="truncate text-sm font-semibold capitalize">
              {result.result.name}
            </span>
          </div>
          <div className="mt-2 flex flex-wrap gap-2 text-xs text-zinc-500 dark:text-zinc-400">
            <span className="rounded border border-zinc-200 bg-zinc-50 px-2 py-1 dark:border-zinc-700 dark:bg-zinc-900">
              {formatCombineSource(result.source)}
            </span>
            {typeof result.confidence === "number" ? (
              <span className="rounded border border-zinc-200 bg-zinc-50 px-2 py-1 dark:border-zinc-700 dark:bg-zinc-900">
                {Math.round(result.confidence * 100)}%
              </span>
            ) : null}
            {result.source === "model_generated" && result.model ? (
              <span className="rounded border border-zinc-200 bg-zinc-50 px-2 py-1 dark:border-zinc-700 dark:bg-zinc-900">
                {getCombinerModelLabel(result.model)}
              </span>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function RecipeCard({ item }: { item: RecipeHistoryItem }) {
  return (
    <div className="rounded-md border border-zinc-200 bg-zinc-50 p-3 dark:border-zinc-700 dark:bg-zinc-900">
      <div className="flex flex-wrap items-center gap-1 text-xs text-zinc-500 dark:text-zinc-400">
        <span>{formatElementName(item.inputA)}</span>
        <span>+</span>
        <span>{formatElementName(item.inputB)}</span>
      </div>
      <div className="mt-2 flex items-center justify-between gap-3">
        <span className="truncate text-sm font-medium capitalize">
          {formatElementName(item.output)}
        </span>
        <span className="rounded border border-zinc-200 bg-white px-2 py-1 text-xs text-zinc-500 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-400">
          {formatCombineSource(item.source)}
        </span>
      </div>
    </div>
  );
}

function createInitialBoard(elements: ElementToken[]): BoardElement[] {
  return elements.slice(0, 4).map((element, index) => ({
    ...element,
    instanceId: `initial:${element.id}:${index}`,
    x: 120 + (index % 2) * 165,
    y: 150 + Math.floor(index / 2) * 82,
    zIndex: index + 1
  }));
}

function createBoardElement(
  element: ElementToken,
  x: number,
  y: number,
  zIndex: number
): BoardElement {
  return {
    ...element,
    instanceId: `${element.id}:${Date.now()}:${Math.random().toString(36).slice(2)}`,
    x,
    y,
    zIndex
  };
}

function clampBoardPosition(
  x: number,
  y: number,
  rect: DOMRect
): { x: number; y: number } {
  return {
    x: Math.max(0, Math.min(x, rect.width - TOKEN_WIDTH)),
    y: Math.max(0, Math.min(y, rect.height - TOKEN_HEIGHT))
  };
}

function getNextZIndex(elements: BoardElement[]): number {
  return Math.max(0, ...elements.map((element) => element.zIndex)) + 1;
}

function isPointInsideRect(clientX: number, clientY: number, rect: DOMRect): boolean {
  return (
    clientX >= rect.left &&
    clientX <= rect.right &&
    clientY >= rect.top &&
    clientY <= rect.bottom
  );
}

function formatElementName(element: ElementToken): string {
  return `${element.emoji ? `${element.emoji} ` : ""}${element.name}`;
}

function formatCombineSource(source: CombineResponse["source"]): string {
  return source === "known_recipe" ? "known recipe" : "model";
}

function toElementToken(element: ElementToken): ElementToken {
  return {
    id: element.id,
    name: element.name,
    emoji: element.emoji,
    discoveredAt: element.discoveredAt
  };
}

function readStoredValue<T>(key: string, fallback: T): T {
  const rawValue = window.localStorage.getItem(key);

  if (!rawValue) {
    return fallback;
  }

  try {
    return JSON.parse(rawValue) as T;
  } catch {
    return fallback;
  }
}
