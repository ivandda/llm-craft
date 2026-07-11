"use client";

import {
  requestCombination,
  requestDpoCandidates,
  requestDpoPreference,
  requestLeaderboard,
  requestLeaderboardSubmission
} from "@/lib/api";
import { getHueForConcept } from "@/lib/emoji";
import { mergeInventory } from "@/lib/craft";
import { selectDpoCandidates } from "@/lib/dpo";
import { getInitialInventoryForMode } from "@/lib/gameModes";
import { createGameStorageKey } from "@/lib/gameStorage";
import { setTheme, useTheme } from "@/lib/theme";
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
  ArrowLeft,
  Brush,
  Clock,
  Combine,
  LayoutGrid,
  LogOut,
  Moon,
  RotateCcw,
  Search,
  Settings as SettingsIcon,
  Sparkles,
  Sun,
  Target,
  Trophy,
  UserCircle,
  X
} from "lucide-react";
import {
  forwardRef,
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type PointerEvent,
  type ReactNode
} from "react";

const TOKEN_WIDTH = 150;
const TOKEN_HEIGHT = 48;
const RESULT_OFFSET = 18;
// "Help train the AI" rounds fire on combinations 3, 8, 13, ... instead of on
// every multi-candidate recipe, so the modal stays occasional.
const DPO_ROUND_START = 3;
const DPO_ROUND_INTERVAL = 5;

function shouldTriggerDpoRound(combinationIndex: number): boolean {
  return (
    combinationIndex >= DPO_ROUND_START &&
    (combinationIndex - DPO_ROUND_START) % DPO_ROUND_INTERVAL === 0
  );
}

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

type Discovery = {
  name: string;
  emoji?: string;
  isModelGenerated: boolean;
};

type SidebarTab = "elements" | "goal" | "recent" | "settings";

// Below `lg`, the game switches to a touch-first model: elements/recent/
// settings live in pop-up sheets and combining is tap-based, because
// drag-from-a-scrolling-list fights the scroll on touch screens.
function useIsDesktop(): boolean {
  const [isDesktop, setIsDesktop] = useState(true);

  useEffect(() => {
    const query = window.matchMedia("(min-width: 1024px)");
    const update = () => setIsDesktop(query.matches);

    update();
    query.addEventListener("change", update);

    return () => query.removeEventListener("change", update);
  }, []);

  return isDesktop;
}

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
  const [dragState, setDragStateValue] = useState<DragState | null>(null);
  // Pointer handlers read this ref instead of React state: a fast click fires
  // pointerdown+up before the state commit, which left a ghost drag preview
  // following the cursor.
  const dragStateRef = useRef<DragState | null>(null);
  // Pointer position lives in a ref, not state: updating state per mousemove
  // re-rendered every token on the board and made drags stutter.
  const dragPointerRef = useRef<Point | null>(null);

  function setDragState(next: DragState | null) {
    dragStateRef.current = next;
    dragPointerRef.current = next
      ? { x: next.pointerX, y: next.pointerY }
      : null;
    setDragStateValue(next);
  }
  // Theme is global (persisted app-wide, applied to <html>), so toggling it
  // here stays consistent with every other screen.
  const isDarkMode = useTheme() === "dark";
  const isDesktop = useIsDesktop();
  // Which pop-up sheet is open on mobile (null = none). Desktop uses the
  // persistent sidebar instead.
  const [mobileSheet, setMobileSheet] = useState<SidebarTab | null>(null);
  // Tap-to-combine selection on touch: first tapped board token is highlighted,
  // tapping a second combines them.
  const [selectedInstanceId, setSelectedInstanceId] = useState<string | null>(
    null
  );
  const [activeSidebarTab, setActiveSidebarTab] = useState<SidebarTab>(
    mode === "goal" ? "goal" : "elements"
  );
  // The goal tab only exists in goal mode; fall back instead of rendering an
  // empty panel if the stored selection no longer applies.
  const sidebarTab: SidebarTab =
    activeSidebarTab === "goal" && mode !== "goal" ? "elements" : activeSidebarTab;
  const sidebarTabs = useMemo<{ id: SidebarTab; label: string }[]>(
    () => [
      { id: "elements", label: "Elements" },
      ...(mode === "goal" ? [{ id: "goal" as const, label: "Goal" }] : []),
      { id: "recent", label: "Recent" },
      { id: "settings", label: "Settings" }
    ],
    [mode]
  );
  const [consumeInputsOnCombine, setConsumeInputsOnCombine] = useState(false);
  const [isDpoTestMode, setIsDpoTestMode] = useState(true);
  const [pendingDpoChoice, setPendingDpoChoice] = useState<PendingDpoChoice | null>(
    null
  );
  const [isSweeping, setIsSweeping] = useState(false);
  const [sweepTarget, setSweepTarget] = useState<Point | null>(null);
  const [hasHydratedStorage, setHasHydratedStorage] = useState(false);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [goalCompletion, setGoalCompletion] = useState<GoalCompletion | null>(null);
  const [discovery, setDiscovery] = useState<Discovery | null>(null);
  const discoveryTimeoutRef = useRef<number | null>(null);
  const [isOverBlackHole, setIsOverBlackHole] = useState(false);
  const [vanishingToken, setVanishingToken] = useState<{
    instanceId: string;
    target: Point;
  } | null>(null);
  const vanishTimeoutRef = useRef<number | null>(null);

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
    setConsumeInputsOnCombine(readStoredValue(storageKeys.consumeInputs, false));
    setIsDpoTestMode(readStoredValue(storageKeys.dpoTestMode, true));
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

  // Initial board positions are precomputed for a wide board; on a narrow
  // phone (or after an orientation change) they would strand tokens off the
  // right edge. Clamp tokens into view on mount and when the board's WIDTH
  // changes — deliberately ignoring height, because mobile browser chrome
  // constantly changes the dvh-based height and clamping on that jitter would
  // slowly migrate tokens toward the top-left (and persist the damage).
  useEffect(() => {
    const board = boardRef.current;

    if (!board || !hasHydratedStorage) {
      return;
    }

    let lastWidth = 0;

    const clampIntoViewOnWidthChange = () => {
      // Never reposition while dragging — it would yank the token under the
      // pointer out from under the user.
      if (dragStateRef.current) {
        return;
      }

      const rect = board.getBoundingClientRect();

      if (rect.width === 0 || rect.height === 0 || rect.width === lastWidth) {
        return;
      }

      lastWidth = rect.width;

      setBoardElements((currentElements) => {
        let hasChanges = false;
        const nextElements = currentElements.map((element) => {
          const clamped = clampBoardPosition(element.x, element.y, rect);

          if (clamped.x !== element.x || clamped.y !== element.y) {
            hasChanges = true;
            return { ...element, x: clamped.x, y: clamped.y };
          }

          return element;
        });

        return hasChanges ? nextElements : currentElements;
      });
    };

    clampIntoViewOnWidthChange();
    const observer = new ResizeObserver(clampIntoViewOnWidthChange);
    observer.observe(board);

    return () => observer.disconnect();
  }, [hasHydratedStorage]);

  useEffect(() => {
    return () => {
      if (sweepTimeoutRef.current !== null) {
        window.clearTimeout(sweepTimeoutRef.current);
      }
      if (discoveryTimeoutRef.current !== null) {
        window.clearTimeout(discoveryTimeoutRef.current);
      }
      if (vanishTimeoutRef.current !== null) {
        window.clearTimeout(vanishTimeoutRef.current);
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
    // A release outside the board (including a plain click on an inventory
    // item) still places the element, near the board center.
    const position =
      getBoardPosition(clientX, clientY) ?? getFallbackBoardPosition();

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

      if (isDpoTestMode && shouldTriggerDpoRound(combinationsUsed)) {
        const dpoCandidates = await fetchDpoCandidates(firstInput, secondInput);

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

  async function fetchDpoCandidates(
    firstInput: ElementToken,
    secondInput: ElementToken
  ): Promise<ElementToken[]> {
    try {
      const { candidates } = await requestDpoCandidates({
        inputA: firstInput,
        inputB: secondInput,
        inventory,
        model: selectedCombinerModel
      });

      return selectDpoCandidates(candidates);
    } catch {
      // A failed round never blocks the game; the combination applies normally.
      return [];
    }
  }

  function skipDpoChoice() {
    if (!pendingDpoChoice) {
      return;
    }

    applyCombinationResult({
      firstInput: pendingDpoChoice.firstInput,
      secondInput: pendingDpoChoice.secondInput,
      response: pendingDpoChoice.response,
      output: pendingDpoChoice.response.result,
      clientX: pendingDpoChoice.clientX,
      clientY: pendingDpoChoice.clientY,
      usedInstanceIds: pendingDpoChoice.usedInstanceIds,
      combinationsUsed: pendingDpoChoice.combinationIndex
    });
    setPendingDpoChoice(null);
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
    const isNewDiscovery = !inventory.some((element) => element.id === output.id);

    setResult(nextResult);
    setInventory((currentInventory) => mergeInventory(currentInventory, output));

    if (isNewDiscovery) {
      setDiscovery({
        name: output.name,
        emoji: output.emoji,
        isModelGenerated: input.response.source === "model_generated"
      });

      if (discoveryTimeoutRef.current !== null) {
        window.clearTimeout(discoveryTimeoutRef.current);
      }

      discoveryTimeoutRef.current = window.setTimeout(() => {
        setDiscovery(null);
        discoveryTimeoutRef.current = null;
      }, 2500);
    }
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

  function getFallbackBoardPosition(): Point | null {
    const rect = boardRef.current?.getBoundingClientRect();

    if (!rect) {
      return null;
    }

    return clampBoardPosition(
      rect.width / 2 - TOKEN_WIDTH / 2 + (Math.random() - 0.5) * 140,
      rect.height / 2 - TOKEN_HEIGHT / 2 + (Math.random() - 0.5) * 140,
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

    if (!blackHoleRect) {
      return false;
    }

    // The visual is a circle; a rect test deletes drops that look outside it.
    const centerX = blackHoleRect.left + blackHoleRect.width / 2;
    const centerY = blackHoleRect.top + blackHoleRect.height / 2;

    return (
      Math.hypot(clientX - centerX, clientY - centerY) <= blackHoleRect.width / 2
    );
  }

  function swallowBoardElement(instanceId: string) {
    if (vanishTimeoutRef.current !== null) {
      window.clearTimeout(vanishTimeoutRef.current);

      if (vanishingToken) {
        removeBoardElement(vanishingToken.instanceId);
      }
    }

    setVanishingToken({ instanceId, target: getBlackHoleCenterInBoard() });
    vanishTimeoutRef.current = window.setTimeout(() => {
      removeBoardElement(instanceId);
      setVanishingToken(null);
      vanishTimeoutRef.current = null;
    }, 320);
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
    const drag = dragStateRef.current;

    if (drag?.kind !== "inventory") {
      return;
    }

    dragPointerRef.current = { x: event.clientX, y: event.clientY };
    setIsOverBlackHole(isReleaseOnBlackHole(event.clientX, event.clientY));
  }

  async function finishInventoryDrag(event: PointerEvent<HTMLButtonElement>) {
    const drag = dragStateRef.current;

    if (drag?.kind !== "inventory") {
      return;
    }

    event.currentTarget.releasePointerCapture(event.pointerId);

    const target = findDropTarget(event.clientX, event.clientY);
    const element = drag.element;

    setDragState(null);
    setIsOverBlackHole(false);

    if (isReleaseOnBlackHole(event.clientX, event.clientY)) {
      return;
    }

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
    const drag = dragStateRef.current;

    if (drag?.kind !== "board" || !drag.instanceId) {
      return;
    }

    dragPointerRef.current = { x: event.clientX, y: event.clientY };
    moveBoardElement(drag.instanceId, event.clientX, event.clientY);
    setIsOverBlackHole(isReleaseOnBlackHole(event.clientX, event.clientY));
  }

  function cancelActiveDrag() {
    setDragState(null);
    setIsOverBlackHole(false);
  }

  async function finishBoardDrag(event: PointerEvent<HTMLButtonElement>) {
    const drag = dragStateRef.current;

    if (drag?.kind !== "board" || !drag.instanceId) {
      return;
    }

    event.currentTarget.releasePointerCapture(event.pointerId);

    const target = findDropTarget(
      event.clientX,
      event.clientY,
      drag.instanceId
    );
    const element = drag.element;
    const instanceId = drag.instanceId;

    setDragState(null);
    setIsOverBlackHole(false);

    if (isReleaseOnBlackHole(event.clientX, event.clientY)) {
      swallowBoardElement(instanceId);
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

  // --- Touch (tap) interaction: no dragging from the scrolling list ---

  // Add an element from a pop-up sheet to the board near its centre.
  function placeElementOnBoard(element: ElementToken) {
    setErrorMessage(null);
    const position = getFallbackBoardPosition();

    if (!position) {
      return;
    }

    setBoardElements((currentElements) => [
      ...currentElements,
      createBoardElement(element, position.x, position.y, getNextZIndex(currentElements))
    ]);
    setMobileSheet(null);
  }

  // Tap a board token: select it, or combine it with the already-selected one.
  function handleBoardTokenTap(
    element: BoardElement,
    clientX: number,
    clientY: number
  ) {
    if (isCombining || isSweeping) {
      return;
    }

    if (!selectedInstanceId) {
      setSelectedInstanceId(element.instanceId);
      return;
    }

    if (selectedInstanceId === element.instanceId) {
      setSelectedInstanceId(null);
      return;
    }

    const firstElement = boardElements.find(
      (candidate) => candidate.instanceId === selectedInstanceId
    );
    setSelectedInstanceId(null);

    if (!firstElement) {
      return;
    }

    void combineElementsOnBoard(firstElement, element, clientX, clientY, [
      firstElement.instanceId,
      element.instanceId
    ]);
  }

  function removeSelectedToken() {
    if (selectedInstanceId) {
      removeBoardElement(selectedInstanceId);
      setSelectedInstanceId(null);
    }
  }

  const selectedElement = selectedInstanceId
    ? boardElements.find((element) => element.instanceId === selectedInstanceId) ??
      null
    : null;

  // Shared panel bodies, rendered in the desktop sidebar and the mobile
  // pop-up sheets from a single source of truth.
  const renderElements = (interaction: "drag" | "tap") => (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="mb-3 shrink-0">
        <label className="relative block">
          <Search
            aria-hidden="true"
            className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-soot"
            size={16}
          />
          <input
            className="h-10 w-full rounded-md border border-linen bg-paper pl-9 pr-3 text-sm outline-none transition placeholder:text-soot focus:border-cobalt focus:bg-surface"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search elements"
            value={query}
          />
        </label>
        <p className="mt-2 text-right font-mono text-xs text-soot">
          {filteredInventory.length} of {inventory.length} elements
        </p>
      </div>

      <div
        className={`grid min-h-0 flex-1 content-start gap-2 overflow-y-auto pr-1 ${
          interaction === "tap" ? "grid-cols-3 sm:grid-cols-4" : "grid-cols-2"
        }`}
      >
        {filteredInventory.map((element) => (
          <InventoryToken
            element={element}
            interactionMode={interaction}
            key={element.id}
            onPick={placeElementOnBoard}
            onPointerCancel={cancelActiveDrag}
            onPointerDown={beginInventoryDrag}
            onPointerMove={updateInventoryDrag}
            onPointerUp={finishInventoryDrag}
          />
        ))}
        {filteredInventory.length === 0 ? (
          <p className="col-span-full rounded-md border border-dashed border-linen p-4 text-center text-sm text-soot">
            No elements match your search.
          </p>
        ) : null}
      </div>

      {interaction === "tap" ? (
        <p className="mt-2 shrink-0 text-center text-xs text-soot">
          Tap an element to drop it on the board.
        </p>
      ) : null}
    </div>
  );

  const renderRecent = () => (
    <div className="min-h-0 flex-1 overflow-y-auto pr-1">
      <div className="grid gap-2">
        {history.length === 0 ? (
          <div className="rounded-md border border-dashed border-linen p-4 text-sm text-soot">
            No recipes yet.
          </div>
        ) : (
          history.map((item) => <RecipeCard item={item} key={item.id} />)
        )}
      </div>
    </div>
  );

  const renderSettings = () => (
    <div className="grid min-h-0 flex-1 content-start gap-2 overflow-y-auto pr-1">
      <label className="flex min-h-11 items-center gap-3 rounded-md border border-linen bg-paper px-3 text-sm font-medium">
        <Combine className="shrink-0 text-soot" size={16} />
        <span className="flex-1">Consume inputs</span>
        <input
          checked={consumeInputsOnCombine}
          className="size-4 accent-cobalt"
          onChange={(event) => setConsumeInputsOnCombine(event.target.checked)}
          type="checkbox"
        />
      </label>
      <label className="flex min-h-11 items-center gap-3 rounded-md border border-linen bg-paper px-3 text-sm font-medium">
        <Sparkles className="shrink-0 text-soot" size={16} />
        <span className="flex-1">Help train the AI</span>
        <input
          checked={isDpoTestMode}
          className="size-4 accent-cobalt"
          onChange={(event) => setIsDpoTestMode(event.target.checked)}
          type="checkbox"
        />
      </label>
      <label className="flex min-h-11 items-center gap-3 rounded-md border border-linen bg-paper px-3 text-sm font-medium">
        {isDarkMode ? (
          <Sun className="shrink-0 text-soot" size={16} />
        ) : (
          <Moon className="shrink-0 text-soot" size={16} />
        )}
        <span className="flex-1">Dark mode</span>
        <input
          checked={isDarkMode}
          className="size-4 accent-cobalt"
          onChange={(event) => setTheme(event.target.checked ? "dark" : "light")}
          type="checkbox"
        />
      </label>

      <div className="my-1 border-t border-linen" />

      <button
        className="flex min-h-11 w-full items-center gap-3 rounded-md border border-linen bg-paper px-3 text-left text-sm font-medium transition hover:bg-surface"
        onClick={() => onOpenProfile({ inventory, history })}
        type="button"
      >
        <UserCircle className="shrink-0 text-soot" size={16} />
        Profile
      </button>
      {mode === "sandbox" ? (
        <>
          <button
            className="flex min-h-11 w-full items-center gap-3 rounded-md border border-linen bg-paper px-3 text-left text-sm font-medium transition hover:bg-surface disabled:cursor-not-allowed disabled:opacity-45"
            disabled={boardElements.length === 0 || isSweeping}
            onClick={clearSandboxWithSweep}
            type="button"
          >
            <Brush className="shrink-0 text-soot" size={16} />
            Clear board
          </button>
          <button
            className="flex min-h-11 w-full items-center gap-3 rounded-md border border-linen bg-paper px-3 text-left text-sm font-medium transition hover:bg-surface"
            onClick={() => resetGame()}
            type="button"
          >
            <RotateCcw className="shrink-0 text-soot" size={16} />
            Reset progress
          </button>
        </>
      ) : null}
      <button
        className="flex min-h-11 w-full items-center gap-3 rounded-md border border-linen bg-paper px-3 text-left text-sm font-medium transition hover:bg-surface"
        onClick={onLogout}
        type="button"
      >
        <LogOut className="shrink-0 text-soot" size={16} />
        Log out
      </button>
    </div>
  );

  const renderGoalPanel = () =>
    mode === "goal" && goalPreset ? (
      <div className="min-h-0 flex-1 overflow-y-auto pr-1">
        <GoalTabPanel
          goalDepth={goalDepth}
          goalGenerationMessage={goalGenerationMessage}
          goalPreset={goalPreset}
          isGeneratingGoal={isGeneratingGoal}
          leaderboard={leaderboard}
          onGenerateNewGoal={generateNewGoal}
          onGoalDepthChange={onGoalDepthChange}
          onReset={() => resetGame()}
        />
      </div>
    ) : null;

  const mobileTabs: { id: SidebarTab; label: string; icon: ReactNode; badge?: number }[] = [
    { id: "elements", label: "Elements", icon: <LayoutGrid size={18} />, badge: inventory.length },
    ...(mode === "goal"
      ? [{ id: "goal" as const, label: "Goal", icon: <Target size={18} /> }]
      : []),
    { id: "recent", label: "Recent", icon: <Clock size={18} />, badge: history.length || undefined },
    { id: "settings", label: "Settings", icon: <SettingsIcon size={18} /> }
  ];

  return (
    <main className="h-[100dvh] overflow-hidden bg-paper text-ink">
      <div className="grid h-[100dvh] grid-cols-1 grid-rows-[minmax(0,1fr)_auto] overflow-hidden lg:grid-cols-[minmax(0,1fr)_320px] lg:grid-rows-1">
        <section className="relative min-h-0 overflow-hidden border-b border-linen bg-paper lg:border-b-0 lg:border-r">
          <div className="absolute left-3 top-3 z-30 flex items-center gap-2 rounded-md border border-linen bg-surface/90 px-2.5 py-2 shadow-hairline backdrop-blur">
            <button
              className="flex h-9 items-center gap-1.5 rounded-md border border-linen px-2.5 text-sm font-medium text-soot transition hover:bg-paper hover:text-ink"
              onClick={onBackToMenu}
              title="Back to modes"
              type="button"
            >
              <ArrowLeft size={16} />
              Modes
            </button>
            <div className="min-w-0">
              <h1 className="font-display text-sm font-semibold tracking-normal">llm-craft</h1>
              <p className="font-mono text-xs text-soot">
                {mode === "goal" ? "Goal" : "Sandbox"} · {boardElements.length} on board
              </p>
            </div>
          </div>

          <div
            className="absolute inset-0 z-0"
            onClick={() => {
              // Tap empty board (touch mode) clears the current selection.
              if (!isDesktop) {
                setSelectedInstanceId(null);
              }
            }}
            ref={boardRef}
            style={{
              backgroundImage: isDarkMode
                ? "radial-gradient(circle at 1px 1px, rgba(244,244,245,0.11) 1px, transparent 0)"
                : "radial-gradient(circle at 1px 1px, rgba(38,34,27,0.1) 1px, transparent 0)",
              backgroundSize: "28px 28px"
            }}
          >
            {/* The black hole is a drag-to-delete target; touch uses the
                selection action bar to remove instead. */}
            {isDesktop ? (
              <BlackHoleDropZone
                isActive={dragState !== null || isSweeping || vanishingToken !== null}
                isHot={isOverBlackHole}
                ref={blackHoleRef}
              />
            ) : null}
            {isSweeping && sweepTarget ? (
              <SweepAnimation target={sweepTarget} />
            ) : null}
            {boardElements.map((element) => (
              <BoardToken
                element={element}
                interactionMode={isDesktop ? "drag" : "tap"}
                isCombining={isCombining}
                isDragging={dragState?.instanceId === element.instanceId}
                isSelected={selectedInstanceId === element.instanceId}
                isSweeping={isSweeping}
                key={element.instanceId}
                onPointerCancel={cancelActiveDrag}
                onPointerDown={beginBoardDrag}
                onPointerMove={updateBoardDrag}
                onPointerUp={finishBoardDrag}
                onTap={handleBoardTokenTap}
                sweepTarget={sweepTarget}
                vanishTarget={
                  vanishingToken?.instanceId === element.instanceId
                    ? vanishingToken.target
                    : null
                }
              />
            ))}
          </div>

          {mode === "goal" && goalPreset ? (
            <GoalBanner
              combinationsUsed={history.length}
              isComplete={goalCompletion !== null}
              par={goalPreset.metadata.minDepth ?? goalPreset.metadata.depth}
              target={goalPreset.target}
            />
          ) : null}

          {history.length === 0 && !isCombining && !goalCompletion ? (
            <div className="pointer-events-none absolute inset-x-0 bottom-8 z-10 flex justify-center px-4">
              <p className="max-w-md rounded-md border border-dashed border-linen bg-surface/85 px-4 py-3 text-center text-sm text-soot backdrop-blur">
                {isDesktop
                  ? "Drag one element onto another to combine them"
                  : "Add elements, then tap two of them to combine"}
                {mode === "goal" ? " — craft your way to the goal element" : ""}.
              </p>
            </div>
          ) : null}

          <StatusToast
            errorMessage={errorMessage}
            isCombining={isCombining}
            result={result}
          />

          {discovery ? (
            <div className="pointer-events-none absolute inset-x-0 top-20 z-40 flex justify-center">
              <div
                className="discovery-badge flex max-w-[calc(100%-2rem)] items-center gap-3 rounded-md border border-linen bg-surface px-4 py-2.5 shadow-lift"
                key={discovery.name}
              >
                <span className="text-2xl">{discovery.emoji ?? "·"}</span>
                <span className="truncate text-sm font-semibold capitalize">
                  {discovery.name}
                </span>
                <span className="shrink-0 font-mono text-xs uppercase tracking-wider text-accent">
                  {discovery.isModelGenerated
                    ? "New — invented by the model"
                    : "First discovery"}
                </span>
              </div>
            </div>
          ) : null}
        </section>

        <aside
          className={`hidden min-h-0 flex-col overflow-hidden p-4 transition-colors lg:flex ${
            dragState?.kind === "board" ? "bg-paper" : "bg-surface"
          }`}
        >
          <div className="mb-4 flex shrink-0 gap-1 rounded-md border border-linen bg-paper p-1">
            {sidebarTabs.map((tab) => (
              <button
                className={`h-8 flex-1 rounded font-mono text-xs uppercase tracking-wider transition ${
                  sidebarTab === tab.id
                    ? "bg-surface text-ink shadow-hairline"
                    : "text-soot hover:text-ink"
                }`}
                key={tab.id}
                onClick={() => setActiveSidebarTab(tab.id)}
                type="button"
              >
                {tab.label}
              </button>
            ))}
          </div>

          {sidebarTab === "elements" ? renderElements("drag") : null}
          {sidebarTab === "goal" ? renderGoalPanel() : null}
          {sidebarTab === "recent" ? renderRecent() : null}
          {sidebarTab === "settings" ? renderSettings() : null}
        </aside>

        {/* Mobile: a bottom tab bar opens pop-up sheets instead of a drag-from
            list, so scrolling never fights a drag. Hidden on desktop. */}
        <nav className="flex items-stretch gap-1 border-t border-linen bg-surface p-2 lg:hidden">
          {mobileTabs.map((tab) => (
            <button
              className="relative flex flex-1 flex-col items-center justify-center gap-0.5 rounded-md py-1.5 text-soot transition hover:bg-paper hover:text-ink"
              key={tab.id}
              onClick={() => setMobileSheet(tab.id)}
              type="button"
            >
              {tab.icon}
              <span className="font-mono text-[11px] uppercase tracking-wide">
                {tab.label}
              </span>
              {tab.badge ? (
                <span className="absolute right-2 top-0.5 rounded-full bg-cobalt px-1.5 text-[10px] font-semibold text-white">
                  {tab.badge}
                </span>
              ) : null}
            </button>
          ))}
        </nav>
      </div>

      {/* Mobile selection action bar for tap-to-combine / remove. */}
      {!isDesktop && selectedElement ? (
        <div className="fixed inset-x-0 bottom-[4.75rem] z-40 flex justify-center px-4 lg:hidden">
          <div className="flex max-w-full items-center gap-2 rounded-md border border-linen bg-surface px-3 py-2 shadow-lift">
            <span className="text-lg">{selectedElement.emoji ?? "·"}</span>
            <span className="min-w-0 flex-1 truncate text-sm font-semibold capitalize">
              {selectedElement.name}
            </span>
            <span className="shrink-0 text-xs text-soot">tap another →</span>
            <button
              className="shrink-0 rounded-md border border-linen px-2 py-1 text-xs font-medium text-soot transition hover:bg-paper hover:text-ink"
              onClick={removeSelectedToken}
              type="button"
            >
              Remove
            </button>
            <button
              className="shrink-0 rounded-md border border-linen px-2 py-1 text-xs font-medium text-soot transition hover:bg-paper hover:text-ink"
              onClick={() => setSelectedInstanceId(null)}
              type="button"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : null}

      {/* Mobile pop-up sheets. */}
      {!isDesktop && mobileSheet ? (
        <MobileSheet
          title={mobileSheet}
          onClose={() => setMobileSheet(null)}
        >
          {mobileSheet === "elements" ? renderElements("tap") : null}
          {mobileSheet === "goal" ? renderGoalPanel() : null}
          {mobileSheet === "recent" ? renderRecent() : null}
          {mobileSheet === "settings" ? renderSettings() : null}
        </MobileSheet>
      ) : null}

      {dragState?.kind === "inventory" ? (
        <DragPreview dragState={dragState} pointerRef={dragPointerRef} />
      ) : null}
      {pendingDpoChoice ? (
        <DpoChoiceModal
          choice={pendingDpoChoice}
          onSelect={selectDpoOutput}
          onSkip={skipDpoChoice}
        />
      ) : null}
      {goalCompletion && goalPreset ? (
        <GoalCelebration
          completion={goalCompletion}
          leaderboard={leaderboard}
          target={goalPreset.target}
          userId={user.id}
          onBackToMenu={onBackToMenu}
          onReset={() => resetGame()}
        />
      ) : null}
    </main>
  );
}

function GoalTabPanel({
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
    <div className="grid gap-3 rounded-md border border-linen bg-paper p-3">
      <div className="flex items-start gap-3">
        <Target className="mt-0.5 shrink-0 text-accent" size={18} />
        <div className="min-w-0">
          <p className="font-mono text-xs font-semibold uppercase tracking-wider text-accent">
            Goal
          </p>
          <h2 className="truncate text-sm font-semibold">{goalPreset.title}</h2>
          <p className="mt-1 text-sm text-soot">
            {goalPreset.objective}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="rounded-md border border-linen bg-surface px-3 py-2">
          <p className="font-mono text-xs uppercase tracking-wider text-soot">Depth</p>
          <p className="mt-1 font-mono text-sm font-semibold">
            {goalPreset.metadata.minDepth ?? goalPreset.metadata.depth}
          </p>
        </div>
        <div className="rounded-md border border-linen bg-surface px-3 py-2">
          <p className="font-mono text-xs uppercase tracking-wider text-soot">Start</p>
          <p className="mt-1 truncate font-mono text-sm font-semibold capitalize">
            {goalPreset.metadata.initialInventoryId ?? "fallback"}
          </p>
        </div>
      </div>

      {goalPreset.metadata.minDepth !== undefined &&
      goalPreset.metadata.minDepth !== goalPreset.metadata.depth ? (
        <p className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800">
          No goal exists at depth {goalPreset.metadata.depth} yet, so you got
          the closest available: depth {goalPreset.metadata.minDepth}. Deeper
          goals appear as players and the model discover more recipes.
        </p>
      ) : null}

      <div>
        <p className="mb-2 font-mono text-xs font-semibold uppercase tracking-wider text-soot">
          Initial inventory
        </p>
        <div className="flex flex-wrap gap-1.5">
          {goalPreset.initialInventory.map((element) => (
            <span
              className="element-card max-w-full truncate rounded border px-2 py-1 text-xs capitalize"
              key={element.id}
              style={{ "--el-hue": getHueForConcept(element.name) } as CSSProperties}
            >
              {formatElementName(element)}
            </span>
          ))}
        </div>
      </div>

      <label className="grid gap-1 text-sm font-medium text-ink">
        <span>Goal depth</span>
        <select
          className="h-9 rounded-md border border-linen bg-surface px-3 text-sm outline-none transition focus:border-cobalt disabled:cursor-wait disabled:opacity-60"
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
          className="flex h-10 items-center justify-center gap-2 rounded-md border border-linen bg-surface px-3 text-sm font-semibold transition hover:bg-paper"
          disabled={isGeneratingGoal}
          onClick={onReset}
          type="button"
        >
          <RotateCcw size={15} />
          Reset
        </button>
        <button
          className="flex h-10 items-center justify-center gap-2 rounded-md bg-cobalt px-3 text-sm font-semibold text-white transition hover:bg-cobalt-deep disabled:cursor-wait disabled:opacity-60"
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
  onSelect,
  onSkip
}: {
  choice: PendingDpoChoice;
  onSelect: (output: ElementToken) => void;
  onSkip: () => void;
}) {
  return (
    <div className="fixed inset-0 z-[2147483647] grid place-items-center overflow-y-auto bg-black/55 px-4 py-6 backdrop-blur-sm">
      <div className="my-auto max-h-[calc(100dvh-3rem)] w-full max-w-xl overflow-y-auto rounded-md border border-linen bg-surface p-5 text-ink shadow-lift">
        <div>
          <p className="font-mono text-xs font-semibold uppercase tracking-wider text-accent">
            Help train the AI
          </p>
          <h2 className="mt-1 font-display text-2xl font-black tracking-normal">
            Choose the best result
          </h2>
          <p className="mt-2 text-sm text-soot">
            {choice.firstInput.name} + {choice.secondInput.name} — your pick
            teaches the model which results people prefer.
          </p>
        </div>

        <div className="mt-5 grid gap-2">
          {choice.candidates.map((candidate) => (
            <button
              className="element-card flex min-h-16 items-center justify-between gap-3 rounded-md border px-4 py-3 text-left transition hover:shadow-lift disabled:cursor-wait disabled:opacity-60"
              disabled={choice.isSaving}
              key={candidate.id}
              onClick={() => onSelect(candidate)}
              style={{ "--el-hue": getHueForConcept(candidate.name) } as CSSProperties}
              type="button"
            >
              <span className="truncate text-lg font-semibold capitalize">
                {candidate.name}
              </span>
              <span className="text-sm font-medium text-accent">Select</span>
            </button>
          ))}
        </div>

        {choice.errorMessage ? (
          <p className="mt-4 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {choice.errorMessage}
          </p>
        ) : null}

        <button
          className="mt-4 h-9 w-full rounded-md border border-linen bg-surface text-sm font-medium text-soot transition hover:bg-paper hover:text-ink disabled:cursor-wait disabled:opacity-60"
          disabled={choice.isSaving}
          onClick={onSkip}
          type="button"
        >
          Skip
        </button>
      </div>
    </div>
  );
}

function LeaderboardPreview({ entries }: { entries: LeaderboardEntry[] }) {
  return (
    <div>
      <div className="flex items-center gap-2 font-mono text-xs font-semibold uppercase tracking-wider text-soot">
        <Trophy size={13} />
        Leaderboard
      </div>
      <div className="mt-2 grid gap-1">
        {entries.length === 0 ? (
          <p className="text-xs text-soot">
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
              <span className="shrink-0 font-mono font-semibold">
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
    <div className="fixed inset-0 z-[2147483647] grid place-items-center overflow-y-auto bg-black/55 px-4 py-6 backdrop-blur-sm">
      <div className="my-auto max-h-[calc(100dvh-3rem)] w-full max-w-lg overflow-y-auto rounded-md border border-linen bg-surface p-5 text-ink shadow-lift">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="font-mono text-xs font-semibold uppercase tracking-wider text-accent">
              Goal complete
            </p>
            <h2 className="mt-1 font-display text-3xl font-black tracking-normal">
              {target.emoji ?? "🎯"} {target.name}
            </h2>
          </div>
          <div className="grid size-14 place-items-center rounded-md border border-linen bg-paper text-3xl">
            🏁
          </div>
        </div>

        <div className="mt-5 grid grid-cols-2 gap-3">
          <div className="rounded-md border border-linen bg-paper p-3">
            <p className="font-mono text-2xl font-semibold">{completion.combinationsUsed}</p>
            <p className="mt-1 text-xs text-soot">Combinations used</p>
          </div>
          <div className="rounded-md border border-linen bg-paper p-3">
            <p className="font-mono text-2xl font-semibold">{rank ? `#${rank}` : "..."}</p>
            <p className="mt-1 text-xs text-soot">Leaderboard rank</p>
          </div>
        </div>

        {completion.isSaving ? (
          <p className="mt-4 rounded-md border border-linen bg-paper px-3 py-2 text-sm text-soot">
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
              <p className="rounded-md border border-dashed border-linen p-3 text-sm text-soot">
                No completed runs yet.
              </p>
            ) : (
              leaderboard.slice(0, 5).map((entry, index) => (
                <div
                  className={`flex items-center justify-between gap-3 rounded-md border px-3 py-2 text-sm ${
                    entry.userId === userId
                      ? "border-cobalt bg-cobalt/5"
                      : "border-linen bg-paper"
                  }`}
                  key={entry.id}
                >
                  <span className="truncate">
                    {index + 1}. {entry.displayName}
                  </span>
                  <span className="font-mono font-semibold">
                    {entry.combinationsUsed} combos
                  </span>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="mt-5 grid gap-2 sm:grid-cols-2">
          <button
            className="h-10 rounded-md border border-linen bg-surface px-3 text-sm font-semibold transition hover:bg-paper"
            onClick={onReset}
            type="button"
          >
            Try again
          </button>
          <button
            className="h-10 rounded-md bg-cobalt px-3 text-sm font-semibold text-white transition hover:bg-cobalt-deep"
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

function GoalBanner({
  combinationsUsed,
  isComplete,
  par,
  target
}: {
  combinationsUsed: number;
  isComplete: boolean;
  par: number;
  target: ElementToken;
}) {
  return (
    <div className="pointer-events-none absolute left-1/2 top-[4.75rem] z-20 max-w-[calc(100%-2rem)] -translate-x-1/2 md:top-4">
      <div className="flex items-center gap-2.5 rounded-md border border-linen bg-surface/95 px-3 py-2 shadow-hairline backdrop-blur">
        <Target className="shrink-0 text-accent" size={14} />
        <span className="font-mono text-xs font-semibold uppercase tracking-wider text-soot">
          Craft
        </span>
        <span
          className="element-card flex items-center gap-1.5 rounded-md border px-2 py-1 text-sm font-semibold capitalize"
          style={{ "--el-hue": getHueForConcept(target.name) } as CSSProperties}
        >
          <span>{target.emoji ?? "·"}</span>
          <span className="max-w-36 truncate">{target.name}</span>
        </span>
        <span className="shrink-0 font-mono text-xs text-soot">
          {isComplete
            ? `done in ${combinationsUsed}`
            : `${combinationsUsed} used · doable in ${par}`}
        </span>
      </div>
    </div>
  );
}

const BlackHoleDropZone = forwardRef<
  HTMLDivElement,
  { isActive: boolean; isHot: boolean }
>(function BlackHoleDropZone({ isActive, isHot }, ref) {
  return (
    <div
      aria-label="Drop here to delete"
      className={`pointer-events-none absolute bottom-5 right-5 z-[2147483647] grid size-32 place-items-center rounded-full transition-all duration-300 ease-out sm:size-36 ${
        isActive
          ? isHot
            ? "scale-110 opacity-100"
            : "scale-100 opacity-95"
          : "scale-50 opacity-0"
      }`}
      ref={ref}
      style={{
        background:
          "radial-gradient(circle at center, #0b0a08 0 34%, #26221b 35% 46%, rgba(43,75,223,0.55) 48% 55%, rgba(43,75,223,0.28) 56% 61%, transparent 62%)",
        boxShadow: isHot
          ? "0 0 40px rgba(43,75,223,0.55), 0 0 70px rgba(43,75,223,0.3)"
          : "0 0 24px rgba(43,75,223,0.3), 0 0 42px rgba(38,34,27,0.25)"
      }}
      title="Drop here to delete"
    >
      <div
        className={`blackhole-ring absolute inset-3 rounded-full border-2 border-cobalt/30 transition-colors duration-300 ${
          isHot ? "border-t-white" : "border-t-cobalt"
        }`}
      />
      <div className="size-14 rounded-full bg-black shadow-[inset_0_0_18px_rgba(255,255,255,0.08)] sm:size-16" />
    </div>
  );
});

function SweepAnimation({ target }: { target: Point }) {
  return (
    <div className="pointer-events-none absolute inset-0 z-[2147483000] overflow-hidden">
      <div
        className="sweep-dust absolute h-20 w-20 rounded-full bg-cobalt/25 blur-2xl"
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
  interactionMode,
  isCombining,
  isDragging,
  isSelected,
  isSweeping,
  onPointerCancel,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onTap,
  sweepTarget,
  vanishTarget
}: {
  element: BoardElement;
  interactionMode: "drag" | "tap";
  isCombining: boolean;
  isDragging: boolean;
  isSelected: boolean;
  isSweeping: boolean;
  onPointerDown: (
    event: PointerEvent<HTMLButtonElement>,
    element: BoardElement
  ) => void;
  onPointerCancel: () => void;
  onPointerMove: (event: PointerEvent<HTMLButtonElement>) => void;
  onPointerUp: (event: PointerEvent<HTMLButtonElement>) => void;
  onTap: (element: BoardElement, clientX: number, clientY: number) => void;
  sweepTarget: Point | null;
  vanishTarget: Point | null;
}) {
  const isTap = interactionMode === "tap";
  const vanishTransform = vanishTarget
    ? `translate(${vanishTarget.x - element.x - TOKEN_WIDTH / 2}px, ${
        vanishTarget.y - element.y - TOKEN_HEIGHT / 2
      }px) scale(0.1) rotate(120deg)`
    : undefined;
  const sweepTransform =
    isSweeping && sweepTarget
      ? `translate(${sweepTarget.x - element.x - TOKEN_WIDTH / 2}px, ${
          sweepTarget.y - element.y - TOKEN_HEIGHT / 2
        }px) scale(0.35)`
      : undefined;
  const [isFreshSpawn] = useState(() => {
    const spawnTimestamp = Number(element.instanceId.split(":")[1]);

    return Number.isFinite(spawnTimestamp) && Date.now() - spawnTimestamp < 1000;
  });

  const pointerHandlers = isTap
    ? {
        onClick: (event: PointerEvent<HTMLButtonElement>) => {
          // Stop the board's background onClick from immediately deselecting.
          event.stopPropagation();
          onTap(element, event.clientX, event.clientY);
        }
      }
    : {
        onPointerCancel,
        onPointerDown: (event: PointerEvent<HTMLButtonElement>) =>
          onPointerDown(event, element),
        onPointerMove,
        onPointerUp
      };

  return (
    <button
      className={`element-card absolute flex h-12 w-[150px] touch-none select-none items-center gap-2 rounded-md border px-2.5 text-left shadow-sm transition-shadow hover:shadow-md ${
        isDragging || isSelected ? "opacity-95 shadow-lg ring-2 ring-cobalt" : ""
      } ${
        isCombining || isSweeping
          ? "cursor-wait"
          : isTap
            ? "cursor-pointer"
            : "cursor-grab active:cursor-grabbing"
      } ${isSweeping ? "board-token-sweeping" : ""} ${
        vanishTarget ? "board-token-vanishing" : ""
      } ${isFreshSpawn && !vanishTarget ? "element-pop" : ""}`}
      data-board-token-id={element.instanceId}
      disabled={isCombining || isSweeping || vanishTarget !== null}
      {...pointerHandlers}
      style={
        {
          left: element.x,
          top: element.y,
          transform: vanishTransform ?? sweepTransform,
          zIndex: element.zIndex,
          "--el-hue": getHueForConcept(element.name)
        } as CSSProperties
      }
      type="button"
    >
      <span className="grid size-8 shrink-0 place-items-center rounded border border-linen bg-surface/70 text-lg">
        {element.emoji ?? "·"}
      </span>
      <span className="min-w-0 break-words text-[13px] font-medium capitalize leading-[1.15] line-clamp-2">
        {element.name}
      </span>
    </button>
  );
}

function InventoryToken({
  element,
  interactionMode,
  onPick,
  onPointerCancel,
  onPointerDown,
  onPointerMove,
  onPointerUp
}: {
  element: ElementToken;
  interactionMode: "drag" | "tap";
  onPick: (element: ElementToken) => void;
  onPointerCancel: () => void;
  onPointerDown: (
    event: PointerEvent<HTMLButtonElement>,
    element: ElementToken
  ) => void;
  onPointerMove: (event: PointerEvent<HTMLButtonElement>) => void;
  onPointerUp: (event: PointerEvent<HTMLButtonElement>) => void;
}) {
  const isTap = interactionMode === "tap";
  // Touch: a plain tap that adds to the board — no pointer-drag handlers, so it
  // never captures the pointer and the list scrolls normally.
  const handlers = isTap
    ? { onClick: () => onPick(element) }
    : {
        onPointerCancel,
        onPointerDown: (event: PointerEvent<HTMLButtonElement>) =>
          onPointerDown(event, element),
        onPointerMove,
        onPointerUp
      };

  return (
    <button
      className={`element-card flex min-h-16 select-none flex-col items-center justify-center rounded-md border px-2 py-2 text-center transition hover:shadow-lift ${
        isTap
          ? "cursor-pointer active:scale-95"
          : "cursor-grab touch-none active:cursor-grabbing"
      }`}
      {...handlers}
      style={{ "--el-hue": getHueForConcept(element.name) } as CSSProperties}
      type="button"
    >
      <span className="text-2xl">{element.emoji ?? "·"}</span>
      <span className="mt-1 w-full hyphens-auto break-words text-xs font-medium capitalize leading-tight text-ink">
        {element.name}
      </span>
    </button>
  );
}

function MobileSheet({
  title,
  onClose,
  children
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex flex-col justify-end bg-black/40 backdrop-blur-sm lg:hidden"
      onClick={onClose}
    >
      <div
        className="flex max-h-[85dvh] min-h-[50dvh] flex-col rounded-t-2xl border-t border-linen bg-surface"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-linen px-4 py-3">
          <h2 className="font-display text-base font-semibold capitalize">
            {title}
          </h2>
          <button
            aria-label="Close"
            className="grid size-9 place-items-center rounded-md border border-linen text-soot transition hover:bg-paper hover:text-ink"
            onClick={onClose}
            type="button"
          >
            <X size={18} />
          </button>
        </div>
        <div className="flex min-h-0 flex-1 flex-col p-3">{children}</div>
      </div>
    </div>
  );
}

function DragPreview({
  dragState,
  pointerRef
}: {
  dragState: DragState;
  pointerRef: { current: Point | null };
}) {
  const nodeRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let frame = requestAnimationFrame(function follow() {
      const point = pointerRef.current;

      if (nodeRef.current && point) {
        nodeRef.current.style.transform = `translate(${
          point.x - dragState.offsetX
        }px, ${point.y - dragState.offsetY}px)`;
      }

      frame = requestAnimationFrame(follow);
    });

    return () => cancelAnimationFrame(frame);
  }, [dragState.offsetX, dragState.offsetY, pointerRef]);

  return (
    <div
      className="element-card pointer-events-none fixed left-0 top-0 z-50 flex h-12 w-[150px] items-center gap-2 rounded-md border px-2.5 text-left opacity-90 shadow-lg ring-2 ring-cobalt"
      ref={nodeRef}
      style={
        {
          transform: `translate(${dragState.pointerX - dragState.offsetX}px, ${
            dragState.pointerY - dragState.offsetY
          }px)`,
          "--el-hue": getHueForConcept(dragState.element.name)
        } as CSSProperties
      }
    >
      <span className="grid size-8 shrink-0 place-items-center rounded border border-linen bg-surface/70 text-lg">
        {dragState.element.emoji ?? "·"}
      </span>
      <span className="min-w-0 break-words text-[13px] font-medium capitalize leading-[1.15] line-clamp-2">
        {dragState.element.name}
      </span>
    </div>
  );
}

function StatusToast({
  errorMessage,
  isCombining,
  result
}: {
  errorMessage: string | null;
  isCombining: boolean;
  result: CombineResponse | null;
}) {
  if (!errorMessage && !isCombining && !result) {
    return null;
  }

  return (
    <div
      className={`absolute bottom-3 left-3 z-40 flex max-w-[calc(100%-1.5rem)] items-center gap-2 rounded-md border border-linen bg-surface/95 px-3 py-1.5 shadow-hairline backdrop-blur sm:max-w-md`}
    >
      {errorMessage ? (
        <p className="text-sm text-soot">{errorMessage}</p>
      ) : isCombining ? (
        <p className="text-sm text-soot">Resolving combination…</p>
      ) : result ? (
        <>
          <span className="text-base">{result.result.emoji ?? "·"}</span>
          <span className="truncate text-sm font-semibold capitalize">
            {result.result.name}
          </span>
          <span className="ml-auto shrink-0 rounded border border-linen bg-paper px-1.5 py-0.5 font-mono text-[11px] text-soot">
            {formatCombineSource(result.source)}
            {typeof result.confidence === "number"
              ? ` · ${Math.round(result.confidence * 100)}%`
              : ""}
          </span>
        </>
      ) : null}
    </div>
  );
}

function RecipeCard({ item }: { item: RecipeHistoryItem }) {
  return (
    <div className="rounded-md border border-linen bg-paper p-3">
      <div className="flex flex-wrap items-center gap-1 text-xs text-soot">
        <span>{formatElementName(item.inputA)}</span>
        <span>+</span>
        <span>{formatElementName(item.inputB)}</span>
      </div>
      <div className="mt-2 flex items-center justify-between gap-3">
        <span className="truncate text-sm font-medium capitalize">
          {formatElementName(item.output)}
        </span>
        <span className="rounded border border-linen bg-surface px-2 py-1 font-mono text-xs text-soot">
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
