import type {
  CombineRequest,
  CombineResponse,
  ElementToken
} from "@/lib/types";

export const BASE_ELEMENTS: ElementToken[] = [
  { id: "water", name: "water", emoji: "💧" },
  { id: "fire", name: "fire", emoji: "🔥" },
  { id: "earth", name: "earth", emoji: "🌍" },
  { id: "air", name: "air", emoji: "💨" }
];

type KnownRecipe = {
  output: ElementToken;
  alternatives?: ElementToken[];
  rationale: string;
};

const KNOWN_RECIPES: Record<string, KnownRecipe> = {
  "earth+water": {
    output: createElementToken("mud", "🟤"),
    rationale: "earth absorbs water and becomes mud"
  },
  "air+fire": {
    output: createElementToken("smoke", "💨"),
    alternatives: [createElementToken("heat", "♨️")],
    rationale: "fire moving through air produces smoke"
  },
  "earth+fire": {
    output: createElementToken("lava", "🌋"),
    rationale: "fire melts earth into lava"
  },
  "air+water": {
    output: createElementToken("rain", "🌧️"),
    alternatives: [createElementToken("cloud", "☁️")],
    rationale: "water suspended in air returns as rain"
  },
  "fire+water": {
    output: createElementToken("steam", "♨️"),
    alternatives: [
      createElementToken("mist", "🌫️"),
      createElementToken("hot spring", "♨️")
    ],
    rationale: "fire heats water into steam"
  },
  "earth+air": {
    output: createElementToken("dust", "🌫️"),
    rationale: "air lifts earth into dust"
  },
  "earth+rain": {
    output: createElementToken("plant", "🌱"),
    rationale: "rain lets earth grow plants"
  },
  "fire+mud": {
    output: createElementToken("brick", "🧱"),
    rationale: "fire hardens mud into brick"
  },
  "lava+water": {
    output: createElementToken("stone", "🪨"),
    alternatives: [createElementToken("obsidian", "⚫")],
    rationale: "water cools lava into stone"
  },
  "plant+water": {
    output: createElementToken("tree", "🌳"),
    rationale: "water helps a plant grow into a tree"
  },
  "fire+stone": {
    output: createElementToken("metal", "🔩"),
    rationale: "fire extracts metal from stone"
  },
  "air+steam": {
    output: createElementToken("cloud", "☁️"),
    rationale: "steam disperses into air as cloud"
  }
};

const FALLBACK_PREFIXES = [
  "quiet",
  "pale",
  "bright",
  "dense",
  "soft",
  "sharp",
  "silver",
  "clear"
];

const FALLBACK_NOUNS = [
  "matter",
  "spark",
  "trace",
  "form",
  "field",
  "blend",
  "echo",
  "compound"
];

export function normalizeConcept(value: string): string {
  return value.trim().replace(/\s+/g, " ").toLowerCase();
}

export function createElementToken(name: string, emoji?: string): ElementToken {
  const normalizedName = normalizeConcept(name);

  return {
    id: normalizedName,
    name: normalizedName,
    emoji
  };
}

export function makePairKey(inputA: string, inputB: string): string {
  const sortedInputs = [normalizeConcept(inputA), normalizeConcept(inputB)].sort();
  return `${sortedInputs[0]}+${sortedInputs[1]}`;
}

export function combineElements(request: CombineRequest): CombineResponse {
  const pairKey = makePairKey(request.inputA.name, request.inputB.name);
  const knownRecipe = KNOWN_RECIPES[pairKey];

  if (knownRecipe) {
    const knownOutputs = [
      knownRecipe.output,
      ...(knownRecipe.alternatives ?? [])
    ];

    return {
      result: knownRecipe.output,
      source: "known_recipe",
      confidence: 0.98,
      knownOutputs,
      rationale: knownRecipe.rationale
    };
  }

  const fallback = createFallbackElement(pairKey);

  return {
    result: fallback,
    source: "mock_model",
    confidence: 0.42,
    knownOutputs: [fallback],
    rationale: "mock response reserved for the future model integration"
  };
}

export function mergeInventory(
  inventory: ElementToken[],
  nextElement: ElementToken
): ElementToken[] {
  if (inventory.some((element) => element.id === nextElement.id)) {
    return inventory;
  }

  return [
    ...inventory,
    {
      ...nextElement,
      discoveredAt: nextElement.discoveredAt ?? new Date().toISOString()
    }
  ].sort((left, right) => left.name.localeCompare(right.name));
}

function createFallbackElement(pairKey: string): ElementToken {
  const hash = hashString(pairKey);
  const prefix = FALLBACK_PREFIXES[hash % FALLBACK_PREFIXES.length];
  const noun = FALLBACK_NOUNS[Math.floor(hash / 7) % FALLBACK_NOUNS.length];

  return createElementToken(`${prefix} ${noun}`);
}

function hashString(value: string): number {
  let hash = 0;

  for (let index = 0; index < value.length; index += 1) {
    hash = (hash * 31 + value.charCodeAt(index)) >>> 0;
  }

  return hash;
}
