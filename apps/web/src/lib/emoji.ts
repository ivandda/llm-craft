const CONCEPT_EMOJI: Record<string, string> = {
  water: "💧",
  fire: "🔥",
  earth: "🌍",
  wind: "💨",
  air: "🌬️",
  steam: "♨️",
  mist: "🌫️",
  fog: "🌫️",
  smoke: "💨",
  cloud: "☁️",
  rain: "🌧️",
  storm: "⛈️",
  thunder: "🌩️",
  lightning: "⚡",
  snow: "❄️",
  ice: "🧊",
  glacier: "🏔️",
  mountain: "⛰️",
  volcano: "🌋",
  lava: "🌋",
  magma: "🌋",
  obsidian: "⚫",
  stone: "🪨",
  rock: "🪨",
  sand: "🏖️",
  dust: "🌫️",
  mud: "🟤",
  clay: "🧱",
  brick: "🧱",
  glass: "🔮",
  metal: "🔩",
  iron: "⚙️",
  gold: "🥇",
  silver: "🥈",
  diamond: "💎",
  crystal: "🔮",
  energy: "⚡",
  electricity: "⚡",
  heat: "♨️",
  sun: "☀️",
  moon: "🌙",
  star: "⭐",
  sky: "🌌",
  space: "🚀",
  planet: "🪐",
  ocean: "🌊",
  sea: "🌊",
  wave: "🌊",
  lake: "🏞️",
  river: "🏞️",
  geyser: "⛲",
  swamp: "🥬",
  island: "🏝️",
  beach: "🏖️",
  desert: "🏜️",
  forest: "🌲",
  tree: "🌳",
  plant: "🌱",
  seed: "🌰",
  flower: "🌸",
  grass: "🌿",
  leaf: "🍃",
  wood: "🪵",
  mushroom: "🍄",
  fruit: "🍎",
  wine: "🍷",
  beer: "🍺",
  alcohol: "🍸",
  spirit: "👻",
  ghost: "👻",
  life: "🧬",
  animal: "🐾",
  fish: "🐟",
  bird: "🐦",
  dragon: "🐉",
  human: "🧑",
  house: "🏠",
  city: "🏙️",
  farm: "🚜",
  bread: "🍞",
  dough: "🥖",
  flour: "🌾",
  wheat: "🌾",
  salt: "🧂",
  sugar: "🍬",
  oil: "🛢️",
  gunpowder: "🧨",
  explosion: "💥",
  bomb: "💣",
  ash: "🌫️",
  charcoal: "🪵",
  coal: "⚫",
  time: "⏳",
  love: "❤️",
  music: "🎵",
  rainbow: "🌈",
  gold_rush: "⛏️",
  tool: "🔧",
  engine: "⚙️",
  robot: "🤖",
  computer: "💻",
  internet: "🌐",
  book: "📖",
  paper: "📄",
  boat: "⛵",
  ship: "🚢",
  car: "🚗",
  train: "🚆",
  plane: "✈️",
  extinguish: "🧯"
};

const FALLBACK_EMOJI = [
  "✨", "🌀", "🔷", "🔶", "🟢", "🟣", "🟠", "🧿", "🎐", "🫧",
  "🌟", "💠", "🔺", "🧩", "🎲", "🪄", "🎇", "🌠", "🫟", "🪩"
];

export function getHueForConcept(name: string): number {
  return hashConcept(name.trim().toLowerCase()) % 360;
}

export function getEmojiForConcept(name: string): string {
  const normalized = name.trim().toLowerCase();

  if (CONCEPT_EMOJI[normalized]) {
    return CONCEPT_EMOJI[normalized];
  }

  for (const word of normalized.split(/[\s-]+/).reverse()) {
    if (CONCEPT_EMOJI[word]) {
      return CONCEPT_EMOJI[word];
    }
  }

  return FALLBACK_EMOJI[hashConcept(normalized) % FALLBACK_EMOJI.length];
}

function hashConcept(value: string): number {
  let hash = 2166136261;

  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }

  return hash >>> 0;
}
