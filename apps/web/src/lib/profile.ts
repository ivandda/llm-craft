import type {
  ElementToken,
  FeaturedAchievement,
  UserProfile
} from "@/lib/types";

export const FEATURED_ACHIEVEMENT_LIMIT = 6;

export function selectFeaturedAchievements(
  inventory: ElementToken[],
  requestedElementIds: string[],
  now = new Date().toISOString()
): FeaturedAchievement[] {
  const availableElements = new Map(
    inventory.map((element) => [element.id, element] as const)
  );
  const selectedIds = [...new Set(requestedElementIds)];

  return selectedIds
    .map((elementId) => availableElements.get(elementId))
    .filter((element): element is ElementToken => Boolean(element))
    .slice(0, FEATURED_ACHIEVEMENT_LIMIT)
    .map((element) => ({
      elementId: element.id,
      name: element.name,
      emoji: element.emoji,
      featuredAt: now
    }));
}

export function createDefaultProfile(
  userId: string,
  displayName: string
): UserProfile {
  return {
    userId,
    displayName,
    featuredAchievements: [],
    updatedAt: new Date().toISOString()
  };
}
