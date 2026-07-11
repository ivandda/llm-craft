export const GUEST_USERNAME_PREFIX = "guest-";

export function isGuestUser(user: { username: string }): boolean {
  return user.username.startsWith(GUEST_USERNAME_PREFIX);
}
