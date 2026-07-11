const BASIC_PREFIX = "basic ";

export function isAdminDashboardConfigured(): boolean {
  return Boolean(
    process.env.ADMIN_DASH_USER?.trim() && process.env.ADMIN_DASH_PASSWORD?.trim()
  );
}

export function isAuthorizedAdminRequest(request: Request): boolean {
  if (!isAdminDashboardConfigured()) {
    return false;
  }

  const header = request.headers.get("authorization") ?? "";

  if (!header.toLowerCase().startsWith(BASIC_PREFIX)) {
    return false;
  }

  const decoded = decodeBase64(header.slice(BASIC_PREFIX.length).trim());
  const separator = decoded.indexOf(":");

  if (separator < 0) {
    return false;
  }

  const user = decoded.slice(0, separator);
  const password = decoded.slice(separator + 1);

  return (
    constantTimeEquals(user, process.env.ADMIN_DASH_USER?.trim() ?? "") &&
    constantTimeEquals(password, process.env.ADMIN_DASH_PASSWORD?.trim() ?? "")
  );
}

export function adminUnauthorizedResponse(): Response {
  return new Response("Authentication required", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="llm-craft admin"' }
  });
}

// Works in both the edge (middleware) and Node (route) runtimes.
function decodeBase64(value: string): string {
  try {
    return atob(value);
  } catch {
    return "";
  }
}

function constantTimeEquals(a: string, b: string): boolean {
  const length = Math.max(a.length, b.length);
  let mismatch = a.length ^ b.length;

  for (let index = 0; index < length; index += 1) {
    mismatch |= (a.charCodeAt(index) || 0) ^ (b.charCodeAt(index) || 0);
  }

  return mismatch === 0;
}
