// Minimal, dependency-free service worker for llm·craft.
// Goals: make the app installable + resilient offline for its shell and static
// assets, without ever caching dynamic/authenticated API responses.

const CACHE = "llm-craft-v1";
const APP_SHELL = ["/", "/manifest.webmanifest", "/icon.svg"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE)
      // Per-entry (not addAll): one transiently-unreachable asset must not
      // reject the whole install and leave the shell entirely uncached.
      .then((cache) =>
        Promise.all(APP_SHELL.map((url) => cache.add(url).catch(() => undefined)))
      )
      .catch(() => undefined)
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(keys.filter((key) => key !== CACHE).map((key) => caches.delete(key)))
      )
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const { request } = event;

  // Only handle GET; never intercept POST/PUT (combine, auth, dpo, ...).
  if (request.method !== "GET") {
    return;
  }

  const url = new URL(request.url);

  // Same-origin only.
  if (url.origin !== self.location.origin) {
    return;
  }

  // Never cache API calls — they are dynamic and often authenticated.
  if (url.pathname.startsWith("/api/")) {
    return;
  }

  // Navigations: network-first, fall back to the cached "/" shell when
  // offline. Only the home route is used as the offline shell so that a
  // last-visited page (e.g. an error page) can never overwrite it.
  if (request.mode === "navigate") {
    event.respondWith(
      fetch(request)
        .then((response) => {
          if (url.pathname === "/" && response.ok) {
            const copy = response.clone();
            caches.open(CACHE).then((cache) => cache.put("/", copy)).catch(() => undefined);
          }

          return response;
        })
        .catch(() =>
          caches
            .match(request)
            .then((cached) => cached ?? caches.match("/"))
            .then((cached) => cached ?? Response.error())
        )
    );
    return;
  }

  // Static assets (Next chunks, icons, fonts): cache-first, then network.
  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) {
        return cached;
      }

      return fetch(request)
        .then((response) => {
          if (response.ok && response.type === "basic") {
            const copy = response.clone();
            caches.open(CACHE).then((cache) => cache.put(request, copy)).catch(() => undefined);
          }

          return response;
        })
        // Nothing cached and the network failed: return a real (error)
        // Response — respondWith must never resolve to undefined.
        .catch(() => Response.error());
    })
  );
});
