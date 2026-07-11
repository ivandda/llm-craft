"use client";

import { useEffect } from "react";

/**
 * Registers the PWA service worker once, on the client, after the page loads.
 * Registration is best-effort: any failure (unsupported browser, insecure
 * origin) is swallowed so it never affects gameplay.
 */
export function ServiceWorkerRegistrar() {
  useEffect(() => {
    if (typeof window === "undefined" || !("serviceWorker" in navigator)) {
      return;
    }

    // Dev builds change assets constantly; only register in production so the
    // cache never serves stale chunks while developing.
    if (process.env.NODE_ENV !== "production") {
      return;
    }

    const register = () => {
      navigator.serviceWorker.register("/sw.js").catch(() => {
        // Ignore: the app works fine without the service worker.
      });
    };

    if (document.readyState === "complete") {
      register();
    } else {
      window.addEventListener("load", register, { once: true });
      return () => window.removeEventListener("load", register);
    }
  }, []);

  return null;
}
