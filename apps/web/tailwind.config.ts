import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      colors: {
        // Theme-aware semantic tokens (see globals.css). The `<alpha-value>`
        // placeholder lets `bg-surface/85`, `text-soot`, etc. keep working.
        paper: "rgb(var(--color-paper) / <alpha-value>)",
        surface: "rgb(var(--color-surface) / <alpha-value>)",
        ink: "rgb(var(--color-ink) / <alpha-value>)",
        soot: "rgb(var(--color-soot) / <alpha-value>)",
        linen: "rgb(var(--color-linen) / <alpha-value>)",
        // `accent` is theme-aware (brightens in dark) for text/icons/rings.
        accent: "rgb(var(--color-accent) / <alpha-value>)",
        // cobalt stays constant: it backs buttons that carry white text in
        // both themes (white on #2B4BDF passes AA).
        cobalt: "#2B4BDF",
        "cobalt-deep": "#1E38B6"
      },
      fontFamily: {
        display: ["var(--font-display)", "ui-sans-serif", "sans-serif"],
        sans: ["var(--font-body)", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"]
      },
      boxShadow: {
        hairline: "0 0 0 1px rgb(38 34 27 / 0.08)",
        lift: "0 1px 2px rgb(38 34 27 / 0.06), 0 6px 18px rgb(38 34 27 / 0.07)"
      }
    }
  },
  plugins: []
};

export default config;
