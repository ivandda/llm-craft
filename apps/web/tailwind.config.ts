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
        paper: "#FAF6EE",
        surface: "#FFFDF8",
        ink: "#26221B",
        soot: "#6F6759",
        linen: "#E3DACA",
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
