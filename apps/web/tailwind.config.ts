import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/**/*.{js,ts,jsx,tsx,mdx}"
  ],
  theme: {
    extend: {
      boxShadow: {
        hairline: "0 0 0 1px rgb(24 24 27 / 0.08)"
      }
    }
  },
  plugins: []
};

export default config;
