import type { Config } from "tailwindcss";

// Neon Arcade palette: the values live as CSS variables in app/globals.css (one source of truth, lane V reuses them).
const v = (name: string) => `var(--${name})`;
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: v("bg"), surface: v("surface"), raised: v("raised"), border: v("border"), "border-ui": v("border-ui"),
        pink: v("pink"), cyan: v("cyan"), yellow: v("yellow"), green: v("green"),
        "pink-dark": v("pink-dark"), "cyan-dark": v("cyan-dark"), "yellow-dark": v("yellow-dark"), "green-dark": v("green-dark"),
        "chip-grey": v("chip-grey"), "chip-grey-bg": v("chip-grey-bg"),
        text: v("text"), soft: v("soft"), muted: v("muted"), "muted-2": v("muted-2"), ink: v("ink"), "ink-cyan": v("ink-cyan"),
        // v1 names kept so nothing outside lane U breaks: panel = surface, accent = cyan, warn = yellow.
        panel: v("surface"), accent: v("cyan"), warn: v("yellow"),
      },
    },
  },
  plugins: [],
};
export default config;
