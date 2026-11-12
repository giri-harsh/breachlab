import { defineConfig, devices } from "@playwright/test";

// E2E_BASE_URL: test an existing deployment; otherwise `next start` is launched on :3100 (run `npm run build` first).
// E2E_LIVE=1: also run the specs that need the real gateway (e2e/live.spec.ts, the round trip in play.spec.ts).
// PW_CHROMIUM: path to a Chromium binary (the dev container has /opt/pw-browsers/chromium).
// HTTPS_PROXY: when set (sandboxed dev container) the browser goes through it for everything except localhost.
// SHOTS=1: e2e/screens.spec.ts writes the Neon Arcade screenshots to docs/screenshots/arcade/.
const base = process.env.E2E_BASE_URL ?? "http://localhost:3100";
const proxy = process.env.HTTPS_PROXY ?? process.env.https_proxy;

export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  expect: { timeout: 15_000 },
  reporter: [["list"], ["json", { outputFile: "test-results/report.json" }]],
  use: {
    baseURL: base,
    ...(proxy ? { proxy: { server: proxy, bypass: "localhost,127.0.0.1" }, ignoreHTTPSErrors: true } : {}),
    launchOptions: process.env.PW_CHROMIUM ? { executablePath: process.env.PW_CHROMIUM, args: ["--no-sandbox"] } : {},
  },
  projects: [
    // Phone first: 390x844 with touch and a mobile UA. The live walk-through runs here only (one real session per level).
    { name: "mobile-390", use: { ...devices["Pixel 7"], viewport: { width: 390, height: 844 } } },
    { name: "desktop-1440", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } }, testIgnore: /live\.spec\.ts/ },
  ],
  webServer: process.env.E2E_BASE_URL
    ? undefined
    : { command: "npx next start -p 3100", url: "http://localhost:3100", reuseExistingServer: true, timeout: 60_000 },
});
